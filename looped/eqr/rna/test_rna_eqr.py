"""Tests for the RNA part of the EqR overlay: the refold evaluator and EqR's attention block on RNA-shaped data.
Run from inside an EqR checkout with the overlay installed (looped/eqr/install_overlay.sh) and ViennaRNA:

    cd $EQR_DIR && python -m pytest -q tests_coupled/test_rna_eqr.py
"""
import csv
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

sys.path.insert(0, os.getcwd())                       # the EqR checkout

from evaluators import rna_design as rd               # noqa: E402
from evaluators.rna_refold import RNARefold           # noqa: E402
from models.coupled_eqr import CoupledEqRModel        # noqa: E402
from models.eqr import EqRModel                       # noqa: E402
from models.losses import ACTLossHead                 # noqa: E402

pytestmark = pytest.mark.skipif(rd.RNA is None, reason="ViennaRNA not installed")

SEQ_LEN = 16


def _onehot_logits(seqs, struct_noise=True):
    """Logits that decode to `seqs`; structure tokens get even larger logits, which decoding must ignore."""
    x = torch.zeros(len(seqs), SEQ_LEN, rd.VOCAB_SIZE)
    for b, s in enumerate(seqs):
        for i, c in enumerate(s):
            x[b, i, rd.NUC_TOKENS[c]] = 5.0
    if struct_noise:
        x[..., rd.STRUCT_TOKENS["("]] = 9.0
    return x


def test_evaluator_scores_by_refolding(tmp_path):
    t0, t1 = "((((...))))", "((((((....))))))"
    inputs = torch.stack([torch.from_numpy(rd.encode_structure(t, SEQ_LEN)) for t in (t0, t1)]
                         + [torch.zeros(SEQ_LEN, dtype=torch.uint8)])               # a padding row
    labels = torch.stack([torch.from_numpy(rd.encode_sequence(s, SEQ_LEN)) for s in ("GGGGAAACCCC", "G" * 6 + "A" * 4 + "C" * 6)]
                         + [torch.zeros(SEQ_LEN, dtype=torch.uint8)])
    preds = {"logits": _onehot_logits(["GGGGAAACCCC", "A" * 16, ""])}               # right, wrong, padding
    ev = RNARefold(data_path=None, eval_metadata=types.SimpleNamespace(vocab_size=rd.VOCAB_SIZE))
    assert ev.required_outputs == {"logits"}
    ev.begin_eval()
    ev.update_batch({"inputs": inputs, "labels": labels}, preds)
    m = ev.result(str(tmp_path))["rna"]
    assert m["n"] == 2 and m["solved"] == 0.5
    assert m["pairs_ok"] == 0.5                       # poly-A pairs nothing
    assert m["label_agree"] == pytest.approx((1.0 + 4 / 16) / 2)
    assert m["solved_len1-32"] == 0.5
    rows = list(csv.DictReader(open(tmp_path / "rna_refold.csv")))
    assert [r["solved"] for r in rows] == ["1", "0"] and rows[0]["prediction"] == "GGGGAAACCCC"
    ev.begin_eval()
    assert ev.result(None) == {}                     # nothing seen since begin_eval
    with pytest.raises(ValueError):
        RNARefold(eval_metadata=types.SimpleNamespace(vocab_size=11))


RNA_ARCH = dict(batch_size=4, seq_len=24, vocab_size=rd.VOCAB_SIZE, H_cycles=2, L_cycles=2, H_layers=0, L_layers=2,
                hidden_size=32, expansion=4, num_heads=2, pos_encodings="rope", board_height=None, board_width=None,
                halt_max_steps=3, halt_exploration_prob=0.1, mlp_t=False, noise_scale=0.01, forward_dtype="float32",
                puzzle_emb_ndim=32, puzzle_emb_len=2, num_puzzle_identifiers=1)


def _rna_batch(b=4, seed=0):
    rng = np.random.default_rng(seed)
    ts = [rd.sample_structure(rng, int(rng.integers(12, 25))) for _ in range(b)]
    seqs = [rd.design_sequence(rng, t) for t in ts]
    return {"inputs": torch.from_numpy(np.stack([rd.encode_structure(t, 24) for t in ts])).long(),
            "labels": torch.from_numpy(np.stack([rd.encode_sequence(s, 24) for s in seqs])).long(),
            "puzzle_identifiers": torch.zeros(b, dtype=torch.int32)}


@pytest.mark.parametrize("model", ["eqr", "relax", "twin"])
def test_attention_block_on_rna_batches(model):
    """EqR's attention block (1D RoPE, puzzle-embedding prefix) and the coupled arms on RNA-shaped data:
    ACT steps through the loss head, logits over the RNA vocabulary at the sequence positions only."""
    torch.manual_seed(0)
    if model == "eqr":
        m = EqRModel(dict(RNA_ARCH))
    elif model == "relax":
        m = CoupledEqRModel(dict(RNA_ARCH, coupling="relax", rate=3.0, leak=1.0, state_dtype="float32"))
    else:
        m = CoupledEqRModel(dict(RNA_ARCH, coupling="relax", rate=3.0, leak=2.0, inject_h=False, schedule="eqr",
                                 twin_l=True, state_dtype="float32"))
    head = ACTLossHead(m, loss_type="stablemax_cross_entropy")
    head.train()
    batch = _rna_batch()
    carry = head.initial_carry(batch)
    for _ in range(2):
        carry, loss, metrics, outputs, _ = head(carry=carry, batch=batch, return_keys=["logits"])
        loss.backward()
    assert outputs["logits"].shape == (4, 24, rd.VOCAB_SIZE)
    # min_*/max_* start at +-inf (EqR's reduction sentinels) until some sample halts
    assert torch.isfinite(loss) and all(torch.isfinite(v).all() for k, v in metrics.items()
                                        if not k.startswith(("min_", "max_")))
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    ev = RNARefold()
    ev.begin_eval()
    ev.update_batch(batch, outputs)
    assert ev.result(None)["rna"]["n"] == 4           # the evaluator consumes the model's outputs as they are


def test_sdpa_fallback_patch_applied():
    """The SDPA fallback is in models/layers.py (training with attention on CUDA needs it without FlashAttention)."""
    src = open(os.path.join("models", "layers.py")).read()
    assert "flash_attn_func is not None" in src and "flash_attn is not installed but CUDA attention" not in src
    out = subprocess.run(["git", "diff", "--stat", "models/layers.py"], capture_output=True, text=True)
    assert "layers.py" in out.stdout
