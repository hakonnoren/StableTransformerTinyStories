"""Tests of the multi-hop couplings, data handling, training and evaluation (tiny random model).

Needs a checkout of OSU-NLP-Group/Loop-Think-Generalize (for RecurrentGPT2Block) and transformers 4.x:
    LTG_DIR=/path/to/Loop-Think-Generalize python -m pytest -q looped/multihop/test_multihop.py
Optional LTG_DATA (a data/multi_hop directory with vocab.json and test_small.json) adds the end-to-end smoke tests.
"""
import json
import os
import subprocess
import sys

import pytest
import torch

LTG = os.environ.get("LTG_DIR", "")
DATA = os.environ.get("LTG_DATA", os.path.join(LTG, "data", "multi_hop") if LTG else "")
pytestmark = pytest.mark.skipif(not LTG or not os.path.exists(os.path.join(LTG, "gpt_utils_extrapolation.py")),
                                reason="set LTG_DIR to a Loop-Think-Generalize checkout")


def tiny(coupling="vanilla", **kw):
    from looped.multihop.model import Coupled, load_base
    torch.manual_seed(0)
    base = load_base(LTG, 40, None, d_model=32, n_layer=2, n_head=4, max_len=64).eval()
    for p in base.parameters():                                   # upstream zero-inits the output projections
        if p.dim() >= 1:
            torch.nn.init.normal_(p, std=0.2) if p.dim() > 1 else None
    return Coupled(base, coupling, **kw).eval()


def batch(b=6, n=9, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 40, (b, n), generator=g)
    ids[:, -1] = 0                                                 # the answer position is a pad
    ids[0, -3:] = 0                                                # one shorter sequence
    return ids, (ids != 0).long()


def test_relax_at_large_gamma_and_zero_delta_is_the_released_recurrence():
    van = tiny("vanilla")
    rel = tiny("relax", gamma=60.0, delta=0.0)
    rel.base = van.base
    ids, mask = batch()
    a = van.trajectory(ids, mask, [1, 2, 3, 7, 12], want=("logp",))["logp"]
    b = rel.trajectory(ids, mask, [1, 2, 3, 7, 12], want=("logp",))["logp"]
    assert torch.allclose(a, b, atol=1e-5)


def test_port_warmup_start_is_undamped_and_twin_reads_last_then_mean():
    rel0 = tiny("relax", gamma=3.0, delta=0.0)
    reld = tiny("relax", gamma=3.0, delta=2.0)
    reld.base = rel0.base
    reld.mix = 0.0
    ids, mask = batch()
    a = rel0.trajectory(ids, mask, [5], want=("logp",))["logp"]
    b = reld.trajectory(ids, mask, [5], want=("logp",))["logp"]
    assert torch.allclose(a, b, atol=1e-6)
    tw = tiny("twin", gamma=3.0, delta=2.0)
    tw.base = rel0.base
    s = (torch.randn(2, 3, 32), torch.randn(2, 3, 32))
    tw.mix = 0.0
    assert torch.equal(tw.read(s, 1), s[1])
    tw.mix = 1.0
    assert torch.allclose(tw.read(s, 1), 0.5 * (s[0] + s[1]))


def test_carry_then_gradient_iterations_equal_one_trajectory():
    m = tiny("relax", gamma=3.0, delta=1.0)
    ids, mask = batch()
    with torch.no_grad():
        lg = m(ids, mask, n_iter=3, n_carry=5)
    lp = m.trajectory(ids, mask, [8], want=("logp",))["logp"][0]
    assert torch.allclose(torch.log_softmax(lg, -1), lp, atol=1e-5)


def test_noise_is_reproducible_and_zero_noise_is_exact():
    m = tiny("twin", gamma=3.0, delta=2.0)
    ids, mask = batch()
    run = lambda s, seed: m.trajectory(ids, mask, [6], want=("logp",), sigma=s,
                                       gen=torch.Generator().manual_seed(seed))["logp"]
    assert torch.equal(run(0.0, 1), m.trajectory(ids, mask, [6], want=("logp",))["logp"])
    assert torch.equal(run(0.2, 1), run(0.2, 1))
    assert not torch.allclose(run(0.2, 1), run(0.2, 2))


def test_prediction_does_not_depend_on_padding():
    m = tiny("relax", gamma=3.0, delta=1.0)
    ids, mask = batch()
    pad = torch.cat([ids, torch.zeros(ids.shape[0], 5, dtype=ids.dtype)], 1)
    a = m.trajectory(ids, mask, [4], want=("logp",))["logp"]
    b = m.trajectory(pad, (pad != 0).long(), [4], want=("logp",))["logp"]
    assert torch.allclose(a, b, atol=1e-5)


def test_gradients_flow_only_through_the_last_iterations():
    m = tiny("relax", gamma=3.0, delta=1.0).train()
    ids, mask = batch()
    m(ids, mask, n_iter=2, n_carry=4).sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.mark.skipif(not DATA or not os.path.exists(os.path.join(DATA, "test_small.json")), reason="no LTG_DATA")
def test_data_split_and_end_to_end(tmp_path):
    from looped.multihop.data import MultiHop, collate
    d = tmp_path / "data"
    d.mkdir()
    small = json.load(open(os.path.join(DATA, "test_small.json")))
    (d / "vocab.json").write_text(open(os.path.join(DATA, "vocab.json")).read())
    (d / "test_small.json").write_text(json.dumps(small))
    (d / "train.json").write_text(json.dumps([x for x in small if int(x["type"].split("hop")[0]) <= 6]))
    mh = MultiHop(str(d))
    k = 10
    assert len(mh.dev[k]) + len(mh.test[k]) == 50 and len(mh.dev[k]) == 17
    ids, mask, y = collate(mh.test[k][:4])
    assert ids.shape[1] == k + 2 and (ids[:, -1] == 0).all()
    # a tiny random model, saved in the released format, then trained a few steps and evaluated
    from looped.multihop.model import load_base
    base = load_base(LTG, len(mh.vocab), None, d_model=32, n_layer=2, n_head=4, max_len=128)
    ck = tmp_path / "init.pt"
    torch.save({"model_state_dict": base.state_dict()}, ck)
    env = {**os.environ, "PYTHONPATH": os.getcwd()}
    out = tmp_path / "run"
    common = ["--ltg-dir", LTG, "--data-dir", str(d)]
    subprocess.run([sys.executable, "-m", "looped.multihop.train", *common, "--init-ckpt", str(ck), "--out", str(out),
                    "--coupling", "twin", "--delta", "2", "--carry-max", "4", "--sigma-train", "0.01", "--steps", "6",
                    "--eval-every", "3", "--port-warmup", "4", "--start-hop", "3", "--max-hop", "6", "--batch", "8",
                    "--d-model", "32", "--n-layer", "2", "--n-head", "4", "--device", "cpu"], check=True, env=env, timeout=600)
    assert (out / "final.pt").exists()
    lines = [json.loads(x) for x in open(out / "curriculum.jsonl")]
    assert lines[0]["step"] == 0 and lines[-1]["step"] == 6
    model_args = ["--hops", "2-12", "--depths", "1,2,4,8", "--halt-max", "6", "--sigmas", "0.1", "--breadth", "3",
                  "--breadth-depths", "4,8", "--breadth-hops", "4,8", "--breadth-n", "5", "--batch", "16", "--device", "cpu"]
    subprocess.run([sys.executable, "-m", "looped.multihop.evaluate", *common, "--ckpt", str(out / "final.pt"),
                    *model_args], check=True, env=env, timeout=600)
    res = json.load(open(out / "eval_final.json"))
    assert set(res) >= {"depth", "overthink", "halting", "noise"}
    assert len(res["depth"]["acc"]["4"]) == 4
    assert set(res["noise"]["per_sigma"]["0.1"]["8"]["8"]) == {"single", "vote", "conv", "any"}
