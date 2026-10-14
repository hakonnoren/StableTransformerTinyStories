"""
Causality / future-token-leak test for the language models in this repo.

A causal LM's logits at position i must depend ONLY on input tokens 0..i. We
test this directly: change one input token at position j and check whether the
logits at any position i < j move. For a causal model they must not (Δ == 0); a
nonzero change means that position is seeing a future token (a leak).

Result (see the leak investigation): baseline / RevFormer / YuriiFormer are
exactly causal; the presymplectic SympFormer (presymp_*) LEAKS, because its
momentum force G = -∂H/∂x is the gradient of a sequence-global symmetric
Hamiltonian — the gradient w.r.t. a key position m collects contributions from
every (future) query that attends to m, even though the attention kernel E is
correctly causal-masked. That's fine for the non-autoregressive tasks the
construction was designed for, but it's a future-token leak for next-token LM.

Run:  python test_causality.py        (exits nonzero if a causal model leaks)
"""
import sys

import torch

from model import ModelConfig, GPTModel, YuriiFormerModel, PresympModelETDAB2
from revformer import RevFormerModel, RevConfig

TOL = 1e-5          # max |Δlogit| at earlier positions allowed for a "causal" model
VOCAB = 65
SEQ_LEN = 12


def _make_presymp(cfg, lookahead=True):
    return PresympModelETDAB2(
        cfg, h=0.01, t0=1.0, eta_learnable=True, eta_mode="loglin",
        eta_log_init=3, eta_lin_init=1e-4, eta_clip=12, presymp_lnp="end",
        use_v0_init=False, mlp_use_attn_vel=True, lookahead=lookahead,
    )


def build_models(cfg):
    """(name, model, expect_causal) tuples."""
    return [
        ("baseline", GPTModel(cfg), True),
        ("reversible_vpb_baseline", RevFormerModel(cfg, RevConfig(regime="vpb_baseline")), True),
        ("reversible_vf_scaling", RevFormerModel(cfg, RevConfig(regime="vf_scaling")), True),
        ("reversible_vpm_scaling", RevFormerModel(cfg, RevConfig(regime="vpm_scaling")), True),
        ("yurii_lt", YuriiFormerModel(cfg), True),
        ("presymp_etd_ab2 (lookahead=ON)", _make_presymp(cfg, True), False),
        ("presymp_etd_ab2 (lookahead=OFF)", _make_presymp(cfg, False), False),
    ]


@torch.no_grad()
def max_leak(model, vocab=VOCAB, T=SEQ_LEN, edit_positions=(3, 6, 9), seed=1):
    """Largest |Δlogit| at positions strictly before an edited input token,
    maximised over a few edit positions. 0 ⇒ causal."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, vocab, (1, T), generator=g)
    logits, _ = model(idx)
    worst = 0.0
    for j in edit_positions:
        if j <= 0 or j >= T:
            continue
        idx2 = idx.clone()
        idx2[0, j] = (int(idx[0, j]) + 7) % vocab
        logits2, _ = model(idx2)
        before = (logits[:, :j] - logits2[:, :j]).abs().max().item()
        worst = max(worst, before)
    return worst


def main():
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=VOCAB, block_size=64, n_layer=4, n_head=4,
                      n_embd=32, dropout=0.0, bias=False)
    print(f"{'model':36s}{'max|Δlogit| (i<j)':>20}  verdict")
    failures = []
    for name, model, expect_causal in build_models(cfg):
        leak = max_leak(model)
        is_causal = leak <= TOL
        if expect_causal:
            ok = is_causal
            verdict = "causal OK" if ok else "LEAK — UNEXPECTED (regression!)"
            if not ok:
                failures.append(name)
        else:
            # documented known leak; flag if it ever becomes causal (someone fixed it)
            ok = True
            verdict = "LEAK (known / expected)" if not is_causal else "now causal?? (update test)"
        print(f"{name:36s}{leak:20.2e}  {verdict}")

    if failures:
        print(f"\nFAIL: causal models leaked: {failures}")
        sys.exit(1)
    print("\nPASS: all causal architectures are causal.")
    sys.exit(0)


if __name__ == "__main__":
    main()
