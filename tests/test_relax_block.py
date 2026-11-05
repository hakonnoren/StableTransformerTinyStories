"""
Tests for the relaxed reversible coupling (RevConfig(regime="relax")).

    x' = e^-g x + (1 - e^-g)(z + Attn(LN_1 z))
    z' = e^-a z + (1 - e^-a)(x' + MLP(LN_2 x'))      g, a = softplus(raw) > 0

Checked here, all in float64 on tiny models:
  * rates start at relax_rate; relax_fixed makes them buffers; the optimizer puts
    learned rates in the no-decay rev_scale group;
  * the block equals the conformal (vf_scaling) block with forces
    (e^g - 1)(z + Attn) and (1 - e^-a)(x + MLP);
  * inverse(forward(y)) == y;
  * log|det| of one block == -T * sum(g + a), and the H = F = 0 spectrum is
    {1, e^-(g+a)} per channel;
  * --mem_efficient gradients (segment 0 and n_layer) equal eager autograd;
  * the causal mask still holds;
  * RevConfig rejects flags that do nothing under relax.

Run:  python tests/test_relax_block.py   (or pytest tests/test_relax_block.py)
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root

from model import ModelConfig                       # noqa: E402
from revformer import RevFormerModel, RevConfig     # noqa: E402
from revformer.revformer import ReversibleBlock     # noqa: E402

VOCAB, T, D, H, L = 61, 6, 16, 2, 4


def _cfg(n_layer=L, n_embd=D):
    return ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=n_layer, n_head=H,
                       n_embd=n_embd, dropout=0.0, bias=False)


def _model(rate=0.5, learn=True, seed=0, **kw):
    torch.manual_seed(seed)
    m = RevFormerModel(_cfg(**kw), RevConfig(regime="relax", relax_rate=rate,
                                             relax_learn=learn, embed="narrow"))
    m.double()
    # Spread the rates so per-channel bugs (wrong broadcast, g/a swapped) show up.
    with torch.no_grad():
        for b in m.blocks:
            if isinstance(b.gamma_bias, torch.nn.Parameter):
                b.gamma_bias.add_(0.3 * torch.randn_like(b.gamma_bias))
                b.alpha_bias.add_(0.3 * torch.randn_like(b.alpha_bias))
    return m


def test_rate_init_and_grouping():
    from train import build_optimizer
    m = RevFormerModel(_cfg(), RevConfig(regime="relax", relax_rate=3.0))
    for b in m.blocks:
        assert torch.allclose(b.get_gamma(), torch.full((1, 1, D), 3.0), atol=1e-6)
        assert torch.allclose(b.get_alpha(), torch.full((1, 1, D), 3.0), atol=1e-6)
    names = {n for n, _ in m.named_parameters()}
    assert "blocks.0.gamma_bias" in names and "blocks.0.alpha_bias" in names

    opt = build_optimizer(m, peak_lr=1e-3, rev_scale_lr_mult=10.0)
    adamw = opt[0] if isinstance(opt, (list, tuple)) else opt
    rates = {id(b.gamma_bias) for b in m.blocks} | {id(b.alpha_bias) for b in m.blocks}
    grp = [g for g in adamw.param_groups if any(id(p) in rates for p in g["params"])]
    assert len(grp) == 1 and grp[0]["weight_decay"] == 0.0 and grp[0]["lr_mult"] == 10.0
    assert {id(p) for p in grp[0]["params"]} == rates

    mf = RevFormerModel(_cfg(), RevConfig(regime="relax", relax_rate=0.7, relax_learn=False))
    assert not any("gamma_bias" in n or "alpha_bias" in n for n, _ in mf.named_parameters())
    assert torch.allclose(mf.blocks[0].get_gamma(), torch.full((1, 1, D), 0.7), atol=1e-6)


def test_equals_conformal_with_modified_forces():
    m = _model()
    torch.manual_seed(1)
    y = torch.randn(2, T, 2 * D, dtype=torch.float64)
    for b in m.blocks:
        g, a = b.get_gamma(), b.get_alpha()
        x, z = y.split(D, dim=-1)
        # eq. (11) with H~(z) = (e^g - 1)(z + H(z)) and F~(x) = (1 - e^-a)(x + F(x))
        x1 = torch.exp(-g) * (x + torch.expm1(g) * (z + b.attn(b.ln_1(z))))
        z1 = torch.exp(-a) * z + (-torch.expm1(-a)) * (x1 + b.mlp(b.ln_2(x1)))
        ref = torch.cat([x1, z1], dim=-1)
        out = b(y)
        assert (out - ref).abs().max() < 1e-13, (out - ref).abs().max()
        y = out


def test_inverse_roundtrip():
    m = _model(rate=0.5)
    torch.manual_seed(2)
    y = torch.randn(3, T, 2 * D, dtype=torch.float64)
    for b in m.blocks:
        y_rec = b.inverse(b(y))
        assert (y_rec - y).abs().max() / y.abs().max() < 1e-11
    with torch.no_grad():
        acts = [y]
        for b in m.blocks:
            acts.append(b(acts[-1]))
        y_rec = acts[-1]
        for b in reversed(m.blocks):
            y_rec = b.inverse(y_rec)
    assert (y_rec - y).abs().max() / y.abs().max() < 1e-10


def test_logdet_and_linear_spectrum():
    m = _model(rate=0.8, n_layer=1, n_embd=8)
    b = m.blocks[0]
    torch.manual_seed(3)
    Tn, d = 3, 8
    y = torch.randn(1, Tn, 2 * d, dtype=torch.float64)
    J = torch.autograd.functional.jacobian(lambda v: b(v.view(1, Tn, 2 * d)).reshape(-1),
                                           y.reshape(-1))
    _, logabsdet = torch.linalg.slogdet(J)
    expected = -Tn * float((b.get_gamma() + b.get_alpha()).sum().detach())
    assert abs(float(logabsdet) - expected) < 1e-9, (float(logabsdet), expected)

    # H = F = 0: per channel the 2x2 map has eigenvalues 1 (x = z kept) and
    # e^-(g+a) (x - z contracted).
    g, a = b.get_gamma().detach().reshape(-1), b.get_alpha().detach().reshape(-1)
    for i in range(d):
        kx, kz = math.exp(-float(g[i])), math.exp(-float(a[i]))
        M = torch.tensor([[kx, 1 - kx], [(1 - kz) * kx, kz + (1 - kz) * (1 - kx)]],
                         dtype=torch.float64)
        ev = sorted(torch.linalg.eigvals(M).real.tolist())
        assert abs(ev[1] - 1.0) < 1e-12 and abs(ev[0] - kx * kz) < 1e-12


def _grads(m, idx, tgt, mem_eff, segment):
    m.zero_grad(set_to_none=True)
    m.mem_efficient, m.mem_efficient_segment = mem_eff, segment
    _, loss = m(idx, tgt)
    loss.backward()
    return float(loss), {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def test_mem_efficient_matches_eager():
    m = _model(rate=0.5)
    m.train()
    torch.manual_seed(4)
    idx = torch.randint(0, VOCAB, (2, T))
    tgt = torch.randint(0, VOCAB, (2, T))
    l_ref, g_ref = _grads(m, idx, tgt, False, 0)
    assert any(n.endswith("gamma_bias") for n in g_ref)
    for seg in (0, L):
        l, g = _grads(m, idx, tgt, True, seg)
        assert abs(l - l_ref) < 1e-12
        assert g.keys() == g_ref.keys()
        for n in g_ref:
            err = (g[n] - g_ref[n]).abs().max() / (g_ref[n].abs().max() + 1e-30)
            assert err < 1e-8, (seg, n, float(err))


def test_causal():
    m = _model(rate=1.0)
    m.eval()
    torch.manual_seed(5)
    idx = torch.randint(0, VOCAB, (1, T))
    with torch.no_grad():
        base, _ = m(idx)
        for j in range(1, T):
            idx2 = idx.clone()
            idx2[0, j] = (idx2[0, j] + 1) % VOCAB
            out, _ = m(idx2)
            assert (out[0, :j] - base[0, :j]).abs().max() < 1e-10


def test_config_validation():
    for kw in (dict(tanh_scale=True), dict(randn_init=True), dict(init_offset=0.1),
               dict(relax_rate=0.0), dict(relax_rate=-1.0)):
        try:
            RevConfig(regime="relax", **kw)
        except ValueError:
            continue
        raise AssertionError(f"RevConfig(regime='relax', {kw}) should have raised")
    for kw in (dict(relax_rate=1.0), dict(relax_learn=False)):
        try:
            RevConfig(regime="vf_scaling", **kw)
        except ValueError:
            continue
        raise AssertionError(f"RevConfig(regime='vf_scaling', {kw}) should have raised")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} relax tests passed")
