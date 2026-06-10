"""Integration tests for the linear-map reversible block (lowrank_cayley).

Run:  python -m revformer.test_linear_map_block      (from repo root)
Checks: build across regimes/rotations, train step, exact reversibility,
log|det| vs numeric Jacobian, causality, vpm volume-preservation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model import ModelConfig
from revformer.revformer import RevFormerModel, RevConfig, LinearMixedReversibleBlock

torch.manual_seed(0)


def _cfg(d=16, L=3, V=40, T=12):
    return ModelConfig(vocab_size=V, block_size=T, n_layer=L, n_head=2, n_embd=d,
                       dropout=0.0, bias=True)


def test_build_train_all():
    cfg = _cfg()
    for regime in ("vpb_baseline", "vpb_scaling", "vpm_scaling", "vf_scaling"):
        for rotation in ("none", "householder", "cayley"):
            rc = RevConfig(regime=regime, linear_map="lowrank_cayley",
                           lowrank_r=4, rotation=rotation, n_householder=4)
            m = RevFormerModel(cfg, rev_cfg=rc)
            idx = torch.randint(0, cfg.vocab_size, (2, cfg.block_size))
            logits, loss = m(idx, idx)
            loss.backward()
            assert torch.isfinite(loss), (regime, rotation)
    print("  build+train across 4 regimes x 3 rotations: OK")


def test_reversibility():
    cfg = _cfg()
    for rotation in ("none", "householder", "cayley"):
        rc = RevConfig(regime="vf_scaling", linear_map="lowrank_cayley",
                       lowrank_r=4, rotation=rotation)
        blk = LinearMixedReversibleBlock(cfg, rc)
        with torch.no_grad():
            blk.linear_map.rho.copy_(torch.randn(4) * 0.3)   # non-trivial volume
            y = torch.randn(2, cfg.block_size, 2 * cfg.n_embd)
            err = (blk.inverse(blk(y, avg=0.1), avg=0.1) - y).abs().max().item()
        assert err < 1e-4, (rotation, err)
        print(f"  reversibility rotation={rotation:11s}: inv_err {err:.2e}")


def test_logdet_vs_jacobian():
    cfg = _cfg(d=8, T=1)        # single position so block Jac == L (shears are det 1)
    for rotation in ("none", "householder", "cayley"):
        rc = RevConfig(regime="vf_scaling", linear_map="lowrank_cayley",
                       lowrank_r=3, rotation=rotation)
        blk = LinearMixedReversibleBlock(cfg, rc).eval()
        with torch.no_grad():
            blk.linear_map.rho.copy_(torch.randn(3) * 0.5)
        m = 2 * cfg.n_embd
        y0 = torch.randn(m)
        J = torch.autograd.functional.jacobian(lambda v: blk(v.reshape(1, 1, m), avg=0.0).reshape(m), y0)
        num = torch.linalg.slogdet(J)[1].item()
        ana = blk.linear_map.logdet(0.0).item()
        assert abs(num - ana) < 1e-3, (rotation, num, ana)
        print(f"  logdet vs Jacobian rotation={rotation:11s}: numeric {num:+.4f} | analytic {ana:+.4f}")


def test_causality():
    cfg = _cfg()
    rc = RevConfig(regime="vf_scaling", linear_map="lowrank_cayley", rotation="householder")
    m = RevFormerModel(cfg, rev_cfg=rc).eval()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size))
    with torch.no_grad():
        base, _ = m(idx)
        t = cfg.block_size // 2
        idx2 = idx.clone(); idx2[0, t + 1:] = torch.randint(0, cfg.vocab_size, (cfg.block_size - t - 1,))
        pert, _ = m(idx2)
        delta = (base[0, :t + 1] - pert[0, :t + 1]).abs().max().item()
    assert delta < 1e-5, delta
    print(f"  causality (future edit -> earlier logits): max_delta {delta:.2e}")


def test_vpm_volume():
    cfg = _cfg()
    for lambd in (0.0, 2.0):
        rc = RevConfig(regime="vpm_scaling", lambd=lambd, linear_map="lowrank_cayley",
                       lowrank_r=4, rotation="householder")
        m = RevFormerModel(cfg, rev_cfg=rc)
        with torch.no_grad():
            for b in m.blocks:
                b.linear_map.rho.copy_(torch.randn(4) * 0.4)
            T = cfg.block_size
            avg = m._avg_corr(T)
            total = sum(b.linear_map.logdet(avg).item() for b in m.blocks) * T
        assert abs(total - (-lambd)) < 1e-3, (lambd, total)
        print(f"  vpm volume lambd={lambd}: total stack log|det| {total:+.4f}  (target {-lambd:+.1f})")


def test_optimizer_routing():
    from train import build_optimizer
    cfg = _cfg()
    rc = RevConfig(regime="vf_scaling", linear_map="lowrank_cayley", rotation="householder")
    m = RevFormerModel(cfg, rev_cfg=rc)
    opt = build_optimizer(m, peak_lr=6e-4, optimizer="muon", rev_scale_lr_mult=10.0)
    muon_ids = getattr(opt, "muon_param_ids", set()) or set()
    gen = [(n, p) for n, p in m.named_parameters() if n.endswith((".U_param", ".K_param", ".V"))]
    assert gen, "no generator params found"
    bad = [n for n, p in gen if id(p) in muon_ids]
    assert not bad, f"generators wrongly on Muon: {bad}"
    rho = [(n, p) for n, p in m.named_parameters() if n.endswith(".rho")]
    on_muon_rho = [n for n, p in rho if id(p) in muon_ids]
    assert not on_muon_rho, on_muon_rho
    print(f"  optimizer routing: {len(gen)} generators on AdamW (not Muon), {len(rho)} rho on rev_scale: OK")


def test_validation():
    try:
        RevConfig(regime="damped", linear_map="lowrank_cayley")
        raise AssertionError("expected ValueError for damped + lowrank_cayley")
    except ValueError:
        print("  validation: damped + lowrank_cayley correctly rejected")


if __name__ == "__main__":
    print("== linear-map reversible block tests ==")
    test_build_train_all()
    test_reversibility()
    test_logdet_vs_jacobian()
    test_causality()
    test_vpm_volume()
    test_optimizer_routing()
    test_validation()
    print("ALL PASS")
