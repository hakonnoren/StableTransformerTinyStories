"""Structural checks for the looped couplings (models/looped.py):

* kick and relax are exactly invertible;
* their Jacobian log|det| is the state-independent -T·Σ(γ+α) (conformal);
* relax at a large rate is the looped standard block on stream H;
* the trajectory helper reproduces forward().
"""
import pytest
import torch

from looped.model import LoopedConfig, LoopedGPT, CouplingLayer


def _cfg(**kw):
    base = dict(vocab_size=11, block_size=6, n_embd=8, n_head=2)
    base.update(kw)
    return LoopedConfig(**base)


@pytest.mark.parametrize("coupling,rate,per_channel", [("kick", 0.3, False), ("kick", 0.0, False),
                                                       ("relax", 3.0, False), ("relax", 0.7, True)])
def test_inverse_exact(coupling, rate, per_channel):
    torch.manual_seed(0)
    layer = CouplingLayer(_cfg(coupling=coupling, rate=rate, rate_h=rate * 0.5 + 0.1 if rate else 0.0,
                               per_channel=per_channel, learn_rates=True)).double()
    s = torch.randn(3, 6, 16, dtype=torch.double)
    e = torch.randn(3, 6, 8, dtype=torch.double)
    rec = layer.inverse(layer(s, e), e)
    assert torch.allclose(rec, s, atol=1e-10), (rec - s).abs().max()


@pytest.mark.parametrize("coupling,rate", [("kick", 0.2), ("relax", 1.5), ("relax", 3.0)])
def test_logdet_constant(coupling, rate):
    torch.manual_seed(1)
    T, d = 4, 4
    layer = CouplingLayer(_cfg(coupling=coupling, rate=rate, rate_h=rate + 0.4, n_embd=d, block_size=T)).double()
    e = torch.randn(1, T, d, dtype=torch.double)
    expected = float(layer.log_volume(T))
    for trial in range(3):
        s = 3 * torch.randn(1, T, 2 * d, dtype=torch.double)
        J = torch.autograd.functional.jacobian(lambda v: layer(v.view(1, T, 2 * d), e).flatten(), s.flatten())
        _, logdet = torch.linalg.slogdet(J)
        assert abs(float(logdet) - expected) < 1e-8, (float(logdet), expected)


def test_relax_large_rate_is_looped_block():
    torch.manual_seed(2)
    relax = LoopedGPT(_cfg(coupling="relax", rate=40.0)).double()
    van = LoopedGPT(_cfg(coupling="vanilla")).double()
    van.load_state_dict({k: v for k, v in relax.state_dict().items() if "raw" not in k})
    idx = torch.randint(0, 11, (2, 6))
    a, _ = relax(idx, n_iter=5)
    b, _ = van(idx, n_iter=5)
    assert torch.allclose(a, b, atol=1e-9)


def test_trajectory_matches_forward():
    torch.manual_seed(3)
    m = LoopedGPT(_cfg(coupling="relax", rate=2.0, learn_rates=True))
    idx = torch.randint(0, 11, (2, 6))
    traj = m.trajectory(idx, 7, record=[3, 7])
    for r in (3, 7):
        logits, _ = m(idx, n_iter=r)
        assert torch.allclose(traj[r][0], logits, atol=1e-6)


# --- constrained damping -------------------------------------------------------------------------
CONSTRAINED = [
    dict(coupling="relax", rate=2.0, rate_h=1.0, leak=0.3),
    dict(coupling="relax", rate=2.0, leak=0.1, learn_leak=True, learn_rates=True, per_channel=True),
    dict(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.1, learn_rates=True),
    dict(coupling="relax", rate=2.0, rate_constraint="bounded", rate_min=0.5, rate_max=4.0, learn_rates=True),
    dict(coupling="relax", rate=3.0, rate_h=1.0, rate_constraint="budget", learn_rates=True, per_channel=True),
]


@pytest.mark.parametrize("kw", CONSTRAINED)
def test_constrained_inverse_and_logdet(kw):
    torch.manual_seed(4)
    T, d = 4, 4
    layer = CouplingLayer(_cfg(n_embd=d, block_size=T, **kw)).double()
    with torch.no_grad():                 # move learned raws off their init
        for n, p in layer.named_parameters():
            if n.endswith("_raw"):
                p.add_(0.2 * torch.randn_like(p))
    e = torch.randn(1, T, d, dtype=torch.double)
    s = 2 * torch.randn(1, T, 2 * d, dtype=torch.double)
    assert torch.allclose(layer.inverse(layer(s, e), e), s, atol=1e-10)
    J = torch.autograd.functional.jacobian(lambda v: layer(v.view(1, T, 2 * d), e).flatten(), s.flatten())
    assert abs(float(torch.linalg.slogdet(J)[1]) - float(layer.log_volume(T))) < 1e-8


def test_budget_and_positive():
    l = CouplingLayer(_cfg(coupling="relax", rate=3.0, rate_h=1.0, rate_constraint="budget", learn_rates=True))
    with torch.no_grad():
        l.split_raw.fill_(5.0)
    assert torch.allclose(l.gamma() + l.alpha(), torch.tensor([4.0]))
    k = CouplingLayer(_cfg(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.1, learn_rates=True))
    with torch.no_grad():
        k.gamma_raw.fill_(-50.0)
    assert float(k.gamma()) >= 0.1 - 1e-6


@pytest.mark.parametrize("leak", [0.0, 0.2, 1.0])
def test_linear_part_radius(leak):
    """With zero forces the map is linear; its spectral radius is the analytic value, and the
    power-iteration estimate recovers it."""
    torch.manual_seed(5)
    m = LoopedGPT(_cfg(coupling="relax", rate=1.5, rate_h=0.7, leak=leak)).double()
    layer = m.core[0]
    with torch.no_grad():
        for mod in (layer.attn.c_proj, layer.mlp.c_proj):
            mod.weight.zero_(); mod.bias.zero_()
    T, d = 6, 8
    s = torch.randn(1, T, 2 * d, dtype=torch.double)
    e = torch.zeros(1, T, d, dtype=torch.double)
    J = torch.autograd.functional.jacobian(lambda v: layer(v.view(1, T, 2 * d), e).flatten(), s.flatten())
    rho = float(torch.linalg.eigvals(J).abs().max())
    assert abs(rho - layer.linear_part_radius()) < 1e-9
    assert (rho < 1 - 1e-6) == (leak > 0)
    est = m.step_spectral_radius(torch.randint(0, 11, (3, 6)), r=2, iters=200)
    assert torch.allclose(est, torch.full_like(est, rho), atol=1e-3)


@pytest.mark.parametrize("coupling", ["kick", "relax"])
def test_post_injection_inverse(coupling):
    torch.manual_seed(6)
    layer = CouplingLayer(_cfg(coupling=coupling, rate=1.0, inject_mode="post", leak=0.2 if coupling == "relax" else 0.0)).double()
    s = torch.randn(2, 6, 16, dtype=torch.double); e = torch.randn(2, 6, 8, dtype=torch.double)
    assert torch.allclose(layer.inverse(layer(s, e), e), s, atol=1e-10)
