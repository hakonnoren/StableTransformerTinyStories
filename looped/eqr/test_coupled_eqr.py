"""Tests for the coupled EqR overlay. Run from inside an EqR checkout that has the
overlay installed (looped/eqr/install_overlay.sh):

    cd $EQR_DIR && python -m pytest -q tests_coupled/test_coupled_eqr.py
"""
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.getcwd())                     # the EqR checkout

from models.eqr import EqRModel                     # noqa: E402
from models.coupled_eqr import CoupledEqRModel      # noqa: E402

BASE = dict(batch_size=4, seq_len=9, vocab_size=11, H_cycles=2, L_cycles=2, H_layers=0, L_layers=2,
            hidden_size=16, expansion=4, num_heads=2, pos_encodings="none", halt_max_steps=3,
            halt_exploration_prob=0.1, mlp_t=True, noise_scale=0.0, board_height=3, board_width=3)


def _cfg(dtype="float64", **kw):
    kw.setdefault("state_dtype", "float64" if dtype == "float64" else "float32")
    return dict(BASE, forward_dtype=dtype, **kw)


def _coupled(seed=0, **kw):
    torch.manual_seed(seed)
    return CoupledEqRModel(_cfg(**kw))


def _batch(b=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {"inputs": torch.randint(1, 11, (b, 9), generator=g),
            "labels": torch.randint(1, 11, (b, 9), generator=g),
            "puzzle_identifiers": torch.zeros(b, dtype=torch.int32)}


def test_same_parameters_and_init_as_eqr():
    torch.manual_seed(0)
    ref = EqRModel(_cfg())
    m = _coupled(seed=0, coupling="relax")
    ref_sd, sd = ref.state_dict(), m.state_dict()
    extra = {k for k in sd if k.startswith("inner.rates.")}
    assert set(sd) - extra == set(ref_sd)
    for k, v in ref_sd.items():
        assert torch.equal(v, sd[k]), k
    n_ref = sum(p.numel() for p in ref.parameters())
    assert sum(p.numel() for p in m.parameters()) == n_ref                       # fixed rates: buffers
    ml = _coupled(seed=0, coupling="relax", learn_rates=True)
    assert sum(p.numel() for p in ml.parameters()) == n_ref + 2
    mc = _coupled(seed=0, coupling="kick", learn_rates=True, per_channel=True)
    assert sum(p.numel() for p in mc.parameters()) == n_ref + 2 * 16
    assert len(mc.rate_parameters()) == 2
    assert _coupled(coupling="kick", learn_rates=True, decay_rates=True).rate_parameters() == []


def test_schedule_and_depth():
    m = _coupled()
    assert "".join(m.inner.schedule) == "HLH"                          # L_cycles + 1 = 3, ends on H
    assert m.inner.depth_per_outer_step() == {"visited_half_steps": 6, "differentiated_half_steps": 3}
    torch.manual_seed(0)
    assert "".join(CoupledEqRModel(dict(_cfg(), L_cycles=6)).inner.schedule) == "HLHLHLH"
    assert "".join(_coupled(half_steps=4).inner.schedule) == "LHLH"
    assert "".join(_coupled(schedule="eqr").inner.schedule) == "LLH"                 # EqR's order
    assert "".join(_coupled(schedule="eqr", twin_l=True).inner.schedule) == "LMH"     # M = twin L'
    torch.manual_seed(0)
    m6 = CoupledEqRModel(dict(_cfg(schedule="eqr", twin_l=True), L_cycles=6))
    assert "".join(m6.inner.schedule) == "LMLMLMH"
    assert m6.inner.depth_per_outer_step() == {"visited_half_steps": 14, "differentiated_half_steps": 7}


@pytest.mark.parametrize("kw", [
    dict(coupling="kick", rate=0.005),
    dict(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.05, learn_rates=True),
    dict(coupling="relax", rate=3.0),
    dict(coupling="relax", rate=3.0, leak=0.25),
    dict(coupling="relax", rate=0.7, rate_h=1.3, per_channel=True, learn_rates=True),
    dict(coupling="relax", rate=3.0, leak=1.0, inject_h=False),
    dict(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.05, learn_rates=True, inject_h=False),
])
def test_update_formula_and_exact_inverse(kw):
    m = _coupled(**kw)
    inner = m.inner
    with torch.no_grad():
        for p in inner.rates.parameters():                           # spread per-channel rates
            p.add_(0.2 * torch.randn_like(p))
    torch.manual_seed(3)
    H, L = torch.randn(4, 9, 16, dtype=torch.float64), torch.randn(4, 9, 16, dtype=torch.float64)
    x = inner._input_embeddings(_batch()["inputs"], _batch()["puzzle_identifiers"])
    seq = {"cos_sin": None}
    with torch.no_grad():
        g, a = inner.rates.gamma().double(), inner.rates.alpha().double()
        c = inner.rates.c()
        c = 1.0 if c is None else c.double()
        fH = inner.L_level(H + x, **seq)
        L1 = inner.half_step_L(H, L, x, seq)
        if kw["coupling"] == "kick":
            ref = torch.exp(-g) * (L + fH)
        else:
            ref = torch.exp(-g) * L + (1 - torch.exp(-g)) * (c * H + fH)
        assert (L1 - ref).abs().max() < 1e-12
        fL = inner.L_level(L1 + x if kw.get("inject_h", True) else L1, **seq)
        H1 = inner.half_step_H(H, L1, x, seq)
        if kw["coupling"] == "kick":
            ref = torch.exp(-a) * H + fL
        else:
            ref = torch.exp(-a) * H + (1 - torch.exp(-a)) * (c * L1 + fL)
        assert (H1 - ref).abs().max() < 1e-12
        # exact inverses
        H0, L0 = inner.inverse_half_step("H", H1, L1, x, seq)
        assert torch.equal(L0, L1) and (H0 - H).abs().max() < 1e-12
        H0, L0 = inner.inverse_half_step("L", H, L1, x, seq)
        assert (L0 - L).abs().max() < 1e-12


@pytest.mark.parametrize("kw", [dict(coupling="kick", rate=0.3), dict(coupling="relax", rate=1.0, leak=0.5),
                                dict(coupling="relax", rate=2.0, per_channel=True, learn_rates=True),
                                dict(coupling="relax", rate=1.0, leak=0.5, inject_h=False)])
def test_log_det_of_a_recursion(kw):
    torch.manual_seed(0)
    m = CoupledEqRModel(dict(_cfg(**kw), hidden_size=8, num_heads=2, seq_len=4, board_height=2, board_width=2))
    inner = m.inner
    with torch.no_grad():
        for p in inner.rates.parameters():
            p.add_(0.3 * torch.randn_like(p))
    x = torch.randn(1, 4, 8, dtype=torch.float64)
    s = torch.randn(2 * 4 * 8, dtype=torch.float64)

    def f(v):
        H, L = v.view(2, 1, 4, 8)
        H, L = inner.latent_recursion(H, L, x, {"cos_sin": None})
        return torch.cat([H.reshape(-1), L.reshape(-1)])

    J = torch.autograd.functional.jacobian(f, s)
    logdet = float(torch.linalg.slogdet(J)[1])
    expected = float(inner.log_volume_per_recursion(4).detach())
    # EqR's rms_norm runs in float32 internally, so the autograd Jacobian of the force is only
    # float32-accurate; a wrong volume formula would be off by O(1), not by 1e-8 relative.
    assert abs(logdet - expected) < 1e-6 * abs(expected), (logdet, expected)


@pytest.mark.parametrize("inject_h", [True, False])
def test_inject_h_breaks_the_stream_symmetry(inject_h):
    """With x in both forces, relax's L and H half-steps are the same map of (own, other): from H = L
    they give the same state, which is why round-1 relax streams merged. Without x in H's force they
    differ, and H's half-step no longer depends on x."""
    m = _coupled(coupling="relax", rate=3.0, leak=1.0, inject_h=inject_h)
    inner = m.inner
    torch.manual_seed(5)
    S = torch.randn(4, 9, 16, dtype=torch.float64)
    b = _batch()
    x = inner._input_embeddings(b["inputs"], b["puzzle_identifiers"])
    seq = {"cos_sin": None}
    with torch.no_grad():
        L1 = inner.half_step_L(S, S, x, seq)
        H1 = inner.half_step_H(S, S, x, seq)
        H1_other_x = inner.half_step_H(S, S, torch.randn_like(x), seq)
    if inject_h:
        assert (L1 - H1).abs().max() < 1e-12
        assert (H1 - H1_other_x).abs().max() > 1e-3
    else:
        assert (L1 - H1).abs().max() > 1e-3
        assert torch.equal(H1, H1_other_x)


def test_self_read_diagnostic_formula_and_no_inverse():
    """self_read: the forces also read the updated stream (EqR-style), so there is no exact inverse."""
    m = _coupled(coupling="relax", rate=3.0, leak=1.0, inject_h=False, self_read=True)
    inner = m.inner
    torch.manual_seed(7)
    H, L = torch.randn(4, 9, 16, dtype=torch.float64), torch.randn(4, 9, 16, dtype=torch.float64)
    b = _batch()
    x = inner._input_embeddings(b["inputs"], b["puzzle_identifiers"])
    seq = {"cos_sin": None}
    with torch.no_grad():
        g, a, c = inner.rates.gamma().double(), inner.rates.alpha().double(), inner.rates.c().double()
        L1 = inner.half_step_L(H, L, x, seq)
        ref = torch.exp(-g) * L + (1 - torch.exp(-g)) * (c * H + inner.L_level(L + H + x, **seq))
        assert (L1 - ref).abs().max() < 1e-12
        H1 = inner.half_step_H(H, L1, x, seq)
        ref = torch.exp(-a) * H + (1 - torch.exp(-a)) * (c * L1 + inner.L_level(H + L1, **seq))
        assert (H1 - ref).abs().max() < 1e-12
    with pytest.raises(RuntimeError):
        inner.inverse_half_step("L", H, L1, x, seq)
    with pytest.raises(RuntimeError):
        inner.log_volume_per_recursion(9)


def _relax(own, cross, f, rate, c):
    k = torch.exp(-rate)
    return k * own + (1 - k) * (c * cross + f)


@pytest.mark.parametrize("leak", [1.0, 2.0, 3.0, 10.0])
def test_twin_update_formula(leak):
    """twin_l: L and L' read each other where EqR's L reads itself, H reads their mean, EqR's order."""
    torch.manual_seed(0)
    m = CoupledEqRModel(dict(_cfg(coupling="relax", rate=3.0, leak=leak, inject_h=False, schedule="eqr",
                                  twin_l=True), L_cycles=4))
    inner = m.inner
    assert "".join(inner.schedule) == "LMLMH"
    carry = inner.empty_carry(4)
    assert carry.z_H.shape == (4, 9, 16) and carry.z_L.shape == (4, 18, 16)        # [L, L'] along seq
    torch.manual_seed(9)
    H, L, M = (torch.randn(4, 9, 16, dtype=torch.float64) for _ in range(3))
    b = _batch()
    x = inner._input_embeddings(b["inputs"], b["puzzle_identifiers"])
    seq = {"cos_sin": None}
    B = lambda u: inner.L_level(u, **seq)
    with torch.no_grad():
        g, a, c = inner.rates.gamma().double(), inner.rates.alpha().double(), inner.rates.c().double()
        assert abs(float(c) - math.exp(-leak)) < 1e-7                                # rates are float32 buffers
        h, l, mm = H, L, M
        for s in "LMLM":
            if s == "L":
                l = _relax(l, mm, B(mm + h + x), g, c)
            else:
                mm = _relax(mm, l, B(l + h + x), g, c)
        avg = 0.5 * (l + mm)
        h = _relax(h, avg, B(avg), a, c)                                             # no x in H's force
        H1, Z1 = inner.latent_recursion(H, torch.cat([L, M], dim=1), x, seq)
    assert (H1 - h).abs().max() < 1e-12
    assert (Z1 - torch.cat([l, mm], dim=1)).abs().max() < 1e-12
    with pytest.raises(RuntimeError):
        inner.log_volume_per_recursion(9)


def test_twin_large_leak_with_merged_copies_is_eqr_form():
    """twinL_leak10: with L = L' one twin update is a*L + (1-a)*B(L + H + x), EqR's z_L update with
    lambda = 1 - e^-3 (EqR: 0.95), up to the cross term (1-a)c*L with c = e^-10."""
    torch.manual_seed(0)
    m = CoupledEqRModel(dict(_cfg(coupling="relax", rate=3.0, leak=10.0, inject_h=False, schedule="eqr",
                                  twin_l=True), L_cycles=2))
    inner = m.inner
    torch.manual_seed(5)
    H, L = torch.randn(4, 9, 16, dtype=torch.float64), torch.randn(4, 9, 16, dtype=torch.float64)
    b = _batch()
    x = inner._input_embeddings(b["inputs"], b["puzzle_identifiers"])
    seq = {"cos_sin": None}
    with torch.no_grad():
        a = math.exp(-3.0)
        eqr_form = a * L + (1 - a) * inner.L_level(L + H + x, **seq)
        twin = inner.half_step_twin(L, L.clone(), H, x, seq)
    assert (twin - eqr_form).abs().max() < 1e-4 * L.abs().max()
    assert (twin - eqr_form).abs().max() > 0                                          # c = e^-10, not 0


def test_eqr_schedule_self_read_formula():
    """schedule='eqr' + self_read: EqR's inner loop (L_cycles L updates reading L, then H) in relax form."""
    torch.manual_seed(0)
    m = CoupledEqRModel(dict(_cfg(coupling="relax", rate=3.0, leak=1.0, inject_h=False, self_read=True,
                                  schedule="eqr"), L_cycles=3))
    inner = m.inner
    torch.manual_seed(4)
    H, L = torch.randn(4, 9, 16, dtype=torch.float64), torch.randn(4, 9, 16, dtype=torch.float64)
    b = _batch()
    x = inner._input_embeddings(b["inputs"], b["puzzle_identifiers"])
    seq = {"cos_sin": None}
    B = lambda u: inner.L_level(u, **seq)
    with torch.no_grad():
        g, a, c = inner.rates.gamma().double(), inner.rates.alpha().double(), inner.rates.c().double()
        l = L
        for _ in range(3):
            l = _relax(l, H, B(l + H + x), g, c)
        h = _relax(H, l, B(H + l), a, c)
        H1, L1 = inner.latent_recursion(H, L, x, seq)
    assert (L1 - l).abs().max() < 1e-12 and (H1 - h).abs().max() < 1e-12


@pytest.mark.parametrize("kw", [dict(schedule="eqr", twin_l=True, leak=1.0, inject_h=False),
                                dict(schedule="eqr", self_read=True, leak=1.0, inject_h=False)])
def test_act_forward_and_gradients_new_schedules(kw):
    """Full ACT steps through EqR's loss head (carry, random reset, z_H/z_L metrics) for the round-three arms."""
    from models.losses import ACTLossHead
    m = _coupled(coupling="relax", rate=3.0, dtype="float32", noise_scale=0.01, **kw)
    head = ACTLossHead(m, loss_type="stablemax_cross_entropy")
    head.train()
    batch = _batch()
    carry = head.initial_carry(batch)
    for _ in range(3):
        carry, loss, metrics, _, done = head(carry=carry, batch=batch, return_keys=[])
        loss.backward()
    n_l = 18 if kw.get("twin_l") else 9
    assert carry.inner_carry.z_L.shape == (4, n_l, 16)
    assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in metrics.values())
    grads = [p.grad for p in m.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(gr).all() for gr in grads)


def test_act_forward_and_gradients_through_loss_head():
    from models.losses import ACTLossHead
    m = _coupled(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.05, learn_rates=True,
                 dtype="float32", noise_scale=0.01)
    head = ACTLossHead(m, loss_type="stablemax_cross_entropy")
    head.train()
    batch = _batch()
    carry = head.initial_carry(batch)
    for _ in range(3):                                   # carry is detached between outer steps
        carry, loss, metrics, _, done = head(carry=carry, batch=batch, return_keys=[])
        loss.backward()
    assert carry.inner_carry.z_H.dtype == torch.float32
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.rate_parameters())
    assert torch.isfinite(loss)


def test_small_kick_rate_survives_in_float32_state():
    m = _coupled(coupling="kick", rate=0.005, dtype="bfloat16")
    assert m.inner.state_dtype == torch.float32
    k = torch.exp(-m.inner.rates.gamma())
    assert abs(float(k) - math.exp(-0.005)) < 1e-7                # bf16 would round it to 0.99609


def test_config_validation():
    for kw in (dict(coupling="nope"), dict(coupling="kick", leak=0.1), dict(coupling="relax", rate=0.0),
               dict(coupling="kick", rate=0.04, rate_constraint="positive", rate_min=0.05),
               dict(rate_constraint="bounded", rate=3.0, rate_min=0.1, rate_max=2.0),
               dict(schedule="nope"), dict(twin_l=True), dict(schedule="eqr", half_steps=4)):
        with pytest.raises(Exception):
            _coupled(**kw)
