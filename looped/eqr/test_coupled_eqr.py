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


@pytest.mark.parametrize("kw", [
    dict(coupling="kick", rate=0.005),
    dict(coupling="kick", rate=1.0, rate_constraint="positive", rate_min=0.05, learn_rates=True),
    dict(coupling="relax", rate=3.0),
    dict(coupling="relax", rate=3.0, leak=0.25),
    dict(coupling="relax", rate=0.7, rate_h=1.3, per_channel=True, learn_rates=True),
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
        fL = inner.L_level(L1 + x, **seq)
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
                                dict(coupling="relax", rate=2.0, per_channel=True, learn_rates=True)])
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
               dict(rate_constraint="bounded", rate=3.0, rate_min=0.1, rate_max=2.0)):
        with pytest.raises(Exception):
            _coupled(**kw)
