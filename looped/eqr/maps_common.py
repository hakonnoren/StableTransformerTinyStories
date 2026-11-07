"""Shared helpers for the map analysis of trained EqR / coupled-EqR Sudoku checkpoints.

Loads a checkpoint exported by export_for_analysis.py (or a full one) into EqR's own model classes
without EqR's data loaders, reads the exported test puzzles directly, and runs the iteration map.
All functions take the model's inner network (`inner`), whose call `inner(carry, batch)` is one
outer iteration (H_cycles latent recursions; one EqR ACT step) and returns (carry, logits, q_halt).
"""
import json
import os
import sys

import numpy as np
import torch


def setup_eqr(eqr_dir):
    eqr_dir = os.path.abspath(eqr_dir)
    os.chdir(eqr_dir)
    if eqr_dir not in sys.path:
        sys.path.insert(0, eqr_dir)


def pick_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(ckpt_path, data_dir, device, dtype="float32"):
    """EqR model (EMA weights) from a checkpoint; forward dtype overridden (float32 for analysis)."""
    from dataset.common import PuzzleDatasetMetadata
    from evaluate import _base_config
    from evaluators.eval_config import EvalConfig
    from pretrain import init_train_state
    from utils.checkpoint import load_ema, load_model_state_dict

    # EqR's evaluate.py / pretrain.py set TF32 matmuls on import ("high"): round-off ~1e-4 instead of float32's
    # ~1e-7, which hides how far a state has settled. The analysis wants full float32.
    torch.set_float32_matmul_precision("highest")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ck["config"]["arch"]["forward_dtype"] = dtype
    config = _base_config(EvalConfig(global_batch_size=64), ck, ckpt_path, rank=0)
    meta = PuzzleDatasetMetadata(**json.load(open(os.path.join(data_dir, "train", "dataset.json"))))
    train_state = init_train_state(config, meta, rank=0, world_size=1)
    load_model_state_dict(train_state.model, ck["model"], strict=False, assign=True)
    if config.ema and "ema" in ck:
        train_state = load_ema(ck, train_state, config, 0)
    model = train_state.model
    model.eval()
    inner = model.model.inner
    inner.to(device)
    for p in inner.parameters():
        p.data = p.data.to(torch.float32 if dtype == "float32" else p.dtype)
        p.requires_grad_(False)                         # analysis only: no autograd graphs (forward-mode JVPs only)
    if hasattr(inner, "state_dtype") and dtype == "float32":
        inner.state_dtype = torch.float32
    inner.forward_dtype = getattr(torch, dtype)
    return inner, config, int(ck.get("step", -1))


def load_puzzles(data_dir, n, start=0, device="cpu"):
    d = os.path.join(data_dir, "test")
    inp = np.load(os.path.join(d, "all__inputs.npy"))[start:start + n]
    lab = np.load(os.path.join(d, "all__labels.npy"))[start:start + n]
    ids = np.load(os.path.join(d, "all__puzzle_identifiers.npy"))[start:start + n]
    return {"inputs": torch.from_numpy(inp.astype(np.int64)).to(device),
            "labels": torch.from_numpy(lab.astype(np.int64)).to(device),
            "puzzle_identifiers": torch.from_numpy(ids.astype(np.int32)).to(device)}


def repeat_batch(batch, k):
    return {key: v.repeat_interleave(k, dim=0) for key, v in batch.items()}


def set_noise(inner, value):
    for m in inner.modules():
        if hasattr(m, "noise_scale"):
            m.noise_scale = float(value)


def random_carry(inner, b, device, gen=None):
    """EqR's random state reset (truncated normal, std = H/L_init_std) for b rows."""
    from models.common import trunc_normal_init_
    carry = inner.empty_carry(b, device=device)
    stds = getattr(inner, "_random_reset_field_stds", {"z_H": 1.0, "z_L": 1.0})
    for name in ("z_H", "z_L"):
        t = getattr(carry, name)
        if gen is None:
            trunc_normal_init_(t, std=float(stds[name]))
        else:
            t.copy_(torch.randn(t.shape, generator=gen, device="cpu").clamp_(-2, 2).mul_(float(stds[name])).to(t))
    return carry


def flat_state(carry):
    return torch.cat([carry.z_H, carry.z_L], dim=1).flatten(1).float()


def exact_and_cell(logits, labels):
    from models.losses import IGNORE_LABEL_ID
    pred = logits.argmax(-1)
    mask = labels != IGNORE_LABEL_ID
    ok = (pred == labels) | ~mask
    return ok.all(-1), ((pred == labels) & mask).sum(-1).float() / mask.sum(-1).float(), pred


def stream_views(inner, carry):
    """Named per-stream tensors: H, L (and the twin L' for twin_l models)."""
    out = {"H": carry.z_H}
    if getattr(inner, "twin_l", False):
        out["L"], out["L2"] = carry.z_L.chunk(2, dim=1)
    else:
        out["L"] = carry.z_L
    return out
