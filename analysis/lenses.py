"""
Depth readouts: raw logit lens and tuned lens.

Why the tuned lens is not optional here. The raw logit lens reads an intermediate
state through the model's FINAL unembedding, which is only valid if the residual
basis is stable across depth. On revparityreg_24957997 it reported 25.17 and 15.46
nats at layer 0 for vpm_scaling / vpb_scaling — values that almost certainly say
"this arm's early residual basis differs from its final one", not "this arm
predicts terribly at L0". The tuned lens (Belrose et al. 2023, arXiv 2303.08112)
fits a per-layer affine translator on frozen activations and removes exactly that
confound. Report both: the RAW-MINUS-TUNED GAP IS ITSELF THE DIAGNOSTIC, and it is
plausibly largest for the volume-scaling regimes.

Translator parameterisation follows the paper: T_l(h) = h + A_l h + b_l with A_l
initialised to zero, so the lens starts exactly at the raw logit lens and can only
improve on it. Training objective is KL(p_final || p_lens) — matching the model's
own final distribution, which is what makes it a *lens* on the model rather than
just a probe fitted to the labels. Set objective="ce" for cross-entropy against
the true next token instead.

Cost note: the lm_head projection to 50304 classes for every layer is the
expensive part, so training subsamples token positions per batch (`n_pos`). That
is standard and keeps a 12-layer fit to minutes rather than hours.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .loader import LensPolicy, bridge, layer_states, forward_logits


class TunedLens(nn.Module):
    """Per-layer affine translators into the final residual basis."""

    def __init__(self, n_layer: int, d_model: int):
        super().__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.translators = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layer)])
        for lin in self.translators:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        return h + self.translators[layer](h)

    @torch.no_grad()
    def identity_check(self) -> float:
        """Max |A| over layers — 0.0 means the lens is still the raw logit lens."""
        return max(float(l.weight.abs().max()) for l in self.translators)


def lens_logits(model, lens: TunedLens | None, layer: int, h: torch.Tensor,
                policy: str = LensPolicy.SUM) -> torch.Tensor:
    """Early-exit logits for a block state, optionally through a tuned lens."""
    x = bridge(model, h, policy)
    if lens is not None:
        x = lens(layer, x)
    return model.lm_head(model.ln_f(x))


def train_tuned_lens(model, stories, device: str = "cpu", policy: str = LensPolicy.SUM,
                     steps: int = 300, batch_size: int = 4, n_pos: int = 128,
                     lr: float = 1e-3, objective: str = "kl", seed: int = 0,
                     log_every: int = 50, verbose: bool = True) -> TunedLens:
    """Fit translators on frozen activations from `stories`.

    The model is frozen throughout (its params get requires_grad=False); gradients
    reach only the translators. Activations are captured under no_grad and enter as
    constants, so this is a convex-ish regression per layer and converges fast.
    """
    from .corpus import pad_batch, token_mask

    if objective not in ("kl", "ce"):
        raise ValueError(f"objective must be 'kl' or 'ce', got {objective!r}")

    for p in model.parameters():
        p.requires_grad_(False)

    d_head = model.cfg.n_embd if getattr(model, "narrow_emb", False) else \
        (model.ln_f.weight.shape[0])
    lens = TunedLens(len(model.blocks), d_head).to(device)
    opt = torch.optim.Adam(lens.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)

    n = len(stories)
    for step in range(steps):
        pick = torch.randint(0, n, (batch_size,), generator=gen).tolist()
        batch = [stories[i] for i in pick]
        ids, lens_t = pad_batch(batch, device=device)
        mask = token_mask(lens_t, ids.shape[1])

        with torch.no_grad():
            states = layer_states(model, ids)
            final = forward_logits(model, ids).float()
            p_final = F.log_softmax(final[:, :-1], dim=-1)

        # score only real positions that have a real target
        valid = (mask[:, :-1] & mask[:, 1:]).reshape(-1).nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            continue
        sel = valid[torch.randperm(valid.numel(), generator=gen)[:n_pos].to(valid.device)]
        tgt = ids[:, 1:].reshape(-1)[sel]
        tgt_lp = p_final.reshape(-1, p_final.shape[-1])[sel]

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for li, h in enumerate(states):
            hb = bridge(model, h, policy)[:, :-1]
            hb = hb.reshape(-1, hb.shape[-1])[sel]
            logits = model.lm_head(model.ln_f(lens(li, hb))).float()
            if objective == "kl":
                # KL(p_final || p_lens), i.e. match the model's own prediction
                loss = F.kl_div(F.log_softmax(logits, -1), tgt_lp,
                                reduction="batchmean", log_target=True)
            else:
                loss = F.cross_entropy(logits, tgt)
            loss.backward()
            total += float(loss.detach())
        opt.step()

        if verbose and (step % log_every == 0 or step == steps - 1):
            print(f"    tuned-lens step {step+1}/{steps}  mean {objective}/layer="
                  f"{total/len(states):.4f}", flush=True)

    return lens


@torch.no_grad()
def eval_lens_kl(model, lens: TunedLens, stories, device: str = "cpu",
                 policy: str = LensPolicy.SUM, batch_size: int = 4,
                 n_batches: int = 6, n_pos: int = 256, seed: int = 0) -> list[dict]:
    """Per-layer KL(p_final || p_lens) for the raw and tuned lens.

    The RAW-MINUS-TUNED GAP is the basis-change diagnostic: an affine translator can
    only fix a change of basis, so whatever it removes was never missing information
    in the first place. What remains after fitting is the layer genuinely not
    encoding the final prediction.

    Returns one row per (layer, lens), so the gap can be read as a function of depth
    rather than collapsed to a single mean.
    """
    from .corpus import pad_batch, token_mask

    gen = torch.Generator().manual_seed(seed)
    n_layer = len(model.blocks)
    acc = {("raw", l): [] for l in range(n_layer)}
    acc.update({("tuned", l): [] for l in range(n_layer)})

    for bi in range(min(n_batches, max(1, len(stories) // batch_size))):
        batch = stories[bi * batch_size:(bi + 1) * batch_size]
        if not batch:
            break
        ids, lens_t = pad_batch(batch, device=device)
        mask = token_mask(lens_t, ids.shape[1])
        states = layer_states(model, ids)
        final = forward_logits(model, ids).float()
        p_final = F.log_softmax(final[:, :-1], dim=-1)

        valid = (mask[:, :-1] & mask[:, 1:]).reshape(-1).nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            continue
        sel = valid[torch.randperm(valid.numel(), generator=gen)[:n_pos].to(valid.device)]
        tgt_lp = p_final.reshape(-1, p_final.shape[-1])[sel]

        for li, h in enumerate(states):
            hb = bridge(model, h, policy)[:, :-1]
            hb = hb.reshape(-1, hb.shape[-1])[sel]
            for name, ln in (("raw", None), ("tuned", lens)):
                x = ln(li, hb) if ln is not None else hb
                logits = model.lm_head(model.ln_f(x)).float()
                kl = F.kl_div(F.log_softmax(logits, -1), tgt_lp,
                              reduction="batchmean", log_target=True)
                acc[(name, li)].append(float(kl))

    out = []
    for (name, li), vals in acc.items():
        if vals:
            out.append({"layer": li, "lens": name, "kl": float(np.mean(vals)),
                        "n_batches": len(vals)})
    return out


def translator_magnitudes(lens: TunedLens) -> list[dict]:
    """Per-layer size of the affine correction, relative to the identity map.

    rel_A = ||A_l||_F / ||I||_F, so 0 means "no correction needed" (the layer already
    lives in the final basis) and 1 means the correction is as large as the identity.
    Needs no data — it is a property of the fitted lens.
    """
    import math
    out = []
    for li, lin in enumerate(lens.translators):
        A = lin.weight.detach()
        out.append({"layer": li,
                    "rel_A": float(A.norm()) / math.sqrt(A.shape[0]),
                    "bias_norm": float(lin.bias.detach().norm())})
    return out


def save_lens(lens: TunedLens, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": lens.state_dict(), "n_layer": lens.n_layer,
                "d_model": lens.d_model}, path)


def load_lens(path: str, device: str = "cpu") -> TunedLens:
    ck = torch.load(path, map_location=device, weights_only=False)
    lens = TunedLens(ck["n_layer"], ck["d_model"])
    lens.load_state_dict(ck["state_dict"])
    return lens.to(device).eval()
