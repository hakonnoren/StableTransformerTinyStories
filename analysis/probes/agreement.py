"""
Score minimal-pair stimuli, and measure at which depth the decision resolves.

Two measurements per item:
  accuracy  — is logP(target) > logP(foil) at the decision position?
  margin    — logP(target) - logP(foil), a continuous version that does not throw
              away magnitude and so needs fewer items for the same power.

And optionally, per layer, the same margin read through the lens (plan E1):
  resolution depth — the first layer where the margin turns positive AND STAYS
  positive to the last layer. "Stays" matters: a margin that flickers positive
  mid-stack and then goes negative has not resolved anything.

Resolution depth is computed on a subsample (`lens_items`) because it costs one
lm_head projection per layer per item.
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from ..lenses import lens_logits
from ..loader import LensPolicy, forward_logits, layer_states, supports_policy


def load_stimuli(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def encode_prompts(items: list[dict], enc) -> list[list[int]]:
    return [enc.encode_ordinary(it["prompt"]) for it in items]


@torch.no_grad()
def score_items(arm, items: list[dict], prompts: list[list[int]], device: str = "cpu",
                batch_size: int = 64, pad_id: int = 50256,
                verbose: bool = True) -> list[dict]:
    """One record per item: accuracy flag and margin at the decision position.

    Right-padding is safe: causal attention means the prompt's last real position
    cannot see the pad tokens that follow it. We gather logits at len-1 per row.
    """
    recs = []
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        pchunk = prompts[i:i + batch_size]
        T = max(len(p) for p in pchunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        at = torch.empty(len(chunk), dtype=torch.long)
        for j, p in enumerate(pchunk):
            ids[j, :len(p)] = torch.tensor(p, dtype=torch.long)
            at[j] = len(p) - 1
        logits = forward_logits(arm.model, ids.to(device)).float()
        lp = F.log_softmax(logits[torch.arange(len(chunk)), at.to(device)], dim=-1).cpu()

        for j, it in enumerate(chunk):
            t = lp[j, it["target_id"]].item()
            f = lp[j, it["foil_id"]].item()
            recs.append({
                "arm": arm.name, "item_id": it["item_id"], "paradigm": it["paradigm"],
                "cell": it["cell"], "n_attractors": it["n_attractors"],
                "attractor_match": it["attractor_match"], "number": it["number"],
                "correct": float(t > f), "margin": t - f,
                "logp_target": t, "logp_foil": f,
            })
        if verbose and (i // batch_size) % 25 == 0:
            print(f"    {arm.name}: {min(i+batch_size, len(items))}/{len(items)} items",
                  flush=True)
    return recs


@torch.no_grad()
def resolution_depth(arm, items: list[dict], prompts: list[list[int]], device: str = "cpu",
                     batch_size: int = 32, tuned=None, policy: str = LensPolicy.SUM,
                     pad_id: int = 50256, verbose: bool = True) -> list[dict]:
    """Per-layer margin and the resulting resolution depth, per item."""
    if not supports_policy(arm.model, policy):
        return []
    recs = []
    n_layer = len(arm.model.blocks)
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        pchunk = prompts[i:i + batch_size]
        T = max(len(p) for p in pchunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        at = torch.empty(len(chunk), dtype=torch.long)
        for j, p in enumerate(pchunk):
            ids[j, :len(p)] = torch.tensor(p, dtype=torch.long)
            at[j] = len(p) - 1
        ids = ids.to(device)
        at_d = at.to(device)
        states = layer_states(arm.model, ids)
        rows = torch.arange(len(chunk))

        margins = np.zeros((len(chunk), n_layer), dtype=np.float32)
        for li, h in enumerate(states):
            hl = h[rows.to(device), at_d]                    # (B, d) at decision pos
            logits = lens_logits(arm.model, tuned, li, hl, policy).float()
            lp = F.log_softmax(logits, dim=-1).cpu()
            for j, it in enumerate(chunk):
                margins[j, li] = lp[j, it["target_id"]].item() - lp[j, it["foil_id"]].item()

        for j, it in enumerate(chunk):
            m = margins[j]
            pos = m > 0
            # first layer that is positive and stays positive through the last layer
            depth = n_layer
            for li in range(n_layer):
                if pos[li:].all():
                    depth = li
                    break
            recs.append({
                "arm": arm.name, "item_id": it["item_id"], "cell": it["cell"],
                "n_attractors": it["n_attractors"],
                "attractor_match": it["attractor_match"],
                "policy": policy, "lens": "tuned" if tuned is not None else "raw",
                "resolution_depth": int(depth),
                "resolved": bool(depth < n_layer),
                "final_margin": float(m[-1]),
                "margins": [float(v) for v in m],
            })
        if verbose and (i // batch_size) % 10 == 0:
            print(f"    {arm.name} depth: {min(i+batch_size, len(items))}/{len(items)}",
                  flush=True)
    return recs
