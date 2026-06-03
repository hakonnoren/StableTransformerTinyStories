"""
Cheap training diagnostics (forward/backward only — no Jacobian/SVD), logged to
Weights & Biases. Adapted from the run_para.py `is_cheap` callback to this repo's
models (GPTModel / RevFormerModel; blocks live in `model.blocks`, attention in
`block.attn`). Metric groups:

  1. token accuracy on a fixed eval batch (loss/val_loss are logged elsewhere)
  2. effective alpha/gamma per reversible block (log|det| contributions)   [reversible only]
  3. (ReZero resweights — N/A: these models have no ReZero gates)
  4. activation-input grad norms  ‖dL/d(residual stream)‖   [persistent backward hooks]
  5. weight grad norms per block                              [param.grad read]
  6. attention distribution stats (entropy/support/top1/hhi/k90/key-sink)  [fwd capture, fixed batch]
  7. representation geometry (act_rms, act_erank, act_token_sim)           [fwd hooks, fixed batch]

Cadence: groups 4/5 accumulate every optimizer step and flush at each eval;
groups 1/2/6/7 are one-shot on a fixed eval batch at each eval.

W&B format: per-layer scalars (e.g. ``cheap/actgrad/L0``) plus one
``wandb.Histogram`` per metric over the layer/head axis (``cheap/actgrad_hist``).
"""

import math
from typing import Optional

import torch


# ----------------------------------------------------------------------------
# Group 4 & 5: gradient norms (persistent hooks + param.grad reads)
# ----------------------------------------------------------------------------
class GradNormTracker:
    """Per-block activation-input grad norms (#4, via backward hooks) and weight
    grad norms (#5, via param.grad). Accumulates per optimizer step; ``flush()``
    returns the mean over accumulated steps and resets."""

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.n_layer = len(self.blocks)
        # within-step accumulators (averaged over micro-steps)
        self._cur_sum = [0.0] * self.n_layer
        self._cur_cnt = [0] * self.n_layer
        # cross-step accumulators (averaged at flush)
        self._act_sum = [0.0] * self.n_layer
        self._wgt_sum = [0.0] * self.n_layer
        self._steps = 0
        self._handles = []
        for i, blk in enumerate(self.blocks):
            self._handles.append(blk.register_full_backward_hook(self._make_hook(i)))

    def _make_hook(self, idx):
        def hook(_module, _grad_input, grad_output):
            g = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
            if g is None:
                return
            # per-sequence L2 norm over (T, C), then mean over batch
            with torch.no_grad():
                per_seq = g.reshape(g.shape[0], -1).norm(dim=1)
                self._cur_sum[idx] += float(per_seq.mean())
                self._cur_cnt[idx] += 1
        return hook

    @torch.no_grad()
    def next_step(self):
        """Call once per optimizer step, after backward and before zero_grad."""
        for i, blk in enumerate(self.blocks):
            # finalize activation-grad for this step (mean over micro-steps)
            if self._cur_cnt[i] > 0:
                self._act_sum[i] += self._cur_sum[i] / self._cur_cnt[i]
            # weight grad norm for this block
            tot = 0.0
            for p in blk.parameters():
                if p.grad is not None:
                    tot += float(p.grad.detach().pow(2).sum())
            self._wgt_sum[i] += math.sqrt(tot)
            self._cur_sum[i] = 0.0
            self._cur_cnt[i] = 0
        self._steps += 1

    def flush(self):
        """Return ({'actgrad': [...], 'wgrad': [...]}) averaged over accumulated
        steps, and reset. Empty dict if no steps accumulated."""
        if self._steps == 0:
            return {}
        act = [s / self._steps for s in self._act_sum]
        wgt = [s / self._steps for s in self._wgt_sum]
        self._act_sum = [0.0] * self.n_layer
        self._wgt_sum = [0.0] * self.n_layer
        self._steps = 0
        return {"actgrad": act, "wgrad": wgt}

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []


# ----------------------------------------------------------------------------
# Group 2: effective alpha/gamma per reversible block
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_alpha_gamma(model):
    """Per-block effective gamma/alpha for reversible models. Returns dict of
    per-layer lists, or {} if the model has no reversible blocks."""
    blocks = getattr(model, "blocks", [])
    if not blocks or not hasattr(blocks[0], "_effective_gamma_alpha"):
        return {}
    # Mirror the model's centering: vpm uses a global avg correction; others 0.
    avg = 0.0
    rev_cfg = getattr(model, "rev_cfg", None)
    if rev_cfg is not None and getattr(rev_cfg, "regime", "") == "vpm_scaling":
        # full-block T is irrelevant to per-block gamma/alpha means; use T=1.
        avg = model._avg_corr(1)
    gamma_mean, alpha_mean, logdet = [], [], []
    for blk in blocks:
        g, a = blk._effective_gamma_alpha(avg)
        gamma_mean.append(float(g.mean()))
        alpha_mean.append(float(a.mean()))
        # per-position log|det| contribution of this block = -(sum gamma + sum alpha)
        logdet.append(float(-(g.sum() + a.sum())))
    return {"gamma_mean": gamma_mean, "alpha_mean": alpha_mean, "block_logdet": logdet}


# ----------------------------------------------------------------------------
# Group 6: attention distribution stats (one-shot on fixed batch)
# ----------------------------------------------------------------------------
class _AttnCapture:
    """Attaches `_capture_attn` to each block's attention and reduces the
    post-softmax matrix to per-layer scalars (means over batch/head/query)."""

    def __init__(self, blocks):
        self.blocks = list(blocks)
        self.n_layer = len(self.blocks)
        self._buf = [None] * self.n_layer

    def _make(self, idx):
        def capture(att):  # att: (B, H, T, T), causal-masked entries are exactly 0
            with torch.no_grad():
                B, H, T, _ = att.shape
                eps = 1e-30
                dev, dt = att.device, att.dtype
                k = torch.arange(1, T + 1, device=dev, dtype=dt).view(1, 1, T)  # allowed keys per query
                logp = torch.log(att.clamp_min(eps))
                ent = -(att * logp).sum(-1)                       # (B,H,T)
                eff_support = ent.exp()
                log_k = torch.log(k.clamp_min(1.0 + eps))
                entropy_norm = torch.where(k > 1, ent / log_k.clamp_min(eps), torch.zeros_like(ent))
                top1 = att.max(-1).values
                hhi = (att * att).sum(-1)
                inv_k = 1.0 / k
                hhi_norm = torch.where(k > 1, (hhi - inv_k) / (1.0 - inv_k).clamp_min(eps),
                                       torch.ones_like(hhi))
                sorted_p = torch.sort(att, dim=-1, descending=True).values
                k90 = ((sorted_p.cumsum(-1) < 0.9).sum(-1).clamp_max(T - 1) + 1).to(dt)
                n_valid = torch.arange(T, 0, -1, device=dev, dtype=dt).view(1, 1, T)
                key_mass_max = (att.sum(-2) / n_valid).amax(dim=-1)  # (B,H) sink summary
                # reduce to a per-HEAD value (mean over batch and query>0); pos 0 is
                # the trivial deterministic causal row, excluded.
                def mh(x):  # x: (B,H,T) -> (H,) mean over batch and query>0
                    xx = x[..., 1:] if T > 1 else x
                    return xx.mean(dim=(0, 2))
                self._buf[idx] = {
                    "attn_entropy_norm": mh(entropy_norm).detach().cpu(),
                    "attn_eff_support": mh(eff_support).detach().cpu(),
                    "attn_top1": mh(top1).detach().cpu(),
                    "attn_hhi_norm": mh(hhi_norm).detach().cpu(),
                    "attn_k90": mh(k90).detach().cpu(),
                    "attn_key_mass_max": key_mass_max.mean(dim=0).detach().cpu(),  # (H,)
                }
        return capture

    def attach(self):
        for i, blk in enumerate(self.blocks):
            blk.attn._capture_attn = self._make(i)

    def detach(self):
        for blk in self.blocks:
            if hasattr(blk.attn, "_capture_attn"):
                blk.attn._capture_attn = None

    def reduce_perhead(self):
        """Return {stat: [per-layer list of per-head values]} from captured
        buffers. Layers whose block missed a capture get an empty list."""
        stats = ("attn_entropy_norm", "attn_eff_support", "attn_top1",
                 "attn_hhi_norm", "attn_k90", "attn_key_mass_max")
        out = {s: [[] for _ in range(self.n_layer)] for s in stats}
        for i, b in enumerate(self._buf):
            if b is not None:
                for s in stats:
                    out[s][i] = b[s].tolist()
        return out


# ----------------------------------------------------------------------------
# Group 7: representation geometry (one-shot on fixed batch)
# ----------------------------------------------------------------------------
@torch.no_grad()
def _repr_from_outputs(outs):
    """outs: list of per-block output tensors (B,T,C). Returns per-layer dict."""
    rms, erank, tok_sim = [], [], []
    for x in outs:
        B, T, C = x.shape
        M = x.reshape(B * T, C).float()
        rms.append(float(M.pow(2).mean().sqrt()))
        # effective rank of the centered covariance (Roy & Vetterli entropy rank)
        Mc = M - M.mean(0, keepdim=True)
        cov = (Mc.t() @ Mc) / max(1, Mc.shape[0])
        ev = torch.linalg.eigvalsh(cov).clamp_min(0.0)
        s = ev.sum()
        if float(s) <= 0:
            erank.append(float("nan"))
        else:
            p = ev / s
            h = -(p * (p + 1e-12).log()).sum()
            erank.append(float(h.exp()))
        # common-mode / rank-collapse: ||mean of unit token vectors|| in [0,1]
        u = M / (M.norm(dim=1, keepdim=True) + 1e-12)
        tok_sim.append(float(u.mean(0).norm()))
    return {"act_rms": rms, "act_erank": erank, "act_token_sim": tok_sim}


@torch.no_grad()
def representation_and_accuracy(model, x, y):
    """One no-grad forward over the fixed batch: capture per-block outputs for
    representation geometry, and compute next-token accuracy. Returns
    (repr_dict, token_accuracy)."""
    blocks = list(getattr(model, "blocks", []))
    outs = [None] * len(blocks)
    handles = []
    for i, blk in enumerate(blocks):
        def mk(idx):
            def hook(_m, _inp, out):
                outs[idx] = out.detach()
            return hook
        handles.append(blk.register_forward_hook(mk(i)))
    was_training = model.training
    model.eval()
    try:
        logits, _ = model(x)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    acc = float((logits.argmax(-1) == y).float().mean())
    rep = _repr_from_outputs([o for o in outs if o is not None])
    return rep, acc


# ----------------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------------
class CheapMetrics:
    """Owns the persistent grad hooks and the fixed eval batch; produces a
    snapshot of all cheap metrics and logs it to W&B (or prints a summary)."""

    def __init__(self, model, fixed_x, fixed_y, wandb_run=None,
                 track_grads=True, track_attn=True, track_repr=True):
        self.model = model
        self.x = fixed_x
        self.y = fixed_y
        self.wandb_run = wandb_run
        self.track_attn = track_attn
        self.track_repr = track_repr
        self.blocks = list(getattr(model, "blocks", []))
        self.has_attn = bool(self.blocks) and all(hasattr(b, "attn") for b in self.blocks)
        self.grad = GradNormTracker(self.blocks) if (track_grads and self.blocks) else None

    def on_optim_step(self):
        if self.grad is not None:
            self.grad.next_step()

    def _log(self, step, perlayer, perhead=None):
        """perlayer: {name: [per-layer values] or scalar}. perhead: {name:
        [per-layer list of per-head values]}. Logs per-layer scalars + a
        histogram per multi-value metric to W&B (per-head metrics also get a
        per-layer head histogram and a pooled head histogram). Prints if no
        wandb."""
        perhead = perhead or {}
        if self.wandb_run is not None:
            import wandb
            payload = {}
            for name, val in perlayer.items():
                if isinstance(val, (list, tuple)):
                    for i, v in enumerate(val):
                        payload[f"cheap/{name}/L{i}"] = v
                    finite = [v for v in val if v == v]  # drop NaN
                    if finite:
                        payload[f"cheap/{name}_hist"] = wandb.Histogram(finite)
                        payload[f"cheap/{name}_mean"] = sum(finite) / len(finite)
                else:
                    payload[f"cheap/{name}"] = val
            for name, layers in perhead.items():
                pooled = []
                for i, heads in enumerate(layers):
                    finite = [h for h in heads if h == h]
                    if not finite:
                        continue
                    pooled.extend(finite)
                    # per-layer scalar (mean over heads) for line plots
                    payload[f"cheap/{name}/L{i}"] = sum(finite) / len(finite)
                    # per-layer histogram over this layer's heads
                    payload[f"cheap/{name}/L{i}_heads"] = wandb.Histogram(finite)
                if pooled:
                    payload[f"cheap/{name}_mean"] = sum(pooled) / len(pooled)
                    payload[f"cheap/{name}_head_hist"] = wandb.Histogram(pooled)  # all heads, all layers
            self.wandb_run.log(payload, step=step)
        else:
            bits = []
            for name, val in perlayer.items():
                if isinstance(val, (list, tuple)):
                    finite = [v for v in val if v == v]
                    if finite:
                        bits.append(f"{name}~{sum(finite)/len(finite):.3g}")
                else:
                    bits.append(f"{name}={val:.3g}")
            for name, layers in perhead.items():
                pooled = [h for heads in layers for h in heads if h == h]
                if pooled:
                    bits.append(f"{name}~{sum(pooled)/len(pooled):.3g}(/{len(pooled)}h)")
            print(f"[cheap][step {step}] " + " | ".join(bits))

    @torch.no_grad()
    def snapshot(self, step):
        perlayer = {}
        perhead = {}

        # #4/#5 grad norms (flush accumulators)
        if self.grad is not None:
            perlayer.update(self.grad.flush())

        # #2 alpha/gamma (reversible only)
        perlayer.update(extract_alpha_gamma(self.model))

        # #6 attention stats (one-shot on fixed batch), kept per head
        if self.track_attn and self.has_attn:
            cap = _AttnCapture(self.blocks)
            cap.attach()
            was_training = self.model.training
            self.model.eval()
            try:
                self.model(self.x)
            finally:
                cap.detach()
                if was_training:
                    self.model.train()
            perhead.update(cap.reduce_perhead())

        # #1 accuracy + #7 representation geometry (one-shot on fixed batch)
        if self.track_repr:
            rep, acc = representation_and_accuracy(self.model, self.x, self.y)
            perlayer.update(rep)
            perlayer["token_accuracy"] = acc

        self._log(step, perlayer, perhead)

    def close(self):
        if self.grad is not None:
            self.grad.close()
