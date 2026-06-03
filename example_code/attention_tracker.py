"""Track per-row distributional statistics of post-softmax attention.

Snapshot-only tracker (mirrors the SV / task-SV cadence in run_para.py): for a
given evaluation batch, computes summaries of the row-stochastic attention
matrices in every block, never materialising the full (B, n_head, T, T) tensor
beyond the in-hook reduction.

Attaches by setting the ``_capture_attn`` callable on each ``CausalSelfAttention``
module (added in mingpt/model.py); the model forward calls it with the
post-softmax pre-dropout matrix.

Per-row stats (shape ``(n_layer, B, n_head, T_query)``), with ``k_i = i+1`` the
size of the causal-allowed key set for query position ``i``:

    Primary (recommended dashboard):
        entropy_norm  = H(p_i) / log(k_i)        # ∈ [0, 1]; 0 sharp, 1 uniform
        eff_support   = exp(H(p_i))              # ∈ [1, k_i]; effective # tokens attended
        eff_fraction  = exp(H) / k_i             # ∈ [1/k_i, 1]; mask-normalised version
        top1          = max_j p_{i,j}            # = p_max
        top5_mass     = Σ over top-5 entries     # mass on the 5 most-attended keys
        k90           = min k s.t. top-k mass ≥ .9   # # tokens covering 90% mass
        hhi_norm      = (Σ p² − 1/k_i)/(1 − 1/k_i)   # 0 uniform, 1 one-hot

    Compatibility / detail:
        mean          = 1/k_i (positional only — kept for backward compat)
        std           = within-row std over the k_i unmasked entries

Per-column stat (key sink detection), shape ``(n_layer, B, n_head, T_key)``:

    key_mass[j]   = mean attention received by key j across queries that can
                    see it: (1/(T−j)) · Σ_{i ≥ j} A[i, j].
                    A "sink" appears as a single column with disproportionate
                    mass; ``key_mass.max(-1)`` per (layer, batch, head) is a
                    convenient sink summary.

Pos-0 (the trivial deterministic causal row) is included in storage and
filtered at plot time.

Usage:

    tracker = AttentionTracker(model)
    tracker.attach()
    snapshot = tracker.capture(model, x_eval)
    # snapshot = {'attn_entropy_norm': (n_layer, B, n_head, T), ...}
    tracker.detach()
"""
import numpy as np
import torch


# Per-row stats stored on the (B, n_head, T_query) axis.
ROW_STATS = ('entropy_norm', 'eff_support', 'eff_fraction', 'top1',
             'top5_mass', 'k90', 'hhi_norm', 'mean', 'std')

# Per-column stats stored on the (B, n_head, T_key) axis.
COL_STATS = ('key_mass',)


class AttentionTracker:
    DEFAULT_STATS = ROW_STATS + COL_STATS

    def __init__(self, model, stats=DEFAULT_STATS):
        self.blocks = list(model.transformer.h)
        self.n_layer = len(self.blocks)
        self.stats = tuple(stats)
        self._buffers = [None] * self.n_layer

    def _make_capture(self, idx):
        def capture(att):
            # att: (B, n_head, T, T) post-softmax. Causal-masked entries are exactly 0
            # (softmax of -inf), so summing/squaring them contributes nothing.
            with torch.no_grad():
                B, H, T, _ = att.shape
                eps = 1e-30
                dtype = att.dtype
                device = att.device

                # k_i = i+1, the size of the causal-allowed key set for query i.
                k = torch.arange(1, T + 1, device=device, dtype=dtype).view(1, 1, T)
                causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=dtype))

                # ---- Entropy & derived ----
                # 0·log(0) := 0 by convention; masked entries (att=0) zero out
                # the log-of-eps term, so the entropy is computed only over
                # unmasked keys without an explicit gating multiply.
                logp = torch.log(att.clamp_min(eps))
                ent = -(att * logp).sum(dim=-1)                              # (B, H, T)
                eff_support = ent.exp()                                      # (B, H, T)
                # entropy_norm: H / log(k_i). Row 0 (k=1) ⇒ log(k)=0; define =0.
                log_k = torch.log(k.clamp_min(1.0 + eps))
                entropy_norm = torch.where(
                    k > 1, ent / log_k.clamp_min(eps), torch.zeros_like(ent)
                )
                eff_fraction = eff_support / k                               # ∈ [1/k_i, 1]

                # ---- p_max & top-k mass ----
                top1 = att.max(dim=-1).values                                # (B, H, T)
                k_top = min(5, T)
                top5_mass = torch.topk(att, k=k_top, dim=-1).values.sum(dim=-1)  # (B, H, T)

                # ---- k90: smallest k whose top-k mass ≥ 0.9 ----
                sorted_p = torch.sort(att, dim=-1, descending=True).values
                cdf = sorted_p.cumsum(dim=-1)
                # (cdf < 0.9).sum gives # of entries strictly below the 90% line; +1
                # is the smallest k that crosses it. Capped at T.
                k90 = (cdf < 0.9).sum(dim=-1).clamp_max(T - 1) + 1
                k90 = k90.to(dtype)                                          # (B, H, T)

                # ---- Normalized HHI ----
                hhi = (att * att).sum(dim=-1)                                # (B, H, T) ∈ [1/k_i, 1]
                inv_k = 1.0 / k
                hhi_norm = torch.where(
                    k > 1,
                    (hhi - inv_k) / (1.0 - inv_k).clamp_min(eps),
                    torch.ones_like(hhi),
                )                                                            # 0 uniform, 1 one-hot

                # ---- Per-row mean & std (numerically stable centered formula) ----
                row_mean_unsq = inv_k.unsqueeze(-1)                          # (1,1,T,1)
                centered = (att - row_mean_unsq) * causal_mask
                row_var = ((centered * centered).sum(dim=-1) / k).clamp_min(0.0)
                row_std = row_var.sqrt()                                     # (B, H, T)
                row_mean_b = inv_k.expand(B, H, T)                           # (B, H, T)

                # ---- Column-wise key mass (sink detection) ----
                # n_valid_queries_per_col[j] = T - j   (queries i ∈ [j, T-1] can see j).
                n_valid_queries_per_col = torch.arange(T, 0, -1, device=device, dtype=dtype).view(1, 1, T)
                key_mass = att.sum(dim=-2) / n_valid_queries_per_col         # (B, H, T_key)

                self._buffers[idx] = {
                    'entropy_norm':  entropy_norm.detach().cpu(),
                    'eff_support':   eff_support.detach().cpu(),
                    'eff_fraction':  eff_fraction.detach().cpu(),
                    'top1':          top1.detach().cpu(),
                    'top5_mass':     top5_mass.detach().cpu(),
                    'k90':           k90.detach().cpu(),
                    'hhi_norm':      hhi_norm.detach().cpu(),
                    'mean':          row_mean_b.detach().cpu(),
                    'std':           row_std.detach().cpu(),
                    'key_mass':      key_mass.detach().cpu(),
                }
        return capture

    def attach(self):
        for i, block in enumerate(self.blocks):
            block.attn._capture_attn = self._make_capture(i)

    def detach(self):
        for block in self.blocks:
            if hasattr(block.attn, '_capture_attn'):
                block.attn._capture_attn = None

    def reset(self):
        self._buffers = [None] * self.n_layer

    def capture(self, model, x_eval) -> dict:
        """One-shot capture: run a no-grad forward over ``x_eval`` and return
        per-stat ndarrays of shape ``(n_layer, B, n_head, T)``.

        If a block missed (e.g. an architecture that bypasses attention), its
        slice is filled with NaN. The tracker must be attached before calling.
        """
        was_training = model.training
        model.eval()
        try:
            self.reset()
            with torch.no_grad():
                model(x_eval)

            # Reference shape from the first populated buffer.
            ref = next((b for b in self._buffers if b is not None), None)
            out = {}
            if ref is None:
                return {f'attn_{s}': np.zeros((self.n_layer, 0, 0, 0), dtype=np.float32)
                        for s in self.stats}
            for stat in self.stats:
                shape = ref[stat].shape  # (B, H, T)
                arr = np.full((self.n_layer, *shape), np.nan, dtype=np.float32)
                for i, buf in enumerate(self._buffers):
                    if buf is not None:
                        arr[i] = buf[stat].numpy()
                out[f'attn_{stat}'] = arr
            return out
        finally:
            if was_training:
                model.train()
