r"""Couplings of the multi-hop composition model of Kohli et al. (2604.07822, "Loop, Think & Generalize").

Their model (`RecurrentGPT2Block`, OSU-NLP-Group/Loop-Think-Generalize, gpt_utils_extrapolation.py) iterates one stack
of 4 GPT-2 layers, f, on the hidden states: h_0 = token embedding (NoPE, no input injection), h_{r+1} = f(h_r), and
reads the answer at the last position through ln_f and the tied LM head. f is residual: f(u) = u + D(u). This module
wraps a loaded `RecurrentGPT2Block` (its weights are the parameters) and runs one of these recurrences, all counted in
evaluations of f (one "iteration" = one f, so depths are compute-matched across couplings):

  vanilla     h <- f(h)                                                    (the released model, unchanged)
  relax       two streams L, H, updated alternately (L at even t, H at odd t), each from the other one:
                  L <- a L + (1-a) (f(H) - (1-c) H)        H <- b H + (1-b) (f(L) - (1-c) L)
              a = e^-gamma, b = e^-alpha, c = e^-delta. f(H) - (1-c) H = c H + D(H): delta = 0 (c = 1) is the
              conformal relax form of the October note (L <- a L + (1-a)(H + D(H))); delta > 0 damps the common
              mode (the Sudoku runs' relax_delta). Read-out: the stream updated last.
  twin        the same two-stream recurrence, read out through the mean (L + H)/2: the copy difference L - H is
              invisible to the read-out (the single-level twin of looped/dynlab; Sudoku's twin_l adds an H level
              that has no counterpart in this single-level model).

Ports: both streams start at the embedding. With gamma large (a ~ 0) and delta = 0, relax is exactly the released
model at every depth (L_1 = f(e), H_1 = f(L_1), ...: the stream updated last is f^r(e)); see test_multihop.py.

Port warm-up: `mix` in [0, 1] blends from the released-equivalent recurrence (mix = 0: c = 1, and twin reads the stream
updated last) to the target coupling (mix = 1: c = e^-delta, twin reads the mean); train.py ramps it over the first
--port-warmup steps, so that a damped coupling starts from the released model's function instead of a broken one.

Test-time noise (EqR's Langevin search, relative here because the residual stream's scale grows with depth): after
each update the updated stream gets sigma * rms(stream) * N(0, 1), rms per token. Training noise likewise (sigma_train).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

COUPLINGS = ("vanilla", "relax", "twin")


class Coupled(nn.Module):
    def __init__(self, base, coupling="vanilla", gamma=3.0, alpha=None, delta=0.0):
        super().__init__()
        assert coupling in COUPLINGS, coupling
        self.base = base
        self.coupling = coupling
        self.gamma, self.alpha, self.delta = float(gamma), float(gamma if alpha is None else alpha), float(delta)
        self.mix = 1.0

    def describe(self):
        return {"coupling": self.coupling, "gamma": self.gamma, "alpha": self.alpha, "delta": self.delta}

    # ---- pieces of the base model ------------------------------------------------------------------------------
    def embed(self, ids):
        return self.base.dropout(self.base.token_embedding(ids))         # NoPE, as the released models

    @staticmethod
    def ext_mask(mask, dtype):
        return (1.0 - mask[:, None, None, :].to(dtype)) * -10000.0      # as RecurrentGPT2Block.forward

    def f(self, u, ext):
        for block in self.base.blocks:
            u = block(u, attention_mask=ext)[0]
        return u

    def logits_of(self, h, pos=-1):
        return self.base.lm_head(self.base.ln_f(h[:, pos]))

    # ---- the recurrence ----------------------------------------------------------------------------------------
    def init_state(self, e):
        return (e,) if self.coupling == "vanilla" else (e, e)

    def step(self, s, t, ext, sigma=0.0, gen=None):
        """One evaluation of f. t: the index of this update (0, 1, ...), which picks the stream in relax/twin."""
        if self.coupling == "vanilla":
            return (_noisy(self.f(s[0], ext), sigma, gen),)
        L, H = s
        c = 1.0 - self.mix * (1.0 - math.exp(-self.delta))
        if t % 2 == 0:
            a = math.exp(-self.gamma)
            L = _noisy(a * L + (1 - a) * (self.f(H, ext) - (1 - c) * H), sigma, gen)
        else:
            b = math.exp(-self.alpha)
            H = _noisy(b * H + (1 - b) * (self.f(L, ext) - (1 - c) * L), sigma, gen)
        return (L, H)

    def read(self, s, t):
        """The read-out state after update t."""
        if self.coupling == "vanilla":
            return s[0]
        last = s[0] if t % 2 == 0 else s[1]
        if self.coupling == "twin":
            return (1.0 - self.mix) * last + self.mix * 0.5 * (s[0] + s[1])
        return last

    def forward(self, ids, mask, n_iter, n_carry=0, sigma=0.0, gen=None):
        """Logits at the last position after n_carry iterations without gradient and n_iter with (training)."""
        e = self.embed(ids)
        ext = self.ext_mask(mask, e.dtype)
        s = self.init_state(e)
        t = 0
        if n_carry:
            with torch.no_grad():
                for _ in range(n_carry):
                    s = self.step(s, t, ext, sigma, gen)
                    t += 1
            s = tuple(x.detach() for x in s)
        for _ in range(n_iter):
            s = self.step(s, t, ext, sigma, gen)
            t += 1
        return self.logits_of(self.read(s, t - 1))

    @torch.no_grad()
    def trajectory(self, ids, mask, depths, sigma=0.0, gen=None, want=("pred",)):
        """Iterate to max(depths); per recorded depth the prediction at the last position, plus (optional) the
        relative change of the read-out state there and the logit margin. Returns dict of [len(depths), b] tensors."""
        depths = sorted(set(int(d) for d in depths))
        e = self.embed(ids)
        ext = self.ext_mask(mask, e.dtype)
        s = self.init_state(e)
        out = {k: [] for k in want}
        prev = None
        resid = None
        for t in range(depths[-1]):
            s = self.step(s, t, ext, sigma, gen)
            r = self.read(s, t)[:, -1].float()
            if prev is not None:
                resid = (r - prev).norm(dim=-1) / r.norm(dim=-1).clamp_min(1e-30)
            prev = r
            if t + 1 in depths:
                lg = self.logits_of(self.read(s, t)).float()
                if "pred" in out:
                    out["pred"].append(lg.argmax(-1))
                if "resid" in out:
                    out["resid"].append(resid if resid is not None else torch.full_like(lg[:, 0], float("nan")))
                if "logp" in out:
                    out["logp"].append(F.log_softmax(lg, -1))
        return {k: torch.stack(v) for k, v in out.items()}


def _noisy(x, sigma, gen):
    if not sigma:
        return x
    rms = x.float().pow(2).mean(-1, keepdim=True).sqrt().to(x.dtype)
    eps = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype) if gen is not None else torch.randn_like(x)
    return x + sigma * rms * eps


def load_base(ltg_dir, vocab_size, ckpt=None, d_model=768, n_layer=4, n_head=12, max_len=128, device="cpu"):
    """The released architecture (upstream class, imported from the upstream checkout), optionally with weights."""
    import sys
    if ltg_dir not in sys.path:
        sys.path.insert(0, ltg_dir)
    from transformers import GPT2Config
    from gpt_utils_extrapolation import RecurrentGPT2Block
    config = GPT2Config(vocab_size=vocab_size, n_positions=max_len, n_ctx=max_len, n_embd=d_model, n_layer=n_layer,
                        n_head=n_head, embd_pdrop=0.0, _attn_implementation="eager")
    base = RecurrentGPT2Block(config, 8, positional_embedding_type="none", input_injection=False, c_scale=0.0)
    if ckpt is not None:
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model_state_dict", sd.get("model", sd))
        sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
        sd = {k[len("base."):] if k.startswith("base.") else k: v for k, v in sd.items()}
        base.load_state_dict(sd, strict=True)
    return base.to(device)
