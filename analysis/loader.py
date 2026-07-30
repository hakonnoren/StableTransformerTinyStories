"""
Checkpoint loading, batched scoring, and the 2*D residual-state lens policy.

Canonical loader for the v2 suite. Three things it fixes relative to the v1
loader in behavior_suite.py:

1. RevConfig is built from ALL the training args, mirroring train.py. Omitting
   embed/lift/readout silently builds wide (2*n_embd) embedding tables for a
   `--rev_embed narrow` checkpoint and fails to load.
2. The 2*D state bridge is EXPLICIT and selectable (LensPolicy), not implicit.
   A narrow RevFormer's blocks emit (x, z) at 2*n_embd while ln_f/lm_head sit at
   n_embd, so any depth-resolved readout has to choose: lens x, z, or x+z. v1 got
   this wrong by feeding the raw 1536-d state to a 768-d head. Which stream
   carries the prediction, and whether that shifts with depth, is a question this
   architecture uniquely allows — see plan §5.3.
3. Scoring is BATCHED. v1 ran one prompt per forward, which is why a 5-arm run
   took minutes; N=2000-story probes are not feasible one prompt at a time.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# numpy<2 unpickling shim, kept from v1 (harmless on numpy>=2)
try:
    import numpy.core, numpy.core.multiarray, numpy.core.numeric  # noqa
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ModelConfig, GPTModel, YuriiFormerModel, PresympModelETDAB2  # noqa: E402
from revformer import RevFormerModel, RevConfig  # noqa: E402


# ---------------------------------------------------------------- lens policy
class LensPolicy:
    """How to map a block's output to the width ln_f/lm_head expect.

    SUM  — the model's own readout (x+z for readout='sum'). The faithful choice:
           this is exactly what forward() does, so layer L-1 reproduces the real
           next-token loss. Default.
    X, Z — lens one stream only (narrow reversible models). Not faithful to
           forward(), and deliberately so: comparing the two curves says which
           stream carries the prediction at each depth.
    """
    SUM = "sum"
    X = "x"
    Z = "z"
    ALL = (SUM, X, Z)


def bridge(model, h: torch.Tensor, policy: str = LensPolicy.SUM) -> torch.Tensor:
    """Block output -> head width, under `policy`.

    Baseline/yurii/presymp and wide-embed reversible models already match the head
    width and pass through untouched (their state IS what ln_f consumes).
    """
    narrow = getattr(model, "narrow_emb", False)
    if not narrow:
        if policy != LensPolicy.SUM:
            raise ValueError(
                f"lens policy {policy!r} needs a narrow-embed reversible model; this "
                f"model's blocks already emit head-width states (use 'sum')")
        return h
    D = model.cfg.n_embd
    if policy == LensPolicy.SUM:
        return model._readout(h)                       # the model's own bridge
    x, z = torch.split(h, D, dim=-1)
    return x if policy == LensPolicy.X else z


def supports_policy(model, policy: str) -> bool:
    return policy == LensPolicy.SUM or bool(getattr(model, "narrow_emb", False))


# ---------------------------------------------------------------- loading
def _model_cfg(cfg_dict: dict) -> ModelConfig:
    # attn_impl is absent from checkpoints predating it; ModelConfig's default
    # ("manual") then applies. The kernels are mathematically equivalent, so an old
    # ckpt read back as "manual" is still the trained model — but the difference is
    # recorded in the manifest, because reduction order is not identical.
    fields = {"vocab_size", "block_size", "n_layer", "n_head", "n_embd",
              "dropout", "bias", "attn_impl", "presymp_mlp_use_attn_vel"}
    return ModelConfig(**{k: v for k, v in cfg_dict.items() if k in fields})


def build_model(cfg: ModelConfig, a: dict):
    """Instantiate the architecture described by a checkpoint's saved args.

    Mirrors train.py's construction field for field. Keep it that way: any arg that
    changes parameter SHAPES (rev_embed, rev_linear_map, no_mlp) will otherwise
    fail to load, and any arg that changes only the FUNCTION will load fine and
    silently compute something else.
    """
    g = a.get
    arch = g("arch", "baseline")
    if arch == "baseline":
        return GPTModel(cfg, no_mlp=g("no_mlp", False))
    if arch == "reversible":
        return RevFormerModel(cfg, RevConfig(
            regime=g("rev_regime", "vpm_scaling"), lambd=g("rev_lambda", 0.0),
            epsilon=g("rev_epsilon", 1.0), randn_init=g("rev_randn_init", False),
            tanh_scale=g("rev_tanh", False),
            kappa_min=g("rev_kappa_min", 0.005), kappa_max=g("rev_kappa_max", 0.08),
            kappa_mem=g("rev_kappa_mem", 0.001),
            linear_map=g("rev_linear_map", "diag"), lowrank_r=g("rev_lowrank_r", 4),
            cayley_h=g("rev_cayley_h", 1.0), rotation=g("rev_rotation", "none"),
            n_householder=g("rev_n_householder", 4),
            embed=g("rev_embed", "wide"), lift=g("rev_lift", "dup"),
            readout=g("rev_readout", "sum")))
    if arch == "yurii_lt":
        return YuriiFormerModel(cfg, use_v0_init=not g("no_v0_init", False),
                                noise_eta=g("yurii_noise_eta", 0.0),
                                noise_gamma=g("yurii_noise_gamma", 0.55),
                                noise_loc=g("yurii_noise_loc", "v"),
                                restart_mode=g("yurii_restart", "none"),
                                restart_min_layer=g("yurii_restart_min_layer", 1),
                                no_mlp=g("no_mlp", False))
    if arch == "presymp_etd_ab2":
        return PresympModelETDAB2(
            cfg, h=g("presymp_h", 1.0), t0=g("presymp_t0", 1.0),
            eta_mu=g("eta_mu"), eta_log_coef=g("eta_log_coef"), eta_lin_coef=g("eta_lin_coef"),
            eta_log_init=g("eta_log_init"), eta_lin_init=g("eta_lin_init"),
            eta_learnable=g("eta_learnable", False), eta_mode=g("eta_mode", "log"),
            eta_init=g("eta_init"), eta_clip=g("eta_clip", 50.0),
            presymp_lnp=g("presymp_lnp", "end"), use_v0_init=not g("no_v0_init", False),
            mlp_use_attn_vel=g("presymp_mlp_use_attn_vel", False),
            mlp_use_p_vel=g("presymp_mlp_use_p_vel", False),
            no_mlp=g("no_mlp", False), lookahead=g("presymp_lookahead", False))
    raise ValueError(f"unknown arch {arch}")


def load_model(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    cfg = _model_cfg(ck["cfg"])
    m = build_model(cfg, a)
    m.load_state_dict(ck["model"])
    m.eval().to(device)
    meta = {
        "arch": a.get("arch", "baseline"),
        "regime": a.get("rev_regime") if a.get("arch") == "reversible" else None,
        "embed": a.get("rev_embed") if a.get("arch") == "reversible" else None,
        # absent in pre-sdpa checkpoints -> what ModelConfig actually used
        "attn_impl": ck["cfg"].get("attn_impl", cfg.attn_impl),
        "best_val": ck.get("best_val"),
        "step": ck.get("step"),
        "max_steps": a.get("max_steps"),
        "train_seed": a.get("seed"),
        "n_layer": cfg.n_layer,
        "n_embd": cfg.n_embd,
        "n_params": sum(p.numel() for p in m.parameters()),
        "narrow": bool(getattr(m, "narrow_emb", False)),
        "is_causal": a.get("arch", "baseline") != "presymp_etd_ab2",
    }
    return m, meta


def discover(ckpt_dir: str, prefix: str = "best_") -> dict:
    out = {}
    for root, _, files in os.walk(ckpt_dir):
        for f in files:
            if f.startswith(prefix) and f.endswith(".pt"):
                out[os.path.basename(root)] = os.path.join(root, f)
    return dict(sorted(out.items()))


@dataclass
class Arm:
    """A loaded checkpoint plus its metadata."""
    name: str
    model: object
    meta: dict
    path: str

    @property
    def n_layer(self) -> int:
        return len(self.model.blocks)


def load_arms(ckpt_dir: str, device: str = "cpu", only: list[str] | None = None) -> list[Arm]:
    arms = []
    for name, p in discover(ckpt_dir).items():
        if only and name not in only:
            continue
        m, meta = load_model(p, device)
        arms.append(Arm(name, m, meta, p))
    if not arms:
        raise SystemExit(f"no best_*.pt under {ckpt_dir}")
    return arms


# ---------------------------------------------------------------- scoring
def forward_logits(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def token_logprobs(model, ids: torch.Tensor, mask: torch.Tensor | None = None):
    """Per-token log P(next token) for a batch.

    Returns (logprobs, valid) both (B, T-1): logprobs[b, t] is the model's
    log-probability of the true token ids[b, t+1]. `valid` accounts for padding.
    """
    logits = forward_logits(model, ids).float()
    lp = F.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    got = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    if mask is None:
        valid = torch.ones_like(got, dtype=torch.bool)
    else:
        # a position is scored only if both it and its target are real tokens
        valid = mask[:, :-1] & mask[:, 1:]
    return got, valid


@torch.no_grad()
def layer_states(model, ids: torch.Tensor) -> list[torch.Tensor]:
    """Per-block output states for one batch, via forward hooks.

    Returns the raw block outputs — for a narrow reversible model these are the
    full 2*n_embd state, so pass them through bridge() before any head.
    """
    caps: list[torch.Tensor | None] = [None] * len(model.blocks)
    handles = []

    def mk(i):
        def hook(_mod, _inp, out):
            caps[i] = (out[0] if isinstance(out, (tuple, list)) else out).detach()
        return hook

    for i, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(mk(i)))
    try:
        model(ids)
    finally:
        for h in handles:
            h.remove()
    return [c for c in caps]


def head_logits(model, h: torch.Tensor, policy: str = LensPolicy.SUM) -> torch.Tensor:
    """Early-exit logits from a block state through the model's own head."""
    return model.lm_head(model.ln_f(bridge(model, h, policy)))


@torch.no_grad()
def last_position_logits(model, prompts: list[list[int]], device: str = "cpu",
                         batch_size: int = 32, pad_id: int = 50256):
    """Logits at the final real token of each prompt, batched.

    Right-padding is safe here for the same reason as in corpus.pad_batch: a causal
    model's position i cannot see i+1, so pad tokens after the prompt do not affect
    the prompt's own last position. We gather at len-1 per row.
    """
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i + batch_size]
        T = max(len(p) for p in chunk)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        idx = torch.zeros(len(chunk), dtype=torch.long)
        for j, p in enumerate(chunk):
            ids[j, :len(p)] = torch.tensor(p, dtype=torch.long)
            idx[j] = len(p) - 1
        logits = forward_logits(model, ids.to(device)).float().cpu()
        out.append(logits[torch.arange(len(chunk)), idx])
    return torch.cat(out, dim=0)
