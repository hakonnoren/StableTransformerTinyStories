# Muon — MomentUm Orthogonalized by Newton-schulz (single-GPU, vendored fallback).
#
# This is a drop-in stand-in for torch.optim.Muon (added in torch 2.12) for the
# older torch we run on the cluster/locally (2.1.x). train.py prefers the built-in
# when available and falls back to this; the constructor mirrors the built-in's
# signature (lr, weight_decay, momentum, nesterov, ns_coefficients, eps, ns_steps,
# adjust_lr_fn) so the two are interchangeable.
#
# Reference: Keller Jordan's Muon (https://github.com/KellerJordan/Muon, MIT).
# Use ONLY for 2D hidden weight matrices (attention q/k/v/proj, MLP fc/proj).
# Embeddings, the LM head/unembedding, norms and all 1D / scalar params must go
# to AdamW — orthogonalizing a vocab x d matrix is the classic Muon failure mode.
import torch
from torch import Tensor

# adjust_lr_fn modes (match the built-in's two options):
#   "original"        -> lr * sqrt(max(1, fan_out/fan_in))  (Keller default; pairs with lr~0.02)
#   "match_rms_adamw" -> lr * 0.2 * sqrt(max(fan_out, fan_in))  (Moonlight; pairs with AdamW-scale lr)
_ADJUST_LR_FNS = ("original", "match_rms_adamw")


def _adjusted_lr(lr: float, rows: int, cols: int, mode: str) -> float:
    if mode == "match_rms_adamw":
        return lr * 0.2 * (max(rows, cols) ** 0.5)
    return lr * (max(1.0, rows / cols) ** 0.5)   # "original"


@torch.no_grad()
def zeropower_via_newtonschulz5(G: Tensor, steps: int,
                                coeffs=(3.4445, -4.7750, 2.0315), eps: float = 1e-7) -> Tensor:
    """Orthogonalize G (compute its zeroth matrix power U V^T) via a quintic
    Newton-Schulz iteration. Runs in bf16; coefficients tuned by Keller Jordan so
    the iteration pushes the singular values toward 1 from a normalized start."""
    assert G.ndim >= 2
    a, b, c = coeffs
    X = G.bfloat16()
    transposed = G.size(-2) > G.size(-1)
    if transposed:                       # iterate on the thinner orientation
        X = X.mT
    # scale so the largest singular value is <= 1 (Frobenius norm is an upper bound)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon optimizer for 2D hidden-weight matrices (vendored fallback).

    Per step, per param: accumulate a (Nesterov) momentum buffer, orthogonalize
    the update with Newton-Schulz, scale the lr per ``adjust_lr_fn``, apply
    decoupled weight decay, then step.

    Standard torch.optim.Optimizer subclass so the training loop can mutate
    ``param_groups[i]["lr"]`` for the cosine schedule exactly as it does for AdamW.
    """

    def __init__(self, params, lr: float = 0.02, weight_decay: float = 0.1,
                 momentum: float = 0.95, nesterov: bool = True,
                 ns_coefficients=(3.4445, -4.7750, 2.0315), eps: float = 1e-7,
                 ns_steps: int = 5, adjust_lr_fn=None):
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if adjust_lr_fn is None:
            adjust_lr_fn = "original"
        if adjust_lr_fn not in _ADJUST_LR_FNS:
            raise ValueError(f"adjust_lr_fn must be one of {_ADJUST_LR_FNS}, got {adjust_lr_fn!r}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, ns_coefficients=ns_coefficients,
                        eps=eps, ns_steps=ns_steps, adjust_lr_fn=adjust_lr_fn)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]
            coeffs = group["ns_coefficients"]
            eps = group["eps"]
            mode = group["adjust_lr_fn"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                # flatten any conv-like >2D weight to a matrix for orthogonalization
                g_mat = g if g.ndim == 2 else g.reshape(g.size(0), -1)

                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g_mat)
                buf.mul_(momentum).add_(g_mat)
                update = g_mat.add(buf, alpha=momentum) if nesterov else buf

                update = zeropower_via_newtonschulz5(update, ns_steps, coeffs, eps).to(g.dtype)
                alr = _adjusted_lr(lr, update.size(-2), update.size(-1), mode)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)          # decoupled weight decay (uses base lr)
                p.add_(update.reshape_as(p), alpha=-alr)

        return loss
