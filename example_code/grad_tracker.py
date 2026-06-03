import torch
from collections import defaultdict



import torch
from torch.func import vjp, vmap

def jacobian_metrics(f, x,
                     metrics=("logdet",),
                     power_iter_steps=10):
    """
    Evaluate y = f(x)   and, optionally, metrics of J = dy/dx.

    Args
    ----
    f        : (B,d,n) -> (B,d,n)   -- must be deterministic
    x        : (B,d,n)              -- no grad required
    metrics  : iterable[str]        -- any of {"logdet","fro","spectral"}
    power_iter_steps : int          -- for spectral norm power-iteration

    Returns
    -------
    y        : (B,d,n)
    stats    : {name: tensor(B,)}   -- one scalar per sample
    """
    B, d, n = x.shape
    DN = d * n

    # ---- single-sample pullback ---------------------------------------------
    def _pullback(xi):
        yi, vjp_fn = vjp(lambda z: f(z.unsqueeze(0)).squeeze(0), xi)
        I = torch.eye(DN, dtype=xi.dtype, device=xi.device).reshape(DN, d, n)
        J = vmap(vjp_fn)(I)[0].reshape(DN, DN)           #   (DN,DN)

        out = {}
        if "logdet" in metrics:
            out["logdet"] = torch.linalg.slogdet(J).logabsdet.detach()
        if "fro" in metrics:
            out["fro"] = torch.linalg.norm(J, ord='fro').detach()
        if "spectral" in metrics:
            if False:
            #v = torch.randn(DN, 1, device=xi.device)
                v = torch.ones(J.shape[1], device=J.device)
                for _ in range(power_iter_steps):
                    v = J @ v
                    v /= v.norm()
                out["spectral"] = (v.t() @ (J @ v)).sqrt().squeeze().detach()
            out["spectral"] = torch.linalg.norm(J, ord=2).detach()
        return yi, out

    ys, list_of_dicts = vmap(_pullback)(x)            # batched & fused
    #stats = {k: torch.stack([d[k] for d in list_of_dicts])  # (B,)
    #         for k in list_of_dicts[0]}
    return ys, list_of_dicts



class GradTracker:
    """
    Tracks per-parameter gradient statistics and (optionally) logs them to TensorBoard.
    """

    def __init__(self, model: torch.nn.Module):
        self.stats = defaultdict(list)      # {param_name: [{norm:…, max:…, …}, …]}
        self.step   = 0
        self.hooks  = []

        for name, p in model.named_parameters():
            if not p.requires_grad:                       # frozen layers are ignored
                continue
            # Register a *post-grad* hook
            self.hooks.append(
                p.register_hook(self._make_hook(name))
            )

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _make_hook(self, name):
        def hook(grad: torch.Tensor):
            stat = {
                "norm": grad.norm().item(),
                "max" : grad.abs().max().item(),
                "mean": grad.mean().item(),
                "std" : grad.std().item()
            }
            self.stats[name].append(stat)

            # Live alert (optional)
            if stat["norm"] > 1e3 or torch.isnan(grad).any():
                print(f"[Exploding?] step {self.step:>6} | {name:<40} norm={stat['norm']:.2e}")
            if stat["norm"] < 1e-8:
                print(f"[Vanishing?] step {self.step:>6} | {name:<40} norm={stat['norm']:.2e}")


        return hook

    def next_step(self):
        self.step += 1

    def close(self):
        for h in self.hooks:
            h.remove()
        if self.writer:
            self.writer.flush()
            self.writer.close()

import torch
from collections import defaultdict
from typing import Dict, List

class GradTrackerSimple:
    """
    Collects gradient statistics in memory.

    After training you will have:
        tracker.history      # param_name -> metric -> list[step_values]
        tracker.global_norms # list[float] (one per optimisation step)
    """

    MetricDict = Dict[str, List[float]]    # {"norm": [...], "max": [...], ...}

    def __init__(self, model: torch.nn.Module):
        self.history: Dict[str, GradTrackerSimple.MetricDict] = defaultdict(
            lambda: defaultdict(list)
        )
        self.global_norms: List[float] = []
        self._param_refs = [p for p in model.parameters() if p.requires_grad]

        # Register one hook per parameter
        self._hooks = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self._hooks.append(p.register_hook(self._make_hook(name)))

        self._step_has_run = False   # helps us compute a *single* global-norm per step

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _make_hook(self, name: str):
        def hook(grad: torch.Tensor):
            h = self.history[name]
            h["norm" ].append(grad.norm().item())
            h["max"  ].append(grad.abs().max().item())
            h["mean" ].append(grad.mean().item())
            h["std"  ].append(grad.std().item())

            # Warn about pathological values
            if h["norm"][-1] > 1e3 or torch.isnan(grad).any():
                print(f"[Exploding?] {name:<40} | norm={h['norm'][-1]:.2e}")
            if h["norm"][-1] < 1e-8:
                print(f"[Vanishing?] {name:<40} | norm={h['norm'][-1]:.2e}")

            # Mark that at least one hook fired this optimisation step
            self._step_has_run = True
        return hook

    def next_step(self):
        """
        Call **once after every `optimizer.step()`**.
        Computes and stores the *global* grad-norm for that step.
        """
        if not self._step_has_run:        # backward() wasn’t called?
            return

        total = torch.zeros((), device=self._param_refs[0].device)
        for p in self._param_refs:
            if p.grad is not None:
                total = total + p.grad.pow(2).sum()
        self.global_norms.append(torch.sqrt(total).item())
        self._step_has_run = False        # reset for next round

    # ------------------------------------------------------------------ #
    # Convenience extras
    # ------------------------------------------------------------------ #
    def as_dataframe(self):
        """Return a tidy pandas DataFrame (requires pandas)."""
        import pandas as pd
        records = []
        for name, metrics in self.history.items():
            for metric, series in metrics.items():
                for step, value in enumerate(series):
                    records.append(dict(step=step, param=name,
                                        metric=metric, value=value))
        return pd.DataFrame.from_records(records)

    def close(self):
        for h in self._hooks:
            h.remove()


import torch, math, itertools
from collections import defaultdict
from typing import Dict, List, Iterable, Tuple

class GradTrackerSpectral:
    """
    Tracks
      • per-parameter gradient stats          (same as before)
      • per-layer spectral & Frobenius norms  (1 power-it + 1 Hutchinson draw / step)
    """

    def __init__(self,
                 model: torch.nn.Module,
                 layers_to_track,
                 power_iters_per_step: int = 1,
                 eps: float = 1e-12):
        self.eps   = eps
        self.k     = power_iters_per_step

        # -----------------------------------------------------------------
        # 1) Per-parameter gradient statistics
        # -----------------------------------------------------------------
        self.grad_hist   = defaultdict(lambda: defaultdict(list))
        self._param_refs = [p for p in model.parameters() if p.requires_grad]
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        for name, p in model.named_parameters():
            if p.requires_grad:
                self._hooks.append(p.register_hook(self._make_grad_hook(name)))

        # -----------------------------------------------------------------
        # 2) Per-layer spectral / Frobenius norms
        # -----------------------------------------------------------------
        if layers_to_track is None:
            layers_to_track = [
                m for m in model.modules()
                if any(p in self._param_refs for p in m.parameters(recurse=False))
            ]

        self.spectral_hist   = defaultdict(list)   # layer -> σ_max over time
        self.frobenius_hist  = defaultdict(list)   # layer -> ∥J∥_F over time
        self._power_state    = {}                  # layer -> current v  (output-space)
        self._input_cache    = {}                  # layer -> latest input Tensor

        # forward hooks just to capture a *representative* input per step
        for layer in layers_to_track:
            lname = self._qualname(model, layer)
            self._hooks.append(layer.register_forward_hook(
                self._make_fwd_hook(lname)))

        self._step_has_run = False
        self.global_norms  = []                    # total grad ℓ2 per step

    # -------------------- gradient-stat hook --------------------------- #
    def _make_grad_hook(self, name: str):
        def hook(grad: torch.Tensor):
            h = self.grad_hist[name]
            h["norm"]. append(grad.norm().item())
            h["max"].  append(grad.abs().max().item())
            h["mean"]. append(grad.mean().item())
            h["std"].  append(grad.std().item())

            if h["norm"][-1] > 1e3 or torch.isnan(grad).any():
                print(f"[Exploding?] {name:<40} | norm={h['norm'][-1]:.2e}")
            if h["norm"][-1] < 1e-8:
                print(f"[Vanishing?] {name:<40} | norm={h['norm'][-1]:.2e}")

            self._step_has_run = True
        return hook

    # -------------------- forward hook (cache x) ----------------------- #
    def _make_fwd_hook(self, layer_name: str):
        def hook(_module, args, _output):
            # Assumes first positional arg is the differentiable input tensor
            x = args[0]
            if isinstance(x, torch.Tensor):
                # store *detached* copy on same device; we'll re-enable grad later
                self._input_cache[layer_name] = x.detach()
        return hook

    @staticmethod
    def _qualname(root: torch.nn.Module, sub: torch.nn.Module) -> str:
        for name, m in root.named_modules():
            if m is sub:
                return name or root.__class__.__name__
        return f"<unnamed_{id(sub)}>"

    # -------------------- per-layer norm update ------------------------ #
    @torch.no_grad()
    def _update_layer_norms(self, layer_name: str, layer: torch.nn.Module):
        if layer_name not in self._input_cache:
            return                                       # no forward pass yet

        x0 = self._input_cache[layer_name].detach().clone().requires_grad_(True)

        # (y0, vjp)   gives   Jᵀ · (·)
        y0,  vjp_fn = torch.autograd.functional.vjp(layer, x0)

        # --- initialise / fetch power-iteration vector in output space
        v = self._power_state.get(layer_name)
        if v is None or v.shape != y0.shape:
            v = torch.randn_like(y0)
        v = v / (v.norm() + self.eps)

        # --- run k power iterations (default 1) ------------------------
        for _ in range(self.k):
            (w,) = vjp_fn(v)                                   # w = Jᵀ v
            jw, _ = torch.autograd.functional.jvp(layer, (x0,), (w,))
            v = jw / (jw.norm() + self.eps)
        self._power_state[layer_name] = v.detach()             # warm start

        # Rayleigh quotient  →  spectral norm estimate
        (w,) = vjp_fn(v)
        sigma = w.norm() / (v.norm() + self.eps)
        self.spectral_hist[layer_name].append(sigma.item())

        # Hutchinson estimator for ∥J∥_F
        r = torch.empty_like(x0).bernoulli_(0.5).mul_(2).sub_(1)   # ±1
        jr, _ = torch.autograd.functional.jvp(layer, (x0,), (r,))
        frob = jr.norm().pow(2).item() ** 0.5                     # √E[...] ~ ∥J∥_F
        self.frobenius_hist[layer_name].append(frob)

    # -------------------- call once per optim step -------------------- #
    def next_step(self):
        if not self._step_has_run:       # backward() did not happen
            return

        # global grad norm
        total = torch.zeros((), device=self._param_refs[0].device)
        for p in self._param_refs:
            if p.grad is not None:
                total += p.grad.pow(2).sum()
        self.global_norms.append(total.sqrt().item())

        # update spectral / Frobenius norms for each cached layer
        for lname, layer in list(self._input_cache.items()):
            pass  # placeholder (see below)

    # Iterate over actual layer objects (need names ↔ modules map)
    def _layer_objects(self):
        seen = set(self._input_cache.keys())
        for h in self._hooks:
            if hasattr(h, "_hooked_module"):
                m = h._hooked_module
                name = next((n for n, mod in m.named_modules() if mod is m), None)
                if name in seen:
                    yield name, m

    def next_step(self):
        if not self._step_has_run:
            return

        total = torch.zeros((), device=self._param_refs[0].device)
        for p in self._param_refs:
            if p.grad is not None:
                total += p.grad.pow(2).sum()
        self.global_norms.append(total.sqrt().item())

        # compute norms layer-wise
        for lname, layer in self._layer_objects():
            self._update_layer_norms(lname, layer)

        self._step_has_run = False          # reset for next optimisation step

    # -------------------- clean-up ------------------------------------ #
    def close(self):
        for h in self._hooks:
            h.remove()

    # -------------------- convenience: tidy DataFrame ----------------- #
    def as_dataframe(self):
        import pandas as pd
        recs = []
        for pname, metr in self.grad_hist.items():
            for m, series in metr.items():
                for s, v in enumerate(series):
                    recs.append(dict(step=s, kind="param_grad",
                                     name=pname, metric=m, value=v))
        for lname, series in self.spectral_hist.items():
            for s, v in enumerate(series):
                recs.append(dict(step=s, kind="jacobian",
                                 name=lname, metric="spectral", value=v))
        for lname, series in self.frobenius_hist.items():
            for s, v in enumerate(series):
                recs.append(dict(step=s, kind="jacobian",
                                 name=lname, metric="frobenius", value=v))
        return pd.DataFrame.from_records(recs)
