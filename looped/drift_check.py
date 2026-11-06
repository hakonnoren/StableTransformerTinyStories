"""Checks prompted by the draft review addendum (frozen checkpoints, no training).

A. Long horizon (seed 1238, 256 test sequences, r up to 2048):
   norm_q   mean ||P s_r||                      (P removes each stream's per-token channel mean)
   step_q   mean ||P(s_r - s_{r-1})||
   acc      answer accuracy
   relax with delta = 0 only (Lemma 2: C_{r+1} - C_r = kappa * v_r exactly):
   vbar     mean ||P(C_r - C_0)|| / (r kappa)    = norm of the running average force (1/r) sum_k v_k
   cos      mean cosine between the current force v_r and that running average
   Drift in the sense of the conditional statement means vbar -> nonzero constant, cos -> 1.
   For kick with gamma = alpha = 0.005 (linear part bounded, e-folding time 200 iterations): does norm_q saturate?

B. Prediction persistence (all seeds, full test set): with predictions at r_hi and at 128 on the answer positions,
   lose  P(correct at r_hi, wrong at 128)
   gain  P(wrong at r_hi, correct at 128)        (acc_128 - acc_hi = gain - lose)
   changed P(prediction at 128 != prediction at r_hi)
   over  P(wrong at 128 | correct at r_hi)

    python looped/drift_check.py     # -> results/looped/<task>/drift_check.json, printed summary
    python looped/drift_check.py --tasks khop_K3 --persist-only   # after new seeds
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "looped"))
import dynamics                                                          # noqa: E402

TASKS = ["listops", "khop_K3"]
LONG = ["vanilla", "relax3", "kick_fixed", "kick_pos", "relax3_leak4", "kick_pos_carry", "relax3_leak4_carry"]
R_LONG = [8, 32, 128, 256, 512, 1024, 2048]
PERSIST = ["vanilla", "relax3", "kick_fixed", "kick_learned", "kick_pos", "relax3_leak16", "relax3_leak4",
           "kick_pos_carry", "relax3_leak4_carry"]


def flat_norm(t):
    return t.flatten(1).norm(dim=1)


@torch.no_grad()
def long_horizon(model, x, y):
    x = torch.as_tensor(x[:256], dtype=torch.long); y = torch.as_tensor(y[:256], dtype=torch.long)
    mask = y != -1
    s, e = model.embed(x)
    layer = model.core[0]
    relax0 = layer.kind == "relax" and not layer.has_leak
    if relax0:
        a, b = torch.exp(-layer.gamma()), torch.exp(-layer.alpha())
        assert a.numel() == 1 and b.numel() == 1, "scalar rates expected"
        kappa = float((1 - a) * (1 - b) / (1 - a * b))
        cen = lambda t: t - t.mean(-1, keepdim=True)
        C = lambda st: cen((a * (1 - b) * st[..., :model.d] + (1 - a) * st[..., model.d:]) / (1 - a * b))
        C0 = C(s)
    out = {}
    for r in range(1, max(R_LONG) + 1):
        s_new = model.step(s, e)
        if r in R_LONG:
            pred = model.readout(s_new).argmax(-1)
            rec = dict(acc=float((pred == y)[mask].float().mean()),
                       norm_q=float(flat_norm(model.center(s_new)).mean()),
                       step_q=float(flat_norm(model.center(s_new) - model.center(s)).mean()))
            if relax0:
                Cr, Cprev = C(s_new), C(s)
                avg = (Cr - C0) / (r * kappa)
                v = (Cr - Cprev) / kappa
                rec["vbar"] = float(flat_norm(avg).mean())
                rec["v"] = float(flat_norm(v).mean())
                rec["cos"] = float(torch.nn.functional.cosine_similarity(v.flatten(1), avg.flatten(1), dim=1).mean())
            out[r] = rec
        s = s_new
    return out


@torch.no_grad()
def persistence(model, x, y, r_hi, bs=500):
    lose = gain = changed = correct_hi = n = 0
    for i in range(0, len(x), bs):
        xb = torch.as_tensor(x[i:i + bs], dtype=torch.long); yb = torch.as_tensor(y[i:i + bs], dtype=torch.long)
        mask = yb != -1
        s, e = model.embed(xb)
        p_hi = None
        for r in range(1, 129):
            s = model.step(s, e)
            if r == r_hi:
                p_hi = model.readout(s).argmax(-1)
        p_128 = model.readout(s).argmax(-1)
        ok_hi, ok_128 = (p_hi == yb)[mask], (p_128 == yb)[mask]
        lose += int((ok_hi & ~ok_128).sum()); gain += int((~ok_hi & ok_128).sum())
        changed += int((p_hi != p_128)[mask].sum()); correct_hi += int(ok_hi.sum()); n += int(mask.sum())
    return dict(lose=lose / n, gain=gain / n, changed=changed / n, over=lose / max(correct_hi, 1))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--persist-only", action="store_true", help="keep the stored long-horizon part")
    cli = ap.parse_args()
    torch.set_num_threads(2)
    for task in cli.tasks:
        d = ROOT / "results/looped" / task
        old = d / "drift_check.json"
        res = {"long": json.loads(old.read_text())["long"] if (cli.persist_only and old.exists()) else {}, "persist": {}}
        for name in ([] if cli.persist_only else LONG):
            pt = d / f"{name}_seed1238.pt"
            if not pt.exists():
                continue
            model, args = dynamics.load(pt)
            x, y, _ = dynamics.task_data(args)["tests"]["test"]
            res["long"][name] = lh = long_horizon(model, x, y)
            line = "  ".join(f"r={r}: |Ps| {v['norm_q']:.3g} step {v['step_q']:.2g} acc {v['acc']:.3f}"
                             + (f" vbar {v['vbar']:.3g} cos {v['cos']:.2f}" if "vbar" in v else "")
                             for r, v in lh.items() if r in (128, 512, 2048))
            print(f"[long] {task}/{name}: {line}", flush=True)
        for name in PERSIST:
            for pt in sorted(d.glob(f"{name}_seed*.pt")):
                if pt.stem.rsplit("_seed", 1)[0] != name:
                    continue
                model, args = dynamics.load(pt)
                x, y, _ = dynamics.task_data(args)["tests"]["test"]
                p = persistence(model, x, y, args["r_hi"])
                res["persist"].setdefault(name, {})[pt.stem.rsplit("_seed", 1)[1]] = p
                print(f"[persist] {task}/{pt.stem}: lose {p['lose']:.3f} gain {p['gain']:.3f} "
                      f"changed {p['changed']:.3f} over {p['over']:.3f}", flush=True)
        (d / "drift_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
