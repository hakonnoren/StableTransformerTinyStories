r"""Test-time scaling on multi-hop composition, 2 to 100 hops: depth, adaptive halting, noise x breadth, overthinking.

    python -m looped.multihop.evaluate --ltg-dir $LTG --data-dir $LTG/data/multi_hop --ckpt runs/multihop/X/final.pt
    python -m looped.multihop.evaluate ... --ckpt $LTG/checkpoints/multi_hop/final_models/r_dyn/checkpoint_epoch_5388.pt --released

Sections (all on the test split of data.py: 500 held-out facts per hop level; the curriculum never saw them):
  depth     noise-free exact accuracy per hop level k and depth (iterations = evaluations of the block stack), and the
            median relative change of the read-out state at the answer position (how settled it is)
  overthink of the answers right at depth a, the share wrong at depth b: (8, 128), (16, 128), (8, 256), (16, 256),
            pooled over k <= 20 (the October note's column) and over all k
  halting   the authors' adaptive halting (stop at the first t >= 2 with KL(p_t || p_t-1) < --eps-kl and entropy of p_t
            < --entropy, at most --halt-max iterations): accuracy and mean iterations per k
  noise     per test noise sigma (relative, model.py) and --breadth independent noisy runs, at --breadth-depths:
            single (mean over runs), vote (majority answer), conv (the run whose read-out state changed least in its
            last update; EqR's most-converged selection), any (some run right), on --breadth-hops
  seen      accuracy on the training-set samples in test.json ("{k}hop_train", k <= 40), for the train-test gap

Writes --out (default: next to the checkpoint, eval_<name>.json) and prints a summary.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from looped.multihop.data import MultiHop, collate
from looped.multihop.model import Coupled, load_base


def parse_hops(s, avail):
    out = set()
    for part in s.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out |= set(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return sorted(out & set(avail))


def run(model, rows, depths, device, bs, **kw):
    """Concatenated trajectory outputs over batches of rows (rows of one hop level have one length)."""
    outs = {}
    for i in range(0, len(rows), bs):
        ids, mask, y = collate(rows[i:i + bs], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tr = model.trajectory(ids, mask, depths, **kw)
        tr["y"] = y
        for k, v in tr.items():
            outs.setdefault(k, []).append(v)
    return {k: torch.cat(v, dim=-1 if k == "y" else 1) if k != "logp" else torch.cat(v, dim=1) for k, v in outs.items()}


def load_model(a, n_vocab, device):
    if a.released:
        base = load_base(a.ltg_dir, n_vocab, a.ckpt, max_len=128, device=device)    # 768 / 4 layers / 12 heads
        return Coupled(base, "vanilla").eval(), {"coupling": "vanilla", "released": a.ckpt}
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    base = load_base(a.ltg_dir, n_vocab, None, max_len=128, device=device, **ck.get("arch", {}))
    base.load_state_dict(ck["model"])
    cp = dict(ck["coupling"])
    model = Coupled(base, cp.pop("coupling"), **cp).to(device).eval()
    return model, {**ck["coupling"], "step": ck.get("step"), "hop": ck.get("hop")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ltg-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--released", action="store_true", help="--ckpt is one of the authors' checkpoints (vanilla)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--hops", default="2-100")
    ap.add_argument("--depths", default="1,2,4,6,8,12,16,24,32,48,64,96,128,192,256")
    ap.add_argument("--halt-max", type=int, default=64)
    ap.add_argument("--eps-kl", type=float, default=0.01)
    ap.add_argument("--entropy", type=float, default=3.0)
    ap.add_argument("--sigmas", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--breadth", type=int, default=16)
    ap.add_argument("--breadth-depths", default="16,64,256")
    ap.add_argument("--breadth-hops", default="10,15,20,25,30,35,40,45,50,60,70,80,100")
    ap.add_argument("--breadth-n", type=int, default=200, help="test facts per hop level in the noise section")
    ap.add_argument("--seen-depths", default="8,16,32")
    ap.add_argument("--skip", default="", help="comma list of sections to skip: depth,halting,noise,seen")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    skip = set(a.skip.split(","))
    data = MultiHop(a.data_dir, need_train=False)
    model, meta = load_model(a, len(data.vocab), device)
    hops = parse_hops(a.hops, data.test)
    depths = sorted({int(d) for d in a.depths.split(",")})
    res = {"ckpt": a.ckpt, "model": meta, "hops": hops, "depths": depths}
    t0 = time.time()

    if "depth" not in skip:
        acc, resid, pred = {}, {}, {}
        for k in hops:
            tr = run(model, data.test[k], depths, device, a.batch, want=("pred", "resid"))
            ok = (tr["pred"] == tr["y"]).float().cpu().numpy()                     # [D, n]
            acc[k] = ok.mean(1).round(4).tolist()
            resid[k] = np.nanmedian(tr["resid"].cpu().numpy(), axis=1).round(5).tolist()
            pred[k] = ok
        res["depth"] = {"acc": acc, "resid_median": resid}
        ov = {}
        for (da, db) in ((8, 128), (16, 128), (8, 256), (16, 256)):
            if da not in depths or db not in depths:
                continue
            ia, ib = depths.index(da), depths.index(db)
            for name, ks in (("k<=20", [k for k in hops if k <= 20]), ("all", hops)):
                ra = np.concatenate([pred[k][ia] for k in ks]); rb = np.concatenate([pred[k][ib] for k in ks])
                n = ra.sum()
                ov[f"{da}->{db} {name}"] = {"lost": float(((ra == 1) & (rb == 0)).sum() / max(n, 1)), "n_right": int(n)}
        res["overthink"] = ov
        print(f"[eval] depth done ({time.time() - t0:.0f} s)", flush=True)

    if "halting" not in skip:
        hd = list(range(1, a.halt_max + 1))
        halt = {}
        for k in hops:
            tr = run(model, data.test[k], hd, device, a.batch, want=("logp",))
            lp = tr["logp"]                                                          # [T, n, V]
            p = lp.exp()
            ent = -(p * lp).sum(-1)                                                  # [T, n]
            kl = torch.full_like(ent, float("inf"))
            kl[1:] = (p[1:] * (lp[1:] - lp[:-1])).sum(-1)
            stop = (kl < a.eps_kl) & (ent < a.entropy)
            stop[-1] = True
            t_stop = stop.float().argmax(0)                                          # first True
            pr = lp.argmax(-1).gather(0, t_stop[None]).squeeze(0)
            halt[k] = {"acc": round(float((pr == tr["y"]).float().mean()), 4), "iters": round(float(t_stop.float().mean() + 1), 2)}
        res["halting"] = {"eps_kl": a.eps_kl, "entropy": a.entropy, "max": a.halt_max, "per_hop": halt}
        print(f"[eval] halting done ({time.time() - t0:.0f} s)", flush=True)

    if "noise" not in skip and a.breadth > 0:
        bd = sorted({int(d) for d in a.breadth_depths.split(",")})
        bh = parse_hops(a.breadth_hops, data.test)
        noise = {}
        for sigma in [float(s) for s in a.sigmas.split(",") if s]:
            per = {}
            for k in bh:
                rows = data.test[k][:a.breadth_n]
                P, R = [], []
                for b in range(a.breadth):
                    gen = torch.Generator(device=device).manual_seed(a.seed * 100003 + b * 7919 + k)
                    tr = run(model, rows, bd, device, a.batch, want=("pred", "resid"), sigma=sigma, gen=gen)
                    P.append(tr["pred"]); R.append(tr["resid"])
                y = tr["y"]
                P = torch.stack(P); R = torch.stack(R)                                # [B, D, n]
                ok = (P == y).float()
                vote = torch.mode(P, dim=0).values
                conv = P.gather(0, R.nan_to_num(float("inf")).argmin(0, keepdim=True)).squeeze(0)
                per[k] = {str(d): {"single": round(float(ok[:, i].mean()), 4),
                                   "vote": round(float((vote[i] == y).float().mean()), 4),
                                   "conv": round(float((conv[i] == y).float().mean()), 4),
                                   "any": round(float(ok[:, i].max(0).values.mean()), 4)} for i, d in enumerate(bd)}
            noise[str(sigma)] = per
            print(f"[eval] noise {sigma} done ({time.time() - t0:.0f} s)", flush=True)
        res["noise"] = {"breadth": a.breadth, "depths": bd, "n_per_hop": a.breadth_n, "per_sigma": noise}

    if "seen" not in skip and data.seen:
        sd = sorted({int(d) for d in a.seen_depths.split(",")})
        seen = {}
        for k in sorted(data.seen):
            tr = run(model, data.seen[k], sd, device, a.batch, want=("pred",))
            seen[k] = (tr["pred"] == tr["y"]).float().mean(1).cpu().numpy().round(4).tolist()
        res["seen"] = {"depths": sd, "acc": seen}

    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ckpt)),
                                f"eval_{os.path.splitext(os.path.basename(a.ckpt))[0]}.json")
    with open(out, "w") as f:
        json.dump(res, f)
    summary(res)
    print(f"[eval] wrote {out} ({time.time() - t0:.0f} s)", flush=True)


def summary(res):
    groups = [(2, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 70), (71, 100)]
    hops = res["hops"]
    if "depth" in res:
        D = res["depths"]
        show = [d for d in (8, 16, 32, 64, 128, 256) if d in D]
        print("[eval] noise-free accuracy, mean over hop levels in each group, at depth " + " / ".join(map(str, show)))
        for lo, hi in groups:
            ks = [k for k in hops if lo <= k <= hi]
            if ks:
                row = [np.mean([res["depth"]["acc"][k][D.index(d)] for k in ks]) for d in show]
                print(f"[eval]   {lo:3d}-{hi:<3d} " + "  ".join(f"{v:.3f}" for v in row))
        for key, v in res.get("overthink", {}).items():
            print(f"[eval]   right at {key.split()[0].split('->')[0]}, wrong at {key.split()[0].split('->')[1]} ({key.split()[1]}): "
                  f"{100 * v['lost']:.1f}% of {v['n_right']}")
    if "halting" in res:
        h = res["halting"]["per_hop"]
        print("[eval] adaptive halting: " + "  ".join(f"{lo}-{hi}: {np.mean([h[k]['acc'] for k in hops if lo <= k <= hi]):.3f} "
              f"({np.mean([h[k]['iters'] for k in hops if lo <= k <= hi]):.1f} it)" for lo, hi in groups
              if any(lo <= k <= hi for k in hops)))
    if "noise" in res:
        for s, per in res["noise"]["per_sigma"].items():
            d = str(max(res["noise"]["depths"]))
            print(f"[eval] sigma {s}, depth {d}, breadth {res['noise']['breadth']}: " + "  ".join(
                f"{k}: {v[d]['single']:.2f}/{v[d]['vote']:.2f}/{v[d]['conv']:.2f}/{v[d]['any']:.2f}" for k, v in per.items())
                + "   (single/vote/conv/any)")


if __name__ == "__main__":
    main()
