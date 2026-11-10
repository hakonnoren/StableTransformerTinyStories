r"""Fine-tune a released multi-hop model (Kohli et al.) in one coupling, with the authors' curriculum, towards long chains.

    python -m looped.multihop.train --ltg-dir $LTG --data-dir $LTG/data/multi_hop \
        --init-ckpt $LTG/checkpoints/multi_hop/final_models/r_dyn/checkpoint_epoch_5388.pt \
        --coupling relax --delta 1 --carry-max 64 --sigma-train 0.01 --out runs/multihop/relax_d1_carry_s0

Curriculum (the authors'): train on all facts with at most k hops; when held-out k-hop accuracy reaches --threshold,
add k + 1, up to --max-hop (40, the longest in train.json). Decisions use the dev split (data.py), at the best of
--curr-depths iterations (one rule for every coupling; the authors used the training maximum, 8). The released r_dyn
model learned up to 22 hops, so the curriculum starts at --start-hop 23 for every arm.

Per step (the authors' recipe): a uniform batch from the current pool, R ~ clip(Poisson(--r-mean), --r-min, --r-max)
iterations with gradient, cross-entropy on the answer at the last position. Optional, from EqR's recipe (the Sudoku
runs): --carry-max C runs U{1..C} iterations without gradient first, with probability --carry-prob (a detached carry:
the gradient is truncated to the last R iterations, as EqR's), and --sigma-train adds relative noise after every update.

Output (--out): train_log.jsonl (loss every --log-every steps), curriculum.jsonl (one line per evaluation: hop k, dev
accuracy per depth, stage advances), stage_k<k>.pt when hop k is reached, final.pt; config.json. With --wandb-project,
the same to W&B.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from looped.multihop.data import MultiHop, collate
from looped.multihop.model import Coupled, load_base


def evaluate_split(model, rows, depths, device, bs=1024):
    """Exact accuracy per depth on a list of rows (noise-free)."""
    hits = np.zeros(len(depths))
    for i in range(0, len(rows), bs):
        ids, mask, y = collate(rows[i:i + bs], device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            tr = model.trajectory(ids, mask, depths)
        hits += (tr["pred"] == y).float().sum(1).cpu().numpy()
    return hits / max(len(rows), 1)


def save(model, path, cfg, extra=None):
    arch = {k: cfg[k] for k in ("d_model", "n_layer", "n_head")}
    torch.save({"model": model.base.state_dict(), "coupling": model.describe(), "arch": arch, "config": cfg,
                **(extra or {})}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ltg-dir", required=True, help="checkout of OSU-NLP-Group/Loop-Think-Generalize")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coupling", default="vanilla", choices=["vanilla", "relax", "twin"])
    ap.add_argument("--gamma", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--delta", type=float, default=0.0)
    ap.add_argument("--port-warmup", type=int, default=2000, help="steps over which the coupling blends in (mix 0 -> 1)")
    ap.add_argument("--start-hop", type=int, default=23)
    ap.add_argument("--max-hop", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--curr-depths", default="8,16,32")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--steps", type=int, default=150000, help="step budget (the run also stops at --max-hop)")
    ap.add_argument("--hours", type=float, default=20.0, help="wall-clock budget for training")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--r-min", type=int, default=2)
    ap.add_argument("--r-max", type=int, default=8)
    ap.add_argument("--r-mean", type=float, default=4.0)
    ap.add_argument("--carry-max", type=int, default=0)
    ap.add_argument("--carry-prob", type=float, default=0.5)
    ap.add_argument("--sigma-train", type=float, default=0.0)
    ap.add_argument("--d-model", type=int, default=768, help="architecture of --init-ckpt (the released: 768/4/12)")
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--wandb-project", default="")
    ap.add_argument("--wandb-entity", default="")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator(device=device).manual_seed(a.seed + 1)
    data = MultiHop(a.data_dir)
    arch = dict(d_model=a.d_model, n_layer=a.n_layer, n_head=a.n_head)
    base = load_base(a.ltg_dir, len(data.vocab), a.init_ckpt, max_len=128, device=device, **arch)
    model = Coupled(base, a.coupling, a.gamma, a.alpha, a.delta).to(device)
    cfg = {**vars(a), "coupling_params": model.describe(), "params": sum(p.numel() for p in model.parameters())}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    wb = None
    if a.wandb_project:
        try:
            import wandb
            wb = wandb.init(project=a.wandb_project, entity=a.wandb_entity or None, name=a.run_name or None,
                            id=a.run_name or None, resume="allow", config=cfg, dir=a.out)
        except Exception as ex:                                    # never let logging kill a run
            print(f"[train] W&B disabled: {ex}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    if a.compile:                                                  # the block stack only: R and the carry vary per step
        model.f = torch.compile(model.f, dynamic=True)
    depths = [int(d) for d in a.curr_depths.split(",")]

    k = a.start_hop
    pool = data.train_upto(k)
    flog = open(os.path.join(a.out, "train_log.jsonl"), "a")
    fcur = open(os.path.join(a.out, "curriculum.jsonl"), "a")

    def curr_eval(step, k):
        model.eval()
        rec = {"step": step, "hop": k, "mix": model.mix}
        for kk in sorted({2, 10, 20, k - 1, k, k + 5} & set(data.dev)):
            acc = evaluate_split(model, data.dev[kk], depths, device)
            rec[f"dev{kk}"] = {str(d): round(float(v), 4) for d, v in zip(depths, acc)}
        model.train()
        return rec

    rec = curr_eval(0, k)
    print(f"[train] start: {a.coupling} {model.describe()}, {cfg['params']:,} parameters, hop {k}, "
          f"dev at start {json.dumps({kk: v for kk, v in rec.items() if kk.startswith('dev')})}", flush=True)
    fcur.write(json.dumps(rec) + "\n"); fcur.flush()
    t0 = time.time()
    model.train()
    run_loss, run_n, last_loss = 0.0, 0, float("nan")
    done = False
    for step in range(1, a.steps + 1):
        model.mix = min(1.0, step / a.port_warmup) if a.port_warmup else 1.0
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / a.warmup)
        ids, mask, y = data.batch(pool[rng.integers(len(pool), size=a.batch)], device)
        r = int(min(a.r_max, max(a.r_min, rng.poisson(a.r_mean))))
        n_carry = int(rng.integers(1, a.carry_max + 1)) if a.carry_max and rng.random() < a.carry_prob else 0
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids, mask, r, n_carry, a.sigma_train, gen)
        loss = F.cross_entropy(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if a.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        if not math.isfinite(loss.item()):
            print(f"[train] DIVERGED at step {step}", flush=True)
            break
        run_loss += loss.item(); run_n += 1
        if step % a.log_every == 0:
            lr_rec = {"step": step, "loss": run_loss / run_n, "hop": k, "mix": model.mix, "sec": round(time.time() - t0, 1)}
            flog.write(json.dumps(lr_rec) + "\n"); flog.flush()
            if wb:
                wb.log({"train/loss": lr_rec["loss"], "curriculum/hop": k}, step=step)
            last_loss, run_loss, run_n = lr_rec["loss"], 0.0, 0
        if step % a.eval_every == 0:
            rec = curr_eval(step, k)
            best = max(rec[f"dev{k}"].values()) if f"dev{k}" in rec else 0.0
            rec["best_k"] = best
            if best >= a.threshold and model.mix >= 1.0:
                rec["advance"] = True
                save(model, os.path.join(a.out, f"stage_k{k}.pt"), cfg, {"step": step, "hop": k})
                print(f"[train] step {step}: hop {k} reached ({best:.3f}), {(time.time() - t0) / 3600:.2f} h", flush=True)
                k += 1
                if k > a.max_hop:
                    done = True
                else:
                    pool = data.train_upto(k)
            fcur.write(json.dumps(rec) + "\n"); fcur.flush()
            if wb:
                wb.log({f"dev/{kk}@{d}": v for kk, dd in rec.items() if kk.startswith("dev") for d, v in dd.items()}
                       | {"curriculum/best_k": best}, step=step)
            if step % (10 * a.eval_every) == 0:
                print(f"[train] step {step}: hop {k}, loss {last_loss:.3f}, dev best {best:.3f}, "
                      f"{(time.time() - t0) / 3600:.2f} h", flush=True)
        if done or (time.time() - t0) / 3600 > a.hours:
            break
    save(model, os.path.join(a.out, "final.pt"), cfg, {"step": step, "hop": k, "done": done})
    print(f"[train] finished at step {step}: curriculum hop {k}{' (all reached)' if done else ''}, "
          f"{(time.time() - t0) / 3600:.2f} h", flush=True)
    if wb:
        wb.finish()


if __name__ == "__main__":
    main()
