"""EqR evaluator for RNA inverse folding: refold every prediction and score it against its target structure.

Installed by looped/eqr/install_overlay.sh as <EqR>/evaluators/rna_refold.py, with rna_design.py next to it;
config/dataset/rna_design.yaml enables it as `rna_refold@RNARefold`. EqR's own accuracy / exact_accuracy
compare a prediction with the training label, which is one solution among many. These metrics do not.

Reported under "rna" (evaluate.py writes rna/<metric>; with breadth > 1 EqR passes the first start only):
  solved                  MFE structure of the prediction == target (Eterna's criterion)
  gc_a_solved             the same for Eterna's beginner strategy (GC pairs, A loops) on the same targets: the floor
  bp_dist                 base-pair distance of the prediction's MFE structure to the target / length
  pair_gc                 share of predicted pairs that are GC or CG
  loop_a                  share of unpaired positions predicted A (pair_gc = loop_a = 1 is the beginner strategy)
  label_agree             share of positions whose base equals the training label's (where the target has one)
  solved_len<lo>-<hi>     `solved` by target length
  n                       targets scored
Predictions are decoded with the target structure (rna_design.decode_sequence: bases at unpaired positions, one
pair type per pair from both positions' logits), so every pair is complementary. With a save path (evaluate.py's
eval dir) it writes rna_refold.csv with one row per target.
"""
import csv
import os

import numpy as np
import torch.distributed as dist

try:                                                   # installed in an EqR checkout
    from evaluators import rna_design as rd
except ImportError:                                    # imported from this repo (tests)
    import rna_design as rd


class RNARefold:
    required_outputs = {"logits"}

    def __init__(self, data_path=None, eval_metadata=None, length_buckets=(32, 64, 96, 128, 256, 512), **kwargs):
        rd._need_rna()                                 # fail at start-up, not at the first evaluation
        if eval_metadata is not None and eval_metadata.vocab_size != rd.VOCAB_SIZE:
            raise ValueError(f"RNARefold expects the RNA vocabulary ({rd.VOCAB_SIZE} tokens), "
                             f"got vocab_size {eval_metadata.vocab_size}")
        self.buckets = tuple(int(b) for b in length_buckets)
        self.rows = []

    def begin_eval(self):
        self.rows = []                                 # (target, prediction, label or "")

    def update_batch(self, batch, preds):
        if "logits" not in preds:
            raise KeyError("RNARefold needs the model's logits (required_outputs = {'logits'})")
        logits = preds["logits"].float().cpu().numpy()
        inputs = batch["inputs"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        for x, lg, y in zip(inputs, logits, labels):
            target = rd.decode_structure(x)
            if not target:                              # padding rows of a last, partial batch
                continue
            n = len(target)
            label = rd.decode_sequence(y, target) if (y[:n] >= 4).all() else ""
            self.rows.append((target, rd.decode_sequence(lg, target), label))

    def result(self, save_path=None, rank=0, world_size=1, group=None):
        rows = self.rows
        if world_size > 1:
            parts = [None] * world_size if rank == 0 else None
            dist.gather_object(rows, parts, dst=0, group=group)
            if rank != 0:
                return None
            rows = [r for part in parts for r in part]
        if not rows:
            return {}
        out = []
        n_pairs = n_gc = n_loop = n_a = 0
        for target, pred, label in rows:
            v = rd.verify(pred, target)
            pt = rd.pair_table(target)
            for i, j in enumerate(pt):
                if j < 0:
                    n_loop += 1
                    n_a += pred[i] == "A"
                elif j > i:
                    n_pairs += 1
                    n_gc += pred[i] + pred[j] in ("GC", "CG")
            out.append({"length": len(target), "solved": v["solved"], "bp_distance": v["bp_distance"],
                        "gc_a_solved": rd.verify(rd.gc_a_design(target), target)["solved"],
                        "label_agree": np.mean([a == b for a, b in zip(pred, label)]) if label else np.nan,
                        "target": target, "prediction": pred, "label": label, "structure": v["structure"]})
        lengths = np.array([r["length"] for r in out])
        solved = np.array([r["solved"] for r in out], dtype=float)
        metrics = {"solved": float(solved.mean()),
                   "gc_a_solved": float(np.mean([r["gc_a_solved"] for r in out])),
                   "bp_dist": float(np.mean([r["bp_distance"] / r["length"] for r in out])),
                   "pair_gc": n_gc / max(n_pairs, 1),
                   "loop_a": n_a / max(n_loop, 1),
                   "n": float(len(out))}
        agree = [r["label_agree"] for r in out if not np.isnan(r["label_agree"])]
        if agree:
            metrics["label_agree"] = float(np.mean(agree))
        lo = 0
        for hi in self.buckets:
            sel = (lengths > lo) & (lengths <= hi)
            if sel.any():
                metrics[f"solved_len{lo + 1}-{hi}"] = float(solved[sel].mean())
            lo = hi
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "rna_refold.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["length", "solved", "gc_a_solved", "bp_distance", "target", "prediction", "mfe_structure",
                            "label"])
                for r in out:
                    w.writerow([r["length"], int(r["solved"]), int(r["gc_a_solved"]), r["bp_distance"],
                                r["target"], r["prediction"], r["structure"], r["label"]])
        return {"rna": metrics}
