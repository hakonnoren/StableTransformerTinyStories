"""EqR evaluator for RNA inverse folding: refold every prediction and score it against its target structure.

Installed by looped/eqr/install_overlay.sh as <EqR>/evaluators/rna_refold.py, with rna_design.py next to it;
config/dataset/rna_design.yaml enables it as `rna_refold@RNARefold`. EqR's own accuracy / exact_accuracy
compare a prediction with the training label, which is one solution among many. These metrics do not.

Reported under "rna" (evaluate.py writes rna/<metric>; with breadth > 1 EqR passes the first start only):
  solved                  MFE structure of the prediction == target (Eterna's criterion)
  bp_dist                 base-pair distance of the prediction's MFE structure to the target / length
  pairs_ok                share of target pairs whose two predicted bases can pair (GC, AU, GU)
  label_agree             share of positions equal to the training label (where the target has one)
  solved_len<lo>-<hi>     `solved` by target length
  n                       targets scored
Predictions are decoded with an argmax over A, C, G, U only, so every prediction is a sequence. With a save path
(evaluate.py's eval dir) it writes rna_refold.csv with one row per target.
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
        ids = (4 + preds["logits"][..., 4:8].float().argmax(-1)).cpu().numpy()
        inputs = batch["inputs"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        for x, p, y in zip(inputs, ids, labels):
            target = rd.decode_structure(x)
            if not target:                              # padding rows of a last, partial batch
                continue
            n = len(target)
            label = rd.decode_sequence(y, n) if (y[:n] >= 4).all() else ""
            self.rows.append((target, rd.decode_sequence(p, n), label))

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
        for target, pred, label in rows:
            v = rd.verify(pred, target)
            pairs = rd.pairs_of(target)
            out.append({"length": len(target), "solved": v["solved"], "bp_distance": v["bp_distance"],
                        "pairs_ok": np.mean([pred[i] + pred[j] in rd.PAIRS for i, j in pairs]) if pairs else 1.0,
                        "label_agree": np.mean([a == b for a, b in zip(pred, label)]) if label else np.nan,
                        "target": target, "prediction": pred, "label": label, "structure": v["structure"]})
        lengths = np.array([r["length"] for r in out])
        solved = np.array([r["solved"] for r in out], dtype=float)
        metrics = {"solved": float(solved.mean()),
                   "bp_dist": float(np.mean([r["bp_distance"] / r["length"] for r in out])),
                   "pairs_ok": float(np.mean([r["pairs_ok"] for r in out])),
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
                w.writerow(["length", "solved", "bp_distance", "pairs_ok", "target", "prediction", "mfe_structure",
                            "label"])
                for r in out:
                    w.writerow([r["length"], int(r["solved"]), r["bp_distance"], f"{r['pairs_ok']:.3f}",
                                r["target"], r["prediction"], r["structure"], r["label"]])
        return {"rna": metrics}
