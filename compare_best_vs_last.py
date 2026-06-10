"""Best-vs-last test-metric comparison across all six ablation arms.

For every arm with both `test_predictions.json` (from best_model.pt) and
`test_predictions_last.json` (from last.pt, end-of-cosine), compute the
headline metrics (accuracy, weighted-F1, macro-F1-evaluable, macro-AUC over
evaluable classes) and report a side-by-side table. The "highest-leverage
no-retrain win" recommendation from `best_epoch_audit.md` is exactly this:
if the cosine-annealed end-of-training checkpoint scores higher than the
trainer's noise-pick best on the headline metric, we have an immediate
performance gain with zero additional training.

Outputs:
    D:/kvasir_capsule/outputs/best_vs_last.csv
    D:/kvasir_capsule/outputs/best_vs_last.md
"""
from __future__ import annotations

import csv
import json
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)

warnings.filterwarnings("ignore", message="Only one class is present in y_true")


ARMS = [
    ("RGB-only",                "stage2_rgb_effb0"),
    ("+P_blood only",           "stage2_pi_blood_only_effb0"),
    ("+H_AFI only",             "stage2_pi_hafi_only_effb0"),
    ("Physics-only",            "stage2_pi_physics_only_effb0"),
    ("Full PI",                 "stage2_pi_effb0"),
    ("Distill",                 "stage2_distill_effb0"),
]


def _metrics(rec) -> Dict[str, float]:
    y = np.asarray(rec["labels"])
    p = np.asarray(rec["preds"])
    probs = np.asarray(rec["probs"])
    classes = rec["class_names"]
    acc = float(accuracy_score(y, p))
    wf1 = float(f1_score(y, p, average="weighted", zero_division=0))
    # macro F1 over evaluable classes (positive support in test)
    eval_idx = [i for i in range(len(classes)) if (y == i).sum() > 0]
    f1s = []
    for i in eval_idx:
        f1s.append(f1_score((y == i).astype(int), (p == i).astype(int),
                             zero_division=0))
    mf1e = float(np.mean(f1s)) if f1s else float("nan")
    aucs = []
    for i in eval_idx:
        try:
            aucs.append(float(roc_auc_score((y == i).astype(int), probs[:, i])))
        except ValueError:
            pass
    macro_auc = float(np.mean(aucs)) if aucs else float("nan")
    return {"acc": acc, "weighted_f1": wf1, "macro_f1_eval": mf1e,
            "macro_auc_eval": macro_auc, "n_evaluable": len(eval_idx)}


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out_dir = "D:/kvasir_capsule/outputs"
    csv_out = os.path.join(out_dir, "best_vs_last.csv")
    md_out = os.path.join(out_dir, "best_vs_last.md")

    rows: List[Dict] = []
    for label, dirname in ARMS:
        b = os.path.join(out_dir, dirname, "test_predictions.json")
        l = os.path.join(out_dir, dirname, "test_predictions_last.json")
        if not (os.path.isfile(b) and os.path.isfile(l)):
            print(f"[best_vs_last] WARN missing predictions for {dirname}; skipping")
            continue
        with open(b, "r", encoding="utf-8") as f:
            rb = json.load(f)
        with open(l, "r", encoding="utf-8") as f:
            rl = json.load(f)
        mb = _metrics(rb)
        ml = _metrics(rl)
        rows.append({"arm": label, "dirname": dirname, "best": mb, "last": ml})

    # CSV
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arm", "metric", "best", "last", "delta_last_minus_best"])
        for r in rows:
            for m in ["acc", "weighted_f1", "macro_f1_eval", "macro_auc_eval"]:
                d = r["last"][m] - r["best"][m]
                w.writerow([r["arm"], m, f"{r['best'][m]:.4f}",
                            f"{r['last'][m]:.4f}", f"{d:+.4f}"])
    print(f"[best_vs_last] wrote {csv_out}")

    # Markdown
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# best_model.pt vs last.pt across the six ablation arms\n\n")
        f.write("Test-set headline metrics (n=6,423 frames; macro-F1 and macro-AUC "
                "over the 11 classes with positive test support).\n")
        f.write("Cells are `best → last (Δ)`. Δ > 0 means the end-of-cosine "
                "`last.pt` outperforms the trainer's chosen `best_model.pt`.\n\n")
        f.write("| Arm | Accuracy | Weighted-F1 | Macro-F1 (eval) | Macro-AUC (eval) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            cells = []
            for m in ["acc", "weighted_f1", "macro_f1_eval", "macro_auc_eval"]:
                d = r["last"][m] - r["best"][m]
                arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
                cells.append(f"{r['best'][m]:.3f} → {r['last'][m]:.3f} ({d:+.3f} {arrow})")
            f.write(f"| {r['arm']} | " + " | ".join(cells) + " |\n")
        # Net findings
        f.write("\n## Findings\n\n")
        any_last_wins = False
        for m in ["acc", "weighted_f1", "macro_f1_eval", "macro_auc_eval"]:
            wins = sum(1 for r in rows if r["last"][m] > r["best"][m])
            ties = sum(1 for r in rows if r["last"][m] == r["best"][m])
            if wins > 0:
                any_last_wins = True
            mean_d = float(np.mean([r["last"][m] - r["best"][m] for r in rows]))
            f.write(f"- {m}: `last.pt` wins on {wins}/{len(rows)} arms (mean delta {mean_d:+.4f}).\n")
        if any_last_wins:
            f.write("\nWhere `last.pt` wins, switching the deployed checkpoint requires no retraining "
                    "and is the highest-leverage immediate fix per `best_epoch_audit.md`.\n")
    print(f"[best_vs_last] wrote {md_out}")

    # Console summary
    print("\n=== best vs last (test set) ===")
    for r in rows:
        print(f"\n  {r['arm']}  ({r['dirname']})")
        for m in ["acc", "weighted_f1", "macro_f1_eval", "macro_auc_eval"]:
            d = r["last"][m] - r["best"][m]
            arrow = "↑" if d > 0 else ("↓" if d < 0 else "·")
            print(f"    {m:18s}  best={r['best'][m]:.4f}  last={r['last'][m]:.4f}  "
                  f"Δ={d:+.4f} {arrow}")


if __name__ == "__main__":
    main()
