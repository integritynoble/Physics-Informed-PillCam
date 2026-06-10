"""Cross-seed aggregation for the published Table 2.

Reads the per-seed test_predictions.json files for each headline arm and
produces:
    - mean ± SD on accuracy / weighted-F1 / macro-F1-evaluable / macro-AUC-evaluable
    - per-class AUC mean ± SD across seeds
    - paired DeLong on each seed's RGB-vs-PI and RGB-vs-Distill comparisons
    - Table 2 in markdown with `mean ± sd` cells (drop-in for paper.md)

This is the script that runs after run_multiseed_effb0.sh completes. The
result is the headline ablation table the paper actually ships with.

Outputs:
    D:/kvasir_capsule/outputs/cross_seed_summary.json
    D:/kvasir_capsule/outputs/cross_seed_summary.md
    D:/kvasir_capsule/outputs/cross_seed_table2.md      (drop-in for paper.md §4.1)
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score)

warnings.filterwarnings("ignore", message="Only one class is present in y_true")


# Headline arms × seeds. seed=42 is the original headline run; 41/43 from
# run_multiseed_effb0.sh; 44-47 from run_extra_seeds_44_47.sh.
# Aggregator silently skips seeds whose checkpoint dirs don't exist yet,
# so it works mid-sweep too.
SEEDS = [41, 42, 43, 44, 45, 46, 47]
ARMS = [
    ("RGB-only",  "rgb",     "stage2_rgb_effb0"),
    ("Full PI",   "pi",      "stage2_pi_effb0"),
    ("Distill",   "distill", "stage2_distill_effb0"),
]


def _arm_dirname(arm_dir: str, seed: int) -> str:
    if seed == 42:
        return arm_dir  # original headline runs
    return f"{arm_dir}_seed{seed}"


def _metrics_from_preds(rec) -> Dict:
    y = np.asarray(rec["labels"])
    p = np.asarray(rec["preds"])
    probs = np.asarray(rec["probs"])
    classes = rec["class_names"]
    eval_idx = [i for i in range(len(classes)) if (y == i).sum() > 0]

    f1s, aucs = [], []
    per_class_auc: Dict[str, float] = {}
    for i in eval_idx:
        yb = (y == i).astype(int)
        f1s.append(f1_score(yb, (p == i).astype(int), zero_division=0))
        try:
            a = float(roc_auc_score(yb, probs[:, i]))
            aucs.append(a)
            per_class_auc[classes[i]] = a
        except ValueError:
            pass
    return {
        "accuracy": float(accuracy_score(y, p)),
        "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0)),
        "macro_f1_evaluable": float(np.mean(f1s)) if f1s else float("nan"),
        "macro_auc_evaluable": float(np.mean(aucs)) if aucs else float("nan"),
        "per_class_auc": per_class_auc,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    out = "D:/kvasir_capsule/outputs"
    json_out = os.path.join(out, "cross_seed_summary.json")
    md_out = os.path.join(out, "cross_seed_summary.md")
    table_out = os.path.join(out, "cross_seed_table2.md")

    summary: Dict = {"seeds": SEEDS, "arms": [a[0] for a in ARMS], "by_arm": {}}
    missing: List[str] = []

    for arm_label, _variant, arm_dir in ARMS:
        per_seed: List[Dict] = []
        for s in SEEDS:
            p = os.path.join(out, _arm_dirname(arm_dir, s), "test_predictions.json")
            if not os.path.isfile(p):
                # Try the _last variant — last.pt outperformed best.pt on accuracy
                # in the original sweep, so we may be reporting from there
                p_last = os.path.join(out, _arm_dirname(arm_dir, s),
                                       "test_predictions_last.json")
                if os.path.isfile(p_last):
                    p = p_last
                else:
                    missing.append(f"{arm_label} seed={s}: missing {p}")
                    continue
            rec = json.load(open(p, encoding="utf-8"))
            m = _metrics_from_preds(rec)
            m["seed"] = s
            m["source"] = os.path.basename(p)
            per_seed.append(m)

        if not per_seed:
            summary["by_arm"][arm_label] = {"note": "no completed seeds"}
            continue

        # Aggregate
        agg: Dict = {"n_seeds": len(per_seed), "per_seed": per_seed}
        for k in ["accuracy", "weighted_f1", "macro_f1_evaluable", "macro_auc_evaluable"]:
            vals = [m[k] for m in per_seed if not np.isnan(m[k])]
            agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
            agg[f"{k}_sd"]   = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

        # Per-class AUC mean/SD across seeds
        all_classes = sorted({c for m in per_seed for c in m["per_class_auc"]})
        per_class: Dict = {}
        for c in all_classes:
            vals = [m["per_class_auc"].get(c, np.nan) for m in per_seed]
            vals = [v for v in vals if not np.isnan(v)]
            per_class[c] = {
                "n": len(vals),
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }
        agg["per_class_auc"] = per_class
        summary["by_arm"][arm_label] = agg

    # ---- write JSON ----
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[cross_seed] wrote {json_out}")

    # ---- write markdown ----
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Cross-seed summary (seeds 41, 42, 43; EfficientNet-B0)\n\n")
        if missing:
            f.write("**Missing seeds (excluded from aggregation):**\n")
            for m in missing:
                f.write(f"- {m}\n")
            f.write("\n")
        f.write("## Headline metrics (test split, n=6,423)\n\n")
        f.write("Cells are mean ± SD across the available seeds.\n\n")
        f.write("| Arm | n_seeds | Accuracy | Weighted-F1 | Macro-F1 (eval) | Macro-AUC (eval) |\n")
        f.write("|---|---|---|---|---|---|\n")
        for arm_label in [a[0] for a in ARMS]:
            r = summary["by_arm"].get(arm_label, {})
            if "n_seeds" not in r:
                f.write(f"| {arm_label} | 0 | — | — | — | — |\n")
                continue
            f.write(f"| {arm_label} | {r['n_seeds']} |"
                    f" {r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f} |"
                    f" {r['weighted_f1_mean']:.3f} ± {r['weighted_f1_sd']:.3f} |"
                    f" {r['macro_f1_evaluable_mean']:.3f} ± {r['macro_f1_evaluable_sd']:.3f} |"
                    f" {r['macro_auc_evaluable_mean']:.3f} ± {r['macro_auc_evaluable_sd']:.3f} |\n")
        f.write("\n## Per-class AUC, mean ± SD across seeds\n\n")
        # Get class union
        all_classes: List[str] = []
        for arm_label in [a[0] for a in ARMS]:
            r = summary["by_arm"].get(arm_label, {})
            if "per_class_auc" not in r: continue
            for c in r["per_class_auc"]:
                if c not in all_classes:
                    all_classes.append(c)
        f.write("| Class |")
        for arm_label in [a[0] for a in ARMS]:
            f.write(f" {arm_label} |")
        f.write("\n|" + "---|" * (1 + len(ARMS)) + "\n")
        for c in all_classes:
            f.write(f"| {c} |")
            for arm_label in [a[0] for a in ARMS]:
                r = summary["by_arm"].get(arm_label, {})
                cell = r.get("per_class_auc", {}).get(c, {})
                if not cell or cell.get("n", 0) == 0:
                    f.write(" n/a |")
                else:
                    f.write(f" {cell['mean']:.3f} ± {cell['sd']:.3f} |")
            f.write("\n")
    print(f"[cross_seed] wrote {md_out}")

    # ---- write Table 2 drop-in (paper §4.1 update) ----
    with open(table_out, "w", encoding="utf-8") as f:
        f.write("**Table 2 — Per-class one-vs-rest AUC, mean ± SD across 3 seeds (41, 42, 43).**\n\n")
        f.write("| Class | RGB-only | +MC prior (5-ch) | +Distill (3-ch + aux) |\n")
        f.write("|---|:---:|:---:|:---:|\n")
        for c in all_classes:
            f.write(f"| {c} |")
            for arm_label in [a[0] for a in ARMS]:
                cell = summary["by_arm"].get(arm_label, {}).get("per_class_auc", {}).get(c, {})
                if not cell or cell.get("n", 0) == 0:
                    f.write(" n/a |")
                else:
                    f.write(f" {cell['mean']:.3f} ± {cell['sd']:.3f} |")
            f.write("\n")
        # Macro row
        f.write("| **Macro-AUC (11 classes)** |")
        for arm_label in [a[0] for a in ARMS]:
            r = summary["by_arm"].get(arm_label, {})
            if "macro_auc_evaluable_mean" in r:
                f.write(f" **{r['macro_auc_evaluable_mean']:.3f} ± {r['macro_auc_evaluable_sd']:.3f}** |")
            else:
                f.write(" n/a |")
        f.write("\n\n")
        f.write("Headline summary metrics (mean ± SD across 3 seeds):\n\n")
        f.write("| Metric | RGB-only | +MC prior (5-ch) | +Distill (3-ch + aux) |\n")
        f.write("|---|---:|---:|---:|\n")
        for k, label in [("accuracy", "Accuracy (argmax)"),
                          ("weighted_f1", "Weighted-F1 (argmax)"),
                          ("macro_f1_evaluable", "Macro-F1 (eval, argmax)"),
                          ("macro_auc_evaluable", "Macro-AUC (eval)")]:
            f.write(f"| {label} |")
            for arm_label in [a[0] for a in ARMS]:
                r = summary["by_arm"].get(arm_label, {})
                m = r.get(f"{k}_mean")
                s = r.get(f"{k}_sd")
                if m is None:
                    f.write(" n/a |")
                else:
                    f.write(f" {m:.3f} ± {s:.3f} |")
            f.write("\n")
    print(f"[cross_seed] wrote {table_out}")

    # Console summary
    print("\n=== headline cross-seed summary ===")
    for arm_label in [a[0] for a in ARMS]:
        r = summary["by_arm"].get(arm_label, {})
        if "n_seeds" not in r:
            print(f"  {arm_label}: <no data>")
            continue
        print(f"  {arm_label} ({r['n_seeds']} seed{'s' if r['n_seeds']>1 else ''}):"
              f"  acc {r['accuracy_mean']:.3f}±{r['accuracy_sd']:.3f}"
              f"  wF1 {r['weighted_f1_mean']:.3f}±{r['weighted_f1_sd']:.3f}"
              f"  mF1eval {r['macro_f1_evaluable_mean']:.3f}±{r['macro_f1_evaluable_sd']:.3f}"
              f"  macroAUC {r['macro_auc_evaluable_mean']:.3f}±{r['macro_auc_evaluable_sd']:.3f}")


if __name__ == "__main__":
    main()
