"""Clinical operating-point analysis on the Windows-trained EffB0 paper checkpoints.

For each (RGB, +PI) arm and each of the 4 clinically-priority classes
(Lymphangiectasia, Angiectasia, Blood - fresh, Ulcer):

  - sensitivity (TPR) at fixed false-positive rate (FPR) = 1%, 5%, 10%
  - FPR required to reach fixed sensitivity = 80%, 90%, 95% (NaN if unreachable)
  - average precision (PR-AUC)

Aggregates across 6 paper-headline seeds {41, 42, 43, 44, 45, 47}.
Writes per-class results to docs/operating_point_summary.md.
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, average_precision_score, roc_auc_score


ROOT = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone")
SEEDS = [41, 42, 43, 44, 45, 47]
ARMS = {"rgb": "RGB-only", "pi": "+PI input-fusion"}
PRIORITY_CLASSES = ["Lymphangiectasia", "Angiectasia", "Blood - fresh", "Ulcer"]
FPR_FIXED = [0.01, 0.05, 0.10]
TPR_FIXED = [0.80, 0.90, 0.95]


def load_seed(arm: str, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    d = np.load(ROOT / f"effb0_paper_seed{seed}_{arm}" / "test_predictions.npz", allow_pickle=True)
    return d["probs"], d["labels"], list(d["classes"])


def sensitivity_at_fpr(y_true: np.ndarray, score: np.ndarray, target_fpr: float) -> float:
    """TPR at the threshold whose FPR is the highest value <= target_fpr.
    If no such threshold exists (all FPRs > target_fpr) returns NaN."""
    fpr, tpr, _ = roc_curve(y_true, score)
    valid = fpr <= target_fpr
    if not valid.any():
        return float("nan")
    return float(tpr[valid].max())


def fpr_at_tpr(y_true: np.ndarray, score: np.ndarray, target_tpr: float) -> float:
    """Smallest FPR achieving TPR >= target_tpr. NaN if unreachable."""
    fpr, tpr, _ = roc_curve(y_true, score)
    valid = tpr >= target_tpr
    if not valid.any():
        return float("nan")
    return float(fpr[valid].min())


def analyze_class(arm: str, cls_name: str, cls_idx: int) -> dict:
    """Aggregate per-seed operating-point metrics for one (arm, class)."""
    sens_at_fpr = {f"{int(f*100)}%": [] for f in FPR_FIXED}
    fpr_at_sens = {f"{int(t*100)}%": [] for t in TPR_FIXED}
    aucs = []
    aps = []
    for s in SEEDS:
        probs, labels, classes = load_seed(arm, s)
        assert classes[cls_idx] == cls_name, f"class mismatch: {classes[cls_idx]} != {cls_name}"
        y = (labels == cls_idx).astype(int)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        score = probs[:, cls_idx]
        aucs.append(roc_auc_score(y, score))
        aps.append(average_precision_score(y, score))
        for f in FPR_FIXED:
            sens_at_fpr[f"{int(f*100)}%"].append(sensitivity_at_fpr(y, score, f))
        for t in TPR_FIXED:
            fpr_at_sens[f"{int(t*100)}%"].append(fpr_at_tpr(y, score, t))
    return {
        "auc_mean": float(np.mean(aucs)),
        "auc_std":  float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        "ap_mean":  float(np.mean(aps)),
        "ap_std":   float(np.std(aps, ddof=1)) if len(aps) > 1 else 0.0,
        "n_seeds":  len(aucs),
        "sens_at_fpr": {k: {"mean": float(np.nanmean(v)) if v else float("nan"),
                            "std":  float(np.nanstd(v, ddof=1)) if (v and len(v) > 1) else 0.0}
                        for k, v in sens_at_fpr.items()},
        "fpr_at_sens": {k: {"mean": float(np.nanmean(v)) if v else float("nan"),
                            "std":  float(np.nanstd(v, ddof=1)) if (v and len(v) > 1) else 0.0}
                        for k, v in fpr_at_sens.items()},
    }


def main():
    # Confirm class indices
    _, _, classes = load_seed("rgb", 41)
    cls_idx = {c: i for i, c in enumerate(classes)}
    print(f"All 14 classes: {classes}")
    print()

    results = {arm: {} for arm in ARMS}
    for arm in ARMS:
        for cls_name in PRIORITY_CLASSES:
            results[arm][cls_name] = analyze_class(arm, cls_name, cls_idx[cls_name])

    # Pretty print
    print(f"{'class':<22} {'arm':<22} {'AUC':>14} {'AP':>14} {'Sens@5%FPR':>13} {'Sens@10%FPR':>13} {'FPR@80%Sens':>13} {'FPR@90%Sens':>13}")
    print("-" * 150)
    for cls_name in PRIORITY_CLASSES:
        for arm, label in ARMS.items():
            r = results[arm][cls_name]
            s_5 = r["sens_at_fpr"]["5%"]
            s_10 = r["sens_at_fpr"]["10%"]
            f_80 = r["fpr_at_sens"]["80%"]
            f_90 = r["fpr_at_sens"]["90%"]
            print(f"{cls_name:<22} {label:<22} {r['auc_mean']:.3f}±{r['auc_std']:.3f}  {r['ap_mean']:.3f}±{r['ap_std']:.3f}  {s_5['mean']:.3f}±{s_5['std']:.3f}  {s_10['mean']:.3f}±{s_10['std']:.3f}  {f_80['mean']:.3f}±{f_80['std']:.3f}  {f_90['mean']:.3f}±{f_90['std']:.3f}")

    out = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/operating_point_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
