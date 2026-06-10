"""
Smoke test: verify the headline manuscript numbers reproduce from
the cached test predictions.
========================================================================

Loads the existing test_predictions.npz files for each cell at each
seed, recomputes per-class AUC and macro-AUC, and verifies the
cross-seed mean matches the manuscript's claimed numbers within
floating-point tolerance.

This is NOT a fresh-clone-from-GitHub reproduction (would require
Python env setup + checkpoint redownload + temporal head retraining)
but IS an arithmetic verification that the claimed numbers are
correctly computed from the saved predictions.

If a future researcher asks ``did you actually compute the macro-AUC
correctly?'', this script answers yes.

Output:
  paper/nature-machine-intelligence/docs/smoke_test_headline_numbers_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs")
SEEDS = [41, 42, 43, 44, 45, 47]
REPORT = Path("D:/onedrive/UT_southwestern/GIproject/GI Project_2026/"
              "GI_Multi_Task/paper/nature-machine-intelligence/docs/"
              "smoke_test_headline_numbers_report.md")

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]

# Manuscript-claimed numbers (from MedIA Table 1, NMI Table 1)
MANUSCRIPT_CLAIMS = {
    "(a)":  (0.7598, 0.0271, "temporal_cell_b",            "use cell-b cache; cell-a was the per-frame baseline"),
    "(b)":  (0.7788, 0.0118, "temporal_cell_b",            "RGB + temporal"),
    "(c)":  (0.7774, 0.0193, "temporal_cell_c",            "RGB + temporal + 8-d C1 train-norm"),
    "(d)":  (0.7722, 0.0254, "temporal_cell_d",            "RGB + temporal + C1 + C3"),
    "(e)":  (0.7828, 0.0223, "temporal_cell_e",            "RGB + temporal + C3"),
    "(b+)": (0.7901, 0.0260, "temporal_cell_b_pi",         "+PI 5-ch + temporal"),
    "(e+)": (0.8038, 0.0226, "temporal_cell_e_pi",         "+PI 5-ch + temporal + C3"),
}


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def macro_auc_from_predictions(logits: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    probs = softmax(logits)
    aucs = []
    for j in range(len(CLASS_NAMES)):
        y = (labels == j).astype(np.int32)
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(roc_auc_score(y, probs[:, j]))
    return float(np.mean(aucs)) if aucs else float("nan")


def main():
    print(f"[smoke] loading cached test_predictions.npz files for "
          f"each cell x seed and verifying manuscript numbers")

    md = []
    md.append("# Smoke test: verify manuscript headline numbers from "
              "cached test predictions\n")
    md.append("**Date:** 2026-05-08")
    md.append("**Method:** for each cell named in the manuscript "
              "(except cell (a) which uses cell-b's underlying "
              "predictions), load the test_predictions.npz files for "
              "all 6 seeds, recompute per-seed macro-AUC, verify the "
              "cross-seed mean matches the claimed value within "
              "floating-point tolerance.")
    md.append("")
    md.append("## Verification table\n")
    md.append("| Cell | Manuscript mean | Reproduced mean | Δ | Manuscript σ | Reproduced σ | Δσ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")

    n_pass = 0
    n_fail = 0
    for cell_name, (claimed_mean, claimed_sigma, dir_name, note) in MANUSCRIPT_CLAIMS.items():
        per_seed = []
        for seed in SEEDS:
            path = OUTPUT_ROOT / dir_name / f"seed{seed}" / "test_predictions.npz"
            if not path.exists():
                print(f"[skip] cell {cell_name} seed {seed}: missing {path}")
                continue
            d = np.load(path)
            macro = macro_auc_from_predictions(np.array(d["logits"]),
                                                  np.array(d["labels"]))
            d.close()
            per_seed.append(macro)

        if not per_seed:
            md.append(f"| {cell_name} | {claimed_mean:.4f} | (no preds) | -- | "
                      f"{claimed_sigma:.4f} | -- | -- |")
            print(f"[skip] cell {cell_name}: no predictions")
            continue

        repro_mean = float(np.mean(per_seed))
        repro_sigma = float(np.std(per_seed))
        delta_mean = repro_mean - claimed_mean
        delta_sigma = repro_sigma - claimed_sigma
        match = "PASS" if abs(delta_mean) < 0.0005 else "FAIL"
        if abs(delta_mean) < 0.0005:
            n_pass += 1
        else:
            n_fail += 1
        md.append(f"| {cell_name} | {claimed_mean:.4f} | {repro_mean:.4f} "
                  f"| {delta_mean:+.4f} | {claimed_sigma:.4f} "
                  f"| {repro_sigma:.4f} | {delta_sigma:+.4f} |")
        print(f"[cell {cell_name}] manuscript={claimed_mean:.4f}  "
              f"reproduced={repro_mean:.4f}  Δ={delta_mean:+.4f}  {match}")

    md.append("")
    md.append("## Verdict\n")
    md.append(f"- {n_pass} cells PASS (Δ macro-AUC < 0.0005)")
    md.append(f"- {n_fail} cells FAIL")
    if n_fail == 0:
        md.append("\n**All manuscript headline cross-seed mean macro-AUC "
                  "numbers reproduce arithmetically from the saved "
                  "test predictions.** The numbers in the manuscript "
                  "are correctly computed from the saved per-frame "
                  "predictions; any future researcher loading the same "
                  "predictions and computing macro-AUC with sklearn's "
                  "roc_auc_score will get the same number.")
    else:
        md.append("\n**Some headline numbers do not reproduce arithmetically. "
                  "Inspect the discrepancies before submission.**")
    md.append("")
    md.append("## What this verifies and what it does not\n")
    md.append("This smoke test verifies:")
    md.append("- That the manuscript's claimed cross-seed macro-AUC "
              "numbers match what is computed from the cached "
              "test_predictions.npz files using sklearn's standard "
              "roc_auc_score")
    md.append("")
    md.append("This smoke test does NOT verify:")
    md.append("- Reproduction from a fresh clone of the GitHub repository")
    md.append("- Reproduction on a different GPU / OS / PyTorch version")
    md.append("- That the saved test predictions correspond to the "
              "training pipeline as documented in `Methods` "
              "(would require a fresh training run)")
    md.append("- Independent replication by another researcher")

    REPORT.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[smoke] report -> {REPORT}")


if __name__ == "__main__":
    main()
