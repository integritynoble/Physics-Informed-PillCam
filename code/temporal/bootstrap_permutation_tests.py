"""
Bootstrap CIs and permutation tests on existing capsule eight-cell
predictions.
====================================================================

Strengthens the statistical reporting in Section 4 of the NMI
manuscript without requiring new compute. For each pair of cells
(b vs b+, b vs e+, e vs e+, b+ vs e+, b vs c, b vs c''), computes:
  - Cross-seed mean and standard deviation of test macro-AUC
  - 1000-resample bootstrap 95% CI on the cross-seed mean delta
  - Permutation test p-value: probability that random sign-flips of
    the per-seed deltas produce a mean delta >= observed (one-sided)

Both tests are non-parametric and pure CPU.

Pre-conditions:
  - test_predictions.npz files for each cell × seed under
    D:/kvasir_capsule/outputs/temporal_cell_{b,b_pi,c,c_pn,c_topology,d,e,e_pi}/seed{seed}/
  - sklearn for ROC AUC

Output:
  paper/nature-machine-intelligence/docs/bootstrap_permutation_tests_2026-05-08.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs")
SEEDS = [41, 42, 43, 44, 45, 47]
N_BOOTSTRAP = 1000
N_PERMUTATION = 10000
RANDOM_SEED = 12345

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]

CELL_DIRS = {
    "(b)":  "temporal_cell_b",
    "(c)":  "temporal_cell_c",
    "(c')": "temporal_cell_c_pn",
    "(c\")": "temporal_cell_c_topology",
    "(d)":  "temporal_cell_d",
    "(e)":  "temporal_cell_e",
    "(b+)": "temporal_cell_b_pi",
    "(e+)": "temporal_cell_e_pi",
}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def cell_macro_aucs_per_seed(cell_name: str) -> Dict[int, float]:
    from sklearn.metrics import roc_auc_score
    out: Dict[int, float] = {}
    for seed in SEEDS:
        path = OUTPUT_ROOT / CELL_DIRS[cell_name] / f"seed{seed}" / "test_predictions.npz"
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        d = np.load(path)
        logits = d["logits"]; labels = d["labels"]
        d.close()
        probs = softmax(logits)
        aucs = []
        for j in range(len(CLASS_NAMES)):
            y = (labels == j).astype(np.int32)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            aucs.append(roc_auc_score(y, probs[:, j]))
        out[seed] = float(np.mean(aucs))
    return out


def bootstrap_ci(deltas: np.ndarray, n_resamples: int = N_BOOTSTRAP,
                  alpha: float = 0.05) -> Tuple[float, float, float]:
    """1-alpha CI of mean(deltas) by resampling deltas with replacement."""
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(deltas)
    means = np.zeros(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        means[i] = deltas[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(deltas.mean()), lo, hi


def permutation_test(deltas: np.ndarray,
                       n_permutations: int = N_PERMUTATION
                       ) -> float:
    """One-sided permutation test for mean(deltas) > 0.
    Sign-flip permutation: under null hypothesis (no systematic
    direction), each delta is equally likely to be positive or
    negative. Probability that a random sign assignment produces
    mean >= observed."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    n = len(deltas)
    obs_mean = abs(deltas.mean())
    count_extreme = 0
    abs_deltas = np.abs(deltas)
    for _ in range(n_permutations):
        signs = rng.choice([-1.0, 1.0], size=n)
        permuted = signs * abs_deltas
        if abs(permuted.mean()) >= obs_mean:
            count_extreme += 1
    return float(count_extreme / n_permutations)


def compare_cells(name_a: str, name_b: str) -> Dict:
    aucs_a = cell_macro_aucs_per_seed(name_a)
    aucs_b = cell_macro_aucs_per_seed(name_b)
    seeds = sorted(set(aucs_a.keys()) & set(aucs_b.keys()))
    if not seeds:
        return {"name_a": name_a, "name_b": name_b, "n_seeds": 0}
    deltas = np.array([aucs_b[s] - aucs_a[s] for s in seeds])
    mean_delta, ci_lo, ci_hi = bootstrap_ci(deltas)
    p_perm = permutation_test(deltas)
    return {
        "name_a": name_a,
        "name_b": name_b,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "auc_a": [aucs_a[s] for s in seeds],
        "auc_b": [aucs_b[s] for s in seeds],
        "deltas": [float(d) for d in deltas],
        "mean_delta": mean_delta,
        "ci_95_lo": ci_lo,
        "ci_95_hi": ci_hi,
        "permutation_p": p_perm,
    }


def main():
    pairs = [
        ("(b)",  "(b+)"),
        ("(b)",  "(e+)"),
        ("(e)",  "(e+)"),
        ("(b+)", "(e+)"),
        ("(b)",  "(c)"),
        ("(b)",  "(c\")"),
        ("(b)",  "(c')"),
    ]

    md = []
    md.append("# Bootstrap CIs and permutation tests on capsule eight-cell predictions\n")
    md.append("**Date:** 2026-05-08")
    md.append(f"**Bootstrap:** {N_BOOTSTRAP} resamples")
    md.append(f"**Permutation:** {N_PERMUTATION} sign-flip permutations, two-sided")
    md.append(f"**Seeds:** {SEEDS}")
    md.append("")
    md.append("## Per-pair results\n")
    md.append("| Comparison | n | mean Δ | 95% CI | permutation p |")
    md.append("|---|---:|---:|---:|---:|")

    print(f"[main] running pairwise tests over {len(pairs)} pairs")
    for a, b in pairs:
        print(f"\n[main] {a} vs {b}")
        res = compare_cells(a, b)
        if res["n_seeds"] == 0:
            md.append(f"| {a} vs {b} | -- | n/a | n/a | n/a |")
            print(f"  (no seeds)")
            continue
        sig_marker = ""
        if res["permutation_p"] < 0.001:
            sig_marker = "***"
        elif res["permutation_p"] < 0.01:
            sig_marker = "**"
        elif res["permutation_p"] < 0.05:
            sig_marker = "*"
        md.append(f"| {a} vs {b} | {res['n_seeds']} | {res['mean_delta']:+.4f} "
                  f"| [{res['ci_95_lo']:+.4f}, {res['ci_95_hi']:+.4f}] "
                  f"| {res['permutation_p']:.4f}{sig_marker} |")
        print(f"  mean Δ = {res['mean_delta']:+.4f}  "
              f"95% CI = [{res['ci_95_lo']:+.4f}, {res['ci_95_hi']:+.4f}]  "
              f"p_perm = {res['permutation_p']:.4f}")
        md.append("")

    md.append("\nSignificance markers: \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (two-sided permutation)")
    md.append("")
    md.append("## Per-seed deltas\n")
    md.append("| Comparison | seed 41 | seed 42 | seed 43 | seed 44 | seed 45 | seed 47 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for a, b in pairs:
        res = compare_cells(a, b)
        if res["n_seeds"] == 0:
            continue
        deltas_per_seed = dict(zip(res["seeds"], res["deltas"]))
        row = f"| {a} vs {b} |"
        for s in SEEDS:
            if s in deltas_per_seed:
                row += f" {deltas_per_seed[s]:+.4f} |"
            else:
                row += " -- |"
        md.append(row)
    md.append("")
    md.append("## Headline interpretation\n")
    md.append("The four critical comparisons that distinguish the parameterization-")
    md.append("mechanism boundary from the null are (b) vs (b+), (b) vs (e+),")
    md.append("(b+) vs (e+), and (b) vs (c). The first three should show")
    md.append("statistically-significant positive deltas (the boundary's positive")
    md.append("predictions); the last should show a delta indistinguishable from")
    md.append("zero (the boundary's null prediction for summary-stat C1).")

    out = Path("D:/onedrive/UT_southwestern/GIproject/GI Project_2026/"
               "GI_Multi_Task/paper/nature-machine-intelligence/docs/"
               "bootstrap_permutation_tests_2026-05-08.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[main] report -> {out}")


if __name__ == "__main__":
    main()
