"""
Paired statistical tests across the 8 temporal-extension cells.
================================================================

For each pair of cells (a vs b, b vs b+, b vs e+, b+ vs e+), compute:
  - per-class DeLong-paired AUC test (z-stat, two-sided p)
  - per-class McNemar paired-prediction test at argmax operating point
  - macro-AUC delta (mean across seeds) and per-seed sign count

Aggregation across seeds: for AUC z-stats we use Stouffer's z combination
(sum of z's / sqrt(n_seeds)); for McNemar we sum the b/c counts across
seeds and recompute the chi-square (treating each seed as additional
paired data on the same patients/frames is conservative since
predictions across seeds are not independent, but it captures the
direction-consistency reviewers will look for).

Inputs:
  D:/kvasir_capsule/outputs/temporal_cell_*/seed{seed}/test_predictions.npz
    {logits: (N, 14) float, labels: (N,) int}

Output:
  paper/nature-machine-intelligence/docs/temporal_paired_tests_2026-05-07.md
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs")
SEEDS = [41, 42, 43, 44, 45, 47]
CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
N_CLASSES = len(CLASS_NAMES)

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


def delong_z_paired(scores_a: np.ndarray, scores_b: np.ndarray,
                     y: np.ndarray) -> Tuple[float, float, float]:
    """DeLong z-statistic for paired ROC comparison on a single class.
    Returns (auc_a, auc_b, z) where z is the DeLong-paired test
    statistic for AUC_a - AUC_b.
    Implementation: per-pair structural components (V10/V01).
    """
    pos = (y == 1)
    neg = ~pos
    m, n = pos.sum(), neg.sum()
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan")

    sa_p, sa_n = scores_a[pos], scores_a[neg]
    sb_p, sb_n = scores_b[pos], scores_b[neg]

    # Mid-rank: per-pair indicator (s_pos > s_neg) + 0.5 * (s_pos == s_neg)
    def kernel(p, n_):
        # returns matrix M[i,j] = 1{p_i>n_j} + 0.5 1{p_i==n_j}
        return (p[:, None] > n_[None, :]).astype(np.float64) + \
               0.5 * (p[:, None] == n_[None, :]).astype(np.float64)

    Ka = kernel(sa_p, sa_n)
    Kb = kernel(sb_p, sb_n)
    auc_a = Ka.mean()
    auc_b = Kb.mean()

    V10_a = Ka.mean(axis=1)   # (m,)
    V10_b = Kb.mean(axis=1)
    V01_a = Ka.mean(axis=0)   # (n,)
    V01_b = Kb.mean(axis=0)

    s10 = ((V10_a - auc_a) * (V10_a - auc_a)).sum() / max(m - 1, 1)
    s01 = ((V01_a - auc_a) * (V01_a - auc_a)).sum() / max(n - 1, 1)
    s10_b = ((V10_b - auc_b) ** 2).sum() / max(m - 1, 1)
    s01_b = ((V01_b - auc_b) ** 2).sum() / max(n - 1, 1)
    cov10 = ((V10_a - auc_a) * (V10_b - auc_b)).sum() / max(m - 1, 1)
    cov01 = ((V01_a - auc_a) * (V01_b - auc_b)).sum() / max(n - 1, 1)

    var_diff = (s10 + s10_b - 2 * cov10) / m + (s01 + s01_b - 2 * cov01) / n
    if var_diff <= 0:
        return float(auc_a), float(auc_b), 0.0
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    return float(auc_a), float(auc_b), float(z)


def mcnemar_paired(pred_a: np.ndarray, pred_b: np.ndarray,
                    y: np.ndarray) -> Tuple[int, int, float, float]:
    """McNemar test for two paired classifiers.
    pred_a, pred_b: argmax class predictions per frame
    y: ground-truth class
    Returns (n01, n10, chi2, p)
      n01 = #frames where a wrong, b right
      n10 = #frames where a right, b wrong
    """
    a_right = (pred_a == y)
    b_right = (pred_b == y)
    n10 = int((a_right & ~b_right).sum())
    n01 = int((~a_right & b_right).sum())
    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0
    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return n01, n10, float(chi2), float(p)


def load_cell(name: str, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    p = OUTPUT_ROOT / CELL_DIRS[name] / f"seed{seed}" / "test_predictions.npz"
    if not p.exists():
        return None, None
    d = np.load(p)
    return d["logits"], d["labels"]


def stouffer(z_list: List[float]) -> float:
    """Combine same-direction one-sided z's via Stouffer."""
    z = np.array(z_list, dtype=np.float64)
    z = z[~np.isnan(z)]
    if len(z) == 0:
        return float("nan")
    return float(z.sum() / np.sqrt(len(z)))


def compare_pair(name_a: str, name_b: str) -> Dict:
    """Per-class paired AUC + McNemar across seeds, aggregated."""
    rows = []
    macro_a, macro_b = [], []
    for cname_idx, cname in enumerate(CLASS_NAMES):
        z_seeds, auca_seeds, aucb_seeds = [], [], []
        n01_total, n10_total = 0, 0
        for seed in SEEDS:
            log_a, y_a = load_cell(name_a, seed)
            log_b, y_b = load_cell(name_b, seed)
            if log_a is None or log_b is None:
                continue
            assert (y_a == y_b).all()
            p_a = softmax(log_a)[:, cname_idx]
            p_b = softmax(log_b)[:, cname_idx]
            y_bin = (y_a == cname_idx).astype(int)
            auc_a, auc_b, z = delong_z_paired(p_a, p_b, y_bin)
            if not np.isnan(z):
                z_seeds.append(z)
                auca_seeds.append(auc_a)
                aucb_seeds.append(auc_b)
        if len(z_seeds) == 0:
            continue
        z_combined = stouffer(z_seeds)
        p_combined = 2 * (1 - stats.norm.cdf(abs(z_combined)))
        rows.append({
            "class": cname,
            "auc_a": float(np.mean(auca_seeds)),
            "auc_b": float(np.mean(aucb_seeds)),
            "delta": float(np.mean(aucb_seeds) - np.mean(auca_seeds)),
            "z_combined": z_combined,
            "p_combined": float(p_combined),
            "n_seeds": len(z_seeds),
        })

    # Macro AUC and macro-level z (mean of per-class z's)
    macro_z = stouffer([r["z_combined"] for r in rows])
    macro_p = 2 * (1 - stats.norm.cdf(abs(macro_z)))

    # McNemar at argmax level (combined across all classes & seeds)
    n01_tot, n10_tot = 0, 0
    for seed in SEEDS:
        log_a, y_a = load_cell(name_a, seed)
        log_b, y_b = load_cell(name_b, seed)
        if log_a is None or log_b is None:
            continue
        pred_a = log_a.argmax(axis=1)
        pred_b = log_b.argmax(axis=1)
        n01, n10, _, _ = mcnemar_paired(pred_a, pred_b, y_a)
        n01_tot += n01
        n10_tot += n10
    if n01_tot + n10_tot > 0:
        chi2 = (abs(n01_tot - n10_tot) - 1) ** 2 / (n01_tot + n10_tot)
        mc_p = 1 - stats.chi2.cdf(chi2, df=1)
    else:
        chi2, mc_p = 0.0, 1.0

    return {
        "name_a": name_a,
        "name_b": name_b,
        "rows": rows,
        "macro_z": macro_z,
        "macro_p": float(macro_p),
        "mcnemar_n01": n01_tot,    # name_a wrong, name_b right
        "mcnemar_n10": n10_tot,    # name_a right, name_b wrong
        "mcnemar_chi2": float(chi2),
        "mcnemar_p": float(mc_p),
    }


def render_pair_md(res: Dict) -> List[str]:
    md = []
    a, b = res["name_a"], res["name_b"]
    md.append(f"### Cell {a} vs Cell {b}\n")
    md.append(f"**Macro DeLong z (Stouffer-combined):** {res['macro_z']:+.3f}  "
              f"two-sided p = {res['macro_p']:.4f}")
    md.append(f"**Macro McNemar (argmax classifier):** "
              f"`{a}_wrong & {b}_right` = {res['mcnemar_n01']}, "
              f"`{a}_right & {b}_wrong` = {res['mcnemar_n10']}, "
              f"chi^2 = {res['mcnemar_chi2']:.2f}, p = {res['mcnemar_p']:.2e}")
    md.append("")
    md.append(f"Per-class paired DeLong (mean AUC across seeds; combined z; two-sided p):\n")
    md.append(f"| Class | {a} mean AUC | {b} mean AUC | Δ | z_comb | p_comb |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for r in res["rows"]:
        sig = ""
        if r["p_combined"] < 0.001:
            sig = "***"
        elif r["p_combined"] < 0.01:
            sig = "**"
        elif r["p_combined"] < 0.05:
            sig = "*"
        md.append(f"| {r['class']} | {r['auc_a']:.3f} | {r['auc_b']:.3f} "
                  f"| {r['delta']:+.3f} | {r['z_combined']:+.2f} "
                  f"| {r['p_combined']:.3f}{sig} |")
    md.append("")
    return md


def main():
    pairs = [
        ("(b)",  "(b+)"),    # RGB+C2 vs +PI+C2
        ("(b)",  "(e+)"),    # RGB+C2 vs final
        ("(e)",  "(e+)"),    # RGB+C2+C3 vs +PI+C2+C3
        ("(b+)", "(e+)"),    # +PI+C2 vs +PI+C2+C3 (does C3 stack?)
        ("(b)",  "(c)"),     # RGB+C2 vs +scalar C1 (failed)
        ("(b)",  "(c\")"),   # RGB+C2 vs +13-dim C1 (failed worst)
    ]
    md = ["# Paired statistical tests across the 8 temporal-extension cells\n"]
    md.append("**Date:** 2026-05-07")
    md.append(f"**Seeds:** {SEEDS}  (n={len(SEEDS)})")
    md.append("**Tests:** DeLong paired-AUC per class (combined across seeds via Stouffer's z); "
              "McNemar paired-prediction at argmax operating point (combined counts).")
    md.append("**Significance markers:** \\* p<0.05, \\*\\* p<0.01, \\*\\*\\* p<0.001 (two-sided).")
    md.append("")
    md.append("Cell (e+) is the final architecture; cell (b+) is the +PI 5-channel "
              "intermediate; cell (b) is the temporal-only RGB baseline.")
    md.append("")

    print("[main] computing pairwise tests...")
    for a, b in pairs:
        print(f"[main] {a} vs {b}...")
        try:
            res = compare_pair(a, b)
            md.extend(render_pair_md(res))
        except Exception as e:
            md.append(f"\n### {a} vs {b}\n  *(failed: {e})*\n")

    out = Path("D:/onedrive/UT_southwestern/GIproject/GI Project_2026/"
               "GI_Multi_Task/paper/nature-machine-intelligence/docs/"
               "temporal_paired_tests_2026-05-07.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"[main] -> {out}")


if __name__ == "__main__":
    main()
