"""
Per-patient disaggregation: is the cell (b+)/(e+) lift driven by
a few patients, or is it consistent across patients?
======================================================================

For each test patient on Kvasir-Capsule, computes per-patient
macro-AUC for cells (b), (b+), (e), and (e+). Reports:
  - Per-patient lift distributions (b+ over b, e+ over e, e+ over b)
  - Number of patients where each lift is positive vs negative
  - Worst-case patient (where the lift is most negative)

This addresses the natural reviewer concern: "is your cross-seed
mean macro-AUC lift coming from genuine improvement across patients,
or from a few patients where the model gets very lucky?" Per-patient
disaggregation answers this directly.

Pre-conditions:
  - test_predictions.npz files for cells (b), (b+), (e), (e+) at
    D:/kvasir_capsule/outputs/temporal_cell_*/seed42/test_predictions.npz
  - video_index.json with patient (video_id) annotations

Output:
  paper/nature-machine-intelligence/docs/per_patient_lift_report.md
  paper/nature-machine-intelligence/manuscript/figures/fig7_per_patient_lift.pdf
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = Path("D:/kvasir_capsule/outputs")
INDEX_JSON = HERE / "video_index.json"
REPORT_DIR = HERE.parent.parent / "docs"
FIG_DIR = HERE.parent.parent / "manuscript" / "figures"
SEEDS = [41, 42, 43, 44, 45, 47]

CELL_DIRS = {
    "(b)":  "temporal_cell_b",
    "(e)":  "temporal_cell_e",
    "(b+)": "temporal_cell_b_pi",
    "(e+)": "temporal_cell_e_pi",
}

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]


def load_cell_predictions_with_patients(cell_name: str, seed: int
                                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load logits + labels for a cell at the given seed and align
    each frame to a patient via the video_index.

    Returns (logits, labels, patient_indices, patient_ids)
    """
    path = OUTPUT_ROOT / CELL_DIRS[cell_name] / f"seed{seed}" / "test_predictions.npz"
    if not path.exists():
        raise SystemExit(f"missing: {path}")

    d = np.load(path)
    logits = np.array(d["logits"])
    labels = np.array(d["labels"])
    d.close()

    # Use the video_index to recover the per-frame ordering. The cells
    # all use the same CachedSequenceDataset with deterministic
    # ordering, so the i-th prediction corresponds to the i-th frame
    # in the test split as enumerated by the index. We need to walk
    # the index in the same order to recover patient ids.
    video_index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
    patient_ids: List[str] = []
    for video_id, frames in video_index["by_video"].items():
        in_test = sorted([f for f in frames
                            if f.get("split") == "test"],
                           key=lambda f: f["frame_number"])
        if not in_test:
            continue
        for _ in in_test:
            patient_ids.append(video_id)

    # Truncate to match logits length (in case of any indexing offsets)
    n = min(len(logits), len(patient_ids))
    if n < len(logits):
        print(f"[load {cell_name}] truncated logits {len(logits)} -> {n}")
    if n < len(patient_ids):
        print(f"[load {cell_name}] truncated patient_ids {len(patient_ids)} -> {n}")
    logits = logits[:n]
    labels = labels[:n]
    patient_ids = patient_ids[:n]

    unique_patients = sorted(set(patient_ids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_patients)}
    patient_indices = np.array([pid_to_idx[pid] for pid in patient_ids],
                                  dtype=np.int64)

    return logits, labels, patient_indices, unique_patients


def per_patient_macro_auc(logits: np.ndarray, labels: np.ndarray,
                              patient_indices: np.ndarray,
                              unique_patients: List[str]) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    out: Dict[str, float] = {}
    for pi, pid in enumerate(unique_patients):
        mask = patient_indices == pi
        if mask.sum() < 5:
            continue
        p_logits = probs[mask]
        p_labels = labels[mask]
        aucs = []
        for j in range(len(CLASS_NAMES)):
            y = (p_labels == j).astype(np.int32)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            aucs.append(roc_auc_score(y, p_logits[:, j]))
        if aucs:
            out[pid] = float(np.mean(aucs))
    return out


def main():
    print(f"[per-patient] aggregating across seeds {SEEDS}")
    # For each cell, accumulate per-patient AUCs across all seeds
    # then average. This gives a more reliable per-patient lift
    # estimate than any single seed alone.
    cell_aucs_per_seed: Dict[str, Dict[int, Dict[str, float]]] = {
        c: {} for c in CELL_DIRS
    }
    for cell_name in CELL_DIRS:
        for seed in SEEDS:
            try:
                logits, labels, pi, pids = load_cell_predictions_with_patients(
                    cell_name, seed)
                aucs = per_patient_macro_auc(logits, labels, pi, pids)
                cell_aucs_per_seed[cell_name][seed] = aucs
                print(f"[load {cell_name} seed={seed}] {len(aucs)} "
                      f"patients with valid AUC")
            except SystemExit as exc:
                print(f"[skip {cell_name} seed={seed}] {exc}")

    # Average per-patient AUCs across seeds for each cell
    cell_aucs: Dict[str, Dict[str, float]] = {}
    common_patients = None
    for cell_name, per_seed_dict in cell_aucs_per_seed.items():
        if not per_seed_dict:
            continue
        seeds_with_data = list(per_seed_dict.keys())
        all_pids = set()
        for d in per_seed_dict.values():
            all_pids.update(d.keys())
        avg = {}
        for pid in all_pids:
            vals = [per_seed_dict[s][pid] for s in seeds_with_data
                    if pid in per_seed_dict[s]]
            if len(vals) >= 3:    # require >= 3 seeds for stability
                avg[pid] = float(np.mean(vals))
        cell_aucs[cell_name] = avg
        if common_patients is None:
            common_patients = set(avg.keys())
        else:
            common_patients &= set(avg.keys())

    common_patients = sorted(common_patients)
    print(f"[per-patient] {len(common_patients)} patients with "
          f"per-patient AUC averaged across >=3 seeds and present in all 4 cells")

    # Compute per-patient lifts
    lift_b_to_bplus = {pid: cell_aucs["(b+)"][pid] - cell_aucs["(b)"][pid]
                        for pid in common_patients}
    lift_e_to_eplus = {pid: cell_aucs["(e+)"][pid] - cell_aucs["(e)"][pid]
                        for pid in common_patients}
    lift_b_to_eplus = {pid: cell_aucs["(e+)"][pid] - cell_aucs["(b)"][pid]
                        for pid in common_patients}

    # Summary stats per lift
    def stats(lift_dict):
        vals = np.array(list(lift_dict.values()))
        n_pos = int((vals > 0).sum())
        n_neg = int((vals < 0).sum())
        return vals, n_pos, n_neg

    md = []
    md.append("# Per-patient lift disaggregation\n")
    md.append("**Date:** 2026-05-08")
    md.append(f"**Seeds aggregated:** {SEEDS} (per-patient AUCs averaged)")
    md.append(f"**Patients in test split:** {len(common_patients)}")
    md.append("")
    md.append("Addresses the reviewer concern: ``is the cross-seed mean "
              "macro-AUC lift driven by a few outlier patients, or is "
              "the boundary's lift consistent across patients?''")
    md.append("")
    md.append("## Per-cell test macro-AUC, aggregated over patients\n")
    md.append("| Cell | Mean per-patient AUC | $\\sigma$ | min | max |")
    md.append("|---|---:|---:|---:|---:|")
    for cell_name in ["(b)", "(e)", "(b+)", "(e+)"]:
        vals = np.array([cell_aucs[cell_name][pid] for pid in common_patients])
        md.append(f"| {cell_name} | {vals.mean():.4f} | {vals.std():.4f} "
                  f"| {vals.min():.4f} | {vals.max():.4f} |")
    md.append("")
    md.append("## Per-patient lift distributions\n")
    md.append("| Comparison | Mean lift | $\\sigma$ | n positive | n negative | min | max |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for label, lift_dict in [
        ("(b+) vs (b)", lift_b_to_bplus),
        ("(e+) vs (e)", lift_e_to_eplus),
        ("(e+) vs (b)", lift_b_to_eplus),
    ]:
        vals, np_, nn_ = stats(lift_dict)
        md.append(f"| {label} | {vals.mean():+.4f} | {vals.std():.4f} "
                  f"| {np_}/{len(common_patients)} | {nn_}/{len(common_patients)} "
                  f"| {vals.min():+.4f} | {vals.max():+.4f} |")
    md.append("")
    md.append("## Per-patient table (sorted by (b+) vs (b) lift)\n")
    md.append("| Patient ID | (b) | (b+) | (e) | (e+) | $\\Delta$(b+)-(b) | $\\Delta$(e+)-(e) |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    sorted_pids = sorted(common_patients,
                            key=lambda p: -lift_b_to_bplus[p])
    for pid in sorted_pids:
        md.append(f"| {pid} | {cell_aucs['(b)'][pid]:.3f} | {cell_aucs['(b+)'][pid]:.3f} "
                  f"| {cell_aucs['(e)'][pid]:.3f} | {cell_aucs['(e+)'][pid]:.3f} "
                  f"| {lift_b_to_bplus[pid]:+.3f} | {lift_e_to_eplus[pid]:+.3f} |")
    md.append("")
    n_pos_bp = sum(1 for v in lift_b_to_bplus.values() if v > 0)
    n_pos_ep = sum(1 for v in lift_e_to_eplus.values() if v > 0)
    n_pos_bep = sum(1 for v in lift_b_to_eplus.values() if v > 0)
    md.append("## Verdict\n")
    md.append(f"- **(b+) vs (b)**: lift is positive on "
              f"{n_pos_bp} of {len(common_patients)} test patients.")
    md.append(f"- **(e+) vs (e)**: lift is positive on "
              f"{n_pos_ep} of {len(common_patients)} test patients.")
    md.append(f"- **(e+) vs (b)**: lift is positive on "
              f"{n_pos_bep} of {len(common_patients)} test patients.")
    md.append("")
    if n_pos_bep / max(1, len(common_patients)) >= 0.7:
        md.append(f"The cell (e+) advantage is patient-broad: positive on "
                  f"$\\geq 70\\%$ of test patients. The cross-seed mean "
                  f"lift is not driven by a few outlier patients.")
    else:
        md.append(f"The cell (e+) advantage is patient-narrow: positive on "
                  f"only {n_pos_bep}/{len(common_patients)} patients. The "
                  f"cross-seed mean lift may be driven by a subset.")

    out = REPORT_DIR / "per_patient_lift_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[per-patient] report -> {out}")

    # Generate figure
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
    })
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))

    for ax, (label, lift_dict) in zip(axes, [
        ("(b$^+$) vs (b)", lift_b_to_bplus),
        ("(e$^+$) vs (e)", lift_e_to_eplus),
        ("(e$^+$) vs (b)", lift_b_to_eplus),
    ]):
        vals = sorted(lift_dict.values())
        x = np.arange(len(vals))
        colors = ["#10b981" if v > 0 else "#ef4444" for v in vals]
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5,
                  alpha=0.85)
        ax.axhline(y=0, color="black", linewidth=0.6)
        ax.axhline(y=np.mean(vals), color="grey", linestyle="--",
                      linewidth=0.6, alpha=0.6,
                      label=f"mean = {np.mean(vals):+.3f}")
        ax.set_xlabel("Test patients (sorted)")
        ax.set_ylabel(r"$\Delta$ macro-AUC")
        ax.set_title(label, fontsize=9)
        n_pos = sum(1 for v in vals if v > 0)
        ax.text(0.05, 0.95, f"{n_pos}/{len(vals)} positive",
                  transform=ax.transAxes, fontsize=7, va="top",
                  bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="none", alpha=0.7))
        ax.legend(loc="lower right", frameon=False, fontsize=7)

    fig.suptitle(f"Per-patient lift disaggregation on the Kvasir-Capsule "
                    f"test split ({len(common_patients)} patients, "
                    f"per-patient AUCs averaged across {len(SEEDS)} seeds)",
                    fontsize=9, y=1.02)
    plt.tight_layout()
    out_pdf = FIG_DIR / "fig7_per_patient_lift"
    plt.savefig(f"{out_pdf}.pdf", bbox_inches="tight")
    plt.savefig(f"{out_pdf}.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"[per-patient] figure -> {out_pdf}.pdf")


if __name__ == "__main__":
    main()
