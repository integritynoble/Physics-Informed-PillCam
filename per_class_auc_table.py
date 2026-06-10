"""Per-class AUC table across all ablation arms.

Reads the test_predictions.json files for each arm, computes per-class
one-vs-rest AUC with bootstrap 95% CIs, and emits:
    - per_class_auc.csv          (machine-readable)
    - per_class_auc.md           (paper-ready Markdown table)
    - per_class_auc.json         (full record incl. CIs and macros)

Bleeding (Blood-fresh) and the abnormal-finding classes are flagged in the
Markdown output. Polyp and Blood-hematin have zero positive support in the
test split (single-video classes per the §3.2 split policy) and are listed
as "n/a" with that reason.

Usage (from paper_draft/):
    python per_class_auc_table.py --out_dir D:/kvasir_capsule/outputs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from typing import Dict, List

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", message="Only one class is present in y_true")


# ---- arms in display order ----------------------------------------------------
ARMS = [
    ("RGB-only",                "stage2_rgb_effb0"),
    ("+P_blood only (no H_AFI)","stage2_pi_blood_only_effb0"),
    ("+H_AFI only (no P_blood)","stage2_pi_hafi_only_effb0"),
    ("Physics-only (no RGB)",   "stage2_pi_physics_only_effb0"),
    ("Full PI (RGB+P_blood+H_AFI)","stage2_pi_effb0"),
    ("Distillation",            "stage2_distill_effb0"),
]

# Highlight bleeding + paper headline targets
HEADLINE_CLASSES = {"Blood - fresh", "Blood - hematin", "Polyp",
                    "Angiectasia", "Erosion", "Ulcer", "Erythema",
                    "Lymphangiectasia"}


def _auc_with_bootstrap_ci(y_bin: np.ndarray, scores: np.ndarray,
                            n_boot: int = 1000, seed: int = 42) -> Dict:
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return {"auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "n_pos": int(y_bin.sum()), "n_neg": int(len(y_bin) - y_bin.sum()),
                "note": "no positive support in test split"}
    point = float(roc_auc_score(y_bin, scores))
    rng = np.random.default_rng(seed)
    samples: List[float] = []
    N = len(y_bin)
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        try:
            samples.append(float(roc_auc_score(y_bin[idx], scores[idx])))
        except ValueError:
            continue
    if samples:
        lo = float(np.quantile(samples, 0.025))
        hi = float(np.quantile(samples, 0.975))
    else:
        lo = hi = float("nan")
    return {"auc": point, "lo": lo, "hi": hi,
            "n_pos": int(y_bin.sum()), "n_neg": int(len(y_bin) - y_bin.sum())}


def _load_arm(out_dir: str, run_dir: str) -> Dict:
    p = os.path.join(out_dir, run_dir, "test_predictions.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="D:/kvasir_capsule/outputs")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    ap.add_argument("--csv_out", default=None)
    ap.add_argument("--md_out", default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    csv_out = args.csv_out or os.path.join(args.out_dir, "per_class_auc.csv")
    md_out = args.md_out or os.path.join(args.out_dir, "per_class_auc.md")
    json_out = args.json_out or os.path.join(args.out_dir, "per_class_auc.json")

    # Load all arms; require alignment on labels and class_names so AUCs
    # are paired across the same test frames.
    loaded = []
    ref_classes = ref_labels = None
    for label, dirname in ARMS:
        rec = _load_arm(args.out_dir, dirname)
        if rec is None:
            print(f"[per_class_auc] WARN: missing test_predictions.json for {dirname}, skipping")
            continue
        if ref_classes is None:
            ref_classes = rec["class_names"]
            ref_labels = rec["labels"]
        else:
            if rec["class_names"] != ref_classes:
                raise SystemExit(f"class_names mismatch in {dirname}")
            if rec["labels"] != ref_labels:
                raise SystemExit(f"labels mismatch in {dirname} (different test split?)")
        loaded.append((label, dirname, rec))

    y_true = np.asarray(ref_labels)
    n_classes = len(ref_classes)

    # Compute per-class AUC + bootstrap CI for each (arm, class).
    table = {cls: {} for cls in ref_classes}
    macros: Dict[str, float] = {}
    for label, dirname, rec in loaded:
        probs = np.asarray(rec["probs"])
        per_cls_aucs = []
        for ci, cls in enumerate(ref_classes):
            y_bin = (y_true == ci).astype(int)
            cell = _auc_with_bootstrap_ci(y_bin, probs[:, ci],
                                           n_boot=args.n_bootstrap)
            table[cls][label] = cell
            if not np.isnan(cell["auc"]):
                per_cls_aucs.append(cell["auc"])
        macros[label] = float(np.mean(per_cls_aucs)) if per_cls_aucs else float("nan")

    # ---- write CSV ----
    arm_labels = [lbl for lbl, _, _ in loaded]
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["class", "n_pos", "n_neg"]
        for lbl in arm_labels:
            header += [f"{lbl} AUC", f"{lbl} CI95_lo", f"{lbl} CI95_hi"]
        w.writerow(header)
        for cls in ref_classes:
            cells = table[cls]
            first = next(iter(cells.values()))
            row = [cls, first["n_pos"], first["n_neg"]]
            for lbl in arm_labels:
                c = cells.get(lbl, {})
                row += [c.get("auc", float("nan")),
                        c.get("lo", float("nan")),
                        c.get("hi", float("nan"))]
            w.writerow(row)
        macro_row = ["MACRO_AUC (evaluable)", "", ""]
        for lbl in arm_labels:
            macro_row += [macros[lbl], "", ""]
        w.writerow(macro_row)
    print(f"[per_class_auc] wrote {csv_out}")

    # ---- write Markdown ----
    def _fmt_cell(c):
        if c is None or np.isnan(c.get("auc", float("nan"))):
            return "n/a"
        return f"{c['auc']:.3f} ({c['lo']:.3f}–{c['hi']:.3f})"
    with open(md_out, "w", encoding="utf-8") as f:
        f.write("# Per-class one-vs-rest AUC (test split, n=6423)\n\n")
        f.write("Cells are AUC (95% bootstrap CI; 1000 resamples).\n")
        f.write("**Bold** rows mark the abnormal-finding classes that drive paper headline numbers.\n")
        f.write("Classes with zero positive support in the test split (Ampulla, Blood-hematin, Polyp) "
                "are training-only per the §3.2 split policy — they cannot have a test-set AUC.\n\n")
        # header
        f.write("| Class | n_pos |")
        for lbl in arm_labels:
            f.write(f" {lbl} |")
        f.write("\n|" + "---|" * (2 + len(arm_labels)) + "\n")
        for cls in ref_classes:
            cells = table[cls]
            first = next(iter(cells.values()))
            n_pos = first["n_pos"]
            row_label = f"**{cls}**" if cls in HEADLINE_CLASSES else cls
            f.write(f"| {row_label} | {n_pos} |")
            for lbl in arm_labels:
                f.write(f" {_fmt_cell(cells.get(lbl))} |")
            f.write("\n")
        # macro row
        f.write(f"| **macro-AUC (evaluable)** | — |")
        for lbl in arm_labels:
            f.write(f" **{macros[lbl]:.3f}** |")
        f.write("\n")
        # delta row vs RGB-only baseline
        if "RGB-only" in arm_labels:
            base = macros["RGB-only"]
            f.write(f"| Δ macro-AUC vs RGB-only | — |")
            for lbl in arm_labels:
                d = macros[lbl] - base
                marker = "↑" if d >= 0 else "↓"
                f.write(f" {d:+.3f} {marker} |")
            f.write("\n")
    print(f"[per_class_auc] wrote {md_out}")

    # ---- write JSON ----
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "n_frames": len(ref_labels),
            "arms": arm_labels,
            "classes": ref_classes,
            "per_class": table,
            "macro_auc_evaluable": macros,
        }, f, indent=2, default=lambda x: None if (isinstance(x, float) and np.isnan(x)) else x)
    print(f"[per_class_auc] wrote {json_out}")

    # Console preview
    print("\n=== headline classes ===")
    for cls in ["Blood - fresh", "Angiectasia", "Erosion", "Ulcer", "Erythema",
                "Lymphangiectasia"]:
        if cls not in table:
            continue
        cells = table[cls]
        print(f"\n{cls}  (n_pos={next(iter(cells.values()))['n_pos']}):")
        for lbl in arm_labels:
            c = cells.get(lbl)
            if c and not np.isnan(c.get("auc", float("nan"))):
                print(f"  {lbl:35s} AUC={c['auc']:.3f}  CI95=[{c['lo']:.3f},{c['hi']:.3f}]")
    print("\n=== macro-AUC (evaluable, 11 classes) ===")
    base = macros.get("RGB-only", float("nan"))
    for lbl in arm_labels:
        d = macros[lbl] - base if not np.isnan(base) else float("nan")
        print(f"  {lbl:35s} macro-AUC={macros[lbl]:.4f}  Δ vs RGB={d:+.4f}")


if __name__ == "__main__":
    main()
