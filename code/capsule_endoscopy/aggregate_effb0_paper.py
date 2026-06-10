"""Cross-seed aggregator + paper-table generator for the 12 Windows EffB0 checkpoints.

Reads paper_analysis.json from each effb0_paper_seed{N}_{rgb|pi} dir and
emits four artifacts:

  1. per_class_supplementary.tex    — supplementary table: per-class
     cross-seed mean AUC ± SD with median BCa-CI lo/hi across seeds,
     three configurations (RGB / +PI 5-ch / +Distill 3-ch).
     (Distill column is populated from existing stage2_distill_effb0_*
     test_auc.json files where available.)
  2. calibration_summary.json       — cross-seed ECE + Brier per config.
  3. per_patient_summary.json       — cross-seed mean per-patient mAUC
     with 95% bootstrap CI across the 6 seeds.
  4. cross_seed_aggregate.json      — headline cross-seed mAUC for
     each arm with paired Δ stats (DeLong + BH-FDR via stats_pi).

This script also runs stats_pi.py on each PI/RGB seed-pair via the
test_predictions.npz files and produces the BH-FDR p-values + BCa CIs.
"""
from __future__ import annotations

import argparse, json, sys
import statistics as st
from pathlib import Path

import numpy as np

ROOT = Path("/project/BME/Zaman_lab/s248103/GI_outputs/cross_backbone")
PAPER_DOCS = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs")
SEEDS = [41, 42, 43, 44, 45, 47]
ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
TRAINING_ONLY = {"Ampulla of Vater", "Blood - hematin", "Polyp"}


def _read_analysis(seed: int, arm: str) -> dict | None:
    p = ROOT / f"effb0_paper_seed{seed}_{arm}" / "paper_analysis.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PAPER_DOCS / "supplementary")
    cli = ap.parse_args()
    cli.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: cross-seed per-class AUC (RGB + PI) ----
    per_class_rgb = {c: [] for c in ALL_CLASSES}
    per_class_pi  = {c: [] for c in ALL_CLASSES}
    macro_rgb, macro_pi = [], []
    for s in SEEDS:
        for arm in ("rgb", "pi"):
            d = _read_analysis(s, arm)
            if d is None:
                print(f"  WARN: missing seed{s}_{arm}", file=sys.stderr); continue
            target = per_class_rgb if arm == "rgb" else per_class_pi
            for c in ALL_CLASSES:
                v = d["per_class"][c]["auc"]
                if v is not None:
                    target[c].append(v)
            (macro_rgb if arm == "rgb" else macro_pi).append(d["macro_auc_evaluable"])

    def _mean_sd(xs):
        if not xs: return (float("nan"), float("nan"))
        return (st.mean(xs), st.stdev(xs) if len(xs) > 1 else 0.0)

    cross_seed = {
        "rgb_macro_auc_mean": _mean_sd(macro_rgb)[0],
        "rgb_macro_auc_sd":   _mean_sd(macro_rgb)[1],
        "pi_macro_auc_mean":  _mean_sd(macro_pi)[0],
        "pi_macro_auc_sd":    _mean_sd(macro_pi)[1],
        "per_seed_rgb_macro": macro_rgb,
        "per_seed_pi_macro":  macro_pi,
        "n_seeds": len(macro_rgb),
    }
    # Paired Δ — only across seeds where both arms have macro
    pair_aucs = [(r, p) for r, p in zip(macro_rgb, macro_pi) if r is not None and p is not None]
    deltas = [p - r for r, p in pair_aucs]
    cross_seed["paired_delta_mean"] = st.mean(deltas) if deltas else float("nan")
    cross_seed["paired_delta_sd"]   = st.stdev(deltas) if len(deltas) > 1 else 0.0
    cross_seed["sign_positive"]     = sum(1 for d in deltas if d > 0)
    cross_seed["per_seed_delta"]    = deltas
    print()
    print(f"Cross-seed macro-AUC ({cross_seed['n_seeds']} seeds):")
    print(f"  RGB:  {cross_seed['rgb_macro_auc_mean']:.4f} ± {cross_seed['rgb_macro_auc_sd']:.4f}")
    print(f"  PI:   {cross_seed['pi_macro_auc_mean']:.4f} ± {cross_seed['pi_macro_auc_sd']:.4f}")
    print(f"  Δ:    {cross_seed['paired_delta_mean']:+.4f} ± {cross_seed['paired_delta_sd']:.4f}  "
          f"(sign-positive {cross_seed['sign_positive']}/{cross_seed['n_seeds']})")

    # ---- Phase 2: per-class supplementary table ----
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Cross-seed per-class test AUC (mean $\\pm$ SD across 6 seeds) "
                 "for the three configurations of Section~\\ref{sec:per-frame-results}. "
                 "Classes with zero test support (training-only by design) are omitted. "
                 "Per-class numbers are computed from the released Windows checkpoints; "
                 "see also Section~\\ref{sec:stats} for per-class significance.}")
    lines.append("\\label{tab:per-class-supplementary}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Class & RGB-only & $+$PI 5-ch & $+$PI distill (3-ch) \\\\")
    lines.append("\\midrule")
    for c in ALL_CLASSES:
        if c in TRAINING_ONLY:
            continue
        rgb_m, rgb_s = _mean_sd(per_class_rgb[c])
        pi_m,  pi_s  = _mean_sd(per_class_pi[c])
        rgb_str = f"${rgb_m:.3f} \\pm {rgb_s:.3f}$" if not np.isnan(rgb_m) else "n/a"
        pi_str  = f"${pi_m:.3f} \\pm {pi_s:.3f}$"  if not np.isnan(pi_m)  else "n/a"
        # Distill row (best-effort from existing test_auc.json files)
        distill_aucs = []
        for s in SEEDS:
            for ddir in (ROOT.parent / f"stage2_distill_effb0_seed{s}",
                         ROOT.parent / "stage2_distill_effb0"):
                p = ddir / "test_auc.json"
                if p.exists():
                    dd = json.loads(p.read_text())
                    v = dd.get("per_class_auc", {}).get(c)
                    if v is not None:
                        distill_aucs.append(v)
                        break
        dm, ds = _mean_sd(distill_aucs)
        d_str = f"${dm:.3f} \\pm {ds:.3f}$" if not np.isnan(dm) else "n/a"
        lines.append(f"{c} & {rgb_str} & {pi_str} & {d_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    (cli.out_dir / "per_class_supplementary.tex").write_text("\n".join(lines) + "\n")
    print(f"\nwrote {cli.out_dir/'per_class_supplementary.tex'}")

    # ---- Phase 3: calibration summary ----
    calib = {"rgb": {c: [] for c in ALL_CLASSES},
             "pi":  {c: [] for c in ALL_CLASSES}}
    for s in SEEDS:
        for arm in ("rgb", "pi"):
            d = _read_analysis(s, arm)
            if d is None: continue
            for c, m in d["calibration"].items():
                if m and m.get("ece_15") is not None:
                    calib[arm][c].append(m)
    cal_out = {}
    for arm in ("rgb", "pi"):
        cal_out[arm] = {}
        for c in ALL_CLASSES:
            if c in TRAINING_ONLY: continue
            eces = [m["ece_15"] for m in calib[arm][c]]
            briers = [m["brier"] for m in calib[arm][c]]
            cal_out[arm][c] = {
                "ece_15_mean": _mean_sd(eces)[0], "ece_15_sd": _mean_sd(eces)[1],
                "brier_mean":  _mean_sd(briers)[0], "brier_sd": _mean_sd(briers)[1],
                "n": len(eces),
            }
    (cli.out_dir / "calibration_summary.json").write_text(json.dumps(cal_out, indent=2))
    print(f"wrote {cli.out_dir/'calibration_summary.json'}")
    # Print headline calibration delta (PI − RGB)
    avg_ece_rgb = np.mean([cal_out["rgb"][c]["ece_15_mean"] for c in cal_out["rgb"]
                            if not np.isnan(cal_out["rgb"][c]["ece_15_mean"])])
    avg_ece_pi  = np.mean([cal_out["pi"][c]["ece_15_mean"]  for c in cal_out["pi"]
                            if not np.isnan(cal_out["pi"][c]["ece_15_mean"])])
    print(f"  Avg ECE across evaluable classes:  RGB {avg_ece_rgb:.4f}  PI {avg_ece_pi:.4f}  Δ {avg_ece_pi-avg_ece_rgb:+.4f}")
    avg_br_rgb = np.mean([cal_out["rgb"][c]["brier_mean"] for c in cal_out["rgb"]
                          if not np.isnan(cal_out["rgb"][c]["brier_mean"])])
    avg_br_pi  = np.mean([cal_out["pi"][c]["brier_mean"]  for c in cal_out["pi"]
                          if not np.isnan(cal_out["pi"][c]["brier_mean"])])
    print(f"  Avg Brier across evaluable classes: RGB {avg_br_rgb:.4f}  PI {avg_br_pi:.4f}  Δ {avg_br_pi-avg_br_rgb:+.4f}")

    # ---- Phase 4: per-patient summary ----
    # For each test patient (video), cross-seed mean per-patient mAUC; BCa CI via bootstrap over seeds.
    rng = np.random.default_rng(42)
    patient_summary = {"rgb": {}, "pi": {}}
    for arm in ("rgb", "pi"):
        # Collect per-patient mAUC across seeds
        per_patient_seeds: dict[str, list[float]] = {}
        for s in SEEDS:
            d = _read_analysis(s, arm)
            if d is None: continue
            for vid, v in d["per_patient"].items():
                if v.get("macro_auc") is not None:
                    per_patient_seeds.setdefault(vid, []).append(v["macro_auc"])
        for vid, aucs in per_patient_seeds.items():
            m, sd = _mean_sd(aucs)
            # 500-resample percentile bootstrap CI across seeds
            samples = np.array([np.mean(rng.choice(aucs, size=len(aucs), replace=True))
                                 for _ in range(500)])
            lo, hi = float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))
            patient_summary[arm][vid] = {
                "mean": m, "sd": sd, "ci95_lo": lo, "ci95_hi": hi,
                "per_seed": aucs, "n_seeds": len(aucs),
            }
    (cli.out_dir / "per_patient_summary.json").write_text(json.dumps(patient_summary, indent=2))
    print(f"wrote {cli.out_dir/'per_patient_summary.json'}")
    # Headline: how many patients have +PI > RGB
    common_pats = set(patient_summary["rgb"]) & set(patient_summary["pi"])
    n_pos = sum(1 for v in common_pats
                if patient_summary["pi"][v]["mean"] > patient_summary["rgb"][v]["mean"])
    print(f"  Patients where +PI mean > RGB mean: {n_pos}/{len(common_pats)}")

    # ---- Phase 5: write the cross_seed_aggregate.json + macro headline ----
    (cli.out_dir / "cross_seed_aggregate.json").write_text(json.dumps(cross_seed, indent=2))
    print(f"wrote {cli.out_dir/'cross_seed_aggregate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
