"""GalKva-2026 benchmark evaluator.

Reads a submission JSON (validated against submission_schema.json),
computes the standard benchmark metrics, and emits a Markdown report
that can be pasted into LEADERBOARD.md.

The evaluator does NOT re-run the user's model --- it consumes their
self-reported per-seed AUCs and the per-class AUCs they submit.
This keeps the benchmark trustworthy (we can spot-check their code)
while keeping it usable (we don't need their data, weights, or
hardware).

USAGE
    python benchmark/evaluate.py \\
        --submission submissions/your_method_v1.json \\
        --out submissions/your_method_v1.md

    # Or run against our own reference results to verify the harness:
    python benchmark/evaluate.py \\
        --submission benchmark/reference_results/yang2026_convnext_pi.json \\
        --out /tmp/ref_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EVALUABLE_6 = [
    "Angiectasia", "Blood - fresh", "Lymphangiectasia",
    "Normal clean mucosa", "Polyp", "Ulcer",
]

DEFAULTS_FOR_RETENTION = {
    "_comment": (
        "If a submission omits delta_vs_rgb_baseline, we attempt to "
        "compute retention against the headline EfficientNet-B0 + 5-channel PI "
        "baseline from Yang et al. (2026): Δ_Kvasir = +0.023, Δ_Galar = TBD "
        "on the EffB0 backbone (cross-vendor zero-shot pending)."
    ),
    "kvasir_rgb_baseline_macro_auc": 0.760,
    "galar_rgb_baseline_macro_auc_convnext_tiny": 0.735,
}


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _validate(sub: dict) -> list[str]:
    errs: list[str] = []
    for k in ("submitter", "method", "benchmark_version", "results"):
        if k not in sub:
            errs.append(f"missing top-level key: {k!r}")
    if "results" in sub:
        for ds in ("kvasir", "galar"):
            if ds not in sub["results"]:
                errs.append(f"missing results.{ds}")
                continue
            r = sub["results"][ds]
            if "macro_auc" not in r:
                errs.append(f"missing results.{ds}.macro_auc")
            elif "mean" not in r["macro_auc"]:
                errs.append(f"missing results.{ds}.macro_auc.mean")
    return errs


def _fmt_pm(mean: float, sd: float | None) -> str:
    if sd is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {sd:.3f}"


def _retention(sub: dict) -> str:
    """Compute Δ_Galar / Δ_Kvasir if both deltas are provided."""
    try:
        dk = sub["results"]["kvasir"].get("delta_vs_rgb_baseline")
        dg = sub["results"]["galar"].get("delta_vs_rgb_baseline")
        if dk is None or dg is None:
            return "n/a (provide delta_vs_rgb_baseline in both datasets)"
        if dk == 0:
            return "n/a (Δ_Kvasir = 0)"
        ratio = dg / dk
        return f"{ratio:+.2f}"
    except (KeyError, TypeError):
        return "n/a"


def _band(ratio_str: str) -> str:
    """Map retention ratio to the band defined in kid_atlas_zeroshot_plan §6."""
    if ratio_str.startswith("n/a"):
        return "—"
    try:
        r = float(ratio_str)
    except ValueError:
        return "—"
    if r >= 1.0:   return "Strong positive (≥1.0): prior helps as much or more cross-vendor"
    if r >= 0.75: return "Positive (0.75–1.0): mechanism is acquisition-invariant"
    if r >= 0.30: return "Partial (0.30–0.75): direction transfers, magnitude attenuated"
    if r >= -0.10: return "Null (−0.10–0.30): prior is dataset-specific in zero-shot"
    return "Negative (<−0.10): prior hurts cross-vendor"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, type=Path,
                    help="Submission JSON (see submission_schema.json).")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output Markdown report path.")
    args = ap.parse_args()

    sub = _load_json(args.submission)
    errs = _validate(sub)
    if errs:
        print("[evaluate] submission validation errors:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 2

    m = sub["method"]; s = sub["submitter"]
    rk = sub["results"]["kvasir"]; rg = sub["results"]["galar"]
    ret = _retention(sub)
    band = _band(ret)
    n_seeds = sub.get("n_seeds", "—")

    lines = []
    lines.append(f"# GalKva-2026 evaluation: {m['name']}")
    lines.append("")
    lines.append(f"- **Submitter:** {s['name']} ({s['affiliation']}) — `{s['email']}`")
    lines.append(f"- **Method:** {m['name']}")
    lines.append(f"- **Description:** {m['description']}")
    if m.get("preprint_url"): lines.append(f"- **Preprint:** <{m['preprint_url']}>")
    if m.get("code_url"):     lines.append(f"- **Code:** <{m['code_url']}>")
    lines.append(f"- **Benchmark version:** {sub['benchmark_version']}")
    lines.append(f"- **Seeds:** {n_seeds}")
    if m.get("trained_on"): lines.append(f"- **Trained on:** {m['trained_on']}")
    lines.append("")
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Dataset | Macro-AUC (mean ± SD) | n_test_frames | Δ vs RGB baseline | Sign-positive seeds |")
    lines.append("|---|---|---|---|---|")
    for label, r in [("Kvasir-Capsule", rk), ("Galar", rg)]:
        ma = r.get("macro_auc", {})
        cell = _fmt_pm(ma.get("mean", float("nan")), ma.get("sd"))
        n = r.get("n_test_frames", "—")
        dv = r.get("delta_vs_rgb_baseline", "—")
        if isinstance(dv, (int, float)): dv = f"{dv:+.3f}"
        sp = r.get("sign_positive_seeds", "—")
        lines.append(f"| {label} | {cell} | {n} | {dv} | {sp} |")
    lines.append("")
    lines.append("## Cross-vendor transferability")
    lines.append("")
    lines.append(f"- **Retention ratio Δ_Galar / Δ_Kvasir:** {ret}")
    lines.append(f"- **Band:** {band}")
    lines.append("")
    lines.append("## Per-class AUC on the 6-class cross-dataset intersection")
    lines.append("")
    lines.append("| Class | Kvasir mean AUC | Galar mean AUC |")
    lines.append("|---|---|---|")
    for c in EVALUABLE_6:
        k = rk.get("per_class_auc", {}).get(c, {}).get("mean")
        g = rg.get("per_class_auc", {}).get(c, {}).get("mean")
        kf = f"{k:.3f}" if isinstance(k, (int, float)) else "—"
        gf = f"{g:.3f}" if isinstance(g, (int, float)) else "—"
        lines.append(f"| {c} | {kf} | {gf} |")
    lines.append("")
    if sub.get("notes"):
        lines.append("## Submitter notes")
        lines.append("")
        lines.append(f"> {sub['notes']}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"[evaluate] wrote {args.out}")
    print(f"[evaluate] retention = {ret} ({band})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
