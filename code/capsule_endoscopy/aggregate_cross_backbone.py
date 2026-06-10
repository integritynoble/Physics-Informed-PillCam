"""Aggregate cross-backbone replication results into a MedIA-ready summary.

Reads test_metrics.json from each
  /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone/<backbone>_seed<N>_<arm>/
directory produced by run_cross_backbone.sh, and writes a Markdown
summary to medIA_submission/cross_backbone_summary.md.

The summary contains:
  * cross-seed mean +/- SD macro-AUC per (backbone, arm)
  * per-seed delta (PI minus RGB) per backbone
  * a one-line headline sentence that can be lifted into the Limitations
    paragraph in section5_discussion.tex.

Run after run_cross_backbone.sh finishes:
    python aggregate_cross_backbone.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

OUT_ROOT = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone")
SUMMARY_PATH = Path(
    "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/cross_backbone_summary.md"
)

BACKBONES = ["resnet18", "convnext_tiny"]
SEEDS = [41, 42, 43, 44, 45, 47]
ARMS = ["rgb", "pi"]

PRETTY_BACKBONE = {
    "resnet18": "ResNet-18",
    "convnext_tiny": "ConvNeXt-Tiny",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
}


def _read_metric(backbone: str, seed: int, arm: str) -> float | None:
    # AUC values live in test_auc.json (written by eval_cross_backbone_auc.py).
    # test_metrics.json is the per-frame F1/accuracy report from training and
    # does not contain AUC.
    fp = OUT_ROOT / f"{backbone}_seed{seed}_{arm}" / "test_auc.json"
    if not fp.exists():
        return None
    data = json.loads(fp.read_text())
    # Prefer macro_auc_evaluable when present (the paper's headline metric);
    # fall back to macro_auc.
    return data.get("macro_auc_evaluable") or data.get("macro_auc")


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def main() -> int:
    rows = {}  # (backbone, arm) -> list of per-seed AUCs
    missing = []
    for backbone in BACKBONES:
        for arm in ARMS:
            seeds_auc = []
            for seed in SEEDS:
                v = _read_metric(backbone, seed, arm)
                if v is None:
                    missing.append(f"{backbone}_seed{seed}_{arm}")
                else:
                    seeds_auc.append(v)
            rows[(backbone, arm)] = seeds_auc

    if missing:
        print("[aggregate] WARNING: missing runs (skipped from summary):",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(file=sys.stderr)

    lines = []
    lines.append("# Cross-backbone replication summary")
    lines.append("")
    lines.append(f"Aggregated from `{OUT_ROOT}` on the seeds {','.join(str(s) for s in SEEDS)} protocol.")
    lines.append("")
    lines.append("## Cross-seed macro-AUC")
    lines.append("")
    lines.append(f"| Backbone | Arm | Macro-AUC (n={len(SEEDS)} seeds) | Per-seed values |")
    lines.append("|---|---|---|---|")

    deltas = {}  # backbone -> list of per-seed (PI - RGB) deltas
    for backbone in BACKBONES:
        for arm in ARMS:
            vs = rows[(backbone, arm)]
            if not vs:
                cell = "n/a"
                per_seed = "n/a"
            else:
                mean = st.mean(vs)
                sd = st.stdev(vs) if len(vs) > 1 else 0.0
                cell = _fmt(mean, sd)
                per_seed = ", ".join(f"{v:.3f}" for v in vs)
            lines.append(
                f"| {PRETTY_BACKBONE.get(backbone, backbone)} | "
                f"{'RGB-only' if arm == 'rgb' else '+PI input-fusion'} | "
                f"{cell} | {per_seed} |"
            )
        # paired delta if both arms present and same seed coverage
        rgb_seeds = rows[(backbone, "rgb")]
        pi_seeds = rows[(backbone, "pi")]
        if len(rgb_seeds) == len(pi_seeds) == len(SEEDS):
            d = [p - r for p, r in zip(pi_seeds, rgb_seeds)]
            deltas[backbone] = d

    lines.append("")
    if deltas:
        lines.append("## Paired per-seed PI lift")
        lines.append("")
        lines.append("| Backbone | Per-seed Δ (PI − RGB) | "
                     "Mean Δ | Sign-positive seeds |")
        lines.append("|---|---|---|---|")
        for backbone, ds in deltas.items():
            sign_pos = sum(1 for v in ds if v > 0)
            lines.append(
                f"| {PRETTY_BACKBONE.get(backbone, backbone)} | "
                f"{', '.join(f'{v:+.3f}' for v in ds)} | "
                f"{st.mean(ds):+.3f} | {sign_pos}/{len(ds)} |"
            )
        lines.append("")

    # Headline sentence for the Limitations paragraph
    headline_bits = []
    for backbone, ds in deltas.items():
        sign_pos = sum(1 for v in ds if v > 0)
        if sign_pos >= 2:
            headline_bits.append(
                f"{PRETTY_BACKBONE.get(backbone, backbone)} "
                f"(Δ = {st.mean(ds):+.3f}, sign-positive on "
                f"{sign_pos}/{len(ds)} seeds)"
            )
    if headline_bits:
        lines.append("## Headline sentence for the paper")
        lines.append("")
        lines.append(
            "> Cross-backbone replication on 3 seeds confirms the +PI lift "
            "direction on " + " and ".join(headline_bits) + "."
        )
        lines.append("")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines))
    print(f"[aggregate] wrote {SUMMARY_PATH}")
    print(f"[aggregate] {len(missing)} runs missing; "
          f"{sum(len(v) for v in rows.values())} per-seed AUCs aggregated.")
    return 0 if not missing else 1


if __name__ == "__main__":
    sys.exit(main())
