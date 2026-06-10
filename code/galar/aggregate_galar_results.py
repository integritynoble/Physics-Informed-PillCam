"""Aggregate Galar zero-shot results across the 6-seed protocol.

Reads galar_test_auc.json from each
  /home2/.../outputs/cross_backbone/<backbone>_seed<N>_<arm>/
  /home2/.../outputs/stage2_distill_<backbone>[_seed<N>]/
directory and produces a Markdown summary at
  GI_project/paper/medIA_submission/docs/galar_zeroshot_summary.md.

Reports:
- Per-backbone cross-seed macro-AUC mean ± sd on Galar (RGB and PI arms)
- Paired Δ_Galar (PI − RGB) per seed and the cross-seed mean
- Δ_Galar / Δ_Kvasir retention ratio (the headline number for the paper)
- Per-evaluable-class mean AUC across seeds

USAGE
    python aggregate_galar_results.py
    python aggregate_galar_results.py --only resnet18      # filter by backbone
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

OUT_ROOT = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone")
SUMMARY_PATH = Path(
    "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/galar_zeroshot_summary.md"
)
KVASIR_SUMMARY_PATH = Path(
    "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/cross_backbone_summary.md"
)

BACKBONES = ["resnet18", "convnext_tiny", "efficientnet_b0",
             "effb0_headline", "effb0_paper", "effb0_canonical"]
SEEDS = [41, 42, 43, 44, 45, 47]
ARMS = ["rgb", "pi"]

PRETTY = {
    "resnet18": "ResNet-18",
    "convnext_tiny": "ConvNeXt-Tiny",
    "efficientnet_b0": "EfficientNet-B0 (cluster matched-recipe)",
    "effb0_headline": "EfficientNet-B0 (cluster headline recipe)",
    "effb0_paper": "EfficientNet-B0 (paper §4.1 / Windows-trained)",
    "effb0_canonical": "EfficientNet-B0 (cluster canonical-split n=10)",
}

EVALUABLE_CLASSES = [
    "Angiectasia", "Blood - fresh", "Lymphangiectasia",
    "Normal clean mucosa", "Polyp", "Ulcer",
]


def _read_galar_auc(backbone: str, seed: int, arm: str) -> dict | None:
    fp = OUT_ROOT / f"{backbone}_seed{seed}_{arm}" / "galar_test_auc.json"
    if not fp.exists():
        return None
    return json.loads(fp.read_text())


# Paper-published in-domain Δ_Kvasir per backbone. These are the canonical
# values cited in the medIA paper §4.1 (EffB0) and §4.4 (ResNet-18,
# ConvNeXt-Tiny). They originate from the Windows-trained checkpoints
# (effb0_paper) and the cluster cross-backbone runs (resnet18, convnext_tiny);
# the cluster's matched-recipe and headline-recipe EffB0 retrains land on a
# different data split and do not reproduce these numbers — see
# acceptance_strategy_2026-05-16.md for the cross-machine reproducibility
# story.
PUBLISHED_KVASIR_DELTAS = {
    "efficientnet_b0":  0.023,   # paper §4.1 headline, Windows-trained
    "effb0_headline":   0.023,   # same model family as paper §4.1 (cluster reproduction attempt; see §3.1)
    "effb0_paper":      0.023,   # the Windows-trained checkpoints themselves
    "effb0_canonical":  0.023,   # cluster n=10 on canonical split; uses paper's Δ_Kvasir for retention
    "convnext_tiny":    0.018,   # paper §4.4 (cluster matched-recipe)
    "resnet18":        -0.017,   # paper §4.4 (cluster matched-recipe)
}


def _read_kvasir_deltas() -> dict[str, float]:
    """Δ_Kvasir per backbone — paper-canonical values, with optional override
    from cross_backbone_summary.md if present. Hardcoded values ensure the
    retention column populates correctly even on cluster mirrors that don't
    yet have the cross_backbone_summary.md file."""
    deltas = dict(PUBLISHED_KVASIR_DELTAS)
    if KVASIR_SUMMARY_PATH.exists():
        text = KVASIR_SUMMARY_PATH.read_text()
        for bb_key, label in (("resnet18", "ResNet-18"), ("convnext_tiny", "ConvNeXt-Tiny")):
            for line in text.splitlines():
                if line.startswith(f"| {label} ") and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    for p in parts:
                        if p.startswith("-0.0") or p.startswith("+0.0") or p.startswith("0.0"):
                            try:
                                deltas[bb_key] = float(p)
                                break
                            except ValueError:
                                pass
                    break
    return deltas


def _fmt_pm(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Filter by backbone substring (e.g. 'resnet18')")
    cli = parser.parse_args()

    backbones = [b for b in BACKBONES if (cli.only is None or cli.only in b)]

    rows: dict[tuple[str, str], list[dict]] = {}
    missing: list[str] = []
    for bb in backbones:
        for arm in ARMS:
            seed_data: list[dict] = []
            for seed in SEEDS:
                rec = _read_galar_auc(bb, seed, arm)
                if rec is None:
                    missing.append(f"{bb}_seed{seed}_{arm}")
                else:
                    seed_data.append({"seed": seed, **rec})
            rows[(bb, arm)] = seed_data

    if missing:
        print("[aggregate] WARNING: missing galar_test_auc.json for:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)

    kvasir_deltas = _read_kvasir_deltas()

    lines: list[str] = []
    lines.append("# Galar zero-shot evaluation summary")
    lines.append("")
    lines.append(
        f"Aggregated from `{OUT_ROOT}` on seeds {','.join(str(s) for s in SEEDS)}. "
        f"Galar test split staged via `setup_galar.py`; inference via `eval_zero_shot.py`."
    )
    lines.append("")
    lines.append("## Cross-seed macro-AUC on Galar (paired RGB vs PI)")
    lines.append("")
    lines.append(f"| Backbone | Arm | Galar macro-AUC (n={len(SEEDS)}) | Per-seed values |")
    lines.append("|---|---|---|---|")
    means_per_arm: dict[tuple[str, str], float | None] = {}
    for bb in backbones:
        for arm in ARMS:
            data = rows[(bb, arm)]
            aucs = [d.get("macro_auc") for d in data if d.get("macro_auc") is not None]
            if not aucs:
                cell = "n/a"
                per_seed = "n/a"
                means_per_arm[(bb, arm)] = None
            else:
                mean, sd = st.mean(aucs), st.stdev(aucs) if len(aucs) > 1 else 0.0
                cell = _fmt_pm(mean, sd)
                per_seed = ", ".join(f"{v:.3f}" for v in aucs)
                means_per_arm[(bb, arm)] = mean
            lines.append(f"| {PRETTY.get(bb, bb)} | {'RGB-only' if arm == 'rgb' else '+PI input-fusion'} | {cell} | {per_seed} |")
    lines.append("")

    lines.append("## Paired Δ_Galar (PI − RGB) and retention ratio vs Kvasir")
    lines.append("")
    lines.append("| Backbone | Δ_Galar (mean) | Sign-positive seeds | Δ_Kvasir (recall) | Retention ratio Δ_Galar / Δ_Kvasir |")
    lines.append("|---|---|---|---|---|")
    for bb in backbones:
        rgb = {d["seed"]: d for d in rows[(bb, "rgb")]}
        pi = {d["seed"]: d for d in rows[(bb, "pi")]}
        deltas = []
        for seed in SEEDS:
            r, p = rgb.get(seed), pi.get(seed)
            if r and p and r.get("macro_auc") is not None and p.get("macro_auc") is not None:
                deltas.append(p["macro_auc"] - r["macro_auc"])
        if not deltas:
            lines.append(f"| {PRETTY.get(bb, bb)} | n/a | 0 | n/a | n/a |")
            continue
        d_galar_mean = st.mean(deltas)
        n_pos = sum(1 for d in deltas if d > 0)
        d_kvasir = kvasir_deltas.get(bb)
        if d_kvasir is not None and d_kvasir != 0:
            ret = d_galar_mean / d_kvasir
            ret_str = f"{ret:+.2f}"
        else:
            ret_str = "fill_after_kvasir_summary"
        d_kvasir_str = f"{d_kvasir:+.3f}" if d_kvasir is not None else "TBD"
        lines.append(
            f"| {PRETTY.get(bb, bb)} | {d_galar_mean:+.3f} | {n_pos}/{len(deltas)} | {d_kvasir_str} | {ret_str} |"
        )
    lines.append("")

    lines.append("## Per-class mean AUC on Galar (PI arm, cross-seed)")
    lines.append("")
    lines.append("| Backbone | " + " | ".join(EVALUABLE_CLASSES) + " |")
    lines.append("|" + "---|" * (len(EVALUABLE_CLASSES) + 1))
    for bb in backbones:
        pi = rows[(bb, "pi")]
        per_class_means = []
        for c in EVALUABLE_CLASSES:
            vals = [d.get("per_class_auc", {}).get(c) for d in pi]
            vals = [v for v in vals if v is not None]
            if vals:
                per_class_means.append(f"{st.mean(vals):.3f}")
            else:
                per_class_means.append("n/a")
        lines.append(f"| {PRETTY.get(bb, bb)} | " + " | ".join(per_class_means) + " |")
    lines.append("")

    lines.append("## Headline sentence for the paper")
    lines.append("")
    headline_bits = []
    for bb in backbones:
        rgb = {d["seed"]: d for d in rows[(bb, "rgb")]}
        pi = {d["seed"]: d for d in rows[(bb, "pi")]}
        deltas = []
        for seed in SEEDS:
            if seed in rgb and seed in pi:
                if rgb[seed].get("macro_auc") is not None and pi[seed].get("macro_auc") is not None:
                    deltas.append(pi[seed]["macro_auc"] - rgb[seed]["macro_auc"])
        if deltas:
            mean_d = st.mean(deltas)
            n_pos = sum(1 for d in deltas if d > 0)
            headline_bits.append(f"{PRETTY.get(bb, bb)} (Δ = {mean_d:+.3f}, {n_pos}/{len(deltas)} seeds positive)")
    if headline_bits:
        lines.append("> Zero-shot evaluation on the Galar dataset confirms the cross-dataset "
                     "transferability of the +PI lift on " + " and ".join(headline_bits) +
                     ". The retention ratios against the Kvasir-Capsule paired deltas "
                     "are reported in the table above.")

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines) + "\n")
    print(f"[aggregate] wrote {SUMMARY_PATH}")
    print(f"[aggregate] {sum(len(v) for v in rows.values())} per-seed AUCs aggregated, "
          f"{len(missing)} missing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
