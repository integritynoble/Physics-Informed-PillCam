"""Subset-block transplant experiment.

EfficientNet-B0 has 9 feature blocks (0-8) + classifier. For each subset of
blocks, transplant ALL parameters in those blocks (conv weights + BN affine +
BN running stats) from +PI → RGB-only, keeping everything else from RGB-only.

Then run inference. The subset that recovers the most of the +PI lift is
the locus.

Subsets evaluated:
    A: features.0 (stem) — SKIP, first-conv shape mismatch
    B: features.1-2 (early blocks)
    C: features.3-5 (mid blocks)
    D: features.6-7 (late blocks)
    E: features.8 + classifier (head)
    F: features.1-8 + classifier (everything except stem) — sanity check

For each subset × each seed (6 seeds), output mAUC and Δ vs RGB-only baseline.
Compare against:
    - RGB-only baseline (un-transplanted)
    - +PI baseline (full +PI model, paper headline 0.783)

Reconstruction ratio = Δ / 0.0232 (the headline +PI − RGB delta).
"""
from __future__ import annotations

import json
import os
import sys
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

OUT_ROOT = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone")
GASTRO_DIR = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/gastroscopy_code_package"
CAPSULE_PKG = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/Capsule-Endoscopy"
DATA_DIR_OVERRIDE = "/project/BME/Zaman_lab/s248103/stage2_data_canonical"

ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
TRAINING_ONLY = {"Ampulla of Vater", "Blood - hematin", "Polyp"}
EVALUABLE_IDX = [i for i, c in enumerate(ALL_CLASSES) if c not in TRAINING_ONLY]

SEEDS = [41, 42, 43, 44, 45, 47]

SUBSETS = {
    "B_early_blocks_1_2":        [1, 2],
    "C_mid_blocks_3_4_5":        [3, 4, 5],
    "D_late_blocks_6_7":         [6, 7],
    "E_head_block_8_plus_clf":   [8, "classifier"],  # classifier is a sentinel
    "F_everything_except_stem":  [1, 2, 3, 4, 5, 6, 7, 8, "classifier"],
}


def setup_imports() -> None:
    for p in (GASTRO_DIR, CAPSULE_PKG):
        if p not in sys.path:
            sys.path.insert(0, p)


def block_of_key(k: str) -> int | str | None:
    """Return block index for a state_dict key, or 'classifier', or None for stem (features.0)."""
    m = re.match(r"backbone\.features\.(\d+)\.", k)
    if m:
        b = int(m.group(1))
        return b
    if "classifier" in k:
        return "classifier"
    return None


def transplant_subset(rgb_state: dict, pi_state: dict, blocks: list) -> tuple[dict, int]:
    """Copy all params whose block is in `blocks` from pi → rgb. Returns (new_state, n_copied)."""
    new_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in rgb_state.items()}
    n = 0
    for k in rgb_state:
        b = block_of_key(k)
        if b in blocks and k in pi_state:
            if pi_state[k].shape == rgb_state[k].shape:
                new_state[k] = pi_state[k].clone()
                n += 1
    return new_state, n


def build_model_and_transform(args: dict, device: str):
    from datasets_pi import build_transforms_pi
    from models_pi import ImageClassifierPI
    from datasets import build_transforms
    from models import ImageClassifier
    if args.get("use_physics_prior", False):
        tf = build_transforms_pi(
            args["image_size"], train=False,
            alpha=args.get("physics_alpha", 4.0),
            lambda_eff=args.get("physics_lambda_eff"),
            version=args.get("physics_prior_version", "v1"),
            pivot_v2=args.get("physics_pivot_v2", 0.30),
        )
        model = ImageClassifierPI(args["model_name"], num_classes=len(ALL_CLASSES), pretrained=False).to(device)
    else:
        tf = build_transforms(args["image_size"], False)
        model = ImageClassifier(args["model_name"], num_classes=len(ALL_CLASSES), pretrained=False).to(device)
    return model, tf


def infer_macro_auc(model, transform, device: str, batch_size: int = 128) -> float:
    from datasets import FolderDatasetWithPaths
    from metrics_pi import ensure_class_folders
    test_dir = os.path.join(DATA_DIR_OVERRIDE, "test")
    ensure_class_folders(test_dir, ALL_CLASSES)
    ds = FolderDatasetWithPaths(test_dir, transform=transform, allow_empty=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device == "cuda"))
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    evaluable = []
    for i, name in enumerate(ALL_CLASSES):
        y_true = (labels == i).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        if name not in TRAINING_ONLY:
            evaluable.append(float(roc_auc_score(y_true, probs[:, i])))
    return float(np.mean(evaluable))


def main():
    setup_imports()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[subset-transplant] device = {device}")
    print(f"[subset-transplant] subsets: {list(SUBSETS.keys())}")

    # results[subset_name][seed] = mAUC
    results: dict[str, dict[int, float]] = {name: {} for name in SUBSETS}
    rgb_baselines: dict[int, float] = {}
    pi_baselines: dict[int, float] = {}

    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        rgb_ckpt = torch.load(OUT_ROOT / f"effb0_paper_seed{seed}_rgb" / "best_model.pt", map_location=device, weights_only=False)
        pi_ckpt  = torch.load(OUT_ROOT / f"effb0_paper_seed{seed}_pi"  / "best_model.pt", map_location=device, weights_only=False)
        rgb_state = rgb_ckpt["model_state"]
        pi_state  = pi_ckpt["model_state"]

        # RGB-only baseline
        model_rgb, tf_rgb = build_model_and_transform(rgb_ckpt["args"], device)
        model_rgb.load_state_dict(rgb_state)
        m_rgb = infer_macro_auc(model_rgb, tf_rgb, device)
        rgb_baselines[seed] = m_rgb
        print(f"  RGB-only baseline:                 {m_rgb:.4f}")

        # +PI baseline (uses tf_pi because of 5-channel input + PI transform)
        model_pi, tf_pi = build_model_and_transform(pi_ckpt["args"], device)
        model_pi.load_state_dict(pi_state)
        m_pi = infer_macro_auc(model_pi, tf_pi, device)
        pi_baselines[seed] = m_pi
        print(f"  +PI baseline:                       {m_pi:.4f}")

        for subset_name, blocks in SUBSETS.items():
            transplanted_state, n = transplant_subset(rgb_state, pi_state, blocks)
            model_tr, _ = build_model_and_transform(rgb_ckpt["args"], device)
            model_tr.load_state_dict(transplanted_state)
            m_tr = infer_macro_auc(model_tr, tf_rgb, device)
            results[subset_name][seed] = m_tr
            print(f"  {subset_name:<30}: mAUC={m_tr:.4f}  (Δ vs RGB={m_tr - m_rgb:+.4f},  n_copied={n})")

    # Aggregate
    print("\n=== AGGREGATE (n=6) ===")
    rgb_arr = np.array([rgb_baselines[s] for s in SEEDS])
    pi_arr  = np.array([pi_baselines[s]  for s in SEEDS])
    print(f"  RGB-only baseline:           {rgb_arr.mean():.4f} ± {rgb_arr.std(ddof=1):.4f}")
    print(f"  +PI baseline:                {pi_arr.mean():.4f} ± {pi_arr.std(ddof=1):.4f}")
    headline_delta = (pi_arr - rgb_arr).mean()
    print(f"  Headline Δ(+PI − RGB):       {headline_delta:+.4f} ± {(pi_arr-rgb_arr).std(ddof=1):.4f}")
    print()
    for subset_name in SUBSETS:
        tr_arr = np.array([results[subset_name][s] for s in SEEDS])
        d = tr_arr - rgb_arr
        recon = d.mean() / headline_delta if headline_delta != 0 else float('nan')
        print(f"  {subset_name:<35}  mAUC={tr_arr.mean():.4f}±{tr_arr.std(ddof=1):.4f}  Δ={d.mean():+.4f}±{d.std(ddof=1):.4f}  sign-pos={int((d>0).sum())}/6  recon={recon:+.0%}")

    out_path = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/subset_transplant_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "seeds": SEEDS,
        "rgb_baselines": {str(s): rgb_baselines[s] for s in SEEDS},
        "pi_baselines":  {str(s): pi_baselines[s]  for s in SEEDS},
        "results": {name: {str(s): results[name][s] for s in SEEDS} for name in SUBSETS},
        "subsets": {name: blocks for name, blocks in SUBSETS.items()},
        "headline_delta_mean": float(headline_delta),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[subset-transplant] saved {out_path}")


if __name__ == "__main__":
    main()
