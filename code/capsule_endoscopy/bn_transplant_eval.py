"""BatchNorm-transplant experiment.

Tests whether the +PI training effect lives in BN running statistics.
For each seed s in {41,42,43,44,45,47}:
    1. Load RGB-only-trained checkpoint
    2. Load +PI-trained checkpoint
    3. Copy ALL BN parameters (γ/β/running_mean/running_var/num_batches_tracked)
       from +PI → RGB-only state_dict
    4. Re-run test inference on the canonical Linux split
    5. Compute macro-AUC

Compares against:
    - RGB-only baseline (un-transplanted)
    - +PI baseline (the 5-channel teacher)
    - +PI strip-and-serve (5-ch trained, prior channels zeroed at inference)

If BN-transplanted RGB matches +PI baseline → BN statistics ARE the mechanism.
If BN-transplanted RGB stays near RGB-only baseline → BN ruled out.
"""
from __future__ import annotations

import json
import os
import sys
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


def setup_imports() -> None:
    for p in (GASTRO_DIR, CAPSULE_PKG):
        if p not in sys.path:
            sys.path.insert(0, p)


def find_bn_keys(state: dict) -> list[str]:
    """Return list of BN parameter keys (γ/β/running_mean/running_var/num_batches_tracked).

    Identifies BN layers by the presence of a *.running_mean entry, then collects all
    siblings under the same prefix that are part of the BN affine + running buffers.
    """
    bn_layer_prefixes = sorted({k.rsplit(".", 1)[0] for k in state if k.endswith(".running_mean")})
    bn_keys = []
    for prefix in bn_layer_prefixes:
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            k = f"{prefix}.{suffix}"
            if k in state:
                bn_keys.append(k)
    return bn_keys


def transplant_bn(rgb_state: dict, pi_state: dict, bn_keys: list[str]) -> dict:
    """Copy BN params from pi_state → rgb_state. Returns a NEW dict (does not mutate)."""
    new_state = {k: v.clone() if torch.is_tensor(v) else v for k, v in rgb_state.items()}
    n_copied = 0
    for k in bn_keys:
        if k in pi_state and k in rgb_state:
            if pi_state[k].shape == rgb_state[k].shape:
                new_state[k] = pi_state[k].clone()
                n_copied += 1
            else:
                # Should not happen for BN params — they only depend on output channels,
                # which are identical between RGB and +PI EfficientNet-B0.
                print(f"  shape mismatch on {k}: pi={pi_state[k].shape} vs rgb={rgb_state[k].shape}  (SKIP)")
    return new_state, n_copied


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


def infer_macro_auc(model, transform, device: str, batch_size: int = 128) -> tuple[float, dict]:
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

    per_class = {}
    for i, name in enumerate(ALL_CLASSES):
        y_true = (labels == i).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_class[name] = None
            continue
        per_class[name] = float(roc_auc_score(y_true, probs[:, i]))
    evaluable = [per_class[ALL_CLASSES[i]] for i in EVALUABLE_IDX if per_class[ALL_CLASSES[i]] is not None]
    macro = float(np.mean(evaluable)) if evaluable else None
    return macro, per_class


def main():
    setup_imports()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bn-transplant] device = {device}")

    results = []
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        rgb_dir = OUT_ROOT / f"effb0_paper_seed{seed}_rgb"
        pi_dir  = OUT_ROOT / f"effb0_paper_seed{seed}_pi"
        rgb_ckpt = torch.load(rgb_dir / "best_model.pt", map_location=device, weights_only=False)
        pi_ckpt  = torch.load(pi_dir  / "best_model.pt", map_location=device, weights_only=False)
        rgb_state = rgb_ckpt["model_state"]
        pi_state  = pi_ckpt["model_state"]

        bn_keys = find_bn_keys(rgb_state)
        n_bn_layers = len([k for k in bn_keys if k.endswith(".running_mean")])
        print(f"  found {n_bn_layers} BN layers, {len(bn_keys)} total BN parameters")

        # Build the RGB-only model and load its weights as baseline
        model_rgb, tf = build_model_and_transform(rgb_ckpt["args"], device)
        model_rgb.load_state_dict(rgb_state)
        m_rgb, pc_rgb = infer_macro_auc(model_rgb, tf, device)
        print(f"  RGB-only baseline:                 {m_rgb:.4f}")

        # Build the BN-transplanted RGB model: same conv weights as RGB-only, BN params from +PI
        transplanted_state, n_copied = transplant_bn(rgb_state, pi_state, bn_keys)
        print(f"  copied {n_copied} BN parameter tensors from +PI → RGB")
        model_tr, _ = build_model_and_transform(rgb_ckpt["args"], device)
        model_tr.load_state_dict(transplanted_state)
        m_tr, pc_tr = infer_macro_auc(model_tr, tf, device)
        delta_rgb = m_tr - m_rgb
        print(f"  BN-transplanted (rgb-conv + pi-BN): {m_tr:.4f}   Δ vs RGB-only = {delta_rgb:+.4f}")

        results.append({
            "seed": seed,
            "rgb_baseline": m_rgb,
            "bn_transplanted": m_tr,
            "delta_vs_rgb": delta_rgb,
            "per_class_rgb": pc_rgb,
            "per_class_transplanted": pc_tr,
        })

    # Aggregate
    rgb_vals  = np.array([r["rgb_baseline"]    for r in results])
    tr_vals   = np.array([r["bn_transplanted"] for r in results])
    delta     = tr_vals - rgb_vals
    print("\n=== AGGREGATE ===")
    print(f"  RGB-only baseline (n=6):         {rgb_vals.mean():.4f} ± {rgb_vals.std(ddof=1):.4f}")
    print(f"  BN-transplanted (rgb+pi-BN):     {tr_vals.mean():.4f} ± {tr_vals.std(ddof=1):.4f}")
    print(f"  Δ(transplanted − RGB):           {delta.mean():+.4f} ± {delta.std(ddof=1):.4f}   sign-positive: {(delta>0).sum()}/{len(delta)}")
    print(f"  Paper headline (+PI − RGB):      +0.0232")
    print(f"  Reconstruction ratio:            {delta.mean() / 0.0232:.0%}")

    out = {"per_seed": results, "aggregate": {
        "rgb_baseline_mean": float(rgb_vals.mean()), "rgb_baseline_std": float(rgb_vals.std(ddof=1)),
        "bn_transplanted_mean": float(tr_vals.mean()), "bn_transplanted_std": float(tr_vals.std(ddof=1)),
        "delta_mean": float(delta.mean()), "delta_std": float(delta.std(ddof=1)),
        "sign_positive": int((delta>0).sum()),
        "n_seeds": len(delta),
    }}
    out_path = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/bn_transplant_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[bn-transplant] saved {out_path}")


if __name__ == "__main__":
    main()
