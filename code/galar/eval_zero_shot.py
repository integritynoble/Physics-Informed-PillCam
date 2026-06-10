"""Galar zero-shot evaluation for Kvasir-Capsule trained checkpoints.

Loads a Kvasir-Capsule-trained classifier (RGB baseline, 5-channel PI input
fusion, or 3-channel distill) and runs inference on the Galar test split
that was staged by setup_galar.py. Computes macro-AUC over the 6
cross-dataset-evaluable classes.

USAGE
    # one checkpoint
    python eval_zero_shot.py \\
        --checkpoint_dir /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone/resnet18_seed41_pi \\
        --galar_test_dir /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/data/galar_eval/test

    # all completed checkpoints (cross_backbone + distill)
    python eval_zero_shot.py --galar_test_dir ...

    # avoid GPU contention while another job is training
    python eval_zero_shot.py --device cpu --galar_test_dir ...

Outputs (per checkpoint, written into the checkpoint dir):
    galar_test_auc.json
    galar_test_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

GASTRO_DIR = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/gastroscopy_code_package"
CAPSULE_PKG = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/Capsule-Endoscopy"

# Match the 14-class output head of every Kvasir-Capsule-trained model.
ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]

# Updated 2026-05-15 after inspecting the Galar metadata. Galar has direct
# labels for 12 Kvasir classes (plus Normal clean mucosa derived from
# section=='small intestine' + no pathology); only Reduced Mucosal View is
# omitted because the view-quality labels exist on only 6 of 80 videos.
# Includes the three Kvasir-train-only classes (Ampulla, Blood-hematin,
# Polyp) which become testable cross-dataset via Galar.
GALAR_EVALUABLE_CLASSES = [
    "Ampulla of Vater",
    "Angiectasia",
    "Blood - fresh",
    "Blood - hematin",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Polyp",
    "Pylorus",
    "Ulcer",
]
GALAR_EVALUABLE_IDX = [ALL_CLASSES.index(c) for c in GALAR_EVALUABLE_CLASSES]


def _setup_imports() -> None:
    for p in (GASTRO_DIR, CAPSULE_PKG):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_model_and_transform(args: dict, device: str):
    """Reproduce the eval-time model + transform for any of:
      - RGB baseline       (datasets.build_transforms + models.ImageClassifier)
      - 5-channel PI fusion (datasets_pi.build_transforms_pi + models_pi.ImageClassifierPI)
      - 3-channel distill   (datasets.build_transforms + models_pi.ImageClassifierPIDistill,
                             with deploy_mode=True at inference)
    """
    use_pi = bool(args.get("use_physics_prior", False))
    use_distill = bool(args.get("distill_lambda", 0)) and not use_pi

    if use_pi:
        from datasets_pi import build_transforms_pi  # noqa: WPS433
        from models_pi import ImageClassifierPI  # noqa: WPS433
        tf_eval = build_transforms_pi(
            args["image_size"], train=False,
            alpha=args.get("physics_alpha", 4.0),
            lambda_eff=args.get("physics_lambda_eff"),
            version=args.get("physics_prior_version", "v1"),
            pivot_v2=args.get("physics_pivot_v2", 0.30),
        )
        model = ImageClassifierPI(
            args["model_name"], num_classes=len(ALL_CLASSES), pretrained=False,
        ).to(device)
    elif use_distill:
        from datasets import build_transforms  # noqa: WPS433
        from models_pi import ImageClassifierPIDistill  # noqa: WPS433
        tf_eval = build_transforms(args["image_size"], False)
        model = ImageClassifierPIDistill(
            args["model_name"], num_classes=len(ALL_CLASSES), pretrained=False,
        ).to(device)
        model.deploy_mode = True
    else:
        from datasets import build_transforms  # noqa: WPS433
        from models import ImageClassifier  # noqa: WPS433
        tf_eval = build_transforms(args["image_size"], False)
        model = ImageClassifier(
            args["model_name"], num_classes=len(ALL_CLASSES), pretrained=False,
        ).to(device)
    return model, tf_eval, ("pi" if use_pi else ("distill" if use_distill else "rgb"))


def evaluate(checkpoint_dir: Path, galar_test_dir: Path,
             device: str, batch_size: int) -> dict | None:
    ckpt_path = checkpoint_dir / "best_model.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: no best_model.pt under {checkpoint_dir}")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    class_names = ckpt["class_names"]
    if class_names != ALL_CLASSES:
        raise RuntimeError(f"unexpected class set in {ckpt_path}: {class_names}")

    model, tf_eval, arm = _build_model_and_transform(args, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    from datasets import FolderDatasetWithPaths  # noqa: WPS433
    from metrics_pi import ensure_class_folders  # noqa: WPS433

    ensure_class_folders(str(galar_test_dir), GALAR_EVALUABLE_CLASSES)
    test_ds = FolderDatasetWithPaths(str(galar_test_dir), transform=tf_eval, allow_empty=True)
    if test_ds.classes != GALAR_EVALUABLE_CLASSES:
        raise RuntimeError(
            f"Galar test_ds.classes != GALAR_EVALUABLE_CLASSES: "
            f"{test_ds.classes} vs {GALAR_EVALUABLE_CLASSES}"
        )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    all_probs: list[np.ndarray] = []
    all_galar_labels: list[np.ndarray] = []
    n_test = len(test_ds)
    print(f"  inference on {n_test} Galar frames "
          f"(model={args['model_name']}, arm={arm.upper()})")
    with torch.no_grad():
        for images, labels, _paths in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_galar_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    galar_labels = np.concatenate(all_galar_labels, axis=0)

    # Per-class AUC restricted to the 6 evaluable classes; model emits 14-class
    # probs but we only score the columns corresponding to the evaluable mapping.
    per_class_auc: dict[str, float | None] = {}
    per_class_n_pos: dict[str, int] = {}
    for k_idx, c in enumerate(GALAR_EVALUABLE_CLASSES):
        kvasir_idx = ALL_CLASSES.index(c)
        y_true = (galar_labels == k_idx).astype(int)
        n_pos = int(y_true.sum())
        per_class_n_pos[c] = n_pos
        if n_pos == 0 or n_pos == len(y_true):
            per_class_auc[c] = None
            continue
        try:
            per_class_auc[c] = float(roc_auc_score(y_true, probs[:, kvasir_idx]))
        except Exception as e:  # noqa: BLE001
            print(f"  warn: AUC for {c} failed: {e}")
            per_class_auc[c] = None

    valid = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(valid)) if valid else None

    # F1 via argmax over the 6 evaluable Kvasir logits (deployment-realistic).
    pred_idx = probs[:, [ALL_CLASSES.index(c) for c in GALAR_EVALUABLE_CLASSES]].argmax(axis=1)
    macro_f1 = float(f1_score(galar_labels, pred_idx, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(galar_labels, pred_idx, average="weighted", zero_division=0))

    auc_out = {
        "macro_auc": macro_auc,
        "per_class_auc": per_class_auc,
        "per_class_n_positive": per_class_n_pos,
        "n_test_frames": int(n_test),
        "n_evaluable_classes_with_signal": len(valid),
        "evaluable_classes": GALAR_EVALUABLE_CLASSES,
        "dataset": "galar",
        "model_name": args["model_name"],
        "use_physics_prior": bool(args.get("use_physics_prior", False)),
        "is_distill": (arm == "distill"),
        "seed": int(args["seed"]),
    }
    metrics_out = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test_frames": int(n_test),
        "per_class_n_positive": per_class_n_pos,
        "arm": arm,
        "seed": int(args["seed"]),
        "model_name": args["model_name"],
        "dataset": "galar",
    }
    (checkpoint_dir / "galar_test_auc.json").write_text(json.dumps(auc_out, indent=2))
    (checkpoint_dir / "galar_test_metrics.json").write_text(json.dumps(metrics_out, indent=2))
    print(f"  -> Galar macro_auc = {macro_auc:.4f}  Galar macro_f1 = {macro_f1:.4f}")
    return auc_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Galar zero-shot eval")
    parser.add_argument("--galar_test_dir", required=True, type=Path,
                        help="Staged Galar test split (ImageFolder layout over the 6 evaluable classes)")
    parser.add_argument("--checkpoint_dir", default=None, type=Path,
                        help="Single checkpoint dir. If omitted, iterates over cross_backbone/* and stage2_distill_*/.")
    parser.add_argument("--scan_root", default=None, type=Path,
                        help="Root to scan. Default: GI_project/outputs/")
    parser.add_argument("--rerun", action="store_true",
                        help="Overwrite existing galar_test_auc.json files")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--batch_size", type=int, default=64)
    cli = parser.parse_args()

    _setup_imports()
    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[galar-eval] device = {device}")

    if not cli.galar_test_dir.exists():
        print(f"[galar-eval] ERROR: Galar test dir not found: {cli.galar_test_dir}")
        return 2

    if cli.checkpoint_dir:
        targets = [cli.checkpoint_dir]
    else:
        scan_root = cli.scan_root or Path(
            "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs"
        )
        targets = []
        for sub in ("cross_backbone",):
            d = scan_root / sub
            if d.exists():
                targets.extend(sorted(p for p in d.iterdir() if p.is_dir()))
        for d in sorted(scan_root.glob("stage2_distill_*")):
            if d.is_dir():
                targets.append(d)
        print(f"[galar-eval] scanning {len(targets)} checkpoint dirs")

    n_done, n_skipped, n_failed = 0, 0, 0
    for d in targets:
        if not (d / "best_model.pt").exists():
            continue
        if (d / "galar_test_auc.json").exists() and not cli.rerun:
            n_skipped += 1
            continue
        print(f"\n[galar-eval] {d.name}")
        try:
            evaluate(d, cli.galar_test_dir, device=device, batch_size=cli.batch_size)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}")
            traceback.print_exc()
            n_failed += 1

    print(f"\n[galar-eval] summary: {n_done} evaluated, "
          f"{n_skipped} already-done skipped, {n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
