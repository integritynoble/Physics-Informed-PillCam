"""Post-hoc macro-AUC evaluator for cross-backbone runs.

For each cross_backbone/<backbone>_seed<N>_<arm>/ directory that has
a completed run (best_model.pt + test_metrics.json present), loads the
best checkpoint, runs the test split through it saving full softmax
probabilities, and computes per-class + macro-AUC.

The macro-AUC field is the paper's headline metric. The train_stage2_pi
test loop only saves argmax-derived F1/accuracy, not AUC -- hence this
post-hoc step.

Idempotent: skips dirs that already have test_auc.json (use --rerun to
overwrite). Safe to invoke while training continues, but it will share
the GPU with the live training process. To avoid contention while
training is still running, pass --device cpu (slower, ~3 min per run).

USAGE
    python eval_cross_backbone_auc.py
    python eval_cross_backbone_auc.py --rerun
    python eval_cross_backbone_auc.py --device cpu
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
from sklearn.metrics import roc_auc_score

OUT_ROOT = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone")
GASTRO_DIR = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/gastroscopy_code_package"
CAPSULE_PKG = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/Capsule-Endoscopy"

# 14 classes in alphabetical (ImageFolder default) order, matching the
# class_to_idx mapping produced by FolderDatasetWithPaths on the
# Kvasir-Capsule stage2 split.
ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]

# Three classes ship with only one source video each and are training-only
# per the paper's video-level 70/15/15 split (Section 3.1).
TRAINING_ONLY = {"Ampulla of Vater", "Blood - hematin", "Polyp"}
EVALUABLE_IDX = [i for i, c in enumerate(ALL_CLASSES) if c not in TRAINING_ONLY]


def _setup_imports() -> None:
    """Make the gastroscopy package and Capsule-Endoscopy package importable."""
    for p in (GASTRO_DIR, CAPSULE_PKG):
        if p not in sys.path:
            sys.path.insert(0, p)


def _build_model_and_transform(args: dict, device: str):
    """Reproduce the model + eval transform configuration used during training."""
    if args.get("use_physics_prior", False):
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
            args["model_name"],
            num_classes=len(ALL_CLASSES),
            pretrained=False,  # will load state dict immediately
        ).to(device)
    else:
        from datasets import build_transforms  # noqa: WPS433
        from models import ImageClassifier  # noqa: WPS433
        tf_eval = build_transforms(args["image_size"], False)
        model = ImageClassifier(
            args["model_name"],
            num_classes=len(ALL_CLASSES),
            pretrained=False,
        ).to(device)
    return model, tf_eval


def _read_best_epoch(run_dir: Path) -> int | None:
    """Best-epoch index from training_history.json (1-indexed); None if missing."""
    hist_path = run_dir / "training_history.json"
    if not hist_path.exists():
        return None
    history = json.loads(hist_path.read_text())["history"]
    if not history:
        return None
    best_epoch = max(
        history,
        key=lambda e: e["val"].get("macro_f1_evaluable", e["val"]["macro_f1"]),
    )["epoch"]
    return int(best_epoch)


def evaluate(run_dir: Path, device: str, batch_size: int,
             force_ablate: list[int] | None = None,
             output_name: str = "test_auc.json",
             data_dir_override: str | None = None) -> dict | None:
    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        return None
    test_metrics_path = run_dir / "test_metrics.json"
    if not test_metrics_path.exists():
        print(f"  SKIP: training not complete yet (no test_metrics.json)")
        return None

    out_path = run_dir / output_name

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    class_names = ckpt["class_names"]
    if class_names != ALL_CLASSES:
        raise RuntimeError(
            f"unexpected class set in {ckpt_path}: {class_names}"
        )

    model, tf_eval = _build_model_and_transform(args, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if force_ablate is not None:
        ablate_idx = list(force_ablate)
    else:
        ablate_spec = str(args.get("ablate_input_channels", "") or "").strip()
        ablate_idx = [int(c) for c in ablate_spec.split(",") if c.strip()]

    from datasets import FolderDatasetWithPaths  # noqa: WPS433
    from metrics_pi import ensure_class_folders  # noqa: WPS433

    data_dir = data_dir_override or args["data_dir"]
    test_dir = os.path.join(data_dir, "test")
    ensure_class_folders(test_dir, class_names)
    test_ds = FolderDatasetWithPaths(test_dir, transform=tf_eval, allow_empty=True)
    if test_ds.classes != class_names:
        raise RuntimeError(
            f"test_ds.classes != class_names: {test_ds.classes} vs {class_names}"
        )

    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=(device == "cuda"),
    )

    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    n_test = len(test_ds)
    arm_tag = "PI" if args.get("use_physics_prior") else "RGB"
    if ablate_idx:
        arm_tag += f" ablate={ablate_idx}"
    print(f"  inference on {n_test} test frames "
          f"(model={args['model_name']}, arm={arm_tag})")
    with torch.no_grad():
        for images, labels, _paths in test_loader:
            images = images.to(device, non_blocking=True)
            if ablate_idx:
                for ci in ablate_idx:
                    images[:, ci, :, :] = 0.0
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    per_class_auc: dict[str, float | None] = {}
    for i, name in enumerate(ALL_CLASSES):
        y_true = (labels == i).astype(int)
        n_pos = int(y_true.sum())
        if n_pos == 0 or n_pos == len(y_true):
            per_class_auc[name] = None  # cannot compute (no positives or all-positives)
            continue
        try:
            per_class_auc[name] = float(roc_auc_score(y_true, probs[:, i]))
        except Exception as e:  # noqa: BLE001
            print(f"  warn: AUC for {name} failed: {e}")
            per_class_auc[name] = None

    evaluable_aucs = [
        per_class_auc[ALL_CLASSES[i]]
        for i in EVALUABLE_IDX
        if per_class_auc[ALL_CLASSES[i]] is not None
    ]
    macro_auc_evaluable = (
        float(np.mean(evaluable_aucs)) if evaluable_aucs else None
    )
    all_valid = [v for v in per_class_auc.values() if v is not None]
    macro_auc = float(np.mean(all_valid)) if all_valid else None

    out = {
        "macro_auc": macro_auc,
        "macro_auc_evaluable": macro_auc_evaluable,
        "per_class_auc": per_class_auc,
        "n_test_frames": int(n_test),
        "n_evaluable_classes_with_signal": len(evaluable_aucs),
        "best_epoch": _read_best_epoch(run_dir),
        "best_val_macro_f1_evaluable": ckpt.get("best_val_macro_f1_evaluable"),
        "model_name": args["model_name"],
        "use_physics_prior": bool(args.get("use_physics_prior", False)),
        "ablate_input_channels": ablate_idx,
        "seed": int(args["seed"]),
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  -> macro_auc_evaluable = {macro_auc_evaluable:.4f}  "
          f"(saved {out_path.name})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="Overwrite existing test_auc.json files")
    parser.add_argument("--device", default=None,
                        choices=["cuda", "cpu"],
                        help="Override device. Default: cuda if available else cpu. "
                             "Pass 'cpu' to avoid contention with live training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Inference batch size (default 64; larger than the "
                             "training batch is fine because no gradients are tracked).")
    parser.add_argument("--only", type=str, default=None,
                        help="Substring filter on run directory name (e.g. 'resnet18').")
    parser.add_argument("--force_ablate_channels", type=str, default=None,
                        help="Comma-separated channel indices to zero at inference time, "
                             "overriding the training-time ablation stored in the checkpoint. "
                             "Use to probe what a fully-trained model relies on.")
    parser.add_argument("--output_name", type=str, default="test_auc.json",
                        help="Output filename (default test_auc.json). Use a distinct name "
                             "(e.g. test_auc_infer_ablate4.json) when using --force_ablate_channels.")
    parser.add_argument("--data_dir_override", type=str, default=None,
                        help="Override the checkpoint-stored data_dir (e.g. for Windows-trained "
                             "checkpoints whose D:/... path doesn't exist on this filesystem).")
    cli = parser.parse_args()

    _setup_imports()

    device = cli.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device = {device}")
    if device == "cuda":
        try:
            print(f"[eval] gpu = {torch.cuda.get_device_name(0)}, "
                  f"free mem = {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
        except Exception:  # noqa: BLE001
            pass

    if not OUT_ROOT.exists():
        print(f"[eval] {OUT_ROOT} does not exist; nothing to do")
        return 0

    run_dirs = sorted(
        d for d in OUT_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and (cli.only is None or cli.only in d.name)
    )
    print(f"[eval] scanning {len(run_dirs)} directories under {OUT_ROOT}"
          + (f" (filter: '{cli.only}')" if cli.only else ""))

    n_done = 0
    n_skipped_no_ckpt = 0
    n_skipped_no_test = 0
    n_skipped_existing = 0
    n_failed = 0

    for d in run_dirs:
        if not (d / "best_model.pt").exists():
            n_skipped_no_ckpt += 1
            continue
        if not (d / "test_metrics.json").exists():
            n_skipped_no_test += 1
            continue
        if (d / cli.output_name).exists() and not cli.rerun:
            n_skipped_existing += 1
            continue
        print(f"\n[eval] {d.name}")
        force_ablate = None
        if cli.force_ablate_channels:
            force_ablate = [int(c) for c in cli.force_ablate_channels.split(",") if c.strip()]
        try:
            evaluate(d, device=device, batch_size=cli.batch_size,
                     force_ablate=force_ablate, output_name=cli.output_name,
                     data_dir_override=cli.data_dir_override)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {e}")
            traceback.print_exc()
            n_failed += 1

    print(f"\n[eval] summary: "
          f"{n_done} evaluated, "
          f"{n_skipped_existing} already-done skipped, "
          f"{n_skipped_no_test} incomplete-training skipped, "
          f"{n_skipped_no_ckpt} no-checkpoint skipped, "
          f"{n_failed} failed")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
