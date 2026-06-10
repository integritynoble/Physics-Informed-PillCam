"""Stage 2 abnormal-subtype training with optional physics-informed prior.

This is a copy of train_stage2_abnormal_subtype.py with a single new flag:

    --use_physics_prior    swap 3-channel RGB for 5-channel RGB+physics input.
                           Uses ImageClassifierPI and build_transforms_pi.
                           Default: off (matches original baseline).

Run from the paper_draft/ directory, e.g.:

    python train_stage2_pi.py \
      --data_dir ./stage2_data \
      --model_name efficientnet_b0 \
      --epochs 40 \
      --batch_size 24 \
      --image_size 224 \
      --lr 1e-4 \
      --output_dir ./outputs/stage2_pi_on \
      --pretrained \
      --use_physics_prior \
      --gastroscopy_code_dir "${GASTROSCOPY_CODE_DIR:-./gastroscopy_code_package}"

A paired RGB-baseline run for the ablation table is the same command without
--use_physics_prior and with --output_dir ./outputs/stage2_pi_off.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: abnormal subtype training (physics-informed optional)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--use_physics_prior", action="store_true",
                        help="Use 5-channel RGB+physics input. Default: off (RGB baseline).")
    parser.add_argument("--physics_alpha", type=float, default=4.0,
                        help="Sharpness of the hemoglobin sigmoid in the physics prior.")
    parser.add_argument("--physics_lambda_eff", type=float, default=None,
                        help="Effective fluence decay length (pixels). Default: 0.25 x image diagonal.")

    parser.add_argument("--gastroscopy_code_dir", type=str, default=None,
                        help="Path to the original gastroscopy_code_package folder (for datasets.py, models.py, utils.py). "
                             "Required unless that package is already on PYTHONPATH.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Ignore last.pt in --output_dir even if it exists; start training from scratch.")
    parser.add_argument("--mixed_precision", action="store_true",
                        help="Wrap forward/backward in torch.amp.autocast(dtype=fp16) and use GradScaler. "
                             "Matches paper §3.4 claim. Recommended on every CUDA device.")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine"],
                        help="LR schedule. 'cosine' applies CosineAnnealingLR(T_max=epochs); "
                             "'none' keeps lr constant. Default 'cosine' matches paper §3.4.")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Set torch.use_deterministic_algorithms(True), cudnn.deterministic, "
                             "CUBLAS_WORKSPACE_CONFIG, and DataLoader worker seeding. Matches paper §4.9 "
                             "reproducibility checklist. On by default.")
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false",
                        help="Disable deterministic mode (faster but non-reproducible).")
    parser.add_argument("--ablate_input_channels", type=str, default="",
                        help="Comma-separated channel indices to zero in the model input "
                             "(e.g. '4' to ablate H_AFI in a 5-channel PI run; '3' to ablate "
                             "P_blood; '0,1,2' for physics-only with RGB zeroed). Channel "
                             "convention: 0=R, 1=G, 2=B, 3=P_blood, 4=H_AFI*Phi.")

    # ---- physics prior version (added 2026-04-28) -------------------------
    parser.add_argument("--physics_prior_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="v1: paper-draft per-image quantile-normalized H. "
                             "v2: scale-fixed NDVI_red index — recommended for new runs "
                             "(per-class AUC analysis showed v1 P_blood actively hurt "
                             "Blood-fresh classification).")
    parser.add_argument("--physics_pivot_v2", type=float, default=0.30,
                        help="Pivot for the v2 sigmoid; ignored when --physics_prior_version=v1.")

    # ---- regularization knobs (added 2026-04-28 to address train–val gap) ---
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label-smoothing epsilon for cross-entropy. 0.05–0.1 typical. "
                             "Default 0.0 reproduces the as-drafted paper.")
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup interpolation strength (Beta(alpha, alpha)). 0.0 disables. "
                             "0.2 is a safe default if you turn it on.")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Stop training if val macro-F1-evaluable has not improved for "
                             "this many epochs. 0 (default) disables early stopping.")
    return parser.parse_args()


def _register_gastroscopy_package(path: str | None) -> None:
    """Make datasets/models/utils from the original package importable."""
    if path and os.path.isdir(path):
        sys.path.insert(0, os.path.abspath(path))
    # Defer the imports until after the path is set up
    globals()["FolderDatasetWithPaths"] = __import__("datasets", fromlist=["FolderDatasetWithPaths"]).FolderDatasetWithPaths
    globals()["build_transforms"] = __import__("datasets", fromlist=["build_transforms"]).build_transforms
    globals()["ImageClassifier"] = __import__("models", fromlist=["ImageClassifier"]).ImageClassifier
    _utils = __import__("utils", fromlist=["compute_class_weights", "count_trainable_parameters",
                                           "save_checkpoint", "save_json", "set_seed"])
    globals()["compute_class_weights"] = _utils.compute_class_weights
    globals()["count_trainable_parameters"] = _utils.count_trainable_parameters
    globals()["save_checkpoint"] = _utils.save_checkpoint
    globals()["save_json"] = _utils.save_json
    globals()["set_seed"] = _utils.set_seed
    # Use the robust shadow that passes labels=range(num_classes) so val/test
    # passes don't crash when the model fails to predict every class.
    from metrics_pi import summarize_classification as _summarize
    globals()["summarize_classification"] = _summarize


def run_epoch(model, loader, criterion, optimizer, device, class_names, train: bool,
              scaler=None, amp_enabled: bool = False, ablate_channels=None,
              mixup_alpha: float = 0.0):
    model.train(train)
    running_loss = 0.0
    y_true, y_pred = [], []

    use_cuda = (str(device).startswith("cuda") or device == "cuda")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if (amp_enabled and use_cuda)
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )
    ablate_idx = list(ablate_channels) if ablate_channels else []

    for images, labels, paths in tqdm(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # Per-channel ablation for Table 3 component analysis.
        # Zeroing happens inside the training loop (post-transform) so the
        # rest of the pipeline does not need to change. Saved per-frame
        # predictions reflect the ablated configuration.
        if ablate_idx:
            images = images.clone()
            for c in ablate_idx:
                images[:, c] = 0.0

        # MixUp (train-only). Pairs each example with a random other example
        # in the batch using lam ~ Beta(alpha, alpha); the criterion is
        # evaluated as a convex combination of the two label CEs. This is the
        # standard "input-and-target mixup" formulation.
        mixup_lam = None
        labels_b = None
        if train and mixup_alpha > 0:
            import numpy as _np
            mixup_lam = float(_np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(images.size(0), device=device)
            images = mixup_lam * images + (1.0 - mixup_lam) * images[perm]
            labels_b = labels[perm]

        with torch.set_grad_enabled(train):
            with autocast_ctx:
                logits = model(images)
                if mixup_lam is not None:
                    loss = mixup_lam * criterion(logits, labels) \
                         + (1.0 - mixup_lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        # For mixup, the per-frame y_true is ill-defined; record the dominant
        # label so the train-time macro-F1 still reflects classification
        # behavior on real labels. Val/test never use mixup, so this branch
        # is irrelevant outside training.
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = summarize_classification(y_true, y_pred, class_names)
    metrics["loss"] = running_loss / len(loader.dataset)
    return metrics


def _worker_init_fn(worker_id: int) -> None:
    """Seed numpy / random per DataLoader worker so augmentation order is stable."""
    import random as _random
    import numpy as _np
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    _np.random.seed(seed)
    _random.seed(seed)


def _enable_deterministic(seed: int) -> None:
    """Best-effort deterministic execution. Required by paper §4.9."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    _register_gastroscopy_package(args.gastroscopy_code_dir)
    set_seed(args.seed)
    if args.deterministic:
        _enable_deterministic(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # -- transforms --
    if args.use_physics_prior:
        from datasets_pi import build_transforms_pi
        tf_train = build_transforms_pi(args.image_size, train=True,
                                        alpha=args.physics_alpha,
                                        lambda_eff=args.physics_lambda_eff,
                                        version=args.physics_prior_version,
                                        pivot_v2=args.physics_pivot_v2)
        tf_eval = build_transforms_pi(args.image_size, train=False,
                                       alpha=args.physics_alpha,
                                       lambda_eff=args.physics_lambda_eff,
                                       version=args.physics_prior_version,
                                       pivot_v2=args.physics_pivot_v2)
        print(f"[train_stage2_pi] physics prior ENABLED "
              f"(version={args.physics_prior_version}, alpha={args.physics_alpha}, "
              f"lambda_eff={args.physics_lambda_eff}, pivot_v2={args.physics_pivot_v2})")
    else:
        tf_train = build_transforms(args.image_size, True)
        tf_eval = build_transforms(args.image_size, False)
        print(f"[train_stage2_pi] physics prior OFF (RGB baseline)")

    train_ds = FolderDatasetWithPaths(os.path.join(args.data_dir, "train"), transform=tf_train,
                                       allow_empty=True)
    class_names = train_ds.classes
    # Align val/test class_to_idx to train's by ensuring every class folder
    # exists in val and test (empty folders are required so ImageFolder
    # produces the same alphabetical class_to_idx mapping across all splits).
    from metrics_pi import ensure_class_folders
    ensure_class_folders(os.path.join(args.data_dir, "val"), class_names)
    ensure_class_folders(os.path.join(args.data_dir, "test"), class_names)
    val_ds = FolderDatasetWithPaths(os.path.join(args.data_dir, "val"), transform=tf_eval,
                                     allow_empty=True)
    test_ds = FolderDatasetWithPaths(os.path.join(args.data_dir, "test"), transform=tf_eval,
                                      allow_empty=True)
    assert val_ds.classes == class_names, "val class_to_idx mismatch"
    assert test_ds.classes == class_names, "test class_to_idx mismatch"
    print(f"[train_stage2_pi] classes = {class_names}")

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader_kwargs = dict(num_workers=args.num_workers,
                         pin_memory=str(args.device).startswith("cuda"),
                         worker_init_fn=_worker_init_fn,
                         generator=loader_gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    # -- model --
    if args.use_physics_prior:
        from models_pi import ImageClassifierPI
        model = ImageClassifierPI(args.model_name, num_classes=len(class_names),
                                   pretrained=args.pretrained).to(args.device)
    else:
        model = ImageClassifier(args.model_name, num_classes=len(class_names),
                                 pretrained=args.pretrained).to(args.device)

    print(f"[train_stage2_pi] trainable params: {count_trainable_parameters(model):,}")

    train_labels = [label for _, label in train_ds.samples]
    class_weights = compute_class_weights(train_labels, len(class_names)).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                     label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.label_smoothing > 0 or args.mixup_alpha > 0 or args.early_stopping_patience > 0:
        print(f"[train_stage2_pi] regularization: label_smoothing={args.label_smoothing}  "
              f"mixup_alpha={args.mixup_alpha}  early_stopping_patience={args.early_stopping_patience}")

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    amp_enabled = args.mixed_precision and str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    ablate_channels = []
    if args.ablate_input_channels.strip():
        ablate_channels = [int(c) for c in args.ablate_input_channels.split(",") if c.strip()]
    print(f"[train_stage2_pi] mixed_precision={amp_enabled}  scheduler={args.scheduler}  deterministic={args.deterministic}  ablate_channels={ablate_channels}")

    best_score = -1.0
    history = []
    start_epoch = 1
    last_path = os.path.join(args.output_dir, "last.pt")
    if not args.no_resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=args.device, weights_only=False)
        if state.get("class_names") != class_names:
            raise SystemExit(
                f"resume aborted: class_names in {last_path} do not match the current "
                f"data split. Pass --no_resume to discard, or delete {last_path}."
            )
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and state.get("scheduler_state") is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        if scaler is not None and state.get("scaler_state") is not None:
            scaler.load_state_dict(state["scaler_state"])
        start_epoch = state["epoch"] + 1
        best_score = state.get("best_score", -1.0)
        history = state.get("history", [])
        print(f"[train_stage2_pi] resumed from {last_path}: start_epoch={start_epoch}, best_val_macro_f1={best_score:.4f}")

    epochs_since_best = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, args.device, class_names, True,
                                   scaler=scaler, amp_enabled=amp_enabled, ablate_channels=ablate_channels,
                                   mixup_alpha=args.mixup_alpha)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, args.device, class_names, False,
                                 scaler=None, amp_enabled=amp_enabled, ablate_channels=ablate_channels,
                                 mixup_alpha=0.0)
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"Train loss={train_metrics['loss']:.4f} macro_f1={train_metrics['macro_f1']:.4f}")
        print(f"Val   loss={val_metrics['loss']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

        # Select best by val macro-F1 over evaluable classes (those with positive
        # support in the val split). The 14-class macro_f1 averages in three
        # always-zero per-class F1s (Ampulla, Blood-hematin, Polyp are training-
        # only per §3.2), turning checkpoint selection into a noisy 11/14-scaled
        # version of the true objective. Switched 2026-04-28 to align checkpoint
        # selection with the paper's headline metric (`macro_f1_evaluable`).
        sel_score = val_metrics.get("macro_f1_evaluable", val_metrics["macro_f1"])
        if sel_score > best_score:
            best_score = sel_score
            epochs_since_best = 0
            save_checkpoint({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "args": vars(args),
                "best_val_macro_f1_evaluable": best_score,
                "best_val_macro_f1": val_metrics["macro_f1"],
            }, os.path.join(args.output_dir, "best_model.pt"))
        else:
            epochs_since_best += 1

        # Persist full resumable state every epoch (model + optimizer + scheduler + scaler + epoch + history).
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "best_score": best_score,
            "class_names": class_names,
            "history": history,
            "args": vars(args),
        }, last_path)
        # Also update training_history.json each epoch so partial runs are inspectable.
        save_json({"history": history}, os.path.join(args.output_dir, "training_history.json"))

        # Early stopping. Triggered on epochs_since_best, computed against the
        # macro-F1-evaluable selection criterion above so it agrees with the
        # checkpoint we'll evaluate on test.
        if args.early_stopping_patience > 0 and epochs_since_best >= args.early_stopping_patience:
            print(f"[train_stage2_pi] early stopping at epoch {epoch}: "
                  f"{epochs_since_best} epochs without improvement on "
                  f"val_macro_f1_evaluable (patience={args.early_stopping_patience})")
            break

    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = run_epoch(model, test_loader, criterion, optimizer=None, device=args.device,
                             class_names=class_names, train=False,
                             scaler=None, amp_enabled=amp_enabled, ablate_channels=ablate_channels)
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))
    print("\nFinal test macro_f1:", test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
