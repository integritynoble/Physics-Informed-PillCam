"""Stage 2 training with physics-distillation auxiliary head.

3-channel RGB input (deployment-friendly — no extra-channel injection at
inference). The backbone produces both a pooled feature vector for class
logits AND a spatial feature map fed to a small decoder that predicts the
analytic P_blood map. The model is trained with

    total_loss = CE(logits, label) + lambda_distill * BCE(pblood_logits, P_blood_target)

where P_blood_target is computed at training time from the un-normalized RGB
via physics_prior.blood_probability (the same per-frame percentile-clipped
formulation used by the 5-channel variant). At inference the auxiliary head
can be discarded.

Run from paper_draft/ (mirror the train_stage2_pi.py CLI):

    python train_stage2_pi_distill.py \
      --data_dir D:/kvasir_capsule/stage2_data \
      --model_name efficientnet_b0 \
      --epochs 30 --batch_size 24 --image_size 224 --lr 1e-4 \
      --output_dir D:/kvasir_capsule/outputs/stage2_distill_effb0 \
      --pretrained \
      --distill_lambda 1.0 \
      --gastroscopy_code_dir "${GASTROSCOPY_CODE_DIR:-./gastroscopy_code_package}"
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from physics_prior import blood_probability


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2: physics-distillation training (3-channel RGB + aux pblood head)")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--distill_lambda", type=float, default=1.0,
                        help="Weight on the auxiliary P_blood-prediction loss. 0 = no distillation.")
    parser.add_argument("--physics_alpha", type=float, default=4.0,
                        help="Sharpness of the hemoglobin sigmoid in the analytic P_blood teacher.")
    parser.add_argument("--physics_lambda_eff", type=float, default=None,
                        help="Effective fluence decay length (pixels). Default: 0.25 x image diagonal.")

    parser.add_argument("--gastroscopy_code_dir", type=str, default=None,
                        help="Path to the original gastroscopy_code_package folder (datasets.py, utils.py).")
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
                        help="Set torch.use_deterministic_algorithms(True) and DataLoader worker seeding.")
    parser.add_argument("--no_deterministic", dest="deterministic", action="store_false",
                        help="Disable deterministic mode (faster but non-reproducible).")
    # ---- physics-prior teacher version + regularization (added 2026-04-28) ----
    parser.add_argument("--physics_prior_version", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Teacher P_blood form: v1 (per-image quantile-norm) or "
                             "v2 (scale-fixed NDVI_red; recommended for new runs).")
    parser.add_argument("--physics_pivot_v2", type=float, default=0.30)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--mixup_alpha", type=float, default=0.0,
                        help="Mixup is on RGB inputs only; the auxiliary teacher target "
                             "is computed AFTER mixing so the prior is consistent with "
                             "the (mixed) input.")
    parser.add_argument("--early_stopping_patience", type=int, default=0)
    parser.add_argument("--init_backbone_from", type=str, default=None,
                        help="Path to a Stage-1 synthetic-pretrained backbone "
                             "checkpoint (saved by stage1_pretrain_synthetic.py "
                             "with key 'backbone_state'). Loaded AFTER the "
                             "ImageNet-pretrained init in models_pi but BEFORE "
                             "training begins. Used by the NMI extension to "
                             "swap ImageNet init for synthetic-data init.")
    return parser.parse_args()


def _register_gastroscopy_package(path: str | None) -> None:
    if path and os.path.isdir(path):
        sys.path.insert(0, os.path.abspath(path))
    globals()["FolderDatasetWithPaths"] = __import__("datasets", fromlist=["FolderDatasetWithPaths"]).FolderDatasetWithPaths
    globals()["build_transforms"] = __import__("datasets", fromlist=["build_transforms"]).build_transforms
    _utils = __import__("utils", fromlist=["compute_class_weights", "count_trainable_parameters",
                                           "save_checkpoint", "save_json", "set_seed"])
    globals()["compute_class_weights"] = _utils.compute_class_weights
    globals()["count_trainable_parameters"] = _utils.count_trainable_parameters
    globals()["save_checkpoint"] = _utils.save_checkpoint
    globals()["save_json"] = _utils.save_json
    globals()["set_seed"] = _utils.set_seed
    from metrics_pi import summarize_classification as _summarize
    globals()["summarize_classification"] = _summarize


def _to_rgb01(images_norm: torch.Tensor) -> torch.Tensor:
    """Reverse ImageNet normalization to recover [0,1] RGB for the analytic prior."""
    mean = IMAGENET_MEAN.to(images_norm.device, dtype=images_norm.dtype)
    std = IMAGENET_STD.to(images_norm.device, dtype=images_norm.dtype)
    return (images_norm * std + mean).clamp(0.0, 1.0)


def run_epoch(model, loader, ce_loss, optimizer, device, class_names, train: bool,
              distill_lambda: float, physics_alpha: float, physics_lambda_eff,
              scaler=None, amp_enabled: bool = False,
              prior_version: str = "v1", pivot_v2: float = 0.30,
              mixup_alpha: float = 0.0):
    model.train(train)
    running_total = 0.0
    running_ce = 0.0
    running_aux = 0.0
    y_true, y_pred = [], []
    bce = nn.BCEWithLogitsLoss()

    use_cuda = (str(device).startswith("cuda") or device == "cuda")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.float16)
        if (amp_enabled and use_cuda)
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    for images, labels, _paths in tqdm(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp on RGB inputs (train only). The teacher target below is
        # computed from the mixed RGB so the prior remains consistent with
        # the input the student actually sees.
        mixup_lam = None
        labels_b = None
        if train and mixup_alpha > 0:
            import numpy as _np
            mixup_lam = float(_np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(images.size(0), device=device)
            images = mixup_lam * images + (1.0 - mixup_lam) * images[perm]
            labels_b = labels[perm]

        # Compute the analytic teacher target in fp32 OUTSIDE autocast.
        # Inside fp16 autocast, torch.quantile in blood_probability silently
        # returned NaN on the 1660 Ti, propagating through BCE → backprop →
        # weights, and the model never trained. Found 2026-04-25.
        if distill_lambda > 0:
            with torch.no_grad():
                rgb01 = _to_rgb01(images.float())
                if prior_version == "v2":
                    from physics_prior import blood_probability_v2
                    target = blood_probability_v2(rgb01.float(),
                                                   alpha=max(physics_alpha, 6.0),
                                                   pivot=pivot_v2,
                                                   lambda_eff=physics_lambda_eff)
                else:
                    target = blood_probability(rgb01.float(),
                                                alpha=physics_alpha,
                                                lambda_eff=physics_lambda_eff)
                target = target.unsqueeze(1)  # [B,1,H,W]
        else:
            target = None

        with torch.set_grad_enabled(train):
            with autocast_ctx:
                out = model(images)
                # In deploy_mode the model returns logits only; otherwise a
                # (logits, pblood_logits) tuple. Test eval calls in deploy mode
                # to bypass the fp16-overflow-prone decoder.
                if isinstance(out, tuple):
                    logits, pblood_logits = out
                else:
                    logits, pblood_logits = out, None
                if mixup_lam is not None:
                    loss_ce = mixup_lam * ce_loss(logits, labels) \
                            + (1.0 - mixup_lam) * ce_loss(logits, labels_b)
                else:
                    loss_ce = ce_loss(logits, labels)
            # BCE is computed in fp32 explicitly — under autocast the
            # logits/target combination can produce inf/NaN on smaller
            # GPUs even with BCEWithLogitsLoss's internal numerics.
            if distill_lambda > 0 and target is not None and pblood_logits is not None:
                target_down = F.adaptive_avg_pool2d(target,
                                                     pblood_logits.shape[-2:]).float()
                # Clamp logits before BCE so a single fp16-overflowed activation
                # cannot poison the whole-batch loss. Found 2026-04-28 — this is
                # what was leaving NaN in the test eval's loss field.
                aux_logits = pblood_logits.float().clamp(-30.0, 30.0)
                loss_aux = bce(aux_logits, target_down)
                loss = loss_ce.float() + distill_lambda * loss_aux
            else:
                loss_aux = torch.tensor(0.0, device=device)
                loss = loss_ce

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        bs = images.size(0)
        running_total += loss.item() * bs
        running_ce += loss_ce.item() * bs
        running_aux += loss_aux.item() * bs
        preds = logits.argmax(dim=1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    metrics = summarize_classification(y_true, y_pred, class_names)
    n = len(loader.dataset)
    metrics["loss"] = running_total / n
    metrics["loss_ce"] = running_ce / n
    metrics["loss_aux"] = running_aux / n
    return metrics


def _worker_init_fn(worker_id: int) -> None:
    import random as _random
    import numpy as _np
    seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    _np.random.seed(seed)
    _random.seed(seed)


def _enable_deterministic() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    _register_gastroscopy_package(args.gastroscopy_code_dir)
    set_seed(args.seed)
    if args.deterministic:
        _enable_deterministic()
    os.makedirs(args.output_dir, exist_ok=True)

    # 3-channel RGB input — use the original (non-physics) transforms
    tf_train = build_transforms(args.image_size, True)
    tf_eval = build_transforms(args.image_size, False)
    print(f"[train_distill] 3-channel RGB input; distill_lambda={args.distill_lambda}, alpha={args.physics_alpha}")

    train_ds = FolderDatasetWithPaths(os.path.join(args.data_dir, "train"), transform=tf_train,
                                       allow_empty=True)
    class_names = train_ds.classes
    from metrics_pi import ensure_class_folders
    ensure_class_folders(os.path.join(args.data_dir, "val"), class_names)
    ensure_class_folders(os.path.join(args.data_dir, "test"), class_names)
    val_ds = FolderDatasetWithPaths(os.path.join(args.data_dir, "val"), transform=tf_eval,
                                     allow_empty=True)
    test_ds = FolderDatasetWithPaths(os.path.join(args.data_dir, "test"), transform=tf_eval,
                                      allow_empty=True)
    assert val_ds.classes == class_names, "val class_to_idx mismatch"
    assert test_ds.classes == class_names, "test class_to_idx mismatch"
    print(f"[train_distill] classes = {class_names}")

    loader_gen = torch.Generator()
    loader_gen.manual_seed(args.seed)
    loader_kwargs = dict(num_workers=args.num_workers,
                         pin_memory=str(args.device).startswith("cuda"),
                         worker_init_fn=_worker_init_fn,
                         generator=loader_gen)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    from models_pi import ImageClassifierPIDistill
    model = ImageClassifierPIDistill(args.model_name, num_classes=len(class_names),
                                      pretrained=args.pretrained).to(args.device)
    print(f"[train_distill] trainable params: {count_trainable_parameters(model):,}")

    if args.init_backbone_from:
        ckpt_path = args.init_backbone_from
        if not os.path.exists(ckpt_path):
            raise SystemExit(f"--init_backbone_from path does not exist: {ckpt_path}")
        stage1 = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        if "backbone_state" not in stage1:
            raise SystemExit(f"--init_backbone_from checkpoint missing key "
                             f"'backbone_state': {ckpt_path}")
        bb = stage1["backbone_state"]
        bb_unprefixed = {k[len("backbone."):]: v for k, v in bb.items()
                         if k.startswith("backbone.")}
        missing, unexpected = model.backbone.load_state_dict(bb_unprefixed, strict=False)
        print(f"[train_distill] loaded Stage-1 backbone from {ckpt_path}: "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        # NOTE: deliberately NOT loading distill_decoder. It gets retrained
        # from random init on real data via the existing CE+BCE loop. This
        # also avoids fp16 overflow on the 1660 Ti when synth-trained
        # decoder activations hit real-image feature maps (observed
        # 2026-05-06: aux loss NaN on epoch 1 with decoder loaded).

    train_labels = [label for _, label in train_ds.samples]
    class_weights = compute_class_weights(train_labels, len(class_names)).to(args.device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights,
                                   label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.label_smoothing > 0 or args.mixup_alpha > 0 or args.early_stopping_patience > 0:
        print(f"[train_distill] regularization: label_smoothing={args.label_smoothing}  "
              f"mixup_alpha={args.mixup_alpha}  early_stopping_patience={args.early_stopping_patience}  "
              f"prior_version={args.physics_prior_version}")

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    amp_enabled = args.mixed_precision and str(args.device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if amp_enabled else None
    print(f"[train_distill] mixed_precision={amp_enabled}  scheduler={args.scheduler}  deterministic={args.deterministic}")

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
        print(f"[train_distill] resumed from {last_path}: start_epoch={start_epoch}, best_val_macro_f1={best_score:.4f}")

    epochs_since_best = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}  lr={optimizer.param_groups[0]['lr']:.2e}")
        train_metrics = run_epoch(model, train_loader, ce_loss, optimizer, args.device, class_names,
                                   True, args.distill_lambda, args.physics_alpha, args.physics_lambda_eff,
                                   scaler=scaler, amp_enabled=amp_enabled,
                                   prior_version=args.physics_prior_version,
                                   pivot_v2=args.physics_pivot_v2,
                                   mixup_alpha=args.mixup_alpha)
        val_metrics = run_epoch(model, val_loader, ce_loss, optimizer, args.device, class_names,
                                 False, args.distill_lambda, args.physics_alpha, args.physics_lambda_eff,
                                 scaler=None, amp_enabled=amp_enabled,
                                 prior_version=args.physics_prior_version,
                                 pivot_v2=args.physics_pivot_v2,
                                 mixup_alpha=0.0)
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(f"Train  total={train_metrics['loss']:.4f}  ce={train_metrics['loss_ce']:.4f}  aux={train_metrics['loss_aux']:.4f}  macro_f1={train_metrics['macro_f1']:.4f}")
        print(f"Val    total={val_metrics['loss']:.4f}  ce={val_metrics['loss_ce']:.4f}  aux={val_metrics['loss_aux']:.4f}  macro_f1={val_metrics['macro_f1']:.4f}")

        # Select on val macro-F1 over evaluable classes (matches §3.4 headline).
        # See train_stage2_pi.py for the rationale; mirrored here 2026-04-28.
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
        save_json({"history": history}, os.path.join(args.output_dir, "training_history.json"))

        if args.early_stopping_patience > 0 and epochs_since_best >= args.early_stopping_patience:
            print(f"[train_distill] early stopping at epoch {epoch}: "
                  f"{epochs_since_best} epochs without improvement on "
                  f"val_macro_f1_evaluable (patience={args.early_stopping_patience})")
            break

    ckpt = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    # Test eval: skip the auxiliary distill loss. We only need test classification
    # metrics here, and computing the BCE under fp16 autocast was producing NaN
    # because pblood_logits could overflow fp16 (>65k) on the GTX 1660 Ti.
    # Setting deploy_mode=True returns class_logits only and short-circuits
    # the decoder; passing distill_lambda=0.0 skips the analytic teacher target.
    model.deploy_mode = True
    test_metrics = run_epoch(model, test_loader, ce_loss, optimizer=None, device=args.device,
                              class_names=class_names, train=False,
                              distill_lambda=0.0,
                              physics_alpha=args.physics_alpha,
                              physics_lambda_eff=args.physics_lambda_eff,
                              scaler=None, amp_enabled=amp_enabled)
    save_json(test_metrics, os.path.join(args.output_dir, "test_metrics.json"))
    print("\nFinal test macro_f1:", test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
