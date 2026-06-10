"""DINOv2 linear probe on Kvasir-Capsule.

A foundation-model baseline for the medIA paper that pre-empts the "did
2024-2025 foundation models obviate task-specific priors?" reviewer
concern. We freeze DINOv2-base (vitb14) ImageNet-pretrained backbone and
train a 768->14 linear classifier head on Kvasir-Capsule under the same
6-seed protocol as the rest of the paper.

Cached features mode: on first run, extract 768-d [CLS] embeddings for
every frame in train/val/test and save to a single .npz. Subsequent runs
re-use the cache (the embeddings depend only on the backbone, not on the
seed) so each per-seed run is just linear-classifier training (~5 min).

USAGE
    # one-time embedding extraction (~1-2 GPU-h on V100)
    python dinov2_linear_probe.py --extract_embeddings \\
        --data_dir /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/data/stage2_data \\
        --out_dir  /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/dinov2_linear

    # per-seed linear probe training (~5 min each on V100, ~10 min on CPU)
    for seed in 41 42 43 44 45 47; do
        python dinov2_linear_probe.py --train \\
            --out_dir /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/dinov2_linear \\
            --seed $seed
    done

Output (per seed under <out_dir>/seed<seed>/):
    test_metrics.json   {macro_auc, per_class_auc, per_class_n_pos,
                         macro_f1_evaluable, accuracy, seed}
    classifier.pt       trained linear head (small; ~50 KB)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, f1_score

# 14 classes, ImageFolder alphabetical order (same as the rest of the paper)
ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
N_CLASSES = len(ALL_CLASSES)

# DINOv2 vitb14: 768-d output via the [CLS] token
DINOV2_VARIANT = "dinov2_vitb14"
EMBED_DIM = 768
IMAGE_SIZE = 224  # 16 * 14, divisible by patch size 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def extract_embeddings(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dinov2] device={device}")
    print(f"[dinov2] loading {DINOV2_VARIANT} via torch.hub")
    backbone = torch.hub.load("facebookresearch/dinov2", DINOV2_VARIANT)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    tf = _build_transform()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "embeddings_cache.npz"
    if cache_path.exists() and not args.rerun:
        print(f"[dinov2] cache already exists: {cache_path}  (use --rerun to overwrite)")
        return

    feats: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[int] = []
    paths: list[str] = []
    SPLIT_IDX = {"train": 0, "val": 1, "test": 2}

    for split_name, idx in SPLIT_IDX.items():
        split_root = Path(args.data_dir) / split_name
        if not split_root.is_dir():
            print(f"[dinov2] WARN: {split_root} missing — skipping")
            continue
        # Some splits have empty class folders (Ampulla, Blood-hematin, and
        # Polyp are train-only by design — see Table 1 of the paper).
        # Materialize all 14 class dirs first so ImageFolder's class_to_idx
        # aligns with ALL_CLASSES, then pass allow_empty=True (torchvision
        # >= 0.20) so empty classes don't trigger FileNotFoundError.
        for c in ALL_CLASSES:
            (split_root / c).mkdir(parents=True, exist_ok=True)
        ds = datasets.ImageFolder(str(split_root), transform=tf, allow_empty=True)
        if ds.classes != ALL_CLASSES:
            raise SystemExit(
                f"split={split_name} class order mismatch: got {ds.classes} "
                f"expected {ALL_CLASSES}"
            )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))
        print(f"[dinov2] extracting {split_name}: {len(ds):,} frames")
        with torch.no_grad():
            for batch_i, (imgs, lab) in enumerate(loader):
                imgs = imgs.to(device, non_blocking=True)
                emb = backbone(imgs)  # (B, 768)
                feats.append(emb.cpu().numpy())
                labels.extend(int(x) for x in lab.numpy())
                splits.extend([idx] * imgs.size(0))
                if batch_i % 20 == 0:
                    print(f"   batch {batch_i}/{len(loader)}")
        for p_, _ in ds.samples:
            paths.append(p_)

    feats_arr = np.concatenate(feats, axis=0).astype(np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    splits_arr = np.asarray(splits, dtype=np.int8)
    np.savez_compressed(
        cache_path,
        embeddings=feats_arr, labels=labels_arr, splits=splits_arr,
        paths=np.asarray(paths),
        classes=np.asarray(ALL_CLASSES),
    )
    print(f"[dinov2] saved {cache_path}  shape={feats_arr.shape}")


def train_linear_probe(args: argparse.Namespace) -> int:
    cache_path = Path(args.out_dir) / "embeddings_cache.npz"
    if not cache_path.exists():
        raise SystemExit(f"missing embeddings cache: {cache_path}. Run with --extract_embeddings first.")
    npz = np.load(cache_path, allow_pickle=True)
    X = npz["embeddings"]
    y = npz["labels"]
    s = npz["splits"]

    Xtr, ytr = X[s == 0], y[s == 0]
    Xva, yva = X[s == 1], y[s == 1]
    Xte, yte = X[s == 2], y[s == 2]
    print(f"[dinov2] sizes:  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")

    # Seed everything
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Inverse-frequency class weights (matches the rest of the paper's recipe)
    class_counts = np.bincount(ytr, minlength=N_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    inv_freq = float(class_counts.sum()) / (N_CLASSES * class_counts)
    class_weight = torch.tensor(inv_freq, dtype=torch.float32, device=device)

    head = nn.Linear(EMBED_DIM, N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(weight=class_weight)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.long, device=device)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    yte_t = torch.tensor(yte, dtype=torch.long, device=device)

    bs = args.batch_size
    best_val_auc = -1.0
    best_state: dict | None = None
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(Xtr_t.size(0), device=device)
        total_loss = 0.0
        for i in range(0, perm.size(0), bs):
            idx = perm[i : i + bs]
            logits = head(Xtr_t[idx])
            loss = crit(logits, ytr_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * idx.size(0)
        sched.step()

        head.eval()
        with torch.no_grad():
            val_probs = F.softmax(head(Xva_t), dim=1).cpu().numpy()
        val_macro_auc = _compute_macro_auc(val_probs, yva)
        if val_macro_auc > best_val_auc:
            best_val_auc = val_macro_auc
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"   ep {ep:3d}  train_loss={total_loss/Xtr_t.size(0):.4f}  val_mAUC={val_macro_auc:.4f}  (best={best_val_auc:.4f})")

    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        test_probs = F.softmax(head(Xte_t), dim=1).cpu().numpy()
    per_class, n_pos = _compute_per_class_auc(test_probs, yte)
    macro_auc = float(np.nanmean([v for v in per_class.values() if v is not None]))
    pred = test_probs.argmax(axis=1)
    f1_eval_classes = [c for c, n in n_pos.items() if n > 0]
    f1_eval_indices = [ALL_CLASSES.index(c) for c in f1_eval_classes]
    mask = np.isin(yte, f1_eval_indices)
    if mask.any():
        macro_f1_eval = float(f1_score(yte[mask], pred[mask], average="macro", zero_division=0,
                                        labels=f1_eval_indices))
    else:
        macro_f1_eval = float("nan")
    accuracy = float((pred == yte).mean())

    out = {
        "macro_auc": macro_auc,
        "per_class_auc": per_class,
        "per_class_n_positive": n_pos,
        "macro_f1_evaluable": macro_f1_eval,
        "accuracy": accuracy,
        "seed": int(args.seed),
        "model_name": DINOV2_VARIANT,
        "method": "linear_probe_frozen_backbone",
        "n_test_frames": int(len(yte)),
        "best_val_macro_auc": float(best_val_auc),
    }
    seed_dir = Path(args.out_dir) / f"seed{args.seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "test_metrics.json").write_text(json.dumps(out, indent=2))
    torch.save(head.state_dict(), seed_dir / "classifier.pt")
    print(f"[dinov2] seed {args.seed}  test macro-AUC = {macro_auc:.4f}")
    print(f"[dinov2] wrote {seed_dir/'test_metrics.json'}")
    return 0


def _compute_macro_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    aucs = []
    for c in range(N_CLASSES):
        y_bin = (labels == c).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            try:
                aucs.append(roc_auc_score(y_bin, probs[:, c]))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def _compute_per_class_auc(probs: np.ndarray, labels: np.ndarray):
    per_class: dict[str, float | None] = {}
    n_pos: dict[str, int] = {}
    for c, name in enumerate(ALL_CLASSES):
        y_bin = (labels == c).astype(int)
        n = int(y_bin.sum())
        n_pos[name] = n
        if n == 0 or n == len(y_bin):
            per_class[name] = None
        else:
            try:
                per_class[name] = float(roc_auc_score(y_bin, probs[:, c]))
            except ValueError:
                per_class[name] = None
    return per_class, n_pos


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract_embeddings", action="store_true",
                    help="Run feature extraction over train/val/test and cache to embeddings_cache.npz.")
    ap.add_argument("--train", action="store_true",
                    help="Train a linear classifier on the cached embeddings for one seed.")
    ap.add_argument("--data_dir", type=Path, default=None,
                    help="Kvasir stage2_data root (with train/val/test subdirs). Required for --extract_embeddings.")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=512,
                    help="Linear-classifier mini-batch (in feature space). Embedding-extraction uses 32.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--rerun", action="store_true",
                    help="Re-extract embeddings even if the cache exists.")
    args = ap.parse_args()

    if args.extract_embeddings:
        if args.data_dir is None:
            raise SystemExit("--data_dir is required for --extract_embeddings")
        args.batch_size = 32  # smaller during extraction to fit GPU
        extract_embeddings(args)
        if args.train:
            args.batch_size = 512
            return train_linear_probe(args)
        return 0
    elif args.train:
        return train_linear_probe(args)
    else:
        raise SystemExit("pass either --extract_embeddings or --train")


if __name__ == "__main__":
    sys.exit(main())
