"""BiomedCLIP linear probe on Kvasir-Capsule.

Medical-domain foundation-model baseline. Companion to dinov2_linear_probe.py.
DINOv2-base lost to our +PI by 0.12 macro-AUC; this checks whether a
medical-domain pretrained model closes that gap.

We use open_clip's `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
image encoder (ViT-B/16, pretrained on PubMed image-text pairs by Zhang
et al. 2023, "Large-Scale Domain-Specific Pretraining for Biomedical
Vision-Language Processing"). The image encoder produces 512-d projected
embeddings.

Single-script flow: extract features once, train 6-seed linear probes,
write per-seed test_metrics.json + an aggregated summary.

USAGE
    python biomedclip_linear_probe.py \\
        --data_dir /project/BME/Zaman_lab/s248103/stage2_data_canonical \\
        --out_dir  /home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/biomedclip_linear

Embeddings are cached in <out_dir>/embeddings.npz (~150 MB) and re-used
across seeds.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import roc_auc_score, f1_score

ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
N_CLASSES = len(ALL_CLASSES)
TRAINING_ONLY = {"Ampulla of Vater", "Blood - hematin", "Polyp"}
EVALUABLE_IDX = [i for i, c in enumerate(ALL_CLASSES) if c not in TRAINING_ONLY]

HF_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
IMAGE_SIZE = 224


def ensure_class_folders(root: Path) -> None:
    for c in ALL_CLASSES:
        (root / c).mkdir(parents=True, exist_ok=True)


def extract_split(model, preprocess, split_dir: Path, device: str, batch_size: int = 64):
    """Run BiomedCLIP image encoder on all frames in split_dir; return (X, y, paths)."""
    ds = datasets.ImageFolder(str(split_dir), transform=preprocess, allow_empty=True)
    # ds.classes may differ in ordering — assert it matches ALL_CLASSES
    if ds.classes != ALL_CLASSES:
        raise RuntimeError(f"class mismatch in {split_dir}: {ds.classes}")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=(device == "cuda"))
    feats, labels = [], []
    print(f"  [{split_dir.name}] extracting {len(ds)} frames")
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            # BiomedCLIP's image encoder returns projected features when using
            # `model.encode_image(imgs)` (size 512 by default).
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                emb = model.encode_image(imgs)
            feats.append(emb.float().cpu().numpy())
            labels.append(lbls.numpy())
            if (i + 1) % 50 == 0:
                print(f"    batch {i+1}/{len(loader)}  ({time.time()-t0:.1f}s)")
    return np.concatenate(feats), np.concatenate(labels)


def train_linear(X_tr, y_tr, X_val, y_val, device: str, seed: int,
                 lr: float = 5e-4, epochs: int = 80, batch_size: int = 256,
                 weight_decay: float = 1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    embed_dim = X_tr.shape[1]
    head = nn.Linear(embed_dim, N_CLASSES).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    n = X_tr.shape[0]
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    best_val_f1, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        idx = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            j = idx[i:i+batch_size]
            logits = head(X_tr_t[j])
            loss = F.cross_entropy(logits, y_tr_t[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(X_val_t)
            preds = val_logits.argmax(dim=1).cpu().numpy()
        # macro-F1 over labels present in val
        present = sorted(set(int(y) for y in y_val))
        f1 = f1_score(y_val, preds, labels=present, average="macro", zero_division=0)
        if f1 > best_val_f1:
            best_val_f1, best_state = f1, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best_val_f1


def evaluate(head, X_te, y_te, device: str):
    head.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = head(X_te_t)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    per_class_auc = {}
    for i, name in enumerate(ALL_CLASSES):
        y_true = (y_te == i).astype(int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            per_class_auc[name] = None
            continue
        per_class_auc[name] = float(roc_auc_score(y_true, probs[:, i]))
    evaluable = [per_class_auc[ALL_CLASSES[i]] for i in EVALUABLE_IDX if per_class_auc[ALL_CLASSES[i]] is not None]
    macro_auc_evaluable = float(np.mean(evaluable)) if evaluable else None
    acc = float((preds == y_te).mean())
    return {
        "macro_auc_evaluable": macro_auc_evaluable,
        "per_class_auc": per_class_auc,
        "accuracy": acc,
        "n_test_frames": int(len(y_te)),
        "n_evaluable_classes_with_signal": len(evaluable),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43, 44, 45, 47])
    ap.add_argument("--force_extract", action="store_true")
    cli = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(cli.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[biomedclip] device={device}  out={out}")

    cache = out / "embeddings.npz"
    if not cache.exists() or cli.force_extract:
        print(f"[biomedclip] loading {HF_MODEL}")
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(HF_MODEL)
        model.eval().to(device)
        # Ensure each split folder has all 14 class subfolders so ImageFolder works
        for split in ("train", "val", "test"):
            ensure_class_folders(Path(cli.data_dir) / split)
        data = {}
        for split in ("train", "val", "test"):
            X, y = extract_split(model, preprocess, Path(cli.data_dir) / split, device)
            data[f"{split}_X"] = X
            data[f"{split}_y"] = y
            print(f"  {split}: X.shape={X.shape}  y.unique={np.unique(y, return_counts=True)}")
        np.savez_compressed(cache, **data)
        print(f"[biomedclip] saved {cache}")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None
    else:
        print(f"[biomedclip] re-using {cache}")
    data = np.load(cache)
    X_tr, y_tr = data["train_X"], data["train_y"]
    X_va, y_va = data["val_X"],   data["val_y"]
    X_te, y_te = data["test_X"],  data["test_y"]
    print(f"[biomedclip] embeddings: train={X_tr.shape} val={X_va.shape} test={X_te.shape}")

    aucs = []
    for seed in cli.seeds:
        seed_dir = out / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        head, val_f1 = train_linear(X_tr, y_tr, X_va, y_va, device, seed)
        metrics = evaluate(head, X_te, y_te, device)
        metrics["seed"] = int(seed)
        metrics["best_val_f1"] = float(val_f1)
        (seed_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
        torch.save(head.state_dict(), seed_dir / "classifier.pt")
        print(f"  seed {seed}: macro_auc_evaluable = {metrics['macro_auc_evaluable']:.4f}")
        aucs.append(metrics["macro_auc_evaluable"])

    print()
    print(f"=== AGGREGATE (n={len(aucs)}) ===")
    print(f"  mean macro_auc_evaluable = {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}")
    print(f"  per-seed: {[f'{a:.4f}' for a in aucs]}")
    (out / "aggregate.json").write_text(json.dumps({
        "mean": float(np.mean(aucs)),
        "std":  float(np.std(aucs, ddof=1)),
        "per_seed": {str(s): a for s, a in zip(cli.seeds, aucs)},
        "model": HF_MODEL,
    }, indent=2))


if __name__ == "__main__":
    main()
