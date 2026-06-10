"""Evaluate a trained checkpoint on the test split and dump per-frame predictions.

Loads best_model.pt produced by train_stage2_pi.py or train_stage2_pi_distill.py,
runs inference on stage2_data/test/, and saves a JSON with:
    {
      "config": {...args from checkpoint...},
      "class_names": [...],
      "labels": [int, ...],
      "preds":  [int, ...],
      "probs":  [[float; num_classes], ...],
      "paths":  [str, ...]   # absolute paths so video_id can be inferred for F5
    }

The downstream figure scripts (make_fig3_roc, make_fig4_cm, make_fig5_fp_per_kframe)
read this JSON. Decoupling inference from plotting means we only run the GPU
once per checkpoint regardless of how many figures we want.

Usage:
    python eval_test_predictions.py \
      --ckpt   D:/kvasir_capsule/outputs/stage2_rgb_effb0/best_model.pt \
      --data_dir D:/kvasir_capsule/stage2_data \
      --variant rgb \
      --out    D:/kvasir_capsule/outputs/stage2_rgb_effb0/test_predictions.json \
      --gastroscopy_code_dir "D:/onedrive/UT_southwestern/GIproject/Dr. Zaman/gastroscopy_code_package (2)/gastroscopy_code_package"

    --variant in {rgb, pi, distill} selects the model + transform pipeline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to best_model.pt")
    p.add_argument("--data_dir", required=True, help="Path to stage2_data/")
    p.add_argument("--variant", required=True, choices=["rgb", "pi", "distill"])
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--gastroscopy_code_dir", required=True,
                   help="Path to gastroscopy_code_package (for datasets.py)")
    p.add_argument("--ablate_input_channels", type=str, default="",
                   help="Comma-separated input-channel indices to zero before the forward "
                        "pass, matching the trainer's --ablate_input_channels semantics. "
                        "If unset, defaults to the value stored in the checkpoint args.")
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, os.path.abspath(args.gastroscopy_code_dir))
    from datasets import FolderDatasetWithPaths, build_transforms

    # Add the paper_draft folder to path so models_pi / metrics_pi / datasets_pi import.
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)  # paper_draft/
    sys.path.insert(0, parent)
    from metrics_pi import ensure_class_folders

    # Load checkpoint
    print(f"[eval] loading {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model_args = ckpt["args"]
    class_names = ckpt["class_names"]
    print(f"[eval] checkpoint trained with model_name={model_args.get('model_name')}, "
          f"epochs={model_args.get('epochs')}, "
          f"best_val_macro_f1={ckpt.get('best_val_macro_f1', 'n/a')}")
    print(f"[eval] {len(class_names)} classes")

    # Build transforms + model per variant
    if args.variant == "rgb":
        from models import ImageClassifier
        tf = build_transforms(args.image_size, train=False)
        model = ImageClassifier(model_args["model_name"], num_classes=len(class_names),
                                 pretrained=False).to(args.device)
    elif args.variant == "pi":
        from datasets_pi import build_transforms_pi
        from models_pi import ImageClassifierPI
        tf = build_transforms_pi(args.image_size, train=False,
                                  alpha=model_args.get("physics_alpha", 4.0),
                                  lambda_eff=model_args.get("physics_lambda_eff"))
        model = ImageClassifierPI(model_args["model_name"], num_classes=len(class_names),
                                   pretrained=False).to(args.device)
    elif args.variant == "distill":
        from models_pi import ImageClassifierPIDistill
        tf = build_transforms(args.image_size, train=False)
        model = ImageClassifierPIDistill(model_args["model_name"], num_classes=len(class_names),
                                          pretrained=False).to(args.device)
    else:
        raise ValueError(args.variant)

    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # For distill, the auxiliary decoder is training-only; skip it at inference
    # so the deployed model takes plain 3-channel RGB and pays no decoder cost.
    if args.variant == "distill":
        model.deploy_mode = True

    # Ensure test/ has all class folders so class_to_idx aligns to train
    test_dir = os.path.join(args.data_dir, "test")
    ensure_class_folders(test_dir, class_names)
    test_ds = FolderDatasetWithPaths(test_dir, transform=tf, allow_empty=True)
    assert test_ds.classes == class_names, "test class_to_idx mismatch"
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    print(f"[eval] {len(test_ds)} test frames")

    # Resolve channel ablation: CLI flag overrides; otherwise inherit from
    # checkpoint args so eval reproduces the trainer's final test pass.
    if args.ablate_input_channels.strip():
        ablate_idx = [int(c) for c in args.ablate_input_channels.split(",") if c.strip()]
    else:
        ck_abl = (model_args or {}).get("ablate_input_channels", "")
        ablate_idx = [int(c) for c in str(ck_abl).split(",") if str(c).strip()] if ck_abl else []
    if ablate_idx:
        print(f"[eval] ablating input channels: {ablate_idx}")

    labels_all, preds_all, probs_all, paths_all = [], [], [], []
    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader):
            images = images.to(args.device)
            if ablate_idx:
                images = images.clone()
                for c in ablate_idx:
                    images[:, c] = 0.0
            out = model(images)
            logits = out[0] if isinstance(out, tuple) else out
            probs = F.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            labels_all.extend(labels.cpu().tolist())
            preds_all.extend(preds.cpu().tolist())
            probs_all.extend(probs.cpu().tolist())
            paths_all.extend(list(paths))

    out_obj = {
        "config": model_args,
        "variant": args.variant,
        "class_names": class_names,
        "labels": labels_all,
        "preds": preds_all,
        "probs": probs_all,
        "paths": paths_all,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f)
    print(f"[eval] wrote {args.out}  ({len(labels_all)} predictions)")


if __name__ == "__main__":
    main()
