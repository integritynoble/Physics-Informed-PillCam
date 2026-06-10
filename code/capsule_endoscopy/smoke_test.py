"""End-to-end smoke test for the 5-channel physics-informed pipeline.

Loads a handful of images from --image_dir, runs them through
    build_transforms_pi  ->  ImageClassifierPI / MultiTaskClassifierPI
and verifies shapes + that a forward pass completes on CPU.

Run from the paper_draft/ directory:

    python smoke_test.py --image_dir "./stage2_data/train/Blood - fresh"
"""
from __future__ import annotations

import argparse
import os
from glob import glob

import torch
from PIL import Image

from datasets_pi import build_transforms_pi
from models_pi import ImageClassifierPI, MultiTaskClassifierPI
from physics_prior import physics_channels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir", type=str, required=True,
                   help="Folder with .png/.jpg images")
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--model_name", type=str, default="resnet18",
                   choices=["resnet18", "resnet50", "efficientnet_b0", "convnext_tiny"])
    p.add_argument("--num_classes", type=int, default=14,
                   help="Kvasir-Capsule has 14 classes")
    p.add_argument("--max_images", type=int, default=4)
    p.add_argument("--pretrained", action="store_true",
                   help="Download ImageNet weights (~50MB). Off by default for a fast smoke test.")
    return p.parse_args()


def main():
    args = parse_args()

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for e in exts:
        paths.extend(glob(os.path.join(args.image_dir, e)))
    paths = sorted(paths)[: args.max_images]
    if not paths:
        raise SystemExit(f"No images found in {args.image_dir}")
    print(f"[smoke] found {len(paths)} images")

    tf = build_transforms_pi(image_size=args.image_size, train=False)
    imgs = [tf(Image.open(p).convert("RGB")) for p in paths]
    batch = torch.stack(imgs, dim=0)
    print(f"[smoke] batch shape = {tuple(batch.shape)}  expected (N, 5, {args.image_size}, {args.image_size})")
    assert batch.shape[1] == 5, "Transform did not emit 5 channels"

    print(f"[smoke] channel-wise mean/std (sanity):")
    for i in range(5):
        ch = batch[:, i]
        print(f"  ch{i}: mean={ch.mean().item():+.3f}  std={ch.std().item():.3f}  min={ch.min().item():+.3f}  max={ch.max().item():+.3f}")

    print(f"[smoke] building ImageClassifierPI({args.model_name}, num_classes={args.num_classes}, pretrained={args.pretrained})")
    model = ImageClassifierPI(args.model_name, num_classes=args.num_classes, pretrained=args.pretrained)
    model.eval()
    with torch.no_grad():
        logits = model(batch)
    print(f"[smoke] logits shape = {tuple(logits.shape)}  expected ({len(paths)}, {args.num_classes})")
    assert logits.shape == (len(paths), args.num_classes)

    print(f"[smoke] building MultiTaskClassifierPI(num_triage=3, num_subtype={args.num_classes})")
    mtl = MultiTaskClassifierPI(args.model_name, num_triage_classes=3, num_subtype_classes=args.num_classes,
                                pretrained=args.pretrained)
    mtl.eval()
    with torch.no_grad():
        triage, subtype = mtl(batch)
    print(f"[smoke] triage shape  = {tuple(triage.shape)}")
    print(f"[smoke] subtype shape = {tuple(subtype.shape)}")

    # verify zero-init invariance: at pretrained=False the model still runs;
    # at pretrained=True the physics channels' contribution at init should be exactly 0
    if args.pretrained:
        baseline_rgb = batch[:, :3]
        pad = torch.zeros_like(batch[:, 3:])
        batch_rgb_only = torch.cat([baseline_rgb, pad], dim=1)
        with torch.no_grad():
            logits_rgb = model(batch_rgb_only)
        max_diff = (logits - logits_rgb).abs().max().item()
        print(f"[smoke] zero-init invariance check: |logits(PI) - logits(RGB+zeros)| max = {max_diff:.2e}")
        print("       (should be ~0 at step 0; nonzero after training = physics channels learned to contribute)")

    print("[smoke] OK")


if __name__ == "__main__":
    main()
