"""Quantify the §5.3 claim: the +PI backbone's Grad-CAM attention
shifts toward regions where the analytic P_blood map has high response.

For each of N test frames per focal class, compute:
  - Grad-CAM saliency from the +PI 5-channel backbone (w.r.t. true-class logit)
  - The analytic P_blood map from the frame's RGB pixels
  - Dice and IoU between the two maps after thresholding (top-quintile for both)

Compares against the RGB-only baseline backbone (which doesn't see P_blood as
input) so the shift can be quantified.

USAGE
    python gradcam_prior_overlap.py
        # uses seed=42 PI + RGB checkpoints (representative)
        # 50 frames per focal class × 6 classes = ~300 forward passes
        # ~5 min on V100

Output:
    GI_project/paper/medIA_submission/docs/supplementary/gradcam_prior_overlap.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

GASTRO = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/gastroscopy_code_package"
CAPSULE = "/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/Capsule-Endoscopy"
TEST_DIR = Path("/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/data/stage2_data/test")
ROOT = Path("/project/BME/Zaman_lab/s248103/GI_outputs/cross_backbone")

ALL_CLASSES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
FOCAL_CLASSES = ["Angiectasia", "Blood - fresh", "Erosion",
                 "Lymphangiectasia", "Pylorus", "Normal clean mucosa"]


def _setup():
    for p in (GASTRO, CAPSULE):
        if p not in sys.path:
            sys.path.insert(0, p)


class GradCAM:
    """Last-conv-layer Grad-CAM. Returns (H, W) saliency in [0, 1]."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, output):
        self.acts = output

    def _bwd(self, _m, grad_in, grad_out):
        self.grads = grad_out[0]

    def __call__(self, image_tensor: torch.Tensor, class_idx: int) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        score = logits[0, class_idx]
        score.backward()
        # weights = mean over spatial dims of gradients
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach()
        cmin, cmax = cam.min(), cam.max()
        if cmax - cmin < 1e-8:
            return torch.zeros_like(cam)
        return (cam - cmin) / (cmax - cmin)


def _build(args_d: dict, device: str):
    use_pi = bool(args_d.get("use_physics_prior", False))
    if use_pi:
        from datasets_pi import build_transforms_pi  # noqa: WPS433
        from models_pi import ImageClassifierPI  # noqa: WPS433
        tf = build_transforms_pi(args_d["image_size"], train=False,
                                 alpha=args_d.get("physics_alpha", 4.0),
                                 lambda_eff=args_d.get("physics_lambda_eff"),
                                 version=args_d.get("physics_prior_version", "v1"),
                                 pivot_v2=args_d.get("physics_pivot_v2", 0.30))
        m = ImageClassifierPI(args_d["model_name"], num_classes=14, pretrained=False).to(device)
    else:
        from datasets import build_transforms  # noqa: WPS433
        from models import ImageClassifier  # noqa: WPS433
        tf = build_transforms(args_d["image_size"], False)
        m = ImageClassifier(args_d["model_name"], num_classes=14, pretrained=False).to(device)
    return m, tf, ("pi" if use_pi else "rgb")


def _analytic_p_blood(rgb_tensor: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    """Re-compute P_blood from a transformed RGB tensor (3, H, W) in [0, 1]
    space. Returns (H, W) map."""
    # Undo ImageNet normalization back to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406], device=rgb_tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=rgb_tensor.device).view(3, 1, 1)
    rgb = (rgb_tensor * std + mean).clamp(0, 1)
    R, G, B = rgb[0], rgb[1], rgb[2]
    H_norm = R / (R + G + B + 1e-8)
    # Percentile clip per-image
    flat = H_norm.flatten()
    q01, q99 = torch.quantile(flat, 0.01), torch.quantile(flat, 0.99)
    H_clip = H_norm.clamp(q01, q99)
    H_resc = (H_clip - q01) / (q99 - q01 + 1e-8)
    P = torch.sigmoid(alpha * (H_resc - 0.5))
    h, w = P.shape
    # Radial fluence
    yy, xx = torch.meshgrid(torch.arange(h, device=P.device), torch.arange(w, device=P.device),
                              indexing="ij")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2.0)
    Phi = 1.0 / (1.0 + 4 * r * r)
    return (P * Phi)


def _dice_iou(map_a: torch.Tensor, map_b: torch.Tensor, top_q: float = 0.8) -> tuple[float, float]:
    """Threshold both maps at their top-quintile and compute Dice + IoU."""
    a_th = torch.quantile(map_a.flatten(), top_q)
    b_th = torch.quantile(map_b.flatten(), top_q)
    A = (map_a >= a_th).float()
    B = (map_b >= b_th).float()
    inter = (A * B).sum().item()
    sumA = A.sum().item(); sumB = B.sum().item()
    union = sumA + sumB - inter
    dice = (2 * inter) / (sumA + sumB) if (sumA + sumB) > 0 else float("nan")
    iou  = inter / union if union > 0 else float("nan")
    return float(dice), float(iou)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_per_class", type=int, default=50,
                    help="Max test frames per focal class (random sample if fewer available).")
    ap.add_argument("--out", type=Path,
                    default="/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/paper/medIA_submission/docs/supplementary/gradcam_prior_overlap.json")
    cli = ap.parse_args()
    _setup()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gradcam] device={device}  seed={cli.seed}")

    results = {"seed": cli.seed, "n_per_class": cli.n_per_class, "focal_classes": FOCAL_CLASSES,
               "rgb": {}, "pi": {}}
    rng = np.random.default_rng(cli.seed)

    for arm in ("rgb", "pi"):
        ckpt_path = ROOT / f"effb0_paper_seed{cli.seed}_{arm}" / "best_model.pt"
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
        model, tf, _ = _build(a, device)
        model.load_state_dict(ck["model_state"]); model.eval()
        # Last-conv-block target: EffB0 .features[-1] is a Conv2dNormActivation
        try:
            target_layer = model.backbone.features[-1]
        except AttributeError:
            target_layer = model.model.features[-1]
        gradcam = GradCAM(model, target_layer)

        for cls in FOCAL_CLASSES:
            cls_idx = ALL_CLASSES.index(cls)
            cls_dir = TEST_DIR / cls
            if not cls_dir.is_dir(): continue
            frames = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in (".jpg", ".png", ".jpeg"))
            if not frames: continue
            sample_idx = rng.choice(len(frames), size=min(cli.n_per_class, len(frames)), replace=False)
            dices, ious = [], []
            for i in sample_idx:
                f = frames[int(i)]
                try:
                    img = Image.open(f).convert("RGB")
                    inp = tf(img).unsqueeze(0).to(device)
                    inp.requires_grad_(True)
                    saliency = gradcam(inp, cls_idx)
                    # Analytic P_blood map at the model's input resolution; use only RGB ch
                    rgb_only = inp[0, :3]
                    p_blood = _analytic_p_blood(rgb_only)
                    dice, iou = _dice_iou(saliency, p_blood, top_q=0.8)
                    dices.append(dice); ious.append(iou)
                except Exception as e:  # noqa: BLE001
                    print(f"  [{arm} {cls}] skip {f.name}: {e}")
            results[arm][cls] = {
                "n_frames": len(dices),
                "dice_mean": float(np.mean(dices)) if dices else float("nan"),
                "dice_sd":   float(np.std(dices)) if len(dices) > 1 else 0.0,
                "iou_mean":  float(np.mean(ious))  if ious  else float("nan"),
                "iou_sd":    float(np.std(ious))  if len(ious)  > 1 else 0.0,
            }
            print(f"  {arm:3s}  {cls:25s}  Dice={results[arm][cls]['dice_mean']:.3f} ± {results[arm][cls]['dice_sd']:.3f}  "
                  f"IoU={results[arm][cls]['iou_mean']:.3f} ± {results[arm][cls]['iou_sd']:.3f}  "
                  f"(n={results[arm][cls]['n_frames']})")

    cli.out.parent.mkdir(parents=True, exist_ok=True)
    cli.out.write_text(json.dumps(results, indent=2))
    print(f"\n[gradcam] wrote {cli.out}")
    # Headline: PI Dice − RGB Dice per class
    print()
    print(f"  Headline: Dice(saliency, P_blood) shift PI − RGB:")
    for cls in FOCAL_CLASSES:
        if cls in results["rgb"] and cls in results["pi"]:
            d_rgb = results["rgb"][cls]["dice_mean"]
            d_pi  = results["pi"][cls]["dice_mean"]
            shift = d_pi - d_rgb
            print(f"    {cls:25s}  RGB {d_rgb:.3f} → PI {d_pi:.3f}  Δ = {shift:+.3f}")


if __name__ == "__main__":
    main()
