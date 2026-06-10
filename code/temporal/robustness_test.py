"""
Robustness test: does the parameterization-mechanism boundary
survive realistic perturbations of the test images?
======================================================================

Tests whether the cell (b+) lift over cell (b) on Kvasir-Capsule
holds when test images are perturbed via:
  - JPEG compression (quality 80, 50, 25)
  - Gaussian noise (sigma 0.02, 0.05, 0.10 in [0,1] space)
  - Brightness shift (-20%, +20%)

For each perturbation, forward the perturbed test images through
the RGB-only and +PI 5-channel backbones (seed 42 for both),
compute per-class AUC, and report the cell-b vs cell-(b+) lift
gap relative to the unperturbed baseline.

If the boundary's lift is robust, the cell-(b+) lift should
remain positive under all perturbations. If the lift collapses
under modest perturbation, the result is fragile and that should
be reported.

Compute: ~30-60 min on GTX 1660 Ti for 7 perturbation conditions
on the test split (6,423 frames each).

Output:
  paper/nature-machine-intelligence/docs/robustness_test_report.md
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent.parent
CAPSULE = REPO_ROOT / "paper" / "Capsule-Endoscopy"
GASTRO = Path("D:/onedrive/UT_southwestern/GIproject/Dr. Zaman/"
              "gastroscopy_code_package (2)/gastroscopy_code_package")
sys.path.insert(0, str(CAPSULE))
sys.path.insert(0, str(GASTRO))

DATA_ROOT = Path("D:/kvasir_capsule/stage2_data")
RGB_OUT = Path("D:/kvasir_capsule/outputs/stage2_rgb_effb0")
PI_OUT = Path("D:/kvasir_capsule/outputs/stage2_pi_effb0")
REPORT_DIR = HERE.parent.parent / "docs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
IMAGE_SIZE = 224

CLASS_NAMES = [
    "Ampulla of Vater", "Angiectasia", "Blood - fresh", "Blood - hematin",
    "Erosion", "Erythema", "Foreign Body", "Ileocecal valve",
    "Lymphangiectasia", "Normal clean mucosa", "Polyp", "Pylorus",
    "Reduced Mucosal View", "Ulcer",
]
N_CLASSES = len(CLASS_NAMES)

PERTURBATIONS = [
    ("baseline", "none", None),
    ("jpeg_q80", "jpeg", 80),
    ("jpeg_q50", "jpeg", 50),
    ("jpeg_q25", "jpeg", 25),
    ("noise_0.02", "noise", 0.02),
    ("noise_0.05", "noise", 0.05),
    ("noise_0.10", "noise", 0.10),
    ("brightness_-20", "brightness", -0.20),
    ("brightness_+20", "brightness", 0.20),
]


def perturb_image(pil_img, kind: str, param):
    from PIL import Image
    if kind == "none":
        return pil_img
    if kind == "jpeg":
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=int(param))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    if kind == "noise":
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        noise = np.random.normal(0, param, size=arr.shape).astype(np.float32)
        arr = np.clip(arr + noise, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    if kind == "brightness":
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        arr = np.clip(arr * (1.0 + param), 0.0, 1.0)
        return Image.fromarray((arr * 255).astype(np.uint8))
    raise ValueError(f"unknown perturbation: {kind}")


class TestSetWithPerturbation(Dataset):
    def __init__(self, kind: str, param, image_size: int = IMAGE_SIZE):
        self.kind = kind
        self.param = param
        self.image_size = image_size
        self.samples: List[Tuple[Path, int]] = []
        sd = DATA_ROOT / "test"
        for cd in sorted(sd.iterdir()):
            if not cd.is_dir() or cd.name not in CLASS_NAMES:
                continue
            cidx = CLASS_NAMES.index(cd.name)
            for f in cd.iterdir():
                if f.suffix.lower() == ".jpg":
                    self.samples.append((f, cidx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        from PIL import Image
        from torchvision import transforms
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        img = perturb_image(img, self.kind, self.param)
        tfm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size),
                                 interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
        ])
        return tfm(img).float(), label


def load_models():
    from models import ImageClassifier
    from models_pi import ImageClassifierPI

    rgb_ckpt = torch.load(RGB_OUT / "best_model.pt", map_location=DEVICE,
                              weights_only=False)
    rgb_args = rgb_ckpt["args"]
    rgb_model = ImageClassifier(rgb_args["model_name"],
                                  num_classes=len(rgb_ckpt["class_names"]),
                                  pretrained=False).to(DEVICE)
    rgb_model.load_state_dict(rgb_ckpt["model_state"], strict=True)
    rgb_model.eval()

    pi_ckpt = torch.load(PI_OUT / "best_model.pt", map_location=DEVICE,
                              weights_only=False)
    pi_args = pi_ckpt["args"]
    pi_model = ImageClassifierPI(pi_args["model_name"],
                                    num_classes=len(pi_ckpt["class_names"]),
                                    pretrained=False,
                                    extra_channels=int(pi_args.get("extra_channels", 2))
                                    ).to(DEVICE)
    pi_model.load_state_dict(pi_ckpt["model_state"], strict=True)
    pi_model.eval()
    return rgb_model, pi_model, pi_args


def unnormalize(x_norm):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_norm.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_norm.device)
    return (x_norm * std + mean).clamp(0.0, 1.0)


def evaluate_models(rgb_model, pi_model, pi_args, kind: str, param):
    from physics_prior import physics_channels
    from sklearn.metrics import roc_auc_score
    ds = TestSetWithPerturbation(kind, param)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
    all_rgb_logits, all_pi_logits, all_labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            rgb_logits = rgb_model(x)
            x_rgb = unnormalize(x.float())
            phys = physics_channels(x_rgb,
                                      alpha=pi_args.get("physics_alpha", 4.0),
                                      lambda_eff=pi_args.get("physics_lambda_eff", None),
                                      version=pi_args.get("physics_prior_version", "v1"),
                                      pivot_v2=pi_args.get("physics_pivot_v2", 0.30))
            x5 = torch.cat([x, phys], dim=1)
            pi_logits = pi_model(x5)
            all_rgb_logits.append(rgb_logits.cpu().numpy())
            all_pi_logits.append(pi_logits.cpu().numpy())
            all_labels.append(y.numpy())
    rgb_logits = np.concatenate(all_rgb_logits, axis=0)
    pi_logits = np.concatenate(all_pi_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    rgb_probs = torch.softmax(torch.from_numpy(rgb_logits), dim=-1).numpy()
    pi_probs = torch.softmax(torch.from_numpy(pi_logits), dim=-1).numpy()

    rgb_aucs = []; pi_aucs = []
    for j in range(N_CLASSES):
        y_bin = (labels == j).astype(np.int32)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        rgb_aucs.append(roc_auc_score(y_bin, rgb_probs[:, j]))
        pi_aucs.append(roc_auc_score(y_bin, pi_probs[:, j]))
    return float(np.mean(rgb_aucs)), float(np.mean(pi_aucs))


def main():
    print(f"[robust] device={DEVICE}")
    rgb_model, pi_model, pi_args = load_models()

    np.random.seed(42)
    torch.manual_seed(42)

    results = []
    t0 = time.time()
    for label, kind, param in PERTURBATIONS:
        print(f"\n[robust] {label} (kind={kind}, param={param})")
        rgb_auc, pi_auc = evaluate_models(rgb_model, pi_model, pi_args, kind, param)
        lift = pi_auc - rgb_auc
        print(f"  RGB macro-AUC = {rgb_auc:.4f}")
        print(f"  +PI macro-AUC = {pi_auc:.4f}")
        print(f"  +PI lift      = {lift:+.4f}")
        results.append({
            "perturbation": label,
            "kind": kind,
            "param": param,
            "rgb_auc": rgb_auc,
            "pi_auc": pi_auc,
            "lift": lift,
        })

    md = []
    md.append("# Robustness test: cell (b+) input-fusion lift under "
              "test-image perturbation\n")
    md.append("**Date:** 2026-05-08")
    md.append("**Backbones:** seed 42 only (single-seed sanity check)")
    md.append("**Test set:** Kvasir-Capsule test split (~6,423 frames)")
    md.append("**Reported metric:** per-class AUC averaged over 14 classes "
              "(macro-AUC), at the per-frame backbone output (no temporal "
              "head).")
    md.append("")
    md.append("## Results\n")
    md.append("| Perturbation | RGB macro-AUC | +PI macro-AUC | +PI lift |")
    md.append("|---|---:|---:|---:|")
    baseline_lift = results[0]["lift"]
    for r in results:
        md.append(f"| {r['perturbation']} | {r['rgb_auc']:.4f} "
                  f"| {r['pi_auc']:.4f} | {r['lift']:+.4f} |")
    md.append("")
    md.append("## Robustness verdict\n")
    md.append("The +PI lift is robust if it remains positive under all "
              "perturbations and degrades gracefully (rather than "
              "catastrophically) at higher perturbation strengths.")
    md.append("")
    n_positive = sum(1 for r in results if r["lift"] > 0.005)
    md.append(f"**+PI lift > 0.005 holds at {n_positive} of {len(results)} "
              f"perturbation conditions** (including baseline). Largest "
              f"perturbations tested: JPEG q=25 (heavy compression artifacts), "
              f"Gaussian noise sigma=0.10 (substantial pixel noise), "
              f"brightness +/-20% (substantial illumination shift).")
    if n_positive == len(results):
        md.append(f"\n**Conclusion: the +PI lift is robust to all tested "
                  f"perturbations.** This addresses the reviewer concern "
                  f"that the cell (b+) result might depend on test-time "
                  f"image fidelity.")
    elif n_positive >= len(results) - 2:
        md.append(f"\n**Conclusion: the +PI lift is mostly robust** "
                  f"({n_positive}/{len(results)}) but degrades at the "
                  f"strongest perturbations. The lift is reliable under "
                  f"realistic clinical-quality images; very heavy "
                  f"degradations may erase it.")
    else:
        md.append(f"\n**Conclusion: the +PI lift is fragile** under "
                  f"perturbation. Only {n_positive}/{len(results)} "
                  f"perturbations preserve the lift. This is a reportable "
                  f"limitation.")
    md.append(f"\nTotal compute: {(time.time() - t0)/60:.1f} min")

    out = REPORT_DIR / "robustness_test_report.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\n[robust] report -> {out}")


if __name__ == "__main__":
    main()
