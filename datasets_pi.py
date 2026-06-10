"""Physics-informed dataset transforms.

Wraps the usual augmentation pipeline, but the final tensor is 5-channel:
    channels 0-2 : ImageNet-normalized RGB
    channel  3   : P_blood (Monte Carlo–informed hemoglobin probability)
    channel  4   : H_AFI_weighted (green/blue AFI surrogate x fluence)

Usage:
    from datasets_pi import build_transforms_pi
    train_tf = build_transforms_pi(image_size=224, train=True)
    ds = datasets.ImageFolder(train_dir, transform=train_tf)
"""
from __future__ import annotations

import torch
from PIL import Image
from torchvision import transforms

from physics_prior import physics_channels

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _rgb01_to_5ch(img01: torch.Tensor,
                  alpha: float = 4.0,
                  lambda_eff: float | None = None,
                  version: str = "v1",
                  pivot_v2: float = 0.30) -> torch.Tensor:
    """img01: [3,H,W] in [0,1]. Returns [5,H,W]."""
    rgb_norm = (img01 - IMAGENET_MEAN) / IMAGENET_STD
    phys = physics_channels(img01.unsqueeze(0),
                            alpha=alpha, lambda_eff=lambda_eff,
                            version=version, pivot_v2=pivot_v2).squeeze(0)
    return torch.cat([rgb_norm, phys], dim=0)


class _PhysicsTransform:
    """Picklable transform that emits a 5-channel tensor.

    A top-level callable (not a closure) is required so DataLoader workers
    on Windows can pickle the transform across the spawn-based IPC. The
    earlier closure-based implementation crashed config 2 of the Week-2
    sweep with `Can't pickle local object 'build_transforms_pi.<locals>._pipeline'`.
    """

    def __init__(self, geom, alpha: float, lambda_eff: float | None,
                 version: str = "v1", pivot_v2: float = 0.30):
        self.geom = geom
        self.alpha = alpha
        self.lambda_eff = lambda_eff
        self.version = version
        self.pivot_v2 = pivot_v2

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img01 = self.geom(img)
        return _rgb01_to_5ch(img01, alpha=self.alpha, lambda_eff=self.lambda_eff,
                              version=self.version, pivot_v2=self.pivot_v2)


def build_transforms_pi(image_size: int = 224, train: bool = True,
                        alpha: float = 4.0,
                        lambda_eff: float | None = None,
                        version: str = "v1",
                        pivot_v2: float = 0.30):
    """Transform that emits a 5-channel tensor (3 ImageNet-norm RGB + 2 physics).

    version: 'v1' (paper-draft default) or 'v2' (scale-fixed P_blood; see
        physics_prior.blood_probability_v2). Use v2 for new training runs.
    """
    if train:
        geom = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
        ])
    else:
        geom = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    return _PhysicsTransform(geom, alpha=alpha, lambda_eff=lambda_eff,
                              version=version, pivot_v2=pivot_v2)
