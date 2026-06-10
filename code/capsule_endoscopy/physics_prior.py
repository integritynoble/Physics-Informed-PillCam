"""Monte Carlo–informed physics priors for capsule endoscopy.

Implements the two-channel physics prior described in the paper outline:
    - P_blood(x,y) = sigmoid(alpha * (H_norm - 0.5)) * Phi(r)
      where H(x,y) = R / (G + B + eps) is the hemoglobin-sensitive index
      and Phi(r) = exp(-r / lambda_eff) is the radial fluence model.
    - H_AFI_weighted(x,y) = log((I_g + eps) / (I_B + eps)) * Phi(r)
      an RGB surrogate for the green-vs-violet AFI log-ratio.

All functions are pure, batched, and autograd-friendly.
"""
from __future__ import annotations

import torch

EPS = 1e-6


def hemoglobin_index(rgb01: torch.Tensor) -> torch.Tensor:
    """H(x,y) = R / (G + B + eps).

    rgb01: [B,3,H,W] in [0,1].
    Returns: [B,H,W].
    """
    r = rgb01[:, 0]
    g = rgb01[:, 1]
    b = rgb01[:, 2]
    return r / (g + b + EPS)


def afi_log_ratio(rgb01: torch.Tensor) -> torch.Tensor:
    """RGB surrogate for the AFI log-ratio, using blue as the excitation proxy.

    H_AFI = log((I_g + eps) / (I_B + eps)).

    rgb01: [B,3,H,W] in [0,1].
    Returns: [B,H,W].
    """
    g = rgb01[:, 1]
    b = rgb01[:, 2]
    return torch.log((g + EPS) / (b + EPS))


def fluence_map(height: int, width: int, lambda_eff: float | None = None,
                device: torch.device | None = None,
                dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Phi(r) = exp(-r / lambda_eff), centered at the image center.

    If lambda_eff is None, defaults to 0.25 * image diagonal
    (moderate falloff that keeps the center ~1 and the corners ~0.02).

    Returns: [H,W].
    """
    if lambda_eff is None:
        lambda_eff = 0.25 * ((height ** 2 + width ** 2) ** 0.5)
    ys = torch.arange(height, device=device, dtype=dtype).view(-1, 1)
    xs = torch.arange(width, device=device, dtype=dtype).view(1, -1)
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    r = torch.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    return torch.exp(-r / lambda_eff)


def blood_probability(rgb01: torch.Tensor, alpha: float = 4.0,
                      lambda_eff: float | None = None,
                      clip_pct: float = 0.01) -> torch.Tensor:
    """P_blood(x,y) = sigmoid(alpha * (H_norm - 0.5)) * Phi(r).

    H_norm is robustly normalized per-image to the [clip_pct, 1-clip_pct]
    quantile range so vignette dark pixels and saturated specular highlights
    do not collapse the dynamic range. alpha controls sharpness of the
    hemoglobin sigmoid; 3-5 is a reasonable range.

    rgb01: [B,3,H,W] in [0,1].
    Returns: [B,H,W] in (0,1).
    """
    h_idx = hemoglobin_index(rgb01)
    flat = h_idx.flatten(start_dim=-2)
    h_lo = torch.quantile(flat, clip_pct, dim=-1).view(-1, 1, 1)
    h_hi = torch.quantile(flat, 1.0 - clip_pct, dim=-1).view(-1, 1, 1)
    h_clipped = torch.maximum(torch.minimum(h_idx, h_hi), h_lo)
    h_norm = (h_clipped - h_lo) / (h_hi - h_lo + EPS)
    sig = torch.sigmoid(alpha * (h_norm - 0.5))
    phi = fluence_map(rgb01.shape[-2], rgb01.shape[-1],
                      lambda_eff=lambda_eff,
                      device=rgb01.device, dtype=rgb01.dtype)
    return sig * phi


def normalized_red_excess(rgb01: torch.Tensor) -> torch.Tensor:
    """NDVI-style red-vs-green index: H_v2 = (R - G) / (R + G + eps).

    Bounded in [-1, +1], scale-invariant within a pixel, no per-image
    normalization. White / neutral mucosa → ~0; fresh blood → ~+0.5–0.8;
    bilious or greenish content → negative. Pure function, autograd-friendly.
    """
    r = rgb01[:, 0]
    g = rgb01[:, 1]
    return (r - g) / (r + g + EPS)


def blood_probability_v2(rgb01: torch.Tensor, alpha: float = 6.0,
                          pivot: float = 0.30,
                          lambda_eff: float | None = None) -> torch.Tensor:
    """Scale-fixed P_blood: sigmoid(alpha * (NDVI_red - pivot)) * Phi(r).

    Why this exists (added 2026-04-28): the original `blood_probability` does
    per-image quantile clipping + min-max normalization, which forces every
    P_blood map to span (≈0, ≈1) regardless of whether blood is actually
    present in the frame. On Kvasir-Capsule this made the +P_blood-only
    ablation arm score 0.613 AUC on Blood-fresh vs the RGB-only baseline at
    0.730 — i.e. the prior was actively *hurting* the bleeding class.

    The v2 form drops per-image normalization. NDVI_red = (R - G) / (R + G)
    is naturally bounded and has a fixed zero (R = G means no red excess).
    The default pivot (0.30) sits between typical pink mucosa (~0.17) and
    fresh-blood pixels (~0.5–0.8), and alpha=6 gives the sigmoid a sharp but
    not saturating transition. Pivot/alpha can be calibrated per-cohort.

    rgb01: [B,3,H,W] in [0,1]. Returns: [B,H,W] in (0,1).
    """
    h = normalized_red_excess(rgb01)
    sig = torch.sigmoid(alpha * (h - pivot))
    phi = fluence_map(rgb01.shape[-2], rgb01.shape[-1],
                      lambda_eff=lambda_eff,
                      device=rgb01.device, dtype=rgb01.dtype)
    return sig * phi


def physics_channels(rgb01: torch.Tensor, alpha: float = 4.0,
                     lambda_eff: float | None = None,
                     version: str = "v1",
                     pivot_v2: float = 0.30) -> torch.Tensor:
    """Two-channel physics prior for concatenation with RGB input.

    Channel 0: P_blood (hemoglobin probability weighted by radial fluence).
    Channel 1: H_AFI_weighted (green/blue log-ratio weighted by radial fluence).

    version="v1" (default; back-compat with paper §3.3 as drafted): per-image
        quantile-normalized hemoglobin index passed through sigmoid(alpha · (H-0.5)).
    version="v2" (added 2026-04-28): scale-fixed NDVI_red index passed
        through sigmoid(alpha · (NDVI_red - pivot)). Use this for new training
        runs targeting the Diagnostics submission — see `blood_probability_v2`.

    rgb01: [B,3,H,W] in [0,1].
    Returns: [B,2,H,W].
    """
    if version == "v1":
        p_blood = blood_probability(rgb01, alpha=alpha, lambda_eff=lambda_eff)
    elif version == "v2":
        # Use a sharper default alpha for v2 since the index range is wider.
        p_blood = blood_probability_v2(rgb01, alpha=max(alpha, 6.0),
                                        pivot=pivot_v2, lambda_eff=lambda_eff)
    else:
        raise ValueError(f"Unknown physics-prior version: {version}")
    h_afi = afi_log_ratio(rgb01)
    phi = fluence_map(rgb01.shape[-2], rgb01.shape[-1],
                     lambda_eff=lambda_eff,
                     device=rgb01.device, dtype=rgb01.dtype)
    h_afi_weighted = h_afi * phi
    return torch.stack([p_blood, h_afi_weighted], dim=1)
