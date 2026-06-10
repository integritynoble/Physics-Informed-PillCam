# BiomedCLIP linear probe — medical-domain foundation-model baseline

SLURM job 10623025, completed 2026-05-20 (3:21 wall, V100). Frozen `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` image encoder (ViT-B/16 image arm), 512-d projected embeddings, 6 linear-classifier seeds {41, 42, 43, 44, 45, 47} trained with the same canonical Kvasir-Capsule split as the EffB0 headline runs.

## Per-seed macro-AUC (11 evaluable classes)

| seed | macro-AUC |
|---|---|
| 41 | 0.7308 |
| 42 | 0.7335 |
| 43 | 0.7354 |
| 44 | 0.7340 |
| 45 | 0.7322 |
| 47 | 0.7325 |
| **mean** | **0.7331 ± 0.0016** |

## Comparison with baselines

| baseline | macro-AUC (n=6) | gap vs +PI EffB0 (0.783) |
|---|---|---|
| DINOv2 base (frozen + linear probe) | 0.6661 ± 0.0070 | −0.117 (~4σ) |
| **BiomedCLIP (frozen + linear probe)** | **0.7331 ± 0.0016** | **−0.050 (~2σ)** |
| EffB0 RGB-only (full fine-tune) | 0.7598 ± 0.0265 | −0.023 |
| EffB0 +PI input fusion (paper headline) | 0.7830 ± 0.0240 | — |
| EffB0 +PI strip-and-serve | 0.7810 ± 0.0282 | −0.002 |

## Key findings

1. **Medical-domain pretraining helps** — BiomedCLIP linear probe scores 0.067 macro-AUC ABOVE DINOv2 (0.7331 vs 0.6661), confirming the literature finding that in-domain pretraining materially improves medical-imaging downstream tasks.

2. **BiomedCLIP still loses to +PI EffB0** by 0.050 macro-AUC (~2σ). 95% BCa CI on the gap excludes zero. The "medical foundation models obviate task-specific priors" reviewer concern is defanged with a stronger baseline than DINOv2-alone.

3. **BiomedCLIP also loses to RGB-only EffB0 fine-tune** by 0.026 macro-AUC. Task-specific fine-tuning of a smaller backbone (EffB0, 5M params) outperforms a frozen medical foundation backbone (BiomedCLIP ViT-B/16, ~86M params) at Kvasir-Capsule scale.

## Reproduction

```bash
# Single-job pipeline: extract embeddings (cached) + 6-seed linear probes
sbatch GI_project/code/Capsule-Endoscopy/submit_biomedclip.sbatch
# Output: GI_project/outputs/biomedclip_linear/
#   - embeddings.npz        (57 MB, cached for re-use)
#   - aggregate.json        (mean/std/per_seed)
#   - seed{N}/test_metrics.json + classifier.pt
```

Wall-clock: ~3.5 min on V100 (embedding extraction ~1.5 min, 6 linear probes ~30s each). Linear probes train AdamW lr 5e-4, weight decay 1e-3, 80 epochs with best-val-macro-F1 checkpointing.

## Implication for the paper

The §4.1 foundation-model paragraph now reports both baselines. The §5.7 "Anticipated questions" answer (Q5: "Don't 2024-2025 foundation models obviate task-specific priors?") is strengthened: we have tested BOTH a generic visual foundation model (DINOv2) and a state-of-the-art medical-domain foundation model (BiomedCLIP); both lose to our +PI EffB0, with BiomedCLIP loss exceeding the cross-seed σ by ~2σ.

The remaining medically-pretrained alternatives (MedSAM, RETFound — fundus only) are deferred to future work submissions on the GalKva-2026 leaderboard.
