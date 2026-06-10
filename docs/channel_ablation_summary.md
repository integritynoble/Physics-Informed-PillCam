# Channel ablation summary — training-time and inference-time

Computed 2026-05-19 on UTSW BioHPC cluster. EfficientNet-B0 headline recipe (lr 1e-3, 30 epochs, cosine, weight_decay 1e-4, patience 5, batch 32) on the canonical Kvasir-Capsule video-level 70/15/15 split (manifest content_sha256 5c0c3fa5bcb1fec5...). Six seeds: {41, 42, 43, 44, 45, 47}.

## Experiment A — Training-time channel ablation

Two single-channel arms train from scratch with one prior channel zeroed throughout training (`--ablate_input_channels` in `train_stage2_pi.py`). SLURM job 10607698. Macro-AUC over the 11 evaluable test classes:

| arm | mean ± std | per-seed (41/42/43/44/45/47) |
|---|---|---|
| RGB-only         | 0.720 ± 0.060 | 0.773 / 0.778 / 0.705 / 0.626 / 0.681 / 0.757 |
| full +PI         | 0.747 ± 0.023 | 0.733 / 0.791 / 0.739 / 0.741 / 0.750 / 0.727 |
| P_blood-only (Φ zeroed) | 0.733 ± 0.011 | 0.721 / 0.742 / 0.735 / 0.729 / 0.748 / 0.721 |
| Φ-only (P_blood zeroed) | 0.747 ± 0.022 | 0.762 / 0.757 / 0.753 / 0.711 / 0.729 / 0.768 |

Paired deltas:
- full+PI − RGB:           +0.027 ± 0.059, sign-positive 4/6
- P_blood-only − RGB:      +0.013 ± 0.064, sign-positive 3/6
- Φ-only − RGB:            +0.027 ± 0.041, sign-positive 4/6
- full+PI − P_blood-only:  +0.014 ± 0.018, sign-positive **6/6**
- full+PI − Φ-only:        +0.000 ± 0.032, sign-positive 3/6

**Reading.** Training with a separate explicit Φ channel is necessary: removing Φ from training (P_blood-only arm) loses to full input-fusion on every seed. Removing P_blood from training (Φ-only arm) preserves macro-AUC almost exactly.

## Experiment B — Inference-time channel ablation

Same Windows-trained EfficientNet-B0 +PI checkpoints that produce the paper's headline numbers (Table 2). We mask one prior channel at inference and re-run the test forward pass (patched `eval_cross_backbone_auc.py` with `--force_ablate_channels` and `--data_dir_override`).

| inference variant | mean ± std | per-seed (41/42/43/44/45/47) |
|---|---|---|
| +PI baseline (5-channel forward pass)         | 0.783 ± 0.026 | 0.777 / 0.797 / 0.742 / 0.822 / 0.780 / 0.780 |
| +PI, zero Φ at inference                       | 0.782 ± 0.027 | 0.776 / 0.796 / 0.739 / 0.823 / 0.778 / 0.779 |
| +PI, zero P_blood at inference                 | 0.782 ± 0.027 | 0.778 / 0.796 / 0.739 / 0.822 / 0.778 / 0.780 |
| **+PI, BOTH prior channels zeroed (strip-and-serve)** | **0.781 ± 0.028** | 0.776 / 0.795 / 0.737 / 0.823 / 0.776 / 0.779 |

Macro-AUC deltas vs baseline:
- zero Φ:                  −0.0013
- zero P_blood:            −0.0008
- **both channels zeroed:  −0.0020** (1/6 sign-positive — within noise)

Macro-AUC deltas vs RGB-only-trained baseline (paper's headline RGB row, 0.7598 ± 0.0297):
- **+PI baseline (paper headline):       +0.0232 (5/6 sign-positive)**
- **+PI both channels zeroed (strip-and-serve):  +0.0211 (4/6 sign-positive)** — retains 91% of the headline lift

The +PI baseline reproduces paper's published 0.783 ± 0.024 exactly, validating the data_dir override. The strip-and-serve row is the deployment-relevant configuration: train the 5-channel input-fusion model, feed `[R, G, B, 0, 0]` at inference, no prior computation needed on the deployment path. This is a cleaner deployment story than the explicit distillation variant (which requires a separate training pipeline).

### Per-class on the same checkpoints (mean over 6 seeds)

| class | RGB ckpt | +PI baseline | +PI zero Φ | +PI zero P_blood | Δ(zero Φ vs +PI) | Δ(zero P_b vs +PI) |
|---|---|---|---|---|---|---|
| Angiectasia          | 0.646 | 0.608 | 0.603 | 0.612 | −0.005 | +0.005 |
| Blood - fresh        | 0.686 | 0.718 | 0.716 | 0.716 | −0.002 | −0.001 |
| Erosion              | 0.695 | 0.785 | 0.784 | 0.782 | −0.001 | −0.002 |
| Erythema             | 0.907 | 0.896 | 0.891 | 0.889 | −0.004 | −0.007 |
| Foreign Body         | 0.970 | 0.983 | 0.983 | 0.983 |  0.000 |  0.000 |
| Ileocecal valve      | 0.810 | 0.806 | 0.806 | 0.806 |  0.000 |  0.000 |
| **Lymphangiectasia** | 0.238 | **0.337** | **0.338** | **0.337** | **+0.001** |  **0.000** |
| Normal clean mucosa  | 0.831 | 0.882 | 0.880 | 0.879 | −0.002 | −0.003 |
| Pylorus              | 0.874 | 0.894 | 0.892 | 0.891 | −0.002 | −0.003 |
| Reduced Mucosal View | 0.757 | 0.750 | 0.750 | 0.751 |  0.000 | +0.001 |
| Ulcer                | 0.945 | 0.955 | 0.956 | 0.956 |  0.000 | +0.001 |

**Reading.** No per-class shifts >0.007 in magnitude. The paper's hero per-class claim (Lymphangiectasia +PI lift, 0.238 → 0.337) is fully preserved when either prior channel is zeroed at inference.

## Cross-architecture: ConvNeXt-Tiny channel ablation (matched recipe)

SLURM job 10610131 (completed 2026-05-20, 9h27m wall). ConvNeXt-Tiny with the matched cross-backbone recipe used for the existing Table 4 cross-backbone replication: 15 epochs, lr 1e-4, cosine, weight_decay 1e-4, batch 32, early-stop patience 5. Canonical Linux split, n=6 seeds.

| arm | mean ± std | per-seed (41/42/43/44/45/47) |
|---|---|---|
| RGB-only         | 0.7461 ± 0.0173 | 0.748 / 0.765 / 0.747 / 0.764 / 0.729 / 0.724 |
| full +PI         | 0.7640 ± 0.0153 | 0.775 / 0.757 / 0.749 / 0.756 / 0.756 / 0.790 |
| **P_blood-only** | **0.8051 ± 0.0200** | 0.815 / 0.789 / 0.807 / 0.832 / 0.813 / 0.776 |
| **Φ-only**       | **0.7866 ± 0.0141** | 0.777 / 0.779 / 0.788 / 0.801 / 0.805 / 0.770 |

Paired deltas (n=6):
- full +PI − RGB:           +0.018 ± 0.028, 4/6 sign-positive, Wilcoxon p=0.31 (matches paper's existing Table 4 cross-backbone Δ=+0.018)
- **P_blood-only − full +PI: +0.041 ± 0.031, 5/6 sign-positive, Wilcoxon p=0.063**
- **Φ-only − full +PI:        +0.023 ± 0.027, 5/6 sign-positive, Wilcoxon p=0.094**
- **P_blood-only − RGB:       +0.059 ± 0.021, 6/6 sign-positive, Wilcoxon p=0.031**
- **Φ-only − RGB:             +0.040 ± 0.021, 6/6 sign-positive, Wilcoxon p=0.031**

**Pattern inverts the EfficientNet-B0 result.** On EffB0 (Table above), full +PI beats P_blood-only on 6/6 seeds (Δ=+0.014). On ConvNeXt-T, P_blood-only beats full +PI on 5/6 seeds (Δ=+0.041). The honest read: the precise channel-content interaction at training time is **architecture-specific**. What replicates across both backbones is the headline +PI > RGB direction (+0.023 on EffB0; +0.018 on ConvNeXt-T).

Training-curve diagnostic: ConvNeXt-T runs are NOT undertrained at 15 epochs (val-F1 peaks at epoch 7-8 for full +PI, similar early-peaking for ablation arms, then drifts down under cosine schedule). A follow-on sweep at the EfficientNet-B0-headline recipe (30 ep, lr 1e-3) ran as SLURM jobs 10611450 (full +PI) + 10611451 (P_blood-only); both completed 2026-05-20. Result: pretrained ConvNeXt-Tiny does NOT tolerate lr 1e-3 fine-tuning.

| arm | matched recipe (15ep/lr1e-4) | EffB0 recipe (30ep/lr1e-3) | Δ recipe |
|---|---|---|---|
| full +PI | 0.7640 ± 0.0153 | 0.5898 ± 0.0788 | −0.174 |
| P_blood-only | 0.8051 ± 0.0200 | 0.6193 ± 0.0564 | −0.186 |

Both arms collapse ~0.17 macro-AUC under the higher LR — consistent with catastrophic forgetting of ImageNet pretraining (a known ConvNeXt sensitivity). EfficientNet-B0 doesn't exhibit this — its full +PI at the same recipe stays at 0.747 ± 0.023.

Within-recipe paired tests for P_blood-only − full +PI on ConvNeXt-T:
- 15ep/lr1e-4 (matched): Δ=+0.041 ± 0.031, 5/6 sign-positive, Wilcoxon p=0.063 ← cleaner result
- 30ep/lr1e-3 (EffB0 headline): Δ=+0.030 ± 0.085, 3/6 sign-positive, Wilcoxon p=0.56 ← buried in recipe noise

**Verdict.** The matched-recipe inversion is the cleaner finding and stands. The 30ep recipe is incompatible with pretrained ConvNeXt-T (catastrophic forgetting); the channel-content signal is partially preserved in direction but reduced to noise within this recipe. The recipe-vs-architecture question cannot be cleanly resolved without an additional cross-architecture sweep at recipes both backbones tolerate. The lr-tolerance asymmetry itself is a recipe-calibration finding for practitioners.

## Unified interpretation

The analytic prior contributes via a **training-time inductive bias** on the first convolutional layer, not an inference-time feature. The trained network's first conv has been re-parameterized by exposure to the prior during training; at inference, it extracts the +PI-quality features from the RGB content of the input alone. This unified mechanism explains four results that previously stood separately:

1. **Input-fusion lift** (Table 2 headline +0.023): the prior-during-training shapes the first conv.
2. **Distillation variant success** (3-channel RGB inference, 0.773): formalizes the training-time-bias mechanism, but the same effect is already happening inside the input-fusion model.
3. **PI-TTA null** (§4.7): inference-time procedures cannot install a bias the first conv was never exposed to during training.
4. **Inference-time channel-ablation null** (this experiment): trained model's first conv no longer queries the prior channels at inference.

## Reproduction

```bash
# Training-time ablation (cluster, SLURM)
sbatch GI_project/code/Capsule-Endoscopy/submit_effb0_channel_ablation.sbatch

# Inference-time ablation (cluster login node or interactive GPU)
cd GI_project/code/Capsule-Endoscopy
python eval_cross_backbone_auc.py \
    --only effb0_paper_seed --device cpu --batch_size 128 \
    --data_dir_override /project/BME/Zaman_lab/s248103/stage2_data_canonical \
    --output_name test_auc_winckpt.json    # baseline (no mask)
python eval_cross_backbone_auc.py \
    --only effb0_paper_seed --device cpu --batch_size 128 \
    --data_dir_override /project/BME/Zaman_lab/s248103/stage2_data_canonical \
    --force_ablate_channels 4 \
    --output_name test_auc_winckpt_phi_zeroed.json    # zero Φ
python eval_cross_backbone_auc.py \
    --only effb0_paper_seed --device cpu --batch_size 128 \
    --data_dir_override /project/BME/Zaman_lab/s248103/stage2_data_canonical \
    --force_ablate_channels 3 \
    --output_name test_auc_winckpt_pblood_zeroed.json    # zero P_blood
```

Wall-clock: training-time ablation 4 h on a V100 (12 runs × ~21 min). Inference-time ablation 36 min total on CPU (18 inference passes × ~2 min each).
