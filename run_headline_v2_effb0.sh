#!/bin/bash
# Headline retraining sweep targeting the Diagnostics submission.
#
# Trains the three arms that drive the paper headline (RGB-only baseline,
# Full PI with v2 prior, Distill with v2 teacher) using:
#   - the corrected checkpoint-selection criterion (val macro-F1-evaluable)
#   - the redesigned v2 P_blood prior (NDVI_red, scale-fixed)
#   - mild regularization (label smoothing + early stopping) — chosen to
#     close the +0.78 train-val gap without changing the data pipeline.
#
# Each arm is a fresh run (--no_resume) writing into a *_v2 sibling folder;
# the original v1 outputs are preserved for the ablation table. Expected
# wall-clock at ~24h/arm on the GTX 1660 Ti (~3 days for the three arms);
# patience=8 early stopping should cut that significantly given the val
# curve plateaus by epoch ~5.
#
# Usage from paper_draft/:
#    bash run_headline_v2_effb0.sh
set -e

# Overridable via environment variables; POSIX relative defaults.
OUT="${OUT_BASE:-./outputs}"
DATA="${DATA_DIR:-./stage2_data}"
GASTRO="${GASTRO_DIR:-./gastroscopy_code_package}"

# Note: --gastroscopy_code_dir is passed as a separate quoted arg below
# because $GASTRO contains spaces and would be word-split if baked into $COMMON.
#
# Why no --label_smoothing here (despite it being available): we measured
# 2026-04-29 that nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
# explodes when the class_weights are large (Foreign Body weight ≈ 700×, etc.)
# because the smoothing penalty applies across ALL classes and is multiplied
# by the per-sample class weight. Train loss climbed to ~14 vs ~0.005 in the
# baseline by epoch 5; the model could not even fit the training set (train
# F1 0.71 vs 0.99 baseline). Smoothing remains available via the trainer flag
# for users who turn off class-weighting — this script keeps the v1 setup.
COMMON="--data_dir $DATA --model_name efficientnet_b0 --epochs 30 \
        --batch_size 24 --image_size 224 --lr 1e-4 --num_workers 2 \
        --pretrained --mixed_precision --scheduler cosine --no_deterministic \
        --early_stopping_patience 8 --no_resume"

# 1) RGB baseline — re-trained with the corrected selection criterion + label smoothing.
#    No v2 to apply (RGB has no physics channels), so this run isolates the
#    selection-criterion + label-smoothing fix from the prior redesign.
echo "==== Run 1/3: RGB baseline (corrected selection criterion, early stopping) ===="
python train_stage2_pi.py $COMMON \
  --gastroscopy_code_dir "$GASTRO" \
  --output_dir "$OUT/stage2_rgb_effb0_v2"  2>&1 | tee -a "$OUT/run_headline_v2.log"

# 2) Full PI with v2 prior.
echo "==== Run 2/3: Full PI with v2 prior ===="
python train_stage2_pi.py $COMMON \
  --gastroscopy_code_dir "$GASTRO" \
  --use_physics_prior --physics_prior_version v2 --physics_pivot_v2 0.30 \
  --output_dir "$OUT/stage2_pi_effb0_v2"  2>&1 | tee -a "$OUT/run_headline_v2.log"

# 3) Distillation with v2 teacher.
echo "==== Run 3/3: Distillation with v2 teacher ===="
python train_stage2_pi_distill.py $COMMON \
  --gastroscopy_code_dir "$GASTRO" \
  --distill_lambda 1.0 --physics_prior_version v2 --physics_pivot_v2 0.30 \
  --output_dir "$OUT/stage2_distill_effb0_v2"  2>&1 | tee -a "$OUT/run_headline_v2.log"

echo "==== headline v2 sweep DONE ===="
