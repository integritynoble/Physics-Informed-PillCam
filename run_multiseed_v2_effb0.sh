#!/usr/bin/env bash
# Multi-seed sweep for the v2 headline arms (3 seeds × 3 arms = 9 runs).
#
# Combined with run_headline_v2_effb0.sh (which uses seed=42), this gives
# (41, 42, 43) for each of {RGB-only_v2, Full PI v2, Distill v2}. Reviewers
# will ask for ≥3 seeds on the headline; this is the cheapest way to
# satisfy that. Wall clock: ~25 h per run × 6 ≈ 6 days, but early-stopping
# (patience=8) typically halves that since val plateaus by epoch ~5.
#
# Outputs land at: D:/kvasir_capsule/outputs/stage2_{rgb,pi,distill}_effb0_v2_seed{41,43}/
set -e

OUT=D:/kvasir_capsule/outputs
DATA=D:/kvasir_capsule/stage2_data
GASTRO="D:/onedrive/UT_southwestern/GIproject/Dr. Zaman/gastroscopy_code_package (2)/gastroscopy_code_package"

# --gastroscopy_code_dir is passed separately because $GASTRO contains spaces.
# No --label_smoothing — see comment in run_headline_v2_effb0.sh (smoothing
# x class_weights blew the loss up to ~14 on the 14-class imbalanced split).
COMMON="--data_dir $DATA --model_name efficientnet_b0 --epochs 30 \
        --batch_size 24 --image_size 224 --lr 1e-4 --num_workers 2 \
        --pretrained --mixed_precision --scheduler cosine --no_deterministic \
        --early_stopping_patience 8 --no_resume"

for SEED in 41 43; do
  echo "==== seed=$SEED, RGB v2 ===="
  python train_stage2_pi.py $COMMON --seed $SEED \
    --gastroscopy_code_dir "$GASTRO" \
    --output_dir "$OUT/stage2_rgb_effb0_v2_seed${SEED}"  2>&1 \
    | tee "$OUT/stage2_rgb_effb0_v2_seed${SEED}.log"

  echo "==== seed=$SEED, Full PI v2 ===="
  python train_stage2_pi.py $COMMON --seed $SEED \
    --gastroscopy_code_dir "$GASTRO" \
    --use_physics_prior --physics_prior_version v2 --physics_pivot_v2 0.30 \
    --output_dir "$OUT/stage2_pi_effb0_v2_seed${SEED}"  2>&1 \
    | tee "$OUT/stage2_pi_effb0_v2_seed${SEED}.log"

  echo "==== seed=$SEED, Distill v2 ===="
  python train_stage2_pi_distill.py $COMMON --seed $SEED \
    --gastroscopy_code_dir "$GASTRO" \
    --distill_lambda 1.0 --physics_prior_version v2 --physics_pivot_v2 0.30 \
    --output_dir "$OUT/stage2_distill_effb0_v2_seed${SEED}"  2>&1 \
    | tee "$OUT/stage2_distill_effb0_v2_seed${SEED}.log"
done
echo "==== multiseed v2 DONE ===="
