#!/usr/bin/env bash
# Sequential 3-config training sweep for the Week-2 headline ablation.
#
# Configs (all efficientnet_b0, 30 epochs, batch 24, lr 1e-4, seed 42):
#   1. RGB-only baseline           train_stage2_pi.py
#   2. +MC prior (5-channel input) train_stage2_pi.py --use_physics_prior
#   3. +Distillation (3-channel    train_stage2_pi_distill.py --distill_lambda 1.0
#      RGB + aux P_blood head)
#
# Estimated wall-clock: ~3.5 h per config x 3 = ~10-12 h. Run overnight.
# Outputs: $OUT_BASE/stage2_{rgb,pi,distill}_effb0/
# Each output dir gets training_history.json, test_metrics.json, best_model.pt.

# Intentionally NOT set -e: if one config fails, the others should still run.
# Each python invocation's exit status is captured and reported below.

# Paths default to POSIX relative dirs under the cwd; override via environment
# variables (reproduce.sh exports these). No machine-specific paths.
: "${DATA_DIR:=./stage2_data}"
: "${OUT_BASE:=./outputs}"
: "${GASTRO_DIR:=./gastroscopy_code_package}"
# Resume support: training scripts auto-resume from $OUT/last.pt if present.
# To force a fresh start (e.g. after re-splitting the data), pass --no_resume
# or delete last.pt + best_model.pt in each output dir before running.
# Pipeline-validation pass uses --no_deterministic so cudnn benchmark autotunes
# the convolutions (~2× speed on the 1660 Ti vs. fully deterministic mode).
# Re-enable deterministic mode for the publishable Week-2 multi-seed runs.
COMMON="--data_dir $DATA_DIR --model_name efficientnet_b0 --epochs 30 --batch_size 24 --image_size 224 --lr 1e-4 --num_workers 2 --pretrained --mixed_precision --scheduler cosine --no_deterministic --gastroscopy_code_dir"

mkdir -p "$OUT_BASE"

echo "=========================================="
echo "[run_paired_effb0] start: $(date)"
echo "=========================================="

echo
echo "[run_paired_effb0] === Config 1/3: RGB-only baseline ==="
python train_stage2_pi.py $COMMON "$GASTRO_DIR" --output_dir "$OUT_BASE/stage2_rgb_effb0"  2>&1 | tee "$OUT_BASE/stage2_rgb_effb0.log"
EXIT_RGB=${PIPESTATUS[0]}; echo "[run_paired_effb0] config 1 exit=$EXIT_RGB"

echo
echo "[run_paired_effb0] === Config 2/3: +MC prior 5-channel ==="
python train_stage2_pi.py $COMMON "$GASTRO_DIR" --use_physics_prior --output_dir "$OUT_BASE/stage2_pi_effb0"  2>&1 | tee "$OUT_BASE/stage2_pi_effb0.log"
EXIT_PI=${PIPESTATUS[0]}; echo "[run_paired_effb0] config 2 exit=$EXIT_PI"

echo
echo "[run_paired_effb0] === Config 3/3: +Distillation 3-channel + aux head ==="
python train_stage2_pi_distill.py $COMMON "$GASTRO_DIR" --distill_lambda 1.0 --output_dir "$OUT_BASE/stage2_distill_effb0"  2>&1 | tee "$OUT_BASE/stage2_distill_effb0.log"
EXIT_DISTILL=${PIPESTATUS[0]}; echo "[run_paired_effb0] config 3 exit=$EXIT_DISTILL"

echo
echo "=========================================="
echo "[run_paired_effb0] done: $(date)"
echo "[run_paired_effb0] summary  rgb=$EXIT_RGB  pi=$EXIT_PI  distill=$EXIT_DISTILL  (0 = success)"
echo "=========================================="
