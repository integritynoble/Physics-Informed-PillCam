#!/usr/bin/env bash
# Multi-seed confirmation runs for the headline ablation.
# Two extra seeds (41, 43) × three configs (RGB, +PI, +Distill) = 6 runs.
# Combined with the existing seed=42 runs, this gives mean ± SD over 3 seeds
# for Table 2 — the standard reviewer demand for ML/medical-imaging papers.
#
# Wall-clock estimate: ~3-3.5 h per run × 6 = ~20 h on the GTX 1660 Ti with AMP.
#
# Outputs land at: ./outputs/stage2_{rgb,pi,distill}_effb0_seed{41,43}/

: "${DATA_DIR:=./stage2_data}"
: "${OUT_BASE:=./outputs}"
: "${GASTRO_DIR:=./gastroscopy_code_package}"
COMMON_RGB_PI="--data_dir $DATA_DIR --model_name efficientnet_b0 --epochs 30 --batch_size 24 --image_size 224 --lr 1e-4 --num_workers 2 --pretrained --mixed_precision --scheduler cosine --no_deterministic --gastroscopy_code_dir"
COMMON_DISTILL="--data_dir $DATA_DIR --model_name efficientnet_b0 --epochs 30 --batch_size 24 --image_size 224 --lr 1e-4 --num_workers 2 --pretrained --scheduler cosine --no_deterministic --distill_lambda 1.0 --gastroscopy_code_dir"

echo "=========================================="
echo "[multiseed] start: $(date)"
echo "=========================================="

for SEED in 41 43; do
    echo
    echo "[multiseed] === seed=$SEED, RGB-only ==="
    python train_stage2_pi.py $COMMON_RGB_PI "$GASTRO_DIR" \
        --seed $SEED \
        --output_dir "$OUT_BASE/stage2_rgb_effb0_seed${SEED}"  2>&1 \
        | tee "$OUT_BASE/stage2_rgb_effb0_seed${SEED}.log"
    EXIT_RGB=${PIPESTATUS[0]}; echo "[multiseed] seed=$SEED rgb exit=$EXIT_RGB"

    echo
    echo "[multiseed] === seed=$SEED, +MC prior 5-channel ==="
    python train_stage2_pi.py $COMMON_RGB_PI "$GASTRO_DIR" \
        --use_physics_prior \
        --seed $SEED \
        --output_dir "$OUT_BASE/stage2_pi_effb0_seed${SEED}"  2>&1 \
        | tee "$OUT_BASE/stage2_pi_effb0_seed${SEED}.log"
    EXIT_PI=${PIPESTATUS[0]}; echo "[multiseed] seed=$SEED pi exit=$EXIT_PI"

    echo
    echo "[multiseed] === seed=$SEED, +Distillation 3-channel + aux ==="
    python train_stage2_pi_distill.py $COMMON_DISTILL "$GASTRO_DIR" \
        --seed $SEED \
        --output_dir "$OUT_BASE/stage2_distill_effb0_seed${SEED}"  2>&1 \
        | tee "$OUT_BASE/stage2_distill_effb0_seed${SEED}.log"
    EXIT_DISTILL=${PIPESTATUS[0]}; echo "[multiseed] seed=$SEED distill exit=$EXIT_DISTILL"
done

echo
echo "=========================================="
echo "[multiseed] done: $(date)"
echo "=========================================="
