#!/usr/bin/env bash
# Component ablation for paper Table 3.
# Three runs, each EfficientNet-B0 + PI 5-channel input, 30 epochs, seed=42:
#   1. blood_only:  zero the H_AFI*Phi channel (idx 4) — isolates P_blood contribution
#   2. hafi_only:   zero the P_blood channel (idx 3)  — isolates H_AFI contribution
#   3. physics_only: zero RGB channels (idx 0,1,2)    — disambiguates "is the prior alone informative or just a regularizer?"
#
# Wall-clock: ~3 h per run × 3 = ~9–10 h on the GTX 1660 Ti with AMP.
# Outputs: D:/kvasir_capsule/outputs/stage2_pi_{blood_only,hafi_only,physics_only}_effb0/

: "${DATA_DIR:=D:/kvasir_capsule/stage2_data}"
: "${OUT_BASE:=D:/kvasir_capsule/outputs}"
: "${GASTRO_DIR:=D:/onedrive/UT_southwestern/GIproject/Dr. Zaman/gastroscopy_code_package (2)/gastroscopy_code_package}"
COMMON="--data_dir $DATA_DIR --model_name efficientnet_b0 --epochs 30 --batch_size 24 --image_size 224 --lr 1e-4 --num_workers 2 --pretrained --use_physics_prior --mixed_precision --scheduler cosine --no_deterministic --gastroscopy_code_dir"

echo "=========================================="
echo "[ablation] start: $(date)"
echo "=========================================="

echo
echo "[ablation] === Run 1/3: +P_blood channel only (H_AFI ablated) ==="
python train_stage2_pi.py $COMMON "$GASTRO_DIR" \
    --ablate_input_channels 4 \
    --output_dir "$OUT_BASE/stage2_pi_blood_only_effb0"  2>&1 | tee "$OUT_BASE/stage2_pi_blood_only_effb0.log"
EXIT_BLOOD=${PIPESTATUS[0]}; echo "[ablation] run 1 exit=$EXIT_BLOOD"

echo
echo "[ablation] === Run 2/3: +H_AFI channel only (P_blood ablated) ==="
python train_stage2_pi.py $COMMON "$GASTRO_DIR" \
    --ablate_input_channels 3 \
    --output_dir "$OUT_BASE/stage2_pi_hafi_only_effb0"  2>&1 | tee "$OUT_BASE/stage2_pi_hafi_only_effb0.log"
EXIT_HAFI=${PIPESTATUS[0]}; echo "[ablation] run 2 exit=$EXIT_HAFI"

echo
echo "[ablation] === Run 3/3: physics-only (RGB zeroed) ==="
python train_stage2_pi.py $COMMON "$GASTRO_DIR" \
    --ablate_input_channels 0,1,2 \
    --output_dir "$OUT_BASE/stage2_pi_physics_only_effb0"  2>&1 | tee "$OUT_BASE/stage2_pi_physics_only_effb0.log"
EXIT_PHYS=${PIPESTATUS[0]}; echo "[ablation] run 3 exit=$EXIT_PHYS"

echo
echo "=========================================="
echo "[ablation] done: $(date)"
echo "[ablation] summary  blood=$EXIT_BLOOD  hafi=$EXIT_HAFI  physics=$EXIT_PHYS  (0 = success)"
echo "=========================================="
