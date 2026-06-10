#!/usr/bin/env bash
# Distillation training — 6 seeds for one backbone (3-channel RGB + aux P_blood decoder).
#
# Usage:
#   bash run_distill.sh                            # default: efficientnet_b0
#   bash run_distill.sh resnet18
#   bash run_distill.sh convnext_tiny
#   bash run_distill.sh efficientnet_b0 --dry-run
#
# Output dir convention (mirrors build_embedding_cache_distill.py:output_dir_for):
#   seed 42 -> $OUT_ROOT/stage2_distill_{backbone_short}
#   other   -> $OUT_ROOT/stage2_distill_{backbone_short}_seed{N}
# where backbone_short = effb0 | resnet18 | convnext_tiny.
#
# Idempotent: skips any seed whose test_metrics.json exists. Interrupted runs
# resume from last.pt via train_stage2_pi_distill.py's built-in resume logic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NEW_ROOT=/home2/s248103/abraham/GI/GI_Multi_Task/GI_project
DATA_DIR="$NEW_ROOT/data/stage2_data"
OUT_ROOT="$NEW_ROOT/outputs"
GASTRO_DIR="$NEW_ROOT/code/gastroscopy_code_package"

SEEDS=(41 42 43 44 45 47)

MODEL_NAME="efficientnet_b0"
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        efficientnet_b0|resnet18|resnet50|convnext_tiny) MODEL_NAME="$arg" ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# Short tag for the output dir (matches existing convention "stage2_distill_effb0")
case "$MODEL_NAME" in
    efficientnet_b0) SHORT="effb0" ;;
    resnet18)        SHORT="resnet18" ;;
    resnet50)        SHORT="resnet50" ;;
    convnext_tiny)   SHORT="convnext_tiny" ;;
    *) echo "unsupported model: $MODEL_NAME" >&2; exit 2 ;;
esac

for d in "$DATA_DIR" "$GASTRO_DIR"; do
    [ -d "$d" ] || { echo "[distill] ERROR: missing $d" >&2; exit 1; }
done

PY="python"
"$PY" -c "import torch; assert torch.cuda.is_available()" || {
    echo "[distill] WARNING: torch.cuda.is_available() is False — will run on CPU (~50x slower)." >&2
}

mkdir -p "$OUT_ROOT"

dir_for() {
    local seed="$1"
    if [ "$seed" = "42" ]; then
        echo "$OUT_ROOT/stage2_distill_${SHORT}"
    else
        echo "$OUT_ROOT/stage2_distill_${SHORT}_seed${seed}"
    fi
}

echo "[distill] backbone=$MODEL_NAME  (short=$SHORT)"
echo "[distill] planned: ${#SEEDS[@]} seeds"
for seed in "${SEEDS[@]}"; do
    out=$(dir_for "$seed")
    status="NEW"
    [ -f "$out/test_metrics.json" ] && status="DONE (skip)"
    echo "  seed=${seed}  ->  ${out}  [${status}]"
done

if [ "$DRY_RUN" = "1" ]; then
    echo "[distill] --dry-run: stopping before training."
    exit 0
fi

echo
echo "[distill] starting at $(date)"
START_TS=$(date +%s)

for seed in "${SEEDS[@]}"; do
    out=$(dir_for "$seed")
    log="$out/train.log"

    if [ -f "$out/test_metrics.json" ]; then
        echo "[distill] SKIP   $MODEL_NAME seed=${seed}  (test_metrics.json exists)"
        continue
    fi

    mkdir -p "$out"
    echo "[distill] START  $MODEL_NAME seed=${seed}  $(date)"

    "$PY" train_stage2_pi_distill.py \
        --data_dir "$DATA_DIR" \
        --model_name "$MODEL_NAME" \
        --epochs 30 \
        --batch_size 24 \
        --num_workers 4 \
        --image_size 224 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --pretrained \
        --seed "$seed" \
        --output_dir "$out" \
        --gastroscopy_code_dir "$GASTRO_DIR" \
        --scheduler cosine \
        --distill_lambda 1.0 \
        2>&1 | tee "$log"

    if [ ! -f "$out/test_metrics.json" ]; then
        echo "[distill] WARN  $MODEL_NAME seed=${seed} produced no test_metrics.json — inspect ${log}" >&2
    else
        echo "[distill] DONE   $MODEL_NAME seed=${seed}"
    fi
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
H=$(( ELAPSED / 3600 )); M=$(( (ELAPSED % 3600) / 60 ))
echo
echo "[distill] $MODEL_NAME finished at $(date)  (elapsed: ${H}h ${M}m)"
