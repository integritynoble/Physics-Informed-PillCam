#!/usr/bin/env bash
# Cross-backbone replication runner for MedIA submission.
#
# Trains ResNet-18 and ConvNeXt-Tiny on Kvasir-Capsule under two arms
# (RGB-only and +PI input-fusion) on 3 seeds (41, 42, 43). 12 runs total.
# Each run is independent and resumable; the script skips runs that already
# have test_metrics.json so partial progress is preserved if interrupted.
#
# Estimated wall-clock on GTX 1660 Ti (6 GB):
#   ResNet-18 per run         ~6   GPU-h (FP32, 15 epochs)
#   ConvNeXt-Tiny per run     ~10  GPU-h (FP32, 15 epochs)
#   Total (12 runs)           ~100 GPU-h  =  ~4 days continuous
#
# Usage:
#   bash run_cross_backbone.sh                 # run the full grid
#   bash run_cross_backbone.sh --dry-run       # print planned runs only
#   bash run_cross_backbone.sh resnet18        # only one backbone
#   bash run_cross_backbone.sh convnext_tiny
#
# When done, run:
#   python aggregate_cross_backbone.py
# to produce `medIA_submission/cross_backbone_summary.md` for the paper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- absolute paths ---------------------------------------------------------
DATA_DIR="/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/data/stage2_data"
OUT_ROOT="/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/outputs/cross_backbone"
GASTRO_DIR="/home2/s248103/abraham/GI/GI_Multi_Task/GI_project/code/gastroscopy_code_package"

# --- grid -------------------------------------------------------------------
BACKBONES=("resnet18" "convnext_tiny")
SEEDS=(41 42 43)
ARMS=("rgb" "pi")          # rgb = baseline; pi = --use_physics_prior

# --- CLI --------------------------------------------------------------------
DRY_RUN=0
FILTER_BACKBONE=""
# --seeds <csv> overrides the default 41,42,43 grid (e.g., --seeds 44,45,47).
prev_arg=""
for arg in "$@"; do
    if [ "$prev_arg" = "--seeds" ]; then
        IFS=',' read -ra SEEDS <<< "$arg"
        prev_arg=""
        continue
    fi
    case "$arg" in
        --dry-run|-n) DRY_RUN=1; prev_arg="" ;;
        resnet18|convnext_tiny|resnet50|efficientnet_b0) FILTER_BACKBONE="$arg"; prev_arg="" ;;
        --seeds) prev_arg="--seeds" ;;
        --help|-h)
            sed -n '2,25p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "unknown arg: $arg" >&2; exit 2 ;;
    esac
done

# --- preflight --------------------------------------------------------------
for d in "$DATA_DIR" "$GASTRO_DIR"; do
    if [ ! -d "$d" ]; then
        echo "[runner] ERROR: missing directory $d" >&2
        exit 1
    fi
done

# Pick a real Python interpreter (Windows often has a Store stub on PATH).
PY=""
for cand in python3 python py; do
    if command -v "$cand" >/dev/null 2>&1; then
        if "$cand" -c "import torch; print(torch.cuda.is_available())" >/dev/null 2>&1; then
            PY="$cand"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    echo "[runner] ERROR: no Python interpreter with torch+cuda found." >&2
    echo "          Activate your conda/venv with torch first." >&2
    exit 1
fi

CUDA_OK=$("$PY" -c "import torch; print(int(torch.cuda.is_available()))")
if [ "$CUDA_OK" != "1" ]; then
    echo "[runner] WARNING: torch.cuda.is_available() is False. Running on CPU will be ~50x slower." >&2
fi

mkdir -p "$OUT_ROOT"

# --- enumerate planned runs ------------------------------------------------
RUN_LIST=()
for backbone in "${BACKBONES[@]}"; do
    if [ -n "$FILTER_BACKBONE" ] && [ "$backbone" != "$FILTER_BACKBONE" ]; then
        continue
    fi
    for seed in "${SEEDS[@]}"; do
        for arm in "${ARMS[@]}"; do
            RUN_LIST+=("${backbone}|${seed}|${arm}")
        done
    done
done

echo "[runner] planned: ${#RUN_LIST[@]} runs"
for spec in "${RUN_LIST[@]}"; do
    IFS='|' read -r backbone seed arm <<< "$spec"
    out="${OUT_ROOT}/${backbone}_seed${seed}_${arm}"
    status="NEW"
    if [ -f "${out}/test_metrics.json" ]; then status="DONE (skip)"; fi
    echo "  ${backbone}  seed=${seed}  arm=${arm}  ->  ${out}  [${status}]"
done

if [ "$DRY_RUN" = "1" ]; then
    echo "[runner] --dry-run: stopping before training."
    exit 0
fi

# --- launch -----------------------------------------------------------------
echo
echo "[runner] starting at $(date)"
START_TS=$(date +%s)

for spec in "${RUN_LIST[@]}"; do
    IFS='|' read -r backbone seed arm <<< "$spec"
    out="${OUT_ROOT}/${backbone}_seed${seed}_${arm}"
    log="${out}/train.log"

    if [ -f "${out}/test_metrics.json" ]; then
        echo "[runner] SKIP   ${backbone} seed=${seed} arm=${arm}  (test_metrics.json exists)"
        continue
    fi

    mkdir -p "$out"
    echo "[runner] START  ${backbone} seed=${seed} arm=${arm}  $(date)"

    physics_flag=""
    if [ "$arm" = "pi" ]; then physics_flag="--use_physics_prior"; fi

    # `last.pt` resume is built into train_stage2_pi.py; we let it handle
    # interrupted runs transparently.
    # NOTE 2026-05-11: lr lowered from 1e-3 -> 1e-4 and mixed-precision
    # disabled after the first attempt produced NaN losses on ResNet-18
    # under FP16 + class-weighted CE. FP32 is slower but stable; ResNet-18
    # and ConvNeXt-Tiny both fit in 6 GB VRAM at batch 32 without AMP.
    "$PY" train_stage2_pi.py \
        --data_dir "$DATA_DIR" \
        --model_name "$backbone" \
        --epochs 15 \
        --batch_size 32 \
        --num_workers 4 \
        --image_size 224 \
        --lr 1e-4 \
        --weight_decay 1e-4 \
        --pretrained \
        --seed "$seed" \
        --output_dir "$out" \
        --gastroscopy_code_dir "$GASTRO_DIR" \
        --scheduler cosine \
        $physics_flag \
        2>&1 | tee "$log"

    if [ ! -f "${out}/test_metrics.json" ]; then
        echo "[runner] WARN  ${backbone} seed=${seed} arm=${arm} produced no test_metrics.json — inspect ${log}" >&2
    else
        echo "[runner] DONE   ${backbone} seed=${seed} arm=${arm}"
    fi
done

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
H=$(( ELAPSED / 3600 )); M=$(( (ELAPSED % 3600) / 60 ))
echo
echo "[runner] finished at $(date)  (elapsed: ${H}h ${M}m)"
echo "[runner] next step:  python aggregate_cross_backbone.py"
