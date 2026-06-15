#!/usr/bin/env bash
# Single-seed SMOKE / sanity reproduction of the headline ablation.
#
# IMPORTANT — what this script is and is not:
#   * This is a single-seed (seed 42) SANITY path that trains the three input
#     variants (RGB, +PI 5-channel, +Distill) and dumps test predictions and
#     a couple of figures. It exists so a reviewer can confirm the pipeline
#     runs end to end on one GPU and produces the expected artifacts.
#   * It is NOT the canonical headline reproduction. The published numbers are
#     the n=10-seed canonical-split runs in
#     scripts/submit_effb0_canonical_n10.sbatch (paper §4.9), which uses the
#     canonical split manifest benchmark/canonical_splits/kvasir_split_manifest_2026-05-18.json.
#     Use the sbatch for the headline; use this script for a quick smoke test.
#
# Prereqs (manual):
#   1. Accept Kvasir-Capsule terms at https://datasets.simula.no/kvasir-capsule/
#      and download labelled_images + metadata.csv to $KVASIR_RAW.
#   2. Provide the upstream gastroscopy_code_package (datasets.py, models.py,
#      utils.py) and point $GASTRO_DIR at it. This package is a hard dependency
#      of the training scripts and is NOT bundled in this repo — see README
#      "Dependencies" for how to obtain/pin it.
#   3. pip install -r requirements.txt (see that file's header for the
#      torch/torchvision CUDA wheel URL).
#
# All paths are overridable via environment variables; defaults are POSIX
# relative paths under the current directory (no machine-specific paths).

set -u  # unset vars are an error; *not* set -e (we run all 3 configs even if one fails)

: "${KVASIR_RAW:=./kvasir_raw}"
: "${DATA_DIR:=./stage2_data}"
: "${OUT_BASE:=./outputs}"
: "${GASTRO_DIR:?set GASTRO_DIR to the gastroscopy_code_package path (see README Dependencies)}"

# Export the names the downstream sweep script reads, so it inherits these
# paths instead of falling back to its own defaults.
export DATA_DIR OUT_BASE GASTRO_DIR

echo "=========================================="
echo "[reproduce] SMOKE reproduction (single seed=42)"
echo "[reproduce]   KVASIR_RAW=$KVASIR_RAW"
echo "[reproduce]   DATA_DIR=$DATA_DIR"
echo "[reproduce]   OUT_BASE=$OUT_BASE"
echo "[reproduce]   GASTRO_DIR=$GASTRO_DIR"
echo "=========================================="

# -- 1. Verify environment ---------------------------------------------------
python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA not available — this paper's results require a CUDA GPU."
print(f"[reproduce] python={sys.version.split()[0]}  torch={torch.__version__}  cuda={torch.version.cuda}")
print(f"[reproduce] gpu={torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
PY

# -- 2. Stage data -----------------------------------------------------------
if [ ! -d "$DATA_DIR/train" ]; then
    echo "[reproduce] staging Kvasir-Capsule into $DATA_DIR"
    python setup_kvasir_capsule.py \
        --images_dir "$KVASIR_RAW/kvasir-capsule-labeled-images" \
        --metadata   "$KVASIR_RAW/metadata.csv" \
        --out_dir    "$DATA_DIR" \
        --mode       hardlink
else
    echo "[reproduce] $DATA_DIR/train already exists — skipping setup"
fi

# -- 3. Train (3 configs, single seed) ---------------------------------------
bash run_paired_effb0.sh

# -- 4. Predictions on the held-out test split -------------------------------
# Map each output config to its eval variant (rgb|pi|distill).
declare -A VARIANT=(
    [stage2_rgb_effb0]=rgb
    [stage2_pi_effb0]=pi
    [stage2_distill_effb0]=distill
)
for cfg in stage2_rgb_effb0 stage2_pi_effb0 stage2_distill_effb0 ; do
    if [ -f "$OUT_BASE/$cfg/best_model.pt" ]; then
        echo "[reproduce] dumping test predictions for $cfg (variant=${VARIANT[$cfg]})"
        python figures/eval_test_predictions.py \
            --ckpt     "$OUT_BASE/$cfg/best_model.pt" \
            --data_dir "$DATA_DIR" \
            --variant  "${VARIANT[$cfg]}" \
            --out      "$OUT_BASE/$cfg/test_predictions.json" \
            --gastroscopy_code_dir "$GASTRO_DIR"
    fi
done

# -- 5. Figures --------------------------------------------------------------
# fig1 is a static schematic; fig3/fig4 consume the test predictions above.
python figures/make_fig1_pipeline.py
python figures/make_fig3_roc.py \
    --rgb     "$OUT_BASE/stage2_rgb_effb0/test_predictions.json" \
    --pi      "$OUT_BASE/stage2_pi_effb0/test_predictions.json" \
    --distill "$OUT_BASE/stage2_distill_effb0/test_predictions.json" \
    --out     fig3_roc.png
python figures/make_fig4_cm.py \
    --rgb     "$OUT_BASE/stage2_rgb_effb0/test_predictions.json" \
    --pi      "$OUT_BASE/stage2_pi_effb0/test_predictions.json" \
    --distill "$OUT_BASE/stage2_distill_effb0/test_predictions.json" \
    --out     fig4_confusion_matrices.png

echo "=========================================="
echo "[reproduce] smoke reproduction done: $(date)"
echo "[reproduce] for the published n=10 headline, run scripts/submit_effb0_canonical_n10.sbatch"
echo "=========================================="
