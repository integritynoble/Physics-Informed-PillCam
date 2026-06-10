#!/usr/bin/env bash
# One-command reproduction of the headline ablation, per paper §4.9.
#
# Prereqs (manual):
#   1. Accept Kvasir-Capsule terms at https://datasets.simula.no/kvasir-capsule/
#      and download labelled_images.zip + metadata.csv to $KVASIR_RAW.
#   2. Place the upstream gastroscopy_code_package on disk and point
#      $GASTROSCOPY_CODE_DIR at it.
#   3. pip install per requirements.txt (see header of that file for the
#      torch/torchvision CUDA wheel URL).
#
# Outputs land under $OUT_BASE (default ./outputs):
#   stage2_rgb_effb0/    test_metrics.json + test_predictions.json + best_model.pt
#   stage2_pi_effb0/     "
#   stage2_distill_effb0/"
#   fig{2,3,4,5}.{png,pdf}
#
# Wall-clock on a GTX 1660 Ti (6 GB) with --mixed_precision: ~10–12 h for the
# 3-config single-seed sweep. Re-runs are idempotent — training scripts auto-
# resume from $OUT/last.pt if present; pass --no_resume to force a fresh run.

set -u  # unset vars are an error; *not* set -e (we want to run all 3 configs even if one fails)

: "${KVASIR_RAW:=./kvasir_raw}"
: "${KVASIR_DATA:=./stage2_data}"
: "${OUT_BASE:=./outputs}"
: "${GASTROSCOPY_CODE_DIR:?set GASTROSCOPY_CODE_DIR to the gastroscopy_code_package path}"

echo "=========================================="
echo "[reproduce] start: $(date)"
echo "[reproduce]   KVASIR_RAW=$KVASIR_RAW"
echo "[reproduce]   KVASIR_DATA=$KVASIR_DATA"
echo "[reproduce]   OUT_BASE=$OUT_BASE"
echo "[reproduce]   GASTROSCOPY_CODE_DIR=$GASTROSCOPY_CODE_DIR"
echo "=========================================="

# -- 1. Verify environment ---------------------------------------------------
python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA not available — this paper's results require a CUDA GPU."
print(f"[reproduce] python={sys.version.split()[0]}  torch={torch.__version__}  cuda={torch.version.cuda}")
print(f"[reproduce] gpu={torch.cuda.get_device_name(0)}  vram={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
PY

# -- 2. Stage data -----------------------------------------------------------
if [ ! -d "$KVASIR_DATA/train" ]; then
    echo "[reproduce] staging Kvasir-Capsule into $KVASIR_DATA"
    python setup_kvasir_capsule.py \
        --images_dir "$KVASIR_RAW/kvasir-capsule-labeled-images" \
        --metadata   "$KVASIR_RAW/metadata.csv" \
        --out_dir    "$KVASIR_DATA" \
        --mode       hardlink
else
    echo "[reproduce] $KVASIR_DATA/train already exists — skipping setup"
fi

# -- 3. Train (3 configs) ----------------------------------------------------
bash run_paired_effb0.sh

# -- 4. Predictions on the held-out test split -------------------------------
for cfg in stage2_rgb_effb0 stage2_pi_effb0 stage2_distill_effb0 ; do
    if [ -f "$OUT_BASE/$cfg/best_model.pt" ]; then
        echo "[reproduce] dumping test predictions for $cfg"
        python figures/eval_test_predictions.py \
            --checkpoint "$OUT_BASE/$cfg/best_model.pt" \
            --data_dir   "$KVASIR_DATA" \
            --out        "$OUT_BASE/$cfg/test_predictions.json" \
            --gastroscopy_code_dir "$GASTROSCOPY_CODE_DIR"
    fi
done

# -- 5. Figures --------------------------------------------------------------
python make_fig2.py --image_dir "$KVASIR_DATA/train/Blood - fresh" --out fig2_blood_kvasir.png
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
echo "[reproduce] done: $(date)"
echo "=========================================="
