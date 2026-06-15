# Physics-Informed PillCam™ AI

Code and benchmark release for the paper

> **Training-Time Optical Priors for Wireless Capsule Endoscopy
> Classification: Hemoglobin-Aware Input Fusion with Cross-Vendor
> Evaluation**
> Yang, Xing, Alam, Entin, Vemulapalli, Casey, Zaman — 2026
> Submitted to *Medical Image Analysis*.

We feed a Monte-Carlo–inspired hemoglobin prior $P_\text{blood}$
(computed analytically from RGB pixels) alongside the standard image
into an EfficientNet-B0 classifier. The prior is consumed only during
training; at inference the network runs on plain 3-channel RGB.

Headline result on **Kvasir-Capsule** (47 238 frames, patient-disjoint
split): macro-AUC $0.760 \pm 0.027 \to 0.783 \pm 0.024$ (5/6 seeds
positive). The combined three-stream architecture reaches
$0.804 \pm 0.023$, with the strongest per-class lift on
*Lymphangiectasia*. The lift is direction-consistent across ConvNeXt-Tiny
on the **GalKva-2026** cross-vendor benchmark, and is statistically
significant under both Bonferroni- and BH-FDR-corrected DeLong tests.

## Repository layout

| Path | What it contains |
|---|---|
| [`physics_prior.py`](./physics_prior.py) | Analytic hemoglobin prior ($P_\text{blood}$, $\Phi$) |
| [`models_pi.py`](./models_pi.py), [`datasets_pi.py`](./datasets_pi.py) | 5-channel input-fusion and distillation variants |
| [`train_stage2_pi.py`](./train_stage2_pi.py), [`train_stage2_pi_distill.py`](./train_stage2_pi_distill.py) | Training entry points |
| [`stats_pi.py`](./stats_pi.py), [`metrics_pi.py`](./metrics_pi.py) | DeLong, BCa, BH-FDR, per-class AUC |
| [`reproduce.sh`](./reproduce.sh) | One-command reproduction |
| [`setup_kvasir_capsule.py`](./setup_kvasir_capsule.py) | Kvasir-Capsule preparation |
| [`figures/`](./figures) | Final paper figures (PDF + PNG) and the scripts that generate them |
| [`benchmark/`](./benchmark) | **GalKva-2026** paired cross-vendor benchmark (splits, evaluator, schema, reference results) |
| [`code/`](./code) | Full training / ablation / baseline code (capsule_endoscopy, temporal, galar) |
| [`docs/`](./docs) | Summary reports for the channel ablation, BiomedCLIP / DINOv2 baselines, calibration, Grad-CAM/prior overlap, per-patient lift |
| [`scripts/`](./scripts) | Released SLURM job scripts |

## Dependencies

The training scripts (`train_stage2_pi.py`, `train_stage2_pi_distill.py`,
`figures/eval_test_predictions.py`) import `datasets.py`, `models.py` and
`utils.py` from the upstream **`gastroscopy_code_package`**, which is **not
bundled in this repo**. Provide it on disk and point `GASTRO_DIR` (or the
`--gastroscopy_code_dir` flag) at it. A vendored, version-pinned copy will be
included in the public release; until then, set `GASTRO_DIR` to your local
checkout.

## Quick start

There are two reproduction paths — a one-GPU smoke test and the published
headline run. Pick the right one:

```bash
pip install -r requirements.txt

# Stage Kvasir-Capsule (downloaded + accepted from datasets.simula.no)
python setup_kvasir_capsule.py \
    --images_dir /path/to/kvasir-capsule-labeled-images \
    --metadata   /path/to/metadata.csv \
    --out_dir    ./stage2_data --mode hardlink

# (A) SMOKE / sanity — single seed (42), 3 input variants. Confirms the
#     pipeline runs end to end and emits predictions + figures. ~10–12 GPU-h.
GASTRO_DIR=/path/to/gastroscopy_code_package bash reproduce.sh

# (B) HEADLINE — the published n=10-seed canonical-split runs (paper §4.9),
#     pinned to benchmark/canonical_splits/kvasir_split_manifest_2026-05-18.json.
sbatch scripts/submit_effb0_canonical_n10.sbatch
```

For ablations, ConvNeXt-Tiny / ResNet-18 cross-backbone replication,
BiomedCLIP / DINOv2 linear-probe baselines, and the C2 temporal +
C3 autoencoder-residual streams, see the scripts under
[`code/capsule_endoscopy/`](./code/capsule_endoscopy) and
[`code/temporal/`](./code/temporal).

## Cross-vendor benchmark — GalKva-2026

We release [`benchmark/`](./benchmark) as a paired cross-vendor
evaluation suite (Kvasir-Capsule × Galar). It provides:

- a canonical 43-patient split manifest with content hash
  (test = 6,423 frames),
- a 6-class evaluable intersection between the two datasets,
- a JSON submission schema and a reference *submission formatter*
  (`benchmark/evaluate.py`) that validates a submission of self-reported
  macro-AUCs and computes the retention ratio
  $\Delta_\text{Galar}/\Delta_\text{Kvasir}$. Prediction-level statistics
  (DeLong, BCa, McNemar) are computed upstream by `stats_pi.py`, not by the
  formatter — see [`benchmark/README.md`](./benchmark/README.md),
- this paper's reference submission as the baseline leaderboard entry.

See [`benchmark/README.md`](./benchmark/README.md) for the submission
protocol.

## Citation

```bibtex
@article{Yang2026PI,
  title   = {Training-Time Optical Priors for Wireless Capsule Endoscopy
             Classification: Hemoglobin-Aware Input Fusion with
             Cross-Vendor Evaluation},
  author  = {Yang, Chengshuai and Xing, Lei and Alam, Keyaan Zawad and
             Entin, Gregory and Vemulapalli, Roopa and Casey, Lisa and
             Zaman, Raiyan Tripti},
  journal = {Medical Image Analysis},
  year    = {2026},
  note    = {Submitted}
}
```

## License

Code is released under the MIT License (see [`LICENSE`](./LICENSE)).
Dataset access: Kvasir-Capsule (research-use license, upstream); Galar
(check upstream license).
