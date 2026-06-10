# GalKva-2026: A Paired Cross-Vendor Capsule Endoscopy Benchmark

**Release:** v1.0 (2026-05-16)
**Citation:** Yang et al., *Computational Imaging Priors for Wireless
Capsule Endoscopy: Monte Carlo-Guided Hemoglobin Mapping with Temporal
Refinement*, Medical Image Analysis 2026 (submitted).
**Companion paper repo:** <https://github.com/integritynoble/GI_Multi_Task>
**Submission portal:** open an issue with a JSON conforming to
[`submission_schema.json`](./submission_schema.json); we run the standard
evaluation and update [`../LEADERBOARD.md`](../LEADERBOARD.md).

---

## Motivation

Most published capsule endoscopy AI is evaluated on a single dataset
acquired from a single capsule vendor. The strongest claims about
"a new architecture / pretraining / prior helps" therefore conflate
**method improvement** with **dataset-specific calibration**.
GalKva-2026 closes this gap by providing a paired, ready-to-use
evaluation suite on two public datasets acquired from disjoint
vendor mixes:

| | **Kvasir-Capsule** (Smedsrud et al. 2021) | **Galar** (Scientific Data 2025) |
|---|---|---|
| Vendor | PillCam SB2/SB3 | MiroCam, PillCam SB2/SB3/Colon2, Olympus |
| Country | Norway | Spain |
| Source frames | 47\,238 (43 patients, 14 classes) | 3.51 M (80 videos, 32 columns multi-label) |
| Annotation style | Per-frame single-label | Per-frame multi-label one-hot |
| License | Research-use (script-only redistribution from this repo) | Verify upstream license; default to script-only redistribution |

The benchmark exposes the **6-class cross-dataset evaluable
intersection** (Angiectasia, Blood-fresh, Lymphangiectasia,
Normal-clean-mucosa, Polyp, Ulcer) and, on Galar, the **13-class
single-dataset extension**. The benchmark's evaluation script
computes the standard MedIA / IEEE TMI reporting suite (cross-seed
macro-AUC, paired DeLong, per-class p-values with both Bonferroni
and BH-FDR corrections, percentile and BCa bootstrap CIs,
McNemar at argmax).

---

## Quickstart for benchmark users

```bash
# 1. Clone this repo
git clone https://github.com/integritynoble/GI_Multi_Task
cd GI_Multi_Task

# 2. Stage the data (you bring your own Kvasir + Galar downloads)
python GI_project/code/Capsule-Endoscopy/setup_kvasir_capsule.py \
    --images_zip <path>/kvasir-capsule-labeled-images.zip \
    --metadata   <path>/metadata.csv \
    --out_dir    data/kvasir_eval --mode hardlink
python GI_project/code/galar/setup_galar.py \
    --galar_root <path>/galar_raw \
    --out_dir    data/galar_eval \
    --mapping    GI_project/code/galar/galar_class_mapping.json \
    --mode       hardlink

# 3. Run your method on both datasets; emit a submission JSON
python your_method.py --eval_dir data/kvasir_eval/test --out kvasir_results.json
python your_method.py --eval_dir data/galar_eval/test  --out galar_results.json

# 4. Evaluate against the benchmark
python benchmark/evaluate.py \
    --kvasir kvasir_results.json --galar galar_results.json \
    --submitter "Your Name" --method "MethodName-v1" \
    --out submissions/your_method_v1.md

# 5. Open a pull request adding the submission JSON to submissions/
#    We'll merge after a sanity-check pass and update LEADERBOARD.md.
```

---

## Three reasons to use GalKva-2026 instead of either dataset alone

1. **Paired evaluation forces architecture-vs-dataset disambiguation.**
   A method that lifts on Kvasir but not on Galar is *dataset-specific*,
   not generally improved. The benchmark prints both numbers
   side-by-side and refuses to headline an unpaired result.

2. **The retention ratio
   $\Delta_\mathrm{Galar} / \Delta_\mathrm{Kvasir}$ is the explicit
   transferability metric.** Methods can be ranked not only by
   in-domain lift but by how much of that lift survives cross-vendor
   zero-shot. Our baseline (Yang et al. 2026) lands at retention =
   $+0.60$ on ConvNeXt-Tiny (4/6 seeds positive).

3. **The 6-class cross-dataset intersection is reviewable.**
   The class mapping
   ([`class_mapping.json`](./class_mapping.json)) is conservative
   (drops 1 Galar label that has no Kvasir analogue) and explicit
   (every Galar `_drop` label is listed). Reviewers can audit the
   mapping and re-run with a stricter or looser version.

---

## Files in this directory

| File | Purpose |
|---|---|
| [`README.md`](./README.md) | This document |
| [`class_mapping.json`](./class_mapping.json) | Canonical Kvasir-Galar 6-class intersection (and 13-class Galar extension) |
| [`submission_schema.json`](./submission_schema.json) | JSON Schema that submissions must conform to |
| [`evaluate.py`](./evaluate.py) | Evaluator: takes a submission JSON, computes standard metrics, emits a Markdown report |
| [`reference_results/`](./reference_results/) | Our baseline reference submissions (RGB + 5-channel PI + distillation, all 6 seeds, both backbones) |
| [`../LEADERBOARD.md`](../LEADERBOARD.md) | Current public results ranked by Galar retention |

---

## License and redistribution

- This benchmark **code** (scripts, evaluator, JSON schemas) is released
  under MIT.
- Kvasir-Capsule frames must be downloaded from
  <https://datasets.simula.no/kvasir-capsule/> by each user (research-
  use license, not redistributable from us).
- Galar frames must be downloaded from the figshare release
  (DOI: see Scientific Data 2025); we redistribute only the staging
  script and the class mapping JSON, not the raw frames, pending
  upstream license verification. Our own reference predictions
  (numerical outputs of our models on these datasets) are MIT.

---

## Versioning

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026-05-16 | Initial release. Kvasir-Capsule full + Galar 10-video subset (15\,298 frames). |
| v1.1 (planned) | TBD | Full Galar 80-video release (3.51 M frames) once we complete extraction. |
| v2.0 (planned) | TBD | Adds KID Atlas (pending access approval) as a third cross-vendor cohort. |

Pinning a published submission to a specific benchmark version is
encouraged --- cite the version tag in your reported numbers so
future readers can reproduce against the same data slice.
