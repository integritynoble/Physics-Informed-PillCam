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
single-dataset extension**. v1.0 scoring is on the **6-class
evaluable intersection**; the 13-class Galar extension is reported
separately and is not part of the v1.0 retention ranking.

**What the evaluator does.** [`evaluate.py`](./evaluate.py) is a
*submission formatter*: it ingests a submission JSON of self-reported
per-seed and per-class macro-AUCs (validated against
[`submission_schema.json`](./submission_schema.json)), computes the
cross-vendor **retention ratio**
$\Delta_\mathrm{Galar}/\Delta_\mathrm{Kvasir}$ and its band, and emits
a Markdown leaderboard entry. It does **not** ingest per-frame
predictions or recompute DeLong / BCa / McNemar — those statistics are
produced upstream from prediction-level outputs by
[`../stats_pi.py`](../stats_pi.py) and reported by the submitter. A
prediction-level evaluator is planned for a future benchmark version.

---

## Quickstart for benchmark users

```bash
# 1. Clone this repo
git clone https://github.com/integritynoble/Physics-Informed-PillCam
cd Physics-Informed-PillCam

# 2. Stage the data (you bring your own Kvasir + Galar downloads)
python setup_kvasir_capsule.py \
    --images_dir <path>/kvasir-capsule-labeled-images \
    --metadata   <path>/metadata.csv \
    --out_dir    data/kvasir_eval --mode hardlink
python code/galar/setup_galar.py \
    --galar_root <path>/galar_raw \
    --out_dir    data/galar_eval \
    --mapping    benchmark/class_mapping.json \
    --mode       hardlink

# 3. Run your method on both datasets and assemble ONE submission JSON
#    conforming to benchmark/submission_schema.json (self-reported per-seed
#    and per-class macro-AUCs for kvasir and galar). See
#    benchmark/reference_results/yang2026_convnext_pi.json for a worked example.

# 4. Format the submission into a leaderboard entry (computes retention ratio)
python benchmark/evaluate.py \
    --submission submissions/your_method_v1.json \
    --out        submissions/your_method_v1.md

# 5. Open a pull request adding your_method_v1.json to submissions/
#    We'll merge after a sanity-check pass and update LEADERBOARD.md.
```

> The evaluator consumes a **single submission JSON**, not raw predictions
> or two per-dataset result files. The schema, an end-to-end worked example,
> and the exact required fields are in `submission_schema.json` and
> `reference_results/yang2026_convnext_pi.json`.

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
| [`evaluate.py`](./evaluate.py) | Submission formatter: takes one submission JSON, computes the retention ratio + band, emits a Markdown leaderboard entry (does not recompute DeLong/BCa/McNemar — see note above) |
| [`canonical_splits/kvasir_split_manifest_2026-05-18.json`](./canonical_splits/kvasir_split_manifest_2026-05-18.json) | Canonical patient-disjoint Kvasir split (train=31,820 / val=8,986 / test=6,423) with SHA-256 content hash; the authoritative split for headline reproduction |
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
| v1.0 | 2026-05-16 | Initial release. Kvasir-Capsule (canonical split per [`canonical_splits/kvasir_split_manifest_2026-05-18.json`](./canonical_splits/kvasir_split_manifest_2026-05-18.json), test=6,423) + Galar 10-video subset (15\,298 frames). Scoring on the 6-class evaluable intersection. |
| v1.1 (planned) | TBD | Full Galar 80-video release (3.51 M frames) once we complete extraction. |
| v2.0 (planned) | TBD | Adds KID Atlas (pending access approval) as a third cross-vendor cohort. |

Pinning a published submission to a specific benchmark version is
encouraged --- cite the version tag in your reported numbers so
future readers can reproduce against the same data slice.
