# GalKva-2026 Leaderboard

Cross-vendor capsule-endoscopy benchmark (Kvasir-Capsule × Galar), ranked by
Galar retention ratio $\Delta_\mathrm{Galar}/\Delta_\mathrm{Kvasir}$ on the
6-class evaluable intersection. See [`benchmark/README.md`](./benchmark/README.md)
for the submission protocol and what the evaluator does.

| Rank | Method | Trained on | Kvasir macro-AUC | Galar macro-AUC | Retention | Band | Seeds |
|---|---|---|---|---|---|---|---|
| 1 | Yang2026 ConvNeXt-Tiny +PI (5-ch input fusion) | Kvasir-Capsule | 0.764 ± 0.015 | 0.745 ± 0.022 | +0.61 | Partial (0.30–0.75) | 6 |

**Baseline entry source:**
[`benchmark/reference_results/yang2026_convnext_pi.json`](./benchmark/reference_results/yang2026_convnext_pi.json).

> ⚠ **Provisional baseline.** As released, the baseline row carries two caveats
> documented in the submission's own `notes`: (1) the Kvasir per-class AUCs are
> placeholders drawn from the paper's Table 4 rather than recomputed from the
> canonical split, and (2) `n_test_frames` (8,623) predates the canonical split
> manifest (test = 6,423; see
> [`benchmark/canonical_splits/kvasir_split_manifest_2026-05-18.json`](./benchmark/canonical_splits/kvasir_split_manifest_2026-05-18.json)).
> Both will be reconciled by regenerating the reference submission from the
> canonical split before the public release is tagged.

## How to appear here

1. Build a submission JSON (see `benchmark/submission_schema.json` and the
   reference example).
2. Run `python benchmark/evaluate.py --submission <your.json> --out <your.md>`.
3. Open a pull request adding your JSON to `submissions/`.
