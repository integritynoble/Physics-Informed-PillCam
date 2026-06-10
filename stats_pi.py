"""Paired statistical tests for the §3.5 reporting block.

Operates on the test_predictions.json files produced by
figures/eval_test_predictions.py (so the GPU only runs once per checkpoint).

Tests implemented (all paired — same test frames in both arms):
    delong_paired(y_true, scores_a, scores_b)
        DeLong et al. (1988) test for the difference between two paired
        AUCs on the same labels. Returns (auc_a, auc_b, z, p_two_sided).
        Sun & Xu (2014) fast formulation; pure numpy, O(n log n).

    mcnemar_paired(correct_a, correct_b)
        McNemar test for paired classification correctness on the same
        frames. Returns (b, c, statistic, p_two_sided) where b is the
        count where A is wrong but B is right and c is the reverse;
        uses the exact binomial when b+c < 25 and the chi-square with
        continuity correction otherwise.

    bootstrap_ci(metric_fn, *arrays, n=1000, seed=42, ci=0.95)
        Frame-stratified resampling 95% CI for any scalar metric_fn
        applied to the supplied arrays.

    bonferroni_correct(p_values, n=None)
        Trivial Bonferroni multiple-comparison correction.

CLI usage (paired comparison from the dumped test_predictions.json):

    python stats_pi.py \\
        --a D:/kvasir_capsule/outputs/stage2_rgb_effb0/test_predictions.json \\
        --b D:/kvasir_capsule/outputs/stage2_pi_effb0/test_predictions.json \\
        --label_a "RGB-only" --label_b "+MC prior" \\
        --out D:/kvasir_capsule/outputs/stats_rgb_vs_pi.json

The output JSON carries per-class DeLong + McNemar + bootstrap-CI plus the
Bonferroni-corrected p-values across the 11 evaluable classes (or whatever
classes have non-zero support in y_true).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as _scipy_stats


# ---------------------------------------------------------------------------
# DeLong (Sun & Xu 2014 fast implementation)
# ---------------------------------------------------------------------------

def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank with ties; pure numpy. x is 1-D."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1  # midrank, 1-indexed
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _delong_covariance(predictions_sorted_transposed: np.ndarray, m: int):
    """Returns (auc_vector_K, covariance_KxK) for K predictors on the same samples.

    predictions_sorted_transposed: shape (K, N). The first m columns are the
    positive samples, the rest are negative samples. Implements the fast
    O(N log N) DeLong covariance from Sun & Xu (2014).
    """
    K, N = predictions_sorted_transposed.shape
    n = N - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((K, m), dtype=float)
    ty = np.empty((K, n), dtype=float)
    tz = np.empty((K, N), dtype=float)
    for r in range(K):
        tx[r] = _compute_midrank(positive_examples[r])
        ty[r] = _compute_midrank(negative_examples[r])
        tz[r] = _compute_midrank(predictions_sorted_transposed[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if sx.ndim == 0:
        sx = np.array([[sx]])
        sy = np.array([[sy]])
    cov = sx / m + sy / n
    return aucs, cov


def delong_paired(y_true: Sequence[int], scores_a: Sequence[float],
                   scores_b: Sequence[float]) -> dict:
    """Paired DeLong test for the difference (auc_a - auc_b).

    y_true: 0/1 labels. scores_*: continuous scores (higher = more positive).
    Returns dict with keys: auc_a, auc_b, delta, var, z, p_two_sided, n_pos, n_neg.
    """
    y = np.asarray(y_true).astype(int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    if not (len(y) == len(sa) == len(sb)):
        raise ValueError(f"length mismatch: y={len(y)} a={len(sa)} b={len(sb)}")

    pos = (y == 1)
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auc_a": float("nan"), "auc_b": float("nan"),
                "delta": float("nan"), "var": float("nan"),
                "z": float("nan"), "p_two_sided": float("nan"),
                "n_pos": n_pos, "n_neg": n_neg, "note": "single-class y_true"}

    # Reorder so positives come first, both predictors in lockstep.
    order = np.concatenate([np.where(pos)[0], np.where(~pos)[0]])
    pred = np.stack([sa[order], sb[order]], axis=0)
    aucs, cov = _delong_covariance(pred, m=n_pos)

    delta = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = delta / math.sqrt(var)
        p = 2.0 * (1.0 - _scipy_stats.norm.cdf(abs(z)))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": delta, "var": var, "z": float(z),
            "p_two_sided": float(p), "n_pos": n_pos, "n_neg": n_neg}


# ---------------------------------------------------------------------------
# McNemar (paired classification correctness)
# ---------------------------------------------------------------------------

def mcnemar_paired(correct_a: Sequence[int], correct_b: Sequence[int]) -> dict:
    """Paired McNemar test on per-frame correctness.

    correct_a/b: 0/1 vectors of equal length.
    Returns dict: b (a wrong, b right), c (a right, b wrong),
                  statistic, p_two_sided, method ('exact' or 'chi2_continuity').
    """
    a = np.asarray(correct_a).astype(int)
    bb = np.asarray(correct_b).astype(int)
    if a.shape != bb.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {bb.shape}")
    b_count = int(((a == 0) & (bb == 1)).sum())
    c_count = int(((a == 1) & (bb == 0)).sum())
    n = b_count + c_count
    if n == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_two_sided": 1.0,
                "method": "trivial"}
    if n < 25:
        # Exact binomial two-sided test under H0: b ~ Binomial(n, 0.5)
        k = min(b_count, c_count)
        p = 2.0 * _scipy_stats.binom.cdf(k, n, 0.5)
        p = min(1.0, p)
        return {"b": b_count, "c": c_count, "statistic": float(k),
                "p_two_sided": float(p), "method": "exact"}
    # Chi-square with continuity correction.
    chi2 = (abs(b_count - c_count) - 1.0) ** 2 / n
    p = 1.0 - _scipy_stats.chi2.cdf(chi2, df=1)
    return {"b": b_count, "c": c_count, "statistic": float(chi2),
            "p_two_sided": float(p), "method": "chi2_continuity"}


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(metric_fn: Callable[..., float], *arrays: Sequence,
                  n: int = 1000, seed: int = 42, ci: float = 0.95) -> dict:
    """Frame-stratified bootstrap CI for an arbitrary metric.

    metric_fn(*arrays) -> scalar. arrays are aligned 1-D vectors of equal
    length; we resample the same indices across all of them.
    """
    rng = np.random.default_rng(seed)
    arrs = [np.asarray(a) for a in arrays]
    N = len(arrs[0])
    if not all(len(a) == N for a in arrs):
        raise ValueError("all arrays must be the same length")

    point = float(metric_fn(*arrs))
    samples = np.empty(n, dtype=float)
    for i in range(n):
        idx = rng.integers(0, N, size=N)
        try:
            samples[i] = float(metric_fn(*[a[idx] for a in arrs]))
        except Exception:
            samples[i] = float("nan")
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "n_resamples": 0}
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(samples, alpha))
    hi = float(np.quantile(samples, 1.0 - alpha))
    return {"point": point, "lo": lo, "hi": hi, "n_resamples": int(samples.size)}


# ---------------------------------------------------------------------------
# Bonferroni
# ---------------------------------------------------------------------------

def bonferroni_correct(p_values: Sequence[float], n: Optional[int] = None) -> List[float]:
    """Trivial Bonferroni: p_corrected = min(1, p * n) for each test.
    n defaults to len(p_values) (correct over the supplied family)."""
    p = np.asarray(p_values, dtype=float)
    n_eff = int(n) if n is not None else len(p)
    return [float(min(1.0, v * n_eff)) for v in p]


# ---------------------------------------------------------------------------
# CLI: paired-comparison report from two test_predictions.json files
# ---------------------------------------------------------------------------

def _load_predictions(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pairwise_report(a: dict, b: dict, label_a: str, label_b: str,
                      n_bootstrap: int = 1000) -> dict:
    if a["class_names"] != b["class_names"]:
        raise SystemExit("class_names mismatch between a and b")
    if a["labels"] != b["labels"]:
        raise SystemExit("labels mismatch between a and b — were both eval runs on the same test split?")
    classes = a["class_names"]
    y_true = np.asarray(a["labels"])
    probs_a = np.asarray(a["probs"])
    probs_b = np.asarray(b["probs"])
    preds_a = np.asarray(a["preds"])
    preds_b = np.asarray(b["preds"])
    correct_a = (preds_a == y_true).astype(int)
    correct_b = (preds_b == y_true).astype(int)

    # Per-class DeLong (one-vs-rest), bootstrap AUCs
    per_class = {}
    delong_ps: List[float] = []
    delong_keys: List[str] = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            per_class[cls] = {"note": "no positive support in test split"}
            continue
        # DeLong paired
        d = delong_paired(y_bin, probs_a[:, i], probs_b[:, i])
        # Bootstrap CIs on each AUC and on delta
        from sklearn.metrics import roc_auc_score
        def _auc(yt, sc):
            try:
                return float(roc_auc_score(yt, sc))
            except ValueError:
                return float("nan")
        ci_a = bootstrap_ci(_auc, y_bin, probs_a[:, i], n=n_bootstrap)
        ci_b = bootstrap_ci(_auc, y_bin, probs_b[:, i], n=n_bootstrap)
        ci_delta = bootstrap_ci(lambda yt, sa, sb: _auc(yt, sa) - _auc(yt, sb),
                                 y_bin, probs_a[:, i], probs_b[:, i], n=n_bootstrap)
        per_class[cls] = {
            "delong": d,
            "auc_a_ci95": ci_a, "auc_b_ci95": ci_b,
            "delta_auc_ci95": ci_delta,
            "support_pos": int(y_bin.sum()), "support_neg": int(len(y_bin) - y_bin.sum()),
        }
        delong_ps.append(d["p_two_sided"])
        delong_keys.append(cls)

    # Bonferroni across the evaluable per-class DeLong tests
    if delong_ps:
        adj = bonferroni_correct(delong_ps)
        for cls, p_adj in zip(delong_keys, adj):
            per_class[cls]["delong"]["p_bonferroni"] = p_adj

    # McNemar at the operating point (argmax)
    mc = mcnemar_paired(correct_a, correct_b)

    # Macro-AUC delta (over evaluable classes only, matching paper §3.4 headline)
    from sklearn.metrics import roc_auc_score
    macro_a, macro_b = [], []
    for i in range(len(classes)):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            continue
        try:
            macro_a.append(float(roc_auc_score(y_bin, probs_a[:, i])))
            macro_b.append(float(roc_auc_score(y_bin, probs_b[:, i])))
        except ValueError:
            pass

    macro = {"a": float(np.mean(macro_a)) if macro_a else float("nan"),
             "b": float(np.mean(macro_b)) if macro_b else float("nan")}
    macro["delta"] = macro["b"] - macro["a"] if not (np.isnan(macro["a"]) or np.isnan(macro["b"])) else float("nan")

    return {
        "label_a": label_a, "label_b": label_b,
        "n_frames": int(len(y_true)),
        "evaluable_classes": delong_keys,
        "per_class": per_class,
        "mcnemar_overall": mc,
        "macro_auc_evaluable": macro,
    }


def main():
    # Force stdout to UTF-8 so Unicode (Δ, ±, etc.) survives Windows cp1252.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="test_predictions.json for arm A")
    ap.add_argument("--b", required=True, help="test_predictions.json for arm B")
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--n_bootstrap", type=int, default=1000)
    args = ap.parse_args()

    a = _load_predictions(args.a)
    b = _load_predictions(args.b)
    rep = _pairwise_report(a, b, args.label_a, args.label_b, args.n_bootstrap)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)

    # Brief stdout summary so the run is human-readable in the log.
    print(f"[stats] {args.label_a} vs {args.label_b}  on {rep['n_frames']} frames, "
          f"{len(rep['evaluable_classes'])} evaluable classes")
    m = rep["macro_auc_evaluable"]
    print(f"[stats] macro-AUC (evaluable):  A={m['a']:.4f}  B={m['b']:.4f}  "
          f"delta={m['delta']:+.4f}")
    print(f"[stats] McNemar overall:  b={rep['mcnemar_overall']['b']}  "
          f"c={rep['mcnemar_overall']['c']}  "
          f"p={rep['mcnemar_overall']['p_two_sided']:.4g}  "
          f"({rep['mcnemar_overall']['method']})")
    print(f"[stats] per-class DeLong p-values (Bonferroni-corrected over "
          f"{len(rep['evaluable_classes'])} evaluable classes):")
    for cls in rep["evaluable_classes"]:
        d = rep["per_class"][cls]["delong"]
        delta = d["delta"]
        p = d["p_two_sided"]; p_adj = d.get("p_bonferroni", float("nan"))
        marker = "*" if (not np.isnan(p_adj) and p_adj < 0.05) else ""
        print(f"    {cls:<24} ΔAUC={delta:+.4f}  p={p:.4g}  p_bonf={p_adj:.4g} {marker}")
    print(f"[stats] wrote {args.out}")


if __name__ == "__main__":
    main()
