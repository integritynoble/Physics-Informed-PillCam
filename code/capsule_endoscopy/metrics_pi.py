"""Robust classification metrics + dataset-alignment helpers.

Replaces the original gastroscopy_code_package summarize_classification with
a version that does NOT raise when y_true / y_pred fail to span all classes.

The original implementation calls sklearn's classification_report without
the `labels` argument; sklearn then infers labels from unique(y_true | y_pred)
and raises ValueError if the inferred count != len(target_names). Under our
class-stratified Kvasir-Capsule split, val carries 9 classes and test carries
11, so this exception fires the moment the model fails to predict every class
at least once on a val/test pass — which is normal early in training.

This shadow always passes labels=list(range(len(class_names))) so the report
is well-defined for the full 14-class set regardless of which appear in y_true
or y_pred. Missing classes get zero-row entries.

Also exposes ensure_class_folders() — must be called AFTER constructing the
train dataset and BEFORE constructing val/test, so that ImageFolder builds
identical class_to_idx mappings across splits. Without it, val (9 classes)
and test (11 classes) get different alphabetical index assignments than
train (14 classes), and the model is evaluated against scrambled labels —
which manifests as catastrophic train/val divergence (val CE rising while
train CE drops).
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List

from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)


def ensure_class_folders(split_dir: str, class_names: Iterable[str]) -> None:
    """Create an empty folder for every class_name under split_dir.

    ImageFolder includes empty subdirectories in its class_to_idx mapping,
    so this aligns the label-index of (e.g.) "Erosion" across train (14
    real folders), val (9 real folders + 5 empty), and test (11 real + 3
    empty). Idempotent — never deletes anything.
    """
    for cls in class_names:
        os.makedirs(os.path.join(split_dir, cls), exist_ok=True)


def summarize_classification(y_true: List[int], y_pred: List[int],
                              class_names: List[str]) -> Dict:
    """Compute classification metrics with two macro-F1 variants.

    Why two macros — `paper_outline.md §3.4` says the headline split treats the
    three single-video classes (Ampulla, Blood-hematin, Polyp) as training-only;
    they have zero support in val/test. The full 14-class macro-F1 averages them
    in as 0, which mechanically drags the headline metric down by 3/14. We
    therefore also report `macro_f1_evaluable` — a macro over only those
    classes with support > 0 in y_true. That is the headline number for
    Tables 2/3; `macro_f1` (full 14-class) is kept for backward compatibility
    and as a conservative bound.
    """
    labels = list(range(len(class_names)))
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Per-class support and F1 from the report dict.
    supports = [int(report[c]["support"]) for c in class_names]
    per_class_f1 = [report[c]["f1-score"] for c in class_names]
    evaluable_idx = [i for i, s in enumerate(supports) if s > 0]
    if evaluable_idx:
        macro_f1_evaluable = float(sum(per_class_f1[i] for i in evaluable_idx) / len(evaluable_idx))
    else:
        macro_f1_evaluable = 0.0
    evaluable_classes = [class_names[i] for i in evaluable_idx]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels,
                              average="macro", zero_division=0),
        "macro_f1_evaluable": macro_f1_evaluable,
        "evaluable_classes": evaluable_classes,
        "weighted_f1": f1_score(y_true, y_pred, labels=labels,
                                 average="weighted", zero_division=0),
        "confusion_matrix": cm,
        "report": report,
    }
