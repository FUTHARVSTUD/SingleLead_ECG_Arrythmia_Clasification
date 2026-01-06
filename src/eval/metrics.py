from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from src.data.aami import AAMI_CLASSES


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict:
    labels = list(range(len(AAMI_CLASSES)))
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    precision, recall, _, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "class_labels": AAMI_CLASSES,
        "macro_f1": float(macro_f1),
        "per_class_f1": per_class_f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm.tolist(),
    }
