"""AAMI beat type mapping utilities."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]
CLASS_TO_INDEX = {c: i for i, c in enumerate(AAMI_CLASSES)}

AAMI_MAPPING: Dict[str, str] = {
    # Normal and bundle branch block beats
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",
    # Supraventricular ectopic
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",
    # Ventricular ectopic
    "V": "V",
    "E": "V",
    "!": "V",
    # Fusion of ventricular and normal
    "F": "F",
    # Unclassifiable / paced beats
    "Q": "Q",
    "f": "Q",
    "|": "Q",
    "~": "Q",
    "x": "Q",
    "P": "Q",  # paced
}

IGNORED_SYMBOLS = {"", "?", "[", "]", "-", "*"}


def map_symbol(symbol: str) -> Optional[int]:
    """Convert a raw beat annotation symbol to an AAMI class index.

    Returns None if the symbol should be ignored.
    """
    if symbol in IGNORED_SYMBOLS:
        return None
    cls = AAMI_MAPPING.get(symbol)
    if cls is None:
        return None
    return CLASS_TO_INDEX[cls]


def class_distribution(labels: Iterable[int]) -> Dict[str, int]:
    counts = {c: 0 for c in AAMI_CLASSES}
    for idx in labels:
        counts[AAMI_CLASSES[idx]] += 1
    return counts
