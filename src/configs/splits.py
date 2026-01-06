"""Dataset split definitions and helper utilities."""
from __future__ import annotations

from typing import Dict, List

# Canonical MIT-BIH DS1 / DS2 split used by the AAMI recommendation.
MITBIH_DS1: List[str] = [
    "101",
    "106",
    "108",
    "109",
    "112",
    "114",
    "115",
    "116",
    "118",
    "119",
    "122",
    "124",
    "201",
    "203",
    "205",
    "207",
    "208",
    "209",
    "215",
    "220",
    "223",
    "230",
]

MITBIH_DS2: List[str] = [
    "100",
    "103",
    "105",
    "111",
    "113",
    "117",
    "121",
    "123",
    "200",
    "202",
    "210",
    "212",
    "213",
    "214",
    "219",
    "221",
    "222",
    "228",
    "231",
    "232",
    "233",
    "234",
]

MITBIH_RECORDS = sorted(set(MITBIH_DS1 + MITBIH_DS2))

# INCART uses a continuous recording per patient. We randomly split by record index
# when writing processed files; exact split happens during preprocessing.
INCART_ADAPT_RATIO = 0.6  # portion used as unlabeled adaptation set (AdaBN)


def dataset_records(dataset: str) -> List[str]:
    dataset = dataset.lower()
    if dataset == "mitbih":
        return MITBIH_RECORDS
    if dataset == "incart":
        # INCART filenames are zero-padded.
        return [f"I{idx:04d}" for idx in range(1, 76)]
    raise ValueError(f"Unknown dataset '{dataset}'")


def describe() -> Dict[str, List[str]]:
    """Return a dictionary with split metadata."""
    return {
        "mitbih_ds1": MITBIH_DS1,
        "mitbih_ds2": MITBIH_DS2,
        "incart_all": dataset_records("incart"),
    }
