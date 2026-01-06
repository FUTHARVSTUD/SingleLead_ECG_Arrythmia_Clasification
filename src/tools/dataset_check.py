from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.configs.splits import MITBIH_RECORDS


def check_mitbih(data_dir: Path):
    missing = []
    for record in MITBIH_RECORDS:
        for ext in ("dat", "hea", "atr"):
            path = data_dir / f"{record}.{ext}"
            if not path.exists():
                missing.append(str(path))
    return {"dataset": "mitbih", "records": len(MITBIH_RECORDS), "missing": missing}


def check_incart(data_dir: Path):
    records = sorted(p.stem for p in data_dir.glob("I*.hea"))
    missing = []
    for stem in records:
        if not (data_dir / f"{stem}.mat").exists():
            missing.append(str(data_dir / f"{stem}.mat"))
    return {"dataset": "incart", "records": len(records), "missing": missing}


def main():
    parser = argparse.ArgumentParser(description="Dataset presence check")
    parser.add_argument("--dataset", type=str, required=True, choices=["mitbih", "incart"])
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    if args.dataset == "mitbih":
        summary = check_mitbih(data_dir)
    else:
        summary = check_incart(data_dir)
    print(json.dumps(summary, indent=2))
    if summary["missing"]:
        raise SystemExit(f"Missing files detected: {summary['missing'][:5]} ...")


if __name__ == "__main__":
    main()
