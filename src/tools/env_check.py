from __future__ import annotations

import json
import platform

import numpy as np
import scipy
import torch
import wfdb


def main():
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "wfdb": wfdb.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
