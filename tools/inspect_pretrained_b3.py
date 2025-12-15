#!/usr/bin/env python3
"""
Simple inspector for the pretrained SPNv2 EfficientNet-B3 checkpoint.

- Loads the .pth.tar file from outputs/pretrained
- Prints top-level keys
- Prints number of state_dict entries and the first few parameter shapes
"""

import sys
from pathlib import Path
import torch

# ----------------------------------------------------------------------
# Setup paths so we can import project modules if needed later
# ----------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Optional: alias core.utils as a top-level 'utils' package
# (this makes imports like `from utils.postprocess import ...` work
# if you extend this script later)
import core.utils as core_utils  # type: ignore
sys.modules["utils"] = core_utils

from core.config import cfg  # not strictly needed now, but handy if you extend


def main():
    ckpt_path = ROOT_DIR / "outputs" / "pretrained" / "spnv2_efficientnetb3_fullconfig_offline.pth.tar"
    print("Loading checkpoint from:", ckpt_path)

    # Load checkpoint on CPU just for inspection
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Some checkpoints store everything directly, some under 'state_dict'
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        print("Checkpoint has 'state_dict' key.")
        state_dict = ckpt["state_dict"]
    else:
        print("Checkpoint appears to be a raw state_dict.")
        state_dict = ckpt

    print("Checkpoint top-level keys:", ckpt.keys() if isinstance(ckpt, dict) else type(ckpt))

    # Basic stats on parameters
    print("\nNumber of state_dict entries:", len(state_dict))
    print("First 10 keys and shapes:")
    for k in list(state_dict.keys())[:10]:
        v = state_dict[k]
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: type={type(v)}")

    # Optional: total parameter count (just for curiosity)
    total_params = 0
    for v in state_dict.values():
        if isinstance(v, torch.Tensor):
            total_params += v.numel()
    print(f"\nTotal number of tensor parameters in state_dict: {total_params/1e6:.2f} M (approx)")


if __name__ == "__main__":
    main()
