import sys, os
from pathlib import Path
import importlib.util

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# ---- Load ShirtRoeDataset directly from its file (avoid core.dataset __init__) ----
shirt_path = ROOT_DIR / "core" / "dataset" / "shirt_dataset.py"
spec = importlib.util.spec_from_file_location("shirt_dataset_module", shirt_path)
shirt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shirt_module)
ShirtRoeDataset = shirt_module.ShirtRoeDataset

from core.utils.mc_dropout import mc_infer
from core.utils.kalman import SimpleKalmanCV3D


# ----- Dummy pose model for now -----
class DummyPoseModel(nn.Module):
    """
    Stand-in for SPNv2 EfficientDet-D0.

    Given an image [B, 3, H, W], returns dict:
      - 't': [B, 3]  (fake translation)
      - 'q': [B, 4]  (fake quaternion)
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 64 * 64, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 7),  # 3 for t, 4 for q
        )

    def forward(self, x):
        # downsample to 64x64 first to keep it small
        x = torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
        y = self.net(x)
        t = y[:, :3]
        q = y[:, 3:]
        return {"t": t, "q": q}


def extract_t_from_out(out):
    # out is dict with 't': [B, 3]
    t = out["t"]
    if t.dim() == 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    return t


def main():
    device = "cpu"  # keep CPU for now; later we can switch to 'cuda'

    # 1. Dataset & DataLoader
    ds = ShirtRoeDataset(
        root="/home/stan711/datasets/shirtv1",
        traj="roe1",
        domain="lightbox",
    )
    print("ROE1 length:", len(ds))

    dl = DataLoader(ds, batch_size=1, shuffle=False)

    # 2. Dummy model
    model = DummyPoseModel().to(device)

    # 3. Simple Kalman filter instance (we'll plug in real μ, Σ later)
    kf = SimpleKalmanCV3D(dt=1.0, device=device)

    # 4. Iterate over first few frames
    for i, batch in enumerate(dl):
        if i >= 5:
            break

        img = batch["image"].to(device)  # [1, 3, H, W]
        t_gt = batch["t_gt"][0]          # [3]

        # MC-Dropout inference on dummy model for translation only
        mu_t, cov_t = mc_infer(
            model,
            img[0],                       # [3, H, W]
            n_samples=10,
            device=device,
            extract_pose_fn=extract_t_from_out,
        )

        # For now, use cov_t as measurement noise R (ensure 3x3)
        # If cov_t is [3,3] good; if [3], make diagonal:
        if cov_t.dim() == 1:
            R = torch.diag(cov_t)
        else:
            R = cov_t

        # Kalman filter predict + update
        kf.predict()
        kf.update(mu_t, R)

        print(f"Frame {i:03d}")
        print("  t_gt:   ", t_gt.numpy())
        print("  mu_t:   ", mu_t.numpy())
        print("  kf.pos: ", kf.position.detach().numpy())
        print("  trace(R):", torch.trace(R).item())

    print("Done dummy SHIRT + MC + Kalman smoke test.")


if __name__ == "__main__":
    main()
