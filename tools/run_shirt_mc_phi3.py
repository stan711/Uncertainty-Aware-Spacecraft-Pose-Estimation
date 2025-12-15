#!/usr/bin/env python3
"""
Run MC-dropout + Kalman filtering on SHIRT ROE1 using the φ0 SPNv2 model.

Outputs:
  outputs/shirt_mc_phi0/roe1_mc_kf.npz
    - t_gt   : [T, 3] ground-truth translations
    - mu_t   : [T, 3] MC mean translations
    - kf_pos : [T, 3] Kalman-filtered positions
    - trace_R: [T]    trace of measurement covariance per frame
"""

from __future__ import annotations

import os
import logging

import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from torch.utils.data import DataLoader
from tqdm import tqdm

import _init_paths  # sets up imports from core/ as flat packages

from config import cfg, update_config
from nets import build_spnv2
from dataset.shirt_dataset import ShirtRoeDataset  # your SHIRT dataset

# ---------------------------------------------------------
# Simple constant-velocity 3D Kalman filter
# ---------------------------------------------------------


class KalmanCV3D:
    """
    6D state: [x, y, z, vx, vy, vz]
    Measurement: position only [x, y, z]
    """

    def __init__(self, dt: float = 5.0,
                 q_pos: float = 1e-4,
                 q_vel: float = 1e-6):
        self.dt = dt

        # State transition
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix (position only)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise
        self.Q = np.diag(
            [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]
        ).astype(np.float64)

        # State & covariance
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0

        self.initialized = False

    def init(self, z0: np.ndarray):
        """Initialize state with first measurement z0 (3,)"""
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[0:3, 0] = z0.astype(np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = True

    def step(self, z: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Perform one Kalman filter step with measurement z (3,) and
        measurement covariance R (3x3).
        Returns the filtered position (3,)
        """
        z = z.astype(np.float64).reshape(3, 1)
        R = R.astype(np.float64)

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x  # innovation

        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        # Return position component
        return self.x[0:3, 0].copy()


# ---------------------------------------------------------
# Helper to build config & model
# ---------------------------------------------------------


def build_model_phi0(device: torch.device) -> tuple[torch.nn.Module, int]:
    """
    Build SPNv2 φ0 model using phi0 GN config and load the trained checkpoint.
    Returns:
      model, efficientpose_head_index
    """
    print("[INFO] Creating SPNv2 ...")

    cfg_file = "experiments/offline_train_full_config_phi3_BN.yaml"

    class Args:
        pass

    args = Args()
    args.cfg = cfg_file
    args.opts = []  # no extra overrides
    update_config(cfg, args)

    # Determine head index for 'efficientpose'
    if "HEAD" in cfg.TEST:
        test_heads = list(cfg.TEST.HEAD)
    else:
        test_heads = list(cfg.MODEL.HEAD.NAMES)

    if "efficientpose" not in test_heads:
        raise RuntimeError("Config's TEST.HEAD / MODEL.HEAD.NAMES must include 'efficientpose'.")

    effi_idx = test_heads.index("efficientpose")

    # Build model
    model = build_spnv2(cfg)

    # Load φ0 checkpoint
    ckpt_path = "outputs/pretrained/spnv2_efficientnetb3_fullconfig_offline.pth.tar"
    print(f"[INFO] Loading φ0 checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    model.to(device)
    model.eval()
    return model, effi_idx


# ---------------------------------------------------------
# MC-dropout + KF loop
# ---------------------------------------------------------


def build_roe1_dataset(cfg_local=None):
    """
    Build SHIRT ROE1 dataset.

    ShirtRoeDataset expects the *root* of SHIRT (the folder that contains `roe1/`),
    and it will internally look for `roe1/roe1.json`.
    """
    # >>> adjust this if you ever move the dataset <<<
    shirt_root = "/home/stan711/datasets/shirtv1"

    ds = ShirtRoeDataset(shirt_root)
    print(f"[INFO] roe1 length: {len(ds)}")
    return ds


def run_mc_kf_roe1(
    model: torch.nn.Module,
    effi_idx: int,
    device: torch.device,
    cfg_local,
    n_mc: int = 20,
    eps_R: float = 1e-4,
    dt_seconds: float = 5.0,
) -> dict:
    """
    Run MC dropout + Kalman filter on ROE1 sequence.
    Returns dict with numpy arrays suitable for np.savez().
    """

    # ---- Dataset / loader ----
    dataset = build_roe1_dataset(cfg_local)

    # IMPORTANT: num_workers=0 to avoid "Too many open files"
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,     # <- key change
        pin_memory=False,  # safer with single-worker CPU loading
    )

    print(f"[INFO] roe1 length: {len(dataset)}")

    # ---- Kalman filter ----
    kf = KalmanCV3D(dt=dt_seconds, q_pos=1e-4, q_vel=1e-6)

    # Storage
    t_gt_list = []
    mu_t_list = []
    kf_pos_list = []
    trace_R_list = []

    # Main loop
    pbar = tqdm(loader, desc=f"ROE1 MC+KF (N={n_mc})")
    for idx, batch in enumerate(pbar):
        # batch is expected to be a dict with keys like ['image', 't_gt', ...]
        images = batch["image"].to(device)   # [1, 3, H, W]
        t_gt_tensor = batch["t_gt"][0]       # [3]
        t_gt = t_gt_tensor.cpu().numpy()

        # ---- MC dropout ----
        # Enable dropout for stochastic passes
        model.train()

        t_samples = []
        with torch.no_grad():
            for _ in range(n_mc):
                outputs = model(images, is_train=False, gpu=device)
                classification, bbox_pred, rotation_raw, translation = outputs[effi_idx]

                # Same selection logic as inference.py
                _, cls_argmax = torch.max(classification, dim=1)
                t_pr = translation[0, cls_argmax].squeeze().cpu().numpy()
                t_samples.append(t_pr)

        # Back to eval mode for any later deterministic use
        model.eval()

        t_samples = np.stack(t_samples, axis=0)  # [N, 3]
        mu_t = t_samples.mean(axis=0)
        if n_mc > 1:
            Sigma_t = np.cov(t_samples, rowvar=False)  # [3, 3]
        else:
            Sigma_t = np.zeros((3, 3), dtype=np.float64)

        # Regularize measurement covariance
        R_t = Sigma_t + eps_R * np.eye(3, dtype=np.float64)
        trace_R = float(np.trace(R_t))

        # ---- Kalman filter update ----
        if not kf.initialized:
            kf.init(mu_t)
        kf_pos = kf.step(mu_t, R_t)  # [3]

        # Store
        t_gt_list.append(t_gt)
        mu_t_list.append(mu_t)
        kf_pos_list.append(kf_pos)
        trace_R_list.append(trace_R)

        # Optional: print first few frames for sanity check
        if idx < 5:
            print(f"Frame {idx:03d}")
            print(f"  t_gt:    {t_gt}")
            print(f"  mu_t:    {mu_t}")
            print(f"  kf.pos:  {kf_pos}")
            print(f"  trace(R): {trace_R}")

    # Stack to arrays
    results = {
        "t_gt":   np.stack(t_gt_list, axis=0),
        "mu_t":   np.stack(mu_t_list, axis=0),
        "kf_pos": np.stack(kf_pos_list, axis=0),
        "trace_R": np.asarray(trace_R_list),
    }
    return results

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build model + locate efficientpose head
    model, effi_idx = build_model_phi0(device)

    # Run MC + KF on ROE1
    results = run_mc_kf_roe1(
        model=model,
        effi_idx=effi_idx,
        device=device,
        cfg_local=cfg,
        n_mc=20,
        eps_R=1e-4,
        dt_seconds=5.0,
    )

    # Save
    out_dir = "outputs/shirt_mc_phi3"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "roe1_mc_kf.npz")
    np.savez(out_path, **results)
    print(f"[INFO] Saved results to {out_path}")


if __name__ == "__main__":
    main()
