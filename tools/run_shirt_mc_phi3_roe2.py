#!/usr/bin/env python3
"""
Run MC-dropout + Kalman filtering on SHIRT ROE2 using the φ3 (EfficientNet-B3) SPNv2 model.

Outputs:
  outputs/shirt_mc_phi3/roe2_mc_kf.npz
    - t_gt   : [T, 3] ground-truth translations (from JSON)
    - mu_t   : [T, 3] MC mean translations
    - kf_pos : [T, 3] Kalman-filtered positions
    - trace_R: [T]    trace of measurement covariance per frame
"""

from __future__ import annotations

import os
import json
import logging

import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

import _init_paths  # sets up imports from core/ as flat packages

from config import cfg, update_config
from nets import build_spnv2


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
# SHIRT ROE2 dataset
# ---------------------------------------------------------


class ShirtRoe2Dataset(Dataset):
    """
    Minimal SHIRT ROE2 dataset for lightbox domain.

    Expected directory layout:
      /home/stan711/datasets/shirtv1/
        roe2/
          roe2.json
          lightbox/
            images/
              img000001.jpg
              ...

    roe2.json should contain a list/dict of frames; we look for:
      - image filename
      - translation vector (3D)
    """

    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.roe2_root = os.path.join(root, "roe2")
        json_path = os.path.join(self.roe2_root, "roe2.json")

        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Could not find roe2.json at: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Assume top-level has a list of frames under some key;
        # we've already verified frames ~2371.
        if isinstance(data, dict):
            if "frames" in data:
                self.frames = data["frames"]
            elif "sequence" in data:
                self.frames = data["sequence"]
            else:
                # If dict but no obvious key, try interpreting values as frames
                # (this matches previous working behavior).
                self.frames = list(data.values())
        elif isinstance(data, list):
            self.frames = data
        else:
            raise ValueError("roe2.json: unsupported JSON structure.")

        if len(self.frames) == 0:
            raise ValueError("roe2.json appears to have 0 frames.")

        self._warned_t = False

    def __len__(self):
        return len(self.frames)

    def _get_image_path(self, frame) -> str:
        """
        Construct full path to the lightbox image.
        We assume something like 'img000001.jpg' is stored under a key like 'image', 'image_name', etc.
        """
        img_name = None
        for k in ["image", "image_name", "filename", "img", "img_name"]:
            if k in frame:
                img_name = frame[k]
                break

        if img_name is None:
            # As fallback, look for a string ending with .jpg/.png in the frame values
            for v in frame.values():
                if isinstance(v, str) and (v.endswith(".jpg") or v.endswith(".png")):
                    img_name = v
                    break

        if img_name is None:
            raise KeyError("Could not find image filename in frame.")

        # In SHIRT structure, lightbox images are under roe2/lightbox/images/
        img_path = os.path.join(self.roe2_root, "lightbox", "images", img_name)
        return img_path

    def _get_translation(self, frame) -> np.ndarray:
        """
        Extract translation as a (3,) numpy array.

        We try several common keys; if none are found, we fall back to the
        first numeric 3-vector we can find and warn once.
        """
        candidate_keys = [
            "r_Vo2To_vbs",   # common in SHIRT documentation
            "t_gt",
            "translation",
            "t",
            "t_vbs",
        ]
        for k in candidate_keys:
            if k in frame:
                arr = np.asarray(frame[k], dtype=np.float64).ravel()
                if arr.shape[0] == 3:
                    return arr

        # Fallback: find first length-3 numeric vector
        if not self._warned_t:
            print("[WARN] _get_translation: using first numeric 3-vector found in frame.")
            self._warned_t = True

        for v in frame.values():
            if isinstance(v, (list, tuple)) and len(v) == 3:
                try:
                    arr = np.asarray(v, dtype=np.float64).ravel()
                    if arr.shape[0] == 3:
                        return arr
                except Exception:
                    continue

        raise KeyError(
            "Could not find translation key in frame. Tried common keys and no 3D vector fallback found."
        )

    def __getitem__(self, idx: int):
        frame = self.frames[idx]

        img_path = self._get_image_path(frame)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        # Minimal transform: convert to tensor and resize as needed.
        # We mimic SPEED+ preprocessing roughly: resize to 768x512 and normalize to [0,1].
        img = img.resize((768, 512))  # (W, H)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # [3, H, W]
        img_tensor = torch.from_numpy(img_np)

        t_gt = self._get_translation(frame)  # [3]

        return {
            "image": img_tensor,
            "t_gt": torch.from_numpy(t_gt).float(),
        }


# ---------------------------------------------------------
# Build φ3 model (EfficientNet-B3 pretrained)
# ---------------------------------------------------------


def build_model_phi3(device: torch.device) -> tuple[torch.nn.Module, int]:
    """
    Build SPNv2 φ3 model using the EfficientNet-B3 offline config
    and load the official pretrained checkpoint.

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

    # Load φ3 pretrained checkpoint
    ckpt_path = "outputs/pretrained/spnv2_efficientnetb3_fullconfig_offline.pth.tar"
    print(f"[INFO] Loading φ3 checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    model.to(device)
    model.eval()
    return model, effi_idx


# ---------------------------------------------------------
# MC-dropout + KF loop on ROE2
# ---------------------------------------------------------


def build_roe2_dataset() -> ShirtRoe2Dataset:
    """
    Build SHIRT ROE2 dataset pointing at your local shirtv1 root.
    """
    shirt_root = "/home/stan711/datasets/shirtv1"
    ds = ShirtRoe2Dataset(shirt_root)
    print(f"[INFO] roe2 length: {len(ds)}")
    return ds


def run_mc_kf_roe2(
    model: torch.nn.Module,
    effi_idx: int,
    device: torch.device,
    n_mc: int = 20,
    eps_R: float = 1e-4,
    dt_seconds: float = 5.0,
) -> dict:
    """
    Run MC dropout + Kalman filter on ROE2 sequence.
    Returns dict with numpy arrays suitable for np.savez().
    """

    # ---- Dataset / loader ----
    dataset = build_roe2_dataset()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,     # avoid "Too many open files"
        pin_memory=False,
    )

    print(f"[INFO] roe2 length: {len(dataset)}")

    # ---- Kalman filter ----
    kf = KalmanCV3D(dt=dt_seconds, q_pos=1e-4, q_vel=1e-6)

    # Storage
    t_gt_list = []
    mu_t_list = []
    kf_pos_list = []
    trace_R_list = []

    # Main loop
    pbar = tqdm(loader, desc=f"ROE2 MC+KF (N={n_mc})")
    for idx, batch in enumerate(pbar):
        images = batch["image"].to(device)   # [1, 3, H, W]
        t_gt_tensor = batch["t_gt"][0]       # [3]
        t_gt = t_gt_tensor.cpu().numpy()

        # ---- MC dropout ----
        model.train()  # enable dropout for stochastic passes

        t_samples = []
        with torch.no_grad():
            for _ in range(n_mc):
                outputs = model(images, is_train=False, gpu=device)
                classification, bbox_pred, rotation_raw, translation = outputs[effi_idx]

                # Same selection logic as inference.py
                _, cls_argmax = torch.max(classification, dim=1)
                t_pr = translation[0, cls_argmax].squeeze().cpu().numpy()
                t_samples.append(t_pr)

        # Back to eval mode afterwards
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

    # Build φ3 model + locate efficientpose head
    model, effi_idx = build_model_phi3(device)

    # Run MC + KF on ROE2
    results = run_mc_kf_roe2(
        model=model,
        effi_idx=effi_idx,
        device=device,
        n_mc=20,
        eps_R=1e-4,
        dt_seconds=5.0,
    )

    # Save
    out_dir = "outputs/shirt_mc_phi3"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "roe2_mc_kf.npz")
    np.savez(out_path, **results)
    print(f"[INFO] Saved results to {out_path}")


if __name__ == "__main__":
    main()
