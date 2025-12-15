#!/usr/bin/env python3
"""
Run MC-dropout + Kalman filtering on SHIRT ROE2 using the φ0 SPNv2 model.

Outputs:
  outputs/shirt_mc_phi0/roe2_mc_kf.npz
    - t_gt   : [T, 3] ground-truth translations
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
import torchvision.transforms as T

import _init_paths  # sets up imports from core/ as flat packages

from config import cfg, update_config
from nets import build_spnv2


# =========================================================
#  Simple constant-velocity 3D Kalman filter
# =========================================================


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


# =========================================================
#  Build φ0 model
# =========================================================


def build_model_phi0(device: torch.device) -> tuple[torch.nn.Module, int]:
    """
    Build SPNv2 φ0 model using phi0 GN config and load the trained checkpoint.
    Returns:
      model, efficientpose_head_index
    """
    print("[INFO] Creating SPNv2 ...")

    cfg_file = "experiments/offline_train_full_config_phi0_gn.yaml"

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
    ckpt_path = "outputs/efficientdet_d0/full_config_phi0_gn/model_final.pth.tar"
    print(f"[INFO] Loading φ0 checkpoint from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    model.to(device)
    model.eval()
    return model, effi_idx


# =========================================================
#  ROE2 Dataset (lightbox)
# =========================================================


class ShirtRoe2Dataset(Dataset):
    """
    Minimal SHIRT ROE2 dataset loader (lightbox domain only).

    Assumes:
      root = "/home/stan711/datasets/shirtv1/roe2"
      JSON at: root/roe2.json
      Images at: root/lightbox/images/*.jpg

    JSON layout is handled flexibly:
      - list at top-level
      - or dict with keys like 'frames', 'sequence', 'lightbox', etc.
    """

    def __init__(self, root: str, resize_hw=(512, 768)):
        super().__init__()
        self.root = root
        self.resize_hw = resize_hw  # (H,W) = (512,768)

        json_path = os.path.join(self.root, "roe2.json")
        if not os.path.isfile(json_path):
            raise FileNotFoundError(f"Could not find roe2.json at: {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        frames = None

        # Case 1: top-level list
        if isinstance(data, list):
            frames = data

        # Case 2: dict with obvious keys
        elif isinstance(data, dict):
            # direct "frames" / "sequence"
            if "frames" in data and isinstance(data["frames"], list):
                frames = data["frames"]
            elif "sequence" in data and isinstance(data["sequence"], list):
                frames = data["sequence"]
            # nested "lightbox"
            elif "lightbox" in data:
                lb = data["lightbox"]
                if isinstance(lb, list):
                    frames = lb
                elif isinstance(lb, dict):
                    for k in ["frames", "sequence", "data"]:
                        if k in lb and isinstance(lb[k], list):
                            frames = lb[k]
                            break
            # fallback: any list-of-dicts value
            if frames is None:
                for v in data.values():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        frames = v
                        break

        if frames is None:
            raise KeyError(
                "roe2.json: could not infer frame list. "
                "Expected a list or a dict containing a list of frame dicts."
            )

        self.frames = frames

        # Basic transform: resize + normalize like SPNv2
        self.transform = T.Compose([
            T.Resize(self.resize_hw),  # (H, W) = (512, 768)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        print(f"[INFO] roe2 length: {len(self.frames)}")

    def __len__(self):
        return len(self.frames)

    def _get_image_path(self, frame: dict) -> str:
        # Try common image name keys
        img_fname = (
            frame.get("img") or
            frame.get("image") or
            frame.get("filename") or
            frame.get("image_path")
        )

        if img_fname is None:
            # Fallback: try 'name' key, or fail
            img_fname = frame.get("name")
            if img_fname is None:
                raise KeyError("Frame does not contain an image filename key.")

        # If the JSON already stores something like "lightbox/images/img000001.jpg"
        # just join it directly to root.
        if "/" in img_fname:
            img_path = os.path.join(self.root, img_fname)
        else:
            # Default: assume it's just "img000001.jpg"
            img_path = os.path.join(self.root, "lightbox", "images", img_fname)

        return img_path

    def _get_translation(self, frame):
        """
        Extract 3D translation from a single JSON frame.

        We first try several common key names, then (optionally) a nested
        'labels' dict. If none of those exist, we fall back to a recursive
        search for the first numeric 3-vector in the frame.
        """

        import numpy as np

        # 1) Try common key names at the top level
        candidate_keys = [
            "r_Vo2To_vbs",      # common SHIRT naming
            "r_Vo2To_vbs_m",    # in case units are encoded in name
            "t_gt",
            "translation",
            "t",
            "t_vbs",
        ]

        def try_from_dict(d):
            """Try to pull a length-3 vector from dict `d` using candidate_keys."""
            for key in candidate_keys:
                if key in d:
                    arr = np.asarray(d[key], dtype=np.float32).reshape(-1)
                    if arr.shape[0] == 3:
                        return arr
            return None

        # Top-level frame
        arr = try_from_dict(frame)
        if arr is not None:
            return arr

        # 2) Try nested 'labels' (common pattern in some trajectory JSONs)
        if "labels" in frame and isinstance(frame["labels"], dict):
            arr = try_from_dict(frame["labels"])
            if arr is not None:
                return arr

        # 3) Fallback: recursively search the frame for a numeric length-3 vector
        def search_vec3(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    res = search_vec3(v)
                    if res is not None:
                        return res
            elif isinstance(obj, (list, tuple)):
                # If it's exactly length-3 and all entries numeric → treat as translation
                if len(obj) == 3 and all(isinstance(x, (int, float)) for x in obj):
                    return np.asarray(obj, dtype=np.float32)
                # Otherwise, recurse over elements
                for v in obj:
                    res = search_vec3(v)
                    if res is not None:
                        return res
            return None

        arr = search_vec3(frame)
        if arr is not None:
            # Only print a warning the first time or just leave it like this
            print("[WARN] _get_translation: using first numeric 3-vector found in frame.")
            return arr

        # If we get here, we truly couldn't find anything usable
        keys = list(frame.keys())
        raise KeyError(
            "Could not find translation key in frame. "
            f"Tried {candidate_keys} at top level and in 'labels', "
            f"and recursive search for a numeric 3-vector. "
            f"Top-level keys in this frame: {keys}"
        )


    def __getitem__(self, idx: int):
        frame = self.frames[idx]

        # Image
        img_path = self._get_image_path(frame)
        img = Image.open(img_path).convert("RGB")

        # Transform
        img_tensor = self.transform(img)  # [3, H, W]

        # Translation GT
        t_gt = self._get_translation(frame)  # [3]

        sample = {
            "image": img_tensor,
            "t_gt": torch.from_numpy(t_gt),
        }
        return sample


# =========================================================
#  MC-dropout + KF loop for ROE2
# =========================================================


def build_roe2_dataset() -> ShirtRoe2Dataset:
    shirt_root_roe2 = "/home/stan711/datasets/shirtv1/roe2"
    ds = ShirtRoe2Dataset(shirt_root_roe2)
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

    # Single worker to avoid "Too many open files" issues
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
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
        model.train()  # enable dropout

        t_samples = []
        with torch.no_grad():
            for _ in range(n_mc):
                outputs = model(images, is_train=False, gpu=device)
                classification, bbox_pred, rotation_raw, translation = outputs[effi_idx]

                _, cls_argmax = torch.max(classification, dim=1)
                t_pr = translation[0, cls_argmax].squeeze().cpu().numpy()
                t_samples.append(t_pr)

        model.eval()  # back to eval mode

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

        # Optional debug for first few frames
        if idx < 5:
            print(f"Frame {idx:03d}")
            print(f"  t_gt:    {t_gt}")
            print(f"  mu_t:    {mu_t}")
            print(f"  kf.pos:  {kf_pos}")
            print(f"  trace(R): {trace_R}")

    # Stack to arrays
    results = {
        "t_gt":    np.stack(t_gt_list, axis=0),
        "mu_t":    np.stack(mu_t_list, axis=0),
        "kf_pos":  np.stack(kf_pos_list, axis=0),
        "trace_R": np.asarray(trace_R_list),
    }
    return results


# =========================================================
#  Main
# =========================================================


def main():
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build model + locate efficientpose head
    model, effi_idx = build_model_phi0(device)

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
    out_dir = "outputs/shirt_mc_phi0"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "roe2_mc_kf.npz")
    np.savez(out_path, **results)
    print(f"[INFO] Saved results to {out_path}")


if __name__ == "__main__":
    main()
