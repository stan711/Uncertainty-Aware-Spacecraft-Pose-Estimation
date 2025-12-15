#!/usr/bin/env python3
"""
Unified Monte Carlo (MC) analysis on SHIRT trajectories (ROE1/ROE2)
for translation and rotation.

- Translation comes from the EfficientPose head (efficientpose), i.e. the
  same regression used for effi_eT in the original SPNv2 code.

- Rotation comes from heatmap + UDP + EPnP, i.e. the same pipeline used
  to compute heat_eR in the original SPNv2 evaluation.

This script:
  * Works for both φ0 (GN) and φ3 (BN) models.
  * Works for both ROE1 and ROE2 lightbox trajectories.
  * Runs N Monte Carlo dropout samples per frame.
  * Computes MC translation mean/covariance and translation error.
  * Computes MC rotation error statistics (mean/median over the N samples).
  * Records per-frame runtime and overall timing.
  * Saves results to an .npz archive.

NOTE:
  - This script does NOT apply any Kalman filtering. It only measures the
    raw MC behaviour of the model (per-frame).
  - For rotation, only the heatmap+PnP orientation is used (Option A).
"""

from __future__ import annotations

import os
import os.path as osp
import time
import argparse
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from tqdm import tqdm

import _init_paths  # repo-specific path setup

from config import cfg, update_config
from nets import build_spnv2
from utils.utils import load_camera_intrinsics, load_tango_3d_keypoints
from utils.postprocess import solve_pose_from_heatmaps


# ---------------------------------------------------------------------------
# SHIRT dataset wrapper (ROE1 / ROE2, lightbox only)
# ---------------------------------------------------------------------------

class ShirtRoeDatasetUnified(Dataset):
    """
    Minimal SHIRT trajectory dataset for ROE1/ROE2, lightbox domain.

    Assumes directory structure:
      <root>/
        roe1/
          roe1.json
          lightbox/
            images/
              <filename>.(jpg|png)
        roe2/
          roe2.json
          lightbox/
            images/
              <filename>.(jpg|png)

    Each JSON file is assumed to be a list of frames, where each frame
    is a dict containing at least:
      - "filename": image file name
      - "q_vbs2tango_true": [4] quaternion
      - "r_Vo2To_vbs_true": [3] translation
    """

    def __init__(self, root: str, traj: str = "roe1",
                 img_size: Tuple[int, int] = (512, 768)) -> None:
        """
        Args:
          root: path to SHIRT root folder (that contains roe1/, roe2/).
          traj: "roe1" or "roe2".
          img_size: (H, W) to which images are resized for the network.
        """
        super().__init__()
        assert traj in ["roe1", "roe2"], "traj must be 'roe1' or 'roe2'"
        self.root = root
        self.traj = traj
        self.img_size = img_size

        json_path = osp.join(root, traj, f"{traj}.json")
        if not osp.exists(json_path):
            raise FileNotFoundError(f"Could not find JSON file: {json_path}")

        import json
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Some variants might store frames under a key
            if "frames" in data:
                self.frames = data["frames"]
            elif "sequence" in data:
                self.frames = data["sequence"]
            else:
                raise KeyError("JSON must be a list or contain 'frames'/'sequence'.")
        elif isinstance(data, list):
            self.frames = data
        else:
            raise TypeError("Unexpected JSON format for SHIRT trajectory.")

        # Basic transforms: resize -> to-tensor -> normalize (ImageNet)
        self.transform = T.Compose([
            T.Resize(self.img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.frames[idx]

        fname = frame.get("filename", None)
        if fname is None:
            raise KeyError("Frame does not contain 'filename'.")

        img_path = osp.join(self.root, self.traj, "lightbox", "images", fname)
        if not osp.exists(img_path):
            raise FileNotFoundError(f"Could not find image: {img_path}")

        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [3, H, W]

        # Translation
        if "r_Vo2To_vbs_true" in frame:
            t_gt = np.array(frame["r_Vo2To_vbs_true"], dtype=np.float64)
        else:
            raise KeyError("Missing 'r_Vo2To_vbs_true' in frame.")

        # Quaternion
        if "q_vbs2tango_true" in frame:
            q_gt = np.array(frame["q_vbs2tango_true"], dtype=np.float64)
        else:
            raise KeyError("Missing 'q_vbs2tango_true' in frame.")

        sample = {
            "image": img_tensor,
            "t_gt": t_gt,
            "q_gt": q_gt,
        }
        return sample


# ---------------------------------------------------------------------------
# Quaternion utilities
# ---------------------------------------------------------------------------

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        return q
    return q / norm


def rotation_error_deg(q_pred: np.ndarray, q_gt: np.ndarray) -> float:
    """
    Orientation error between two quaternions in degrees.

    Mirrors the standard SPNv2 definition:
      E_R = 2 * acos(|<q_pred, q_gt>|)
    """
    q_pred = normalize_quaternion(q_pred)
    q_gt = normalize_quaternion(q_gt)
    dot = np.abs(np.dot(q_pred, q_gt))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot)
    angle_deg = angle_rad * 180.0 / np.pi
    return float(angle_deg)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(phi: int, device: torch.device) -> Tuple[nn.Module, int, int]:
    """
    Build SPNv2 model for a given φ (0 or 3) and return:
      model, heatmap_head_index, effpose_head_index
    """
    if phi == 0:
        cfg_file = "experiments/offline_train_full_config_phi0_gn.yaml"
    elif phi == 3:
        cfg_file = "experiments/offline_train_full_config_phi3_BN.yaml"
    else:
        raise ValueError("phi must be 0 or 3")

    class Args:
        pass

    args = Args()
    args.cfg = cfg_file
    args.opts = []
    update_config(cfg, args)

    print("[INFO] Creating SPNv2 ...")
    model = build_spnv2(cfg)

    # Load checkpoint
    if phi == 0:
        ckpt_path = "outputs/efficientdet_d0/full_config_phi0_gn/model_final.pth.tar"
    else:
        ckpt_path = "outputs/pretrained/spnv2_efficientnetb3_fullconfig_offline.pth.tar"

    if not osp.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading φ{phi} checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[INFO] Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    model.to(device)
    model.eval()

    # Determine head indices
    if "HEAD" in cfg.TEST:
        head_names = list(cfg.TEST.HEAD)
    else:
        head_names = list(cfg.MODEL.HEAD.NAMES)

    if "heatmap" not in head_names or "efficientpose" not in head_names:
        raise RuntimeError("Config must include 'heatmap' and 'efficientpose' heads.")

    heat_idx = head_names.index("heatmap")
    effi_idx = head_names.index("efficientpose")
    return model, heat_idx, effi_idx


# ---------------------------------------------------------------------------
# MC runner (translation + rotation)
# ---------------------------------------------------------------------------

def run_mc_shirt(
    model: nn.Module,
    heat_idx: int,
    effi_idx: int,
    device: torch.device,
    traj: str,
    n_mc: int,
    shirt_root: str,
) -> Dict[str, Any]:
    """
    Run MC dropout on SHIRT ROE1/ROE2 for both translation (EfficientPose)
    and rotation (heatmap+PnP), without Kalman filtering.

    Returns a dict of numpy arrays suitable for np.savez.
    """

    # Dataset / loader
    ds = ShirtRoeDatasetUnified(
        shirt_root,
        traj=traj,
        img_size=(cfg.DATASET.INPUT_SIZE[1], cfg.DATASET.INPUT_SIZE[0])
    )
    num_frames = len(ds)
    print(f"[INFO] {traj} length: {num_frames} frames")

    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    # Camera / keypoints for solve_pose_from_heatmaps (from SPEED+ setup)
    camera = load_camera_intrinsics(cfg.DATASET.CAMERA)
    keypts_true_3D = load_tango_3d_keypoints(cfg.DATASET.KEYPOINTS)

    # Storage
    t_gt_all = []
    q_gt_all = []

    t_mean_all = []
    t_cov_all = []
    t_err_all = []

    rot_err_mean_all = []
    rot_err_med_all = []

    frame_times_ms = []

    # Monte Carlo loop
    pbar = tqdm(loader, desc=f"{traj.upper()} MC (N={n_mc})")
    t0 = time.time()
    for idx, batch in enumerate(pbar):
        start = time.time()

        images = batch["image"].to(device)  # [1, 3, H, W]
        t_gt = batch["t_gt"][0].numpy().astype(np.float64)
        q_gt = batch["q_gt"][0].numpy().astype(np.float64)

        t_gt_all.append(t_gt)
        q_gt_all.append(q_gt)

        t_samples = []
        rot_err_samples = []

        with torch.no_grad():
            # Enable dropout
            model.train()
            for _ in range(n_mc):
                outputs = model(images, is_train=False, gpu=device)

                # EfficientPose head -> translation (effi_eT)
                classification, bbox_pred, rotation_raw, translation = outputs[effi_idx]
                _, cls_argmax = torch.max(classification, dim=1)
                t_pr = translation[0, cls_argmax].squeeze().cpu().numpy().astype(np.float64)
                t_samples.append(t_pr)

                # Heatmap head -> rotation via pose solving (heat_eR path)
                heatmaps = outputs[heat_idx].squeeze(0).cpu()
                keypts_pr, q_pr, t_from_heat, reject = solve_pose_from_heatmaps(
                    heatmaps,
                    cfg.DATASET.IMAGE_SIZE,
                    cfg.TEST.HEATMAP_THRESHOLD,
                    camera,
                    keypts_true_3D,
                )
                if not reject and q_pr is not None:
                    q_pred = q_pr  # already a quaternion
                    eR = rotation_error_deg(q_pred, q_gt)
                    rot_err_samples.append(eR)

            # Back to eval mode for safety
            model.eval()

        # Translation MC stats
        t_samples = np.stack(t_samples, axis=0)  # [N, 3]
        t_mean = t_samples.mean(axis=0)
        t_cov = np.cov(t_samples, rowvar=False) if n_mc > 1 else np.zeros((3, 3), dtype=np.float64)
        t_err = float(np.linalg.norm(t_mean - t_gt))

        t_mean_all.append(t_mean)
        t_cov_all.append(t_cov)
        t_err_all.append(t_err)

        # Rotation MC stats
        if len(rot_err_samples) > 0:
            rot_err_samples = np.asarray(rot_err_samples, dtype=np.float64)
            rot_err_mean = float(rot_err_samples.mean())
            rot_err_median = float(np.median(rot_err_samples))
        else:
            rot_err_mean = np.nan
            rot_err_median = np.nan

        rot_err_mean_all.append(rot_err_mean)
        rot_err_med_all.append(rot_err_median)

        # Timing
        frame_ms = (time.time() - start) * 1000.0
        frame_times_ms.append(frame_ms)

        if idx < 5:
            print(
                f"Frame {idx:04d}: |t_mean - t_gt| = {t_err:.4f} m, "
                f"rot_err_mean = {rot_err_mean:.3f} deg"
            )

    total_time_s = time.time() - t0
    avg_time_ms = float(np.mean(frame_times_ms))

    print(
        f"[INFO] Total time: {total_time_s:.2f} s for {num_frames} frames "
        f"(avg {avg_time_ms:.2f} ms/frame)"
    )

    # Stack arrays
    results = {
        "traj": traj,
        "n_mc": n_mc,
        "num_frames": num_frames,
        "t_gt": np.stack(t_gt_all, axis=0),
        "q_gt": np.stack(q_gt_all, axis=0),
        "t_mean": np.stack(t_mean_all, axis=0),
        "t_cov": np.stack(t_cov_all, axis=0),
        "t_error": np.asarray(t_err_all, dtype=np.float64),
        "rot_error_mean": np.asarray(rot_err_mean_all, dtype=np.float64),
        "rot_error_median": np.asarray(rot_err_med_all, dtype=np.float64),
        "frame_time_ms": np.asarray(frame_times_ms, dtype=np.float64),
        "total_time_s": float(total_time_s),
        "avg_time_ms": float(avg_time_ms),
    }
    return results


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified MC analysis on SHIRT (translation + rotation)."
    )
    parser.add_argument(
        "--phi",
        type=int,
        required=True,
        choices=[0, 3],
        help="Backbone scale φ (0 for EfficientDet-D0 GN, 3 for EfficientNet-B3 BN).",
    )
    parser.add_argument(
        "--traj",
        type=str,
        required=True,
        choices=["roe1", "roe2"],
        help="Trajectory name (roe1 or roe2).",
    )
    parser.add_argument(
        "--n_mc",
        type=int,
        default=5,
        help="Number of MC dropout samples per frame.",
    )
    parser.add_argument(
        "--shirt_root",
        type=str,
        default="/home/stan711/datasets/shirtv1",
        help="Root folder of the SHIRT dataset.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="outputs",
        help="Root folder for outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Build model and locate heads
    model, heat_idx, effi_idx = build_model(args.phi, device)

    # Run MC
    results = run_mc_shirt(
        model=model,
        heat_idx=heat_idx,
        effi_idx=effi_idx,
        device=device,
        traj=args.traj,
        n_mc=args.n_mc,
        shirt_root=args.shirt_root,
    )

    # Save
    out_dir = osp.join(args.out_root, f"shirt_mc_phi{args.phi}")
    os.makedirs(out_dir, exist_ok=True)
    out_name = f"{args.traj}_mc_N{args.n_mc}.npz"
    out_path = osp.join(out_dir, out_name)

    np.savez(out_path, **results)
    print(f"[INFO] Saved results to {out_path}")


if __name__ == "__main__":
    main()
