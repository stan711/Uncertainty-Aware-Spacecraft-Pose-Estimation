#!/usr/bin/env python3
"""
KF sweep on SHIRT MC results for φ0 and φ3.

- Scans:
    outputs/shirt_mc_phi0/roe*_mc_N*.npz
    outputs/shirt_mc_phi3/roe*_mc_N*.npz

- For each file (model, trajectory, N):
    * Loads ground-truth translation t_gt
    * Loads MC mean translation and covariance
    * Runs three variants:
        - noKF        : just MC mean
        - KF_fixed    : constant R from global MC statistics
        - KF_adaptive : per-frame R_t from MC covariance + epsilon I
    * Computes mean / median L2 translation error

- Prints tables and writes a CSV summary for each model directory.
"""

from __future__ import annotations

import os
import re
import csv
import glob
import numpy as np


# -------------------------------------------------------------------
# Simple constant-velocity 3D Kalman filter (translation only)
# -------------------------------------------------------------------

class KalmanCV3D:
    """
    6D state: [x, y, z, vx, vy, vz]^T
    Measurement: position only [x, y, z]
    """

    def __init__(self,
                 dt: float = 5.0,
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


# -------------------------------------------------------------------
# Helpers to load arrays from MC .npz files
# -------------------------------------------------------------------

def load_translation_stats(npz_path: str):
    """
    Load ground-truth translation, MC mean, and covariance
    from a unified MC .npz file.

    This is robust to slightly different key names:
    - t_gt      : [T, 3]
    - t_mean or t_mu or mu_t : [T, 3]
    - t_cov or Sigma_t or cov_t : [T, 3, 3]
    """
    data = np.load(npz_path)

    # t_gt
    if "t_gt" not in data:
        raise KeyError(f"{npz_path}: missing 't_gt' in npz file.")
    t_gt = data["t_gt"]  # [T, 3]

    # mean
    mean_keys = ["t_mean", "t_mu", "mu_t"]
    t_mean = None
    for k in mean_keys:
        if k in data:
            t_mean = data[k]
            break
    if t_mean is None:
        raise KeyError(f"{npz_path}: could not find any of {mean_keys} for translation mean.")

    # covariance
    cov_keys = ["t_cov", "Sigma_t", "cov_t"]
    t_cov = None
    for k in cov_keys:
        if k in data:
            t_cov = data[k]
            break
    if t_cov is None:
        raise KeyError(f"{npz_path}: could not find any of {cov_keys} for translation covariance.")

    # basic sanity
    if t_gt.shape != t_mean.shape:
        raise ValueError(f"{npz_path}: t_gt shape {t_gt.shape} != t_mean shape {t_mean.shape}")
    if t_cov.shape[0] != t_gt.shape[0]:
        raise ValueError(f"{npz_path}: t_cov first dim {t_cov.shape[0]} != T {t_gt.shape[0]}")

    return t_gt, t_mean, t_cov


def run_kf_sequence(t_mean: np.ndarray,
                    t_cov: np.ndarray,
                    mode: str,
                    dt: float = 5.0,
                    eps_R: float = 1e-4) -> np.ndarray:
    """
    Run one of the three KF modes on a translation sequence:

    mode = 'none'   : return t_mean (no KF)
    mode = 'fixed'  : KF with constant R (from global MC stats)
    mode = 'adapt'  : KF with per-frame R_t = Sigma_t + eps_R I
    """
    T = t_mean.shape[0]

    if mode == "none":
        return t_mean.copy()

    # New KF instance per mode
    kf = KalmanCV3D(dt=dt, q_pos=1e-4, q_vel=1e-6)
    preds = np.zeros_like(t_mean)

    if mode == "fixed":
        # Use a constant R from global MC statistics
        traces = np.array([np.trace(S) for S in t_cov])
        avg_trace = float(np.mean(traces))
        scalar = max(avg_trace / 3.0, 1e-6)  # avoid degenerate R
        R_fixed = np.eye(3, dtype=np.float64) * scalar

        for i in range(T):
            if not kf.initialized:
                kf.init(t_mean[i])
            preds[i] = kf.step(t_mean[i], R_fixed)

    elif mode == "adapt":
        for i in range(T):
            if not kf.initialized:
                kf.init(t_mean[i])
            R_t = t_cov[i] + eps_R * np.eye(3, dtype=np.float64)
            preds[i] = kf.step(t_mean[i], R_t)
    else:
        raise ValueError(f"Unknown KF mode: {mode}")

    return preds


def compute_errors(t_gt: np.ndarray, t_pred: np.ndarray):
    """Return mean and median L2 error."""
    err = np.linalg.norm(t_pred - t_gt, axis=1)
    return float(err.mean()), float(np.median(err))


# -------------------------------------------------------------------
# Main sweep
# -------------------------------------------------------------------

def sweep_model_dir(model_name: str, model_dir: str):
    """
    Process all roe*_mc_N*.npz files in a given model directory.

    Returns a list of dictionaries:
        {
          'model': model_name,
          'traj': 'roe1'/'roe2',
          'N': N,
          'mode': 'noKF'/'KF_fixed'/'KF_adapt',
          'mean_err': ...,
          'median_err': ...
        }
    """
    pattern = os.path.join(model_dir, "roe*_mc_N*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[WARN] No MC files found for model {model_name} in {model_dir}")
        return []

    results = []
    print(f"\n=== Model: {model_name} ===")
    print(f"Found {len(files)} MC files")

    for path in files:
        fname = os.path.basename(path)

        # trajectory
        if fname.startswith("roe1"):
            traj = "roe1"
        elif fname.startswith("roe2"):
            traj = "roe2"
        else:
            print(f"[WARN] Skipping {fname}: cannot infer trajectory.")
            continue

        # N from filename e.g. roe1_mc_N20.npz
        m = re.search(r"_mc_N(\d+)\.npz$", fname)
        if not m:
            print(f"[WARN] Skipping {fname}: cannot parse N.")
            continue
        N = int(m.group(1))

        print(f"  Processing {fname} (traj={traj}, N={N})")

        # Load translation stats
        t_gt, t_mean, t_cov = load_translation_stats(path)

        # Modes: no KF, fixed R, adaptive R
        for mode_label, kf_mode in [
            ("noKF", "none"),
            ("KF_fixed", "fixed"),
            ("KF_adapt", "adapt"),
        ]:
            t_pred = run_kf_sequence(t_mean, t_cov, mode=kf_mode)
            mean_err, median_err = compute_errors(t_gt, t_pred)

            results.append({
                "model": model_name,
                "traj": traj,
                "N": N,
                "mode": mode_label,
                "mean_err": mean_err,
                "median_err": median_err,
            })

    return results


def print_summary_table(results, model_name: str):
    """Pretty-print a table for a single model."""
    if not results:
        return

    # Filter for this model
    rows = [r for r in results if r["model"] == model_name]
    if not rows:
        return

    # Sort by traj, N, mode
    rows.sort(key=lambda r: (r["traj"], r["N"],
                             {"noKF": 0, "KF_fixed": 1, "KF_adapt": 2}[r["mode"]]))

    print(f"\n===== Summary for {model_name} =====")
    print(f"{'traj':<5} {'N':>3} {'mode':>9}  {'mean_err [m]':>14}  {'median_err [m]':>16}")
    print("-" * 55)
    for r in rows:
        print(f"{r['traj']:<5} {r['N']:>3} {r['mode']:>9}  {r['mean_err']:>14.4f}  {r['median_err']:>16.4f}")


def save_csv(results, model_name: str, model_dir: str):
    """Save a CSV summary in the corresponding model directory."""
    rows = [r for r in results if r["model"] == model_name]
    if not rows:
        return

    csv_path = os.path.join(model_dir, "kf_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "traj", "N", "mode", "mean_err_m", "median_err_m"])
        for r in rows:
            writer.writerow([
                r["model"], r["traj"], r["N"], r["mode"],
                f"{r['mean_err']:.6f}",
                f"{r['median_err']:.6f}",
            ])
    print(f"[INFO] Saved CSV summary for {model_name} to {csv_path}")


def main():
    # Hard-coded model dirs based on your current structure
    model_configs = [
        ("phi0", "outputs/shirt_mc_phi0"),
        ("phi3", "outputs/shirt_mc_phi3"),
    ]

    all_results = []

    for model_name, model_dir in model_configs:
        if not os.path.isdir(model_dir):
            print(f"[WARN] Model dir not found: {model_dir}")
            continue

        res = sweep_model_dir(model_name, model_dir)
        all_results.extend(res)
        print_summary_table(all_results, model_name)
        save_csv(all_results, model_name, model_dir)


if __name__ == "__main__":
    main()
