#!/usr/bin/env python3
"""
Analyze Kalman Filter variants (noKF, fixedKF, adaptiveKF) using
the Monte Carlo outputs produced by run_shirt_mc_unified.py.

Assumes NPZ files of the form:
  outputs/shirt_mc_phi{phi}/roe{1,2}_mc_N{5,10,20}.npz

Each NPZ is expected to contain at least:
  - t_gt             : [T, 3]   ground truth translation
  - t_mean           : [T, 3]   MC mean translation
  - t_cov            : [T, 3, 3] translation covariance per frame
  - t_error          : [T]      per-frame translation error (MC mean)
  - rot_error_mean   : scalar   mean rotation error in deg
  - rot_error_median : scalar   median rotation error in deg

This script:
  - runs 3 modes per (phi, traj, N):
      * noKF:      use t_mean directly
      * fixedKF:   constant measurement covariance R_fixed
      * adaptiveKF: per-frame R_t from t_cov
  - computes mean/median translation error for each configuration
  - attaches rotation mean/median (from MC) for convenience
  - prints a combined translation+rotation summary table
  - saves CSV at: outputs/shirt_mc_kf_from_mc_summary_rot.csv
"""

import os
import csv
import numpy as np


# ------------------------------------------------------------
# Simple constant-velocity 3D Kalman Filter
# ------------------------------------------------------------

class KalmanCV3D:
    """
    6D state: [x, y, z, vx, vy, vz]
    Measurement: position only [x, y, z]
    """

    def __init__(self, dt=5.0, q_pos=1e-4, q_vel=1e-6):
        self.dt = dt

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 3] = dt
        self.F[1, 4] = dt
        self.F[2, 5] = dt

        # Measurement matrix H (position only)
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Process noise covariance Q
        self.Q = np.diag(
            [q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]
        ).astype(np.float64)

        # State and covariance
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0

        self.initialized = False

    def init(self, z0):
        """
        Initialize with first measurement z0 (shape [3])
        """
        z0 = np.asarray(z0, dtype=np.float64).reshape(3)
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.x[0:3, 0] = z0
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = True

    def step(self, z, R):
        """
        One KF step:
          z : [3] measurement
          R : [3,3] measurement covariance

        Returns filtered position [3]
        """
        z = np.asarray(z, dtype=np.float64).reshape(3, 1)
        R = np.asarray(R, dtype=np.float64).reshape(3, 3)

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Update
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x

        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P

        return self.x[0:3, 0].copy()


# ------------------------------------------------------------
# Loading helper
# ------------------------------------------------------------

def load_mc_npz(phi, traj, N):
    """
    Load the NPZ for given phi, traj, N.
    Returns a dict with numpy arrays:
      t_gt [T,3], t_mean [T,3], t_cov [T,3,3],
      t_error [T], rot_mean (scalar), rot_med (scalar)
    """
    base_dir = f"outputs/shirt_mc_phi{phi}"
    npz_path = os.path.join(base_dir, f"{traj}_mc_N{N}.npz")
    print(f"[INFO] Loading NPZ: {npz_path}")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"NPZ not found: {npz_path}")

    data = np.load(npz_path)

    t_gt = np.array(data["t_gt"], dtype=float)
    t_mean = np.array(data["t_mean"], dtype=float)
    t_cov = np.array(data["t_cov"], dtype=float)
    t_error = np.array(data["t_error"], dtype=float).reshape(-1)

    # Rotation stats from MC heatmap+PnP
    rot_mean = float(np.array(data["rot_error_mean"]).reshape(-1)[0])
    rot_med = float(np.array(data["rot_error_median"]).reshape(-1)[0])

    assert t_gt.shape == t_mean.shape
    T = t_gt.shape[0]
    if t_cov.shape[0] != T:
        raise ValueError(f"t_cov first dim {t_cov.shape[0]} != {T}")
    if t_error.shape[0] != T:
        raise ValueError(f"t_error length {t_error.shape[0]} != {T}")

    # Ensure t_cov is [T,3,3]
    if t_cov.ndim == 2 and t_cov.shape[1] == 3:
        cov_full = np.zeros((T, 3, 3), dtype=float)
        for i in range(T):
            cov_full[i] = np.diag(t_cov[i])
        t_cov = cov_full
    elif t_cov.ndim != 3:
        raise ValueError(f"Unexpected t_cov shape: {t_cov.shape}")

    return {
        "t_gt": t_gt,
        "t_mean": t_mean,
        "t_cov": t_cov,
        "t_error": t_error,
        "rot_mean": rot_mean,
        "rot_med": rot_med,
    }


# ------------------------------------------------------------
# Error utilities
# ------------------------------------------------------------

def translation_error(t_pred, t_gt):
    """
    Per-frame L2 translation error ||pred - gt||_2.
    t_pred, t_gt: [T, 3]
    Returns [T] array.
    """
    diff = t_pred - t_gt
    return np.linalg.norm(diff, axis=1)


# ------------------------------------------------------------
# KF evaluation
# ------------------------------------------------------------

def eval_no_kf(t_error):
    """
    No KF: use precomputed MC translation errors directly.
    """
    return t_error


def eval_fixed_kf(t_mean, t_gt, t_cov, dt=5.0, q_pos=1e-4, q_vel=1e-6, eps_R=1e-4):
    """
    Fixed KF: use a single R_fixed computed from average t_cov.
    """
    T = t_gt.shape[0]

    # average covariance
    R_fixed = t_cov.mean(axis=0)
    # regularization
    R_fixed = R_fixed + eps_R * np.eye(3, dtype=float)

    kf = KalmanCV3D(dt=dt, q_pos=q_pos, q_vel=q_vel)
    kf.init(t_mean[0])

    t_filt = np.zeros_like(t_mean)

    for i in range(T):
        z = t_mean[i]
        t_filt[i] = kf.step(z, R_fixed)

    e = translation_error(t_filt, t_gt)
    return e


def eval_adaptive_kf(t_mean, t_gt, t_cov, dt=5.0, q_pos=1e-4, q_vel=1e-6, eps_R=1e-4):
    """
    Adaptive KF: R_t = t_cov[t] + eps_R * I
    """
    T = t_gt.shape[0]

    kf = KalmanCV3D(dt=dt, q_pos=q_pos, q_vel=q_vel)
    kf.init(t_mean[0])

    t_filt = np.zeros_like(t_mean)

    for i in range(T):
        z = t_mean[i]
        R_t = t_cov[i] + eps_R * np.eye(3, dtype=float)
        t_filt[i] = kf.step(z, R_t)

    e = translation_error(t_filt, t_gt)
    return e


# ------------------------------------------------------------
# Main analysis loop
# ------------------------------------------------------------

def analyze_all():
    """
    Loop over phi in {0,3}, traj in {roe1,roe2}, N in {5,10,20}
    Evaluate noKF/fixedKF/adaptiveKF.
    Return list of summary dicts with both translation and rotation stats.
    """
    PHIS = [0, 3]
    TRAJS = ["roe1", "roe2"]
    N_VALUES = [5, 10, 20]

    summaries = []

    for N in N_VALUES:
        for phi in PHIS:
            for traj in TRAJS:
                mc = load_mc_npz(phi, traj, N)
                t_gt = mc["t_gt"]
                t_mean = mc["t_mean"]
                t_cov = mc["t_cov"]
                t_error = mc["t_error"]
                rot_mean = mc["rot_mean"]
                rot_med = mc["rot_med"]
                T = t_gt.shape[0]

                # noKF
                e_no = eval_no_kf(t_error)
                mean_no = float(e_no.mean())
                med_no = float(np.median(e_no))

                # fixedKF
                e_fix = eval_fixed_kf(t_mean, t_gt, t_cov)
                mean_fix = float(e_fix.mean())
                med_fix = float(np.median(e_fix))

                # adaptiveKF
                e_ad = eval_adaptive_kf(t_mean, t_gt, t_cov)
                mean_ad = float(e_ad.mean())
                med_ad = float(np.median(e_ad))

                # Rotation metrics are MC-only; KF does not change them.
                # They are attached for all rows for convenience.
                summaries.append({
                    "phi": phi,
                    "traj": traj,
                    "N": N,
                    "mode": "noKF",
                    "T": T,
                    "mean_t": mean_no,
                    "med_t": med_no,
                    "mean_r": rot_mean,
                    "med_r": rot_med,
                })
                summaries.append({
                    "phi": phi,
                    "traj": traj,
                    "N": N,
                    "mode": "fixedKF",
                    "T": T,
                    "mean_t": mean_fix,
                    "med_t": med_fix,
                    "mean_r": rot_mean,
                    "med_r": rot_med,
                })
                summaries.append({
                    "phi": phi,
                    "traj": traj,
                    "N": N,
                    "mode": "adaptiveKF",
                    "T": T,
                    "mean_t": mean_ad,
                    "med_t": med_ad,
                    "mean_r": rot_mean,
                    "med_r": rot_med,
                })

    return summaries


def print_table(summaries):
    """
    Pretty-print a table sorted by N, phi, traj, mode.
    """
    summaries_sorted = sorted(
        summaries,
        key=lambda d: (d["N"], d["phi"], d["traj"], d["mode"])
    )

    print("\n===== KF Sweep from MC (Translation + Rotation) =====")
    header = (
        "  N  phi  traj       mode   #frames   mean_t[m]   med_t[m]   mean_r[deg]  med_r[deg]"
    )
    print(header)
    print("-" * len(header))
    for s in summaries_sorted:
        print(
            f"{s['N']:3d}  {s['phi']:3d}  {s['traj']:<4s}  {s['mode']:<10s}"
            f"{s['T']:9d}   {s['mean_t']:10.4f}  {s['med_t']:10.4f}"
            f"{s['mean_r']:12.4f}  {s['med_r']:11.4f}"
        )
    print("=" * len(header))


def save_csv(summaries, out_path="outputs/shirt_mc_kf_from_mc_summary_rot.csv"):
    """
    Save summaries to CSV.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["phi", "traj", "N", "mode", "T", "mean_t", "med_t", "mean_r", "med_r"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in sorted(summaries, key=lambda d: (d["N"], d["phi"], d["traj"], d["mode"])):
            writer.writerow(s)
    print(f"[INFO] KF sweep summary CSV written to {out_path}")


def main():
    summaries = analyze_all()
    print_table(summaries)
    save_csv(summaries)


if __name__ == "__main__":
    main()
