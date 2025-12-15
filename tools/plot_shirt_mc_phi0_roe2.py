#!/usr/bin/env python3
"""
Plot MC-dropout + Kalman filtering results on SHIRT ROE2 using the φ0 SPNv2 model.

Expected input:
  outputs/shirt_mc_phi0/roe2_mc_kf.npz
    - t_gt    : [T, 3] ground-truth translations
    - mu_t    : [T, 3] MC mean translations
    - kf_pos  : [T, 3] Kalman-filtered positions
    - trace_R : [T]    trace of measurement covariance per frame

Outputs (plots):
  outputs/shirt_mc_phi0/plots/roe2_error_over_time.png
  outputs/shirt_mc_phi0/plots/roe2_uncertainty_vs_error.png
  outputs/shirt_mc_phi0/plots/roe2_uncertainty_over_time.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(npz_path: str):
    """Load ROE2 MC+KF results from npz file."""
    print(f"[INFO] Loading results from: {npz_path}")
    data = np.load(npz_path)

    t_gt = data["t_gt"]        # [T, 3]
    mu_t = data["mu_t"]        # [T, 3]
    kf_pos = data["kf_pos"]    # [T, 3]
    trR = data["trace_R"]      # [T]

    return t_gt, mu_t, kf_pos, trR


def compute_errors(t_gt: np.ndarray,
                   mu_t: np.ndarray,
                   kf_pos: np.ndarray):
    """Compute per-frame Euclidean position errors for MC mean and KF."""
    # Euclidean translation error
    e_mc = np.linalg.norm(mu_t - t_gt, axis=1)    # [T]
    e_kf = np.linalg.norm(kf_pos - t_gt, axis=1)  # [T]
    return e_mc, e_kf


def plot_error_over_time(e_mc: np.ndarray,
                         e_kf: np.ndarray,
                         out_path: str):
    """Plot |t - t_gt| over time for MC mean vs KF."""
    timesteps = np.arange(len(e_mc))

    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, e_mc, label="MC mean error")
    plt.plot(timesteps, e_kf, label="KF error")
    plt.xlabel("Frame index")
    plt.ylabel("Translation error [m]")
    plt.title("ROE2: Translation Error Over Time (φ0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saving error-over-time plot to: {out_path}")


def plot_uncertainty_vs_error(trR: np.ndarray,
                              e_mc: np.ndarray,
                              out_path: str):
    """Scatter plot: trace(R_t) vs |t - t_gt| for MC mean."""
    plt.figure(figsize=(5, 5))
    plt.scatter(trR, e_mc, s=4, alpha=0.4)
    plt.xlabel("trace(R_t)")
    plt.ylabel("MC mean translation error [m]")
    plt.title("ROE2: Uncertainty vs Error (φ0)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saving uncertainty-vs-error scatter plot to: {out_path}")


def plot_uncertainty_over_time(trR: np.ndarray,
                               out_path: str):
    """Plot trace(R_t) over time."""
    timesteps = np.arange(len(trR))

    plt.figure(figsize=(10, 4))
    plt.plot(timesteps, trR)
    plt.xlabel("Frame index")
    plt.ylabel("trace(R_t)")
    plt.title("ROE2: Uncertainty Over Time (φ0)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saving uncertainty-over-time plot to: {out_path}")


def main():
    base_dir = "outputs/shirt_mc_phi0"
    npz_path = os.path.join(base_dir, "roe2_mc_kf.npz")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Could not find results file: {npz_path}")

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Load results
    t_gt, mu_t, kf_pos, trR = load_results(npz_path)

    # Compute errors
    print("[INFO] Computing errors...")
    e_mc, e_kf = compute_errors(t_gt, mu_t, kf_pos)

    # Print summary stats
    mc_mean = float(e_mc.mean())
    mc_median = float(np.median(e_mc))
    kf_mean = float(e_kf.mean())
    kf_median = float(np.median(e_kf))

    print(f"[INFO] MC mean error:  mean={mc_mean:.4f} m, median={mc_median:.4f} m")
    print(f"[INFO] KF  error:      mean={kf_mean:.4f} m, median={kf_median:.4f} m")

    # Plots
    err_time_path = os.path.join(plots_dir, "roe2_error_over_time.png")
    unc_err_path = os.path.join(plots_dir, "roe2_uncertainty_vs_error.png")
    unc_time_path = os.path.join(plots_dir, "roe2_uncertainty_over_time.png")

    plot_error_over_time(e_mc, e_kf, err_time_path)
    plot_uncertainty_vs_error(trR, e_mc, unc_err_path)
    plot_uncertainty_over_time(trR, unc_time_path)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
