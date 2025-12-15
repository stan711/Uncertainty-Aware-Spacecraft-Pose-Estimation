#!/usr/bin/env python3
"""
Plot SHIRT ROE1 results for MC-dropout + Kalman filter (φ0).
"""

import os
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("plot_shirt_mc_phi3")


def load_results(npz_path):
    """
    Load results from np.savez file.

    Expected keys:
      - t_gt:    [N, 3]
      - mu_t:    [N, 3]
      - kf_pos:  [N, 3]
      - trace_R or trR: [N]
    """
    data = np.load(npz_path)

    t_gt = data["t_gt"]
    mu_t = data["mu_t"]
    kf_pos = data["kf_pos"]

    # Backward-compatible: accept either "trace_R" or "trR"
    if "trace_R" in data.files:
        trR = data["trace_R"]
    elif "trR" in data.files:
        trR = data["trR"]
    else:
        raise KeyError(
            f"Neither 'trace_R' nor 'trR' found in {npz_path}. "
            f"Available keys: {list(data.files)}"
        )

    return t_gt, mu_t, kf_pos, trR


def compute_errors(t_gt, mu_t, kf_pos):
    """
    Compute per-frame Euclidean translation errors (in meters) for:
      - MC mean (mu_t)
      - Kalman filter (kf_pos)
    """
    # Norm of difference along axis=1
    err_mc = np.linalg.norm(mu_t - t_gt, axis=1)
    err_kf = np.linalg.norm(kf_pos - t_gt, axis=1)
    return err_mc, err_kf


def plot_error_over_time(err_mc, err_kf, out_path):
    """
    Plot MC vs KF error over time.
    """
    n = len(err_mc)
    x = np.arange(n)

    plt.figure(figsize=(8, 4))
    plt.plot(x, err_mc, label="MC mean error")
    plt.plot(x, err_kf, label="KF error", alpha=0.8)
    plt.xlabel("Frame index")
    plt.ylabel("Translation error [m]")
    plt.title("ROE1 Translation Error Over Time")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_uncertainty_vs_error(trR, err_mc, out_path):
    """
    Scatter plot: uncertainty (trace_R) vs MC error.
    """
    plt.figure(figsize=(5, 5))
    plt.scatter(trR, err_mc, s=5, alpha=0.4)
    plt.xlabel("trace(R) (MC covariance)")
    plt.ylabel("MC translation error [m]")
    plt.title("ROE1 Uncertainty vs. MC Error")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_uncertainty_over_time(trR, out_path):
    """
    Plot uncertainty trace over time.
    """
    n = len(trR)
    x = np.arange(n)

    plt.figure(figsize=(8, 4))
    plt.plot(x, trR)
    plt.xlabel("Frame index")
    plt.ylabel("trace(R)")
    plt.title("ROE1 Measurement Uncertainty Over Time")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    # Path to the saved npz from run_shirt_mc_phi0.py
    npz_path = "outputs/shirt_mc_phi3/roe1_mc_kf.npz"
    logger.info(f"Loading results from: {npz_path}")

    t_gt, mu_t, kf_pos, trR = load_results(npz_path)

    # Compute errors
    logger.info("Computing errors...")
    err_mc, err_kf = compute_errors(t_gt, mu_t, kf_pos)

    # Summary stats
    mc_mean = float(err_mc.mean())
    mc_med = float(np.median(err_mc))
    kf_mean = float(err_kf.mean())
    kf_med = float(np.median(err_kf))

    logger.info(f"MC mean error:  mean={mc_mean:.4f} m, median={mc_med:.4f} m")
    logger.info(f"KF  error:      mean={kf_mean:.4f} m, median={kf_med:.4f} m")

    # Output directory for plots
    plot_dir = "outputs/shirt_mc_phi3/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Error over time
    out_error_time = os.path.join(plot_dir, "roe1_error_over_time.png")
    logger.info(f"Saving error-over-time plot to: {out_error_time}")
    plot_error_over_time(err_mc, err_kf, out_error_time)

    # 2) Uncertainty vs error scatter
    out_unc_scatter = os.path.join(plot_dir, "roe1_uncertainty_vs_error.png")
    logger.info(f"Saving uncertainty-vs-error scatter plot to: {out_unc_scatter}")
    plot_uncertainty_vs_error(trR, err_mc, out_unc_scatter)

    # 3) Uncertainty over time
    out_unc_time = os.path.join(plot_dir, "roe1_uncertainty_over_time.png")
    logger.info(f"Saving uncertainty-over-time plot to: {out_unc_time}")
    plot_uncertainty_over_time(trR, out_unc_time)

    logger.info("Done.")


if __name__ == "__main__":
    main()
