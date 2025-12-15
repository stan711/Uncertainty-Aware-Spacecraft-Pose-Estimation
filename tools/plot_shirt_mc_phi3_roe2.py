#!/usr/bin/env python3
"""
Plot SHIRT ROE2 MC-dropout + Kalman results for φ3 (EfficientNet-B3).

Loads:
  outputs/shirt_mc_phi3/roe2_mc_kf.npz

Saves:
  outputs/shirt_mc_phi3/plots/roe2_error_over_time.png
  outputs/shirt_mc_phi3/plots/roe2_uncertainty_vs_error.png
  outputs/shirt_mc_phi3/plots/roe2_uncertainty_over_time.png
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt


def load_results(npz_path: str):
    data = np.load(npz_path)
    t_gt = data["t_gt"]      # [T, 3]
    mu_t = data["mu_t"]      # [T, 3]
    kf_pos = data["kf_pos"]  # [T, 3]
    trR = data["trace_R"]    # [T]
    return t_gt, mu_t, kf_pos, trR


def compute_errors(t_gt, mu_t, kf_pos):
    """
    Returns:
      err_mc : [T] MC mean |mu_t - t_gt|
      err_kf : [T] KF |kf_pos - t_gt|
    """
    diff_mc = mu_t - t_gt
    diff_kf = kf_pos - t_gt
    err_mc = np.linalg.norm(diff_mc, axis=1)
    err_kf = np.linalg.norm(diff_kf, axis=1)
    return err_mc, err_kf


def main():
    npz_path = "outputs/shirt_mc_phi3/roe2_mc_kf.npz"
    print(f"[INFO] Loading results from: {npz_path}")
    t_gt, mu_t, kf_pos, trR = load_results(npz_path)

    print("[INFO] Computing errors...")
    err_mc, err_kf = compute_errors(t_gt, mu_t, kf_pos)

    print(f"[INFO] MC mean error:  mean={err_mc.mean():.4f} m, median={np.median(err_mc):.4f} m")
    print(f"[INFO] KF  error:      mean={err_kf.mean():.4f} m, median={np.median(err_kf):.4f} m")

    out_dir = "outputs/shirt_mc_phi3/plots"
    os.makedirs(out_dir, exist_ok=True)

    T = len(err_mc)
    t_axis = np.arange(T)

    # 1) Error over time
    plt.figure()
    plt.plot(t_axis, err_mc, label="MC mean |μ_t - t_gt|")
    plt.plot(t_axis, err_kf, label="KF |x_t - t_gt|", alpha=0.8)
    plt.xlabel("Frame index")
    plt.ylabel("Translation error [m]")
    plt.title("ROE2 translation error over time (φ3, MC vs KF)")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "roe2_error_over_time.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saving error-over-time plot to: {out_path}")

    # 2) Uncertainty vs absolute error (MC only)
    plt.figure()
    plt.scatter(trR, err_mc, s=4)
    plt.xlabel("trace(R_t)")
    plt.ylabel("|μ_t - t_gt| [m]")
    plt.title("ROE2 uncertainty vs MC translation error (φ3)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "roe2_uncertainty_vs_error.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saving uncertainty-vs-error scatter plot to: {out_path}")

    # 3) Uncertainty over time
    plt.figure()
    plt.plot(t_axis, trR)
    plt.xlabel("Frame index")
    plt.ylabel("trace(R_t)")
    plt.title("ROE2 uncertainty over time (φ3)")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "roe2_uncertainty_over_time.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] Saving uncertainty-over-time plot to: {out_path}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
