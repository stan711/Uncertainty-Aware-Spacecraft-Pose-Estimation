import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_error_vs_time(
    t_indices,
    err_no_filter,
    err_kf_fixed,
    err_kf_adaptive,
    title="Translation error vs time",
    ylabel="Position error [m]",
    save_path=None,
):
    """
    Plot three error curves over time:
      - no filter (raw NN)
      - KF with fixed R
      - KF with adaptive R_t (from uncertainty)
    """
    t_indices = np.asarray(t_indices)
    err_no_filter = np.asarray(err_no_filter)
    err_kf_fixed = np.asarray(err_kf_fixed)
    err_kf_adaptive = np.asarray(err_kf_adaptive)

    plt.figure()
    plt.plot(t_indices, err_no_filter, label="No filter", linewidth=1.5)
    plt.plot(t_indices, err_kf_fixed, label="KF fixed R", linewidth=1.5)
    plt.plot(t_indices, err_kf_adaptive, label="KF adaptive R_t", linewidth=1.5)
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.close()


def plot_uncertainty_vs_error(
    uncertainties,
    errors,
    title="Uncertainty vs error",
    xlabel="Uncertainty (e.g. trace(cov))",
    ylabel="Error [m]",
    save_path=None,
):
    """
    Scatter plot of scalar uncertainty vs scalar error.
    Useful for calibration plots: do higher uncertainty values
    correspond to larger errors?
    """
    uncertainties = np.asarray(uncertainties)
    errors = np.asarray(errors)

    plt.figure()
    plt.scatter(uncertainties, errors, s=8, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=200)

    plt.close()


if __name__ == "__main__":
    # Tiny self-test with fake data
    t = np.arange(0, 50)
    err_no = np.abs(np.sin(t / 10)) * 0.05 + 0.02
    err_fix = err_no * 0.8
    err_adapt = err_no * 0.6

    plot_error_vs_time(
        t,
        err_no,
        err_fix,
        err_adapt,
        title="(Dummy) Translation error vs time",
        save_path="plots/dummy_error_vs_time.png",
    )

    unc = np.random.rand(100)
    err = unc * 0.05 + 0.01 * np.random.rand(100)

    plot_uncertainty_vs_error(
        unc,
        err,
        title="(Dummy) Uncertainty vs error",
        save_path="plots/dummy_unc_vs_err.png",
    )

    print("Dummy plots saved under plots/.")
