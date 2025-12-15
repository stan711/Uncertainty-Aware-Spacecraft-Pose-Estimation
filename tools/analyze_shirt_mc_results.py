#!/usr/bin/env python3
"""
Analyze SHIRT Monte Carlo results (translation + rotation) from NPZ files.

Expected NPZ layout (for each run):
  outputs/shirt_mc_phi{phi}/{traj}_mc_N{N}.npz

Each NPZ is expected to contain at least:
  - t_error          [T]    per-frame translation error [m]
  - frame_time_ms    [T]    per-frame runtime [ms]
  - avg_time_ms      scalar average runtime per frame [ms]
  - total_time_s     scalar total runtime [sec]
  - num_frames       scalar number of frames
  - rot_error_mean   scalar mean rotation error [deg]
  - rot_error_median scalar median rotation error [deg]

Extra keys are ignored.

We analyze:
  phi  in {0, 3}
  traj in {"roe1", "roe2"}
  N    in {5, 10, 20}

Outputs:
  - Printed summary table (sorted by N, then phi, then traj)
  - CSV: outputs/shirt_mc_master_summary_rot.csv
  - Plots in outputs/shirt_mc_master_plots_rot/:
      * Translation mean error vs N
      * Rotation mean error vs N
      * Avg time vs N
      * Translation error-over-time for N=20 (per phi, traj)
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Helper: load one NPZ run
# --------------------------------------------------------------------------


def load_mc_run(phi: int, traj: str, N: int, base_dir: str = "outputs"):
    """
    Load one MC run for given phi, trajectory, and N from a .npz file.

    Returns a dict with:
      - 'phi', 'traj', 'N'
      - 'num_frames'
      - 't_error'        [T] translation error per frame [m]
      - 'frame_time_ms'  [T] runtime per frame [ms]
      - 'avg_time_ms'    scalar
      - 'total_time_s'   scalar
      - 'r_mean'         scalar mean rotation error [deg]
      - 'r_med'          scalar median rotation error [deg]

    Returns None if the file does not exist.
    """
    phi_dir = os.path.join(base_dir, f"shirt_mc_phi{phi}")
    npz_path = os.path.join(phi_dir, f"{traj}_mc_N{N}.npz")

    if not os.path.isfile(npz_path):
        print(f"[WARN] NPZ not found for phi={phi}, traj={traj}, N={N}: {npz_path}")
        return None

    print(f"[INFO] Loading NPZ: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())

    def get_required(name):
        if name not in keys:
            raise KeyError(f"Required key '{name}' missing in {npz_path}")
        return data[name]

    # Per-frame arrays
    t_error = np.array(get_required("t_error"), dtype=float).reshape(-1)
    frame_time_ms = np.array(get_required("frame_time_ms"), dtype=float).reshape(-1)

    # Scalars: avg_time_ms, total_time_s, num_frames
    if "avg_time_ms" in keys:
        avg_time_ms = float(np.array(data["avg_time_ms"]))
    else:
        avg_time_ms = float(frame_time_ms.mean())

    if "total_time_s" in keys:
        total_time_s = float(np.array(data["total_time_s"]))
    else:
        total_time_s = float(frame_time_ms.sum() / 1000.0)

    if "num_frames" in keys:
        num_frames = int(np.array(data["num_frames"]))
    else:
        num_frames = len(t_error)

    # Rotation summary stats
    if "rot_error_mean" in keys:
        r_mean_arr = np.array(data["rot_error_mean"], dtype=float).reshape(-1)
        r_mean = float(r_mean_arr.mean())
    else:
        r_mean = float("nan")

    if "rot_error_median" in keys:
        r_med_arr = np.array(data["rot_error_median"], dtype=float).reshape(-1)
        r_med = float(r_med_arr.mean())
    else:
        r_med = float("nan")

    return {
        "phi": phi,
        "traj": traj,
        "N": N,
        "num_frames": num_frames,
        "t_error": t_error,
        "frame_time_ms": frame_time_ms,
        "avg_time_ms": avg_time_ms,
        "total_time_s": total_time_s,
        "r_mean": r_mean,
        "r_med": r_med,
    }


# --------------------------------------------------------------------------
# Summaries for a single run
# --------------------------------------------------------------------------


def summarize_run(run_dict):
    """
    Given run_dict from load_mc_run, compute scalar summaries:
      - mean / median translation error (from t_error)
      - mean / median rotation error (from stored scalars)
      - avg time / total time
    """
    t_error = run_dict["t_error"]

    t_mean = float(np.mean(t_error))
    t_med = float(np.median(t_error))

    r_mean = run_dict["r_mean"]
    r_med = run_dict["r_med"]

    return {
        "phi": run_dict["phi"],
        "traj": run_dict["traj"],
        "N": run_dict["N"],
        "num_frames": run_dict["num_frames"],
        "t_mean": t_mean,
        "t_med": t_med,
        "r_mean": r_mean,
        "r_med": r_med,
        "avg_time_ms": float(run_dict["avg_time_ms"]),
        "total_time_s": float(run_dict["total_time_s"]),
    }


# --------------------------------------------------------------------------
# Analyze all combinations
# --------------------------------------------------------------------------


def analyze_all():
    phis = [0, 3]
    trajs = ["roe1", "roe2"]
    Ns = [5, 10, 20]

    summaries = []

    for N in Ns:
        for phi in phis:
            for traj in trajs:
                run = load_mc_run(phi, traj, N)
                if run is None:
                    continue
                summary = summarize_run(run)
                summaries.append(summary)

    # Sort by N, then phi, then traj
    summaries.sort(key=lambda d: (d["N"], d["phi"], d["traj"]))
    return summaries


# --------------------------------------------------------------------------
# Printing + CSV
# --------------------------------------------------------------------------


def print_and_save_summary(
    summaries, out_csv="outputs/shirt_mc_master_summary_rot.csv"
):
    if not summaries:
        print("[WARN] No summaries to report.")
        return

    print("\n===== SHIRT MC Translation + Rotation Summary (Sorted by N) =====")
    header = (
        "  N  phi  traj   #frames   mean_t[m]   med_t[m]   mean_r[deg]  med_r[deg]  avg_time[ms]  total_time[s]"
    )
    print(header)
    print("-" * len(header))

    out_dir = os.path.dirname(out_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "N",
                "phi",
                "traj",
                "num_frames",
                "mean_t_m",
                "med_t_m",
                "mean_r_deg",
                "med_r_deg",
                "avg_time_ms",
                "total_time_s",
            ]
        )

        for s in summaries:
            row_str = (
                f"{s['N']:3d}  {s['phi']:3d}  {s['traj']:4s}"
                f"  {s['num_frames']:8d}"
                f"  {s['t_mean']:10.4f}"
                f"  {s['t_med']:10.4f}"
                f"  {s['r_mean']:12.4f}"
                f"  {s['r_med']:11.4f}"
                f"  {s['avg_time_ms']:12.2f}"
                f"  {s['total_time_s']:13.2f}"
            )
            print(row_str)

            writer.writerow(
                [
                    s["N"],
                    s["phi"],
                    s["traj"],
                    s["num_frames"],
                    s["t_mean"],
                    s["t_med"],
                    s["r_mean"],
                    s["r_med"],
                    s["avg_time_ms"],
                    s["total_time_s"],
                ]
            )

    print("=" * len(header))
    print(f"\n[INFO] Summary CSV written to {out_csv}")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------


def make_plots(summaries, base_dir="outputs/shirt_mc_master_plots_rot"):
    if not summaries:
        print("[WARN] No summaries to plot.")
        return

    os.makedirs(base_dir, exist_ok=True)

    phis = sorted(set(s["phi"] for s in summaries))
    trajs = sorted(set(s["traj"] for s in summaries))

    def filt(phi, traj):
        return [s for s in summaries if s["phi"] == phi and s["traj"] == traj]

    # 1) Error vs N + time vs N
    for phi in phis:
        for traj in trajs:
            subset = filt(phi, traj)
            if not subset:
                continue
            subset = sorted(subset, key=lambda d: d["N"])
            N_list = [s["N"] for s in subset]
            t_mean_list = [s["t_mean"] for s in subset]
            r_mean_list = [s["r_mean"] for s in subset]
            time_list = [s["avg_time_ms"] for s in subset]

            # Translation error vs N
            plt.figure()
            plt.plot(N_list, t_mean_list, marker="o")
            plt.xlabel("N (MC samples)")
            plt.ylabel("Mean translation error [m]")
            plt.title(f"phi{phi} {traj}: Translation error vs N")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(
                base_dir, f"phi{phi}_{traj}_trans_error_vs_N.png"
            )
            plt.savefig(out_path)
            plt.close()
            print(f"[INFO] Saved {out_path}")

            # Rotation error vs N (mean)
            plt.figure()
            plt.plot(N_list, r_mean_list, marker="o")
            plt.xlabel("N (MC samples)")
            plt.ylabel("Mean rotation error [deg]")
            plt.title(f"phi{phi} {traj}: Rotation error vs N")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(base_dir, f"phi{phi}_{traj}_rot_error_vs_N.png")
            plt.savefig(out_path)
            plt.close()
            print(f"[INFO] Saved {out_path}")

            # Time vs N
            plt.figure()
            plt.plot(N_list, time_list, marker="o")
            plt.xlabel("N (MC samples)")
            plt.ylabel("Avg time per frame [ms]")
            plt.title(f"phi{phi} {traj}: Time vs N")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(base_dir, f"phi{phi}_{traj}_time_vs_N.png")
            plt.savefig(out_path)
            plt.close()
            print(f"[INFO] Saved {out_path}")

    # 2) Translation error-over-time for N=20 (if available)
    chosen_N = 20
    for phi in phis:
        for traj in trajs:
            run = load_mc_run(phi, traj, chosen_N)
            if run is None:
                continue
            t_err = run["t_error"]

            plt.figure()
            plt.plot(t_err)
            plt.xlabel("Frame")
            plt.ylabel("Translation error [m]")
            plt.title(f"phi{phi} {traj}: Translation error over time (N={chosen_N})")
            plt.grid(True)
            plt.tight_layout()
            out_path = os.path.join(
                base_dir, f"phi{phi}_{traj}_trans_error_over_time_N{chosen_N}.png"
            )
            plt.savefig(out_path)
            plt.close()
            print(f"[INFO] Saved {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    summaries = analyze_all()
    print_and_save_summary(summaries)
    make_plots(summaries)


if __name__ == "__main__":
    main()
