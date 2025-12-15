#!/usr/bin/env python3
"""
Plot KF sweep comparisons from a CSV produced by analyze_shirt_kf_from_mc.py.

Expected CSV columns (either form is accepted):
  - phi, traj, N, mode, T
  - mean_t / med_t / mean_r / med_r
    OR mean_t[m] / med_t[m] / mean_r[deg] / med_r[deg]

Outputs (per phi+traj):
  - mean_t vs N for each KF mode
  - med_t vs N for each KF mode
  - mean_r vs N for each KF mode
  - med_r vs N for each KF mode
  - improvement (fixed/adaptive vs noKF) vs N for mean_t and med_t
  - improvement (fixed/adaptive vs noKF) vs N for mean_r and med_r
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _require_cols(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing columns: {missing}\nFound: {list(df.columns)}")


def _pick_metric_cols(df: pd.DataFrame) -> Dict[str, str]:
    """
    Return a mapping {logical_name: actual_column_name}.
    logical_name in: mean_t, med_t, mean_r, med_r
    """
    candidates = {
        "mean_t": ["mean_t", "mean_t[m]"],
        "med_t":  ["med_t", "med_t[m]"],
        "mean_r": ["mean_r", "mean_r[deg]"],
        "med_r":  ["med_r", "med_r[deg]"],
    }

    mapping: Dict[str, str] = {}
    for logical, opts in candidates.items():
        found = None
        for c in opts:
            if c in df.columns:
                found = c
                break
        if found is None:
            raise KeyError(
                f"Could not find a column for '{logical}'. Tried {opts}. Found: {list(df.columns)}"
            )
        mapping[logical] = found
    return mapping


def _mode_sort_key(mode: str) -> int:
    order = {"noKF": 0, "fixedKF": 1, "adaptiveKF": 2}
    return order.get(mode, 99)


def _savefig(outpath: str) -> None:
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def _plot_metric_vs_N(sub: pd.DataFrame, metric_col: str, title: str, ylabel: str, outpath: str) -> None:
    # modes on same plot
    for mode in sorted(sub["mode"].unique(), key=_mode_sort_key):
        sm = sub[sub["mode"] == mode].sort_values("N")
        plt.plot(sm["N"].to_numpy(), sm[metric_col].to_numpy(), marker="o", label=mode)
    plt.xlabel("MC samples N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(outpath)


def _plot_improvement_vs_N(
    sub: pd.DataFrame,
    metric_col: str,
    title: str,
    ylabel: str,
    outpath: str,
) -> None:
    """
    Plot improvement relative to noKF: (noKF - mode).
    Positive means improvement (lower error).
    """
    # Build a table indexed by N, columns=mode values
    pivot = sub.pivot_table(index="N", columns="mode", values=metric_col, aggfunc="mean")

    if "noKF" not in pivot.columns:
        # Nothing to compare against
        return

    base = pivot["noKF"]

    for mode in ["fixedKF", "adaptiveKF"]:
        if mode not in pivot.columns:
            continue
        imp = base - pivot[mode]
        plt.plot(imp.index.to_numpy(), imp.to_numpy(), marker="o", label=f"{mode} vs noKF")

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("MC samples N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(outpath)


def _plot_percent_improvement_vs_N(
    sub: pd.DataFrame,
    metric_col: str,
    title: str,
    outpath: str,
) -> None:
    """
    Plot percent improvement relative to noKF: 100*(noKF - mode)/noKF.
    """
    pivot = sub.pivot_table(index="N", columns="mode", values=metric_col, aggfunc="mean")
    if "noKF" not in pivot.columns:
        return
    base = pivot["noKF"].replace(0, np.nan)

    for mode in ["fixedKF", "adaptiveKF"]:
        if mode not in pivot.columns:
            continue
        pct = 100.0 * (base - pivot[mode]) / base
        plt.plot(pct.index.to_numpy(), pct.to_numpy(), marker="o", label=f"{mode} vs noKF")

    plt.axhline(0.0, linewidth=1.0)
    plt.xlabel("MC samples N")
    plt.ylabel("Percent improvement (%)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    _savefig(outpath)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to shirt_mc_kf_from_mc_summary_rot.csv")
    ap.add_argument("--outdir", required=True, help="Output directory for plots")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # required structural columns
    _require_cols(df, ["phi", "traj", "N", "mode"])
    metric_map = _pick_metric_cols(df)

    # Normalize types
    df["phi"] = df["phi"].astype(int)
    df["traj"] = df["traj"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["N"] = df["N"].astype(int)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # Plot per (phi, traj)
    for (phi, traj), sub in df.groupby(["phi", "traj"], sort=True):
        tag = f"phi{phi}_{traj}".lower()

        # ---- raw metrics vs N ----
        _plot_metric_vs_N(
            sub, metric_map["mean_t"],
            title=f"{tag}: mean translation error vs N",
            ylabel="Mean translation error (m)",
            outpath=os.path.join(outdir, f"{tag}_mean_t_vs_N.png"),
        )
        _plot_metric_vs_N(
            sub, metric_map["med_t"],
            title=f"{tag}: median translation error vs N",
            ylabel="Median translation error (m)",
            outpath=os.path.join(outdir, f"{tag}_med_t_vs_N.png"),
        )
        _plot_metric_vs_N(
            sub, metric_map["mean_r"],
            title=f"{tag}: mean rotation error vs N",
            ylabel="Mean rotation error (deg)",
            outpath=os.path.join(outdir, f"{tag}_mean_r_vs_N.png"),
        )
        _plot_metric_vs_N(
            sub, metric_map["med_r"],
            title=f"{tag}: median rotation error vs N",
            ylabel="Median rotation error (deg)",
            outpath=os.path.join(outdir, f"{tag}_med_r_vs_N.png"),
        )

        # ---- improvement vs noKF ----
        _plot_improvement_vs_N(
            sub, metric_map["mean_t"],
            title=f"{tag}: absolute improvement in mean translation error vs N",
            ylabel="Improvement (m)  [positive = better]",
            outpath=os.path.join(outdir, f"{tag}_imp_mean_t_vs_N.png"),
        )
        _plot_improvement_vs_N(
            sub, metric_map["med_t"],
            title=f"{tag}: absolute improvement in median translation error vs N",
            ylabel="Improvement (m)  [positive = better]",
            outpath=os.path.join(outdir, f"{tag}_imp_med_t_vs_N.png"),
        )
        _plot_improvement_vs_N(
            sub, metric_map["mean_r"],
            title=f"{tag}: absolute improvement in mean rotation error vs N",
            ylabel="Improvement (deg)  [positive = better]",
            outpath=os.path.join(outdir, f"{tag}_imp_mean_r_vs_N.png"),
        )
        _plot_improvement_vs_N(
            sub, metric_map["med_r"],
            title=f"{tag}: absolute improvement in median rotation error vs N",
            ylabel="Improvement (deg)  [positive = better]",
            outpath=os.path.join(outdir, f"{tag}_imp_med_r_vs_N.png"),
        )

        # ---- percent improvement vs noKF ----
        _plot_percent_improvement_vs_N(
            sub, metric_map["mean_t"],
            title=f"{tag}: percent improvement in mean translation error vs N",
            outpath=os.path.join(outdir, f"{tag}_pct_imp_mean_t_vs_N.png"),
        )
        _plot_percent_improvement_vs_N(
            sub, metric_map["med_t"],
            title=f"{tag}: percent improvement in median translation error vs N",
            outpath=os.path.join(outdir, f"{tag}_pct_imp_med_t_vs_N.png"),
        )

    print(f"[INFO] Plots written to: {outdir}")


if __name__ == "__main__":
    main()
