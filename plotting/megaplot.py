#!/usr/bin/env python3
"""Combine scaling.csv files from multiple machines and produce:
  - one four-platform bar-chart snapshot per GPU count (plot_four_platforms.py)
  - two line plots (GFLOP/s and wall time vs GPU count) across the whole
    sweep, with an IQR ribbon (plot_scaling_lines.py)

Why per GPU count for the bar charts: plot_four_platforms.py takes the
median across every row it's given. Feeding it a sweep CSV that spans
devices=1..8 directly would blend wildly different GPU counts into one
meaningless median (GFLOP/s naturally varies a lot with device count). So
this script splits the combined data by `devices` first and calls
plot_four_platforms.py once per value. The line plots don't have this
problem since GPU count is the x-axis.

Usage:
    python3 plotting/megaplot.py results/scaling.csv results/scaling-mi250.csv \
        --outdir plots --metric fwd_gflops
"""
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="scaling.csv files to combine (one per machine)")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--metric", default="fwd_gflops",
                    choices=["fwd_gflops", "total_gflops", "fwd_wall_s", "total_wall_s"])
    ap.add_argument("--combined-csv", default="results/scaling-combined.csv")
    args = ap.parse_args()

    frames = []
    for c in args.csvs:
        df = pd.read_csv(c)
        df["__source_file"] = c
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    combined_path = Path(args.combined_csv)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.drop(columns="__source_file").to_csv(combined_path, index=False)
    print(f"wrote {combined_path} ({len(combined)} rows from {len(args.csvs)} file(s))")

    if "devices" not in combined.columns:
        print("no 'devices' column found; plotting combined csv as one chart")
        subprocess.run(
            [sys.executable, str(HERE / "plot_four_platforms.py"),
             str(combined_path), "--outdir", args.outdir, "--metric", args.metric],
            check=True,
        )
        return

    outdir = Path(args.outdir)
    for d in sorted(combined["devices"].dropna().unique()):
        d = int(d)
        sub = combined[combined["devices"] == d]
        subpath = outdir / f"_devices-{d}.csv"
        subpath.parent.mkdir(parents=True, exist_ok=True)
        sub.drop(columns="__source_file").to_csv(subpath, index=False)
        d_outdir = outdir / f"devices-{d}"
        print(f"==> devices={d}: {len(sub)} rows -> {d_outdir}")
        subprocess.run(
            [sys.executable, str(HERE / "plot_four_platforms.py"),
             str(subpath), "--outdir", str(d_outdir), "--metric", args.metric],
            check=True,
        )

    gflops_metric = "total_gflops" if args.metric.startswith("total") else "fwd_gflops"
    time_metric = "total_wall_s" if args.metric.startswith("total") else "fwd_wall_s"
    print(f"==> scaling line plots ({gflops_metric}, {time_metric}) -> {outdir}")
    subprocess.run(
        [sys.executable, str(HERE / "plot_scaling_lines.py"),
         str(combined_path), "--outdir", str(outdir),
         "--gflops-metric", gflops_metric, "--time-metric", time_metric],
        check=True,
    )


if __name__ == "__main__":
    main()
