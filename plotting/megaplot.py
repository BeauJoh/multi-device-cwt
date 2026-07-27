#!/usr/bin/env python3
"""Combine scaling.csv files from multiple machines and produce, per
problem size N:
  - one four-platform bar-chart snapshot per GPU count (plot_four_platforms.py)
  - two line plots (GFLOP/s and wall time vs GPU count) across the whole
    sweep, with an IQR ribbon (plot_scaling_lines.py)
plus, if more than one N is present in the data:
  - two line plots (GFLOP/s and wall time vs N) at each implementation's
    own largest swept GPU count (plot_vs_problem_size.py) -- see
    `make sweep-sizes`.

Why facet by N at all: a sweep run with `make sweep-sizes` appends
multiple problem sizes to the same results.csv, and both the bar charts
and the GPU-count line charts take the median across every row they're
given -- mixing rows from different N values into one median would be
meaningless (GFLOP/s and wall time both scale sharply with N). Within a
given N, the bar charts additionally get split by `devices` for the same
reason (see plot_four_platforms.py). The vs-N plot is the opposite: it
deliberately spans every N, since that's the whole point of it.

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


def plot_one_n(df_n, outdir, metric):
    """Bar charts (per devices count) + GPU-count line charts, for rows
    already restricted to a single N."""
    if "devices" not in df_n.columns:
        print("no 'devices' column found; plotting as one chart")
        subpath = outdir / "_all.csv"
        subpath.parent.mkdir(parents=True, exist_ok=True)
        df_n.to_csv(subpath, index=False)
        subprocess.run(
            [sys.executable, str(HERE / "plot_four_platforms.py"),
             str(subpath), "--outdir", str(outdir), "--metric", metric],
            check=True,
        )
        return

    for d in sorted(df_n["devices"].dropna().unique()):
        d = int(d)
        sub = df_n[df_n["devices"] == d]
        subpath = outdir / f"_devices-{d}.csv"
        subpath.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(subpath, index=False)
        d_outdir = outdir / f"devices-{d}"
        print(f"==> devices={d}: {len(sub)} rows -> {d_outdir}")
        subprocess.run(
            [sys.executable, str(HERE / "plot_four_platforms.py"),
             str(subpath), "--outdir", str(d_outdir), "--metric", metric],
            check=True,
        )

    gflops_metric = "total_gflops" if metric.startswith("total") else "fwd_gflops"
    time_metric = "total_wall_s" if metric.startswith("total") else "fwd_wall_s"
    n_csv = outdir / "_n.csv"
    df_n.to_csv(n_csv, index=False)
    print(f"==> scaling line plots ({gflops_metric}, {time_metric}) -> {outdir}")
    subprocess.run(
        [sys.executable, str(HERE / "plot_scaling_lines.py"),
         str(n_csv), "--outdir", str(outdir),
         "--gflops-metric", gflops_metric, "--time-metric", time_metric],
        check=True,
    )


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
    combined = combined.drop(columns="__source_file")

    outdir = Path(args.outdir)

    if "N" not in combined.columns:
        plot_one_n(combined, outdir, args.metric)
        return

    ns = sorted(combined["N"].dropna().unique())
    for n in ns:
        n = int(n)
        df_n = combined[combined["N"] == n]
        n_outdir = outdir / f"N-{n}" if len(ns) > 1 else outdir
        print(f"==> N={n}: {len(df_n)} rows -> {n_outdir}")
        plot_one_n(df_n, n_outdir, args.metric)

    if len(ns) > 1:
        gflops_metric = "total_gflops" if args.metric.startswith("total") else "fwd_gflops"
        time_metric = "total_wall_s" if args.metric.startswith("total") else "fwd_wall_s"
        print(f"==> vs-problem-size plots ({gflops_metric}, {time_metric}) -> {outdir}")
        subprocess.run(
            [sys.executable, str(HERE / "plot_vs_problem_size.py"),
             str(combined_path), "--outdir", str(outdir),
             "--gflops-metric", gflops_metric, "--time-metric", time_metric],
            check=True,
        )


if __name__ == "__main__":
    main()
