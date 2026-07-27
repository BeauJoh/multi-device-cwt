#!/usr/bin/env python3
"""Line plots of GFLOP/s and wall time vs GPU count, across all four platforms.

Unlike plot_four_platforms.py (one bar-chart snapshot per GPU count), this
plots the whole 1..8 GPU sweep as connected lines with an IQR ribbon, so
scaling behaviour is visible directly. Native and SCALE-compiled builds
targeting the same hardware share a base colour (NVIDIA green / AMD red),
but SCALE builds use a darker shade of that colour, a dashed line, and a
square marker (native uses a solid line and a circle marker), so the two
are easy to tell apart at a glance.

The H100 box only has one GPU, so its two series (CUDA / NVCC, CUDA /
SCALE→NVIDIA) will just show as a single point at devices=1 — that's
expected, not a bug.

Usage:
    python3 plotting/plot_scaling_lines.py results/scaling-combined.csv --outdir plots
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ORDER = [
    "CUDA / NVCC",
    "CUDA / SCALE→NVIDIA",
    "HIP / HIPCC",
    "CUDA / SCALE→AMD",
]

BASE_COLORS = {
    "CUDA / NVCC": "#76B900",          # NVIDIA green
    "CUDA / SCALE→NVIDIA": "#76B900",
    "HIP / HIPCC": "#D32F2F",          # AMD red
    "CUDA / SCALE→AMD": "#D32F2F",
}

IS_SCALE = {
    "CUDA / NVCC": False,
    "CUDA / SCALE→NVIDIA": True,
    "HIP / HIPCC": False,
    "CUDA / SCALE→AMD": True,
}

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 22,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 22,
})


def darken(hex_color, factor=0.55):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (max(0, int(c * factor)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def style_for(impl):
    base = BASE_COLORS.get(impl, "tab:blue")
    is_scale = IS_SCALE.get(impl, False)
    return {
        "color": darken(base) if is_scale else base,
        "linestyle": "--" if is_scale else "-",
        "marker": "s" if is_scale else "o",
        # Native's circle marker is drawn on top of SCALE's square where
        # they coincide (zorder, not draw order, so this doesn't affect
        # legend ordering, which follows the ax.plot() call sequence below).
        "zorder": 3 if not is_scale else 2,
    }


def plot_metric(df, metric, ylabel, title, out_stem, outdir):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    plotted = False
    for impl in ORDER:
        sub = df[df["implementation"] == impl]
        if sub.empty:
            continue
        agg = (
            sub.groupby("devices", as_index=False)
               .agg(
                   median=(metric, "median"),
                   q25=(metric, lambda x: x.quantile(0.25)),
                   q75=(metric, lambda x: x.quantile(0.75)),
               )
               .sort_values("devices")
        )
        style = style_for(impl)
        machine = sub["machine"].iloc[0]
        label = f"{impl} ({machine})"
        ax.plot(
            agg["devices"], agg["median"], label=label,
            linewidth=2.5, markersize=9, **style,
        )
        ax.fill_between(
            agg["devices"], agg["q25"], agg["q75"],
            color=style["color"], alpha=0.18, linewidth=0,
            zorder=1,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("GPUs")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    all_devices = sorted(int(d) for d in df["devices"].dropna().unique())
    if all_devices:
        ax.set_xticks(all_devices)
    ax.grid(color="#bdbdbd", alpha=0.35, linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{out_stem}.pdf"
    png = outdir / f"{out_stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--gflops-metric", default="fwd_gflops",
                    choices=["fwd_gflops", "total_gflops"])
    ap.add_argument("--time-metric", default="fwd_wall_s",
                    choices=["fwd_wall_s", "total_wall_s"])
    ap.add_argument("--N", type=int, default=None,
                    help="restrict to this signal length N (default: whatever's in the csv)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["implementation"].isin(ORDER)].copy()
    if df.empty:
        raise SystemExit("No matching rows found")
    df["implementation"] = df["implementation"].astype(str)
    df["machine"] = df["machine"].fillna("Unknown").astype(str)
    df["devices"] = pd.to_numeric(df["devices"], errors="coerce")
    for m in (args.gflops_metric, args.time_metric):
        df[m] = pd.to_numeric(df[m], errors="coerce")

    if args.N is not None:
        df = df[df["N"] == args.N]
        if df.empty:
            raise SystemExit(f"No rows with N={args.N}")

    N = int(df["N"].iloc[0])
    B = int(df["B"].iloc[0])

    plot_metric(
        df, args.gflops_metric, "GFLOP/s",
        f"CWT Forward Scaling vs GPU Count (N={N:,}, B={B})",
        f"scaling_{args.gflops_metric}", args.outdir,
    )
    plot_metric(
        df, args.time_metric, "Wall Time (s)",
        f"CWT Forward Wall Time vs GPU Count (N={N:,}, B={B})",
        f"scaling_{args.time_metric}", args.outdir,
    )


if __name__ == "__main__":
    main()
