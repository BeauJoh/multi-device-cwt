#!/usr/bin/env python3
"""Line plots of GFLOP/s and wall time vs problem size N, at each
implementation's own most-parallel GPU count.

This exists to test a specific hypothesis: that the native-HIP-gets-worse-
with-more-GPUs anomaly is (at least partly) a small-problem-size artifact
-- fixed per-device overhead (kernel launch, allocator, hipSetDevice
context switches) dominating when each GPU's chunk of work is tiny, which
would matter less as N grows and each chunk does proportionally more real
compute. If that's true, the gap between HIP/HIPCC and CUDA/SCALE->AMD
(and the "more GPUs is worse" slope for HIP/HIPCC specifically) should
narrow, flatten, or invert as N increases across a sweep run with
`make sweep-sizes`.

For each implementation, this picks the largest `devices` value present
for that implementation in the data (so CUDA/NVCC and CUDA/SCALE->NVIDIA
naturally use devices=1 on a single-GPU H100 box, while HIP/HIPCC and
CUDA/SCALE->AMD use devices=8 on the MI250 box, where the anomaly is most
extreme) and plots median +/- IQR against N (log2 x-axis).

Usage:
    python3 plotting/plot_vs_problem_size.py results/scaling-combined.csv --outdir plots
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
        "zorder": 2 if is_scale else 3,
    }


def plot_metric(df, metric, ylabel, title, out_stem, outdir, logy):
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")

    plotted = False
    for impl in ORDER:
        sub = df[df["implementation"] == impl]
        if sub.empty:
            continue
        max_devices = sub["devices"].max()
        sub = sub[sub["devices"] == max_devices]

        agg = (
            sub.groupby("N", as_index=False)
               .agg(
                   median=(metric, "median"),
                   q25=(metric, lambda x: x.quantile(0.25)),
                   q75=(metric, lambda x: x.quantile(0.75)),
               )
               .sort_values("N")
        )
        style = style_for(impl)
        machine = sub["machine"].iloc[0]
        label = f"{impl} ({machine}, gpus={int(max_devices)})"
        ax.plot(
            agg["N"], agg["median"], label=label,
            linewidth=2.5, markersize=9, **style,
        )
        ax.fill_between(
            agg["N"], agg["q25"], agg["q75"],
            color=style["color"], alpha=0.18, linewidth=0,
            zorder=1,
        )
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("N (signal length / scales)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=16)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_major_locator(mticker.FixedLocator(sorted(df["N"].dropna().unique())))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    if logy:
        ax.set_yscale("log")
    ax.grid(color="#bdbdbd", alpha=0.35, linewidth=1.0)
    ax.grid(which="minor", alpha=0.0)
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
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["implementation"].isin(ORDER)].copy()
    if df.empty:
        raise SystemExit("No matching rows found")
    df["implementation"] = df["implementation"].astype(str)
    df["machine"] = df["machine"].fillna("Unknown").astype(str)
    df["devices"] = pd.to_numeric(df["devices"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")
    for m in (args.gflops_metric, args.time_metric):
        df[m] = pd.to_numeric(df[m], errors="coerce")

    if df["N"].nunique() < 2:
        print("only one N value present; skipping vs-problem-size plot "
              "(need at least two sizes -- see make sweep-sizes)")
        return

    plot_metric(
        df, args.gflops_metric, "GFLOP/s",
        "CWT Forward Scaling vs Problem Size N\n"
        "(each series at its own largest swept GPU count)",
        f"vs_problem_size_{args.gflops_metric}", args.outdir, logy=False,
    )
    plot_metric(
        df, args.time_metric, "Wall Time (s)",
        "CWT Forward Wall Time vs Problem Size N\n"
        "(each series at its own largest swept GPU count)",
        f"vs_problem_size_{args.time_metric}", args.outdir, logy=True,
    )


if __name__ == "__main__":
    main()
