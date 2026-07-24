#!/usr/bin/env python3
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
COLORS = {
    "CUDA / NVCC": "#76B900",          # NVIDIA green
    "CUDA / SCALE→NVIDIA": "#76B900",
    "HIP / HIPCC": "#D32F2F",          # AMD red
    "CUDA / SCALE→AMD": "#D32F2F",
}
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
})
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument(
        "--metric",
        default="fwd_gflops",
        choices=[
            "fwd_gflops",
            "total_gflops",
            "fwd_wall_s",
            "total_wall_s",
        ],
    )
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df["implementation"].isin(ORDER)].copy()
    if len(df) == 0:
        raise SystemExit("No matching rows found")
    df["implementation"] = df["implementation"].astype(str)
    df["machine"] = df["machine"].fillna("Unknown").astype(str)
    df[args.metric] = pd.to_numeric(df[args.metric], errors="coerce")
    agg = (
        df.groupby(["machine", "implementation"], as_index=False)
          .agg(
              median=(args.metric, "median"),
              q25=(args.metric, lambda x: x.quantile(0.25)),
              q75=(args.metric, lambda x: x.quantile(0.75)),
              n=(args.metric, "count"),
          )
    )
    agg["implementation"] = pd.Categorical(
        agg["implementation"],
        categories=ORDER,
        ordered=True,
    )
    agg = agg.sort_values(["machine", "implementation"])
    agg["label"] = (
        agg["machine"].astype(str)
        + "\n"
        + agg["implementation"].astype(str)
    )
    yerr = [
        agg["median"] - agg["q25"],
        agg["q75"] - agg["median"],
    ]
    colors = [
        COLORS.get(impl, "tab:blue")
        for impl in agg["implementation"]
    ]
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    bars = ax.bar(
        agg["label"],
        agg["median"],
        yerr=yerr,
        capsize=6,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )
    ylabel = (
        "GFLOP/s"
        if "gflops" in args.metric
        else "Wall Time (s)"
    )
    N = int(df["N"].iloc[0])
    B = int(df["B"].iloc[0])
    devices = ", ".join(sorted(df["machine"].unique()))
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"Continuous Wavelet Transform (Forward Only, N={N:,}, B={B})\n"
        f"{devices}",
        pad=20,
    )
    ax.grid(
        axis="y",
        color="#bdbdbd",
        alpha=0.35,
        linewidth=1.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(
        ax.get_xticklabels(),
        rotation=20,
        ha="right",
    )
    ymax = agg["median"].max()
    for bar, val in zip(bars, agg["median"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ymax * 0.015,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=16,
            fontweight="bold",
        )
    fig.tight_layout()
    stem = f"four_platforms_{args.metric}"
    pdf = outdir / f"{stem}.pdf"
    png = outdir / f"{stem}.png"
    csv = outdir / f"{stem}_summary.csv"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    agg.to_csv(csv, index=False)
    print(agg)
    print(f"wrote {pdf}")
    print(f"wrote {png}")
    print(f"wrote {csv}")
if __name__ == "__main__":
    main()
