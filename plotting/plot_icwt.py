#!/usr/bin/env python3
"""
Render the forward-CWT + inverse-CWT (ICWT) round-trip demo as a 3-panel
figure: the 1D synthetic input signal, the 2D CWT coefficient heatmap, and
the 1D reconstructed signal overlaid on the original.

Reads the three CSVs written by run_icwt_demo() in cwt_hip.cpp/cwt_cuda.cu
(via --icwt --icwt-dir <dir>):
    <dir>/signal.csv  -- columns: t, fx, recon
    <dir>/scales.csv  -- columns: a_idx, scale
    <dir>/coeffs.csv  -- one row per scale index, N coefficient values per row

This is a correctness/consistency check, not a performance benchmark: it
confirms forward CWT + this simple inverse-CWT formula reconstructs
(approximately) the same signal that went in. See README for why the scale
grid is log-spaced and wide (needed for the reconstruction formula to work
well) rather than the narrow linear band still used implicitly by nothing
else in this repo.

Usage:
    plot_icwt.py <results-dir> --outdir plots --label cuda-nvcc-N256
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_signal(path):
    t, fx, recon = [], [], []
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            t.append(float(row["t"]))
            fx.append(float(row["fx"]))
            recon.append(float(row["recon"]))
    return t, fx, recon


def read_scales(path):
    scales = []
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            scales.append(float(row["scale"]))
    return scales


def read_coeffs(path):
    coeffs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            coeffs.append([float(v) for v in line.split(",")])
    return coeffs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("resultsdir", help="directory containing signal.csv/scales.csv/coeffs.csv (the --icwt-dir passed to the binary)")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--label", default=None, help="figure title / filename suffix, e.g. cuda-nvcc-N256")
    args = ap.parse_args()

    t, fx, recon = read_signal(os.path.join(args.resultsdir, "signal.csv"))
    scales = read_scales(os.path.join(args.resultsdir, "scales.csv"))
    coeffs = read_coeffs(os.path.join(args.resultsdir, "coeffs.csv"))

    n = len(t)
    a = len(scales)
    if len(coeffs) != a or any(len(row) != n for row in coeffs):
        raise SystemExit(
            f"error: coeffs.csv shape ({len(coeffs)} rows) doesn't match "
            f"scales.csv ({a}) / signal.csv ({n}) -- mismatched --icwt run?"
        )

    sse = sum((r - f) ** 2 for r, f in zip(recon, fx))
    sig_ss = sum(f * f for f in fx)
    rel_rms = ((sse / n) ** 0.5) / ((sig_ss / n) ** 0.5) if sig_ss > 0 else 0.0

    label = args.label or os.path.basename(os.path.normpath(args.resultsdir))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))

    axes[0].plot(t, fx, color="#333333", lw=1)
    axes[0].set_title(f"1D input signal ({label})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("f(t)")

    im = axes[1].imshow(
        coeffs, aspect="auto", origin="lower",
        extent=[t[0], t[-1], scales[0], scales[-1]],
        cmap="RdBu_r",
    )
    axes[1].set_title("2D CWT coefficient heatmap")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("scale a")
    plt.colorbar(im, ax=axes[1], fraction=0.03)

    axes[2].plot(t, fx, color="#333333", lw=1.5, label="original")
    axes[2].plot(t, recon, color="#D32F2F", lw=1, ls="--", label="reconstructed (ICWT)")
    axes[2].set_title(f"1D reconstructed signal -- rel. RMS error = {rel_rms:.2%}")
    axes[2].set_xlabel("t")
    axes[2].legend()

    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    stem_label = label.replace(" ", "_").replace("/", "-")
    stem = os.path.join(args.outdir, f"icwt_demo_{stem_label}")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"==> wrote {stem}.pdf / {stem}.png (rel_rms={rel_rms:.4%})")


if __name__ == "__main__":
    main()
