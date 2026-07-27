#!/usr/bin/env python3
"""
Render a per-device Gantt-style timeline from ##DEVICE_TIMELINE lines emitted
by cwt_hip.cpp/cwt_cuda.cu's own host-side instrumentation: launch_ms (when a
kernel was enqueued on a device) and done_ms (when a non-blocking stream-query
poll first reported that device's work complete), both std::chrono offsets
from the start of the timed region.

Unlike a GPU-vendor profiler, this instrumentation lives in the shared
benchmark harness itself, so it applies identically -- same code path, same
methodology -- to all four implementations (native CUDA/HIP and both SCALE
targets) on either machine. See README for why rocprofv3 could not be used
for the SCALE targets.

Note this gives one (launch_ms, done_ms) interval per device per run, not a
full per-kernel-dispatch trace -- coarser than a real GPU profiler, but
enough to show the thing that matters here: overlapping intervals across
devices means concurrent execution, a staircase of non-overlapping intervals
means the dispatch is effectively serialized.

Usage:
    plot_device_timeline.py <log-file> --outdir plots --label hip-hipcc-N2048
    some_binary ... | plot_device_timeline.py - --outdir plots --label foo
"""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LINE_RE = re.compile(
    r'##DEVICE_TIMELINE\s+impl="(?P<impl>[^"]*)"\s+machine="(?P<machine>[^"]*)"\s+'
    r'gpus=(?P<gpus>\d+)\s+device=(?P<device>\d+)\s+'
    r'launch_ms=(?P<launch_ms>[-\d.eE]+)\s+done_ms=(?P<done_ms>[-\d.eE]+)'
)


def parse(fh):
    rows = []
    for line in fh:
        m = LINE_RE.search(line)
        if m:
            rows.append({
                "impl": m.group("impl"),
                "machine": m.group("machine"),
                "gpus": int(m.group("gpus")),
                "device": int(m.group("device")),
                "launch_ms": float(m.group("launch_ms")),
                "done_ms": float(m.group("done_ms")),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", help="file containing ##DEVICE_TIMELINE lines (or '-' for stdin)")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--label", default=None, help="figure title / filename suffix")
    args = ap.parse_args()

    if args.logfile == "-":
        rows = parse(sys.stdin)
    else:
        with open(args.logfile) as fh:
            rows = parse(fh)

    if not rows:
        print(f"error: no ##DEVICE_TIMELINE lines found in {args.logfile}", file=sys.stderr)
        sys.exit(1)

    impl = rows[0]["impl"]
    machine = rows[0]["machine"]
    gpus = rows[0]["gpus"]
    rows.sort(key=lambda r: r["device"])

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 1.5))
    for r in rows:
        ax.broken_barh([(r["launch_ms"], r["done_ms"] - r["launch_ms"])],
                        (r["device"] - 0.4, 0.8),
                        facecolor="#D32F2F", edgecolor="black", linewidth=0.2)

    ax.set_yticks([r["device"] for r in rows])
    ax.set_yticklabels([f'GPU {r["device"]}' for r in rows])
    ax.set_xlabel("Time since dispatch start (ms)")
    ax.set_ylabel("Device")
    title = args.label or f"{impl} ({machine}, gpus={gpus})"
    ax.set_title(f"Device timeline: {title}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    stem_label = (args.label or f"{impl}-{machine}-gpus{gpus}").replace(" ", "_").replace("/", "-")
    stem = os.path.join(args.outdir, f"device_timeline_{stem_label}")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"==> wrote {stem}.pdf / {stem}.png")


if __name__ == "__main__":
    main()
