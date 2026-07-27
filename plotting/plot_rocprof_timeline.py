#!/usr/bin/env python3
"""
Render a per-GPU kernel-dispatch timeline (Gantt-style) from a rocprofv3
`--kernel-trace --output-format csv` output directory -- a static figure
alternative to opening the trace in the Perfetto UI.

Reads the `*kernel_trace.csv` file rocprofv3 writes (columns include
Agent_Id, Start_Timestamp, End_Timestamp in ns) and draws one horizontal bar
per kernel dispatch, grouped by GPU agent: overlapping bars across agents
means concurrent execution, a staircase means serialized dispatch.

Usage:
    plot_rocprof_timeline.py <rocprofv3-output-dir> --outdir plots --label N2048-hip-hipcc
"""
import argparse
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def find_csv(trace_dir, suffix):
    matches = glob.glob(os.path.join(trace_dir, "**", f"*{suffix}"), recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime)
    return matches[-1]


def agent_sort_key(agent_id):
    m = re.search(r"(\d+)", str(agent_id))
    return int(m.group(1)) if m else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir", help="the -d/--output-directory passed to rocprofv3")
    ap.add_argument("--outdir", default="plots")
    ap.add_argument("--label", default=None, help="figure title / filename suffix (default: trace_dir's basename)")
    args = ap.parse_args()

    kernel_csv = find_csv(args.trace_dir, "kernel_trace.csv")
    if kernel_csv is None:
        print(f"error: no *kernel_trace.csv found under {args.trace_dir} "
              f"(did you run rocprofv3 with --kernel-trace --output-format csv?)", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(kernel_csv)
    df = df[df["Kind"] == "KERNEL_DISPATCH"].copy()
    if df.empty:
        print(f"error: {kernel_csv} has no KERNEL_DISPATCH rows", file=sys.stderr)
        sys.exit(1)

    t0 = df["Start_Timestamp"].min()
    df["start_ms"] = (df["Start_Timestamp"] - t0) / 1e6
    df["dur_ms"] = (df["End_Timestamp"] - df["Start_Timestamp"]) / 1e6

    agents = sorted(df["Agent_Id"].unique(), key=agent_sort_key)
    agent_row = {a: i for i, a in enumerate(agents)}

    fig, ax = plt.subplots(figsize=(10, 0.6 * len(agents) + 1.5))
    for agent in agents:
        sub = df[df["Agent_Id"] == agent]
        bars = list(zip(sub["start_ms"], sub["dur_ms"]))
        ax.broken_barh(bars, (agent_row[agent] - 0.4, 0.8), facecolor="#D32F2F", edgecolor="black", linewidth=0.2)

    ax.set_yticks(list(agent_row.values()))
    ax.set_yticklabels(list(agent_row.keys()))
    ax.set_xlabel("Time since first kernel dispatch (ms)")
    ax.set_ylabel("GPU agent")
    title = args.label or os.path.basename(os.path.normpath(args.trace_dir))
    ax.set_title(f"Kernel dispatch timeline: {title}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.join(args.outdir, f"rocprof_timeline_{title}".replace(" ", "_"))
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=150)
    print(f"==> wrote {stem}.pdf / {stem}.png")


if __name__ == "__main__":
    main()
