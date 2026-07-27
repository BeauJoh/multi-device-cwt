#!/usr/bin/env bash
# Generate a rocprofv3 kernel-dispatch trace for one run of a multi-device-cwt
# binary, with no source changes required, and render it as a static
# per-GPU timeline figure -- no Perfetto UI/install needed.
#
# rocprofv3 (part of ROCm's ROCprofiler-SDK) records genuine device-side
# kernel-dispatch timestamps per GPU agent, which is stronger evidence of
# concurrent-vs-serial multi-GPU execution than the host-side ##DEVICE_TIMELINE
# polling instrumentation in cwt_hip.cpp/cwt_cuda.cu.
#
# Usage:
#   scripts/rocprof_trace.sh <path-to-binary> [binary args...]
#   LABEL=<fixed-name> scripts/rocprof_trace.sh <path-to-binary> [binary args...]
#
# Examples:
#   scripts/rocprof_trace.sh ./cwt-hip-hipcc      --mode explicit --N 2048  --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/rocprof_trace.sh ./cwt-hip-hipcc      --mode explicit --N 16384 --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/rocprof_trace.sh ./cwt-cuda-scale-amd --mode explicit --N 2048  --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/rocprof_trace.sh ./cwt-cuda-scale-amd --mode explicit --N 16384 --B 1 --gpus 8 --csv /dev/null --forward-only
#
# Output: raw rocprofv3 CSVs under results/rocprof-<label>/, plus a rendered
# plots/rocprof_timeline_<label>.pdf/.png Gantt-style figure (one row per GPU
# agent, one bar per kernel dispatch -- overlapping bars across agents means
# concurrent execution, a staircase means serialized dispatch).
#
# By default <label> is "<timestamp>-<binary>" (fine for ad hoc exploration).
# Set LABEL explicitly (e.g. from `make rocprof-timelines`) for a fixed,
# reproducible filename with no timestamp -- reruns overwrite in place, which
# is what you want when pulling a fixed set of figures into a paper.
set -o pipefail  # deliberately no -e: setup-backends.sh/scaleenv rely on
# non-fatal command failures (e.g. missing `module` binary) being tolerated,
# same as how GNU Make runs recipe lines without -e by default.

cd "$(dirname "$0")/.."

# Same environment setup `make build`/`make sweep` use, so this resolves the
# right ROCM_PATH/BACKENDS/MACHINE for the current host.
. ./setup-backends.sh

export PATH="$PATH:${ROCM_PATH:-/opt/rocm}/bin"

# rocprofv3 dlopen()s some of its own support libraries (e.g.
# libhsa-amd-aqlprofile64.so.1) at trace time rather than relying on the
# target binary's own RPATH, so make sure ROCm's lib dirs are explicitly on
# LD_LIBRARY_PATH regardless of what setup-backends.sh already added.
export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib:${ROCM_PATH:-/opt/rocm}/lib64:${LD_LIBRARY_PATH:-}"

BIN=${1:?usage: $0 <path-to-binary> [args...]}
shift

# The cwt-cuda-scale-* binaries are SCALE builds and only run correctly with
# SCALE's own runtime libraries (e.g. libredscale.so) on LD_LIBRARY_PATH,
# which `scaleenv` sets up -- same as the `( source .../scaleenv ... )`
# subshells in `make sweep`. Native binaries (cwt-hip-hipcc, cwt-cuda-nvcc)
# don't need this.
SCALE_ROOT="${SCALE_ROOT:-$(pwd)/scale-1.7.2}"
if [[ "$(basename "$BIN")" == *scale* ]]; then
  if [ ! -f "$SCALE_ROOT/bin/scaleenv" ]; then
    echo "error: $BIN looks like a SCALE build but $SCALE_ROOT/bin/scaleenv is missing (run 'make ensure-scale' first)" >&2
    exit 1
  fi
  hip_arch="${HIP_DEV_TARGET:-gfx908}"
  cuda_sm="sm_${CUDA_DEV_TARGET#sm_}"
  scale_arch="$hip_arch"
  [[ "$(basename "$BIN")" == *scale-nvidia* ]] && scale_arch="$cuda_sm"
  source "$SCALE_ROOT/bin/scaleenv" "$scale_arch"
fi

LABEL="${LABEL:-$(date +%Y%m%d-%H%M%S)-$(basename "$BIN")}"
OUTDIR="results/rocprof-$LABEL"
# Clean any stale run under this label first so a fixed LABEL truly
# overwrites in place (rocprofv3 nests output under a hostname/pid
# subdirectory each run, so old files would otherwise just accumulate).
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

rocprofv3 \
  --kernel-trace \
  --hip-trace \
  --memory-copy-trace \
  --output-format csv \
  -d "$OUTDIR" \
  -- "$BIN" "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo "error: rocprofv3 exited with status $status -- see output above" >&2
  exit "$status"
fi

echo "==> raw trace CSVs written under $OUTDIR"

# Render the static timeline figure automatically -- reuse this project's
# own plotting venv (bootstrapped by `make megaplot`/`make ensure-plot-deps`)
# if it exists, otherwise fall back to system python3.
PLOT_PY="$(pwd)/.venv-plots/bin/python3"
[ -x "$PLOT_PY" ] || PLOT_PY="python3"

if "$PLOT_PY" plotting/plot_rocprof_timeline.py "$OUTDIR" --outdir plots --label "$LABEL"; then
  echo "==> timeline figure: plots/rocprof_timeline_${LABEL}.pdf / .png"
else
  echo "warning: timeline plotting failed -- raw CSVs are still under $OUTDIR for manual inspection" >&2
fi
