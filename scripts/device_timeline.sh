#!/usr/bin/env bash
# Run one multi-device-cwt binary and render its host-side ##DEVICE_TIMELINE
# instrumentation (launch_ms/done_ms per device, from cwt_hip.cpp/cwt_cuda.cu)
# as a static per-device timeline figure.
#
# Unlike a GPU-vendor profiler (rocprofv3 etc. -- see README for why that
# doesn't work for the SCALE targets), this works identically for all four
# implementations (native CUDA/HIP and both SCALE targets) on either
# machine, since the instrumentation lives in the shared benchmark harness
# itself rather than depending on vendor tooling.
#
# Usage:
#   scripts/device_timeline.sh <path-to-binary> [binary args...]
#   LABEL=<fixed-name> scripts/device_timeline.sh <path-to-binary> [binary args...]
#
# Examples:
#   scripts/device_timeline.sh ./cwt-hip-hipcc      --mode explicit --N 2048  --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/device_timeline.sh ./cwt-cuda-scale-amd --mode explicit --N 2048  --B 1 --gpus 8 --csv /dev/null --forward-only
#
# Output: raw stdout under results/device-timeline-<label>.log, plus a
# rendered plots/device_timeline_<label>.pdf/.png Gantt-style figure (one row
# per device -- overlapping bars mean concurrent execution, a staircase
# means serialized dispatch).
#
# By default <label> is "<timestamp>-<binary>" (fine for ad hoc exploration).
# Set LABEL explicitly (e.g. from `make device-timelines`) for a fixed,
# reproducible filename with no timestamp -- reruns overwrite in place, which
# is what you want when pulling a fixed set of figures into a paper.
set -o pipefail  # deliberately no -e: setup-backends.sh/scaleenv rely on
# non-fatal command failures (e.g. missing `module` binary) being tolerated,
# same as how GNU Make runs recipe lines without -e by default.

cd "$(dirname "$0")/.."

# Same environment setup `make build`/`make sweep` use.
. ./setup-backends.sh

# Make sure ROCm's own lib dirs are on LD_LIBRARY_PATH as a fallback *before*
# scaleenv (below) prepends SCALE's bundled dir in front of this. SCALE's own
# libhsa-runtime64.so/libamdhip64.so still win (they're first on the path),
# preserving correctness, but anything SCALE's bundle doesn't ship itself
# (e.g. libamd_comgr.so, used for on-the-fly kernel finalization) still
# resolves instead of failing at runtime -- without this, cwt-cuda-scale-amd
# aborts with "No CUDA devices found" even with no profiler involved at all.
export LD_LIBRARY_PATH="${ROCM_PATH:-/opt/rocm}/lib:${ROCM_PATH:-/opt/rocm}/lib64:${LD_LIBRARY_PATH:-}"

BIN=${1:?usage: $0 <path-to-binary> [args...]}
shift

# The cwt-cuda-scale-* binaries are SCALE builds and only run correctly with
# SCALE's own runtime libraries on LD_LIBRARY_PATH, which `scaleenv` sets up
# -- same as the `( source .../scaleenv ... )` subshells in `make sweep`.
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
mkdir -p results
LOG="results/device-timeline-$LABEL.log"

"$BIN" "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

if [ "$status" -ne 0 ]; then
  echo "error: $BIN exited with status $status -- see $LOG" >&2
  exit "$status"
fi

if ! grep -q '##DEVICE_TIMELINE' "$LOG"; then
  echo "error: no ##DEVICE_TIMELINE lines in $LOG -- was the binary built from the current cwt_hip.cpp/cwt_cuda.cu?" >&2
  exit 1
fi

echo "==> raw run log written to $LOG"

PLOT_PY="$(pwd)/.venv-plots/bin/python3"
[ -x "$PLOT_PY" ] || PLOT_PY="python3"

if "$PLOT_PY" plotting/plot_device_timeline.py "$LOG" --outdir plots --label "$LABEL"; then
  echo "==> timeline figure: plots/device_timeline_${LABEL}.pdf / .png"
else
  echo "warning: timeline plotting failed -- raw log is still at $LOG for manual inspection" >&2
fi
