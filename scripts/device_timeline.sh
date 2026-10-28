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
set -o pipefail  # deliberately no -e: scaleenv relies on non-fatal command
# failures being tolerated, same as GNU Make recipe lines without -e.

cd "$(dirname "$0")/.."

# Toolchain selection (module load/swap, PATH, ...) is the caller's shell
# environment. SCALE targets come from CUDA_SM / HIP_ARCH, which `make
# device-timelines` exports; when running this script by hand, set them
# yourself (e.g. HIP_ARCH=gfx950 or CUDA_SM=sm_90).

BIN=${1:?usage: $0 <path-to-binary> [args...]}
shift

# The cwt-cuda-scale-* binaries are SCALE builds and only run correctly with
# SCALE's own runtime libraries on LD_LIBRARY_PATH, which `scaleenv` sets up
# -- same as the `( source .../scaleenv ... )` subshells in `make sweep`.
SCALE_ROOT="${SCALE_ROOT:-${SCALE_DIR:-$(pwd)/scale-latest}}"
if [[ "$(basename "$BIN")" == *scale* ]]; then
  if [ ! -f "$SCALE_ROOT/bin/scaleenv" ]; then
    echo "error: $BIN looks like a SCALE build but $SCALE_ROOT/bin/scaleenv is missing (run 'make ensure-scale' first)" >&2
    exit 1
  fi
  if [[ "$(basename "$BIN")" == *scale-nvidia* ]]; then
    scale_arch="${CUDA_SM:?set CUDA_SM (e.g. sm_90) for a SCALE->NVIDIA run}"
  else
    scale_arch="${HIP_ARCH:?set HIP_ARCH (e.g. gfx950) for a SCALE->AMD run}"
  fi
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
