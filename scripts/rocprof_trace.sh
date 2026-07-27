#!/usr/bin/env bash
# Generate a Perfetto-viewable rocprofv3 trace for one run of a multi-device-cwt
# binary, with no source changes required.
#
# rocprofv3 (part of ROCm's ROCprofiler-SDK) records genuine device-side
# kernel-dispatch timestamps per GPU agent, which is stronger evidence of
# concurrent-vs-serial multi-GPU execution than the host-side ##DEVICE_TIMELINE
# polling instrumentation in cwt_hip.cpp/cwt_cuda.cu.
#
# Usage:
#   scripts/rocprof_trace.sh <path-to-binary> [binary args...]
#
# Examples:
#   scripts/rocprof_trace.sh ./cwt-hip-hipcc      --mode explicit --N 2048  --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/rocprof_trace.sh ./cwt-hip-hipcc      --mode explicit --N 16384 --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/rocprof_trace.sh ./cwt-cuda-scale-amd --mode explicit --N 2048  --B 1 --gpus 8 --csv /dev/null --forward-only
#   scripts/rocprof_trace.sh ./cwt-cuda-scale-amd --mode explicit --N 16384 --B 1 --gpus 8 --csv /dev/null --forward-only
#
# Output: a .pftrace file under results/rocprof-<timestamp>-<binary>/ that you
# can drag-and-drop at https://ui.perfetto.dev
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

OUTDIR="results/rocprof-$(date +%Y%m%d-%H%M%S)-$(basename "$BIN")"
mkdir -p "$OUTDIR"

rocprofv3 \
  --kernel-trace \
  --hip-trace \
  --memory-copy-trace \
  --output-format pftrace \
  -d "$OUTDIR" \
  -- "$BIN" "$@"
status=$?

if [ "$status" -ne 0 ]; then
  echo "error: rocprofv3 exited with status $status -- see output above" >&2
  exit "$status"
fi

echo "==> trace written under $OUTDIR"
echo "==> open the .pftrace file at https://ui.perfetto.dev and look at the per-Agent"
echo "    kernel-dispatch tracks: overlapping intervals = concurrent, staircase = serial"
