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
# Output: a .pftrace file under results/rocprof-<timestamp>/ that you can
# drag-and-drop at https://ui.perfetto.dev
set -euo pipefail

export PATH="$PATH:/opt/rocm/bin"

BIN=${1:?usage: $0 <path-to-binary> [args...]}
shift

OUTDIR="results/rocprof-$(date +%Y%m%d-%H%M%S)-$(basename "$BIN")"
mkdir -p "$OUTDIR"

rocprofv3 \
  --kernel-trace \
  --hip-trace \
  --memory-copy-trace \
  --output-format pftrace \
  -d "$OUTDIR" \
  -- "$BIN" "$@"

echo "==> trace written under $OUTDIR"
echo "==> open the .pftrace file at https://ui.perfetto.dev and look at the per-Agent"
echo "    kernel-dispatch tracks: overlapping intervals = concurrent, staircase = serial"
