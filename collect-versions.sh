#!/usr/bin/env bash
#
# collect-versions.sh
#
# Prints the driver/toolkit/software versions needed to resolve the
# remaining "confirm version" TODOs in the Hardware Platforms section of
# the whitepaper. Run this on each compute box (H100 and MI250X) and
# paste the output back -- it does not modify anything, and every check
# is skipped cleanly if the relevant tool isn't present on that box.
#
# Usage: ./collect-versions.sh

set -uo pipefail  # not -e: keep going if one tool is missing on this host

section() { printf '\n=== %s ===\n' "$1"; }

section "Host"
hostname -s
uname -srm

section "SCALE"
scale_root="${SCALE_ROOT:-./scale-1.7.2}"
if [ -d "$scale_root" ]; then
  echo "SCALE root: $scale_root"
  # scaleenv has no --version flag as of 1.7.2; the pinned version comes
  # from the Makefile's SCALE_TARBALL_URL instead.
  grep -m1 -i 'scale.*tarball\|scale-1\.' Makefile 2>/dev/null \
    || echo "(pinned version not found in ./Makefile -- check SCALE_TARBALL_URL manually)"
else
  echo "SCALE not found at \$SCALE_ROOT or ./scale-1.7.2"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  section "NVIDIA driver / GPU"
  nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv

  section "CUDA toolkits installed"
  for cuda_dir in /usr/local/cuda-*; do
    [ -d "$cuda_dir" ] || continue
    printf '%s: ' "$cuda_dir"
    "$cuda_dir/bin/nvcc" --version 2>/dev/null | grep -i release
  done
  if command -v nvcc >/dev/null 2>&1; then
    printf 'default nvcc (\$PATH): '
    nvcc --version | grep -i release
  fi
fi

if command -v rocminfo >/dev/null 2>&1; then
  section "ROCm / HSA runtime"
  rocminfo 2>/dev/null | grep -E "Runtime Version|Runtime Ext Version|XNACK enabled"

  section "ROCm package version"
  if [ -f /opt/rocm/.info/version ]; then
    cat /opt/rocm/.info/version
  else
    echo "(/opt/rocm/.info/version not found -- listing /opt/rocm-* and dpkg as fallback)"
    ls -d /opt/rocm-* 2>/dev/null
    dpkg -l 2>/dev/null | grep -E 'rocm-dev |rocm-core ' || true
  fi

  if command -v hipcc >/dev/null 2>&1; then
    section "hipcc"
    hipcc --version
  fi
  if command -v rocm-smi >/dev/null 2>&1; then
    section "amdgpu driver version (rocm-smi)"
    rocm-smi --showdriverversion 2>/dev/null
  fi
fi

section "OS release"
grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null

