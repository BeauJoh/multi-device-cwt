# multi-device-cwt

A small benchmark for scaling Continuous Wavelet Transforms (CWT) across multiple GPU devices, comparing native compilers against [SCALE](https://scale-lang.com) (a CUDA-source compiler that also targets non-NVIDIA hardware).

## Files

- `cwt_cuda.cu` — CUDA implementation of the forward CWT (Mexican hat wavelet), built two ways: natively with `nvcc`, and through SCALE targeting either NVIDIA or AMD hardware.
- `cwt_hip.cpp` — native HIP port of the same kernel, built with `hipcc`.
- `Makefile` — builds the four binaries below.
- `setup-backends.sh` — exports host-specific toolchain/arch environment variables (`CUDA_DEV_TARGET`, `HIP_DEV_TARGET`, `SCALE_ROOT`, etc.) based on `hostname -s`.

## Build targets

| Target | Source | Compiler | Notes |
|---|---|---|---|
| `cwt-cuda-nvcc` | `cwt_cuda.cu` | `nvcc` | Native CUDA build |
| `cwt-hip-hipcc` | `cwt_hip.cpp` | `hipcc` | Native HIP build |
| `cwt-cuda-scale-nvidia` | `cwt_cuda.cu` | SCALE's `nvcc` | CUDA source compiled through SCALE, targeting NVIDIA |
| `cwt-cuda-scale-amd` | `cwt_cuda.cu` | SCALE's `nvcc` | Same CUDA source compiled through SCALE, targeting AMD |

## Prerequisites

- A CUDA toolkit (`nvcc`) for the native and SCALE→NVIDIA builds.
- ROCm/HIP (`hipcc`) for the native HIP build.
- A SCALE install, unpacked at `./scale-1.7.3-Linux` (or point `SCALE_ROOT` elsewhere) for the two `scale-*` builds. You don't need to install this yourself — `make cwt-cuda-scale-nvidia` / `make cwt-cuda-scale-amd` depend on an `ensure-scale` target that downloads and unpacks the latest SCALE tarball automatically if it's missing.
- `hostname -s` must match one of the known hosts in `setup-backends.sh` (`milan2`, `milan0`, `hudson`, `faraday`, `cousteau`, `zenith`), or export `CUDA_DEV_TARGET` / `HIP_DEV_TARGET` yourself.

## Building

```bash
# picks up CUDA_DEV_TARGET / HIP_DEV_TARGET for this host, then builds all 4 binaries
source ./setup-backends.sh
make all
```

Or build a single target, e.g. `make cwt-cuda-nvcc`.

`make clean` removes all four binaries.

## Running

Each binary shares the same CLI:

```bash
./cwt-cuda-nvcc --N <samples> --B <batch> --gpus <n> --mode explicit|single --csv results.csv [--forward-only]
```

- `--N` — signal length (also used as the number of scales).
- `--B` — batch size.
- `--gpus` — number of devices to shard work across (`explicit` mode) or ignored (`single` mode, one device).
- `--csv` — appends a results row (implementation, machine, backend, N, B, devices, wall time, GFLOP/s, etc.) to this file.

Example, comparing all four builds at 4-way scaling on `N=2048`:

```bash
for bin in cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd; do
  CWT_IMPL="$bin" ./$bin --N 2048 --B 1 --gpus 4 --mode explicit --csv results.csv --forward-only
done
```

## Plotting

`plotting/plot_four_platforms.py` takes a results CSV and produces a grouped bar chart comparing all four implementations (median with IQR error bars), one grouping per machine:

```bash
pip install matplotlib pandas
python3 plotting/plot_four_platforms.py results.csv --outdir plots --metric fwd_gflops
```

`--metric` can be `fwd_gflops`, `total_gflops`, `fwd_wall_s`, or `total_wall_s`. Output is written as both `.pdf` and `.png`, plus a `_summary.csv` with the aggregated median/quartile values.
