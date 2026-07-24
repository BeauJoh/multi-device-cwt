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
- A SCALE install, unpacked at `./scale-1.7.2` (or point `SCALE_ROOT` elsewhere) for the two `scale-*` builds. You don't need to install this yourself — `make cwt-cuda-scale-nvidia` / `make cwt-cuda-scale-amd` depend on an `ensure-scale` target that downloads and unpacks SCALE 1.7.2 (pinned, via `SCALE_TARBALL_URL`) automatically if it's missing.
- `hostname -s` must match one of the known hosts in `setup-backends.sh` (`milan2`, `milan0`, `hudson`, `faraday`, `cousteau`, `zenith`), or export `CUDA_DEV_TARGET` / `HIP_DEV_TARGET` yourself.

## Building

Just run:

```bash
make
```

This sources `setup-backends.sh` itself, checks `BACKENDS` for the current host, and builds only what's relevant: `cuda` → `cwt-cuda-nvcc` + `cwt-cuda-scale-nvidia`; `hip` → `cwt-hip-hipcc` + `cwt-cuda-scale-amd`; a host with both (e.g. `zenith`) builds all four. No need to `source` anything first.

To force every target regardless of `BACKENDS` (e.g. cross-checking on a dev box), use `make all`. Or build a single target directly, e.g. `make cwt-cuda-nvcc`.

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

## Sweeping

```bash
make sweep
```

Builds whatever this host supports (same `BACKENDS` filtering as plain `make`), then runs every applicable binary across GPU counts `1..N`, `REPS` times each (default 5), appending every run to `results/scaling.csv`. Device counts are auto-detected — `nvidia-smi -L | wc -l` for CUDA, `rocminfo | grep -c 'Device Type:.*GPU'` for HIP — not hardcoded, so it adapts to whatever's actually plugged into that box (1 on the H100 box, 8 on the MI250 box).

Override any of `N`, `B`, `REPS`, `CSV`, e.g.:

```bash
make sweep N=4096 REPS=10
```

`make clean-results` removes the results directory.

## Plotting

`plotting/plot_four_platforms.py` takes a results CSV and produces a grouped bar chart comparing all four implementations (median with IQR error bars), one grouping per machine:

```bash
pip install matplotlib pandas
python3 plotting/plot_four_platforms.py results.csv --outdir plots --metric fwd_gflops
```

`--metric` can be `fwd_gflops`, `total_gflops`, `fwd_wall_s`, or `total_wall_s`. Output is written as both `.pdf` and `.png`, plus a `_summary.csv` with the aggregated median/quartile values.

Note: this script medians across every row it's given, so only feed it rows from a single `devices` (GPU count) at a time — see `make megaplot` below, which handles that split automatically.

### Combining results from both machines

The H100 and MI250 boxes can ssh to each other and share the same repo path (`/home/smc/multi-device-cwt`), so `make megaplot` pulls the other box's `results/scaling.csv` over `scp`, combines it with this box's own, and produces one four-platform comparison chart per GPU count (since blending device counts into one median wouldn't be meaningful):

```bash
# from the H100 box:
make megaplot REMOTE_HOST=mi250

# from the MI250 box:
make megaplot REMOTE_HOST=h100
```

`REMOTE_HOST` should be whatever ssh alias reaches the other box. Override `REMOTE_PATH` if the repo lives somewhere other than `/home/smc/multi-device-cwt` on the remote, and `METRIC`/`PLOTS_DIR` the same way as above. Output goes to `plots/devices-<N>/` per GPU count, with the combined raw CSV at `results/scaling-combined.csv`.

You can also run this by hand for more control — `python3 plotting/megaplot.py <csv1> <csv2> ... --outdir plots --metric fwd_gflops`.
