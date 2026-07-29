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
- `hostname -s` must match a known host in `setup-backends.sh`, or export `CUDA_DEV_TARGET` / `HIP_DEV_TARGET` yourself.

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
./cwt-cuda-nvcc --N <samples> --B <batch> --gpus <n> --mode explicit|single --csv results.csv [--forward-only] [--verify] [--verify-samples <k>] [--icwt] [--icwt-dir <dir>] [--icwt-max-rel-rms <v>]
```

- `--N` — signal length (also used as the number of scales).
- `--B` — batch size.
- `--gpus` — number of devices to shard work across (`explicit` mode) or ignored (`single` mode, one device).
- `--csv` — appends a results row (implementation, machine, backend, N, B, devices, wall time, GFLOP/s, etc.) to this file.
- `--verify` — after the timed region, check a random sample of the forward-CWT output against a CPU reference computed with the identical math (same summation order, so it's a meaningful bit-level check, not just "close enough"). Off by default: without it, the `rel_err` column is `0.0` meaning *not checked*, not *verified correct* -- this was previously the unconditional, silent behavior for every row in every results CSV, i.e. no correctness check had actually ever been run. Prints `verify=PASS`/`FAIL` and the max sampled relative error; `--verify-samples <k>` controls how many random `(batch, scale, time)` points are checked (default 2000, fixed RNG seed so native and SCALE builds check the exact same points).
- `--icwt` — after the timed region, run a host-side inverse-CWT (ICWT) round-trip demo on the already-computed forward output and dump CSVs for plotting (see "Inverse CWT (ICWT) round-trip demo" below). Requires `--B 1`. `--icwt-dir <dir>` sets where the CSVs land (default `results/icwt`). `--icwt-max-rel-rms <v>` (default `0.05`, i.e. 5%) is a hard gate: if the reconstruction's relative RMS error exceeds it, the program prints `icwt=FAIL` and **exits with a nonzero status** (so `make icwt-demo`'s `|| exit 1` aborts the whole target rather than silently accepting a regression). On the current synthetic signal this normally sits at ~0.7-3.1% (well under the 5% gate) -- a FAIL here means something has actually regressed (e.g. an accidental change to the scale grid, wavelet, or reconstruction formula), not just normal run-to-run noise.

Note on the scale grid: `scales` is a log-spaced range from `0.001` to `2.0` (`A` values, `A=N`), not a narrow linear band -- this matters for the ICWT round-trip below (a wide, log-spaced scale grid is needed for the classical CWT reconstruction formula to work well; a narrow band is fine for the forward transform but reconstructs poorly). Changing the scale *values* has no effect on any previously-collected timing/GFLOP-s numbers, since forward-transform cost only depends on the scale *count* (`A=N`) and `N`, never on the actual scale values.

Example, comparing all four builds at 4-way scaling on `N=2048`:

```bash
for bin in cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd; do
  CWT_IMPL="$bin" ./$bin --N 2048 --B 1 --gpus 4 --mode explicit --csv results.csv --forward-only
done
```

### Per-device timeline diagnostic

Every run also prints one `##DEVICE_TIMELINE` line per device to stdout, e.g.:

```
##DEVICE_TIMELINE impl="HIP / HIPCC" machine="<your machine label>" gpus=8 device=0 launch_ms=0.12 done_ms=42.87
##DEVICE_TIMELINE impl="HIP / HIPCC" machine="<your machine label>" gpus=8 device=1 launch_ms=0.31 done_ms=85.02
...
```

`launch_ms` is the host-side wall-clock offset (from the start of the timed region) at which that device's kernel was enqueued; `done_ms` is the offset at which that device's stream was observed to finish (via a non-blocking, round-robin `hipStreamQuery`/`cudaStreamQuery` poll, not a blind in-order synchronize, so waiting on device 0 can't mask overlap or serialization on the other devices). If `launch_ms` is close together across devices but `done_ms` comes back staggered by roughly one device's worth of compute time each (a "staircase"), that's direct evidence the devices are executing serially rather than concurrently, regardless of what the aggregate wall time / GFLOP/s numbers suggest.

Buffer allocation for each task's output (`d_out`) also happens before the timed region now (previously it was allocated per-task inside the loop, interleaved with device switches — allocator overhead scaling with `--gpus` even as each task's chunk of work shrinks could itself explain part of a "more GPUs look slower" result that has nothing to do with the kernel).

### Device timeline figures

```bash
make device-timelines
```

Renders the `##DEVICE_TIMELINE` data above as a static per-device Gantt-style
figure: one row per device, one bar per `[launch_ms, done_ms]` interval.
Overlapping bars across devices means concurrent execution; a staircase of
non-overlapping bars means the dispatch is effectively serialized. For each
`N` in `DEVICE_TIMELINE_SIZES` (default `2048 16384`) and each implementation
this host's `BACKENDS` supports, at `--gpus` up to `DEVICE_TIMELINE_GPUS`
(default 8, capped to however many devices this host actually has), this
writes a **fixed-filename** figure (no timestamp, so reruns overwrite in
place and the paths are stable to reference from the white paper):

```
plots/device_timeline_cuda-nvcc-N2048.pdf     / .png   (CUDA box)
plots/device_timeline_scale-nvidia-N2048.pdf  / .png   (CUDA box)
plots/device_timeline_hip-hipcc-N2048.pdf     / .png   (HIP box)
plots/device_timeline_scale-amd-N2048.pdf     / .png   (HIP box)
... and the same set for each other N in DEVICE_TIMELINE_SIZES
```

This is deliberately *not* a GPU-vendor profiler (e.g. `rocprofv3`, part of
ROCm's ROCprofiler-SDK) wrapping the binary — that was tried, and works
fine for `cwt-hip-hipcc`, but `cwt-cuda-scale-amd` cannot be traced this
way: SCALE bundles its own private ROCm HSA runtime, and whenever any
HSA-level profiler attaches to it, `libredscale.so`'s own device-detection
throws `No CUDA devices found` a few calls into enumerating the first (CPU)
agent, before it ever reaches a GPU. That's a bug in SCALE's runtime, not
something fixable from this repo. Since a profiling approach that only
covers one of the two implementations being compared isn't usable for an
apples-to-apples comparison, `##DEVICE_TIMELINE` — which lives in the
shared benchmark harness itself and therefore works identically for all
four implementations on either machine — is what backs the timeline
figures instead. See `scripts/device_timeline.sh` /
`plotting/plot_device_timeline.py`.

Under the hood this is `scripts/device_timeline.sh <binary> <args...>`,
which you can also run directly for one-off timelines of any binary/args
(raw stdout lands at `results/device-timeline-<label>.log`, where `<label>`
defaults to a timestamp unless you set `LABEL=` yourself).

### Inverse CWT (ICWT) round-trip demo

```bash
make icwt-demo
```

Runs each applicable binary once (single GPU, `--B 1`, small `N` -- default
`ICWT_N=256`, override with `make icwt-demo ICWT_N=512`) with `--icwt`, then
renders a fixed-filename 3-panel figure per implementation:

```
plots/icwt_demo_cuda-nvcc-N256.pdf     / .png   (CUDA box)
plots/icwt_demo_scale-nvidia-N256.pdf  / .png   (CUDA box)
plots/icwt_demo_hip-hipcc-N256.pdf     / .png   (HIP box)
plots/icwt_demo_scale-amd-N256.pdf     / .png   (HIP box)
```

Each figure shows: the 1D synthetic input signal, the 2D CWT coefficient
heatmap, and the 1D reconstructed signal overlaid on the original, with the
relative RMS reconstruction error in the title. This is a correctness/
consistency check -- forward CWT followed by a simple inverse-CWT should
give back (approximately) the same signal that went in -- not a
performance benchmark, which is why it runs at a small, fixed `N` rather
than the sweep sizes (the CWT coefficient dump this needs is `O(A*N)`,
i.e. `O(N^2)`, and gets large fast: don't point `ICWT_N` at the sweep's
larger sizes).

The reconstruction formula is the simple single-scale-sum form of the
classical Grossmann-Morlet CWT inverse: `f(t) ~= C . sum_a W(a,t) . a^-2 . da(a)`,
computed host-side in `run_icwt_demo()` (in `cwt_hip.cpp`/`cwt_cuda.cu`) from
the already-computed forward output -- it's `O(A*N)`, trivial next to the
`O(A*N*N)` forward transform being benchmarked, so there's no dedicated GPU
kernel for it. The scalar `C` is calibrated once via a least-squares fit
against the known synthetic input signal (the only signal this program ever
generates), the same spirit as `--verify` re-using a known CPU reference; a
production ICWT over an arbitrary, unknown signal would instead use a fixed
`C` derived analytically from the wavelet and scale grid alone. On this
synthetic test signal: ~0.7% relative RMS error at `N=128`, ~3.1% at `N=256`
(both well within visual/plotting resolution) -- versus ~18-84% if the
scale grid were the old narrow linear band instead of the current wide,
log-spaced one (see the scale-grid note in "Running" above).

## Sweeping

```bash
make sweep
```

Builds whatever this host supports (same `BACKENDS` filtering as plain `make`), then runs every applicable binary across GPU counts `1..N`, `REPS` times each (default 5), appending every run to `results/scaling.csv`. Device counts are auto-detected — `nvidia-smi -L | wc -l` for CUDA, `rocminfo | grep -c 'Device Type:.*GPU'` for HIP — not hardcoded, so it adapts to whatever's actually plugged into that box.

Override any of `N`, `B`, `REPS`, `CSV`, e.g.:

```bash
make sweep N=4096 REPS=10
```

`make clean-results` removes the results directory.

`make sweep` also auto-runs `make megaplot` at the end, using the `REMOTE_HOST` default set per host in `setup-backends.sh` — so a plain `make sweep` on either box both collects results and refreshes the combined plots against whatever the other box last swept. If the other box hasn't swept yet (or its results.csv isn't there), this step just warns and skips rather than failing the sweep; rerun `make megaplot REMOTE_HOST=<...>` by hand once both sides are done. Override with `make sweep REMOTE_HOST=<other-alias>` or unset it to skip.

### Sweeping across problem sizes

```bash
make sweep-sizes
```

Runs `make sweep` once per size in `SIZES` (default `2048 4096 8192 16384`), appending every size to the same CSV, then runs `megaplot` once at the end (not once per size). Useful for checking whether an effect (e.g. an implementation getting relatively worse with more GPUs) is a small-problem-size artifact — fixed per-device overhead (kernel launch, allocator, device-context switches) that matters less as each GPU's chunk of real work grows.

Forward CWT cost is `O(B*N^3)` (the number of scales `A` is set to `N`, and the kernel is `O(A*N*N)` per batch element), so doubling `N` is roughly an 8x increase in total compute — this ladder gets slow fast. `N=16384` alone can take tens of minutes on a multi-GPU box at `REPS=5`; go to `32768` (`make sweep-sizes SIZES="2048 4096 8192 16384 32768"`) only if you've got the time budget, and consider a lower `REPS` for it, e.g. `make sweep N=32768 REPS=2 SKIP_MEGAPLOT=1` run by hand.

## Plotting

`plotting/plot_four_platforms.py` takes a results CSV and produces a grouped bar chart comparing all four implementations (median with IQR error bars), one grouping per machine:

```bash
pip install matplotlib pandas
python3 plotting/plot_four_platforms.py results.csv --outdir plots --metric fwd_gflops
```

`--metric` can be `fwd_gflops`, `total_gflops`, `fwd_wall_s`, or `total_wall_s`. Output is written as both `.pdf` and `.png`, plus a `_summary.csv` with the aggregated median/quartile values.

Note: this script medians across every row it's given, so only feed it rows from a single `devices` (GPU count) at a time — see `make megaplot` below, which handles that split automatically.

`plotting/plot_scaling_lines.py` instead plots the full 1..8 GPU sweep as line charts — median line with an IQR ribbon, GPU count on the x-axis — one chart for GFLOP/s and one for wall time:

```bash
python3 plotting/plot_scaling_lines.py results/scaling-combined.csv --outdir plots
```

NVIDIA-targeting and AMD-targeting implementations keep their platform colour (NVIDIA green / AMD red) in both charts. Within each colour, native (`CUDA / NVCC`, `HIP / HIPCC`) is a solid line with a circle marker, and the corresponding SCALE build (`CUDA / SCALE→NVIDIA`, `CUDA / SCALE→AMD`) is a darker shade of the same colour, dashed, with a square marker — so native vs SCALE is easy to tell apart at a glance even across two different colour families, and native's circle is drawn on top where the two coincide. A single-GPU box's two series will just show as a single point at devices=1, which is expected. `--gflops-metric`/`--time-metric` pick which columns to plot (`fwd_*` by default, or `total_*`).

`plotting/plot_vs_problem_size.py` plots the opposite axis — GFLOP/s and wall time vs problem size `N` (log2 x-axis), for a results CSV spanning multiple sizes (i.e. from `make sweep-sizes`):

```bash
python3 plotting/plot_vs_problem_size.py results/scaling-combined.csv --outdir plots
```

Same colour/style scheme as the GPU-count line charts. Each series is plotted at its own largest swept GPU count (so implementations that only ever ran on a single-GPU box naturally show devices=1, while implementations run on a multi-GPU box show its full device count), which is the relevant comparison for checking whether a GPU-count effect is really a small-problem-size overhead artifact: if the gap between native and SCALE (or the "more GPUs is worse" slope) narrows or flattens as `N` grows, that's the signature of fixed per-device overhead losing significance relative to a growing amount of real compute per device. Only produced when the CSV has more than one distinct `N` (a single-size CSV is skipped, not an error).

### Combining results from multiple machines

If you're running this across more than one machine and they can `ssh` to each other with the repo at the same path on both, `make megaplot` pulls the other box's `results/scaling.csv` over `scp`, combines it with this box's own, and produces the per-GPU-count bar charts, the two scaling-vs-GPU-count line charts, and (if more than one problem size is present) the two scaling-vs-N line charts:

```bash
make megaplot REMOTE_HOST=<ssh alias for the other box>
```

Run the same command (with the appropriate alias) from either side. `REMOTE_HOST` should be whatever ssh alias reaches the other box; `setup-backends.sh` can set a per-host default so you don't have to pass it explicitly. Override `REMOTE_PATH` if the repo lives somewhere other than the default path on the remote, and `METRIC`/`PLOTS_DIR` the same way as above. Output is split by problem size first (`plots/N-<n>/...`, skipped if there's only one size), then by GPU count within each size (`plots/N-<n>/devices-<d>/...`, same skip rule); the vs-N charts land directly under `plots/`. Combined raw CSV is written to `results/scaling-combined.csv`.

`make megaplot` installs `pandas`/`matplotlib` itself the first time, into a local venv at `.venv-plots` (no system pip access needed, and it won't touch any other Python on the box). `make clean-plot-deps` removes that venv if you ever want it rebuilt.

You can also run this by hand for more control — `.venv-plots/bin/python3 plotting/megaplot.py <csv1> <csv2> ... --outdir plots --metric fwd_gflops`.
