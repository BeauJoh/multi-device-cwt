SHELL := /usr/bin/env bash

# ------------------------------------------------------------
# Source selection
# ------------------------------------------------------------

CWT_CUDA_SRC ?= cwt_cuda.cu
CWT_HIP_SRC  ?= cwt_hip.cpp

NVIDIA_NVCC ?= $(shell command -v nvcc 2>/dev/null)
CUDA_NVCC   ?= $(NVIDIA_NVCC)
CUDA_ARCH   ?= $(patsubst sm_%,%,$(CUDA_DEV_TARGET))
SCALE_ROOT  ?= $(CURDIR)/scale-1.7.2

HIP_HIPCC ?= hipcc
HIP_ARCH  ?= $(if $(HIP_DEV_TARGET),$(HIP_DEV_TARGET),gfx908)

SCALE_TARBALL_URL ?= https://pkgs.scale-lang.com/tar/scale-1.7.2-amd64.tar.xz

# ------------------------------------------------------------
# Sweep defaults
# ------------------------------------------------------------

N           ?= 2048
B           ?= 1
REPS        ?= 5
RESULTS_DIR ?= results
CSV         ?= $(RESULTS_DIR)/scaling.csv

.PHONY: all build sweep clean clean-results \
        cwt-cuda-nvcc cwt-hip-hipcc \
        cwt-cuda-scale-nvidia cwt-cuda-scale-amd \
        ensure-scale

# Plain `make` builds only what this host actually supports (per BACKENDS
# from setup-backends.sh). `make all` always forces every target, useful on
# a dev box like `zenith` that has both toolchains installed.
.DEFAULT_GOAL := build

all: cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd

# Build only the targets relevant to this host's BACKENDS (cuda and/or hip),
# as reported by setup-backends.sh. No need to `source` it first — this
# sources it itself.
build:
	@. ./setup-backends.sh; \
	targets=""; \
	if [[ "$$BACKENDS" == *"cuda"* ]]; then targets="$$targets cwt-cuda-nvcc cwt-cuda-scale-nvidia"; fi; \
	if [[ "$$BACKENDS" == *"hip"* ]]; then targets="$$targets cwt-hip-hipcc cwt-cuda-scale-amd"; fi; \
	if [ -z "$$targets" ]; then \
		echo "error: no supported backend found in BACKENDS='$$BACKENDS' (host: $$HOST)" >&2; \
		exit 1; \
	fi; \
	echo "==> $$HOST: BACKENDS=$$BACKENDS -> building:$$targets"; \
	$(MAKE) $$targets

# ------------------------------------------------------------
# Native builds
# ------------------------------------------------------------

# Native CUDA build via nvcc. Sources setup-backends.sh itself and resolves
# the arch from the shell env it produces (HIP_DEV_TARGET/CUDA_DEV_TARGET),
# rather than Make's own $(CUDA_ARCH) — that Make variable is only fresh
# when this recipe happens to run inside build's recursive $(MAKE) call; if
# you invoke this target directly without having sourced setup-backends.sh
# yourself first, Make's copy is stale/empty. Falls back to $(CUDA_ARCH) /
# $(CUDA_NVCC) if for some reason the shell env doesn't have it either.
cwt-cuda-nvcc: $(CWT_CUDA_SRC)
	. ./setup-backends.sh; \
	cuda_sm="sm_$${CUDA_DEV_TARGET#sm_}"; \
	[ "$$cuda_sm" != "sm_" ] || cuda_sm="sm_$(CUDA_ARCH)"; \
	nvcc -O3 -std=c++17 -arch="$$cuda_sm" -o $@ $<

# Native HIP build via hipcc. Same self-contained approach as cwt-cuda-nvcc.
cwt-hip-hipcc: $(CWT_HIP_SRC)
	. ./setup-backends.sh; \
	hip_arch="$${HIP_DEV_TARGET:-$(HIP_ARCH)}"; \
	$(HIP_HIPCC) -O3 -std=c++17 --offload-arch="$$hip_arch" -o $@ $<

# ------------------------------------------------------------
# Ensure SCALE is installed
# ------------------------------------------------------------

# Installs SCALE from the tarball distribution into $(SCALE_ROOT) if it
# isn't already there. Tarball is architecture-generic (the GPU target is
# chosen later via `scaleenv <arch>`), so this is identical on NVIDIA and
# AMD hosts.
ensure-scale:
	@if [ ! -x "$(SCALE_ROOT)/bin/scaleenv" ]; then \
		echo "==> SCALE not found at $(SCALE_ROOT)"; \
		echo "==> Downloading and installing SCALE from $(SCALE_TARBALL_URL)"; \
		tmpdir="$$(mktemp -d)"; \
		( wget -q -O "$$tmpdir/scale.tar.xz" "$(SCALE_TARBALL_URL)" \
		  && tar xf "$$tmpdir/scale.tar.xz" -C "$$tmpdir" ) || { rm -rf "$$tmpdir"; exit 1; }; \
		scaleenv_path="$$(find "$$tmpdir" -type f -path '*/bin/scaleenv' | head -n1)"; \
		if [ -z "$$scaleenv_path" ]; then \
			echo "error: could not find bin/scaleenv anywhere in the extracted tarball" >&2; \
			echo "==> extracted tarball contents (up to 3 levels deep):" >&2; \
			find "$$tmpdir" -maxdepth 3 >&2; \
			rm -rf "$$tmpdir"; exit 1; \
		fi; \
		extracted="$$(dirname "$$(dirname "$$scaleenv_path")")"; \
		chmod -R u+rwX,go+rX "$$extracted" 2>/dev/null || true; \
		find "$$extracted/bin" -maxdepth 1 -type f -exec chmod +x {} \; 2>/dev/null || true; \
		rm -rf "$(SCALE_ROOT)"; \
		mv "$$extracted" "$(SCALE_ROOT)"; \
		rm -rf "$$tmpdir"; \
		sudo usermod -a -G render,video "$$LOGNAME" || true; \
		echo "==> Installed SCALE to $(SCALE_ROOT)."; \
		echo "==> If this is the first install, log out/in (or reboot) so the render/video group membership applies."; \
	fi
	@if [ ! -x "$(SCALE_ROOT)/bin/scaleenv" ]; then \
		echo "error: missing SCALE: $(SCALE_ROOT)/bin/scaleenv" >&2; \
		echo "==> debug: contents of $(SCALE_ROOT)/bin (if it exists):" >&2; \
		ls -la "$(SCALE_ROOT)/bin" >&2 2>&1 || echo "  (no such directory)" >&2; \
		exit 1; \
	fi

# ------------------------------------------------------------
# SCALE builds (same CUDA source, compiled via SCALE's nvcc)
# ------------------------------------------------------------

# CUDA source built through SCALE, targeting NVIDIA hardware. Resolves arch
# from the shell env (see cwt-cuda-nvcc comment above for why), not Make's
# $(CUDA_ARCH). If SCALE_CUDA_PATH is set (setup-backends.sh sets it for
# hosts where the installed CUDA version is newer than SCALE's clang can
# parse, e.g. the H100 box: CUDA 13.3 vs SCALE 1.7.2's supported 13.1), it
# overrides CUDA_PATH just for this build, leaving the native cwt-cuda-nvcc
# build on the host's regular CUDA_PATH.
cwt-cuda-scale-nvidia: $(CWT_CUDA_SRC) | ensure-scale
	. ./setup-backends.sh; \
	cuda_sm="sm_$${CUDA_DEV_TARGET#sm_}"; \
	[ "$$cuda_sm" != "sm_" ] || cuda_sm="sm_$(CUDA_ARCH)"; \
	cuda_compute="$${cuda_sm#sm_}"; \
	if [ -n "$${SCALE_CUDA_PATH:-}" ]; then \
		echo "==> overriding CUDA_PATH for SCALE build: $$SCALE_CUDA_PATH"; \
		export CUDA_PATH="$$SCALE_CUDA_PATH"; \
	fi; \
	source "$(SCALE_ROOT)/bin/scaleenv" "$$cuda_sm" && \
	nvcc \
	  -gencode arch=compute_$$cuda_compute,code=$$cuda_sm \
	  $(NVCCFLAGS) \
	  $$CPPFLAGS \
	  -o $@ $< \
	  $$LDFLAGS \
	  $$LDLIBS

# Same CUDA source built through SCALE, targeting AMD hardware.
cwt-cuda-scale-amd: $(CWT_CUDA_SRC) | ensure-scale
	. ./setup-backends.sh; \
	hip_arch="$${HIP_DEV_TARGET:-$(HIP_ARCH)}"; \
	source "$(SCALE_ROOT)/bin/scaleenv" "$$hip_arch" && \
	nvcc -O3 -std=c++17 -o $@ $<

# ------------------------------------------------------------
# Sweep: build (BACKENDS-filtered), then run each applicable binary across
# GPU counts 1..N (N = however many devices this host actually has),
# REPS times each, appending every run to one CSV. Device counts are
# auto-detected per backend (`nvidia-smi -L` / `rocminfo`), not hardcoded,
# so this scales the sweep to whatever's actually plugged in.
# ------------------------------------------------------------
sweep: build
	@. ./setup-backends.sh; \
	mkdir -p "$(RESULTS_DIR)"; \
	ran_any=0; \
	sweep_hip_arch="$${HIP_DEV_TARGET:-gfx908}"; \
	sweep_cuda_sm="sm_$${CUDA_DEV_TARGET#sm_}"; \
	if [[ "$$BACKENDS" == *"cuda"* ]]; then \
		ndev="$$(nvidia-smi -L 2>/dev/null | wc -l)"; \
		if [ -z "$$ndev" ] || [ "$$ndev" -lt 1 ]; then \
			echo "error: BACKENDS contains cuda but nvidia-smi reports no devices" >&2; exit 1; \
		fi; \
		echo "==> $$HOST ($$MACHINE): sweeping CUDA GPUs 1..$$ndev, $(REPS) reps, N=$(N) B=$(B)"; \
		for gpus in $$(seq 1 "$$ndev"); do \
			for rep in $$(seq 1 $(REPS)); do \
				echo "--- CUDA/NVCC gpus=$$gpus rep=$$rep/$(REPS) ---"; \
				CWT_IMPL="CUDA / NVCC" CWT_BACKEND="CUDA" CWT_MACHINE="$$MACHINE" \
				  ./cwt-cuda-nvcc --mode explicit --N $(N) --B $(B) --gpus $$gpus --csv "$(CSV)" --forward-only; \
				echo "--- CUDA/SCALE-NVIDIA gpus=$$gpus rep=$$rep/$(REPS) ---"; \
				( source "$(SCALE_ROOT)/bin/scaleenv" "$$sweep_cuda_sm" && \
				  CWT_IMPL="CUDA / SCALE→NVIDIA" CWT_BACKEND="CUDA" CWT_MACHINE="$$MACHINE" \
				  ./cwt-cuda-scale-nvidia --mode explicit --N $(N) --B $(B) --gpus $$gpus --csv "$(CSV)" --forward-only ); \
			done; \
		done; \
		ran_any=1; \
	fi; \
	if [[ "$$BACKENDS" == *"hip"* ]]; then \
		ndev="$$(rocminfo 2>/dev/null | grep -c 'Device Type:.*GPU')"; \
		if [ -z "$$ndev" ] || [ "$$ndev" -lt 1 ]; then \
			echo "error: BACKENDS contains hip but rocminfo reports no GPU devices" >&2; exit 1; \
		fi; \
		echo "==> $$HOST ($$MACHINE): sweeping HIP GPUs 1..$$ndev, $(REPS) reps, N=$(N) B=$(B)"; \
		for gpus in $$(seq 1 "$$ndev"); do \
			for rep in $$(seq 1 $(REPS)); do \
				echo "--- HIP/HIPCC gpus=$$gpus rep=$$rep/$(REPS) ---"; \
				CWT_IMPL="HIP / HIPCC" CWT_BACKEND="HIP" CWT_MACHINE="$$MACHINE" \
				  ./cwt-hip-hipcc --mode explicit --N $(N) --B $(B) --gpus $$gpus --csv "$(CSV)" --forward-only; \
				echo "--- CUDA/SCALE-AMD gpus=$$gpus rep=$$rep/$(REPS) ---"; \
				( source "$(SCALE_ROOT)/bin/scaleenv" "$$sweep_hip_arch" && \
				  CWT_IMPL="CUDA / SCALE→AMD" CWT_BACKEND="HIP" CWT_MACHINE="$$MACHINE" \
				  ./cwt-cuda-scale-amd --mode explicit --N $(N) --B $(B) --gpus $$gpus --csv "$(CSV)" --forward-only ); \
			done; \
		done; \
		ran_any=1; \
	fi; \
	if [ "$$ran_any" -eq 0 ]; then \
		echo "error: no supported backend found in BACKENDS='$$BACKENDS' (host: $$HOST)" >&2; exit 1; \
	fi; \
	echo "==> results appended to $(CSV)"

# ------------------------------------------------------------
clean:
	rm -f cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd

clean-results:
	rm -rf "$(RESULTS_DIR)"
