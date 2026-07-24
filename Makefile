SHELL := /usr/bin/env bash

# ------------------------------------------------------------
# Source selection
# ------------------------------------------------------------

CWT_CUDA_SRC ?= cwt_cuda.cu
CWT_HIP_SRC  ?= cwt_hip.cpp

NVIDIA_NVCC ?= $(shell command -v nvcc 2>/dev/null)
CUDA_NVCC   ?= $(NVIDIA_NVCC)
CUDA_ARCH   ?= $(patsubst sm_%,%,$(CUDA_DEV_TARGET))
SCALE_ROOT  ?= $(CURDIR)/scale-1.7.3-Linux

HIP_HIPCC ?= hipcc
HIP_ARCH  ?= gfx908

SCALE_TARBALL_URL ?= https://pkgs.scale-lang.com/tar/scale-latest-amd64.tar.xz

.PHONY: all build clean \
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

# Native CUDA build via nvcc.
cwt-cuda-nvcc: $(CWT_CUDA_SRC)
	$(CUDA_NVCC) -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -o $@ $<

# Native HIP build via hipcc.
cwt-hip-hipcc: $(CWT_HIP_SRC)
	$(HIP_HIPCC) -O3 -std=c++17 --offload-arch=$(HIP_ARCH) -o $@ $<

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
		extracted="$$(find "$$tmpdir" -maxdepth 1 -mindepth 1 -type d -name 'scale-*' | head -n1)"; \
		if [ -z "$$extracted" ]; then \
			echo "error: could not find an extracted 'scale-*' directory in the tarball" >&2; \
			rm -rf "$$tmpdir"; exit 1; \
		fi; \
		rm -rf "$(SCALE_ROOT)"; \
		mv "$$extracted" "$(SCALE_ROOT)"; \
		rm -rf "$$tmpdir"; \
		sudo usermod -a -G render,video "$$LOGNAME" || true; \
		echo "==> Installed SCALE to $(SCALE_ROOT)."; \
		echo "==> If this is the first install, log out/in (or reboot) so the render/video group membership applies."; \
	fi
	@test -x "$(SCALE_ROOT)/bin/scaleenv" || { echo "error: missing SCALE: $(SCALE_ROOT)/bin/scaleenv" >&2; exit 1; }

# ------------------------------------------------------------
# SCALE builds (same CUDA source, compiled via SCALE's nvcc)
# ------------------------------------------------------------

# CUDA source built through SCALE, targeting NVIDIA hardware.
cwt-cuda-scale-nvidia: $(CWT_CUDA_SRC) | ensure-scale
	. ./setup-backends.sh && \
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	nvcc \
	  -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	  $(NVCCFLAGS) \
	  $$CPPFLAGS \
	  -o $@ $< \
	  $$LDFLAGS \
	  $$LDLIBS

# Same CUDA source built through SCALE, targeting AMD hardware.
cwt-cuda-scale-amd: $(CWT_CUDA_SRC) | ensure-scale
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc -O3 -std=c++17 -o $@ $<

# ------------------------------------------------------------
clean:
	rm -f cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd
