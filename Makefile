include ./makefile_defs.mk

SHELL := /usr/bin/env bash

# ------------------------------------------------------------
# Source selection
# ------------------------------------------------------------

CWT_CUDA_SRC ?= cwt_cuda.cu
CWT_HIP_SRC  ?= cwt_hip.cpp

NVIDIA_NVCC ?= $(shell command -v nvcc 2>/dev/null)
CUDA_NVCC   ?= $(NVIDIA_NVCC)
CUDA_ARCH   ?= $(patsubst sm_%,%,$(CUDA_DEV_TARGET))
SCALE_ROOT  ?= $(CURDIR)/scale-1.7.0-Linux

HIP_HIPCC ?= hipcc
HIP_ARCH  ?= gfx908

.PHONY: all clean \
        cwt-cuda-nvcc cwt-hip-hipcc \
        cwt-cuda-scale-nvidia cwt-cuda-scale-amd

all: cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd

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
# SCALE builds (same CUDA source, compiled via SCALE's nvcc)
# ------------------------------------------------------------

# CUDA source built through SCALE, targeting NVIDIA hardware.
cwt-cuda-scale-nvidia: $(CWT_CUDA_SRC)
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
cwt-cuda-scale-amd: $(CWT_CUDA_SRC)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc -O3 -std=c++17 -o $@ $<

# ------------------------------------------------------------
clean:
	rm -f cwt-cuda-nvcc cwt-hip-hipcc cwt-cuda-scale-nvidia cwt-cuda-scale-amd
