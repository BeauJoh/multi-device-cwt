HOST ?= $(shell hostname -s)
TOP_LEVEL ?= $(CURDIR)

IMPL_DIR ?= $(TOP_LEVEL)/sycl-implementations/$(HOST)
IRIS_INSTALL_ROOT ?= $(IMPL_DIR)/iris
IRIS := $(IRIS_INSTALL_ROOT)

UNISYCL_INSTALL_ROOT ?= $(IMPL_DIR)/unisycl
UNISYCL := $(UNISYCL_INSTALL_ROOT)/bin/cscc

ADAPTIVECPP_INSTALL_ROOT ?= $(IMPL_DIR)/adaptivecpp
ADAPTIVECPP := $(ADAPTIVECPP_INSTALL_ROOT)/bin/acpp
ADAPTIVECPPCUDA_INSTALL_ROOT ?= $(IMPL_DIR)/adaptivecpp-cuda
ADAPTIVECPPCUDA := $(ADAPTIVECPPCUDA_INSTALL_ROOT)/bin/acpp
ADAPTIVECPPHIP_INSTALL_ROOT ?= $(IMPL_DIR)/adaptivecpp-hip
ADAPTIVECPPHIP := $(ADAPTIVECPPHIP_INSTALL_ROOT)/bin/acpp

DPCPPCPU_INSTALL_ROOT ?= $(IMPL_DIR)/dpc++-cpu
DPCPPCPU := $(DPCPPCPU_INSTALL_ROOT)/bin/clang++

DPCPPCUDA_INSTALL_ROOT ?= $(IMPL_DIR)/dpc++-cuda
DPCPPCUDA := $(DPCPPCUDA_INSTALL_ROOT)/bin/clang++
DPCPPHIP_INSTALL_ROOT ?= $(IMPL_DIR)/dpc++-hip
DPCPPHIP := $(DPCPPHIP_INSTALL_ROOT)/bin/clang++

TBB_INSTALL_ROOT ?= $(IMPL_DIR)/tbb

HIP_DEV_TARGET ?= gfx942
CUDA_DEV_TARGET ?= sm_70

LIBTORCH ?= $(shell python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')/../..
TORCH_LDFLAGS = -Wl,-rpath,$(LIBTORCH)/lib -L$(LIBTORCH)/lib
TORCH_CXXFLAGS = -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1
TORCH_LIBS = -ltorch -ltorch_cpu -lc10 -ltorch
TORCH_CPPFLAGS = -I$(LIBTORCH)/include -I$(LIBTORCH)/include/torch/csrc/api/include

CC ?= gcc
CXX ?= g++
FORTRAN ?= gfortran
NVCC ?= $(CUDA_PATH)/bin/nvcc
HIPCC ?= $(ROCM_PATH)/bin/hipcc

CFLAGS = -O3
CXXFLAGS = -O3
FFLAGS = -g -I$(IRIS)/include/iris
LDFLAGS = -L$(IRIS)/lib -liris -lpthread -ldl
