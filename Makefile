include ./makefile_defs.mk

SHELL := /usr/bin/env bash
PIXI_RUN := ./setup-pixi.sh run

RUN_ENV ?=
EXTRA   ?=
RUN_PYTORCH ?= 0

# ------------------------------------------------------------
# Source selection
# ------------------------------------------------------------

CWT_SYCL_SRC         ?= cwt.cpp
CWT_BUFFER_SRC       ?= cwt-buffer.cpp
CWT_BUFFER_SEQ_SRC   ?= cwt-buffer-seq.cpp
CWT_HOSTBACKED_SRC   ?= cwt-buffer-hostbacked.cpp
CWT_PYTORCH_SRC      ?= cwt.py
CWT_TRITON_SRC       ?= cwt_triton.py
CWT_BLAS_SRC         ?= cwt_blas.py
CWT_DBG_SRC          ?= dbg_cwt.cpp
DBG_REF              ?= acpp
DBG_CWT_REF_DUMP     ?= /tmp/cwt_ref.bin
DBG_CWT_USY_DUMP     ?= /tmp/cwt_usy.bin
DBG_CSV              ?= /tmp/dbg_cwt.csv

#SCALE sources:
CWT_CUDA_SRC         ?= cwt_cuda.cu
NVIDIA_NVCC          ?= $(shell command -v nvcc 2>/dev/null)
CUDA_NVCC            ?= $(NVIDIA_NVCC)
CUDA_ARCH            ?= $(patsubst sm_%,%,$(CUDA_DEV_TARGET))
SCALE_ROOT           ?= $(CURDIR)/scale-1.7.0-Linux
CWT_HIP_SRC          ?= cwt_hip.cpp
HIP_HIPCC            ?= hipcc
HIP_ARCH             ?= gfx908

# ------------------------------------------------------------
# Experiment defaults
# ------------------------------------------------------------

N      ?= 2048
B      ?= 1
GPUS   ?= 1
SHARDS  ?= 1
CSV    ?= results.csv
REPS   ?= 5

EXP_BACKEND ?= cuda,hip
EXP_POLICY  ?= roundrobin
CHUNK_LIST    ?= 1 2 3 4 5 6 7:8 9 10

RESULTS_DIR      ?= results
PLOTS_DIR        ?= plots
ARCHIVE_BASENAME ?= plots
ARCHIVE_FILE     ?= $(ARCHIVE_BASENAME).tar.gz

STRONG_SCALE_CSV ?= $(RESULTS_DIR)/strong_scaling.csv
WEAK_SCALE_CSV   ?= $(RESULTS_DIR)/weak_scaling.csv

# Format: "gpu:N gpu:N ..."
# These defaults are cube-root weak scaling for CWT ~ O(N^3).
#WEAK_CASES ?= 1:1024 2:1290 4:1625 7:1966
WEAK_CASES ?= 1:1024 2:1280 4:1792 7:2048

PLOT_PHASE         ?= forward
PLOT_SCRIPT        ?= plotting/plot_strong_scaling.py
WEAK_PLOT_SCRIPT   ?= plotting/plot_weak_scaling.py
PORTABILITY_SCRIPT ?= plotting/plot_performance_portability.py
PORTABILITY_CSV    ?= $(STRONG_SCALE_CSV)

CWT_FORWARD_FLAG ?= --forward-only
COMMON_LD_PATH := $(IRIS_INSTALL_ROOT)/lib:$(LD_LIBRARY_PATH)
IRIS_LINK_FLAGS := -L$(IRIS_INSTALL_ROOT)/lib -liris -lpthread -ldl
IRIS_BUILD_ENV := LD_LIBRARY_PATH="$(COMMON_LD_PATH)" LIBRARY_PATH="$(IRIS_INSTALL_ROOT)/lib:$${LIBRARY_PATH:-}" CPATH="$(IRIS_INSTALL_ROOT)/include:$${CPATH:-}"

.PHONY: all debug paper-default run-all plot-all archive-all clean help \
        ensure-unisycl ensure-acpp ensure-dpcpp-cpu ensure-dpcpp-cuda ensure-dpcpp-hip cwt-dbg-usy \
        cwt-usy-buffer cwt-usy-buffer-seq cwt-usy-buffer-hostbacked \
        cwt-profile-usy cwt-profile-acpp cwt-profile-dpcpp-cpu cwt-profile-dpcpp-hip cwt-profile-dpcpp-cuda \
        cwt-dbg-acpp cwt-dbg-dpcpp-cuda cwt-dbg-dpcpp-hip \
        run-cwt-cuda run-cwt-cuda-single \
        cwt-hip-hipcc cwt-cuda-scale-amd \
        run-cwt-hip-hipcc run-cwt-hip-hipcc-single \
        run-cwt-cuda-scale-amd run-cwt-cuda-scale-amd-single \
        run-cwt-acpp-single run-cwt-dpcpp-cuda-single run-cwt-dpcpp-hip-single\
        run-cwt-usy run-cwt-acpp run-cwt-dpcpp-cuda run-cwt-dpcpp-hip \
        run-cwt-pytorch run-cwt-triton run-cwt-blas \
        run-dbg-cwt-usy run-dbg-cwt-acpp run-dbg-cwt-dpcpp-cuda run-dbg-cwt-dpcpp-hip \
        prepare-results strong-scale-build strong-scale-build-dpcpp \
        strong-scale-run strong-scale-run-dpcpp strong-scale-run-pytorch strong-scale-default strong-scale-clean \
        weak-scale-build weak-scale-run weak-scale-default weak-scale-clean \
        strong-scale-plot weak-scale-plot plot-pp plot-pp-paper \
        dbg-forward-compare dbg-inverse-usy paper-oversubscription plot-over plot-over-paper


all: paper-default

paper-default: run-all plot-all archive-all

run-all: strong-scale-run weak-scale-run

plot-all: strong-scale-plot weak-scale-plot portability-plot

archive-all:
	tar -czf "$(ARCHIVE_FILE)" "$(RESULTS_DIR)" "$(PLOTS_DIR)"

help:
	@echo "Main workflows:"
	@echo "  make                         Build, run strong+weak, plot, archive"
	@echo "  make strong-scale-default     Run strong scaling and plot"
	@echo "  make weak-scale-default       Run weak scaling and plot"
	@echo "  make plot-all                 Plot existing results only"
	@echo ""
	@echo "Key variables:"
	@echo "  EXP_BACKEND=$(EXP_BACKEND)"
	@echo "  EXP_POLICY=$(EXP_POLICY)"
	@echo "  N=$(N)"
	@echo "  B=$(B)"
	@echo "  REPS=$(REPS)"
	@echo "  CHUNK_LIST=$(CHUNK_LIST)"
	@echo "  WEAK_CASES=$(WEAK_CASES)"
	@echo "  RESULTS_DIR=$(RESULTS_DIR)"
	@echo "  PLOTS_DIR=$(PLOTS_DIR)"

clean:
	rm -f \
	  cwt-usy cwt-dbg-usy \
	  cwt-usy-buffer cwt-usy-buffer-seq cwt-usy-buffer-hostbacked \
	  cwt-acpp cwt-dpcpp-cuda cwt-dpcpp-hip \
	  cwt-profile-usy cwt-profile-acpp \
	  cwt-profile-dpcpp-cpu cwt-profile-dpcpp-hip cwt-profile-dpcpp-cuda \
	  cwt-dbg-acpp cwt-dbg-dpcpp-cuda cwt-dbg-dpcpp-hip \
		cwt-cuda-nvcc cwt-cuda-scale-nvidia cwt-hip-hipcc cwt-cuda-scale-amd
	rm -f __pycache__ \
	  .pytest_cache

clean-results:
	rm -rf \
	  $(RESULTS_DIR) \
	  $(PLOTS_DIR) \
	rm -f "$(ARCHIVE_FILE)"

# ------------------------------------------------------------
# Ensure installed toolchains
# ------------------------------------------------------------

ensure-unisycl:
	@test -x "$(UNISYCL)" || { \
		echo "==> UniSYCL compiler not found at $(UNISYCL)"; \
		echo "==> Running ./install.sh --unisycl-only"; \
		./install.sh --unisycl-only; \
	}
	@test -x "$(UNISYCL)" || { echo "error: missing UniSYCL: $(UNISYCL)" >&2; exit 1; }

ensure-acpp:
	@test -x "$(ADAPTIVECPP)" || { \
		echo "==> AdaptiveCpp compiler not found at $(ADAPTIVECPP)"; \
		echo "==> Running ./install.sh --acpp-only"; \
		./install.sh --acpp-only; \
	}
	@test -x "$(ADAPTIVECPP)" || { echo "error: missing AdaptiveCpp: $(ADAPTIVECPP)" >&2; exit 1; }

ensure-dpcpp-cpu:
	@test -x "$(DPCPPCPU)" || { \
		echo "==> DPC++ CPU compiler not found at $(DPCPPCPU)"; \
		echo "==> Running ./install.sh --dpcpp-only"; \
		./install.sh --dpcpp-only; \
	}
	@test -x "$(DPCPPCPU)" || { echo "error: missing DPC++ CPU: $(DPCPPCPU)" >&2; exit 1; }

ensure-dpcpp-cuda:
	@test -x "$(DPCPPCUDA)" || { \
		echo "==> DPC++ CUDA compiler not found at $(DPCPPCUDA)"; \
		echo "==> Running ./install.sh --dpcpp-only"; \
		./install.sh --dpcpp-only; \
	}
	@test -x "$(DPCPPCUDA)" || { echo "error: missing DPC++ CUDA: $(DPCPPCUDA)" >&2; exit 1; }

ensure-dpcpp-hip:
	@test -x "$(DPCPPHIP)" || { \
		echo "==> DPC++ HIP compiler not found at $(DPCPPHIP)"; \
		echo "==> Running ./install.sh --dpcpp-only"; \
		./install.sh --dpcpp-only; \
	}
	@test -x "$(DPCPPHIP)" || { echo "error: missing DPC++ HIP: $(DPCPPHIP)" >&2; exit 1; }

# ------------------------------------------------------------
# Build targets
# ------------------------------------------------------------

cwt-usy: $(CWT_SYCL_SRC) | ensure-unisycl
	. ./setup-backends.sh && \
	GXX_STDLIB_DIR="$$(dirname "$$(g++ -print-file-name=libstdc++.so)")" && \
	$(IRIS_BUILD_ENV) \
	$(UNISYCL) $(CXXFLAGS) -std=c++17 -o $@ $< \
	  -L$$GXX_STDLIB_DIR $(IRIS_LINK_FLAGS) -lstdc++

cwt-dbg-usy: $(CWT_DBG_SRC) | ensure-unisycl
	. ./setup-backends.sh && \
	GXX_STDLIB_DIR="$$(dirname "$$(g++ -print-file-name=libstdc++.so)")" && \
	$(IRIS_BUILD_ENV) \
	$(UNISYCL) -O0 -g -std=c++17 -o $@ $< \
	  -L$$GXX_STDLIB_DIR $(IRIS_LINK_FLAGS) -lstdc++

cwt-usy-buffer: $(CWT_BUFFER_SRC) | ensure-unisycl
	. ./setup-backends.sh && \
	GXX_STDLIB_DIR="$$(dirname "$$(g++ -print-file-name=libstdc++.so)")" && \
	$(IRIS_BUILD_ENV) \
	$(UNISYCL) $(CXXFLAGS) -std=c++17 -o $@ $< \
	  -L$$GXX_STDLIB_DIR $(IRIS_LINK_FLAGS) -lstdc++

cwt-usy-buffer-seq: $(CWT_BUFFER_SEQ_SRC) | ensure-unisycl
	. ./setup-backends.sh && \
	GXX_STDLIB_DIR="$$(dirname "$$(g++ -print-file-name=libstdc++.so)")" && \
	$(IRIS_BUILD_ENV) \
	$(UNISYCL) $(CXXFLAGS) -std=c++17 -o $@ $< \
	  -L$$GXX_STDLIB_DIR $(IRIS_LINK_FLAGS) -lstdc++

cwt-usy-buffer-hostbacked: $(CWT_HOSTBACKED_SRC) | ensure-unisycl
	. ./setup-backends.sh && \
	GXX_STDLIB_DIR="$$(dirname "$$(g++ -print-file-name=libstdc++.so)")" && \
	$(IRIS_BUILD_ENV) \
	$(UNISYCL) $(CXXFLAGS) -std=c++17 -o $@ $< \
	  -L$$GXX_STDLIB_DIR $(IRIS_LINK_FLAGS) -lstdc++

cwt-acpp: $(CWT_SYCL_SRC) | ensure-acpp
	. ./setup-backends.sh && \
	$(ADAPTIVECPP) $(CXXFLAGS) -std=c++17 --acpp-targets="generic" \
	  -ldl -fPIC -o $@ $<

cwt-dpcpp-cuda: $(CWT_SYCL_SRC) | ensure-dpcpp-cuda
	$(DPCPPCUDA) $(CXXFLAGS) -O3 -std=c++17 -fsycl \
	  -fsycl-targets=nvptx64-nvidia-cuda \
	  -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=$(CUDA_DEV_TARGET) \
	  -Wl,-rpath=$(DPCPPCUDA_INSTALL_ROOT)/lib -L$(DPCPPCUDA_INSTALL_ROOT)/lib -lsycl \
	  -ldl -fPIC -o $@ $<

cwt-dpcpp-hip: $(CWT_SYCL_SRC) | ensure-dpcpp-hip
	$(DPCPPHIP) $(CXXFLAGS) -O3 -std=c++17 -fsycl \
	  -fsycl-targets=amdgcn-amd-amdhsa \
	  -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$(HIP_DEV_TARGET) \
	  -Wl,-rpath=$(DPCPPHIP_INSTALL_ROOT)/lib -L$(DPCPPHIP_INSTALL_ROOT)/lib -lsycl \
	  -ldl -fPIC -o $@ $<

cwt-dbg-acpp: $(CWT_DBG_SRC) | ensure-acpp
	. ./setup-backends.sh && \
	$(ADAPTIVECPP) -O0 -g -std=c++17 --acpp-targets="generic" \
	  -ldl -fPIC -o $@ $<

cwt-dbg-dpcpp-cuda: $(CWT_DBG_SRC) | ensure-dpcpp-cuda
	$(DPCPPCUDA) -O0 -g -std=c++17 -fsycl \
	  -fsycl-targets=nvptx64-nvidia-cuda \
	  -Xsycl-target-backend=nvptx64-nvidia-cuda --cuda-gpu-arch=$(CUDA_DEV_TARGET) \
	  -Wl,-rpath=$(DPCPPCUDA_INSTALL_ROOT)/lib -L$(DPCPPCUDA_INSTALL_ROOT)/lib -lsycl \
	  -ldl -fPIC -o $@ $<

cwt-dbg-dpcpp-hip: $(CWT_DBG_SRC) | ensure-dpcpp-hip
	$(DPCPPHIP) -O0 -g -std=c++17 -fsycl \
	  -fsycl-targets=amdgcn-amd-amdhsa \
	  -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$(HIP_DEV_TARGET) \
	  -Wl,-rpath=$(DPCPPHIP_INSTALL_ROOT)/lib -L$(DPCPPHIP_INSTALL_ROOT)/lib -lsycl \
	  -ldl -fPIC -o $@ $<

cwt-cuda-nvcc: $(CWT_CUDA_SRC)
	$(CUDA_NVCC) -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -o $@ $<

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

cwt-hip-hipcc: $(CWT_HIP_SRC)
	$(HIP_HIPCC) -O3 -std=c++17 --offload-arch=$(HIP_ARCH) -o $@ $<

cwt-cuda-scale-amd: $(CWT_CUDA_SRC)
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	nvcc -O3 -std=c++17 -o $@ $<

# ------------------------------------------------------------
# Run targets
# ------------------------------------------------------------
debug: cwt-usy
	IRIS_ARCHS=hip,cuda \
	IRIS_POLICY=roundrobin \
	IRIS_PROFILE_PATH=zenith_trace.csv \
	IRIS_ASYNC=1 \
	CWT_IMPL="UniSYCL (Trace)" \
	CWT_MACHINE="$(hostname -s)" \
	CWT_BACKEND="IRIS" \
	./cwt-usy --mode unisycl --N 1024 --B 1 --gpus 4 --shards 1 --csv /tmp/usy_trace.csv --forward-only \
  2>&1 | tee /tmp/iris-cwt-trace.log

run-cwt-acpp-single: cwt-acpp
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	$(RUN_ENV) ./cwt-acpp --mode single --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-dpcpp-cuda-single: cwt-dpcpp-cuda
	LD_LIBRARY_PATH="$(COMMON_LD_PATH):$(DPCPPCUDA_INSTALL_ROOT)/lib:$(TBB_INSTALL_ROOT)/lib" \
	ONEAPI_DEVICE_SELECTOR="cuda:*" \
	SYCL_DEVICE_FILTER="cuda:*" \
	$(RUN_ENV) ./cwt-dpcpp-cuda --mode single --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-dpcpp-hip-single: cwt-dpcpp-hip
	LD_LIBRARY_PATH="$(COMMON_LD_PATH):$(DPCPPHIP_INSTALL_ROOT)/lib:$(TBB_INSTALL_ROOT)/lib" \
	ONEAPI_DEVICE_SELECTOR="hip:*" \
	SYCL_DEVICE_FILTER="hip:*" \
	$(RUN_ENV) ./cwt-dpcpp-hip --mode single --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-usy: cwt-usy
	IRIS_POLICY=$(EXP_POLICY) \
	IRIS_ASYNC=1 \
	IRIS_ARCHS=$(EXP_BACKEND) \
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	$(RUN_ENV) ./cwt-usy --mode unisycl --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

profile-cwt-usy: cwt-usy
	@set -u; \
	export IRIS_POLICY="$(EXP_POLICY)"; \
	export IRIS_ASYNC=1; \
	export IRIS_ARCHS="$(EXP_BACKEND)"; \
	export CORNEA_PROFILE_OUT="$(abspath $(EXP_CORNEA_PROFILE))"; \
	export LD_LIBRARY_PATH="$(COMMON_LD_PATH)"; \
	export IRIS_PROFILE=1; \
	rm -f "$$CORNEA_PROFILE_OUT"; \
	echo "Starting Cornea..."; \
	echo "Using cornea-profile: ${IRIS_INSTALL_ROOT}/bin/cornea-profile"; \
	"${IRIS_INSTALL_ROOT}/bin/cornea-profile"  \
	./cwt-usy --mode unisycl \
		--N $(N) \
		--B $(B) \
		--gpus $(GPUS) \
		--shards $(SHARDS) \
		--csv $(CSV) \
		$(EXTRA); \
	ls -lh "$$CORNEA_PROFILE_OUT" 2>/dev/null || true;

run-cwt-acpp: cwt-acpp
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	$(RUN_ENV) ./cwt-acpp --mode conventional --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-dpcpp-cuda: cwt-dpcpp-cuda
	LD_LIBRARY_PATH="$(COMMON_LD_PATH):$(DPCPPCUDA_INSTALL_ROOT)/lib:$(TBB_INSTALL_ROOT)/lib" \
	ONEAPI_DEVICE_SELECTOR="cuda:*" \
	SYCL_DEVICE_FILTER="cuda:*" \
	$(RUN_ENV) ./cwt-dpcpp-cuda --mode conventional --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-dpcpp-hip: cwt-dpcpp-hip
	LD_LIBRARY_PATH="$(COMMON_LD_PATH):$(DPCPPHIP_INSTALL_ROOT)/lib:$(TBB_INSTALL_ROOT)/lib" \
	ONEAPI_DEVICE_SELECTOR="hip:*" \
	SYCL_DEVICE_FILTER="hip:*" \
	$(RUN_ENV) ./cwt-dpcpp-hip --mode conventional --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-pytorch: $(CWT_PYTORCH_SRC)
	$(PIXI_RUN) python3 ./$(CWT_PYTORCH_SRC) --samples $(N) --batch $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-triton: $(CWT_TRITON_SRC)
	$(PIXI_RUN) python3 ./$(CWT_TRITON_SRC) --samples $(N) --batch $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-blas: $(CWT_BLAS_SRC)
	$(PIXI_RUN) python3 ./$(CWT_BLAS_SRC) --samples $(N) --batch $(B) --gpus $(GPUS) --shards $(SHARDS) --block_a 16 --block_n 512 --csv $(CSV) $(EXTRA)

run-cwt-cuda-nvcc: cwt-cuda-nvcc
	$(RUN_ENV) CWT_IMPL="CUDA / NVCC" CWT_BACKEND="CUDA" ./cwt-cuda-nvcc \
		--mode explicit --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-cuda-nvcc-single: cwt-cuda-nvcc
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) CWT_IMPL="CUDA / NVCC (Single Stream)" CWT_BACKEND="CUDA" ./cwt-cuda-nvcc \
		--mode single --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-cuda-scale-nvidia: cwt-cuda-scale-nvidia
	source "$(SCALE_ROOT)/bin/scaleenv" sm_$(CUDA_ARCH) && \
	$(RUN_ENV) CWT_IMPL="CUDA / SCALE→NVIDIA" CWT_BACKEND="CUDA" ./cwt-cuda-scale-nvidia \
		--mode explicit --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-hip-hipcc: cwt-hip-hipcc
	$(RUN_ENV) CWT_IMPL="HIP / HIPCC" CWT_BACKEND="HIP" ./cwt-hip-hipcc \
		--mode explicit --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-hip-hipcc-single: cwt-hip-hipcc
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) CWT_IMPL="HIP / HIPCC (Single Stream)" CWT_BACKEND="HIP" ./cwt-hip-hipcc \
		--mode single --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-cuda-scale-amd: cwt-cuda-scale-amd
	source "$(SCALE_ROOT)/bin/scaleenv" $(HIP_ARCH) && \
	$(RUN_ENV) CWT_IMPL="CUDA / SCALE→AMD" CWT_BACKEND="HIP" ./cwt-cuda-scale-amd \
		--mode explicit --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

run-cwt-cuda-scale-amd-single: cwt-cuda-scale-amd
	$(RUN_ENV) CWT_IMPL="CUDA / SCALE→AMD (Single Stream)" CWT_BACKEND="HIP" ./cwt-cuda-scale-amd \
		--mode single --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(CSV) $(EXTRA)

# ------------------------------------------------------------
# Build helpers
# ------------------------------------------------------------

prepare-results:
	@mkdir -p "$(RESULTS_DIR)"

strong-scale-build-dpcpp:
	@set -e; \
	. ./setup-backends.sh >/dev/null 2>&1; \
	backend="$(EXP_BACKEND)"; \
	if [ "$$backend" = "auto" ]; then \
		if [[ "$$BACKENDS" == *"hip"* ]]; then backend="hip"; \
		elif [[ "$$BACKENDS" == *"cuda"* ]]; then backend="cuda"; \
		else echo "No supported accelerator backend found in BACKENDS=$$BACKENDS" >&2; exit 1; fi; \
	fi; \
	if [ "$$backend" = "hip" ]; then \
		$(MAKE) cwt-dpcpp-hip; \
	elif [ "$$backend" = "cuda" ]; then \
		$(MAKE) cwt-dpcpp-cuda; \
	else \
		echo "Unsupported EXP_BACKEND=$$backend" >&2; exit 1; \
	fi

strong-scale-build: prepare-results
	@echo "Building CWT binaries..."
	@$(MAKE) cwt-usy
	@$(MAKE) cwt-acpp
	@$(MAKE) cwt-cuda-nvcc
	@$(MAKE) strong-scale-build-dpcpp

weak-scale-build: strong-scale-build

# ------------------------------------------------------------
# Strong scaling
# ------------------------------------------------------------

strong-scale-run: prepare-results strong-scale-build
	@rm -f "$(STRONG_SCALE_CSV)"
	@echo "Running strong-scaling comparison..."
	@set -e; \
	. ./setup-backends.sh; \
	backend="$(EXP_BACKEND)"; \
	if [ "$$backend" = "auto" ]; then \
		if [[ "$$BACKENDS" == *"hip"* ]]; then backend="hip"; backend_label="HIP"; \
		elif [[ "$$BACKENDS" == *"cuda"* ]]; then backend="cuda"; backend_label="CUDA"; \
		else echo "No supported accelerator backend found in BACKENDS=$$BACKENDS" >&2; exit 1; fi; \
	else \
		backend_label="$$(printf "%s" "$$backend" | tr "[:lower:]" "[:upper:]")"; \
	fi; \
	for chunks in $(CHUNK_LIST); do \
		for rep in $$(seq 1 $(REPS)); do \
			echo "============================================="; \
			echo "Strong scaling: GPUs=$$chunks rep=$$rep/$(REPS)"; \
			echo "============================================="; \
			$(MAKE) run-cwt-usy \
				IRIS_POLICY=roundrobin \
				EXP_BACKEND="$$backend" \
				N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="UniSYCL (Single Queue)"; \
			$(MAKE) run-cwt-acpp \
				N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="ACPP (Explicit Queues)"; \
			$(MAKE) run-cwt-acpp-single \
				EXP_BACKEND="$$backend" \
				N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="ACPP (Single Queue)"; \
			done; \
	done
	@$(MAKE) strong-scale-run-dpcpp \
		N="$(N)" B="$(B)" REPS="$(REPS)" CHUNK_LIST="$(CHUNK_LIST)" STRONG_SCALE_CSV="$(STRONG_SCALE_CSV)" EXP_BACKEND="$(EXP_BACKEND)"
	ifeq ($(RUN_PYTORCH),1)
		@$(MAKE) strong-scale-run-pytorch \
			N="$(N)" B="$(B)" REPS="$(REPS)" CHUNK_LIST="$(CHUNK_LIST)" STRONG_SCALE_CSV="$(STRONG_SCALE_CSV)" EXP_BACKEND="$(EXP_BACKEND)"
	endif

strong-scale-run-dpcpp:
	@set -e; \
	. ./setup-backends.sh; \
	backend="$(EXP_BACKEND)"; \
	if [ "$$backend" = "auto" ]; then \
		if [[ "$$BACKENDS" == *"hip"* ]]; then backend="hip"; backend_label="HIP"; \
		elif [[ "$$BACKENDS" == *"cuda"* ]]; then backend="cuda"; backend_label="CUDA"; \
		else echo "No supported accelerator backend found in BACKENDS=$$BACKENDS" >&2; exit 1; fi; \
	else \
		backend_label="$$(printf "%s" "$$backend" | tr "[:lower:]" "[:upper:]")"; \
	fi; \
	for chunks in $(CHUNK_LIST); do \
		for rep in $$(seq 1 $(REPS)); do \
			echo "DPC++ strong scaling: GPUs=$$chunks rep=$$rep/$(REPS)"; \
			if [ "$$backend" = "hip" ]; then \
				$(MAKE) run-cwt-dpcpp-hip \
					N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Explicit Queues)"; \
				$(MAKE) run-cwt-dpcpp-hip-single \
					N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Single Queue)"; \
			else \
				$(MAKE) run-cwt-dpcpp-cuda \
					N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Explicit Queues)"; \
				$(MAKE) run-cwt-dpcpp-cuda-single \
					N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Single Queue)"; \
			fi; \
		done; \
	done

strong-scale-run-pytorch:
	@set -e; \
	. ./setup-backends.sh; \
	backend="$(EXP_BACKEND)"; \
	if [ "$$backend" = "auto" ]; then \
		if [[ "$$BACKENDS" == *"hip"* ]]; then backend_label="HIP"; \
		elif [[ "$$BACKENDS" == *"cuda"* ]]; then backend_label="CUDA"; \
		else echo "No supported accelerator backend found in BACKENDS=$$BACKENDS" >&2; exit 1; fi; \
	else \
		backend_label="$$(printf "%s" "$$backend" | tr "[:lower:]" "[:upper:]")"; \
	fi; \
	for chunks in $(CHUNK_LIST); do \
		for rep in $$(seq 1 $(REPS)); do \
			echo "PyTorch strong scaling: GPUs=$$chunks rep=$$rep/$(REPS)"; \
			$(MAKE) run-cwt-pytorch \
				N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="PyTorch"; \
			$(MAKE) run-cwt-triton \
				N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="PyTorch+Triton"; \
			$(MAKE) run-cwt-blas \
				N="$(N)" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(STRONG_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="PyTorch+BLAS"; \
		done; \
	done

# ------------------------------------------------------------
# Weak scaling
# ------------------------------------------------------------

weak-scale-run: prepare-results weak-scale-build
	@rm -f "$(WEAK_SCALE_CSV)"
	@echo "Running weak-scaling comparison..."
	@set -e; \
	. ./setup-backends.sh; \
	backend="$(EXP_BACKEND)"; \
	if [ "$$backend" = "auto" ]; then \
		if [[ "$$BACKENDS" == *"hip"* ]]; then backend="hip"; backend_label="HIP"; \
		elif [[ "$$BACKENDS" == *"cuda"* ]]; then backend="cuda"; backend_label="CUDA"; \
		else echo "No supported accelerator backend found in BACKENDS=$$BACKENDS" >&2; exit 1; fi; \
	else \
		backend_label="$$(printf "%s" "$$backend" | tr "[:lower:]" "[:upper:]")"; \
	fi; \
	for case in $(WEAK_CASES); do \
		chunks="$${case%%:*}"; \
		nval="$${case##*:}"; \
		for rep in $$(seq 1 $(REPS)); do \
			echo "============================================="; \
			echo "Weak scaling: N=$$nval GPUs=$$chunks rep=$$rep/$(REPS)"; \
			echo "============================================="; \
			$(MAKE) run-cwt-usy \
				EXP_BACKEND="$$backend" \
				N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="UniSYCL (Single Queue)"; \
			$(MAKE) run-cwt-acpp \
				N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="ACPP (Explicit Queues)"; \
			$(MAKE) run-cwt-acpp-single \
				N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
				EXTRA="$(CWT_FORWARD_FLAG)" \
				CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="ACPP (Single Queue)"; \
			if [ "$$backend" = "hip" ]; then \
				$(MAKE) run-cwt-dpcpp-hip \
					N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Explicit Queues)"; \
				$(MAKE) run-cwt-dpcpp-hip-single \
					N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Single Queue)"; \
			else \
				$(MAKE) run-cwt-dpcpp-cuda \
					N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Explicit Queues)"; \
				$(MAKE) run-cwt-dpcpp-cuda-single \
					N="$$nval" B="$(B)" GPUS="$$chunks" SHARDS="1" CSV="$(WEAK_SCALE_CSV)" \
					EXTRA="$(CWT_FORWARD_FLAG)" \
					CWT_MACHINE="$$MACHINE" CWT_BACKEND="$$backend_label" CWT_IMPL="DPC++ (Single Queue)"; \
			fi; \
		done; \
	done

paper-strongscaling:
	./run-paper-experiments.sh

paper-oversubscription:
	./run-paper-oversubscription.sh

# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------

# Quick plot-only helpers
# ------------------------------------------------------------

plot-milan:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--machine milan2

plot-cousteau:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--machine cousteau

plot-zenith:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--machine zenith

plot-milan-paper:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--machine milan2 \
		--no-title

plot-cousteau-paper:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--machine cousteau \
		--no-title

plot-zenith-paper:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--machine zenith \
		--no-title

strong-scale-plot:
	$(PIXI_RUN) python $(PLOT_SCRIPT) "$(STRONG_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)"

# -------------------------------------------------------
# Oversubscription plots
# -------------------------------------------------------


plot-over:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "$(PLOT_PHASE)" \
		--machine "$(MACHINE)"

plot-over-cousteau:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "forward" \
		--machine cousteau

plot-over-cousteau-paper:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "forward" \
		--machine cousteau \
		--no-title

plot-over-milan2:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "forward" \
		--machine milan2

plot-over-milan2-paper:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "forward" \
		--machine milan2 \
		--no-title

plot-over-zenith:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "forward" \
		--machine zenith

plot-over-zenith-paper:
	./setup-pixi.sh run python plotting/plot_oversubscription.py \
		"results/oversubscription.csv" \
		--outdir "plots" \
		--phase "forward" \
		--machine zenith \
		--no-title

plot-over: plot-over-cousteau plot-over-milan2 plot-over-zenith

plot-over-paper: plot-over-cousteau-paper plot-over-milan2-paper plot-over-zenith-paper

weak-scale-plot:
	$(PIXI_RUN) python $(WEAK_PLOT_SCRIPT) "$(WEAK_SCALE_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)"

plot-pp:
	$(PIXI_RUN) python $(PORTABILITY_SCRIPT) "$(PORTABILITY_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)"

plot-pp-paper:
	$(PIXI_RUN) python $(PORTABILITY_SCRIPT) "$(PORTABILITY_CSV)" \
		--outdir "$(PLOTS_DIR)" \
		--phase "$(PLOT_PHASE)" \
		--no-title

strong-scale-default: strong-scale-run strong-scale-plot portability-plot
weak-scale-default: weak-scale-run weak-scale-plot

strong-scale-clean:
	rm -f "$(STRONG_SCALE_CSV)"
	rm -rf "$(PLOTS_DIR)"
	rm -f "$(ARCHIVE_FILE)"

weak-scale-clean:
	rm -f "$(WEAK_SCALE_CSV)"
	rm -rf "$(PLOTS_DIR)"

# ------------------------------------------------------------
# Debug helpers
# ------------------------------------------------------------

run-dbg-cwt-usy: cwt-dbg-usy
	CHARM_SYCL_RTS=iris \
	IRIS_ARCHS=$(EXP_BACKEND) \
	IRIS_POLICY=roundrobin \
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	$(RUN_ENV) ./cwt-dbg-usy --mode unisycl --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(DBG_CSV) $(EXTRA)

run-dbg-cwt-acpp: cwt-dbg-acpp
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	$(RUN_ENV) ./cwt-dbg-acpp --mode conventional --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(DBG_CSV) $(EXTRA)

run-dbg-cwt-dpcpp-cuda: cwt-dbg-dpcpp-cuda
	LD_LIBRARY_PATH="$(COMMON_LD_PATH):$(DPCPPCUDA_INSTALL_ROOT)/lib:$(TBB_INSTALL_ROOT)/lib" \
	ONEAPI_DEVICE_SELECTOR="cuda:*" \
	SYCL_DEVICE_FILTER="cuda:*" \
	$(RUN_ENV) ./cwt-dbg-dpcpp-cuda --mode conventional --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(DBG_CSV) $(EXTRA)

run-dbg-cwt-dpcpp-hip: cwt-dbg-dpcpp-hip
	LD_LIBRARY_PATH="$(COMMON_LD_PATH):$(DPCPPHIP_INSTALL_ROOT)/lib:$(TBB_INSTALL_ROOT)/lib" \
	ONEAPI_DEVICE_SELECTOR="hip:*" \
	SYCL_DEVICE_FILTER="hip:*" \
	$(RUN_ENV) ./cwt-dbg-dpcpp-hip --mode conventional --N $(N) --B $(B) --gpus $(GPUS) --shards $(SHARDS) --csv $(DBG_CSV) $(EXTRA)

dbg-forward-compare:
	@set -e; \
	. ./setup-backends.sh; \
	rm -f "$(DBG_CWT_REF_DUMP)" "$(DBG_CWT_USY_DUMP)" "$(DBG_CSV)"; \
	backend="$(EXP_BACKEND)"; \
	if [ "$$backend" = "auto" ]; then \
		if [[ "$$BACKENDS" == *"hip"* ]]; then backend="hip"; \
		elif [[ "$$BACKENDS" == *"cuda"* ]]; then backend="cuda"; \
		else echo "No supported accelerator backend found in BACKENDS=$$BACKENDS" >&2; exit 1; fi; \
	fi; \
	$(MAKE) cwt-dbg-usy; \
	if [ "$(DBG_REF)" = "acpp" ]; then \
		$(MAKE) cwt-dbg-acpp; \
		$(MAKE) run-dbg-cwt-acpp N="$(N)" B="$(B)" GPUS="1" SHARDS="1" DBG_CSV="$(DBG_CSV)" EXTRA="--forward-only --dump-cwt $(DBG_CWT_REF_DUMP)"; \
	elif [ "$(DBG_REF)" = "dpcpp" ] && [ "$$backend" = "hip" ]; then \
		$(MAKE) cwt-dbg-dpcpp-hip; \
		$(MAKE) run-dbg-cwt-dpcpp-hip N="$(N)" B="$(B)" GPUS="1" SHARDS="1" DBG_CSV="$(DBG_CSV)" EXTRA="--forward-only --dump-cwt $(DBG_CWT_REF_DUMP)"; \
	elif [ "$(DBG_REF)" = "dpcpp" ] && [ "$$backend" = "cuda" ]; then \
		$(MAKE) cwt-dbg-dpcpp-cuda; \
		$(MAKE) run-dbg-cwt-dpcpp-cuda N="$(N)" B="$(B)" GPUS="1" SHARDS="1" DBG_CSV="$(DBG_CSV)" EXTRA="--forward-only --dump-cwt $(DBG_CWT_REF_DUMP)"; \
	else \
		echo "Unsupported DBG_REF=$(DBG_REF), use acpp or dpcpp" >&2; exit 1; \
	fi; \
	$(MAKE) run-dbg-cwt-usy N="$(N)" B="$(B)" GPUS="1" SHARDS="1" DBG_CSV="$(DBG_CSV)" EXP_BACKEND="$$backend" EXTRA="--forward-only --dump-cwt $(DBG_CWT_USY_DUMP)"; \
	python3 compare_cwt_dump.py "$(DBG_CWT_REF_DUMP)" "$(DBG_CWT_USY_DUMP)"

dbg-inverse-usy:
	@rm -f "$(DBG_CSV)"
	@$(MAKE) run-dbg-cwt-usy \
		N="$(N)" B="$(B)" GPUS="$(GPUS)" SHARDS="$(SHARDS)" DBG_CSV="$(DBG_CSV)" \
		EXTRA="$(EXTRA)"

#############################################
# IRIS async vs sync trace A/B experiment   #
#############################################

TRACE_N ?= 1024
TRACE_CHUNKS ?= 4
TRACE_B ?= 1

TRACE_CSV ?= /tmp/usy_trace.csv
TRACE_LOG_SYNC ?= /tmp/iris_trace_sync.log
TRACE_LOG_ASYNC ?= /tmp/iris_trace_async.log
TRACE_PROFILE_SYNC ?= /tmp/iris_profile_sync.csv
TRACE_PROFILE_ASYNC ?= /tmp/iris_profile_async.csv

trace-clean:
	rm -f $(TRACE_CSV) $(TRACE_LOG_SYNC) $(TRACE_LOG_ASYNC) \
	      $(TRACE_PROFILE_SYNC) $(TRACE_PROFILE_ASYNC)

trace-sync: cwt-usy
	@echo "=== IRIS SYNC RUN (ASYNC=0) ==="
	IRIS_ARCHS=cuda \
	IRIS_POLICY=roundrobin \
	IRIS_ASYNC=0 \
	IRIS_HISTORY=1 \
	IRIS_PROFILE=1 \
	IRIS_PROFILE_PATH=$(TRACE_PROFILE_SYNC) \
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	CWT_IMPL="UniSYCL (Trace Sync)" \
	CWT_MACHINE="$(shell hostname -s)" \
	CWT_BACKEND="CUDA" \
	./cwt-usy --mode unisycl \
		--N $(TRACE_N) \
		--B $(TRACE_B) \
		--gpus $(TRACE_CHUNKS) \
		--shards 1 \
		--csv $(TRACE_CSV) \
		--forward-only \
	2>&1 | tee $(TRACE_LOG_SYNC)

trace-async: cwt-usy
	@echo "=== IRIS ASYNC RUN (ASYNC=1) ==="
	IRIS_ARCHS=cuda \
	IRIS_POLICY=roundrobin \
	IRIS_ASYNC=1 \
	IRIS_HISTORY=1 \
	IRIS_PROFILE=1 \
	IRIS_PROFILE_PATH=$(TRACE_PROFILE_ASYNC) \
	LD_LIBRARY_PATH="$(COMMON_LD_PATH)" \
	CWT_IMPL="UniSYCL (Trace Async)" \
	CWT_MACHINE="$(shell hostname -s)" \
	CWT_BACKEND="CUDA" \
	./cwt-usy --mode unisycl \
		--N $(TRACE_N) \
		--B $(TRACE_B) \
		--gpus $(TRACE_CHUNKS) \
		--shards 1 \
		--csv $(TRACE_CSV) \
		--forward-only \
	2>&1 | tee $(TRACE_LOG_ASYNC)

trace-ab: trace-clean trace-sync trace-async
	@echo ""
	@echo "=== A/B COMPLETE ==="
	@echo "Sync log:   $(TRACE_LOG_SYNC)"
	@echo "Async log:  $(TRACE_LOG_ASYNC)"
	@echo "Sync prof:  $(TRACE_PROFILE_SYNC)"
	@echo "Async prof: $(TRACE_PROFILE_ASYNC)"

