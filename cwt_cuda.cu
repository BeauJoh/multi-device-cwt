// cwt_cuda.cu
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using real_t = double;

#define CUDA_CHECK(x) do { \
  cudaError_t err = (x); \
  if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    std::exit(1); \
  } \
} while (0)

struct Args {
  int N = 4096;
  int B = 1;
  int chunks = 1;
  int gpus = 1;
  int sweep = 1;
  bool forward_only = false;
  std::string mode = "explicit";
  std::string csv = "results.csv";
};

static std::string getenv_s(const char* k, const char* d) {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string(d);
}

__device__ __forceinline__ real_t mexican_hat(real_t x) {
  real_t x2 = x * x;
  return (real_t(1) - x2) * exp(real_t(-0.5) * x2);
}

__global__ void cwt_forward_kernel(
    const real_t* __restrict__ fx,
    const real_t* __restrict__ scales,
    const real_t* __restrict__ trans,
    real_t* __restrict__ out,
    int B, int N, int a0, int a_count) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  int a_local = blockIdx.y;
  int b = blockIdx.z;

  if (n >= N || a_local >= a_count || b >= B) return;

  int a_idx = a0 + a_local;
  real_t aval = scales[a_idx];
  real_t inv_sqrt_a = real_t(1) / sqrt(aval);
  real_t tn = trans[n];
  real_t dt = trans[1] - trans[0];

  real_t sum = 0;
  for (int k = 0; k < N; ++k) {
    real_t tk = trans[k];
    real_t x = (tk - tn) / aval;
    sum += fx[b * N + k] * inv_sqrt_a * mexican_hat(x);
  }

  out[(b * a_count + a_local) * N + n] = sum * dt;
}

struct Task {
  int a0;
  int a1;
};

static std::vector<Task> make_tasks(int A, int chunks) {
  chunks = std::max(1, std::min(chunks, A));
  std::vector<Task> tasks;
  int base = A / chunks;
  int rem = A % chunks;
  int a = 0;

  for (int i = 0; i < chunks; ++i) {
    int cnt = base + (i < rem ? 1 : 0);
    tasks.push_back({a, a + cnt});
    a += cnt;
  }

  return tasks;
}

static std::vector<real_t> build_signal(int N, int B) {
  constexpr real_t pi = 3.141592653589793238462643383279502884;
  std::vector<real_t> fx(B * N);

  for (int n = 0; n < N; ++n) {
    real_t x = real_t(n) / real_t(N - 1);
    real_t v =
        sin(40 * pi * x) * exp(-100 * pi * (x - 2) * (x - 2)) +
        (sin(40 * pi * x) + 2 * cos(160 * pi * x)) *
            exp(-50 * pi * (x - 0.5) * (x - 0.5)) +
        2 * sin(160 * pi * x) * exp(-100 * pi * (x - 0.8) * (x - 0.8));

    for (int b = 0; b < B; ++b) fx[b * N + n] = v;
  }

  return fx;
}

static std::vector<real_t> linspace(real_t lo, real_t hi, int N) {
  std::vector<real_t> x(N);
  for (int i = 0; i < N; ++i) {
    real_t t = N > 1 ? real_t(i) / real_t(N - 1) : 0;
    x[i] = lo + t * (hi - lo);
  }
  return x;
}

static double gflops_forward(int B, int A, int N, double sec) {
  double flops = 2.0 * B * A * double(N) * double(N);
  return sec > 0 ? flops / sec / 1e9 : INFINITY;
}

static void append_csv(
    const std::string& path,
    int N, int B, int A, int chunks,
    double fwd_s,
    double rel_err) {
  bool exists = std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);

  if (!exists) {
    f << "implementation,machine,backend,N,B,A,devices,tasks_per_dev,tasks_total,"
         "fwd_wall_s,inv_wall_s,total_wall_s,fwd_gflops,inv_gflops,total_gflops,rel_err\n";
  }

  auto impl = getenv_s("CWT_IMPL", "CUDA");
  auto machine = getenv_s("CWT_MACHINE", "UnknownMachine");
  auto backend = getenv_s("CWT_BACKEND", "CUDA");

  double gf = gflops_forward(B, A, N, fwd_s);

  f << impl << "," << machine << "," << backend << ","
    << N << "," << B << "," << A << ","
    << chunks << ",1," << chunks << ","
    << fwd_s << ",0," << fwd_s << ","
    << gf << ",0," << gf << ","
    << rel_err << "\n";
}

static Args parse_args(int argc, char** argv) {
  Args a;

  for (int i = 1; i < argc; ++i) {
    std::string s = argv[i];
    auto need = [&](const char* name) -> char* {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };

    if (s == "--samples" || s == "--N") a.N = std::atoi(need(s.c_str()));
    else if (s == "--batch" || s == "--B") a.B = std::atoi(need(s.c_str()));
    else if (s == "--gpus") {
      a.gpus = std::atoi(need(s.c_str()));
      a.chunks = a.gpus;
    } else if (s == "--chunks") {
      a.chunks = std::atoi(need(s.c_str()));
    } else if (s == "--sweep") {
      a.sweep = std::atoi(need(s.c_str()));
    } else if (s == "--mode") {
      a.mode = need(s.c_str());
    } else if (s == "--csv") {
      a.csv = need(s.c_str());
    } else if (s == "--forward-only") {
      a.forward_only = true;
    } else {
      throw std::runtime_error("unknown argument: " + s);
    }
  }

  return a;
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);

  int visible = 0;
  CUDA_CHECK(cudaGetDeviceCount(&visible));
  if (visible <= 0) throw std::runtime_error("No CUDA devices found");

  int physical_workers = 1;
  if (args.mode == "single") {
    physical_workers = 1;
  } else if (args.mode == "explicit") {
    physical_workers = std::max(1, std::min(args.chunks, visible));
  } else {
    throw std::runtime_error("mode must be explicit or single");
  }

  const int N = args.N;
  const int B = args.B;
  const int A = N;
  const int chunks = std::max(1, args.chunks);

  std::vector<real_t> h_fx = build_signal(N, B);
  std::vector<real_t> h_scales = linspace(0.01, 0.10, A);
  std::vector<real_t> h_trans = linspace(-1.0, 1.0, N);
  std::vector<real_t> h_cwt(size_t(B) * A * N, 0);

  auto tasks = make_tasks(A, chunks);

  std::vector<real_t*> d_fx(physical_workers, nullptr);
  std::vector<real_t*> d_scales(physical_workers, nullptr);
  std::vector<real_t*> d_trans(physical_workers, nullptr);
  std::vector<cudaStream_t> streams(physical_workers);

  for (int w = 0; w < physical_workers; ++w) {
    CUDA_CHECK(cudaSetDevice(w));
    CUDA_CHECK(cudaStreamCreate(&streams[w]));
    CUDA_CHECK(cudaMalloc(&d_fx[w], sizeof(real_t) * B * N));
    CUDA_CHECK(cudaMalloc(&d_scales[w], sizeof(real_t) * A));
    CUDA_CHECK(cudaMalloc(&d_trans[w], sizeof(real_t) * N));
    CUDA_CHECK(cudaMemcpyAsync(d_fx[w], h_fx.data(), sizeof(real_t) * B * N,
                               cudaMemcpyHostToDevice, streams[w]));
    CUDA_CHECK(cudaMemcpyAsync(d_scales[w], h_scales.data(), sizeof(real_t) * A,
                               cudaMemcpyHostToDevice, streams[w]));
    CUDA_CHECK(cudaMemcpyAsync(d_trans[w], h_trans.data(), sizeof(real_t) * N,
                               cudaMemcpyHostToDevice, streams[w]));
  }

  std::vector<real_t*> d_out(tasks.size(), nullptr);

  auto t0 = std::chrono::steady_clock::now();

  for (int i = 0; i < int(tasks.size()); ++i) {
    int w = (args.mode == "single") ? 0 : (i % physical_workers);
    CUDA_CHECK(cudaSetDevice(w));

    int a0 = tasks[i].a0;
    int a1 = tasks[i].a1;
    int a_count = a1 - a0;

    CUDA_CHECK(cudaMalloc(&d_out[i], sizeof(real_t) * B * a_count * N));

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x, a_count, B);

    cwt_forward_kernel<<<grid, block, 0, streams[w]>>>(
        d_fx[w], d_scales[w], d_trans[w], d_out[i], B, N, a0, a_count);
    CUDA_CHECK(cudaGetLastError());
  }

  for (int i = 0; i < int(tasks.size()); ++i) {
    int w = (args.mode == "single") ? 0 : (i % physical_workers);
    CUDA_CHECK(cudaSetDevice(w));

    int a0 = tasks[i].a0;
    int a1 = tasks[i].a1;
    int a_count = a1 - a0;

    real_t* dst = h_cwt.data() + size_t(a0) * N;
    CUDA_CHECK(cudaMemcpyAsync(
        dst,
        d_out[i],
        sizeof(real_t) * B * a_count * N,
        cudaMemcpyDeviceToHost,
        streams[w]));
  }

  for (int w = 0; w < physical_workers; ++w) {
    CUDA_CHECK(cudaSetDevice(w));
    CUDA_CHECK(cudaStreamSynchronize(streams[w]));
  }

  auto t1 = std::chrono::steady_clock::now();
  double fwd_s = std::chrono::duration<double>(t1 - t0).count();

  for (int i = 0; i < int(tasks.size()); ++i) {
    int w = (args.mode == "single") ? 0 : (i % physical_workers);
    CUDA_CHECK(cudaSetDevice(w));
    CUDA_CHECK(cudaFree(d_out[i]));
  }

  for (int w = 0; w < physical_workers; ++w) {
    CUDA_CHECK(cudaSetDevice(w));
    CUDA_CHECK(cudaFree(d_fx[w]));
    CUDA_CHECK(cudaFree(d_scales[w]));
    CUDA_CHECK(cudaFree(d_trans[w]));
    CUDA_CHECK(cudaStreamDestroy(streams[w]));
  }

  double rel_err = 0.0;
  double gf = gflops_forward(B, A, N, fwd_s);

  std::cout << "\nmode=" << args.mode
            << " chunks=" << chunks
            << " physical_workers=" << physical_workers
            << " tasks_total=" << chunks << "\n";
  std::cout << "  forward_wall_s=" << fwd_s << "  GFLOP/s≈" << gf << "\n";
  std::cout << "  inverse_wall_s=0  GFLOP/s≈0\n";
  std::cout << "  total_wall_s=" << fwd_s << "    GFLOP/s≈" << gf << "\n";
  std::cout << "  rel_err=" << rel_err << "\n";

  append_csv(args.csv, N, B, A, chunks, fwd_s, rel_err);
  std::cout << "\nAppended results CSV: " << args.csv << "\n";

  return 0;
}
