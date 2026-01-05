#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cstring>

#include "sol_utils.h"
#include "cuda-ecc-ed25519/sha512.h"
#include "cuda-ecc-ed25519/ge.h"

// Simple CUDA kernel: seed32 -> sha512 -> clamp -> ed25519 basepoint scalar mult -> pub32.
__global__ void sol_priv_to_pub_kernel(const uint8_t* priv32, size_t num, uint8_t* out_pub32) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;

    const uint8_t* seed = priv32 + idx * 32;
    uint8_t digest[64];
    sha512(seed, 32, digest);

    digest[0]  &= 248;
    digest[31] &= 63;
    digest[31] |= 64;

    ge_p3 A;
    ge_scalarmult_base(&A, digest);
    ge_p3_tobytes(out_pub32 + idx * 32, &A);
}

// Reuse device buffers/stream across calls to avoid per-batch malloc/free overhead.
static uint8_t* g_d_in = nullptr;
static uint8_t* g_d_out = nullptr;
static size_t g_capacity = 0;
static cudaStream_t g_stream = nullptr;

// Buffers for GPU-generated priv/pub
static uint8_t* g_d_gen_priv = nullptr;
static uint8_t* g_d_gen_pub = nullptr;
static size_t g_gen_capacity = 0;

__device__ uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__global__ void sol_generate_priv_pub_kernel(uint64_t seed_base,
                                             size_t num,
                                             uint8_t* out_priv32,
                                             uint8_t* out_pub32) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num) return;

    // Pseudorandom 32-byte seed via splitmix64 stream; deterministic per seed_base/idx
    uint64_t state = seed_base + idx * 0x9e3779b97f4a7c15ULL;
    uint8_t priv[32];
    for (int i = 0; i < 4; ++i) {
        uint64_t v = splitmix64(state);
        std::memcpy(priv + i * 8, &v, 8);
    }

    // Derive pub (same as sol_priv_to_pub_kernel)
    uint8_t digest[64];
    sha512(priv, 32, digest);
    digest[0]  &= 248;
    digest[31] &= 63;
    digest[31] |= 64;

    ge_p3 A;
    ge_scalarmult_base(&A, digest);

    // Write results
    std::memcpy(out_priv32 + idx * 32, priv, 32);
    ge_p3_tobytes(out_pub32 + idx * 32, &A);
}

static bool sol_priv_to_pub_gpu_impl(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32) {
    if (!priv32 || num == 0) return false;

    out_pub32.resize(num * 32);

    const size_t bytes_in = num * 32;
    const size_t bytes_out = num * 32;

    // Allocate/resize once with headroom
    if (num > g_capacity) {
        if (g_d_in) cudaFree(g_d_in);
        if (g_d_out) cudaFree(g_d_out);
        size_t new_cap = num * 3 / 2; // 1.5x headroom
        if (cudaMalloc(&g_d_in, new_cap * 32) != cudaSuccess) {
            g_d_in = nullptr;
            g_d_out = nullptr;
            g_capacity = 0;
            return false;
        }
        if (cudaMalloc(&g_d_out, new_cap * 32) != cudaSuccess) {
            cudaFree(g_d_in);
            g_d_in = nullptr;
            g_d_out = nullptr;
            g_capacity = 0;
            return false;
        }
        g_capacity = new_cap;
    }

    if (!g_stream) {
        if (cudaStreamCreate(&g_stream) != cudaSuccess) {
            g_stream = nullptr;
            return false;
        }
    }

    const int threads = 256;
    const int blocks = static_cast<int>((num + threads - 1) / threads);
    cudaError_t err;

    err = cudaMemcpyAsync(g_d_in, priv32, bytes_in, cudaMemcpyHostToDevice, g_stream);
    if (err != cudaSuccess) return false;

    sol_priv_to_pub_kernel<<<blocks, threads, 0, g_stream>>>(g_d_in, num, g_d_out);
    err = cudaGetLastError();
    if (err != cudaSuccess) return false;

    err = cudaMemcpyAsync(out_pub32.data(), g_d_out, bytes_out, cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) return false;

    err = cudaStreamSynchronize(g_stream);
    return err == cudaSuccess;
}

// Generate priv+pub on GPU, copy both to host
bool sol_generate_priv_pub_gpu(size_t num,
                               uint64_t seed_base,
                               std::vector<uint8_t>& out_priv32,
                               std::vector<uint8_t>& out_pub32) {
    if (num == 0) return false;
    out_priv32.resize(num * 32);
    out_pub32.resize(num * 32);

    // Allocate/resize device buffers with headroom
    if (num > g_gen_capacity) {
        if (g_d_gen_priv) cudaFree(g_d_gen_priv);
        if (g_d_gen_pub) cudaFree(g_d_gen_pub);
        size_t new_cap = num * 3 / 2;
        if (cudaMalloc(&g_d_gen_priv, new_cap * 32) != cudaSuccess) {
            g_d_gen_priv = nullptr;
            g_d_gen_pub = nullptr;
            g_gen_capacity = 0;
            return false;
        }
        if (cudaMalloc(&g_d_gen_pub, new_cap * 32) != cudaSuccess) {
            cudaFree(g_d_gen_priv);
            g_d_gen_priv = nullptr;
            g_d_gen_pub = nullptr;
            g_gen_capacity = 0;
            return false;
        }
        g_gen_capacity = new_cap;
    }

    if (!g_stream) {
        if (cudaStreamCreate(&g_stream) != cudaSuccess) {
            g_stream = nullptr;
            return false;
        }
    }

    const int threads = 256;
    const int blocks = static_cast<int>((num + threads - 1) / threads);
    sol_generate_priv_pub_kernel<<<blocks, threads, 0, g_stream>>>(seed_base, num, g_d_gen_priv, g_d_gen_pub);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return false;

    err = cudaMemcpyAsync(out_priv32.data(), g_d_gen_priv, num * 32, cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) return false;
    err = cudaMemcpyAsync(out_pub32.data(), g_d_gen_pub, num * 32, cudaMemcpyDeviceToHost, g_stream);
    if (err != cudaSuccess) return false;

    err = cudaStreamSynchronize(g_stream);
    return err == cudaSuccess;
}

bool sol_priv_to_pub_gpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32) {
    return sol_priv_to_pub_gpu_impl(priv32, num, out_pub32);
}

bool sol_priv_to_pub32_gpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32) {
    return sol_priv_to_pub_gpu_impl(priv32, num, out_pub32);
}

