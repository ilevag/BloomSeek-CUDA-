#pragma once

#include <cstddef>
#include <cstdint>

// Convert a batch of 32-byte private keys to 64-byte uncompressed public keys (X||Y) on GPU.
// Returns true on success. If returns false, caller should fallback to CPU implementation.
// host_priv32: num * 32 bytes
// host_pub64: output buffer, num * 64 bytes
extern "C" bool gpu_secp256k1_priv_to_pub64(const uint8_t* host_priv32, size_t num, uint8_t* host_pub64);

// Experimental fused pipeline: GPU RNG -> priv32 -> pub64 -> addr20 -> Bloom.
// Generates num keys starting from base_nonce, checks against Bloom filter on GPU,
// and copies priv32/addr20/flags back to host.
extern "C" bool gpu_eth_fused_priv_gen_bloom(
    size_t num,
    uint64_t base_seed,
    const uint8_t* d_bloom_bits,
    size_t bloom_blocks,
    uint8_t k_hashes,
    uint8_t* host_priv32,
    uint8_t* host_addr20,
    uint8_t* host_flags);

// OPTIMIZED: Fused pipeline with Batch Modular Inversion (Montgomery's trick).
// Uses 1 modular inversion per GPU block instead of 1 per thread.
// Expected speedup: 10-15x for the scalar multiplication stage.
// Falls back to standard version if this fails.
extern "C" bool gpu_eth_fused_priv_gen_bloom_batch_inv(
    size_t num,
    uint64_t base_seed,
    const uint8_t* d_bloom_bits,
    size_t bloom_blocks,
    uint8_t k_hashes,
    uint8_t* host_priv32,
    uint8_t* host_addr20,
    uint8_t* host_flags);

