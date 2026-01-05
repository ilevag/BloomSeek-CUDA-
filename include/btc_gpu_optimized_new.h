#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <string>
#include <fstream>
#include <ios>

/**
 * Optimized Bitcoin GPU functions with VanitySearch optimizations:
 * - Group operations (1024 keys per batch)
 * - Secp256k1 endomorphisms for ~2x speedup
 * - Optimized mathematical operations
 * Expected 10-50x performance improvement over standard implementation
 */

#include "logger.h"

// Global logger instance
extern Logger* g_logger;

// ========================= Bloom Filter Functions =========================

// Load bloom filter to GPU
extern "C" bool gpu_bloom_load(
    const uint8_t* host_bits,
    size_t size_bytes,
    size_t blocks_count,
    uint8_t hash_functions
);

// Unload bloom filter from GPU
extern "C" void gpu_bloom_unload();

// Check keys against bloom filter
extern "C" bool gpu_bloom_check_var(
    const uint8_t* host_keys,
    size_t num_keys,
    size_t key_len,
    uint8_t* host_match_flags
);

// Get device state info
extern "C" bool gpu_bloom_get_device_state(
    const uint8_t** d_bits_out,
    size_t* blocks_count_out,
    uint8_t* hash_functions_out
);

// Bitcoin-specific GPU functions
extern "C" bool gpu_btc_pub33_to_hash160(
    const uint8_t* host_pub33,
    size_t num_keys,
    uint8_t* host_hash160,
    int addr_type
);

extern "C" bool gpu_btc_fused_pipeline(
    const uint8_t* host_pub33,
    size_t num_keys,
    uint8_t* host_flags,
    int addr_type,
    uint8_t* host_hash160_out
);

// Performance statistics
struct BtcGpuStats {
    float gpu_utilization;
    size_t memory_used_mb;
    float endomorphism_speedup;
};

extern "C" bool gpu_btc_get_performance_stats(BtcGpuStats* stats);

// Benchmark comparison
extern "C" bool gpu_btc_benchmark_comparison(
    size_t test_keys,
    int addr_type,
    double* original_time,
    double* optimized_time,
    double* speedup
);

// ============================================================================
// VanitySearch Optimization Functions
// ============================================================================

// Optimized key computation with group operations and endomorphisms
extern "C" bool btc_compute_keys_optimized(const uint64_t* host_priv_keys, size_t num_keys, uint8_t* host_pub_keys);

// Performance benchmark for optimizations
extern "C" float btc_benchmark_optimizations();

// Get optimization statistics
extern "C" void btc_get_optimization_stats(int* group_size, float* endomorphism_speedup, bool* memory_optimized);

