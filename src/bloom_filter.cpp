#include "bloom_filter.h"
#include "logger.h"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>

// v1 uses std::hash (compat), v2 uses deterministic hashes
#include <functional>

BloomFilter::BloomFilter(size_t expected_items, double fpr) 
    : expected_elements(expected_items), false_positive_rate(fpr), k_hashes(SBBF_HASH_FUNCTIONS) {
    
    // Compute optimal number of bits
    m_bits = calculate_optimal_bits(expected_items, fpr);
    
    // Round up to block count (each block 256 bits)
    m_blocks = calculate_blocks_count(m_bits);
    
    // Recompute exact number of bits
    m_bits = m_blocks * SBBF_BLOCK_SIZE_BITS;
    
    // Initialize bit array (bytes)
    bit_array.resize(m_blocks * SBBF_BLOCK_SIZE_BYTES, 0);
    
    LOG_INFO("BloomFilter created: " + std::to_string(expected_items) + " items, " +
             "FPR: " + std::to_string(fpr) + ", " +
             "size: " + std::to_string(get_size_bytes()) + " bytes");
}

BloomFilter::BloomFilter() 
    : m_bits(0), m_blocks(0), k_hashes(SBBF_HASH_FUNCTIONS), 
      expected_elements(0), false_positive_rate(0.0) {
}

size_t BloomFilter::calculate_optimal_bits(size_t n, double fpr) const {
    // Fixed number of hashes k=SBBF_HASH_FUNCTIONS (SBBF: k=8).
    // Invert formula for m with fixed k:
    // p = (1 - e^{-k n / m})^k  =>  m = n * k / -ln(1 - p^{1/k})
    if (n == 0) return 0;
    const double k = static_cast<double>(SBBF_HASH_FUNCTIONS);
    const double p = std::clamp(fpr, 1e-12, 0.5);
    const double root = std::pow(p, 1.0 / k);
    double denom = -std::log(1.0 - root);
    if (!(denom > 0.0)) {
        // fallback to classic formula (unlikely)
        double ln2 = std::log(2.0);
        double bits_alt = -static_cast<double>(n) * std::log(p) / (ln2 * ln2);
        return static_cast<size_t>(std::ceil(bits_alt));
    }
    double bits = static_cast<double>(n) * k / denom;
    return static_cast<size_t>(std::ceil(bits));
}

size_t BloomFilter::calculate_blocks_count(size_t bits) const {
    // Round up to number of blocks
    return (bits + SBBF_BLOCK_SIZE_BITS - 1) / SBBF_BLOCK_SIZE_BITS;
}

// v1 — compatibility with old filters
uint64_t BloomFilter::hash1_v1(const std::string& item) const {
    return std::hash<std::string>{}(item);
}
uint64_t BloomFilter::hash2_v1(const std::string& item) const {
    std::hash<std::string> hasher;
    uint64_t h = hasher(item);
    return h * 0x9e3779b9 + (h >> 32);
}
uint64_t BloomFilter::get_hash_v1(const std::string& item, int hash_num) const {
    uint64_t h1 = hash1_v1(item);
    uint64_t h2 = hash2_v1(item);
    return h1 + static_cast<uint64_t>(hash_num) * h2;
}

// v2 — deterministic byte hashing (FNV-1a 64 with variation)
uint64_t BloomFilter::rotl64(uint64_t x, int r) const { return (x << r) | (x >> (64 - r)); }
uint64_t BloomFilter::fnv1a64(const uint8_t* data, size_t len, uint64_t seed) const {
    const uint64_t FNV_OFFSET = 14695981039346656037ULL ^ seed;
    const uint64_t FNV_PRIME  = 1099511628211ULL;
    uint64_t hash = FNV_OFFSET;
    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= FNV_PRIME;
    }
    // Extra mixing for stability
    hash ^= (hash >> 33);
    hash *= 0xff51afd7ed558ccdULL;
    hash ^= (hash >> 33);
    hash *= 0xc4ceb9fe1a85ec53ULL;
    hash ^= (hash >> 33);
    return hash;
}
uint64_t BloomFilter::hash1_v2(const uint8_t* data, size_t len) const {
    return fnv1a64(data, len, 0);
}
uint64_t BloomFilter::hash2_v2(const uint8_t* data, size_t len) const {
    // Independent variation via fixed seed and rotation
    uint64_t h = fnv1a64(data, len, 0x9e3779b97f4a7c15ULL);
    return rotl64(h, 31) ^ 0x9e3779b97f4a7c15ULL;
}
uint64_t BloomFilter::get_hash_v2(const uint8_t* data, size_t len, int hash_num) const {
    uint64_t h1 = hash1_v2(data, len);
    uint64_t h2 = hash2_v2(data, len);
    return h1 + static_cast<uint64_t>(hash_num) * h2;
}

void BloomFilter::add(const std::string& item) {
    if (!is_initialized()) {
        LOG_ERROR("Attempt to add into uninitialized filter");
        return;
    }
    
    for (int i = 0; i < k_hashes; ++i) {
        uint64_t hash_value = (file_version == 1)
            ? get_hash_v1(item, i)
            : get_hash_v2(reinterpret_cast<const uint8_t*>(item.data()), item.size(), i);
        
        // Determine block
        size_t block_index = hash_value % m_blocks;
        
        // Determine bit within block
        size_t bit_in_block = (hash_value >> 32) % SBBF_BLOCK_SIZE_BITS;
        
        // Compute absolute byte and bit index
        size_t byte_index = block_index * SBBF_BLOCK_SIZE_BYTES + (bit_in_block / 8);
        size_t bit_index = bit_in_block % 8;
        
        // Set bit
        bit_array[byte_index] |= (1U << bit_index);
    }
}

void BloomFilter::add_bytes(const uint8_t* data, size_t len) {
    if (!is_initialized()) {
        LOG_ERROR("Attempt to add into uninitialized filter");
        return;
    }
    for (int i = 0; i < k_hashes; ++i) {
        uint64_t hash_value = get_hash_v2(data, len, i);
        size_t block_index = hash_value % m_blocks;
        size_t bit_in_block = (hash_value >> 32) % SBBF_BLOCK_SIZE_BITS;
        size_t byte_index = block_index * SBBF_BLOCK_SIZE_BYTES + (bit_in_block / 8);
        size_t bit_index = bit_in_block % 8;
        bit_array[byte_index] |= (1U << bit_index);
    }
}

bool BloomFilter::might_contain(const std::string& item) const {
    if (!is_initialized()) {
        return false;
    }
    
    for (int i = 0; i < k_hashes; ++i) {
        uint64_t hash_value = (file_version == 1)
            ? get_hash_v1(item, i)
            : get_hash_v2(reinterpret_cast<const uint8_t*>(item.data()), item.size(), i);
        
        // Determine block
        size_t block_index = hash_value % m_blocks;
        
        // Determine bit within block
        size_t bit_in_block = (hash_value >> 32) % SBBF_BLOCK_SIZE_BITS;
        
        // Compute absolute byte and bit index
        size_t byte_index = block_index * SBBF_BLOCK_SIZE_BYTES + (bit_in_block / 8);
        size_t bit_index = bit_in_block % 8;
        
        // Check bit
        if ((bit_array[byte_index] & (1U << bit_index)) == 0) {
            return false; // Definitely not present
        }
    }
    
    return true; // Possibly present
}

bool BloomFilter::might_contain_bytes(const uint8_t* data, size_t len) const {
    if (!is_initialized()) {
        return false;
    }
    for (int i = 0; i < k_hashes; ++i) {
        uint64_t hash_value = get_hash_v2(data, len, i);
        size_t block_index = hash_value % m_blocks;
        size_t bit_in_block = (hash_value >> 32) % SBBF_BLOCK_SIZE_BITS;
        size_t byte_index = block_index * SBBF_BLOCK_SIZE_BYTES + (bit_in_block / 8);
        size_t bit_index = bit_in_block % 8;
        if ((bit_array[byte_index] & (1U << bit_index)) == 0) {
            return false;
        }
    }
    return true;
}

bool BloomFilter::save_to_file(const std::string& filename) const {
    if (!is_initialized()) {
        LOG_ERROR("Attempt to save uninitialized filter");
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for write: " + filename);
        return false;
    }
    
    try {
        // File header
        const uint32_t magic_number = 0x424C4F4D; // "BLOM"
        const uint32_t version = file_version; // v2
        
        file.write(reinterpret_cast<const char*>(&magic_number), sizeof(magic_number));
        file.write(reinterpret_cast<const char*>(&version), sizeof(version));
        
        // Filter parameters
        file.write(reinterpret_cast<const char*>(&m_bits), sizeof(m_bits));
        file.write(reinterpret_cast<const char*>(&m_blocks), sizeof(m_blocks));
        file.write(reinterpret_cast<const char*>(&k_hashes), sizeof(k_hashes));
        file.write(reinterpret_cast<const char*>(&expected_elements), sizeof(expected_elements));
        file.write(reinterpret_cast<const char*>(&false_positive_rate), sizeof(false_positive_rate));
        
        // Data size
        size_t data_size = bit_array.size();
        file.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
        
        // Filter data
        file.write(reinterpret_cast<const char*>(bit_array.data()), data_size);
        
        file.close();
        
        LOG_INFO("Filter saved to file: " + filename + " (" + 
                std::to_string(data_size) + " bytes)");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error writing filter to file " + filename + ": " + e.what());
        return false;
    }
}

bool BloomFilter::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open file for read: " + filename);
        return false;
    }
    
    try {
        // Check header
        uint32_t magic_number, version;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));
        
        if (magic_number != 0x424C4F4D) {
            LOG_ERROR("Invalid Bloom filter file format: " + filename);
            return false;
        }
        
        if (version != 1 && version != 2) {
            LOG_ERROR("Unsupported Bloom filter file version: " + std::to_string(version));
            return false;
        }
        file_version = version;
        
        // Read filter parameters
        file.read(reinterpret_cast<char*>(&m_bits), sizeof(m_bits));
        file.read(reinterpret_cast<char*>(&m_blocks), sizeof(m_blocks));
        file.read(reinterpret_cast<char*>(&k_hashes), sizeof(k_hashes));
        file.read(reinterpret_cast<char*>(&expected_elements), sizeof(expected_elements));
        file.read(reinterpret_cast<char*>(&false_positive_rate), sizeof(false_positive_rate));
        
        // Read data size
        size_t data_size;
        file.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
        
        // Read filter data
        bit_array.resize(data_size);
        file.read(reinterpret_cast<char*>(bit_array.data()), data_size);
        
        file.close();
        
        LOG_INFO("Filter loaded from file: " + filename + " (" + 
                std::to_string(data_size) + " bytes, " +
                std::to_string(expected_elements) + " elements)");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("Error reading filter from file " + filename + ": " + e.what());
        return false;
    }
}

BloomFilter::FilterStats BloomFilter::get_stats() const {
    FilterStats stats;
    stats.size_bytes = get_size_bytes();
    stats.size_bits = get_size_bits();
    stats.blocks_count = m_blocks;
    stats.hash_functions = k_hashes;
    stats.false_positive_rate = false_positive_rate;
    stats.expected_elements = expected_elements;
    
    if (expected_elements > 0) {
        stats.bytes_per_element = static_cast<double>(stats.size_bytes) / expected_elements;
    } else {
        stats.bytes_per_element = 0.0;
    }
    
    return stats;
}

void BloomFilter::clear() {
    std::fill(bit_array.begin(), bit_array.end(), 0);
    LOG_INFO("Filter cleared");
}