#pragma once

#include <vector>
#include <cstdint>
#include <array>

struct BtcKeyPairBinary {
    std::array<uint8_t, 32> private_key;
    std::array<uint8_t, 20> address20; // P2PKH or P2SH hash160
};

struct BtcPrivPub {
    std::array<uint8_t, 32> private_key;
    std::array<uint8_t, 33> public_key33; // compressed public key
};

enum class BitcoinAddressType {
    P2PKH = 1,
    P2SH = 2
};

class BitcoinGenerator {
public:
    BitcoinGenerator();
    ~BitcoinGenerator();

    bool initialize(); // CPU-only for now
    std::vector<BtcKeyPairBinary> generate_batch_binary(size_t batch, BitcoinAddressType addr_type);
    std::vector<BtcPrivPub> generate_batch_priv_pub(size_t batch);
    
    // NEW: Optimized methods with endomorphisms (6x faster)
    std::vector<BtcKeyPairBinary> generate_batch_optimized(size_t batch, BitcoinAddressType addr_type);
    std::vector<BtcKeyPairBinary> generate_batch_with_endomorphisms(size_t batch, BitcoinAddressType addr_type);
    std::vector<BtcPrivPub> generate_batch_priv_pub_with_endomorphisms(size_t batch); // 6x version for Mode 6
    
    // Benchmark and performance testing
    double benchmark_performance(size_t test_keys, BitcoinAddressType addr_type);
    void print_performance_comparison(size_t test_keys);
    
    // Set address type for generation
    void set_address_type(BitcoinAddressType type) { address_type = type; }
    
    // Enable/disable optimizations
    void set_use_optimizations(bool enable) { use_optimizations = enable; }
    bool get_use_optimizations() const { return use_optimizations; }
    
    // Compute 6 endomorphic private keys from base key (for match processing)
    bool compute_endomorphic_privkeys(const uint8_t base_priv32[32], uint8_t out_privkeys[6][32]) const;

    // Helper: derive compressed public key from private key
    bool priv_to_pub_compressed(const uint8_t priv32[32], uint8_t pub33[33]) const;
    
private:
    BitcoinAddressType address_type = BitcoinAddressType::P2PKH;
    bool use_optimizations = true; // Use optimized version by default
};

// CPU implementation of HASH160 conversion for BTC (public key 33 bytes -> address hash160 20 bytes)
// For P2PKH: RIPEMD160(SHA256(pubkey))
// For P2SH (nested P2WPKH-in-P2SH): RIPEMD160(SHA256(0x00 0x14 <hash160(pubkey)>))
void btc_pub33_to_hash160_cpu(const uint8_t pub33[33], BitcoinAddressType addr_type, uint8_t out20[20]);
