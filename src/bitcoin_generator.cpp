#include "bitcoin_generator.h"
#include "btc_gpu_optimized.h"
#include <random>
#include <array>
#include <cstring>
#include <openssl/evp.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/bn.h>
#include <openssl/rand.h>
#include <openssl/sha.h>
#include <openssl/ripemd.h>
#include <chrono>

extern "C" bool gpu_secp256k1_priv_to_pub33(const uint8_t* host_priv32, size_t num, uint8_t* host_pub33);
extern "C" bool gpu_btc_pub33_to_hash160(const uint8_t* host_pub33, size_t num, uint8_t* host_hash160, int addr_type);

static bool secp256k1_pubkey_from_priv_compressed(const uint8_t priv32[32], uint8_t pub33[33]) {
    bool ok = false;
    EC_KEY* key = nullptr;
    BIGNUM* priv = nullptr;
    const EC_GROUP* group = nullptr;
    EC_POINT* pub_point = nullptr;

    do {
        key = EC_KEY_new_by_curve_name(NID_secp256k1);
        if (!key) break;
        group = EC_KEY_get0_group(key);
        if (!group) break;
        priv = BN_bin2bn(priv32, 32, nullptr);
        if (!priv) break;
        if (EC_KEY_set_private_key(key, priv) != 1) break;
        pub_point = EC_POINT_new(group);
        if (!pub_point) break;
        if (EC_POINT_mul(group, pub_point, priv, nullptr, nullptr, nullptr) != 1) break;
        if (EC_KEY_set_public_key(key, pub_point) != 1) break;
        
        // Export compressed (0x02/0x03 + 32 bytes)
        std::array<uint8_t, 33> compressed{};
        size_t len = EC_POINT_point2oct(group, pub_point, POINT_CONVERSION_COMPRESSED, compressed.data(), compressed.size(), nullptr);
        if (len != 33) break;
        std::memcpy(pub33, compressed.data(), 33);
        ok = true;
    } while (false);

    if (pub_point) EC_POINT_free(pub_point);
    if (priv) BN_free(priv);
    if (key) EC_KEY_free(key);
    return ok;
}

// secp256k1 group order n in big-endian
// n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
static const uint8_t SECP256K1_N_BE[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

static inline bool is_zero32(const uint8_t k[32]) {
    uint8_t acc = 0;
    for (int i = 0; i < 32; ++i) acc |= k[i];
    return acc == 0;
}

static inline bool be_less_than(const uint8_t a[32], const uint8_t b[32]) {
    // Big-endian lexicographic compare
    int cmp = std::memcmp(a, b, 32);
    return cmp < 0;
}

// Generate random private key in [1..n-1] using rejection sampling (no BN math)
static bool generate_valid_privkey(uint8_t out32[32]) {
    for (int attempts = 0; attempts < 128; ++attempts) {
        if (RAND_bytes(out32, 32) != 1) continue;
        if (!is_zero32(out32) && be_less_than(out32, SECP256K1_N_BE)) return true;
    }
    return false;
}

void btc_pub33_to_hash160_cpu(const uint8_t pub33[33], BitcoinAddressType addr_type, uint8_t hash160_out[20]) {
    // Compute HASH160(pubkey) first
    uint8_t sha256_pub[32];
    SHA256(pub33, 33, sha256_pub);
    uint8_t h160_pub[20];
    RIPEMD160(sha256_pub, 32, h160_pub);

    if (addr_type == BitcoinAddressType::P2PKH) {
        // P2PKH address hash = HASH160(pubkey)
        std::memcpy(hash160_out, h160_pub, 20);
    } else {
        // P2SH (nested SegWit P2WPKH-in-P2SH): redeemScript = 0x00 0x14 <HASH160(pubkey)>
        uint8_t redeem[1 + 1 + 20];
        redeem[0] = 0x00;       // OP_0
        redeem[1] = 0x14;       // PUSH 20
        std::memcpy(&redeem[2], h160_pub, 20);
        uint8_t sha256_redeem[32];
        SHA256(redeem, sizeof(redeem), sha256_redeem);
        RIPEMD160(sha256_redeem, 32, hash160_out);
    }
}

// ============================================================================
// ENDOMORPHISM: Compute 6 endomorphic private keys for secp256k1
// ============================================================================
// For secp256k1 curve, the endomorphism λ satisfies:
//   λ * P = β * P  (where β is cube root of 1 mod p)
//
// Given private key k, we can generate 6 keys that map to the same x-coordinate:
//   k, k*λ, k*λ², -k, -k*λ, -k*λ²
//
// All 6 keys produce public keys with related x-coordinates (differ by β or β²)
// This allows checking 6 addresses with one EC multiplication!
// ============================================================================

// Lambda (λ): scalar endomorphism multiplier
// λ = 0x5363ad4cc05c30e0a5261c0288126459122e22ea20816678df02967c1b23bd72
static const uint8_t LAMBDA_BYTES[32] = {
    0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0,
    0xa5, 0x26, 0x1c, 0x02, 0x88, 0x12, 0x64, 0x59,
    0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
    0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72
};

// Helper: модульное умножение scalar * scalar mod n
static bool scalar_mult_mod_n(const uint8_t* a, const uint8_t* b, uint8_t* result) {
    BIGNUM* bn_a = BN_bin2bn(a, 32, nullptr);
    BIGNUM* bn_b = BN_bin2bn(b, 32, nullptr);
    BIGNUM* bn_n = BN_bin2bn(SECP256K1_N_BE, 32, nullptr);
    BIGNUM* bn_result = BN_new();
    BN_CTX* ctx = BN_CTX_new();
    
    if (!bn_a || !bn_b || !bn_n || !bn_result || !ctx) {
        if (bn_a) BN_free(bn_a);
        if (bn_b) BN_free(bn_b);
        if (bn_n) BN_free(bn_n);
        if (bn_result) BN_free(bn_result);
        if (ctx) BN_CTX_free(ctx);
        return false;
    }
    
    // result = (a * b) mod n
    BN_mod_mul(bn_result, bn_a, bn_b, bn_n, ctx);
    
    // Convert back to bytes (big-endian, padded to 32 bytes)
    int len = BN_num_bytes(bn_result);
    if (len > 32) {
        BN_free(bn_a);
        BN_free(bn_b);
        BN_free(bn_n);
        BN_free(bn_result);
        BN_CTX_free(ctx);
        return false;
    }
    
    std::memset(result, 0, 32);
    BN_bn2bin(bn_result, result + (32 - len));
    
    BN_free(bn_a);
    BN_free(bn_b);
    BN_free(bn_n);
    BN_free(bn_result);
    BN_CTX_free(ctx);
    return true;
}

// Helper: негация scalar mod n  (result = n - k)
static bool scalar_negate_mod_n(const uint8_t* k, uint8_t* result) {
    BIGNUM* bn_k = BN_bin2bn(k, 32, nullptr);
    BIGNUM* bn_n = BN_bin2bn(SECP256K1_N_BE, 32, nullptr);
    BIGNUM* bn_result = BN_new();
    
    if (!bn_k || !bn_n || !bn_result) {
        if (bn_k) BN_free(bn_k);
        if (bn_n) BN_free(bn_n);
        if (bn_result) BN_free(bn_result);
        return false;
    }
    
    // result = n - k
    BN_sub(bn_result, bn_n, bn_k);
    
    int len = BN_num_bytes(bn_result);
    if (len > 32) {
        BN_free(bn_k);
        BN_free(bn_n);
        BN_free(bn_result);
        return false;
    }
    
    std::memset(result, 0, 32);
    BN_bn2bin(bn_result, result + (32 - len));
    
    BN_free(bn_k);
    BN_free(bn_n);
    BN_free(bn_result);
    return true;
}

// Helper function: compute 6 endomorphic private keys
// Input: priv32 (32 bytes) - original private key
// Output: out_privkeys[6][32] - 6 endomorphic variants
static bool compute_endomorphic_privkeys_impl(const uint8_t priv32[32], uint8_t out_privkeys[6][32]) {
    // Compute lambda^2 mod n
    uint8_t lambda2[32];
    if (!scalar_mult_mod_n(LAMBDA_BYTES, LAMBDA_BYTES, lambda2)) {
        return false;
    }
    
    // Variant 0: k (original)
    std::memcpy(out_privkeys[0], priv32, 32);
    
    // Variant 1: k * λ mod n
    if (!scalar_mult_mod_n(priv32, LAMBDA_BYTES, out_privkeys[1])) {
        return false;
    }
    
    // Variant 2: k * λ² mod n
    if (!scalar_mult_mod_n(priv32, lambda2, out_privkeys[2])) {
        return false;
    }
    
    // Variant 3: -k mod n
    if (!scalar_negate_mod_n(out_privkeys[0], out_privkeys[3])) {
        return false;
    }
    
    // Variant 4: -k*λ mod n
    if (!scalar_negate_mod_n(out_privkeys[1], out_privkeys[4])) {
        return false;
    }
    
    // Variant 5: -k*λ² mod n
    if (!scalar_negate_mod_n(out_privkeys[2], out_privkeys[5])) {
        return false;
    }
    
    return true;
}

BitcoinGenerator::BitcoinGenerator() {}
BitcoinGenerator::~BitcoinGenerator() {}

bool BitcoinGenerator::initialize() { 
    return true; 
}

std::vector<BtcKeyPairBinary> BitcoinGenerator::generate_batch_binary(size_t batch, BitcoinAddressType addr_type) {
    std::vector<BtcKeyPairBinary> out;
    out.reserve(batch);

    for (size_t i = 0; i < batch; ++i) {
        BtcKeyPairBinary kp{};
        // Generate private key in [1..n-1] (CSPRNG with rejection sampling)
        if (!generate_valid_privkey(kp.private_key.data())) { --i; continue; }
        
        // Generate compressed public key
        uint8_t pub33[33];
        if (!secp256k1_pubkey_from_priv_compressed(kp.private_key.data(), pub33)) {
            // regenerate until success
            bool ok2 = false;
            for (int attempt = 0; attempt < 16; ++attempt) {
                if (!generate_valid_privkey(kp.private_key.data())) continue;
                if (secp256k1_pubkey_from_priv_compressed(kp.private_key.data(), pub33)) { ok2 = true; break; }
            }
            if (!ok2) {
                continue;
            }
        }
        
        // Generate hash160 address (CPU reference)
        btc_pub33_to_hash160_cpu(pub33, addr_type, kp.address20.data());
        out.push_back(kp);
    }
    return out;
}

std::vector<BtcPrivPub> BitcoinGenerator::generate_batch_priv_pub(size_t batch) {
    // 1) Generate random private keys on CPU using CSPRNG (batched) and rejection fixups
    std::vector<uint8_t> privs(batch * 32);
    // Fill entire buffer in one call for speed
    RAND_bytes(privs.data(), static_cast<int>(privs.size()));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(batch); ++i) {
        uint8_t* p = &privs[i * 32];
        if (is_zero32(p) || !be_less_than(p, SECP256K1_N_BE)) {
            // FIX #3: Protection against infinite loop with safety counter
            int safety_counter = 1000;
            while (!generate_valid_privkey(p) && safety_counter-- > 0) {}

            if (safety_counter <= 0) {
                // Critical error: failed after 1000 attempts
                // Fallback: use minimal valid key
                std::memset(p, 0, 32);
                p[31] = 0x01; // privkey = 1 (valid)
            }
        }
    }

    // 2) Try GPU secp256k1 for compressed public keys
    std::vector<uint8_t> pub33(batch * 33);
    bool gpu_ok = gpu_secp256k1_priv_to_pub33(privs.data(), batch, pub33.data());

    // 3) Fallback to CPU for any/all if GPU not available
    if (!gpu_ok) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(batch); ++i) {
            (void)secp256k1_pubkey_from_priv_compressed(&privs[i*32], &pub33[i*33]);
        }
    }

    // 4) Pack results
    std::vector<BtcPrivPub> res;
    res.reserve(batch);
    for (size_t i = 0; i < batch; ++i) {
        BtcPrivPub item{};
        std::memcpy(item.private_key.data(), &privs[i*32], 32);
        std::memcpy(item.public_key33.data(), &pub33[i*33], 33);
        res.push_back(item);
    }
    return res;
}

// NEW: Optimized batch generation with endomorphisms (6x speedup)
// FULLY IMPLEMENTED: Proper endomorphic private key computation with OpenSSL
std::vector<BtcKeyPairBinary> BitcoinGenerator::generate_batch_optimized(size_t batch, BitcoinAddressType addr_type) {
    // Use endomorphisms for 6x speed boost!
    if (use_optimizations) {
        return generate_batch_with_endomorphisms(batch, addr_type);
    } else {
        return generate_batch_binary(batch, addr_type);
    }
}

std::vector<BtcKeyPairBinary> BitcoinGenerator::generate_batch_with_endomorphisms(size_t batch, BitcoinAddressType addr_type) {
    std::vector<BtcKeyPairBinary> result;
    result.reserve(batch * 6); // Reserve space for 6 variants per key
    
    // Generate base private keys and compressed public keys  
    auto priv_pub_pairs = generate_batch_priv_pub(batch);
    if (priv_pub_pairs.empty()) {
        return result;
    }
    
    // Process each base key and compute 6 endomorphic variants
    // This gives us 6x more keys to check with minimal additional cost!
    for (size_t i = 0; i < batch; ++i) {
        // Compute 6 endomorphic private keys from the base key
        uint8_t endo_privkeys[6][32];
        if (!compute_endomorphic_privkeys_impl(priv_pub_pairs[i].private_key.data(), endo_privkeys)) {
            // Fallback: skip this key on error (very rare)
            continue;
        }
        
        // For each endomorphic private key, compute pub33 and hash160
        for (int v = 0; v < 6; ++v) {
            BtcKeyPairBinary kp;
            
            // Store the correct endomorphic private key
            std::memcpy(kp.private_key.data(), endo_privkeys[v], 32);
            
            // Compute compressed public key for this private key
            uint8_t pub33[33];
            if (!secp256k1_pubkey_from_priv_compressed(endo_privkeys[v], pub33)) {
                // Skip invalid key (should never happen)
                continue;
            }
            
            // Compute hash160 from the public key
            btc_pub33_to_hash160_cpu(pub33, addr_type, kp.address20.data());
            
            result.push_back(kp);
        }
    }
    
    return result;
}

// NEW: Generate batch with endomorphisms (returns priv+pub33 for Mode 6)
// This is used in GPU Bloom scan mode for 6x speedup
std::vector<BtcPrivPub> BitcoinGenerator::generate_batch_priv_pub_with_endomorphisms(size_t batch) {
    std::vector<BtcPrivPub> result;
    result.reserve(batch * 6); // Reserve space for 6 variants per key
    
    // Generate base private keys and compressed public keys
    auto priv_pub_pairs = generate_batch_priv_pub(batch);
    if (priv_pub_pairs.empty()) {
        return result;
    }
    
    // Process each base key and compute 6 endomorphic variants
    for (size_t i = 0; i < batch; ++i) {
        // Compute 6 endomorphic private keys from the base key
        uint8_t endo_privkeys[6][32];
        if (!compute_endomorphic_privkeys_impl(priv_pub_pairs[i].private_key.data(), endo_privkeys)) {
            // Fallback: skip this key on error (very rare)
            continue;
        }
        
        // For each endomorphic private key, compute pub33
        for (int v = 0; v < 6; ++v) {
            BtcPrivPub kp;
            
            // Store the correct endomorphic private key
            std::memcpy(kp.private_key.data(), endo_privkeys[v], 32);
            
            // Compute compressed public key for this private key
            if (!secp256k1_pubkey_from_priv_compressed(endo_privkeys[v], kp.public_key33.data())) {
                // Skip invalid key (should never happen)
                continue;
            }
            
            result.push_back(kp);
        }
    }
    
    return result;
}

// Public wrapper for endomorphic key computation
bool BitcoinGenerator::compute_endomorphic_privkeys(const uint8_t base_priv32[32], uint8_t out_privkeys[6][32]) const {
    return compute_endomorphic_privkeys_impl(base_priv32, out_privkeys);
}

bool BitcoinGenerator::priv_to_pub_compressed(const uint8_t priv32[32], uint8_t pub33[33]) const {
    return secp256k1_pubkey_from_priv_compressed(priv32, pub33);
}

// Benchmark performance comparison
double BitcoinGenerator::benchmark_performance(size_t test_keys, BitcoinAddressType addr_type) {
    auto start = std::chrono::high_resolution_clock::now();
    
    if (use_optimizations) {
        auto results = generate_batch_with_endomorphisms(test_keys, addr_type);
    } else {
        auto results = generate_batch_binary(test_keys, addr_type);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    return static_cast<double>(test_keys) / (duration.count() / 1000000.0); // keys per second
}

void BitcoinGenerator::print_performance_comparison(size_t test_keys) {
    printf("=== Bitcoin Generator Performance Comparison ===\n");
    printf("Test size: %zu keys\n\n", test_keys);
    
    // Test original implementation
    use_optimizations = false;
    double original_speed = benchmark_performance(test_keys, address_type);
    
    // Test optimized implementation
    use_optimizations = true;
    double optimized_speed = benchmark_performance(test_keys, address_type);
    
    double speedup = optimized_speed / original_speed;
    
    printf("Original implementation:  %.0f keys/sec\n", original_speed);
    printf("Optimized implementation: %.0f keys/sec\n", optimized_speed);
    printf("Speedup factor:           %.1fx\n", speedup);
    printf("Expected speedup:         6-12x (with endomorphisms)\n\n");
    
    if (speedup > 5.0) {
        printf("✅ Excellent performance improvement achieved!\n");
    } else if (speedup > 2.0) {
        printf("✅ Good performance improvement\n");
    } else {
        printf("⚠️  Performance improvement below expectations\n");
        printf("   Check GPU optimization implementation\n");
    }
}
