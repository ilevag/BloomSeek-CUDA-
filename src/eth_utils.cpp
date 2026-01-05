#include "eth_utils.h"

#include <algorithm>
#include <cstring>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/bn.h>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "conversion_utils.h"
#include "eth_secp256k1_gpu.h"
#include "eth-vanity-cuda/src/cpu_keccak.h"

namespace {
// Convert _uint256 (8x uint32) returned by cpu_full_keccak to 32 bytes (big-endian)
inline void uint256_to_bytes_be(const _uint256& v, uint8_t out[32]) {
    uint32_t words[8] = {v.a, v.b, v.c, v.d, v.e, v.f, v.g, v.h};
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = static_cast<uint8_t>((words[i] >> 24) & 0xFF);
        out[i * 4 + 1] = static_cast<uint8_t>((words[i] >> 16) & 0xFF);
        out[i * 4 + 2] = static_cast<uint8_t>((words[i] >> 8) & 0xFF);
        out[i * 4 + 3] = static_cast<uint8_t>(words[i] & 0xFF);
    }
}

// Validate private key is in [1, n-1] for secp256k1.
inline bool secp256k1_priv_valid(const uint8_t priv[32]) {
    static EC_GROUP* group = EC_GROUP_new_by_curve_name(NID_secp256k1);
    if (!group) return false;
    const BIGNUM* order = EC_GROUP_get0_order(group);
    if (!order) return false;
    bool ok = false;
    BIGNUM* bn = BN_bin2bn(priv, 32, nullptr);
    if (!bn) return false;
    if (!BN_is_zero(bn) && BN_cmp(bn, order) < 0) {
        ok = true;
    }
    BN_free(bn);
    return ok;
}
} // namespace

void eth_keccak256(const uint8_t* data, size_t len, uint8_t out32[32]) {
    // cpu_full_keccak expects mutable pointer, but does not modify contents.
    _uint256 digest = cpu_full_keccak(const_cast<uint8_t*>(data), static_cast<uint32_t>(len));
    uint256_to_bytes_be(digest, out32);
}

void eth_pub64_to_addr20(const uint8_t* pub64, uint8_t out20[20]) {
    uint8_t hash[32];
    eth_keccak256(pub64, 64, hash);
    std::memcpy(out20, hash + 12, 20); // take last 20 bytes
}

bool eth_parse_address_hex(const std::string& text, std::array<uint8_t, 20>& out) {
    std::string t = text;
    // Strip optional 0x/0X prefix
    if (t.rfind("0x", 0) == 0 || t.rfind("0X", 0) == 0) {
        t = t.substr(2);
    }
    if (t.size() != 40) return false;
    auto bytes = ConversionUtils::hex_decode(t);
    if (bytes.size() != 20) return false;
    std::copy(bytes.begin(), bytes.end(), out.begin());
    return true;
}

bool eth_parse_address_bytes_or_hex(const void* data, int len, std::array<uint8_t, 20>& out) {
    if (!data || len <= 0) return false;
    if (len == 20) {
        std::memcpy(out.data(), data, 20);
        return true;
    }
    // Treat as text (hex) if len >= 40
    const char* cstr = reinterpret_cast<const char*>(data);
    std::string s(cstr, static_cast<size_t>(len));
    // Trim trailing nulls/newlines/spaces that may come from SQLite text blobs
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
        s.pop_back();
    }
    // Trim leading spaces
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return false;
    if (start > 0) s = s.substr(start);
    return eth_parse_address_hex(s, out);
}

std::string eth_addr20_to_hex(const uint8_t addr[20]) {
    return ConversionUtils::hex_encode(std::vector<uint8_t>(addr, addr + 20));
}

bool eth_priv_to_pub64_gpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub64) {
    if (!priv32 || num == 0) return false;
    for (size_t i = 0; i < num; ++i) {
        if (!secp256k1_priv_valid(&priv32[i * 32])) return false;
    }
    out_pub64.resize(num * 64);
    if (gpu_secp256k1_priv_to_pub64(priv32, num, out_pub64.data())) {
        return true;
    }
    // GPU failed; fall back to CPU.
    return eth_priv_to_pub64_cpu(priv32, num, out_pub64);
}

bool eth_priv_to_pub64_cpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub64) {
    if (!priv32 || num == 0) return false;
    out_pub64.resize(num * 64);

    auto to_uncompressed = [](const uint8_t in_priv[32], uint8_t out_pub64[64]) -> bool {
        bool ok = false;
        EC_KEY* key = nullptr;
        BIGNUM* bn = nullptr;
        const EC_GROUP* grp = nullptr;
        EC_POINT* pub = nullptr;
        do {
            key = EC_KEY_new_by_curve_name(NID_secp256k1);
            if (!key) break;
            grp = EC_KEY_get0_group(key);
            if (!grp) break;
            bn = BN_bin2bn(in_priv, 32, nullptr);
            if (!bn) break;
            if (EC_KEY_set_private_key(key, bn) != 1) break;
            pub = EC_POINT_new(grp);
            if (!pub) break;
            if (EC_POINT_mul(grp, pub, bn, nullptr, nullptr, nullptr) != 1) break;
            if (EC_KEY_set_public_key(key, pub) != 1) break;
            std::array<uint8_t, 65> uncompressed{};
            size_t len = EC_POINT_point2oct(grp, pub, POINT_CONVERSION_UNCOMPRESSED, uncompressed.data(), uncompressed.size(), nullptr);
            if (len != 65) break;
            std::memcpy(out_pub64, uncompressed.data() + 1, 64); // drop 0x04
            ok = true;
        } while (false);
        if (pub) EC_POINT_free(pub);
        if (bn) BN_free(bn);
        if (key) EC_KEY_free(key);
        return ok;
    };

    for (size_t i = 0; i < num; ++i) {
        if (!secp256k1_priv_valid(&priv32[i * 32])) return false;
        if (!to_uncompressed(&priv32[i * 32], &out_pub64[i * 64])) {
            return false;
        }
    }
    return true;
}

extern "C" bool gpu_eth_pub64_to_addr20(const uint8_t* host_pub64, size_t num, uint8_t* host_addr20);

bool eth_pub64_to_addr20_gpu(const uint8_t* pub64, size_t num, std::vector<uint8_t>& out_addr20) {
    if (!pub64 || num == 0) return false;
    out_addr20.resize(num * 20);
    return gpu_eth_pub64_to_addr20(pub64, num, out_addr20.data());
}


