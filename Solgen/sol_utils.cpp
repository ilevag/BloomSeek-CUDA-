#include "sol_utils.h"

#include <algorithm>
#include <cstring>
#include <openssl/rand.h>

#include "conversion_utils.h"

#ifndef __CUDACC__
#undef __host__
#undef __device__
#undef __global__
#undef __forceinline__
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#endif

#include "cuda-ecc-ed25519/ed25519.h"
#include "cuda-ecc-ed25519/ge.h"
#include "cuda-ecc-ed25519/sha512.h"

bool sol_address_to_pub32(const std::string& address, std::array<uint8_t, 32>& out) {
    auto bytes = ConversionUtils::solana_address_to_bytes(address);
    if (bytes.size() != 32) return false;
    std::memcpy(out.data(), bytes.data(), 32);
    return true;
}

std::string sol_pub32_to_address(const uint8_t pub32[32]) {
    return ConversionUtils::bytes_to_solana_address(std::vector<uint8_t>(pub32, pub32 + 32));
}

bool sol_parse_address_b58(const std::string& txt, std::array<uint8_t, 32>& out) {
    auto decoded = ConversionUtils::base58_decode(txt);
    if (decoded.size() != 32) return false;
    std::copy(decoded.begin(), decoded.end(), out.begin());
    return true;
}

bool sol_parse_address_bytes_or_b58(const void* data, int len, std::array<uint8_t, 32>& out) {
    if (!data || len <= 0) return false;
    if (len == 32) {
        std::memcpy(out.data(), data, 32);
        return true;
    }
    const char* cstr = reinterpret_cast<const char*>(data);
    std::string s(cstr, static_cast<size_t>(len));
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
        s.pop_back();
    }
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start != std::string::npos && start > 0) s = s.substr(start);
    return sol_parse_address_b58(s, out);
}

bool sol_parse_address_bytes_or_base58(const void* data, int len, std::array<uint8_t, 32>& out) {
    if (!data || len <= 0) return false;
    if (len == 32) {
        std::memcpy(out.data(), data, 32);
        return true;
    }
    const char* cstr = reinterpret_cast<const char*>(data);
    std::string s(cstr, static_cast<size_t>(len));
    while (!s.empty() && (s.back() == '\0' || s.back() == '\n' || s.back() == '\r' || s.back() == ' ')) {
        s.pop_back();
    }
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return false;
    if (start > 0) s = s.substr(start);
    return sol_address_to_pub32(s, out);
}

std::string sol_addr32_to_b58(const uint8_t addr[32]) {
    return ConversionUtils::base58_encode(std::vector<uint8_t>(addr, addr + 32));
}

bool sol_priv_to_pub_cpu(const uint8_t priv32[32], uint8_t pub32[32]) {
    if (!priv32 || !pub32) return false;
    uint8_t priv64[64];
    ed25519_create_keypair(pub32, priv64, priv32);
    return true;
}

bool sol_priv_to_pub32_cpu(const uint8_t priv32[32], uint8_t pub32[32]) {
    uint8_t priv64[64];
    ed25519_create_keypair(pub32, priv64, priv32);
    return true;
}

bool sol_priv_to_pub32_cpu_batch(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32) {
    if (!priv32 || num == 0) return false;
    out_pub32.resize(num * 32);
    for (size_t i = 0; i < num; ++i) {
        if (!sol_priv_to_pub32_cpu(&priv32[i * 32], &out_pub32[i * 32])) return false;
    }
    return true;
}

std::vector<SolPrivPub> sol_generate_batch_cpu(size_t batch) {
    std::vector<SolPrivPub> res;
    res.reserve(batch);
    std::vector<uint8_t> seeds(batch * 32);
    RAND_bytes(seeds.data(), static_cast<int>(seeds.size()));
    for (size_t i = 0; i < batch; ++i) {
        SolPrivPub item{};
        std::memcpy(item.priv.data(), &seeds[i * 32], 32);
        // Do not compute pub on CPU here; GPU path will derive pub for speed.
        res.push_back(item);
    }
    return res;
}

