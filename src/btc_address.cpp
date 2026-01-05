#include "btc_address.h"
#include "conversion_utils.h"
#include <vector>
#include <array>
#include <cstring>

// SHA256 and RIPEMD160 via OpenSSL
#include <openssl/sha.h>
#include <openssl/ripemd.h>

static void sha256(const uint8_t* in, size_t len, uint8_t out32[32]) { SHA256(in, len, out32); }
static void ripemd160(const uint8_t* in, size_t len, uint8_t out20[20]) { RIPEMD160(in, len, out20); }

static std::string base58check(uint8_t version, const uint8_t* payload, size_t payload_len) {
    std::vector<uint8_t> buf;
    buf.reserve(1 + payload_len + 4);
    buf.push_back(version);
    buf.insert(buf.end(), payload, payload + payload_len);
    uint8_t h1[32], h2[32];
    sha256(buf.data(), buf.size(), h1);
    sha256(h1, 32, h2);
    buf.insert(buf.end(), h2, h2 + 4);
    return ConversionUtils::base58_encode(buf);
}

std::string btc_p2pkh_from_hash160(const uint8_t hash160[20]) {
    return base58check(0x00, hash160, 20);
}

std::string btc_p2sh_from_hash160(const uint8_t hash160[20]) {
    return base58check(0x05, hash160, 20);
}

// Minimal Bech32 (v0) encoder for P2WPKH
// Using a tiny implementation sufficient for v0/20-byte program
static const char* bech32_chars = "qpzry9x8gf2tvdw0s3jn54khce6mua7l";

static uint32_t bech32_polymod(const std::vector<uint8_t>& v) {
    uint32_t chk = 1;
    for (uint8_t x : v) {
        uint8_t b = chk >> 25;
        chk = (chk & 0x1ffffff) << 5 ^ x;
        if (b & 1) chk ^= 0x3b6a57b2;
        if (b & 2) chk ^= 0x26508e6d;
        if (b & 4) chk ^= 0x1ea119fa;
        if (b & 8) chk ^= 0x3d4233dd;
        if (b & 16) chk ^= 0x2a1462b3;
    }
    return chk;
}

static std::vector<uint8_t> bech32_hrp_expand(const std::string& hrp) {
    std::vector<uint8_t> ret;
    ret.reserve(hrp.size() * 2 + 1);
    for (char c : hrp) ret.push_back((uint8_t)(c >> 5));
    ret.push_back(0);
    for (char c : hrp) ret.push_back((uint8_t)(c & 31));
    return ret;
}

static std::string bech32_encode(const std::string& hrp, const std::vector<uint8_t>& data) {
    std::vector<uint8_t> values = bech32_hrp_expand(hrp);
    values.insert(values.end(), data.begin(), data.end());
    values.insert(values.end(), {0,0,0,0,0,0});
    uint32_t pm = bech32_polymod(values) ^ 1;
    for (int i = 0; i < 6; ++i) {
        values.push_back((pm >> (5 * (5 - i))) & 31);
    }
    std::string out = hrp + '1';
    for (uint8_t x : data) out.push_back(bech32_chars[x]);
    for (int i = 0; i < 6; ++i) out.push_back(bech32_chars[values[values.size() - 6 + i]]);
    return out;
}

static std::vector<uint8_t> convert_bits_8_to_5(const uint8_t* in, size_t in_len) {
    std::vector<uint8_t> out;
    out.reserve((in_len * 8 + 4) / 5);
    int acc = 0, bits = 0;
    for (size_t i = 0; i < in_len; ++i) {
        acc = (acc << 8) | in[i];
        bits += 8;
        while (bits >= 5) {
            bits -= 5;
            out.push_back((acc >> bits) & 31);
        }
    }
    if (bits > 0) out.push_back((acc << (5 - bits)) & 31);
    return out;
}

std::string btc_p2wpkh_bech32_from_hash160(const uint8_t hash160[20]) {
    std::vector<uint8_t> data;
    data.reserve(1 + ((20 * 8 + 4) / 5));
    data.push_back(0); // witness version 0
    auto prog5 = convert_bits_8_to_5(hash160, 20);
    data.insert(data.end(), prog5.begin(), prog5.end());
    return bech32_encode("bc", data);
}


