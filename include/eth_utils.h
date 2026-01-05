#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Keccak-256 (Ethereum) helpers and GPU wrappers.
// These utilities are lightweight and reuse existing third-party Keccak
// implementation and the existing GPU priv->pub64 kernel.

// Compute Keccak-256 hash of arbitrary bytes.
void eth_keccak256(const uint8_t* data, size_t len, uint8_t out32[32]);

// Derive Ethereum address (last 20 bytes of Keccak-256 of uncompressed pubkey).
// pub64 must be 64 bytes (X||Y big-endian, no 0x04 prefix).
void eth_pub64_to_addr20(const uint8_t* pub64, uint8_t out20[20]);

// Parse Ethereum address from:
//  - raw blob of length 20
//  - hex string with optional "0x" prefix (expects 40 hex chars)
bool eth_parse_address_bytes_or_hex(const void* data, int len, std::array<uint8_t, 20>& out);
bool eth_parse_address_hex(const std::string& text, std::array<uint8_t, 20>& out);

// Format address to lowercase hex (40 chars, no prefix).
std::string eth_addr20_to_hex(const uint8_t addr[20]);

// GPU helper: convert batch of priv32 (num * 32 bytes) to pub64 (num * 64 bytes).
// Returns false on CUDA failure.
bool eth_priv_to_pub64_gpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub64);

// CPU fallback: priv32 -> pub64 using OpenSSL; returns false on error.
bool eth_priv_to_pub64_cpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub64);

// GPU helper: convert batch of pub64 (num * 64 bytes) to addr20 (num * 20 bytes).
// Returns false on CUDA failure.
bool eth_pub64_to_addr20_gpu(const uint8_t* pub64, size_t num, std::vector<uint8_t>& out_addr20);


