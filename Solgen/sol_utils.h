#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Solana (ed25519) helpers: address parsing/formatting and key conversion.

// Convert Solana base58 address to 32-byte pubkey; returns false on invalid.
bool sol_address_to_pub32(const std::string& address, std::array<uint8_t, 32>& out);

// Convert 32-byte pubkey to base58 Solana address.
std::string sol_pub32_to_address(const uint8_t pub32[32]);

// Parse address from either 32-byte blob or base58 text (common for DB import).
bool sol_parse_address_bytes_or_base58(const void* data, int len, std::array<uint8_t, 32>& out);

// Legacy aliases kept for backward compatibility.
bool sol_parse_address_b58(const std::string& txt, std::array<uint8_t, 32>& out);
bool sol_parse_address_bytes_or_b58(const void* data, int len, std::array<uint8_t, 32>& out);
std::string sol_addr32_to_b58(const uint8_t addr[32]);

// CPU ed25519: priv(seed32) -> pub32; returns false on error.
bool sol_priv_to_pub32_cpu(const uint8_t priv32[32], uint8_t pub32[32]);
bool sol_priv_to_pub_cpu(const uint8_t priv32[32], uint8_t pub32[32]);

// Batch CPU variant.
bool sol_priv_to_pub32_cpu_batch(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32);

// GPU path (implemented in sol_gpu_ed25519.cu); returns false if not available.
bool sol_priv_to_pub32_gpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32);
bool sol_priv_to_pub_gpu(const uint8_t* priv32, size_t num, std::vector<uint8_t>& out_pub32);

// Generate batch seeds and public keys (CPU).
struct SolPrivPub {
    std::array<uint8_t, 32> priv;
    std::array<uint8_t, 32> pub;
};

std::vector<SolPrivPub> sol_generate_batch_cpu(size_t batch);

// GPU: generate priv on device, derive pub, copy both to host buffers.
bool sol_generate_priv_pub_gpu(size_t num,
                               uint64_t seed_base,
                               std::vector<uint8_t>& out_priv32,
                               std::vector<uint8_t>& out_pub32);

