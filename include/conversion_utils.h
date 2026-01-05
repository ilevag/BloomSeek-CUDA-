#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace ConversionUtils {

// Base58 conversion functions
std::vector<uint8_t> base58_decode(const std::string& encoded);
std::string base58_encode(const std::vector<uint8_t>& data);

// Hex conversion functions  
std::vector<uint8_t> hex_decode(const std::string& hex_string);
std::string hex_encode(const std::vector<uint8_t>& data);

// Solana address conversion (Base58 <-> Binary)
std::vector<uint8_t> solana_address_to_bytes(const std::string& address);
std::string bytes_to_solana_address(const std::vector<uint8_t>& bytes);

// Private key conversion (Hex <-> Binary)
std::vector<uint8_t> private_key_to_bytes(const std::string& hex_key);
std::string bytes_to_private_key(const std::vector<uint8_t>& bytes);

// Validation functions
bool is_valid_solana_address(const std::string& address);
bool is_valid_private_key_hex(const std::string& hex_key);

}





