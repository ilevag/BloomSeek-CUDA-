#pragma once

#include <string>
#include <cstdint>

// Create Base58Check P2PKH address (version 0x00) from HASH160(pubkey)
std::string btc_p2pkh_from_hash160(const uint8_t hash160[20]);

// Create Base58Check P2SH address (version 0x05) from HASH160(redeemScript)
std::string btc_p2sh_from_hash160(const uint8_t hash160[20]);

// Create Bech32 P2WPKH address (hrp "bc", witness v=0) from HASH160(pubkey)
std::string btc_p2wpkh_bech32_from_hash160(const uint8_t hash160[20]);


