#include "conversion_utils.h"
#include "logger.h"
#include <algorithm>
#include <sstream>
#include <iomanip>

namespace ConversionUtils {

// Base58 alphabet используемый в Solana
static const std::string BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

std::vector<uint8_t> base58_decode(const std::string& encoded) {
    if (encoded.empty()) {
        return {};
    }
    
    // Создаем таблицу декодирования
    int decode_table[256];
    std::fill(decode_table, decode_table + 256, -1);
    for (size_t i = 0; i < BASE58_ALPHABET.size(); i++) {
        decode_table[static_cast<uint8_t>(BASE58_ALPHABET[i])] = static_cast<int>(i);
    }
    
    // Подсчитываем ведущие нули
    size_t leading_zeros = 0;
    for (char c : encoded) {
        if (c == '1') leading_zeros++;
        else break;
    }
    
    // Декодируем в big integer
    std::vector<uint8_t> result;
    result.reserve(encoded.length() * 733 / 1000 + 1); // log(58) / log(256)
    
    for (char c : encoded) {
        int val = decode_table[static_cast<uint8_t>(c)];
        if (val == -1) {
            LOG_ERROR("Invalid Base58 character: " + std::string(1, c));
            return {};
        }
        
        // Умножаем результат на 58 и добавляем текущую цифру
        int carry = val;
        for (size_t i = 0; i < result.size(); i++) {
            carry += static_cast<int>(result[i]) * 58;
            result[i] = static_cast<uint8_t>(carry & 0xFF);
            carry >>= 8;
        }
        
        while (carry > 0) {
            result.push_back(static_cast<uint8_t>(carry & 0xFF));
            carry >>= 8;
        }
    }
    
    // Добавляем ведущие нули
    result.resize(result.size() + leading_zeros, 0);
    
    // Разворачиваем (был little-endian, нужен big-endian)
    std::reverse(result.begin(), result.end());
    
    return result;
}

std::string base58_encode(const std::vector<uint8_t>& data) {
    if (data.empty()) {
        return "";
    }
    
    // Подсчитываем ведущие нули
    size_t leading_zeros = 0;
    for (uint8_t byte : data) {
        if (byte == 0) leading_zeros++;
        else break;
    }
    
    // Копируем данные для обработки
    std::vector<uint8_t> temp(data.begin() + leading_zeros, data.end());
    
    std::string result;
    result.reserve(data.size() * 138 / 100 + 1); // log(256) / log(58)
    
    while (!temp.empty()) {
        // Делим на 58
        int remainder = 0;
        bool non_zero = false;
        
        for (size_t i = 0; i < temp.size(); i++) {
            int current = remainder * 256 + temp[i];
            temp[i] = static_cast<uint8_t>(current / 58);
            remainder = current % 58;
            
            if (temp[i] != 0) {
                non_zero = true;
            }
        }
        
        // Удаляем ведущие нули из temp
        if (!non_zero) {
            temp.clear();
        } else {
            while (!temp.empty() && temp[0] == 0) {
                temp.erase(temp.begin());
            }
        }
        
        result += BASE58_ALPHABET[remainder];
    }
    
    // Добавляем ведущие '1' для ведущих нулевых байтов
    result.append(leading_zeros, '1');
    
    // Разворачиваем результат
    std::reverse(result.begin(), result.end());
    
    return result;
}

std::vector<uint8_t> hex_decode(const std::string& hex_string) {
    if (hex_string.length() % 2 != 0) {
        LOG_ERROR("Invalid hex string length: " + hex_string);
        return {};
    }
    
    std::vector<uint8_t> result;
    result.reserve(hex_string.length() / 2);
    
    for (size_t i = 0; i < hex_string.length(); i += 2) {
        std::string byte_str = hex_string.substr(i, 2);
        char* end_ptr;
        unsigned long byte_val = std::strtoul(byte_str.c_str(), &end_ptr, 16);
        
        if (end_ptr != byte_str.c_str() + 2) {
            LOG_ERROR("Invalid hex byte: " + byte_str);
            return {};
        }
        
        result.push_back(static_cast<uint8_t>(byte_val));
    }
    
    return result;
}

std::string hex_encode(const std::vector<uint8_t>& data) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    
    for (uint8_t byte : data) {
        ss << std::setw(2) << static_cast<unsigned>(byte);
    }
    
    return ss.str();
}

std::vector<uint8_t> solana_address_to_bytes(const std::string& address) {
    if (!is_valid_solana_address(address)) {
        LOG_ERROR("Invalid Solana address format: " + address);
        return {};
    }
    
    auto decoded = base58_decode(address);
    if (decoded.size() != 32) {
        LOG_ERROR("Solana address should be 32 bytes, got: " + std::to_string(decoded.size()));
        return {};
    }
    
    return decoded;
}

std::string bytes_to_solana_address(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 32) {
        LOG_ERROR("Solana address bytes should be 32, got: " + std::to_string(bytes.size()));
        return "";
    }
    
    return base58_encode(bytes);
}

std::vector<uint8_t> private_key_to_bytes(const std::string& hex_key) {
    if (!is_valid_private_key_hex(hex_key)) {
        LOG_ERROR("Invalid private key hex format: " + hex_key);
        return {};
    }
    
    auto decoded = hex_decode(hex_key);
    if (decoded.size() != 32) {
        LOG_ERROR("Private key should be 32 bytes, got: " + std::to_string(decoded.size()));
        return {};
    }
    
    return decoded;
}

std::string bytes_to_private_key(const std::vector<uint8_t>& bytes) {
    if (bytes.size() != 32) {
        LOG_ERROR("Private key bytes should be 32, got: " + std::to_string(bytes.size()));
        return "";
    }
    
    return hex_encode(bytes);
}

bool is_valid_solana_address(const std::string& address) {
    // Solana адреса обычно 32-44 символа в Base58
    if (address.length() < 32 || address.length() > 44) {
        return false;
    }
    
    // Проверяем что все символы из Base58 алфавита
    for (char c : address) {
        if (BASE58_ALPHABET.find(c) == std::string::npos) {
            return false;
        }
    }
    
    // Пробуем декодировать
    auto decoded = base58_decode(address);
    return decoded.size() == 32;
}

bool is_valid_private_key_hex(const std::string& hex_key) {
    // Приватный ключ должен быть 64 символа (32 байта * 2)
    if (hex_key.length() != 64) {
        return false;
    }
    
    // Проверяем что все символы hex
    for (char c : hex_key) {
        if (!std::isxdigit(c)) {
            return false;
        }
    }
    
    return true;
}

}





