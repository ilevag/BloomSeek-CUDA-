#include "base58.h"
#include <algorithm>
#include <stdexcept>

const char* Base58::alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

const int8_t Base58::decode_table[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1,  0,  1,  2,  3,  4,  5,  6,  7,  8, -1, -1, -1, -1, -1, -1,
    -1,  9, 10, 11, 12, 13, 14, 15, 16, -1, 17, 18, 19, 20, 21, -1,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, -1, -1, -1, -1, -1,
    -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -1, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, -1, -1, -1, -1, -1
};

std::string Base58::encode(const std::vector<uint8_t>& data) {
    return encode(data.data(), data.size());
}

std::string Base58::encode(const uint8_t* data, size_t size) {
    if (size == 0) return "";
    
    // Подсчитываем ведущие нули
    size_t leading_zeros = 0;
    while (leading_zeros < size && data[leading_zeros] == 0) {
        leading_zeros++;
    }
    
    // Размер выходного буфера (log(256)/log(58) ≈ 1.37)
    size_t encoded_size = (size - leading_zeros) * 138 / 100 + 1;
    std::vector<uint8_t> encoded(encoded_size, 0);
    
    // Преобразуем в базу 58
    for (size_t i = leading_zeros; i < size; i++) {
        int carry = data[i];
        for (int j = encoded_size - 1; j >= 0; j--) {
            carry += 256 * encoded[j];
            encoded[j] = carry % 58;
            carry /= 58;
        }
    }
    
    // Находим первый ненулевой байт
    size_t first_non_zero = 0;
    while (first_non_zero < encoded_size && encoded[first_non_zero] == 0) {
        first_non_zero++;
    }
    
    // Конвертируем в строку
    std::string result;
    result.reserve(leading_zeros + encoded_size - first_non_zero);
    
    // Добавляем '1' для каждого ведущего нуля
    for (size_t i = 0; i < leading_zeros; i++) {
        result += '1';
    }
    
    // Добавляем закодированные символы
    for (size_t i = first_non_zero; i < encoded_size; i++) {
        result += alphabet[encoded[i]];
    }
    
    return result;
}

std::vector<uint8_t> Base58::decode(const std::string& str) {
    if (str.empty()) return {};
    
    // Подсчитываем ведущие '1'
    size_t leading_ones = 0;
    while (leading_ones < str.length() && str[leading_ones] == '1') {
        leading_ones++;
    }
    
    // Размер выходного буфера
    size_t decoded_size = (str.length() - leading_ones) * 733 / 1000 + 1;
    std::vector<uint8_t> decoded(decoded_size, 0);
    
    // Декодируем
    for (size_t i = leading_ones; i < str.length(); i++) {
        char c = str[i];
        if (c < 0 || c >= 128 || decode_table[c] == -1) {
            throw std::invalid_argument("Недопустимый символ в Base58 строке");
        }
        
        int carry = decode_table[c];
        for (int j = decoded_size - 1; j >= 0; j--) {
            carry += 58 * decoded[j];
            decoded[j] = carry % 256;
            carry /= 256;
        }
    }
    
    // Находим первый ненулевой байт
    size_t first_non_zero = 0;
    while (first_non_zero < decoded_size && decoded[first_non_zero] == 0) {
        first_non_zero++;
    }
    
    // Создаем результат
    std::vector<uint8_t> result;
    result.reserve(leading_ones + decoded_size - first_non_zero);
    
    // Добавляем нули для ведущих '1'
    for (size_t i = 0; i < leading_ones; i++) {
        result.push_back(0);
    }
    
    // Добавляем декодированные байты
    for (size_t i = first_non_zero; i < decoded_size; i++) {
        result.push_back(decoded[i]);
    }
    
    return result;
}

bool Base58::is_valid(const std::string& str) {
    for (char c : str) {
        if (c < 0 || c >= 128 || decode_table[c] == -1) {
            return false;
        }
    }
    return true;
}