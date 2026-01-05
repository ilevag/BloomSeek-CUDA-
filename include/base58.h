#pragma once

#include <string>
#include <vector>
#include <cstdint>

class Base58 {
public:
    // Кодирование массива байт в Base58 строку
    static std::string encode(const std::vector<uint8_t>& data);
    static std::string encode(const uint8_t* data, size_t size);
    
    // Декодирование Base58 строки в массив байт
    static std::vector<uint8_t> decode(const std::string& str);
    
    // Проверка валидности Base58 строки
    static bool is_valid(const std::string& str);
    
private:
    static const char* alphabet;
    static const int8_t decode_table[128];
    
    static void init_decode_table();
};