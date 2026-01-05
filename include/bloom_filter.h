#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>

/**
 * Split Block Bloom Filter (SBBF) реализация
 * Оптимизирован для высокой производительности и совместимости с Parquet
 */
class BloomFilter {
private:
    std::vector<uint8_t> bit_array;    // Массив битов фильтра
    size_t m_bits;                     // Общее количество битов
    size_t m_blocks;                   // Количество блоков (по 256 бит каждый)
    uint8_t k_hashes;                  // Количество хэш-функций (фиксированно 8 для SBBF)
    size_t expected_elements;          // Ожидаемое количество элементов
    double false_positive_rate;        // Заданная вероятность ложных срабатываний
    uint32_t file_version = 2;         // Версия формата файла фильтра (v2 — детерминированные хэши)
    
    // Константы для SBBF
    static constexpr uint8_t SBBF_HASH_FUNCTIONS = 8;
    static constexpr size_t SBBF_BLOCK_SIZE_BITS = 256;
    static constexpr size_t SBBF_BLOCK_SIZE_BYTES = 32;
    
    // Вспомогательные функции хэширования (v1 — по строке, v2 — по байтам)
    // v1 (совместимость): std::hash<string>
    uint64_t hash1_v1(const std::string& item) const;
    uint64_t hash2_v1(const std::string& item) const;
    uint64_t get_hash_v1(const std::string& item, int hash_num) const;

    // v2 (детерминированные, GPU-совместимые): FNV-1a 64 + вариация для второго хэша
    uint64_t fnv1a64(const uint8_t* data, size_t len, uint64_t seed = 0) const;
    uint64_t rotl64(uint64_t x, int r) const;
    uint64_t hash1_v2(const uint8_t* data, size_t len) const;
    uint64_t hash2_v2(const uint8_t* data, size_t len) const;
    uint64_t get_hash_v2(const uint8_t* data, size_t len, int hash_num) const;
    
    // Вычисление параметров фильтра
    size_t calculate_optimal_bits(size_t n, double fpr) const;
    size_t calculate_blocks_count(size_t bits) const;

public:
    /**
     * Конструктор фильтра Блума
     * @param expected_items Ожидаемое количество элементов
     * @param fpr Вероятность ложных срабатываний (например, 0.001 для 0.1%)
     */
    BloomFilter(size_t expected_items, double fpr);
    
    /**
     * Конструктор для загрузки из файла
     */
    BloomFilter();
    
    /**
     * Деструктор
     */
    ~BloomFilter() = default;
    
    /**
     * Добавить элемент в фильтр
     * @param item Строка для добавления
     */
    void add(const std::string& item);
    void add_bytes(const uint8_t* data, size_t len);
    
    /**
     * Проверить может ли элемент содержаться в фильтре
     * @param item Строка для проверки
     * @return true если элемент может быть в множестве, false если точно нет
     */
    bool might_contain(const std::string& item) const;
    bool might_contain_bytes(const uint8_t* data, size_t len) const;
    
    /**
     * Сохранить фильтр в файл
     * @param filename Путь к файлу
     * @return true при успехе
     */
    bool save_to_file(const std::string& filename) const;
    
    /**
     * Загрузить фильтр из файла
     * @param filename Путь к файлу
     * @return true при успехе
     */
    bool load_from_file(const std::string& filename);
    
    /**
     * Получить размер фильтра в байтах
     */
    size_t get_size_bytes() const { return bit_array.size(); }
    
    /**
     * Получить количество битов в фильтре
     */
    size_t get_size_bits() const { return m_bits; }
    
    /**
     * Получить ожидаемую вероятность ложных срабатываний
     */
    double get_false_positive_rate() const { return false_positive_rate; }
    
    /**
     * Получить количество ожидаемых элементов
     */
    size_t get_expected_elements() const { return expected_elements; }
    
    /**
     * Получить количество хэш-функций
     */
    uint8_t get_hash_functions() const { return k_hashes; }
    
    /**
     * Получить статистику фильтра
     */
    struct FilterStats {
        size_t size_bytes;
        size_t size_bits;
        size_t blocks_count;
        uint8_t hash_functions;
        double false_positive_rate;
        size_t expected_elements;
        double bytes_per_element;
    };
    
    FilterStats get_stats() const;
    
    /**
     * Очистить фильтр (обнулить все биты)
     */
    void clear();
    
    /**
     * Проверить инициализирован ли фильтр
     */
    bool is_initialized() const { return !bit_array.empty(); }
    uint32_t get_file_version() const { return file_version; }

    /**
     * Доступ к внутренним данным фильтра для передачи на GPU
     */
    const std::vector<uint8_t>& data() const { return bit_array; }
    size_t get_blocks_count() const { return m_blocks; }
    uint8_t get_hash_function_count() const { return k_hashes; }
};