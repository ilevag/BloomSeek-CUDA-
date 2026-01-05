#pragma once

#include <string>
#include <fstream>
#include <map>
#include <filesystem>

class ConfigManager {
private:
    std::string config_file_path;
    std::map<std::string, std::string> config_data;
    
    bool load_config();
    bool save_config();
    void create_default_config();
    // Helpers
    static bool str_to_bool(const std::string& v);
    static std::string bool_to_str(bool v);
    static long long str_to_ll(const std::string& v, long long def);
    static std::string ll_to_str(long long v);
    
public:
    ConfigManager(const std::string& config_file = "config.ini");
    
    // Configuration methods
    std::string get_database_directory() const;
    std::string get_bloom_filter_directory() const;
    bool is_first_run() const;
    
    void set_database_directory(const std::string& path);
    void set_bloom_filter_directory(const std::string& path);
    void set_first_run_completed();
    
    // Utility methods
    bool ensure_directories_exist();
    void prompt_for_directories();

    // GPU tuning config
    bool is_gpu_autotune_done() const;
    void set_gpu_autotune_done(bool done);

    size_t get_gpu_threads_per_block() const;
    size_t get_gpu_blocks_per_grid() const;
    size_t get_gpu_batch_size() const;
    size_t get_eth_batch_size() const;

    void set_gpu_threads_per_block(size_t v);
    void set_gpu_blocks_per_grid(size_t v);
    void set_gpu_batch_size(size_t v);
    void set_eth_batch_size(size_t v);
};


