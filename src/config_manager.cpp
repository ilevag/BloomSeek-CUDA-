#include "config_manager.h"
#include "logger.h"
#include <iostream>
#include <sstream>

ConfigManager::ConfigManager(const std::string& config_file) : config_file_path(config_file) {
    if (!load_config()) {
        create_default_config();
    }
}

bool ConfigManager::load_config() {
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            // Remove leading/trailing whitespace
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            config_data[key] = value;
        }
    }
    
    return true;
}

bool ConfigManager::save_config() {
    std::ofstream file(config_file_path);
    if (!file.is_open()) {
        return false;
    }
    
    file << "# Solana/Ethereum GPU Generator Configuration\n";
    file << "# Lines starting with '#' are comments. Key=Value format.\n";
    file << "# Directories:\n";
    file << "#  database_directory: path to SQLite databases\n";
    file << "#  bloom_filter_directory: path to Bloom filters\n\n";

    file << "# First run flags:\n";
    file << "#  first_run_completed: internal marker for initial setup\n";
    file << "#  gpu_autotune_done: whether GPU params were auto-tuned on first run\n\n";

    file << "# GPU tuning (override auto if set):\n";
    file << "#  gpu_threads_per_block: CUDA threads per block (e.g., 256/512)\n";
    file << "#  gpu_blocks_per_grid: CUDA blocks per grid (e.g., SM*8)\n";
    file << "#  gpu_batch_size: number of keys per GPU batch (affects VRAM)\n";
    file << "#  eth_batch_size: number of ETH priv/pub per CPU batch (for Mode 6)\n\n";

    for (const auto& pair : config_data) {
        file << pair.first << "=" << pair.second << "\n";
    }
    
    return true;
}

void ConfigManager::create_default_config() {
    config_data["database_directory"] = "./databases";
    config_data["bloom_filter_directory"] = "./bloom_filters";
    config_data["first_run_completed"] = "false";
    // GPU tuning defaults (auto on first run)
    config_data["gpu_autotune_done"] = "false";
    // Leave unset to enable auto; will be filled after autotune
    // config_data["gpu_threads_per_block"]
    // config_data["gpu_blocks_per_grid"]
    // config_data["gpu_batch_size"]
    config_data["eth_batch_size"] = "65536";
    save_config();
}

std::string ConfigManager::get_database_directory() const {
    auto it = config_data.find("database_directory");
    return (it != config_data.end()) ? it->second : "./databases";
}

std::string ConfigManager::get_bloom_filter_directory() const {
    auto it = config_data.find("bloom_filter_directory");
    return (it != config_data.end()) ? it->second : "./bloom_filters";
}

bool ConfigManager::is_first_run() const {
    auto it = config_data.find("first_run_completed");
    return (it == config_data.end()) || (it->second == "false");
}

void ConfigManager::set_database_directory(const std::string& path) {
    config_data["database_directory"] = path;
    save_config();
}

void ConfigManager::set_bloom_filter_directory(const std::string& path) {
    config_data["bloom_filter_directory"] = path;
    save_config();
}

void ConfigManager::set_first_run_completed() {
    config_data["first_run_completed"] = "true";
    save_config();
}

bool ConfigManager::ensure_directories_exist() {
    try {
        std::filesystem::create_directories(get_database_directory());
        std::filesystem::create_directories(get_bloom_filter_directory());
        return true;
    } catch (const std::exception& e) {
        if (g_logger) {
            g_logger->error("Failed to create directories: " + std::string(e.what()));
        }
        return false;
    }
}

void ConfigManager::prompt_for_directories() {
    std::string db_path, bloom_path;
    
    std::cout << "=== First Run Setup ===" << std::endl;
    std::cout << "Please configure directory paths for data storage." << std::endl << std::endl;
    
    std::cout << "Enter database directory path (default: ./databases): ";
    std::getline(std::cin, db_path);
    if (db_path.empty()) {
        db_path = "./databases";
    }
    
    std::cout << "Enter bloom filter directory path (default: ./bloom_filters): ";
    std::getline(std::cin, bloom_path);
    if (bloom_path.empty()) {
        bloom_path = "./bloom_filters";
    }
    
    set_database_directory(db_path);
    set_bloom_filter_directory(bloom_path);
    set_first_run_completed();
    
    std::cout << std::endl << "Configuration saved!" << std::endl;
    std::cout << "Database directory: " << db_path << std::endl;
    std::cout << "Bloom filter directory: " << bloom_path << std::endl;
    std::cout << std::endl;
    
    if (!ensure_directories_exist()) {
        std::cout << "Warning: Failed to create some directories!" << std::endl;
    }
}

// ===== Helpers =====
bool ConfigManager::str_to_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}
std::string ConfigManager::bool_to_str(bool v) { return v ? "true" : "false"; }
long long ConfigManager::str_to_ll(const std::string& v, long long def) {
    try { return std::stoll(v); } catch (...) { return def; }
}
std::string ConfigManager::ll_to_str(long long v) { return std::to_string(v); }

// ===== GPU tuning getters/setters =====
bool ConfigManager::is_gpu_autotune_done() const {
    auto it = config_data.find("gpu_autotune_done");
    return (it != config_data.end()) && str_to_bool(it->second);
}
void ConfigManager::set_gpu_autotune_done(bool done) {
    config_data["gpu_autotune_done"] = bool_to_str(done);
    save_config();
}

size_t ConfigManager::get_gpu_threads_per_block() const {
    auto it = config_data.find("gpu_threads_per_block");
    return (it != config_data.end()) ? (size_t)str_to_ll(it->second, 0) : 0;
}
size_t ConfigManager::get_gpu_blocks_per_grid() const {
    auto it = config_data.find("gpu_blocks_per_grid");
    return (it != config_data.end()) ? (size_t)str_to_ll(it->second, 0) : 0;
}
size_t ConfigManager::get_gpu_batch_size() const {
    auto it = config_data.find("gpu_batch_size");
    return (it != config_data.end()) ? (size_t)str_to_ll(it->second, 0) : 0;
}
size_t ConfigManager::get_eth_batch_size() const {
    auto it = config_data.find("eth_batch_size");
    return (it != config_data.end()) ? (size_t)str_to_ll(it->second, 65536) : 65536;
}

void ConfigManager::set_gpu_threads_per_block(size_t v) {
    config_data["gpu_threads_per_block"] = ll_to_str((long long)v);
    save_config();
}
void ConfigManager::set_gpu_blocks_per_grid(size_t v) {
    config_data["gpu_blocks_per_grid"] = ll_to_str((long long)v);
    save_config();
}
void ConfigManager::set_gpu_batch_size(size_t v) {
    config_data["gpu_batch_size"] = ll_to_str((long long)v);
    save_config();
}
void ConfigManager::set_eth_batch_size(size_t v) {
    config_data["eth_batch_size"] = ll_to_str((long long)v);
    save_config();
}


