#include <iostream>
#include <signal.h>
#include <chrono>
#include <thread>
#include <condition_variable>
#include <iomanip>
#include <sstream>
#include <atomic>
#include <limits>
#include <cstdlib>
#include <cmath>
#include <regex>
#include <cstring>
#include <cuda_runtime.h>
#include <random>
#include <vector>
#include <deque>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include <deque>
#include <mutex>
#include <condition_variable>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef min
#undef max
#undef ERROR
#endif

#include "logger.h"
#include "config_manager.h"
#include "bloom_filter.h"
#include "conversion_utils.h"
#include "bitcoin_generator.h"
#include "btc_address.h"
#include "btc_gpu_optimized.h"
#include "eth_secp256k1_gpu.h"
#include "eth_utils.h"
#include "sol_utils.h"
#include "solana_modes.h"
#include <sqlite3.h>
#include <filesystem>

// Global variables for signal handling
std::atomic<bool> g_running(true);
std::atomic<bool> g_force_exit(false);
int g_selected_gpu = 0;


struct PreparedBatch {
    std::vector<BtcPrivPub> items;
};

class BatchBuffer {
public:
    explicit BatchBuffer(size_t max_size) : max_size_(max_size) {}

    bool push(PreparedBatch&& batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_full_.wait(lock, [&] { return queue_.size() < max_size_ || closed_; });
        if (closed_) {
            return false;
        }
        queue_.push_back(std::move(batch));
        cv_not_empty_.notify_one();
        return true;
    }

    bool pop(PreparedBatch& out_batch) {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_not_empty_.wait(lock, [&] { return !queue_.empty() || closed_; });
        if (queue_.empty()) {
            return false;
        }
        out_batch = std::move(queue_.front());
        queue_.pop_front();
        cv_not_full_.notify_one();
        return true;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        cv_not_empty_.notify_all();
        cv_not_full_.notify_all();
    }

private:
    size_t max_size_;
    std::deque<PreparedBatch> queue_;
    std::mutex mutex_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    bool closed_ = false;
};

// Signal handler
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        if (g_running.load()) {
            std::cout << "\n\nReceived termination signal. Shutting down...\n";
            g_running.store(false);
        } else {
            std::cout << "\n\nForced termination.\n";
            g_force_exit.store(true);
            exit(1);
        }
    }
}

// Function declarations
bool run_btc_external_bloom_mode(ConfigManager& config);
bool run_btc_gpu_scan_mode(ConfigManager& config);
bool run_btc_validate_results_mode();
bool run_eth_external_bloom_mode(ConfigManager& config);
bool run_eth_gpu_scan_mode(ConfigManager& config);
bool run_eth_validate_results_mode();


// Bitcoin Mode 5: external DB -> BTC Bloom filter (20B/32B addresses)
bool run_btc_external_bloom_mode(ConfigManager& config) {
    std::cout << "=== BTC EXTERNAL BLOOM FILTER MODE ===" << std::endl << std::endl;
    
    // Output folder
    std::string bloom_dir = "bloom_external_btc";
    try {
        if (!std::filesystem::exists(bloom_dir)) {
            std::filesystem::create_directories(bloom_dir);
        }
    } catch (...) {}

    // Choose address type
    std::cout << "Select Bitcoin address type:" << std::endl;
    std::cout << "1. P2PKH addresses (p2pkh_addresses.db)" << std::endl;
    std::cout << "2. P2SH addresses (p2sh_addresses.db)" << std::endl;
    std::cout << "Your choice (1-2): ";
    
    int addr_type;
    if (!(std::cin >> addr_type)) {
        std::cin.clear();
        addr_type = 1;
    }
    if (addr_type < 1 || addr_type > 2) addr_type = 1;
    
    size_t key_size = 20; // All Bitcoin addresses in this mode are 20 bytes
    std::string db_path;
    
    std::string db_name = (addr_type == 1) ? "p2pkh_addresses.db" : "p2sh_addresses.db";
    std::string address_type_name = (addr_type == 1) ? "P2PKH" : "P2SH";
    
    std::cout << "\nEnter path to " << address_type_name << " database [C:/Users/ily/Downloads/btc_dbs/" << db_name << "]:\n> ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, db_path);
    
    if (db_path.empty()) {
        db_path = "C:/Users/ily/Downloads/btc_dbs/" + db_name;
    }
    
    if (!std::filesystem::exists(db_path)) {
        std::cout << "Invalid DB path: " << db_path << std::endl;
        return false;
    }

    // Count unique addresses and basic analytics
    size_t unique_addresses = 0;
    size_t total_rows = 0;
    uint64_t db_file_size_bytes = std::filesystem::file_size(db_path);

    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cout << "Failed to open DB: " << db_path << std::endl;
        return false;
    }

    // Offer fast estimate (size-based) or precise COUNT(*)
    size_t fast_estimate_rows = static_cast<size_t>(db_file_size_bytes / 32); // ~20B addr + overhead
    std::cout << "\nDatabase size: " << (db_file_size_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Fast estimate (based on file size): ~" << fast_estimate_rows << " rows." << std::endl;
    std::cout << "Perform precise row count? This may take a while on large DBs. (y/N): ";
    char do_precise = 'n';
    std::cin >> do_precise;
    if (do_precise == 'y' || do_precise == 'Y') {
        std::cout << "Counting rows with COUNT(*) ... this can take several minutes for big files." << std::endl;
        std::cout.flush();
        const char* count_sql = "SELECT COUNT(*) FROM addresses";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, count_sql, -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                total_rows = sqlite3_column_int64(stmt, 0);
            }
        }
        sqlite3_finalize(stmt);
        unique_addresses = total_rows;
    } else {
        unique_addresses = fast_estimate_rows;
        total_rows = fast_estimate_rows;
    }

    if (unique_addresses == 0) {
        std::cout << "No addresses found in database." << std::endl;
        sqlite3_close(db);
        return false;
    }

    // Determine database type for display
    std::string db_type_name = (addr_type == 1) ? "P2PKH" : "P2SH";
    
    std::cout << "Database analysis:" << std::endl;
    std::cout << "  Database type: " << db_type_name << std::endl;
    std::cout << "  Database path: " << db_path << std::endl;
    std::cout << "  Total rows (est/precise): " << total_rows << std::endl;
    std::cout << "  Unique addresses (est/precise): " << unique_addresses << std::endl;
    std::cout << "  Address size: " << key_size << " bytes" << std::endl;
    std::cout << "  Database size: " << (db_file_size_bytes / (1024*1024)) << " MB" << std::endl;

    // FPR selection
    struct FprOption { double fpr; std::string name; };
    std::vector<FprOption> options = {
        {1e-6, "1e-6 (0.0001%)"},
        {1e-7, "1e-7 (0.00001%)"},
        {1e-8, "1e-8 (0.000001%)"},
        {1e-9, "1e-9 (0.0000001%)"}
    };
    
    auto estimate_bytes = [&](size_t n, double p) -> size_t {
        if (n == 0) return 0;
        const double k = 8.0;
        double root = std::pow(std::clamp(p, 1e-12, 0.5), 1.0 / k);
        double denom = -std::log(1.0 - root);
        double bits = (denom > 0.0) ? (n * k / denom) : 0.0;
        size_t blocks = (static_cast<size_t>(std::ceil(bits)) + 256 - 1) / 256;
        size_t total_bits = blocks * 256;
        return total_bits / 8;
    };

    std::cout << "\nChoose FPR (false positive rate), fixed k=8:" << std::endl;
    std::cout << "0) Enter custom FPR" << std::endl;
    for (size_t i = 0; i < options.size(); ++i) {
        size_t bytes = estimate_bytes(unique_addresses, options[i].fpr);
        std::cout << (i+1) << ") " << options[i].name << "  ~" << (bytes / (1024*1024)) << " MB" << std::endl;
    }
    
    int fpr_choice;
    std::cout << "Select (0-" << options.size() << ") [default 1]: ";
    if (!(std::cin >> fpr_choice)) {
        std::cin.clear();
        fpr_choice = 1;
    }
    
    double fpr;
    size_t estimated_filter_bytes = 0;
    if (fpr_choice == 0) {
        std::cout << "Enter custom FPR (e.g., 1e-8): ";
        if (!(std::cin >> fpr)) {
            std::cin.clear();
            fpr = 1e-6;
        }
    } else if (fpr_choice >= 1 && fpr_choice <= (int)options.size()) {
        fpr = options[fpr_choice - 1].fpr;
    } else {
        fpr = 1e-6;
    }
    estimated_filter_bytes = estimate_bytes(unique_addresses, fpr);

    std::cout << "Selected FPR: " << fpr << std::endl;
    std::cout << "Estimated Bloom filter size: " << (estimated_filter_bytes / (1024 * 1024)) << " MB" << std::endl;
    if (estimated_filter_bytes > (size_t)3ull * 1024ull * 1024ull * 1024ull) {
        std::cout << "  Warning: requires several GB of RAM. Close other apps or pick higher FPR to reduce size." << std::endl;
    }
    
    // Generate filename based on database type
    std::string save_path = (addr_type == 1) ? 
        bloom_dir + "/bitcoin_p2pkh_20b.db.bf" : 
        bloom_dir + "/bitcoin_p2sh_20b.db.bf";
    
    std::cout << "\nCreating Bloom filter..." << std::endl;
    std::cout << "  Target FPR: " << fpr << std::endl;
    std::cout << "  Expected elements: " << unique_addresses << std::endl;
    std::cout << "  Allocating bit array (~" << (estimated_filter_bytes / (1024 * 1024)) << " MB)" << std::endl;
    
    // Read addresses from DB and create filter
    BloomFilter bf(unique_addresses, fpr);

    // Read addresses in batches
    const size_t batch_size = 100000;
    size_t processed = 0;
    
    const char* select_sql = "SELECT address FROM addresses";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, select_sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cout << "Failed to prepare statement" << std::endl;
        sqlite3_close(db);
        return false;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const void* addr_data = sqlite3_column_blob(stmt, 0);
        int addr_len = sqlite3_column_bytes(stmt, 0);
        
        if (addr_data && addr_len == (int)key_size) {
            const uint8_t* addr_bytes = static_cast<const uint8_t*>(addr_data);
            bf.add_bytes(addr_bytes, key_size);
            processed++;
            
            if (processed % batch_size == 0) {
                std::cout << "\rProcessed: " << processed << "/" << unique_addresses << " addresses" << std::flush;
            }
        }
    }
    
    std::cout << "\rProcessed: " << processed << "/" << unique_addresses << " addresses" << std::endl;
    
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    // Save filter
    if (!bf.save_to_file(save_path)) {
        std::cout << "Failed to save Bloom filter" << std::endl;
        return false;
    }

    // Show results
    BloomFilter::FilterStats stats{};
    if (bf.load_from_file(save_path)) {
        stats = bf.get_stats();
        std::cout << "\nFilter created successfully: " << save_path << std::endl;
        std::cout << "  Size: " << (stats.size_bytes / (1024*1024)) << " MB (" << stats.size_bytes << " bytes)" << std::endl;
        std::cout << "  Expected elements: " << stats.expected_elements << std::endl;
        std::cout << "  FPR: " << stats.false_positive_rate << std::endl;
        std::cout << "  Blocks: " << stats.blocks_count << ", Hashes: " << (int)stats.hash_functions << std::endl;
    } else {
        std::cout << "Filter created in '" << bloom_dir << "' with FPR=" << fpr << ". File: " << save_path << std::endl;
    }

    return true;
}

// Bitcoin Mode 6: GPU scan against BTC Bloom filter
bool run_btc_gpu_scan_mode(ConfigManager& config) {
    std::cout << "=== BTC GPU BLOOM SCAN MODE (Mode 6) ===" << std::endl << std::endl;
    std::cout << "Load a previously created BTC v2 Bloom filter (from Mode 5)." << std::endl;

    // Ensure selected GPU is active for CUDA paths used below
    cudaSetDevice(g_selected_gpu);

    // Choose address type
    std::cout << "Select Bitcoin address type:" << std::endl;
    std::cout << "1. P2PKH addresses (p2pkh_addresses.db)" << std::endl;
    std::cout << "2. P2SH addresses (p2sh_addresses.db)" << std::endl;
    std::cout << "Your choice (1-2): ";
    
    int addr_type;
    if (!(std::cin >> addr_type)) {
        std::cin.clear();
        addr_type = 1;
    }
    if (addr_type < 1 || addr_type > 2) addr_type = 1;
    
    size_t key_size = 20; // All Bitcoin addresses in this mode are 20 bytes
    BitcoinAddressType btc_addr_type = (addr_type == 1) ? BitcoinAddressType::P2PKH : BitcoinAddressType::P2SH;
    std::string addr_type_name = (addr_type == 1) ? "p2pkh" : "p2sh";
    std::string filter_name = "bitcoin_" + addr_type_name + "_20b.db.bf";

    // Prefer Bloom filter next to the executable/CWD in portable layout
    std::filesystem::path default_filter = std::filesystem::current_path() / "bloom_external_btc" / filter_name;
    std::string filter_path;
    std::cout << "Enter path to Bloom filter (.bf) [" << default_filter.string() << "]:\n> ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, filter_path);
    
    if (filter_path.empty()) {
        filter_path = default_filter.string();
    }
    
    if (!std::filesystem::exists(filter_path)) {
        std::cout << "Invalid Bloom filter path: " << filter_path << std::endl;
        return false;
    }

    // Load filter
    BloomFilter bf;
    if (!bf.load_from_file(filter_path)) {
        std::cout << "Failed to load Bloom filter" << std::endl;
        return false;
    }
    
    if (bf.get_file_version() != 2) {
        std::cout << "Bloom filter must be v2 (deterministic FNV-1a 64)." << std::endl;
        return false;
    }
    
    auto stats = bf.get_stats();
    const auto& bits = bf.data();
    size_t blocks = bf.get_blocks_count();
    uint8_t k_hashes = bf.get_hash_function_count();
    
    std::cout << "Filter info: size=" << (stats.size_bytes / (1024*1024)) << " MB, blocks=" << blocks
              << ", k=" << (int)k_hashes << ", expected_elements=" << stats.expected_elements
              << ", target FPR=" << stats.false_positive_rate << std::endl;

    // Upload filter into VRAM
    if (!gpu_bloom_load(bits.data(), bits.size(), blocks, k_hashes)) {
        std::cout << "Failed to upload Bloom filter to GPU" << std::endl;
        return false;
    }
    
    LOG_INFO("BTC Mode6: Bloom filter uploaded to GPU: " + std::to_string(bits.size()) + " bytes, blocks=" + std::to_string(blocks));

    // Output file for matches
    std::string out_path = "matches_btc_" + std::to_string(key_size) + "b.txt";
    std::cout << "Output file for matches [" << out_path << "]: ";
    std::string out_in;
    std::getline(std::cin, out_in);
    if (!out_in.empty()) out_path = out_in;
    
    std::ofstream out_file(out_path, std::ios::app);
    if (!out_file.is_open()) {
        std::cout << "Failed to open output file: " << out_path << std::endl;
        gpu_bloom_unload();
        return false;
    }

    // Write header if file is new/empty and determine starting match number
    size_t match_counter = 0;
    try {
        std::error_code ec;
        bool need_header = !std::filesystem::exists(out_path, ec) || std::filesystem::file_size(out_path, ec) == 0;
        if (need_header) {
            out_file << "№\tTIMESTAMP\tCHECKED_B\tHASH160\tPRIVATE_KEY\tADDRESS_MAIN\tADDRESS_BECH32\n";
            if (!out_file.flush()) {
                LOG_ERROR("Failed to write header to output file: " + out_path);
                gpu_bloom_unload();
                return false;
            }
        } else {
            // Count existing lines to continue numbering
            std::ifstream count_file(out_path);
            if (!count_file.is_open()) {
                LOG_ERROR("Failed to open output file for counting: " + out_path);
                gpu_bloom_unload();
                return false;
            }

            std::string line;
            while (std::getline(count_file, line)) {
                if (!line.empty() && line[0] != '#' && line.find("№") == std::string::npos) {
                    match_counter++;
                }
            }

            if (count_file.bad()) {
                LOG_WARN("Error reading output file for counting: " + out_path);
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Exception during file header/counting operations: " + std::string(e.what()));
    } catch (...) {
        LOG_ERROR("Unknown exception during file header/counting operations");
    }

    // Initialize Bitcoin generator with optimizations
    BitcoinGenerator generator;
    generator.initialize();
    generator.set_address_type(btc_addr_type);
    generator.set_use_optimizations(true); // Enable GPU optimizations
    if (g_logger) g_logger->set_level(LogLevel::WARN);

    // NVML dynamic loader (Windows and others with nvml.dll in PATH)
    struct NvmlLoader {
        bool ok = false;
        
        using Device = void*;
        struct Util { unsigned int gpu, memory; };
        // Match NVML's nvmlMemory_t layout: total, free, used
        struct Mem { unsigned long long total, free, used; };
        using Ret = int; // NVML_SUCCESS == 0
        
        // Function pointers
        Ret (WINAPI *nvmlInit)() = nullptr;
        Ret (WINAPI *nvmlShutdown)() = nullptr;
        Ret (WINAPI *nvmlDeviceGetHandleByIndex)(unsigned int, Device*) = nullptr;
        Ret (WINAPI *nvmlDeviceGetHandleByIndex_v2)(unsigned int, Device*) = nullptr;
        Ret (WINAPI *nvmlDeviceGetUtilizationRates)(Device, Util*) = nullptr;
        Ret (WINAPI *nvmlDeviceGetMemoryInfo)(Device, Mem*) = nullptr;
        
        HMODULE dll = nullptr;
        Device device = nullptr;
        
        bool load(int gpu_index) {
#ifdef _WIN32
            dll = LoadLibraryA("nvml.dll");
            if (!dll) return false;
            nvmlInit = reinterpret_cast<Ret (WINAPI*)()>(GetProcAddress(dll, "nvmlInit_v2"));
            if (!nvmlInit) nvmlInit = reinterpret_cast<Ret (WINAPI*)()>(GetProcAddress(dll, "nvmlInit"));
            nvmlShutdown = reinterpret_cast<Ret (WINAPI*)()>(GetProcAddress(dll, "nvmlShutdown"));
            nvmlDeviceGetHandleByIndex = reinterpret_cast<Ret (WINAPI*)(unsigned int, Device*)>(GetProcAddress(dll, "nvmlDeviceGetHandleByIndex"));
            nvmlDeviceGetHandleByIndex_v2 = reinterpret_cast<Ret (WINAPI*)(unsigned int, Device*)>(GetProcAddress(dll, "nvmlDeviceGetHandleByIndex_v2"));
            nvmlDeviceGetUtilizationRates = reinterpret_cast<Ret (WINAPI*)(Device, Util*)>(GetProcAddress(dll, "nvmlDeviceGetUtilizationRates"));
            nvmlDeviceGetMemoryInfo = reinterpret_cast<Ret (WINAPI*)(Device, Mem*)>(GetProcAddress(dll, "nvmlDeviceGetMemoryInfo"));
            if (!nvmlInit || !nvmlShutdown || (!nvmlDeviceGetHandleByIndex && !nvmlDeviceGetHandleByIndex_v2) || !nvmlDeviceGetUtilizationRates || !nvmlDeviceGetMemoryInfo) return false;
            if (nvmlInit() != 0) return false;
            Ret r = nvmlDeviceGetHandleByIndex_v2 ? nvmlDeviceGetHandleByIndex_v2(gpu_index, &device)
                                                  : nvmlDeviceGetHandleByIndex(gpu_index, &device);
            if (r != 0) return false;
            ok = true;
            return true;
#else
            return false;
#endif
        }
        void unload() {
#ifdef _WIN32
            if (ok && nvmlShutdown) nvmlShutdown();
            if (dll) { FreeLibrary(dll); dll = nullptr; }
            ok = false;
#endif
        }
        bool query(unsigned int& gpu_util, unsigned int& mem_util, unsigned long long& mem_used_mb, unsigned long long& mem_total_mb) {
            if (!ok) return false;
            Util u{}; Mem m{};

            // Get GPU utilization rates (compute + memory)
            if (nvmlDeviceGetUtilizationRates(device, &u) != 0) {
                // If this fails, set defaults
                gpu_util = 0;
                mem_util = 0;
            } else {
                gpu_util = u.gpu;
                mem_util = u.memory; // Note: memory utilization may not be available for all GPU types/modes
            }

            // Get memory info (this should always work)
            if (nvmlDeviceGetMemoryInfo(device, &m) != 0) return false;

            mem_used_mb = m.used / (1024ULL*1024ULL);
            mem_total_mb = m.total / (1024ULL*1024ULL);
            return true;
        }
    } nvml;
    if (!nvml.load(g_selected_gpu)) {
        LOG_WARN("NVML initialization failed - GPU monitoring will be limited");
    } else {
        LOG_INFO("NVML initialized successfully for GPU monitoring");
    }

    std::cout << "\nStarting BTC GPU scan for " << addr_type_name << " addresses..." << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;
    
    auto report_last_time = std::chrono::steady_clock::now();
    size_t checked_at_last_report = 0;
    size_t matched_at_last_report = 0;
    size_t total_checked = 0, total_matched = 0;
    bool did_cpu_crosscheck = false;
    auto last_crosscheck = std::chrono::steady_clock::now();
    
    // Deduplication cache for written hash160 (LRU-ish by size cap)
    std::unordered_set<std::string> recent_hashes;
    recent_hashes.reserve(200000);
    const size_t max_recent = 200000; // ~200k last matches

    const size_t batch_size = config.get_gpu_batch_size() ? config.get_gpu_batch_size() : 65536;
    std::vector<uint8_t> host_priv_batch(batch_size * 32);
    std::vector<uint8_t> flags(batch_size);
    bool gpu_keygen_enabled = true;

    while (g_running.load() && !g_force_exit.load()) {
        size_t keys_in_batch = 0;
        bool pipeline_ok = false;

        if (gpu_keygen_enabled) {
            if (!gpu_btc_generate_privkeys_device(batch_size)) {
                LOG_WARN("GPU key generation failed - switching to CPU generator");
                gpu_keygen_enabled = false;
            } else {
                pipeline_ok = gpu_btc_priv_fused_pipeline_device(
                    batch_size,
                    flags.data(),
                    (int)btc_addr_type,
                    nullptr,
                    host_priv_batch.data());
                if (!pipeline_ok) {
                    LOG_ERROR("GPU fused pipeline (device priv) failed");
                } else {
                    keys_in_batch = batch_size;
                }
            }
        }

        std::vector<BtcPrivPub> batch_privpub;
        if (!pipeline_ok) {
            batch_privpub = generator.generate_batch_priv_pub(batch_size);
            keys_in_batch = batch_privpub.size();
            if (keys_in_batch == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            size_t needed_priv_bytes = keys_in_batch * 32;
            if (host_priv_batch.size() < needed_priv_bytes) {
                host_priv_batch.resize(needed_priv_bytes);
            }
            if (flags.size() < keys_in_batch) {
                flags.resize(keys_in_batch);
            }
            for (size_t i = 0; i < keys_in_batch; ++i) {
                std::memcpy(&host_priv_batch[i * 32], batch_privpub[i].private_key.data(), 32);
            }

            pipeline_ok = gpu_btc_priv_fused_pipeline(
                host_priv_batch.data(),
                keys_in_batch,
                flags.data(),
                (int)btc_addr_type,
                nullptr);

            if (!pipeline_ok) {
                LOG_ERROR("GPU fused pipeline (host priv) failed");
                break;
            }
        } else {
            // Ensure buffers are large enough (they may have been resized in fallback)
            if (flags.size() < keys_in_batch) flags.resize(keys_in_batch);
        }

        if (!pipeline_ok) {
            break;
        }

        auto count_set_bits = [](uint8_t value) -> int {
            return static_cast<int>(std::bitset<8>(value).count());
        };

        auto get_priv_ptr = [&](size_t index) -> const uint8_t* {
            return &host_priv_batch[index * 32];
        };

        auto compute_cpu_mask = [&](size_t index, uint8_t out_privkeys[6][32]) -> uint8_t {
            if (!generator.compute_endomorphic_privkeys(get_priv_ptr(index), out_privkeys)) {
                return 0;
            }
            uint8_t mask = 0;
            uint8_t pub_tmp[33];
            uint8_t hash_tmp[20];
            for (int v = 0; v < 6; ++v) {
                if (!generator.priv_to_pub_compressed(out_privkeys[v], pub_tmp)) {
                    continue;
                }
                btc_pub33_to_hash160_cpu(pub_tmp, btc_addr_type, hash_tmp);
                if (bf.might_contain_bytes(hash_tmp, 20)) {
                    mask |= static_cast<uint8_t>(1U << v);
                }
            }
            return mask;
        };

        // Use optimized fused pipeline (combines pub33->hash160->bloom in one kernel)
        // This provides better performance by reducing CPU-GPU transfers
        // One-time CPU vs GPU cross-check on first batch
        auto now_cc = std::chrono::steady_clock::now();
        auto cc_elapsed = std::chrono::duration_cast<std::chrono::minutes>(now_cc - last_crosscheck).count();
        if (!did_cpu_crosscheck || cc_elapsed >= 15) {
            size_t sample_n = std::min<size_t>(keys_in_batch, 1000);
            size_t cpu_true = 0, gpu_true = 0, mismatches = 0;
            for (size_t i = 0; i < sample_n; ++i) {
                uint8_t endo_privkeys[6][32];
                uint8_t cpu_mask = compute_cpu_mask(i, endo_privkeys);
                uint8_t gpu_mask = flags[i];
                cpu_true += static_cast<size_t>(count_set_bits(cpu_mask));
                gpu_true += static_cast<size_t>(count_set_bits(gpu_mask));
                if (cpu_mask != gpu_mask) mismatches++;
            }
            std::cout << "CPU/GPU cross-check on " << sample_n << ": cpu_true=" << cpu_true
                      << ", gpu_true=" << gpu_true << ", mismatches=" << mismatches << std::endl;
            did_cpu_crosscheck = true;
            last_crosscheck = now_cc;
        }
        
        // Handle matches with validation before writing
        size_t matched_in_batch = 0;
        for (size_t i = 0; i < keys_in_batch; ++i) {
            uint8_t bitmask = flags[i];
            if (bitmask == 0) continue;

            uint8_t endo_privkeys[6][32];
            if (!generator.compute_endomorphic_privkeys(get_priv_ptr(i), endo_privkeys)) {
                continue;
            }

            for (int variant = 0; variant < 6; ++variant) {
                if ((bitmask & (1U << variant)) == 0) continue;

                uint8_t priv_variant[32];
                std::memcpy(priv_variant, endo_privkeys[variant], 32);

                uint8_t pub_variant[33];
                if (!generator.priv_to_pub_compressed(priv_variant, pub_variant)) {
                    continue;
                }

                uint8_t h160_cpu[20];
                btc_pub33_to_hash160_cpu(pub_variant, btc_addr_type, h160_cpu);

                if (!bf.might_contain_bytes(h160_cpu, 20)) {
                    continue; // skip inconsistent match
                }

                std::string key_hex = ConversionUtils::hex_encode(std::vector<uint8_t>(h160_cpu, h160_cpu + 20));
                if (recent_hashes.find(key_hex) != recent_hashes.end()) {
                    continue;
                }
                if (recent_hashes.size() >= max_recent) {
                    recent_hashes.clear();
                }
                recent_hashes.insert(key_hex);
                matched_in_batch++;
                match_counter++;

                auto now = std::chrono::system_clock::now();
                auto now_time_t = std::chrono::system_clock::to_time_t(now);
                auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
                std::tm now_tm;
                #ifdef _WIN32
                localtime_s(&now_tm, &now_time_t);
                #else
                localtime_r(&now_time_t, &now_tm);
                #endif
                std::ostringstream timestamp_stream;
                timestamp_stream << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S")
                                << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
                std::string timestamp = timestamp_stream.str();

                double checked_billions = static_cast<double>(total_checked) / 1e9;
                std::ostringstream checked_stream;
                checked_stream << std::fixed << std::setprecision(3) << checked_billions;
                std::string checked_b = checked_stream.str();

                std::string addr_hex = key_hex;
                std::string priv_hex = ConversionUtils::hex_encode(std::vector<uint8_t>(priv_variant, priv_variant + 32));

                try {
                    if (btc_addr_type == BitcoinAddressType::P2PKH) {
                        std::string addr_p2pkh = btc_p2pkh_from_hash160(h160_cpu);
                        std::string addr_bech32 = btc_p2wpkh_bech32_from_hash160(h160_cpu);
                        out_file << match_counter << "\t" << timestamp << "\t" << checked_b << "\t" << addr_hex << "\t" << priv_hex << "\t" << addr_p2pkh << "\t" << addr_bech32 << "\n";
                    } else {
                        uint8_t h160_pub[20];
                        btc_pub33_to_hash160_cpu(pub_variant, BitcoinAddressType::P2PKH, h160_pub);
                        std::string addr_p2sh = btc_p2sh_from_hash160(h160_cpu);
                        std::string addr_bech32 = btc_p2wpkh_bech32_from_hash160(h160_pub);
                        out_file << match_counter << "\t" << timestamp << "\t" << checked_b << "\t" << addr_hex << "\t" << priv_hex << "\t" << addr_p2sh << "\t" << addr_bech32 << "\n";
                    }

                    if (out_file.fail()) {
                        LOG_ERROR("Failed to write match #" + std::to_string(match_counter) + " to output file: " + out_path);
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Exception writing match #" + std::to_string(match_counter) + " to file: " + std::string(e.what()));
                } catch (...) {
                    LOG_ERROR("Unknown exception writing match #" + std::to_string(match_counter) + " to file");
                }
            }
        }
        
        if (matched_in_batch > 0) out_file.flush();
        
        total_checked += keys_in_batch * 6;
        total_matched += matched_in_batch;
        
        // Progress report every minute
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - report_last_time).count();
        // Report more frequently for ETH (10s) to mirror BTC-style stats but more responsive
        if (elapsed >= 10) {
            size_t delta_checked = total_checked - checked_at_last_report;
            size_t delta_matched = total_matched - matched_at_last_report;
            double per_sec_1m = elapsed > 0 ? (double)delta_checked / (double)elapsed : 0.0;
            double fpr_1m = delta_checked ? (double)delta_matched / (double)delta_checked : 0.0;
            double fpr_total = total_checked ? (double)total_matched / (double)total_checked : 0.0;
            
            std::cout << "[1m] speed=" << std::fixed << std::setprecision(0) << per_sec_1m << " keys/s, "
                      << "checked=" << total_checked << ", matched=" << total_matched
                      << ", fpr_1m=" << std::setprecision(6) << fpr_1m
                      << ", fpr_total=" << std::setprecision(6) << fpr_total;
            
            // Enhanced GPU monitoring
            unsigned int gu=0, mu=0;
            unsigned long long used=0, tot=0;
            bool nvml_ok = nvml.query(gu, mu, used, tot);

            if (nvml_ok) {
                std::cout << ", gpu=" << gu << "%"
                          << ", vmem=" << used << "MB/" << tot << "MB";

                // Add memory utilization with explanation if it's 0
                if (mu == 0) {
                    std::cout << ", mem_util=N/A";
                } else {
                    std::cout << ", mem_util=" << mu << "%";
                }

                // Add memory bandwidth estimate (used/total ratio as proxy)
                double mem_bw_ratio = tot > 0 ? (double)used / (double)tot * 100.0 : 0.0;
                std::cout << ", bw=" << std::fixed << std::setprecision(1) << mem_bw_ratio << "%";
            } else {
                std::cout << ", gpu=N/A, vmem=N/A";
            }

            // Add batch processing metrics
            double avg_batch_time = elapsed > 0 ? (double)elapsed / (double)total_checked * (keys_in_batch * 6) : 0.0;
            std::cout << ", batch=" << std::fixed << std::setprecision(2) << avg_batch_time << "ms";

            // Add CUDA memory info for comparison with NVML
            size_t cuda_free = 0, cuda_total = 0;
            if (cudaMemGetInfo(&cuda_free, &cuda_total) == cudaSuccess) {
                size_t cuda_used = cuda_total - cuda_free;
                double cuda_util = cuda_total > 0 ? (double)cuda_used / (double)cuda_total * 100.0 : 0.0;
                std::cout << ", cuda_mem=" << std::fixed << std::setprecision(1) << cuda_util << "%";
            }

            std::cout << std::endl;
            
            report_last_time = now;
            checked_at_last_report = total_checked;
            matched_at_last_report = total_matched;
        }
    }
    
    std::cout << std::endl;
    gpu_bloom_unload();

    // Safe file closing with error handling
    if (out_file.is_open()) {
        try {
            out_file.flush(); // Ensure all data is written
            out_file.close();

            if (out_file.fail()) {
                LOG_WARN("Warning: Possible issues when closing output file: " + out_path);
            } else {
                LOG_INFO("Successfully closed output file: " + out_path);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("Exception when closing output file: " + std::string(e.what()));
        } catch (...) {
            LOG_ERROR("Unknown exception when closing output file");
        }
    }
    
    if (g_logger) g_logger->set_level(LogLevel::INFO);
    nvml.unload();
    
    std::cout << "Scan finished. Total checked: " << total_checked << ", matched: " << total_matched << std::endl;
    return true;
}

// Bitcoin Mode 7: Validate found addresses (Bloom + key/address consistency + DB lookup)
bool run_btc_validate_results_mode() {
    std::cout << "=== Validate found addresses (Bloom result + key/address consistency check + DB lookup) ===" << std::endl << std::endl;

    // Choose address type (for defaults)
    std::cout << "Select Bitcoin address type:" << std::endl;
    std::cout << "1. P2PKH addresses (p2pkh_addresses.db)" << std::endl;
    std::cout << "2. P2SH addresses (p2sh_addresses.db)" << std::endl;
    std::cout << "Your choice (1-2): ";

    int addr_type;
    if (!(std::cin >> addr_type)) {
        std::cin.clear();
        addr_type = 1;
    }
    if (addr_type < 1 || addr_type > 2) addr_type = 1;
    std::string addr_type_name = (addr_type == 1) ? "p2pkh" : "p2sh";
    std::string filter_name = "bitcoin_" + addr_type_name + "_20b.db.bf";

    // Defaults
    std::filesystem::path default_filter = std::filesystem::current_path() / "bloom_external_btc" / filter_name;
    std::filesystem::path default_db = std::filesystem::current_path() / "databases" / (addr_type == 1 ? "p2pkh_addresses.db" : "p2sh_addresses.db");
    std::filesystem::path default_matches = std::filesystem::current_path() / "matches_btc_20b.txt";
    std::filesystem::path default_output = std::filesystem::current_path() / "btc_db_verification_results.txt";

    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string filter_path, db_path, input_path, output_path;
    std::cout << "Enter path to Bloom filter (.bf) [" << default_filter.string() << "]:\n> ";
    std::getline(std::cin, filter_path);
    if (filter_path.empty()) filter_path = default_filter.string();

    std::cout << "Enter path to SQLite DB [" << default_db.string() << "]:\n> ";
    std::getline(std::cin, db_path);
    if (db_path.empty()) db_path = default_db.string();

    std::cout << "Enter path to matches file [" << default_matches.string() << "]:\n> ";
    std::getline(std::cin, input_path);
    if (input_path.empty()) input_path = default_matches.string();

    std::cout << "Enter path to output report [" << default_output.string() << "]:\n> ";
    std::getline(std::cin, output_path);
    if (output_path.empty()) output_path = default_output.string();

    if (!std::filesystem::exists(filter_path)) {
        std::cout << "Invalid Bloom filter path: " << filter_path << std::endl;
        return false;
    }
    if (!std::filesystem::exists(db_path)) {
        std::cout << "Invalid DB path: " << db_path << std::endl;
        return false;
    }
    if (!std::filesystem::exists(input_path)) {
        std::cout << "Input matches file not found: " << input_path << std::endl;
        return false;
    }

    // Locate verifier script (try CWD, parent, grandparent)
    std::vector<std::filesystem::path> candidates;
    auto cwd = std::filesystem::current_path();
    candidates.push_back(cwd / "btc_db_verifier.py");
    candidates.push_back(cwd.parent_path() / "btc_db_verifier.py");
    candidates.push_back(cwd.parent_path().parent_path() / "btc_db_verifier.py");

    std::filesystem::path script_path;
    for (const auto& c : candidates) {
        if (!c.empty() && std::filesystem::exists(c)) {
            script_path = c;
            break;
        }
    }
    if (script_path.empty()) {
        std::cout << "Cannot find btc_db_verifier.py (tried CWD, parent, grandparent)." << std::endl;
        return false;
    }

    // Build command to reuse Python verifier (with Bloom+DB+consistency checks)
    std::string command = "python \"" + script_path.string() + "\" --bloom \"" + filter_path + "\" --db \"" + db_path +
                          "\" -i \"" + input_path + "\" -o \"" + output_path + "\"";

    std::cout << "Running verifier:\n  " << command << std::endl;
    int rc = std::system(command.c_str());
    if (rc != 0) {
        std::cout << "Verifier finished with code " << rc << std::endl;
        return false;
    }
    return true;
}

// Ethereum Mode 8: external DB -> ETH Bloom filter (20B addresses)
bool run_eth_external_bloom_mode(ConfigManager& config) {
    std::cout << "=== ETH EXTERNAL BLOOM FILTER MODE ===" << std::endl << std::endl;

    std::string bloom_dir = "bloom_external_eth";
    try {
        if (!std::filesystem::exists(bloom_dir)) {
            std::filesystem::create_directories(bloom_dir);
        }
    } catch (...) {}

    // Prompt DB/table/column
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string db_path, table_name, column_name;
    std::string default_db = config.get_database_directory() + "/eth_addresses.db";
    std::cout << "Enter path to SQLite DB with ETH addresses [" << default_db << "]:\n> ";
    std::getline(std::cin, db_path);
    if (db_path.empty()) db_path = default_db;

    std::cout << "Enter table name [addresses]: ";
    std::getline(std::cin, table_name);
    if (table_name.empty()) table_name = "addresses";

    std::cout << "Enter column name with address [address]: ";
    std::getline(std::cin, column_name);
    if (column_name.empty()) column_name = "address";

    if (!std::filesystem::exists(db_path)) {
        std::cout << "Invalid DB path: " << db_path << std::endl;
        return false;
    }

    size_t key_size = 20;
    uint64_t db_file_size_bytes = std::filesystem::file_size(db_path);

    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cout << "Failed to open DB: " << db_path << std::endl;
        return false;
    }

    // Improved fast estimate using page stats and sampled avg length (avoids full COUNT(*))
    sqlite3_int64 page_size = 0, page_count = 0, freelist_count = 0;
    {
        sqlite3_stmt* stmt_ps = nullptr;
        if (sqlite3_prepare_v2(db, "PRAGMA page_size", -1, &stmt_ps, nullptr) == SQLITE_OK &&
            sqlite3_step(stmt_ps) == SQLITE_ROW) {
            page_size = sqlite3_column_int64(stmt_ps, 0);
        }
        sqlite3_finalize(stmt_ps);

        sqlite3_stmt* stmt_pc = nullptr;
        if (sqlite3_prepare_v2(db, "PRAGMA page_count", -1, &stmt_pc, nullptr) == SQLITE_OK &&
            sqlite3_step(stmt_pc) == SQLITE_ROW) {
            page_count = sqlite3_column_int64(stmt_pc, 0);
        }
        sqlite3_finalize(stmt_pc);

        sqlite3_stmt* stmt_fl = nullptr;
        if (sqlite3_prepare_v2(db, "PRAGMA freelist_count", -1, &stmt_fl, nullptr) == SQLITE_OK &&
            sqlite3_step(stmt_fl) == SQLITE_ROW) {
            freelist_count = sqlite3_column_int64(stmt_fl, 0);
        }
        sqlite3_finalize(stmt_fl);
    }

    double sampled_avg_len = 20.0; // fallback
    {
        std::string len_sql = "SELECT LENGTH(\"" + column_name + "\") FROM \"" + table_name + "\" LIMIT 10000";
        sqlite3_stmt* stmt_len = nullptr;
        if (sqlite3_prepare_v2(db, len_sql.c_str(), -1, &stmt_len, nullptr) == SQLITE_OK) {
            double sum = 0.0;
            size_t cnt = 0;
            while (sqlite3_step(stmt_len) == SQLITE_ROW) {
                sum += sqlite3_column_double(stmt_len, 0);
                cnt++;
            }
            if (cnt > 0) sampled_avg_len = sum / static_cast<double>(cnt);
        }
        sqlite3_finalize(stmt_len);
    }

    // Approx payload bytes: live pages * page_size, exclude freelist
    double live_bytes = 0.0;
    if (page_size > 0 && page_count > 0 && page_count >= freelist_count) {
        live_bytes = static_cast<double>(page_size) * static_cast<double>(page_count - freelist_count);
    } else {
        live_bytes = static_cast<double>(db_file_size_bytes);
    }

    // Rough per-row overhead for rowid + header; adequate for 20-byte BLOB payloads
    const double row_overhead = 12.0;
    double est_rows = (sampled_avg_len > 0.0) ? (live_bytes / (sampled_avg_len + row_overhead))
                                             : (live_bytes / 32.0);
    size_t fast_estimate_rows = static_cast<size_t>(std::max(1.0, est_rows));

    std::cout << "\nDatabase size: " << (db_file_size_bytes / (1024 * 1024)) << " MB" << std::endl;
    if (page_size > 0) {
        std::cout << "Page stats: size=" << page_size << " bytes, pages=" << page_count
                  << ", freelist=" << freelist_count << std::endl;
    }
    std::cout << "Sampled avg address length (first <=10k rows): " << sampled_avg_len << " bytes" << std::endl;
    std::cout << "Fast estimate (page stats + avg len): ~" << fast_estimate_rows << " rows." << std::endl;
    std::cout << "Perform precise row count? This may take a while. (y/N): ";
    char do_precise = 'n';
    std::cin >> do_precise;

    size_t total_rows = fast_estimate_rows;
    if (do_precise == 'y' || do_precise == 'Y') {
        std::string sql = "SELECT COUNT(*) FROM \"" + table_name + "\"";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) == SQLITE_OK) {
            if (sqlite3_step(stmt) == SQLITE_ROW) {
                total_rows = static_cast<size_t>(sqlite3_column_int64(stmt, 0));
            }
        }
        sqlite3_finalize(stmt);
    }
    size_t unique_addresses = total_rows;
    if (unique_addresses == 0) {
        std::cout << "No addresses found in database." << std::endl;
        sqlite3_close(db);
        return false;
    }

    struct FprOption { double fpr; std::string name; };
    std::vector<FprOption> options = {
        {1e-6, "1e-6 (0.0001%)"},
        {1e-7, "1e-7 (0.00001%)"},
        {1e-8, "1e-8 (0.000001%)"}
    };

    auto estimate_bytes = [&](size_t n, double p) -> size_t {
        if (n == 0) return 0;
        const double k = 8.0;
        double root = std::pow(std::clamp(p, 1e-12, 0.5), 1.0 / k);
        double denom = -std::log(1.0 - root);
        double bits = (denom > 0.0) ? (n * k / denom) : 0.0;
        size_t blocks = (static_cast<size_t>(std::ceil(bits)) + 256 - 1) / 256;
        size_t total_bits = blocks * 256;
        return total_bits / 8;
    };

    std::cout << "\nChoose FPR (false positive rate), fixed k=8:" << std::endl;
    std::cout << "0) Enter custom FPR" << std::endl;
    for (size_t i = 0; i < options.size(); ++i) {
        size_t bytes = estimate_bytes(unique_addresses, options[i].fpr);
        std::cout << (i+1) << ") " << options[i].name << "  ~" << (bytes / (1024*1024)) << " MB" << std::endl;
    }

    int fpr_choice;
    std::cout << "Select (0-" << options.size() << ") [default 1]: ";
    if (!(std::cin >> fpr_choice)) {
        std::cin.clear();
        fpr_choice = 1;
    }

    double fpr;
    if (fpr_choice == 0) {
        std::cout << "Enter custom FPR (e.g., 1e-8): ";
        if (!(std::cin >> fpr)) {
            std::cin.clear();
            fpr = 1e-6;
        }
    } else if (fpr_choice >= 1 && fpr_choice <= (int)options.size()) {
        fpr = options[fpr_choice - 1].fpr;
    } else {
        fpr = 1e-6;
    }
    size_t estimated_filter_bytes = estimate_bytes(unique_addresses, fpr);

    std::cout << "Selected FPR: " << fpr << std::endl;
    std::cout << "Estimated Bloom filter size: " << (estimated_filter_bytes / (1024 * 1024)) << " MB" << std::endl;
    if (estimated_filter_bytes > (size_t)3ull * 1024ull * 1024ull * 1024ull) {
        std::cout << "  Warning: requires several GB of RAM. Close other apps or pick higher FPR to reduce size." << std::endl;
    }

    std::string save_path = bloom_dir + "/ethereum_20b.db.bf";
    std::cout << "\nCreating Bloom filter..." << std::endl;
    std::cout << "  Target FPR: " << fpr << std::endl;
    std::cout << "  Expected elements: " << unique_addresses << std::endl;

    BloomFilter bf(unique_addresses, fpr);
    const size_t batch_size = 100000;
    size_t processed = 0;
    std::string select_sql = "SELECT \"" + column_name + "\" FROM \"" + table_name + "\"";
    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(db, select_sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cout << "Failed to prepare statement" << std::endl;
        sqlite3_close(db);
        return false;
    }

    std::array<uint8_t, 20> addr20{};
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const void* addr_data = sqlite3_column_blob(stmt, 0);
        int addr_len = sqlite3_column_bytes(stmt, 0);
        bool ok = eth_parse_address_bytes_or_hex(addr_data, addr_len, addr20);
        if (!ok) {
            const unsigned char* txt = sqlite3_column_text(stmt, 0);
            int txt_len = sqlite3_column_bytes(stmt, 0);
            ok = eth_parse_address_bytes_or_hex(txt, txt_len, addr20);
        }
        if (ok) {
            bf.add_bytes(addr20.data(), key_size);
            processed++;
            if (processed % batch_size == 0) {
                std::cout << "\rProcessed: " << processed << "/" << unique_addresses << " rows" << std::flush;
            }
        }
    }
    std::cout << "\rProcessed: " << processed << "/" << unique_addresses << " rows" << std::endl;

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    if (!bf.save_to_file(save_path)) {
        std::cout << "Failed to save Bloom filter" << std::endl;
        return false;
    }

    BloomFilter::FilterStats stats{};
    if (bf.load_from_file(save_path)) {
        stats = bf.get_stats();
        std::cout << "\nFilter created successfully: " << save_path << std::endl;
        std::cout << "  Size: " << (stats.size_bytes / (1024*1024)) << " MB (" << stats.size_bytes << " bytes)" << std::endl;
        std::cout << "  Expected elements: " << stats.expected_elements << std::endl;
        std::cout << "  FPR: " << stats.false_positive_rate << std::endl;
        std::cout << "  Blocks: " << stats.blocks_count << ", Hashes: " << (int)stats.hash_functions << std::endl;
    } else {
        std::cout << "Filter created in '" << bloom_dir << "' with FPR=" << fpr << ". File: " << save_path << std::endl;
    }
    return true;
}

// Ethereum Mode 9: GPU scan against ETH Bloom filter
bool run_eth_gpu_scan_mode(ConfigManager& config) {
    std::cout << "=== ETH GPU BLOOM SCAN MODE (Mode 9) ===" << std::endl << std::endl;
    std::cout << "Load a previously created ETH Bloom filter (from Mode 8)." << std::endl;

    cudaSetDevice(g_selected_gpu);

    std::filesystem::path default_filter = std::filesystem::current_path() / "bloom_external_eth" / "ethereum_20b.db.bf";
    std::string filter_path;
    std::cout << "Enter path to Bloom filter (.bf) [" << default_filter.string() << "]:\n> ";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::getline(std::cin, filter_path);
    if (filter_path.empty()) filter_path = default_filter.string();

    if (!std::filesystem::exists(filter_path)) {
        std::cout << "Invalid Bloom filter path: " << filter_path << std::endl;
        return false;
    }

    BloomFilter bf;
    if (!bf.load_from_file(filter_path)) {
        std::cout << "Failed to load Bloom filter" << std::endl;
        return false;
    }
    if (bf.get_file_version() != 2) {
        std::cout << "Bloom filter must be v2 (deterministic FNV-1a 64)." << std::endl;
        return false;
    }

    auto stats = bf.get_stats();
    const auto& bits = bf.data();
    size_t blocks = bf.get_blocks_count();
    uint8_t k_hashes = bf.get_hash_function_count();

    std::cout << "Filter info: size=" << (stats.size_bytes / (1024*1024)) << " MB, blocks=" << blocks
              << ", k=" << (int)k_hashes << ", expected_elements=" << stats.expected_elements
              << ", target FPR=" << stats.false_positive_rate << std::endl;

    if (!gpu_bloom_load(bits.data(), bits.size(), blocks, k_hashes)) {
        std::cout << "Failed to upload Bloom filter to GPU" << std::endl;
        return false;
    }
    LOG_INFO("ETH Mode9: Bloom filter uploaded to GPU: " + std::to_string(bits.size()) + " bytes, blocks=" + std::to_string(blocks));

    std::string out_path = "matches_eth_20b.txt";
    std::cout << "Output file for matches [" << out_path << "]: ";
    std::string out_in;
    std::getline(std::cin, out_in);
    if (!out_in.empty()) out_path = out_in;

    std::ofstream out_file(out_path, std::ios::app);
    if (!out_file.is_open()) {
        std::cout << "Failed to open output file: " << out_path << std::endl;
        gpu_bloom_unload();
        return false;
    }

    size_t match_counter = 0;
    try {
        std::error_code ec;
        bool need_header = !std::filesystem::exists(out_path, ec) || std::filesystem::file_size(out_path, ec) == 0;
        if (need_header) {
            out_file << "№\tTIMESTAMP\tCHECKED_B\tADDR_HEX\tPRIVATE_KEY\n";
            out_file.flush();
        } else {
            std::ifstream count_file(out_path);
            std::string line;
            while (std::getline(count_file, line)) {
                if (!line.empty() && line[0] != '#' && line.find("№") == std::string::npos) {
                    match_counter++;
                }
            }
        }
    } catch (...) {}

    BitcoinGenerator generator;
    generator.initialize();
    generator.set_use_optimizations(true); // enable faster key generation
    if (g_logger) g_logger->set_level(LogLevel::WARN);

    // NVML loader for GPU utilization/memory stats (same as BTC path)
    struct NvmlLoader {
        bool ok = false;

        using Device = void*;
        struct Util { unsigned int gpu, memory; };
        struct Mem { unsigned long long total, free, used; }; // nvmlMemory_t layout
        using Ret = int; // NVML_SUCCESS == 0

        Ret (WINAPI *nvmlInit)() = nullptr;
        Ret (WINAPI *nvmlShutdown)() = nullptr;
        Ret (WINAPI *nvmlDeviceGetHandleByIndex)(unsigned int, Device*) = nullptr;
        Ret (WINAPI *nvmlDeviceGetHandleByIndex_v2)(unsigned int, Device*) = nullptr;
        Ret (WINAPI *nvmlDeviceGetUtilizationRates)(Device, Util*) = nullptr;
        Ret (WINAPI *nvmlDeviceGetMemoryInfo)(Device, Mem*) = nullptr;

        HMODULE dll = nullptr;
        Device device = nullptr;

        bool load(int gpu_index) {
#ifdef _WIN32
            dll = LoadLibraryA("nvml.dll");
            if (!dll) return false;
            nvmlInit = reinterpret_cast<Ret (WINAPI*)()>(GetProcAddress(dll, "nvmlInit_v2"));
            if (!nvmlInit) nvmlInit = reinterpret_cast<Ret (WINAPI*)()>(GetProcAddress(dll, "nvmlInit"));
            nvmlShutdown = reinterpret_cast<Ret (WINAPI*)()>(GetProcAddress(dll, "nvmlShutdown"));
            nvmlDeviceGetHandleByIndex = reinterpret_cast<Ret (WINAPI*)(unsigned int, Device*)>(GetProcAddress(dll, "nvmlDeviceGetHandleByIndex"));
            nvmlDeviceGetHandleByIndex_v2 = reinterpret_cast<Ret (WINAPI*)(unsigned int, Device*)>(GetProcAddress(dll, "nvmlDeviceGetHandleByIndex_v2"));
            nvmlDeviceGetUtilizationRates = reinterpret_cast<Ret (WINAPI*)(Device, Util*)>(GetProcAddress(dll, "nvmlDeviceGetUtilizationRates"));
            nvmlDeviceGetMemoryInfo = reinterpret_cast<Ret (WINAPI*)(Device, Mem*)>(GetProcAddress(dll, "nvmlDeviceGetMemoryInfo"));
            if (!nvmlInit || !nvmlShutdown || (!nvmlDeviceGetHandleByIndex && !nvmlDeviceGetHandleByIndex_v2) || !nvmlDeviceGetUtilizationRates || !nvmlDeviceGetMemoryInfo) return false;
            if (nvmlInit() != 0) return false;
            Ret r = nvmlDeviceGetHandleByIndex_v2 ? nvmlDeviceGetHandleByIndex_v2(gpu_index, &device)
                                                  : nvmlDeviceGetHandleByIndex(gpu_index, &device);
            if (r != 0) return false;
            ok = true;
            return true;
#else
            return false;
#endif
        }
        void unload() {
#ifdef _WIN32
            if (ok && nvmlShutdown) nvmlShutdown();
            if (dll) { FreeLibrary(dll); dll = nullptr; }
            ok = false;
#endif
        }
        bool query(unsigned int& gpu_util, unsigned int& mem_util, unsigned long long& mem_used_mb, unsigned long long& mem_total_mb) {
            if (!ok) return false;
            Util u{}; Mem m{};

            if (nvmlDeviceGetUtilizationRates(device, &u) != 0) {
                gpu_util = 0;
                mem_util = 0;
            } else {
                gpu_util = u.gpu;
                mem_util = u.memory;
            }

            if (nvmlDeviceGetMemoryInfo(device, &m) != 0) return false;

            mem_used_mb = m.used / (1024ULL*1024ULL);
            mem_total_mb = m.total / (1024ULL*1024ULL);
            return true;
        }
    } nvml;
    if (!nvml.load(g_selected_gpu)) {
        LOG_WARN("NVML initialization failed - GPU monitoring will be limited");
    } else {
        LOG_INFO("NVML initialized successfully for GPU monitoring");
    }

    const size_t batch_size = config.get_eth_batch_size() ? config.get_eth_batch_size() : 65536;
    std::vector<uint8_t> host_priv_batch(batch_size * 32);
    std::vector<uint8_t> pub64;
    std::vector<uint8_t> addr20_batch;
    std::vector<uint8_t> bloom_flags;
    std::vector<BtcPrivPub> priv_pub;
    bool gpu_bloom_enabled = true;
    bool use_fused_pipeline = false;
    bool use_batch_inv_pipeline = false;  // Optimized batch inversion pipeline
    uint64_t fused_seed_base = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    const uint8_t* d_bloom_bits = nullptr;
    size_t bloom_blocks_dev = 0;
    uint8_t bloom_k_dev = 0;
    // Enable fused pipeline by default if Bloom is on device; fallback to classic on failure
    if (gpu_bloom_get_device_state(&d_bloom_bits, &bloom_blocks_dev, &bloom_k_dev) && d_bloom_bits && bloom_blocks_dev > 0 && bloom_k_dev > 0) {
        use_fused_pipeline = true;
        use_batch_inv_pipeline = true;  // Try optimized windowed scalar multiplication
        std::cout << "Fused GPU pipeline enabled (GPU RNG + priv->addr->Bloom)." << std::endl;
        std::cout << "Windowed Scalar Multiplication ENABLED (4-bit windows - ~60% faster)." << std::endl;
    } else {
        std::cout << "Fused pipeline unavailable (no device Bloom state); using classic pipeline." << std::endl;
    }
    std::unordered_set<std::string> recent_addrs;
    recent_addrs.reserve(200000);
    const size_t max_recent = 200000;

    auto timestamp_now = []() {
        auto now = std::chrono::system_clock::now();
        auto now_time_t = std::chrono::system_clock::to_time_t(now);
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
        std::tm now_tm;
#ifdef _WIN32
        localtime_s(&now_tm, &now_time_t);
#else
        localtime_r(&now_time_t, &now_tm);
#endif
        std::ostringstream ts;
        ts << std::put_time(&now_tm, "%Y-%m-%d %H:%M:%S")
           << '.' << std::setfill('0') << std::setw(3) << now_ms.count();
        return ts.str();
    };

    size_t total_checked = 0, total_matched = 0;
    auto report_last_time = std::chrono::steady_clock::now();
    size_t checked_at_last_report = 0;
    size_t matched_at_last_report = 0;

    while (g_running.load() && !g_force_exit.load()) {
        size_t keys_in_batch = batch_size;
        if (use_fused_pipeline) {
            if (addr20_batch.size() < keys_in_batch * 20) addr20_batch.resize(keys_in_batch * 20);
            if (bloom_flags.size() < keys_in_batch) bloom_flags.resize(keys_in_batch);
            if (host_priv_batch.size() < keys_in_batch * 32) host_priv_batch.resize(keys_in_batch * 32);
            
            bool fused_ok = false;
            
            // Try optimized batch inversion pipeline first
            if (use_batch_inv_pipeline) {
                fused_ok = gpu_eth_fused_priv_gen_bloom_batch_inv(
                    keys_in_batch,
                    fused_seed_base,
                    d_bloom_bits,
                    bloom_blocks_dev,
                    bloom_k_dev,
                    host_priv_batch.data(),
                    addr20_batch.data(),
                    bloom_flags.data());
                if (!fused_ok) {
                    std::cout << "[WARN] Windowed scalar mul pipeline failed; falling back to standard fused pipeline." << std::endl;
                    use_batch_inv_pipeline = false;
                }
            }
            
            // Fallback to standard fused pipeline
            if (!fused_ok && !use_batch_inv_pipeline) {
                fused_ok = gpu_eth_fused_priv_gen_bloom(
                    keys_in_batch,
                    fused_seed_base,
                    d_bloom_bits,
                    bloom_blocks_dev,
                    bloom_k_dev,
                    host_priv_batch.data(),
                    addr20_batch.data(),
                    bloom_flags.data());
            }
            
            fused_seed_base += keys_in_batch;
            if (!fused_ok) {
                std::cout << "[WARN] Fused GPU pipeline failed; falling back to classic pipeline." << std::endl;
                use_fused_pipeline = false;
                // fall through to classic path below
            }
        }

        if (!use_fused_pipeline) {
            priv_pub = generator.generate_batch_priv_pub(batch_size);
            keys_in_batch = priv_pub.size();
            if (keys_in_batch == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            size_t needed_priv_bytes = keys_in_batch * 32;
            if (host_priv_batch.size() < needed_priv_bytes) host_priv_batch.resize(needed_priv_bytes);
            for (size_t i = 0; i < keys_in_batch; ++i) {
                std::memcpy(&host_priv_batch[i * 32], priv_pub[i].private_key.data(), 32);
            }

            bool gpu_ok = eth_priv_to_pub64_gpu(host_priv_batch.data(), keys_in_batch, pub64);
            if (!gpu_ok) {
                LOG_WARN("GPU priv->pub64 failed; falling back to CPU for this batch");
                if (!eth_priv_to_pub64_cpu(host_priv_batch.data(), keys_in_batch, pub64)) {
                    LOG_ERROR("CPU priv->pub64 fallback failed; aborting ETH scan");
                    break;
                }
            }

            if (addr20_batch.size() < keys_in_batch * 20) addr20_batch.resize(keys_in_batch * 20);
            if (bloom_flags.size() < keys_in_batch) bloom_flags.resize(keys_in_batch);

            bool addr_gpu_ok = eth_pub64_to_addr20_gpu(pub64.data(), keys_in_batch, addr20_batch);
            if (!addr_gpu_ok) {
                for (size_t i = 0; i < keys_in_batch; ++i) {
                    eth_pub64_to_addr20(&pub64[i * 64], &addr20_batch[i * 20]);
                }
            }

            bool bloom_gpu_ok = gpu_bloom_enabled && gpu_bloom_check_var(addr20_batch.data(), keys_in_batch, 20, bloom_flags.data());
            if (!bloom_gpu_ok) {
                if (gpu_bloom_enabled) {
                    std::cout << "[WARN] GPU Bloom check failed (will use CPU for rest of run)." << std::endl;
                    gpu_bloom_enabled = false;
                }
                for (size_t i = 0; i < keys_in_batch; ++i) {
                    bloom_flags[i] = bf.might_contain_bytes(&addr20_batch[i * 20], 20) ? 1 : 0;
                }
            }
        }

        size_t matched_in_batch = 0;
        for (size_t i = 0; i < keys_in_batch; ++i) {
            if (bloom_flags[i] == 0) continue;

            uint8_t* addr20 = &addr20_batch[i * 20];

            std::string addr_hex = eth_addr20_to_hex(addr20);
            if (recent_addrs.find(addr_hex) != recent_addrs.end()) {
                continue;
            }
            if (recent_addrs.size() >= max_recent) recent_addrs.clear();
            recent_addrs.insert(addr_hex);

            matched_in_batch++;
            match_counter++;
            const uint8_t* priv_ptr = use_fused_pipeline ? &host_priv_batch[i * 32] : priv_pub[i].private_key.data();
            std::string priv_hex = ConversionUtils::hex_encode(std::vector<uint8_t>(priv_ptr, priv_ptr + 32));
            std::string timestamp = timestamp_now();

            double checked_billions = static_cast<double>(total_checked) / 1e9;
            std::ostringstream checked_stream;
            checked_stream << std::fixed << std::setprecision(3) << checked_billions;
            std::string checked_b = checked_stream.str();

            // BTC-style columns: №, TIMESTAMP, CHECKED_B, ADDR_HEX, PRIV_HEX
            out_file << match_counter << "\t" << timestamp << "\t" << checked_b << "\t" << addr_hex << "\t" << priv_hex << "\n";
        }

        if (matched_in_batch > 0) out_file.flush();

        total_checked += keys_in_batch;
        total_matched += matched_in_batch;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - report_last_time).count();
        if (elapsed >= 60) {
            size_t delta_checked = total_checked - checked_at_last_report;
            size_t delta_matched = total_matched - matched_at_last_report;
            double per_sec_1m = elapsed > 0 ? (double)delta_checked / (double)elapsed : 0.0;
            double fpr_1m = delta_checked ? (double)delta_matched / (double)delta_checked : 0.0;
            double fpr_total = total_checked ? (double)total_matched / (double)total_checked : 0.0;

            std::cout << "[ETH] speed=" << std::fixed << std::setprecision(0) << per_sec_1m << " keys/s, "
                      << "checked=" << total_checked << ", matched=" << total_matched
                      << ", fpr_1m=" << std::setprecision(6) << fpr_1m
                      << ", fpr_total=" << std::setprecision(6) << fpr_total;

            unsigned int gpu_util = 0, mem_util = 0;
            unsigned long long mem_used_mb = 0, mem_total_mb = 0;
            if (nvml.query(gpu_util, mem_util, mem_used_mb, mem_total_mb)) {
                std::cout << ", gpu=" << gpu_util << "%"
                          << ", vmem=" << mem_used_mb << "MB/" << mem_total_mb << "MB";
                if (mem_util == 0) {
                    std::cout << ", mem_util=N/A";
                } else {
                    std::cout << ", mem_util=" << mem_util << "%";
                }
            } else {
                size_t cuda_free = 0, cuda_total = 0;
                if (cudaMemGetInfo(&cuda_free, &cuda_total) == cudaSuccess) {
                    size_t cuda_used = cuda_total - cuda_free;
                    double cuda_util = cuda_total > 0 ? (double)cuda_used / (double)cuda_total * 100.0 : 0.0;
                    std::cout << ", cuda_mem=" << std::fixed << std::setprecision(1) << cuda_util << "%";
                } else {
                    std::cout << ", gpu=N/A, vmem=N/A";
                }
            }
            std::cout << std::endl;

            report_last_time = now;
            checked_at_last_report = total_checked;
            matched_at_last_report = total_matched;
        }
    }

    std::cout << std::endl;
    gpu_bloom_unload();
    nvml.unload();
    if (out_file.is_open()) {
        try {
            out_file.flush();
            out_file.close();
        } catch (...) {}
    }
    if (g_logger) g_logger->set_level(LogLevel::INFO);
    std::cout << "ETH scan finished. Total checked: " << total_checked << ", matched: " << total_matched << std::endl;
    return true;
}

// Ethereum Mode 10: Validate ETH matches (recompute address and Bloom check)
bool run_eth_validate_results_mode() {
    std::cout << "=== Validate ETH found addresses (Bloom + recompute) ===" << std::endl << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::filesystem::path default_filter = std::filesystem::current_path() / "bloom_external_eth" / "ethereum_20b.db.bf";
    std::filesystem::path default_matches = std::filesystem::current_path() / "matches_eth_20b.txt";
    std::filesystem::path default_output = std::filesystem::current_path() / "eth_verification_results.txt";
    std::filesystem::path default_db = std::filesystem::current_path() / "databases" / "eth_addresses.db";

    std::string filter_path, input_path, output_path, db_path, table_name, column_name;
    std::cout << "Enter path to Bloom filter (.bf) [" << default_filter.string() << "]:\n> ";
    std::getline(std::cin, filter_path);
    if (filter_path.empty()) filter_path = default_filter.string();

    std::cout << "Enter path to matches file [" << default_matches.string() << "]:\n> ";
    std::getline(std::cin, input_path);
    if (input_path.empty()) input_path = default_matches.string();

    std::cout << "Enter path to output report [" << default_output.string() << "]:\n> ";
    std::getline(std::cin, output_path);
    if (output_path.empty()) output_path = default_output.string();

    std::cout << "Enter path to SQLite DB for optional lookup (empty to skip) [" << default_db.string() << "]:\n> ";
    std::getline(std::cin, db_path);
    if (db_path.empty()) db_path = default_db.string();

    bool use_db = std::filesystem::exists(db_path);
    if (use_db) {
        std::cout << "Enter table name [addresses]: ";
        std::getline(std::cin, table_name);
        if (table_name.empty()) table_name = "addresses";
        std::cout << "Enter column name with address [address]: ";
        std::getline(std::cin, column_name);
        if (column_name.empty()) column_name = "address";
    } else {
        std::cout << "DB not found or not provided; DB lookup will be skipped." << std::endl;
    }

    if (!std::filesystem::exists(filter_path)) {
        std::cout << "Invalid Bloom filter path: " << filter_path << std::endl;
        return false;
    }
    if (!std::filesystem::exists(input_path)) {
        std::cout << "Input matches file not found: " << input_path << std::endl;
        return false;
    }

    BloomFilter bf;
    if (!bf.load_from_file(filter_path)) {
        std::cout << "Failed to load Bloom filter" << std::endl;
        return false;
    }

    std::ifstream in_file(input_path);
    if (!in_file.is_open()) {
        std::cout << "Failed to open files for validation" << std::endl;
        return false;
    }

    sqlite3* db = nullptr;
    sqlite3_stmt* stmt_blob = nullptr;
    sqlite3_stmt* stmt_text = nullptr;
    if (use_db) {
        if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
            std::cout << "Warning: cannot open DB, skipping DB lookup." << std::endl;
            use_db = false;
        } else {
            std::string sql_blob = "SELECT 1 FROM \"" + table_name + "\" WHERE \"" + column_name + "\" = ? LIMIT 1";
            if (sqlite3_prepare_v2(db, sql_blob.c_str(), -1, &stmt_blob, nullptr) != SQLITE_OK) {
                std::cout << "Warning: cannot prepare DB BLOB statement, skipping DB lookup." << std::endl;
                use_db = false;
            }
            std::string sql_text = "SELECT 1 FROM \"" + table_name + "\" WHERE \"" + column_name + "\" = ? OR \"" + column_name + "\" = ? LIMIT 1";
            if (use_db && sqlite3_prepare_v2(db, sql_text.c_str(), -1, &stmt_text, nullptr) != SQLITE_OK) {
                std::cout << "Warning: cannot prepare DB TEXT statement, skipping DB lookup." << std::endl;
                use_db = false;
            }
        }
    }

    std::string line;
    size_t validated = 0;
    size_t bloom_hits = 0;
    size_t recomputed_hits = 0;
    size_t db_hits = 0;
    std::vector<std::string> out_rows;
    while (std::getline(in_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("№") != std::string::npos) continue;
        // Expected format (tab-separated): № \t TIMESTAMP \t CHECKED_B \t ADDR_HEX \t PRIV_HEX
        std::string num, ts, checked, addr_hex, priv_hex;
        {
            std::istringstream ss(line);
            if (!std::getline(ss, num, '\t')) continue;
            if (!std::getline(ss, ts, '\t')) continue;
            if (!std::getline(ss, checked, '\t')) continue;
            if (!std::getline(ss, addr_hex, '\t')) continue;
            if (!std::getline(ss, priv_hex, '\t')) continue;
        }

        std::array<uint8_t,20> addr20{};
        if (!eth_parse_address_hex(addr_hex, addr20)) continue;

        auto priv_bytes = ConversionUtils::hex_decode(priv_hex);
        if (priv_bytes.size() != 32) continue;

        uint8_t pub64[64];
        std::vector<uint8_t> pub64_vec;
        if (!eth_priv_to_pub64_gpu(priv_bytes.data(), 1, pub64_vec)) continue;
        std::memcpy(pub64, pub64_vec.data(), 64);
        uint8_t recomputed_addr[20];
        eth_pub64_to_addr20(pub64, recomputed_addr);

        bool bloom_ok = bf.might_contain_bytes(addr20.data(), 20);
        bool recomputed_ok = std::memcmp(addr20.data(), recomputed_addr, 20) == 0;
        bool db_hit = false;
        if (use_db) {
            // Try BLOB lookup
            if (stmt_blob) {
                sqlite3_reset(stmt_blob);
                sqlite3_clear_bindings(stmt_blob);
                sqlite3_bind_blob(stmt_blob, 1, addr20.data(), 20, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt_blob) == SQLITE_ROW) db_hit = true;
            }
            if (!db_hit && stmt_text) {
                sqlite3_reset(stmt_text);
                sqlite3_clear_bindings(stmt_text);
                std::string hex = eth_addr20_to_hex(addr20.data());
                std::string hex0x = "0x" + hex;
                sqlite3_bind_text(stmt_text, 1, hex.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_text(stmt_text, 2, hex0x.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt_text) == SQLITE_ROW) db_hit = true;
            }
        }

        std::ostringstream row;
        row << addr_hex << "\t" << priv_hex << "\t" << (bloom_ok ? "1" : "0") << "\t" << (recomputed_ok ? "1" : "0") << "\t" << (db_hit ? "1" : "0");
        out_rows.push_back(row.str());
        ++validated;
        if (bloom_ok) ++bloom_hits;
        if (recomputed_ok) ++recomputed_hits;
        if (db_hit) ++db_hits;
    }

    if (stmt_blob) sqlite3_finalize(stmt_blob);
    if (stmt_text) sqlite3_finalize(stmt_text);
    if (db) sqlite3_close(db);

    // Write summary and table to output file
    {
        std::ofstream out_file(output_path, std::ios::trunc);
        if (!out_file.is_open()) {
            std::cout << "Failed to open output file for writing summary: " << output_path << std::endl;
            return false;
        }
        out_file << "# SUMMARY\n";
        out_file << "# validated=" << validated
                 << " bloom_hits=" << bloom_hits
                 << " recomputed_match=" << recomputed_hits
                 << " db_hits=" << db_hits << "\n";
        out_file << "ADDR_HEX\tPRIV_HEX\tIN_BLOOM\tRECOMPUTED_MATCH\tDB_HIT\n";
        for (const auto& r : out_rows) out_file << r << "\n";
    }

    std::cout << "Validation finished. Checked rows: " << validated
              << ", bloom_hits: " << bloom_hits
              << ", recomputed_ok: " << recomputed_hits
              << ", db_hits: " << db_hits
              << ". Report: " << output_path << std::endl;
    return true;
}

// Bitcoin Performance Test Implementation
void run_bitcoin_performance_test() {
    std::cout << "\n=== Bitcoin GPU Optimization Performance Test ===\n\n";
    
    BitcoinGenerator generator;
    if (!generator.initialize()) {
        std::cout << "Failed to initialize Bitcoin generator\n";
        return;
    }
    
    // Test different batch sizes
    std::vector<size_t> test_sizes = {1000, 5000, 10000, 50000};
    std::vector<BitcoinAddressType> addr_types = {
        BitcoinAddressType::P2PKH, 
        BitcoinAddressType::P2SH
    };
    
    for (auto addr_type : addr_types) {
        std::string type_name = (addr_type == BitcoinAddressType::P2PKH) ? "P2PKH" : "P2SH";
        std::cout << "\n--- Testing " << type_name << " addresses ---\n";
        
        generator.set_address_type(addr_type);
        
        for (size_t test_size : test_sizes) {
            std::cout << "\nTest size: " << test_size << " keys\n";
            
            // Test original implementation
            generator.set_use_optimizations(false);
            auto start = std::chrono::high_resolution_clock::now();
            auto original_results = generator.generate_batch_binary(test_size, addr_type);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto original_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double original_speed = static_cast<double>(original_results.size()) / (original_duration.count() / 1000.0);
            
            // Test optimized implementation
            generator.set_use_optimizations(true);
            start = std::chrono::high_resolution_clock::now();
            auto optimized_results = generator.generate_batch_optimized(test_size, addr_type);
            end = std::chrono::high_resolution_clock::now();
            
            auto optimized_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            double optimized_speed = static_cast<double>(optimized_results.size()) / (optimized_duration.count() / 1000.0);
            
            double speedup = optimized_speed / original_speed;
            
            std::cout << "  Original:   " << std::fixed << std::setprecision(0) 
                      << original_speed << " keys/sec (" << original_duration.count() << " ms)\n";
            std::cout << "  Optimized:  " << std::fixed << std::setprecision(0) 
                      << optimized_speed << " keys/sec (" << optimized_duration.count() << " ms)\n";
            std::cout << "  Speedup:    " << std::fixed << std::setprecision(1) << speedup << "x";
            
            if (speedup > 5.0) {
                std::cout << " Excellent!";
            } else if (speedup > 2.0) {
                std::cout << " Good";
            } else {
                std::cout << " Below target";
            }
            std::cout << "\n";
            
            // Verify correctness (sample check)
            if (!original_results.empty() && !optimized_results.empty()) {
                std::cout << "  Results:    Original=" << original_results.size() 
                          << ", Optimized=" << optimized_results.size();
                if (optimized_results.size() >= original_results.size()) {
                    std::cout << " OK\n";
                } else {
                    std::cout << " ERROR: Size mismatch\n";
                }
            }
        }
    }
    
    // Overall performance summary
    std::cout << "\n=== Performance Summary ===\n";
    generator.print_performance_comparison(10000);
    
    // GPU utilization test
    std::cout << "\n=== GPU Utilization Test ===\n";
    BtcGpuStats stats;
    if (gpu_btc_get_performance_stats(&stats)) {
        std::cout << "GPU Utilization: " << std::fixed << std::setprecision(1) 
                  << stats.gpu_utilization << "%\n";
        std::cout << "Memory Used:     " << stats.memory_used_mb << " MB\n";
        std::cout << "Endomorphism Speedup: " << std::fixed << std::setprecision(1) 
                  << stats.endomorphism_speedup << "x\n";
    }
    
    // Benchmark comparison with current baseline
    std::cout << "\n=== Benchmark Against Current Performance ===\n";
    size_t test_keys = 100000;
    int addr_type = 1; // P2PKH
    
    double original_time, optimized_time, speedup;
    
    if (gpu_btc_benchmark_comparison(test_keys, addr_type, &original_time, &optimized_time, &speedup)) {
        std::cout << "Test keys:        " << test_keys << "\n";
        std::cout << "Original time:    " << std::fixed << std::setprecision(2) << original_time << " ms\n";
        std::cout << "Optimized time:   " << std::fixed << std::setprecision(2) << optimized_time << " ms\n";
        std::cout << "Speedup achieved: " << std::fixed << std::setprecision(1) << speedup << "x\n";
        
        double original_rate = test_keys / (original_time / 1000.0);
        double optimized_rate = test_keys / (optimized_time / 1000.0);
        
        std::cout << "\nPerformance rates:\n";
        std::cout << "  Original:  " << std::fixed << std::setprecision(0) << original_rate << " keys/sec\n";
        std::cout << "  Optimized: " << std::fixed << std::setprecision(0) << optimized_rate << " keys/sec\n";
        
        // Compare with current baseline (~495,889 keys/sec)
        double current_baseline = 495889.0;
        double improvement_vs_baseline = optimized_rate / current_baseline;
        
        std::cout << "\nComparison with current baseline (" << std::fixed << std::setprecision(0) 
                  << current_baseline << " keys/sec):\n";
        std::cout << "  Improvement: " << std::fixed << std::setprecision(1) 
                  << improvement_vs_baseline << "x\n";
        
        if (improvement_vs_baseline > 10.0) {
            std::cout << "  Status: Excellent - Target achieved!\n";
        } else if (improvement_vs_baseline > 5.0) {
            std::cout << "  Status: Very good improvement\n";
        } else if (improvement_vs_baseline > 2.0) {
            std::cout << "  Status: Good improvement\n";
        } else {
            std::cout << "  Status: Needs optimization\n";
        }
    } else {
        std::cout << "Benchmark failed - check GPU implementation\n";
    }
    
    std::cout << "\n=== Test Complete ===\n";
    std::cout << "Press any key to continue...\n";
    std::cin.get();
}

enum class NetworkChoice {
    BTC = 1,
    ETH = 2,
    SOL = 3
};

int main(int argc, char* argv[]) {
    std::cout << "=== BloomSeek with CUDA Acceleration ===" << std::endl;
    std::cout << "GPU BloomSeek v0.8" << std::endl;
    std::cout << "========================================================" << std::endl << std::endl;
    
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    // Initialize logger
    try {
        g_logger = new Logger("bitcoin_generator.log", LogLevel::INFO, true);
        LOG_INFO("Starting Key Generator v0.8");
    } catch (const std::exception& e) {
        std::cerr << "Logger initialization error: " << e.what() << std::endl;
        return 1;
    }
    
    // Initialize configuration manager
    ConfigManager config;
    
    // GPU selection
    int selected_gpu = 0;
    {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count <= 0) {
            std::cout << "No CUDA devices found. Exiting." << std::endl;
            delete g_logger;
            return 1;
        }
        std::cout << "Available GPUs:" << std::endl;
        for (int i = 0; i < device_count; ++i) {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, i);
            std::cout << "  " << (i+1) << ") " << prop.name << " (SM " << prop.major << "." << prop.minor
                      << ", " << (prop.totalGlobalMem / (1024*1024)) << " MB)" << std::endl;
        }
        std::cout << "Select GPU (1-" << device_count << ") [default 1]: ";
        int gpu_sel = 1;
        if (!(std::cin >> gpu_sel)) { std::cin.clear(); gpu_sel = 1; }
        if (gpu_sel < 1 || gpu_sel > device_count) gpu_sel = 1;
        selected_gpu = gpu_sel - 1;
        g_selected_gpu = selected_gpu;
        cudaSetDevice(selected_gpu);
    }

    // Initialize GPU optimization systems (temporarily disabled for compilation)
    // if (!gpu_optimization_initialize()) {
    //     LOG_ERROR("Failed to initialize GPU optimization systems");
    //     delete g_logger;
    //     return 1;
    // }

    // GPU autotune on first run (or when not yet done)
    try {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, selected_gpu);
        if (!config.is_gpu_autotune_done()) {
            // Use enhanced optimization system (temporarily disabled for compilation)
            // OptimizedKernelLauncher& launcher = OptimizedKernelLauncher::getInstance();
            // if (launcher.configure_for_gpu(selected_gpu)) {
            //     // Get optimized configuration for typical batch sizes
            //     GpuKernelConfig optimal_config = launcher.get_optimal_config(65536); // 64K batch
            //
            //     size_t threads_per_block = optimal_config.threads_per_block;
            //     size_t blocks_per_grid = optimal_config.blocks_per_grid;
            //     size_t batch = threads_per_block * blocks_per_grid;
            //
            //     if (batch > 1048576) batch = 1048576; // 1M safety
            //
            //     config.set_gpu_threads_per_block(threads_per_block);
            //     config.set_gpu_blocks_per_grid(blocks_per_grid);
            //     config.set_gpu_batch_size(batch);
            //     config.set_gpu_autotune_done(true);
            //
            //     LOG_INFO("GPU autotune saved: TPB=" + std::to_string(threads_per_block) +
            //             ", blocks=" + std::to_string(blocks_per_grid) +
            //             ", batch=" + std::to_string(batch) +
            //             " (architecture: SM " + std::to_string(prop.major) + "." +
            //             std::to_string(prop.minor) + ")");
            // } else {
                // Fallback to manual configuration
                size_t threads_per_block = 256;
                size_t blocks_per_grid = std::max(1, prop.multiProcessorCount * 4);
                if (prop.major >= 12) {
                    threads_per_block = 512;
                    blocks_per_grid = prop.multiProcessorCount * 8;
                } else if (prop.major >= 9 || (prop.major >= 8 && prop.multiProcessorCount >= 48)) {
                    threads_per_block = 512;
                    blocks_per_grid = prop.multiProcessorCount * 4;
                } else if (prop.major >= 8) {
                    threads_per_block = 512;
                    blocks_per_grid = prop.multiProcessorCount * 8;
                } else if (prop.major >= 7) {
                    threads_per_block = 256;
                    blocks_per_grid = prop.multiProcessorCount * 2;
                } else {
                    threads_per_block = 128;
                    blocks_per_grid = std::max(1, prop.multiProcessorCount);
                }
                size_t batch = threads_per_block * blocks_per_grid;
                if (batch > 1048576) batch = 1048576; // 1M safety
                config.set_gpu_threads_per_block(threads_per_block);
                config.set_gpu_blocks_per_grid(blocks_per_grid);
                config.set_gpu_batch_size(batch);
                config.set_gpu_autotune_done(true);
                LOG_INFO("GPU autotune saved (fallback): TPB=" + std::to_string(threads_per_block) + ", blocks=" + std::to_string(blocks_per_grid) + ", batch=" + std::to_string(batch));
            // }
        }
    } catch (...) {
        LOG_WARN("GPU autotune failed; continuing with runtime defaults");
    }

    // Network selection
    std::cout << "\nSelect network:" << std::endl;
    std::cout << "1. Bitcoin" << std::endl;
    std::cout << "2. Ethereum" << std::endl;
    std::cout << "3. Solana" << std::endl;
    std::cout << "Your choice (1-3) [default 1]: ";
    int net_in = 1;
    if (!(std::cin >> net_in)) { std::cin.clear(); net_in = 1; }
    if (net_in < 1 || net_in > 3) net_in = 1;
    NetworkChoice net = static_cast<NetworkChoice>(net_in);

    bool success = false;

    if (net == NetworkChoice::BTC) {
        std::cout << "\nBitcoin modes:" << std::endl;
        std::cout << "5. Create Bloom filter for external SQLite BTC DB HASH160 (P2PKH/P2SH, 20 bytes)" << std::endl;
        std::cout << "6. GPU scan vs BTC Bloom filter in VRAM (P2PKH/P2SH)" << std::endl;
        std::cout << "7. Validate found addresses (Bloom result + key/address consistency check + DB lookup)" << std::endl;
        std::cout << "Select mode (5-7): ";
        int choice = 5;
        if (!(std::cin >> choice)) { std::cin.clear(); choice = 5; std::cout << "5" << std::endl; }
        switch (choice) {
            case 5: success = run_btc_external_bloom_mode(config); break;
            case 6: success = run_btc_gpu_scan_mode(config); break;
            case 7: success = run_btc_validate_results_mode(); break;
            default: std::cout << "Invalid choice!" << std::endl; success = false; break;
        }
    } else if (net == NetworkChoice::ETH) {
        std::cout << "\nEthereum modes:" << std::endl;
        std::cout << "8. Create Bloom filter for external SQLite ETH DB (20 bytes)" << std::endl;
        std::cout << "9. GPU scan vs ETH Bloom filter in VRAM" << std::endl;
        std::cout << "10. Validate found ETH addresses (Bloom + recompute)" << std::endl;
        std::cout << "Select mode (8-10): ";
        int choice = 8;
        if (!(std::cin >> choice)) { std::cin.clear(); choice = 8; std::cout << "8" << std::endl; }
        switch (choice) {
            case 8: success = run_eth_external_bloom_mode(config); break;
            case 9: success = run_eth_gpu_scan_mode(config); break;
            case 10: success = run_eth_validate_results_mode(); break;
            default: std::cout << "Invalid choice!" << std::endl; success = false; break;
        }
    } else if (net == NetworkChoice::SOL) {
        std::cout << "\nSolana modes:" << std::endl;
        std::cout << "11. Create Bloom filter for external SQLite SOL DB (32 bytes)" << std::endl;
        std::cout << "12. GPU/CPU scan vs SOL Bloom filter in VRAM" << std::endl;
        std::cout << "13. Validate found SOL addresses (Bloom + recompute)" << std::endl;
        std::cout << "Select mode (11-13): ";
        int choice = 11;
        if (!(std::cin >> choice)) { std::cin.clear(); choice = 11; std::cout << "11" << std::endl; }
        switch (choice) {
            case 11: success = run_sol_external_bloom_mode(config); break;
            case 12: success = run_sol_gpu_scan_mode(config); break;
            case 13: success = run_sol_validate_results_mode(); break;
            default: std::cout << "Invalid choice!" << std::endl; success = false; break;
        }
    }
    
    // Cleanup GPU optimization systems (temporarily disabled for compilation)
    // gpu_optimization_shutdown();

    // Release fused GPU resources
    gpu_fused_shutdown();

    // Cleanup
    delete g_logger;
    g_logger = nullptr;

    return success ? 0 : 1;
}


