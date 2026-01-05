#include "solana_modes.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#ifdef ERROR
#undef ERROR
#endif
#ifdef DEBUG
#undef DEBUG
#endif
#endif

#include <cuda_runtime.h>
#include <sqlite3.h>

#include "bloom_filter.h"
#include "btc_gpu_optimized.h"
#include "conversion_utils.h"
#include "logger.h"
#include "sol_utils.h"
#include <cstring>

extern std::atomic<bool> g_running;
extern std::atomic<bool> g_force_exit;
extern int g_selected_gpu;

// ================= SOLANA MODES =================
// Solana Mode 11: external DB -> SOL Bloom filter (32B pubkeys)
bool run_sol_external_bloom_mode(ConfigManager& config) {
    std::cout << "=== SOL EXTERNAL BLOOM FILTER MODE ===" << std::endl << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::string bloom_dir = "bloom_external_sol";
    try {
        if (!std::filesystem::exists(bloom_dir)) {
            std::filesystem::create_directories(bloom_dir);
        }
    } catch (...) {}

    std::string db_path, table_name, column_name;
    std::string default_db = config.get_database_directory() + "/sol_addresses.db";
    std::cout << "Enter path to SQLite DB with SOL addresses [" << default_db << "]:\n> ";
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

    size_t key_size = 32;
    uint64_t db_file_size_bytes = std::filesystem::file_size(db_path);

    sqlite3* db = nullptr;
    if (sqlite3_open_v2(db_path.c_str(), &db, SQLITE_OPEN_READONLY, nullptr) != SQLITE_OK) {
        std::cout << "Failed to open DB: " << db_path << std::endl;
        return false;
    }

    // Heuristic: ~36 bytes per row for 32B blob + SQLite overhead (empirically ~125M rows for 4.4GB DB)
    const double bytes_per_row_est = 36.0;
    size_t fast_estimate_rows = static_cast<size_t>(db_file_size_bytes / bytes_per_row_est);
    std::cout << "\nDatabase size: " << (db_file_size_bytes / (1024 * 1024)) << " MB" << std::endl;
    std::cout << "Fast estimate (based on file size): ~" << fast_estimate_rows << " rows." << std::endl;
    std::cout << "Perform precise row count? (y/N): ";
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
        std::cout << (i + 1) << ") " << options[i].name << "  ~" << (bytes / (1024 * 1024)) << " MB" << std::endl;
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

    std::string save_path = bloom_dir + "/solana_32b.db.bf";
    std::cout << "\nCreating Bloom filter..." << std::endl;
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

    std::array<uint8_t, 32> addr32{};
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const void* addr_data = sqlite3_column_blob(stmt, 0);
        int addr_len = sqlite3_column_bytes(stmt, 0);
        bool ok = sol_parse_address_bytes_or_b58(addr_data, addr_len, addr32);
        if (!ok) {
            const unsigned char* txt = sqlite3_column_text(stmt, 0);
            int txt_len = sqlite3_column_bytes(stmt, 0);
            ok = sol_parse_address_bytes_or_b58(txt, txt_len, addr32);
        }
        if (ok) {
            bf.add_bytes(addr32.data(), key_size);
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
        std::cout << "  Size: " << (stats.size_bytes / (1024 * 1024)) << " MB (" << stats.size_bytes << " bytes)" << std::endl;
        std::cout << "  Expected elements: " << stats.expected_elements << std::endl;
        std::cout << "  FPR: " << stats.false_positive_rate << std::endl;
        std::cout << "  Blocks: " << stats.blocks_count << ", Hashes: " << (int)stats.hash_functions << std::endl;
    } else {
        std::cout << "Filter created in '" << bloom_dir << "' with FPR=" << fpr << ". File: " << save_path << std::endl;
    }
    return true;
}

// Solana Mode 12: GPU/CPU scan against SOL Bloom filter
bool run_sol_gpu_scan_mode(ConfigManager& config) {
    std::cout << "=== SOL GPU BLOOM SCAN MODE (Mode 12) ===" << std::endl << std::endl;
    std::cout << "Load a previously created SOL Bloom filter (from Mode 11)." << std::endl;

    cudaSetDevice(g_selected_gpu);
    // Reduce log noise during high-frequency GPU Bloom checks
    if (g_logger) g_logger->set_level(LogLevel::WARN);

    // Lightweight NVML loader for GPU utilization/memory stats (optional)
    struct NvmlLoader {
        bool ok = false;
#ifdef _WIN32
        using Device = void*;
        struct Util { unsigned int gpu, memory; };
        struct Mem { unsigned long long total, free, used; }; // nvmlMemory_t layout
        using Ret = int; // NVML_SUCCESS == 0

        Ret(WINAPI* nvmlInit)() = nullptr;
        Ret(WINAPI* nvmlShutdown)() = nullptr;
        Ret(WINAPI* nvmlDeviceGetHandleByIndex)(unsigned int, Device*) = nullptr;
        Ret(WINAPI* nvmlDeviceGetHandleByIndex_v2)(unsigned int, Device*) = nullptr;
        Ret(WINAPI* nvmlDeviceGetUtilizationRates)(Device, Util*) = nullptr;
        Ret(WINAPI* nvmlDeviceGetMemoryInfo)(Device, Mem*) = nullptr;

        HMODULE dll = nullptr;
        Device device = nullptr;

        bool load(int gpu_index) {
            dll = LoadLibraryA("nvml.dll");
            if (!dll) return false;
            nvmlInit = reinterpret_cast<Ret(WINAPI*)()>(GetProcAddress(dll, "nvmlInit_v2"));
            if (!nvmlInit) nvmlInit = reinterpret_cast<Ret(WINAPI*)()>(GetProcAddress(dll, "nvmlInit"));
            nvmlShutdown = reinterpret_cast<Ret(WINAPI*)()>(GetProcAddress(dll, "nvmlShutdown"));
            nvmlDeviceGetHandleByIndex = reinterpret_cast<Ret(WINAPI*)(unsigned int, Device*)>(GetProcAddress(dll, "nvmlDeviceGetHandleByIndex"));
            nvmlDeviceGetHandleByIndex_v2 = reinterpret_cast<Ret(WINAPI*)(unsigned int, Device*)>(GetProcAddress(dll, "nvmlDeviceGetHandleByIndex_v2"));
            nvmlDeviceGetUtilizationRates = reinterpret_cast<Ret(WINAPI*)(Device, Util*)>(GetProcAddress(dll, "nvmlDeviceGetUtilizationRates"));
            nvmlDeviceGetMemoryInfo = reinterpret_cast<Ret(WINAPI*)(Device, Mem*)>(GetProcAddress(dll, "nvmlDeviceGetMemoryInfo"));
            if (!nvmlInit || !nvmlShutdown || (!nvmlDeviceGetHandleByIndex && !nvmlDeviceGetHandleByIndex_v2) ||
                !nvmlDeviceGetUtilizationRates || !nvmlDeviceGetMemoryInfo)
                return false;
            if (nvmlInit() != 0) return false;
            Ret r = nvmlDeviceGetHandleByIndex_v2 ? nvmlDeviceGetHandleByIndex_v2(gpu_index, &device)
                                                  : nvmlDeviceGetHandleByIndex(gpu_index, &device);
            if (r != 0) return false;
            ok = true;
            return true;
        }
        void unload() {
            if (ok && nvmlShutdown) nvmlShutdown();
            if (dll) { FreeLibrary(dll); dll = nullptr; }
            ok = false;
        }
        bool query(unsigned int& gpu_util, unsigned int& mem_util, unsigned long long& mem_used_mb, unsigned long long& mem_total_mb) {
            if (!ok) return false;
            Util u{}; Mem m{};
            if (nvmlDeviceGetUtilizationRates(device, &u) != 0) return false;
            if (nvmlDeviceGetMemoryInfo(device, &m) != 0) return false;
            gpu_util = u.gpu;
            mem_util = u.memory;
            mem_used_mb = m.used / (1024ULL * 1024ULL);
            mem_total_mb = m.total / (1024ULL * 1024ULL);
            return true;
        }
#else
        bool load(int) { return false; }
        void unload() {}
        bool query(unsigned int&, unsigned int&, unsigned long long&, unsigned long long&) { return false; }
#endif
    } nvml;
    nvml.load(g_selected_gpu);

    std::filesystem::path default_filter = std::filesystem::current_path() / "bloom_external_sol" / "solana_32b.db.bf";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::string filter_path;
    std::cout << "Enter path to Bloom filter (.bf) [" << default_filter.string() << "]:\n> ";
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

    std::cout << "Filter info: size=" << (stats.size_bytes / (1024 * 1024)) << " MB, blocks=" << blocks
              << ", k=" << (int)k_hashes << ", expected_elements=" << stats.expected_elements
              << ", target FPR=" << stats.false_positive_rate << std::endl;

    if (!gpu_bloom_load(bits.data(), bits.size(), blocks, k_hashes)) {
        std::cout << "Failed to upload Bloom filter to GPU" << std::endl;
        return false;
    }
    LOG_INFO("SOL Mode12: Bloom filter uploaded to GPU: " + std::to_string(bits.size()) + " bytes, blocks=" + std::to_string(blocks));

    std::string out_path = "matches_sol_32b.txt";
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
            out_file << "№\tTIMESTAMP\tCHECKED_B\tADDR_B58\tPRIVATE_KEY_HEX\n";
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

    const size_t batch_size = config.get_eth_batch_size() ? config.get_eth_batch_size() : 65536;
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

    bool gpu_bloom_enabled = true;
    std::vector<uint8_t> priv_flat;
    std::vector<uint8_t> pub32;
    std::vector<uint8_t> bloom_flags;
    uint64_t seed_base = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    while (g_running.load() && !g_force_exit.load()) {
        size_t keys_in_batch = batch_size;
        bool gpu_gen_ok = sol_generate_priv_pub_gpu(keys_in_batch, seed_base, priv_flat, pub32);
        seed_base += keys_in_batch;
        if (!gpu_gen_ok) {
            auto privpub = sol_generate_batch_cpu(batch_size);
            keys_in_batch = privpub.size();
            if (keys_in_batch == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            priv_flat.resize(keys_in_batch * 32);
            for (size_t i = 0; i < keys_in_batch; ++i) {
                std::memcpy(&priv_flat[i * 32], privpub[i].priv.data(), 32);
            }
            if (!sol_priv_to_pub_gpu(priv_flat.data(), keys_in_batch, pub32)) {
                pub32.resize(keys_in_batch * 32);
                for (size_t i = 0; i < keys_in_batch; ++i) {
                    sol_priv_to_pub_cpu(privpub[i].priv.data(), &pub32[i * 32]);
                }
            }
        }

        // GPU Bloom check if available; otherwise CPU Bloom
        bloom_flags.assign(keys_in_batch, 0);
        bool bloom_gpu_ok = gpu_bloom_enabled &&
                            gpu_bloom_check_var(pub32.data(), keys_in_batch, 32, bloom_flags.data());
        if (!bloom_gpu_ok) {
            gpu_bloom_enabled = false;
            for (size_t i = 0; i < keys_in_batch; ++i) {
                if (bf.might_contain_bytes(&pub32[i * 32], 32)) bloom_flags[i] = 1;
            }
        }

        size_t matched_in_batch = 0;
        for (size_t i = 0; i < keys_in_batch; ++i) {
            if (!bloom_flags[i]) continue;
            const uint8_t* addr32 = &pub32[i * 32];
            std::string addr_b58 = sol_addr32_to_b58(addr32);
            if (recent_addrs.find(addr_b58) != recent_addrs.end()) continue;
            if (recent_addrs.size() >= max_recent) recent_addrs.clear();
            recent_addrs.insert(addr_b58);

            matched_in_batch++;
            match_counter++;
            std::string priv_hex = ConversionUtils::hex_encode(std::vector<uint8_t>(&priv_flat[i * 32], &priv_flat[i * 32 + 32]));
            std::string timestamp = timestamp_now();
            double checked_billions = static_cast<double>(total_checked) / 1e9;
            std::ostringstream checked_stream;
            checked_stream << std::fixed << std::setprecision(3) << checked_billions;
            std::string checked_b = checked_stream.str();

            out_file << match_counter << "\t" << timestamp << "\t" << checked_b << "\t" << addr_b58 << "\t" << priv_hex << "\n";
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

            std::ostringstream fpr1s, fprtots;
            fpr1s << std::scientific << std::setprecision(2) << fpr_1m;
            fprtots << std::scientific << std::setprecision(2) << fpr_total;

            std::cout << "[1m][SOL] speed=" << std::fixed << std::setprecision(0) << per_sec_1m << " keys/s, "
                      << "checked=" << total_checked << ", matched=" << total_matched
                      << ", fpr_1m=" << fpr1s.str()
                      << ", fpr_total=" << fprtots.str();

            // NVML-based utilization if available, otherwise fallback to cudaMemGetInfo for memory
            unsigned int gpu_util = 0, mem_util = 0;
            unsigned long long mem_used_mb = 0, mem_total_mb = 0;
            bool nvml_ok = nvml.query(gpu_util, mem_util, mem_used_mb, mem_total_mb);
            if (nvml_ok) {
                std::cout << ", gpu=" << gpu_util << "%, vmem=" << mem_used_mb << "MB/" << mem_total_mb << "MB";
            } else {
                size_t cuda_free = 0, cuda_total = 0;
                if (cudaMemGetInfo(&cuda_free, &cuda_total) == cudaSuccess) {
                    size_t cuda_used = cuda_total - cuda_free;
                    double cuda_util = cuda_total > 0 ? (double)cuda_used / (double)cuda_total * 100.0 : 0.0;
                    std::cout << ", cuda_mem=" << std::fixed << std::setprecision(1) << cuda_util << "%"
                              << " (" << (cuda_used / (1024 * 1024)) << "MB/"
                              << (cuda_total / (1024 * 1024)) << "MB)";
                } else {
                    std::cout << ", cuda_mem=N/A";
                }
            }

            std::cout << ", batch=" << keys_in_batch;
            std::cout << std::endl;

            report_last_time = now;
            checked_at_last_report = total_checked;
            matched_at_last_report = total_matched;
        }
    }

    std::cout << std::endl;
    gpu_bloom_unload();
    if (out_file.is_open()) {
        try { out_file.flush(); out_file.close(); } catch (...) {}
    }
    std::cout << "SOL scan finished. Total checked: " << total_checked << ", matched: " << total_matched << std::endl;
    return true;
}

// Solana Mode 13: Validate SOL matches
bool run_sol_validate_results_mode() {
    std::cout << "=== Validate SOL found addresses (Bloom + recompute) ===" << std::endl << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::filesystem::path default_filter = std::filesystem::current_path() / "bloom_external_sol" / "solana_32b.db.bf";
    std::filesystem::path default_matches = std::filesystem::current_path() / "matches_sol_32b.txt";
    std::filesystem::path default_output = std::filesystem::current_path() / "sol_verification_results.txt";
    std::filesystem::path default_db = std::filesystem::current_path() / "databases" / "sol_addresses.db";

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
    std::ofstream out_file(output_path);
    if (!in_file.is_open() || !out_file.is_open()) {
        std::cout << "Failed to open files for validation" << std::endl;
        return false;
    }

    out_file << "ADDR_B58\tPRIV_HEX\tIN_BLOOM\tRECOMPUTED_MATCH\tDB_HIT\n";

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
            std::string sql_text = "SELECT 1 FROM \"" + table_name + "\" WHERE \"" + column_name + "\" = ? LIMIT 1";
            if (use_db && sqlite3_prepare_v2(db, sql_text.c_str(), -1, &stmt_text, nullptr) != SQLITE_OK) {
                std::cout << "Warning: cannot prepare DB TEXT statement, skipping DB lookup." << std::endl;
                use_db = false;
            }
        }
    }

    std::string line;
    size_t validated = 0;
    while (std::getline(in_file, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.find("№") != std::string::npos) continue;

        // Matches file is tab-delimited: № \t TIMESTAMP \t CHECKED_B \t ADDR_B58 \t PRIV_HEX
        std::vector<std::string> cols;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, '\t')) {
            cols.push_back(cell);
        }
        if (cols.size() < 5) continue;

        const std::string& addr_b58 = cols[3];
        const std::string& priv_hex = cols[4];

        std::array<uint8_t, 32> addr32{};
        if (!sol_parse_address_b58(addr_b58, addr32)) continue;
        auto priv_bytes = ConversionUtils::hex_decode(priv_hex);
        if (priv_bytes.size() != 32) continue;

        uint8_t pub32[32];
        if (!sol_priv_to_pub_cpu(priv_bytes.data(), pub32)) continue;

        bool bloom_ok = bf.might_contain_bytes(addr32.data(), 32);
        bool recomputed_ok = std::memcmp(addr32.data(), pub32, 32) == 0;
        bool db_hit = false;
        if (use_db) {
            if (stmt_blob) {
                sqlite3_reset(stmt_blob);
                sqlite3_clear_bindings(stmt_blob);
                sqlite3_bind_blob(stmt_blob, 1, addr32.data(), 32, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt_blob) == SQLITE_ROW) db_hit = true;
            }
            if (!db_hit && stmt_text) {
                sqlite3_reset(stmt_text);
                sqlite3_clear_bindings(stmt_text);
                sqlite3_bind_text(stmt_text, 1, addr_b58.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(stmt_text) == SQLITE_ROW) db_hit = true;
            }
        }

        out_file << addr_b58 << "\t" << priv_hex << "\t" << (bloom_ok ? "1" : "0") << "\t" << (recomputed_ok ? "1" : "0") << "\t" << (db_hit ? "1" : "0") << "\n";
        std::cout << "[SOL VALID] addr=" << addr_b58
                  << " priv=" << priv_hex
                  << " in_bloom=" << (bloom_ok ? "1" : "0")
                  << " recomputed=" << (recomputed_ok ? "1" : "0")
                  << " db_hit=" << (db_hit ? "1" : "0")
                  << std::endl;
        validated++;
    }

    if (stmt_blob) sqlite3_finalize(stmt_blob);
    if (stmt_text) sqlite3_finalize(stmt_text);
    if (db) sqlite3_close(db);

    std::cout << "Validation finished. Checked rows: " << validated << ". Report: " << output_path << std::endl;
    return true;
}

