#pragma once

#include "config_manager.h"

// Solana-specific workflows (Bloom filter build, GPU scan, validation).
bool run_sol_external_bloom_mode(ConfigManager& config);
bool run_sol_gpu_scan_mode(ConfigManager& config);
bool run_sol_validate_results_mode();

