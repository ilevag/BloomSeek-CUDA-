BloomSeek CUDA Key Generator
============================

High-performance GPU (CUDA) tool for generating keys and matching addresses across Bitcoin, Ethereum, and Solana. Uses Bloom filters stored in GPU VRAM for fast membership checks, supports configurable false-positive rates, and can post-verify matches against the original address databases.

Features
--------
- GPU-accelerated key generation and Bloom filter lookups for BTC, ETH, and SOL.
- Bloom filter creation from large SQLite databases with configurable FPR (e.g., 1e-6..1e-9).
- Validation/verification flows that recompute addresses and optionally check against the source DB.
- Result reports (`matches_*.txt`, `*_verification_results.txt`) with optional Python verifier for BTC.
- Uses CUDA VRAM for Bloom filters; warns when filters become multi-GB.

Requirements
------------
- CUDA-capable GPU and recent NVIDIA drivers.
- CUDA Toolkit with CMake integration (CMakeLists targets SM 86/89 by default; adjust as needed).
- CMake and a C++17 toolchain (e.g., Visual Studio with CUDA support on Windows).
- SQLite3 and OpenSSL libraries (found via CMake or provided DLLs in the repo).
- Large SQLite databases are **not** included (expected under `databases/`).

Third-Party Code (submodules/vendor)
------------------------------------
- Submodule `third_party/eth-vanity-cuda` (ETH GPU primitives).
- Submodule `third_party/solanity` (Ed25519 CUDA/SGX code used for SOL).
- Vendored `third_party/VanitySearch-1.1` sources (BTC address utilities).
Check each project for its license terms before redistribution.

Quick Start (Windows example)
-----------------------------
```powershell
git clone --recurse-submodules https://github.com/your-account/your-repo.git
cd your-repo
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
# Run interactive app (place DLLs next to the exe if needed)
cd build/Release
.\BitcoinGenerator.exe
```

Runtime Data & Artifacts
------------------------
- SQLite DBs expected in `databases/` (e.g., `p2pkh_addresses.db`, `p2sh_addresses.db`, `eth_addresses.db`). These are large and not tracked.
- Bloom filters written/read from `bloom_external_btc/` and `bloom_external_eth/` (ignored by Git).
- Results: `matches_btc_*.txt`, `matches_eth_*.txt`, `*_verification_results.txt` (ignored by Git).

Configuration
-------------
- On first run, `config.ini` is created (not tracked). Key options:
  - `database_directory` (default `./databases`)
  - `bloom_filter_directory` (default `./bloom_filters`)
  - GPU tuning flags (autotune markers, optional manual overrides for threads/blocks/batch sizes)

Usage Examples
--------------
- Interactive main app: run `BitcoinGenerator.exe` (or `run.bat`) and follow prompts to:
  - Build Bloom filters from DBs (BTC/ETH).
  - Scan with GPU against existing Bloom filters.
  - Validate matches (recompute addresses, optional DB lookup).
- BTC DB verification via Python:
```bash
python btc_db_verifier.py --input matches_btc_20b.txt --db databases/p2pkh_addresses.db --bloom bloom_external_btc/bitcoin_p2pkh_20b.db.bf
```
- Quick DB inspection:
```bash
python _tmp_inspect.py databases/eth_addresses.db
```

DBs details
-----------------
BTC - 602 mln lines (BLOB HASH160 P2PKH)
Link (https://disk.yandex.ru/d/L0V7saw_NPo4IQ)
ETH - 252 mln lines (BLOB)
Link (https://disk.yandex.ru/d/kd4HYxmsWWooTQ)
SOL - 126 mln lines (BLOB)
Link (https://disk.yandex.ru/d/nrELoQvJlJsBpw)

For DBs were parsed all addresses, that had at least 1 transaction till july 2025)

Performance Notes
-----------------
- Lower FPR (e.g., 1e-9) increases Bloom filter size; multi-GB filters will consume significant VRAM/host RAM.
- Ensure adequate free VRAM before loading filters; the app prints warnings for large allocations.

Safety / Disclaimer
-------------------
Do not run on real private keys or sensitive data you cannot risk. The software is provided as-is; you are responsible for complying with local laws and third-party licenses.
# BloomSeek-CUDA-

