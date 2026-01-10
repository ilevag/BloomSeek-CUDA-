cmake_minimum_required(VERSION 3.18)
project(BitcoinGenerator LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Найти CUDA
find_package(CUDA REQUIRED)
enable_language(CUDA)

# Настройки CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Найти OpenSSL для secp256k1 криптографии
find_package(OpenSSL REQUIRED)

# Найти SQLite3 через vcpkg
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SQLITE3 QUIET sqlite3)
endif()

if(NOT SQLITE3_FOUND)
    find_package(SQLite3 QUIET)
    if(SQLite3_FOUND)
        set(SQLITE3_LIBRARIES SQLite::SQLite3)
    else()
        # Fallback: поиск библиотек напрямую
        find_library(SQLITE3_LIBRARIES NAMES sqlite3 PATHS
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/sqlite"
            "C:/sqlite3"
        )
        find_path(SQLITE3_INCLUDE_DIRS NAMES sqlite3.h PATHS
            "C:/vcpkg/installed/x64-windows/include"
            "C:/sqlite"
            "C:/sqlite3"
        )
    endif()
endif()

# Включить директории
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/Solgen)
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# Исходные файлы (Bitcoin only)
set(SOURCES
    src/main.cpp
    src/config_manager.cpp
    src/logger.cpp
    src/base58.cpp
    src/bloom_filter.cpp
    src/conversion_utils.cpp
    src/bitcoin_generator.cpp
    src/btc_address.cpp
    src/btc_secp256k1_adapters.cpp
    src/eth_utils.cpp
    Solgen/solana_modes.cpp
    Solgen/sol_utils.cpp
)

set(CUDA_SOURCES
    src/btc_gpu_optimized.cu
    src/eth_secp256k1_gpu.cu
    Solgen/sol_gpu_ed25519.cu
    third_party/solanity/src/cuda-ecc-ed25519/fe.cu
    third_party/solanity/src/cuda-ecc-ed25519/ge.cu
    third_party/solanity/src/cuda-ecc-ed25519/sc.cu
    third_party/solanity/src/cuda-ecc-ed25519/seed.cu
    third_party/solanity/src/cuda-ecc-ed25519/sha512.cu
    third_party/solanity/src/cuda-ecc-ed25519/keypair.cu
)

# Include solanity headers
include_directories(${CMAKE_SOURCE_DIR}/third_party/solanity/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party/solanity/src/cuda-headers)
include_directories(${CMAKE_SOURCE_DIR}/third_party/solanity/src/sgx-ecc-ed25519)

# Создать исполняемый файл
add_executable(${PROJECT_NAME} ${SOURCES} ${CUDA_SOURCES})
# Переименовать выходной бинарник
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "BloomSeek")

if (CUDA_curand_LIBRARY)
    set(CURAND_LIB ${CUDA_curand_LIBRARY})
else()
    set(CURAND_LIB curand)
endif()

# Линковать библиотеки
target_link_libraries(${PROJECT_NAME} 
    ${CUDA_LIBRARIES}
    ${CURAND_LIB}
    ${SQLITE3_LIBRARIES}
    OpenSSL::SSL 
    OpenSSL::Crypto
)

# Ensure stdint.h path in solanity fixedint.h under MSVC + nvcc
target_compile_definitions(${PROJECT_NAME} PRIVATE
    $<$<OR:$<COMPILE_LANGUAGE:CUDA>,$<COMPILE_LANGUAGE:CXX>>:__STDC__=1 __STDC_VERSION__=199901L>
)

# Опции компиляции CUDA
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "86;89") # RTX 3050/5070

# Копилятор флаги
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    $<$<COMPILE_LANGUAGE:CXX>:/openmp>
)

# Allow duplicate symbols from inline-less third-party headers on MSVC
# Pass via proper linker flags for CXX and CUDA device link
if (MSVC)
    target_link_options(${PROJECT_NAME} PRIVATE
        $<$<LINK_LANGUAGE:CXX>:/FORCE:MULTIPLE>
        $<$<LINK_LANGUAGE:CUDA>:-Xlinker /FORCE:MULTIPLE>
    )
endif()

# Директории включения для SQLite3
if(SQLITE3_INCLUDE_DIRS)
    target_include_directories(${PROJECT_NAME} PRIVATE ${SQLITE3_INCLUDE_DIRS})
endif()
