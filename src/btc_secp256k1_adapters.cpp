#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>

// Forward declaration of GPU function
extern "C" bool gpu_secp256k1_priv_to_pub64(const uint8_t* host_priv32, size_t num, uint8_t* host_pub64);

// Compress uncompressed pub (X||Y, 64 bytes) -> compressed 33 bytes (0x02/0x03 || X)
static inline void compress_pub64_to_pub33(const uint8_t in64[64], uint8_t out33[33]) {
    // X is first 32 bytes
    std::memcpy(out33 + 1, in64, 32);
    // Y is last 32 bytes; parity by LSB of Y
    const uint8_t y_lsb = in64[63] & 1U;
    out33[0] = y_lsb ? 0x03 : 0x02;
}

extern "C" bool gpu_secp256k1_priv_to_pub33(const uint8_t* host_priv32, size_t num, uint8_t* host_pub33) {
    if (!host_priv32 || !host_pub33 || num == 0) return false;
    // 1) Use existing GPU path to get uncompressed 64-byte public keys
    std::vector<uint8_t> pub64(num * 64);
    if (!gpu_secp256k1_priv_to_pub64(host_priv32, num, pub64.data())) {
        return false;
    }
    // 2) Compress each point on CPU in parallel
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(num); ++i) {
        compress_pub64_to_pub33(&pub64[i * 64], &host_pub33[i * 33]);
    }
    return true;
}


