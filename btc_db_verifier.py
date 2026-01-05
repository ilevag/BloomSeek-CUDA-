#!/usr/bin/env python3
"""
Bitcoin Database Verifier
Проверяет адреса из matches_btc_20b.txt на наличие в базе данных p2pkh_addresses.db
и (опционально) сверяет с Bloom-фильтром.
"""

import sys
import time
import sqlite3
import argparse
import hashlib
import struct
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# === Cryptography helpers for secp256k1 ===
SECP256K1_P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
SECP256K1_GX = 55066263022277343669578718895168534326250603453777594175500187360389116729240
SECP256K1_GY = 32670510020758816978083085130507043184471273380659243275938904335757337482424
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
BECH32_CHARSET = "qpzry9x8gf2tvdw0s3jn54khce6mua7l"


def _modinv(value: int, modulus: int) -> int:
    """Modular inverse using Fermat little theorem (modulus is prime)."""
    return pow(value, modulus - 2, modulus)


def _point_add(p1: Optional[Tuple[int, int]], p2: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Elliptic curve point addition for secp256k1."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1

    x1, y1 = p1
    x2, y2 = p2

    if x1 == x2 and (y1 + y2) % SECP256K1_P == 0:
        return None

    if p1 == p2:
        if y1 % SECP256K1_P == 0:
            return None
        s = (3 * x1 * x1) * _modinv((2 * y1) % SECP256K1_P, SECP256K1_P)
    else:
        dx = (x2 - x1) % SECP256K1_P
        if dx == 0:
            return None
        s = (y2 - y1) * _modinv(dx, SECP256K1_P)

    s %= SECP256K1_P
    x3 = (s * s - x1 - x2) % SECP256K1_P
    y3 = (s * (x1 - x3) - y1) % SECP256K1_P
    return x3, y3


def _scalar_mult(k: int, point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Multiply a point by an integer using double-and-add."""
    result = None
    addend = point
    while k:
        if k & 1:
            result = _point_add(result, addend)
        addend = _point_add(addend, addend)
        k >>= 1
    return result


def private_key_to_public_key(priv_key_bytes: bytes, compressed: bool = True) -> bytes:
    """Derive a secp256k1 public key from a 32-byte private key."""
    if len(priv_key_bytes) != 32:
        raise ValueError("Private key must be 32 bytes (64 hex characters)")

    priv_int = int.from_bytes(priv_key_bytes, "big")
    if not (1 <= priv_int < SECP256K1_N):
        raise ValueError("Private key is out of range")

    point = _scalar_mult(priv_int, (SECP256K1_GX, SECP256K1_GY))
    if point is None:
        raise ValueError("Failed to derive public key")

    x, y = point
    x_bytes = x.to_bytes(32, "big")
    if compressed:
        prefix = 0x02 | (y & 1)
        return bytes([prefix]) + x_bytes
    return b"\x04" + x_bytes + y.to_bytes(32, "big")


def hash160(data: bytes) -> bytes:
    """RIPEMD160(SHA256(data))."""
    sha = hashlib.sha256(data).digest()
    ripemd = hashlib.new("ripemd160")
    ripemd.update(sha)
    return ripemd.digest()


def base58check_encode(version: int, payload: bytes) -> str:
    """Base58Check encoding with provided version byte."""
    data = bytes([version]) + payload
    checksum = hashlib.sha256(hashlib.sha256(data).digest()).digest()[:4]
    return base58_encode(data + checksum)


def base58_encode(data: bytes) -> str:
    """Encode bytes into Base58 string."""
    value = int.from_bytes(data, "big")
    encoded = ""
    while value > 0:
        value, remainder = divmod(value, 58)
        encoded = BASE58_ALPHABET[remainder] + encoded

    # Preserve leading zeros as '1'
    padding = 0
    for byte in data:
        if byte == 0:
            padding += 1
        else:
            break
    return "1" * padding + encoded


def _bech32_polymod(values):
    generator = [0x3b6a57b2, 0x26508e6d, 0x1ea119fa, 0x3d4233dd, 0x2a1462b3]
    chk = 1
    for value in values:
        top = chk >> 25
        chk = (chk & 0x1FFFFFF) << 5 ^ value
        for i in range(5):
            if (top >> i) & 1:
                chk ^= generator[i]
    return chk


def _bech32_hrp_expand(hrp: str):
    return [ord(x) >> 5 for x in hrp] + [0] + [ord(x) & 31 for x in hrp]


def _convert_bits(data, from_bits, to_bits, pad=True):
    acc = 0
    bits = 0
    ret = []
    maxv = (1 << to_bits) - 1
    for value in data:
        if value < 0 or value >> from_bits:
            raise ValueError("Invalid value for bit conversion")
        acc = (acc << from_bits) | value
        bits += from_bits
        while bits >= to_bits:
            bits -= to_bits
            ret.append((acc >> bits) & maxv)
    if pad:
        if bits:
            ret.append((acc << (to_bits - bits)) & maxv)
    elif bits >= from_bits or ((acc << (to_bits - bits)) & maxv):
        raise ValueError("Invalid padding in bit conversion")
    return ret


def bech32_address_from_hash160(hash160_bytes: bytes) -> str:
    """Construct a Bech32 P2WPKH address from HASH160 bytes."""
    hrp = "bc"
    data = [0] + _convert_bits(hash160_bytes, 8, 5)
    combined = data + _bech32_create_checksum(hrp, data)
    return hrp + "1" + "".join(BECH32_CHARSET[d] for d in combined)


def _bech32_create_checksum(hrp, data):
    values = _bech32_hrp_expand(hrp) + data
    values += [0, 0, 0, 0, 0, 0]
    polymod = _bech32_polymod(values) ^ 1
    return [(polymod >> 5 * (5 - i)) & 31 for i in range(6)]


def derive_expected_values(private_key_hex: str):
    """Return (hash160_hex, p2pkh_address, bech32_address) derived from the key."""
    priv_bytes = bytes.fromhex(private_key_hex)
    pubkey = private_key_to_public_key(priv_bytes, compressed=True)
    hash160_bytes = hash160(pubkey)
    return (
        hash160_bytes.hex(),
        base58check_encode(0x00, hash160_bytes),
        bech32_address_from_hash160(hash160_bytes),
    )


def parse_matches_file(filename: str) -> List[Dict]:
    """
    Парсит файл matches_btc_20b.txt и извлекает данные

    Args:
        filename: Путь к файлу с результатами

    Returns:
        Список словарей с данными
    """
    matches = []

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()

                # Пропускаем заголовок и пустые строки
                if not line or line.startswith('№') or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) >= 6:  # Минимум нужно 6 колонок
                    match_data = {
                        'line_number': line_num,
                        'number': parts[0].strip(),
                        'timestamp': parts[1].strip(),
                        'checked_b': parts[2].strip(),
                        'hash160_hex': parts[3].strip(),  # HASH160 в hex формате
                        'private_key': parts[4].strip(),
                        'address_main': parts[5].strip(),
                        'address_bech32': parts[6].strip() if len(parts) > 6 else ''
                    }
                    matches.append(match_data)

    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        sys.exit(1)

    return matches


def verify_match_fields(match: Dict) -> Tuple[bool, List[str]]:
    """Ensure private key, HASH160 and addresses correspond to each other."""
    errors = []
    hash160_hex = match.get('hash160_hex', '').lower()
    private_key_hex = match.get('private_key', '').lower()
    address_main = match.get('address_main', '')
    address_bech32 = match.get('address_bech32', '')

    if len(hash160_hex) != 40:
        errors.append("HASH160 must be 40 hex characters")
        return False, errors

    if not private_key_hex:
        errors.append("PRIVATE_KEY is empty")
        return False, errors

    if len(private_key_hex) != 64:
        errors.append("PRIVATE_KEY must be 64 hex characters")
        return False, errors

    try:
        derived_hash, derived_p2pkh, derived_bech32 = derive_expected_values(private_key_hex)
    except ValueError as exc:
        errors.append(str(exc))
        return False, errors

    if hash160_hex != derived_hash:
        errors.append("HASH160 does not match derived HASH160")

    if not address_main:
        errors.append("ADDRESS_MAIN is empty")
    elif address_main != derived_p2pkh:
        errors.append("ADDRESS_MAIN does not match derived P2PKH address")

    if not address_bech32:
        errors.append("ADDRESS_BECH32 is empty")
    elif address_bech32.lower() != derived_bech32:
        errors.append("ADDRESS_BECH32 does not match derived Bech32 address")

    return len(errors) == 0, errors


def split_valid_invalid_matches(matches: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split matches into valid (consistent) and invalid (inconsistent) lists."""
    valid = []
    invalid = []
    for match in matches:
        is_valid, errors = verify_match_fields(match)
        if is_valid:
            valid.append(match)
        else:
            match['consistency_errors'] = errors
            invalid.append(match)
    return valid, invalid


# === Bloom filter loader/checker (compatible with C++ BloomFilter v2) ===
def fnv1a64(data: bytes, seed: int) -> int:
    FNV_OFFSET = 14695981039346656037 ^ seed
    FNV_PRIME = 1099511628211
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    # extra mixing as in C++
    h ^= (h >> 33)
    h = (h * 0xFF51AFD7ED558CCD) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    h = (h * 0xC4CEB9FE1A85EC53) & 0xFFFFFFFFFFFFFFFF
    h ^= (h >> 33)
    return h


def bloom_hashes_v2(data: bytes) -> Tuple[int, int]:
    h1 = fnv1a64(data, 0)
    h2 = fnv1a64(data, 0x9E3779B97F4A7C15)
    h2 = ((h2 << 31) & 0xFFFFFFFFFFFFFFFF) | (h2 >> (64 - 31))
    h2 ^= 0x9E3779B97F4A7C15
    return h1, h2


class BloomFilterFile:
    def __init__(self, path: str):
        self.path = path
        self.loaded = False
        self.blocks = 0
        self.k = 0
        self.bits = b""

    def load(self):
        with open(self.path, "rb") as f:
            data = f.read()
        off = 0
        magic, version = struct.unpack_from("<II", data, off)
        off += 8
        if magic != 0x424C4F4D:
            raise ValueError("Invalid bloom file magic")
        if version not in (1, 2):
            raise ValueError(f"Unsupported bloom version {version}")
        # sizes are little-endian; assume 64-bit size_t/double layout from C++
        m_bits, m_blocks = struct.unpack_from("<QQ", data, off)
        off += 16
        k_hashes = struct.unpack_from("<B", data, off)[0]
        off += 1
        # skip expected_elements (Q) and fpr (double)
        off += 8 + 8
        data_size = struct.unpack_from("<Q", data, off)[0]
        off += 8
        bit_array = data[off:off + data_size]
        if len(bit_array) != data_size:
            raise ValueError("Truncated bloom file")
        self.blocks = m_blocks
        self.k = k_hashes
        self.bits = bit_array
        self.loaded = True

    def might_contain(self, h160_hex: str) -> bool:
        if not self.loaded:
            raise RuntimeError("Bloom filter not loaded")
        key = bytes.fromhex(h160_hex)
        h1, h2 = bloom_hashes_v2(key)
        for i in range(8):  # max k=8
            if i >= self.k:
                break
            hv = (h1 + i * h2) & 0xFFFFFFFFFFFFFFFF
            block_index = hv % self.blocks
            bit_in_block = (hv >> 32) % 256
            byte_index = block_index * 32 + (bit_in_block // 8)
            bit_mask = 1 << (bit_in_block % 8)
            if byte_index >= len(self.bits):
                return False
            if (self.bits[byte_index] & bit_mask) == 0:
                return False
        return True


def hex_to_bytes(hex_str: str) -> bytes:
    """
    Конвертирует hex строку в bytes
    """
    try:
        return bytes.fromhex(hex_str)
    except ValueError as e:
        raise ValueError(f"Invalid hex string '{hex_str}': {e}")


def check_addresses_in_db(
    db_path: str,
    matches: List[Dict],
    batch_size: int = 1000,
    bloom: BloomFilterFile = None
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Проверяет наличие адресов в базе данных и, опционально, в Bloom фильтре
    Returns: (found, not_found, not_in_bloom)
    """
    found = []
    not_found = []
    not_in_bloom = []

    try:
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        cursor = conn.cursor()

        total_matches = len(matches)
        print(f"Checking {total_matches} addresses...")
        print(f"Database: {db_path}")
        if bloom:
            print(f"Bloom filter: {bloom.path}")
        print(f"Batch size: {batch_size}")
        print("-" * 60)

        start_time = time.time()

        for i in range(0, total_matches, batch_size):
            batch = matches[i:i + batch_size]
            batch_start = time.time()

            for match_data in batch:
                hash160_hex = match_data['hash160_hex']

                try:
                    hash160_bytes = hex_to_bytes(hash160_hex)

                    if bloom and not bloom.might_contain(hash160_hex):
                        not_in_bloom.append(match_data)

                    cursor.execute('SELECT 1 FROM addresses WHERE address = ? LIMIT 1', (sqlite3.Binary(hash160_bytes),))
                    result = cursor.fetchone()

                    if result:
                        found.append(match_data)
                    else:
                        not_found.append(match_data)

                except ValueError as e:
                    print(f"Warning: Skipping invalid hash160 '{hash160_hex}': {e}")
                    not_found.append(match_data)

            batch_time = time.time() - batch_start
            progress = (i + len(batch)) / total_matches * 100
            addresses_per_sec = len(batch) / batch_time if batch_time > 0 else 0

            print(f"[{progress:5.1f}%] batch {i//batch_size+1}: {len(batch)} addrs, "
                  f"{addresses_per_sec:,.0f} addr/s, elapsed {batch_time:.3f}s")

        total_time = time.time() - start_time
        avg_rate = total_matches / total_time if total_time > 0 else 0
        print(f"Done in {total_time:.2f}s, avg {avg_rate:,.0f} addr/s")

        conn.close()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    return found, not_found, not_in_bloom


def save_results(found: List[Dict], not_found: List[Dict], invalid: List[Dict], not_in_bloom: List[Dict], output_file: str, total_input: int):
    """Сохраняет результаты в файл"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Bitcoin Database Verification Results\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total records in input: {total_input}\n")
            f.write(f"# Valid records checked in DB: {len(found) + len(not_found)}\n")
            f.write(f"# Found in database: {len(found)}\n")
            f.write(f"# Not found in database: {len(not_found)}\n")
            f.write(f"# Invalid key/address pairs: {len(invalid)}\n")
            f.write(f"# Not in Bloom filter (if provided): {len(not_in_bloom)}\n")
            f.write("#" + "="*80 + "\n\n")

            if found:
                f.write("=== ADDRESSES FOUND IN DATABASE ===\n")
                f.write("№\tTIMESTAMP\tCHECKED_B\tHASH160\tPRIVATE_KEY\tADDRESS_MAIN\n")
                for match in found:
                    f.write(f"{match['number']}\t{match['timestamp']}\t{match['checked_b']}\t"
                            f"{match['hash160_hex']}\t{match['private_key']}\t{match['address_main']}\n")
                f.write("\n")

            if not_found:
                f.write("=== ADDRESSES NOT FOUND IN DATABASE ===\n")
                f.write("№\tTIMESTAMP\tCHECKED_B\tHASH160\tPRIVATE_KEY\tADDRESS_MAIN\n")
                for match in not_found:
                    f.write(f"{match['number']}\t{match['timestamp']}\t{match['checked_b']}\t"
                            f"{match['hash160_hex']}\t{match['private_key']}\t{match['address_main']}\n")

            if not_in_bloom:
                f.write("\n=== ADDRESSES NOT IN BLOOM FILTER ===\n")
                f.write("№\tTIMESTAMP\tCHECKED_B\tHASH160\tPRIVATE_KEY\tADDRESS_MAIN\n")
                for match in not_in_bloom:
                    f.write(f"{match['number']}\t{match['timestamp']}\t{match['checked_b']}\t"
                            f"{match['hash160_hex']}\t{match['private_key']}\t{match['address_main']}\n")

            if invalid:
                f.write("\n=== INVALID KEY/ADDRESS MATCHES (SKIPPED) ===\n")
                f.write("№\tTIMESTAMP\tHASH160\tPRIVATE_KEY\tADDRESS_MAIN\tADDRESS_BECH32\tERRORS\n")
                for match in invalid:
                    error_text = " | ".join(match.get('consistency_errors', []))
                    f.write(f"{match['number']}\t{match['timestamp']}\t{match['hash160_hex']}\t"
                            f"{match['private_key']}\t{match['address_main']}\t"
                            f"{match['address_bech32']}\t{error_text}\n")

        print(f"Results saved to: {output_file}")

    except Exception as e:
        print(f"Error saving results to file {output_file}: {e}")


def print_summary(found: List[Dict], not_found: List[Dict], invalid: List[Dict], not_in_bloom: List[Dict], total_input: int):
    """Выводит итоговую статистику"""
    total_valid = len(found) + len(not_found)

    print("\n" + "="*80)
    print("DATABASE VERIFICATION SUMMARY")
    print("="*80)
    print(f"Total records in input file: {total_input}")
    print(f"Valid addresses checked in DB: {total_valid}")
    if total_valid:
        print(f"Found in database: {len(found)} ({len(found)/total_valid*100:.1f}%)")
        print(f"Not found in database: {len(not_found)} ({len(not_found)/total_valid*100:.1f}%)")
    else:
        print("Found in database: 0 (0.0%)")
        print("Not found in database: 0 (0.0%)")
    print(f"Invalid key/address pairs skipped: {len(invalid)}")
    if total_valid:
        print(f"Not in Bloom filter: {len(not_in_bloom)} ({len(not_in_bloom)/total_valid*100:.1f}%)")
    else:
        print(f"Not in Bloom filter: {len(not_in_bloom)}")
    print("="*80)

    if found:
        print("\nFirst 5 found addresses:")
        for i, match in enumerate(found[:5], 1):
            print(f"{i}. {match['address_main']} (HASH160: {match['hash160_hex']})")

    if invalid:
        print("\nFirst invalid entries:")
        for match in invalid[:5]:
            errors = "; ".join(match.get('consistency_errors', []))
            print(f"Line {match['line_number']} ({match['number']}): {errors}")


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Verify Bitcoin addresses from matches_btc_20b.txt against p2pkh_addresses.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python btc_db_verifier.py
  python btc_db_verifier.py -i my_matches.txt -o verification_results.txt
  python btc_db_verifier.py --batch-size 500
  python btc_db_verifier.py --bloom bloom_external_btc/bitcoin_p2pkh_20b.db.bf

Database: p2pkh_addresses.db (602M records, 16GB+)
Input file format: matches_btc_20b.txt (TSV)
        """
    )

    parser.add_argument(
        '-i', '--input',
        default='matches_btc_20b.txt',
        help='Path to input file with matches (default: matches_btc_20b.txt)'
    )

    parser.add_argument(
        '-o', '--output',
        default='btc_db_verification_results.txt',
        help='Path to output file with results (default: btc_db_verification_results.txt)'
    )

    parser.add_argument(
        '--db',
        default='databases/p2pkh_addresses.db',
        help='Path to SQLite database (default: databases/p2pkh_addresses.db)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )

    parser.add_argument(
        '--bloom',
        default=None,
        help='Path to Bloom filter (.bf) to re-check membership (optional)'
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)

    if not Path(args.db).exists():
        print(f"Error: Database file {args.db} does not exist")
        sys.exit(1)

    print("Starting Bitcoin Database Verification")
    print(f"Input file: {args.input}")
    print(f"Database: {args.db}")
    print(f"Output file: {args.output}")
    if args.bloom:
        print(f"Bloom filter: {args.bloom}")
    print("-" * 80)

    try:
        matches = parse_matches_file(args.input)
        if not matches:
            print("No matches found in input file")
            return

        total_records = len(matches)
        print(f"Found {total_records} addresses to verify")

        valid_matches, invalid_matches = split_valid_invalid_matches(matches)
        if invalid_matches:
            print(f"Detected {len(invalid_matches)} invalid entries (key/address mismatch). They will be skipped.")
            for match in invalid_matches[:5]:
                errors = "; ".join(match.get('consistency_errors', []))
                print(f"  Line {match['line_number']} ({match['number']}): {errors}")

        if not valid_matches:
            print("No valid matches remain after key/address validation. Database verification skipped.")
            save_results([], [], invalid_matches, [], args.output, total_records)
            print_summary([], [], invalid_matches, [], total_records)
            return

        bloom = None
        if args.bloom:
            try:
                bloom = BloomFilterFile(args.bloom)
                bloom.load()
            except Exception as e:
                print(f"Warning: failed to load Bloom filter {args.bloom}: {e}")
                bloom = None

        found, not_found, not_in_bloom = check_addresses_in_db(args.db, valid_matches, args.batch_size, bloom)

        print(f"\nBloom mismatches (not in filter): {len(not_in_bloom)}")

        save_results(found, not_found, invalid_matches, not_in_bloom, args.output, total_records)

        print_summary(found, not_found, invalid_matches, not_in_bloom, total_records)

        if not_found:
            print(f"\nWARNING: {len(not_found)} addresses not found in database!")
            print("This may indicate a problem with the Bloom filter or database.")

    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()


