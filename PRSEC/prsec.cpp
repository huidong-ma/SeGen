#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <openssl/kdf.h>
#include <openssl/pem.h>
#include <openssl/rand.h>
#include <openssl/rsa.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <atomic>

#include <omp.h>

#include "bloom_filter.hpp"

namespace fs = std::filesystem;

namespace {
constexpr size_t kWindowLength = 50;
constexpr size_t kSessionKeyLength = 32;
constexpr size_t kCtrIvLength = 16;
constexpr size_t kGcmNonceLength = 12;
constexpr size_t kGcmTagLength = 16;
constexpr size_t kMacLength = 32;
constexpr std::array<unsigned char, 4> kMetadataMagic = {'S', 'G', 'M', 'D'};
constexpr unsigned char kFormatVersion = 2;
constexpr char kMetadataAad[] = "SeGen2-metadata-v1";
constexpr char kMacDomain[] = "SeGen2-hmac-v1";

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

std::vector<unsigned char> read_binary(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), "cannot open " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void write_binary(const fs::path& path, const std::vector<unsigned char>& data) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(output.good(), "cannot write " + path.string());
    output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    require(output.good(), "failed to write " + path.string());
}

std::string read_dna_sequence(const fs::path& path) {
    std::ifstream input(path);
    require(input.good(), "cannot open " + path.string());
    std::string sequence;
    std::getline(input, sequence);
    if (!sequence.empty() && sequence.back() == '\r') sequence.pop_back();
    require(!sequence.empty(), "input DNA sequence is empty");
    for (char base : sequence) {
        require(base == 'A' || base == 'C' || base == 'G' || base == 'T',
                "input must contain one A/C/G/T DNA sequence line");
    }
    return sequence;
}

unsigned char encode_base(char base) {
    switch (base) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: fail("invalid DNA base");
    }
}

char decode_base(unsigned char value) {
    static constexpr char bases[] = {'A', 'C', 'G', 'T'};
    return bases[value & 0x03];
}

void append_u32(std::vector<unsigned char>& output, uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) output.push_back(static_cast<unsigned char>(value >> shift));
}

void append_u64(std::vector<unsigned char>& output, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8) output.push_back(static_cast<unsigned char>(value >> shift));
}

uint32_t read_u32(const std::vector<unsigned char>& data, size_t& offset) {
    require(offset + 4 <= data.size(), "truncated metadata");
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) value = (value << 8) | data[offset++];
    return value;
}

uint64_t read_u64(const std::vector<unsigned char>& data, size_t& offset) {
    require(offset + 8 <= data.size(), "truncated metadata");
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) value = (value << 8) | data[offset++];
    return value;
}

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

bloom_filter load_dictionary(const fs::path& dictionary_path) {
    std::ifstream input(dictionary_path);
    require(input.good(), "cannot open sensitive dictionary " + dictionary_path.string());
    size_t entry_count = 0;
    std::string line;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty()) continue;
        if (line.size() != kWindowLength) continue;
        if (std::any_of(line.begin(), line.end(), [](char base) {
                return base != 'A' && base != 'C' && base != 'G' && base != 'T';
            })) continue;
        ++entry_count;
    }
    require(entry_count > 0, "sensitive dictionary contains no A/C/G/T 50-mers");

    bloom_parameters parameters;
    parameters.projected_element_count = entry_count + 10000;
    parameters.false_positive_probability = 0.0001;
    parameters.random_seed = 0xA5A5A5A5;
    require(!(!parameters), "invalid Bloom filter parameters");
    parameters.compute_optimal_parameters();
    bloom_filter filter(parameters);
    input.clear();
    input.seekg(0);
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.size() != kWindowLength) continue;
        if (std::any_of(line.begin(), line.end(), [](char base) {
                return base != 'A' && base != 'C' && base != 'G' && base != 'T';
            })) continue;
        filter.insert(line);
    }
    return filter;
}

std::vector<unsigned char> sensitive_bitmap(const std::string& sequence, const bloom_filter& filter,
                                             int threads, size_t& match_count) {
    std::vector<std::atomic_uint8_t> marked(sequence.size());
    for (auto& bit : marked) bit.store(0, std::memory_order_relaxed);
    unsigned long long matches = 0;
    const long long last_start = static_cast<long long>(sequence.size()) - static_cast<long long>(kWindowLength);
    if (last_start >= 0) {
#pragma omp parallel for num_threads(threads) reduction(+:matches) schedule(static)
        for (long long start = 0; start <= last_start; ++start) {
            const std::string window = sequence.substr(static_cast<size_t>(start), kWindowLength);
            if (!filter.contains(window)) continue;
            ++matches;
            // Atomic stores allow overlapping 50-mers owned by separate scan chunks.
            for (size_t pos = static_cast<size_t>(start); pos < static_cast<size_t>(start) + kWindowLength; ++pos)
                marked[pos].store(1, std::memory_order_relaxed);
        }
    }
    std::vector<unsigned char> bitmap(sequence.size());
    for (size_t i = 0; i < bitmap.size(); ++i) bitmap[i] = marked[i].load(std::memory_order_relaxed);
    match_count = static_cast<size_t>(matches);
    return bitmap;
}

std::vector<unsigned char> ctr_keystream(const std::array<unsigned char, kSessionKeyLength>& key,
                                          const std::array<unsigned char, kCtrIvLength>& iv, size_t length) {
    require(length <= static_cast<size_t>(INT_MAX), "sequence is too large for OpenSSL EVP update");
    EVP_CIPHER_CTX* context = EVP_CIPHER_CTX_new();
    require(context != nullptr, "cannot allocate AES-CTR context");
    std::vector<unsigned char> stream(length + EVP_CIPHER_block_size(EVP_aes_256_ctr()));
    int written = 0;
    int final_written = 0;
    const bool ok = EVP_EncryptInit_ex(context, EVP_aes_256_ctr(), nullptr, key.data(), iv.data()) == 1 &&
                    EVP_EncryptUpdate(context, stream.data(), &written, stream.data(), static_cast<int>(length)) == 1 &&
                    EVP_EncryptFinal_ex(context, stream.data() + written, &final_written) == 1;
    EVP_CIPHER_CTX_free(context);
    require(ok && static_cast<size_t>(written + final_written) >= length, "AES-CTR keystream generation failed");
    stream.resize(length);
    return stream;
}

std::string apply_mask(const std::string& sequence, const std::vector<unsigned char>& bitmap,
                       const std::vector<unsigned char>& stream, bool inverse) {
    require(sequence.size() == bitmap.size() && sequence.size() == stream.size(), "mask inputs have different lengths");
    std::string result = sequence;
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (!bitmap[i]) continue;
        const unsigned char x = encode_base(sequence[i]);
        const unsigned char r = stream[i] & 0x03;
        result[i] = decode_base(inverse ? (x + 4 - r) & 0x03 : (x + r) & 0x03);
    }
    return result;
}

std::vector<unsigned char> gcm_encrypt(const std::array<unsigned char, kSessionKeyLength>& key,
                                        const std::array<unsigned char, kGcmNonceLength>& nonce,
                                        const std::vector<unsigned char>& aad,
                                        const std::vector<unsigned char>& plain,
                                        std::array<unsigned char, kGcmTagLength>& tag) {
    require(plain.size() <= static_cast<size_t>(INT_MAX), "metadata is too large for OpenSSL EVP update");
    EVP_CIPHER_CTX* context = EVP_CIPHER_CTX_new();
    require(context != nullptr, "cannot allocate AES-GCM context");
    std::vector<unsigned char> cipher(plain.size());
    int ignored = 0, written = 0, final_written = 0;
    const bool ok = EVP_EncryptInit_ex(context, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1 &&
                    EVP_CIPHER_CTX_ctrl(context, EVP_CTRL_GCM_SET_IVLEN, kGcmNonceLength, nullptr) == 1 &&
                    EVP_EncryptInit_ex(context, nullptr, nullptr, key.data(), nonce.data()) == 1 &&
                    EVP_EncryptUpdate(context, nullptr, &ignored, aad.data(), static_cast<int>(aad.size())) == 1 &&
                    EVP_EncryptUpdate(context, cipher.data(), &written, plain.data(), static_cast<int>(plain.size())) == 1 &&
                    EVP_EncryptFinal_ex(context, cipher.data() + written, &final_written) == 1 &&
                    EVP_CIPHER_CTX_ctrl(context, EVP_CTRL_GCM_GET_TAG, kGcmTagLength, tag.data()) == 1;
    EVP_CIPHER_CTX_free(context);
    require(ok, "AES-GCM metadata encryption failed");
    cipher.resize(static_cast<size_t>(written + final_written));
    return cipher;
}

std::vector<unsigned char> gcm_decrypt(const std::array<unsigned char, kSessionKeyLength>& key,
                                        const std::array<unsigned char, kGcmNonceLength>& nonce,
                                        const std::vector<unsigned char>& aad,
                                        const std::vector<unsigned char>& cipher,
                                        const std::array<unsigned char, kGcmTagLength>& tag) {
    require(cipher.size() <= static_cast<size_t>(INT_MAX), "metadata is too large for OpenSSL EVP update");
    EVP_CIPHER_CTX* context = EVP_CIPHER_CTX_new();
    require(context != nullptr, "cannot allocate AES-GCM context");
    std::vector<unsigned char> plain(cipher.size());
    int ignored = 0, written = 0, final_written = 0;
    const bool ok = EVP_DecryptInit_ex(context, EVP_aes_256_gcm(), nullptr, nullptr, nullptr) == 1 &&
                    EVP_CIPHER_CTX_ctrl(context, EVP_CTRL_GCM_SET_IVLEN, kGcmNonceLength, nullptr) == 1 &&
                    EVP_DecryptInit_ex(context, nullptr, nullptr, key.data(), nonce.data()) == 1 &&
                    EVP_DecryptUpdate(context, nullptr, &ignored, aad.data(), static_cast<int>(aad.size())) == 1 &&
                    EVP_DecryptUpdate(context, plain.data(), &written, cipher.data(), static_cast<int>(cipher.size())) == 1 &&
                    EVP_CIPHER_CTX_ctrl(context, EVP_CTRL_GCM_SET_TAG, kGcmTagLength, const_cast<unsigned char*>(tag.data())) == 1 &&
                    EVP_DecryptFinal_ex(context, plain.data() + written, &final_written) == 1;
    EVP_CIPHER_CTX_free(context);
    require(ok, "metadata authentication failed");
    plain.resize(static_cast<size_t>(written + final_written));
    return plain;
}

std::vector<unsigned char> wrap_key(const fs::path& public_pem, const std::array<unsigned char, kSessionKeyLength>& key) {
    FILE* file = fopen(public_pem.c_str(), "r");
    require(file != nullptr, "cannot open recipient public key");
    EVP_PKEY* pkey = PEM_read_PUBKEY(file, nullptr, nullptr, nullptr);
    fclose(file);
    require(pkey != nullptr, "cannot parse recipient public key PEM");
    EVP_PKEY_CTX* context = EVP_PKEY_CTX_new(pkey, nullptr);
    require(context != nullptr, "cannot allocate RSA-OAEP context");
    size_t output_length = 0;
    const bool ok = EVP_PKEY_encrypt_init(context) == 1 &&
                    EVP_PKEY_CTX_set_rsa_padding(context, RSA_PKCS1_OAEP_PADDING) == 1 &&
                    EVP_PKEY_CTX_set_rsa_oaep_md(context, EVP_sha256()) == 1 &&
                    EVP_PKEY_CTX_set_rsa_mgf1_md(context, EVP_sha256()) == 1 &&
                    EVP_PKEY_encrypt(context, nullptr, &output_length, key.data(), key.size()) == 1;
    std::vector<unsigned char> wrapped(output_length);
    const bool encrypted = ok && EVP_PKEY_encrypt(context, wrapped.data(), &output_length, key.data(), key.size()) == 1;
    EVP_PKEY_CTX_free(context);
    EVP_PKEY_free(pkey);
    require(encrypted, "RSA-OAEP key wrapping failed");
    wrapped.resize(output_length);
    return wrapped;
}

std::array<unsigned char, kSessionKeyLength> unwrap_key(const fs::path& private_pem,
                                                          const std::vector<unsigned char>& wrapped) {
    FILE* file = fopen(private_pem.c_str(), "r");
    require(file != nullptr, "cannot open recipient private key");
    EVP_PKEY* pkey = PEM_read_PrivateKey(file, nullptr, nullptr, nullptr);
    fclose(file);
    require(pkey != nullptr, "cannot parse recipient private key PEM");
    EVP_PKEY_CTX* context = EVP_PKEY_CTX_new(pkey, nullptr);
    require(context != nullptr, "cannot allocate RSA-OAEP context");
    size_t output_length = 0;
    const bool ok = EVP_PKEY_decrypt_init(context) == 1 &&
                    EVP_PKEY_CTX_set_rsa_padding(context, RSA_PKCS1_OAEP_PADDING) == 1 &&
                    EVP_PKEY_CTX_set_rsa_oaep_md(context, EVP_sha256()) == 1 &&
                    EVP_PKEY_CTX_set_rsa_mgf1_md(context, EVP_sha256()) == 1 &&
                    EVP_PKEY_decrypt(context, nullptr, &output_length, wrapped.data(), wrapped.size()) == 1;
    std::vector<unsigned char> unwrapped(output_length);
    const bool decrypted = ok && EVP_PKEY_decrypt(context, unwrapped.data(), &output_length, wrapped.data(), wrapped.size()) == 1;
    EVP_PKEY_CTX_free(context);
    EVP_PKEY_free(pkey);
    require(decrypted && output_length == kSessionKeyLength, "RSA-OAEP key unwrap failed");
    std::array<unsigned char, kSessionKeyLength> key{};
    std::copy_n(unwrapped.data(), key.size(), key.data());
    return key;
}

std::array<unsigned char, kMacLength> compute_mac(const std::array<unsigned char, kSessionKeyLength>& key,
                                                   const std::array<unsigned char, kCtrIvLength>& ctr_iv,
                                                   const std::string& masked,
                                                   const std::vector<unsigned char>& metadata,
                                                   const std::vector<unsigned char>& wrapped,
                                                   const std::vector<unsigned char>& aad) {
    HMAC_CTX* context = HMAC_CTX_new();
    require(context != nullptr, "cannot allocate HMAC context");
    unsigned int length = 0;
    std::array<unsigned char, kMacLength> mac{};
    const unsigned char version = kFormatVersion;
    const bool ok = HMAC_Init_ex(context, key.data(), key.size(), EVP_sha256(), nullptr) == 1 &&
                    HMAC_Update(context, reinterpret_cast<const unsigned char*>(kMacDomain), sizeof(kMacDomain) - 1) == 1 &&
                    HMAC_Update(context, &version, 1) == 1 &&
                    HMAC_Update(context, ctr_iv.data(), ctr_iv.size()) == 1 &&
                    HMAC_Update(context, reinterpret_cast<const unsigned char*>(masked.data()), masked.size()) == 1 &&
                    HMAC_Update(context, metadata.data(), metadata.size()) == 1 &&
                    HMAC_Update(context, wrapped.data(), wrapped.size()) == 1 &&
                    HMAC_Update(context, aad.data(), aad.size()) == 1 &&
                    HMAC_Final(context, mac.data(), &length) == 1;
    HMAC_CTX_free(context);
    require(ok && length == mac.size(), "HMAC-SHA-256 failed");
    return mac;
}

struct Metadata {
    std::array<unsigned char, kCtrIvLength> ctr_iv{};
    struct Interval {
        uint64_t start;
        uint64_t length;
    };
    uint64_t sequence_length = 0;
    std::vector<Interval> intervals;
};

std::vector<Metadata::Interval> bitmap_to_intervals(const std::vector<unsigned char>& bitmap) {
    std::vector<Metadata::Interval> intervals;
    for (size_t pos = 0; pos < bitmap.size();) {
        if (bitmap[pos] == 0) {
            ++pos;
            continue;
        }
        const size_t start = pos;
        while (pos < bitmap.size() && bitmap[pos] == 1) ++pos;
        intervals.push_back({static_cast<uint64_t>(start), static_cast<uint64_t>(pos - start)});
    }
    return intervals;
}

std::vector<unsigned char> intervals_to_bitmap(const Metadata& metadata, size_t expected_length) {
    require(metadata.sequence_length == expected_length, "metadata does not match masked sequence");
    std::vector<unsigned char> bitmap(expected_length, 0);
    for (const auto& interval : metadata.intervals) {
        require(interval.start <= expected_length && interval.length <= expected_length - interval.start,
                "invalid sensitive interval");
        std::fill(bitmap.begin() + static_cast<std::ptrdiff_t>(interval.start),
                  bitmap.begin() + static_cast<std::ptrdiff_t>(interval.start + interval.length), 1);
    }
    return bitmap;
}

struct DerivedKeys {
    std::array<unsigned char, kSessionKeyLength> mask{};
    std::array<unsigned char, kSessionKeyLength> meta{};
    std::array<unsigned char, kSessionKeyLength> mac{};
};

std::array<unsigned char, kSessionKeyLength> hkdf_derive(
        const std::array<unsigned char, kSessionKeyLength>& master, const char* info) {
    EVP_PKEY_CTX* context = EVP_PKEY_CTX_new_id(EVP_PKEY_HKDF, nullptr);
    require(context != nullptr, "cannot allocate HKDF context");
    std::array<unsigned char, kSessionKeyLength> output{};
    size_t output_length = output.size();
    const bool ok = EVP_PKEY_derive_init(context) == 1 &&
                    EVP_PKEY_CTX_set_hkdf_md(context, EVP_sha256()) == 1 &&
                    EVP_PKEY_CTX_set1_hkdf_salt(context, reinterpret_cast<const unsigned char*>("SeGen2-HKDF-v1"), 14) == 1 &&
                    EVP_PKEY_CTX_set1_hkdf_key(context, master.data(), master.size()) == 1 &&
                    EVP_PKEY_CTX_add1_hkdf_info(context, reinterpret_cast<const unsigned char*>(info), std::strlen(info)) == 1 &&
                    EVP_PKEY_derive(context, output.data(), &output_length) == 1;
    EVP_PKEY_CTX_free(context);
    require(ok && output_length == output.size(), "HKDF-SHA-256 key derivation failed");
    return output;
}

DerivedKeys derive_keys(const std::array<unsigned char, kSessionKeyLength>& master) {
    return {hkdf_derive(master, "SeGen/mask"), hkdf_derive(master, "SeGen/meta"),
            hkdf_derive(master, "SeGen/mac")};
}

std::vector<unsigned char> archive_aad(const std::vector<unsigned char>& manifest,
                                       const std::vector<unsigned char>& params) {
    std::vector<unsigned char> aad;
    const char domain[] = "SeGen2-archive-aad-v1";
    aad.insert(aad.end(), domain, domain + sizeof(domain) - 1);
    append_u32(aad, kFormatVersion);
    append_u64(aad, manifest.size());
    aad.insert(aad.end(), manifest.begin(), manifest.end());
    append_u64(aad, params.size());
    aad.insert(aad.end(), params.begin(), params.end());
    return aad;
}

std::vector<unsigned char> serialize_plain_metadata(const Metadata& metadata) {
    std::vector<unsigned char> plain;
    plain.reserve(8 + 4 + metadata.ctr_iv.size() + 8 + metadata.intervals.size() * 16);
    append_u64(plain, metadata.sequence_length);
    append_u32(plain, kWindowLength);
    plain.insert(plain.end(), metadata.ctr_iv.begin(), metadata.ctr_iv.end());
    append_u64(plain, metadata.intervals.size());
    for (const auto& interval : metadata.intervals) {
        append_u64(plain, interval.start);
        append_u64(plain, interval.length);
    }
    return plain;
}

Metadata parse_plain_metadata(const std::vector<unsigned char>& plain, size_t expected_length) {
    size_t plain_offset = 0;
    const uint64_t sequence_length = read_u64(plain, plain_offset);
    const uint32_t window_length = read_u32(plain, plain_offset);
    require(sequence_length == expected_length && window_length == kWindowLength,
            "metadata does not match masked sequence");
    require(plain_offset + kCtrIvLength + 8 <= plain.size(), "invalid metadata length");
    Metadata metadata;
    metadata.sequence_length = sequence_length;
    std::copy_n(plain.data() + plain_offset, kCtrIvLength, metadata.ctr_iv.data());
    plain_offset += kCtrIvLength;
    const uint64_t interval_count = read_u64(plain, plain_offset);
    require(interval_count <= sequence_length && interval_count <= (plain.size() - plain_offset) / 16 &&
            plain_offset + interval_count * 16 == plain.size(), "invalid interval metadata");
    uint64_t previous_end = 0;
    for (uint64_t i = 0; i < interval_count; ++i) {
        const uint64_t start = read_u64(plain, plain_offset);
        const uint64_t length = read_u64(plain, plain_offset);
        require(length > 0 && start >= previous_end && start <= sequence_length && length <= sequence_length - start,
                "invalid sensitive interval");
        metadata.intervals.push_back({start, length});
        previous_end = start + length;
    }
    return metadata;
}

std::vector<unsigned char> pack_metadata(const std::array<unsigned char, kSessionKeyLength>& key, const Metadata& metadata,
                                         const std::vector<unsigned char>& aad) {
    std::array<unsigned char, kGcmNonceLength> nonce{};
    require(RAND_bytes(nonce.data(), nonce.size()) == 1, "random GCM nonce generation failed");
    std::array<unsigned char, kGcmTagLength> tag{};
    const auto cipher = gcm_encrypt(key, nonce, aad, serialize_plain_metadata(metadata), tag);
    std::vector<unsigned char> result(kMetadataMagic.begin(), kMetadataMagic.end());
    result.push_back(kFormatVersion);
    result.insert(result.end(), nonce.begin(), nonce.end());
    result.insert(result.end(), cipher.begin(), cipher.end());
    result.insert(result.end(), tag.begin(), tag.end());
    return result;
}

Metadata unpack_metadata(const std::array<unsigned char, kSessionKeyLength>& key,
                         const std::vector<unsigned char>& packed, const std::vector<unsigned char>& aad) {
    require(packed.size() >= kMetadataMagic.size() + 1 + kGcmNonceLength + kGcmTagLength, "metadata is truncated");
    require(std::equal(kMetadataMagic.begin(), kMetadataMagic.end(), packed.begin()) &&
            packed[kMetadataMagic.size()] == kFormatVersion, "unsupported metadata format");
    size_t offset = kMetadataMagic.size() + 1;
    std::array<unsigned char, kGcmNonceLength> nonce{};
    std::copy_n(packed.data() + offset, nonce.size(), nonce.data());
    offset += nonce.size();
    std::array<unsigned char, kGcmTagLength> tag{};
    std::copy_n(packed.end() - static_cast<std::ptrdiff_t>(tag.size()), tag.size(), tag.data());
    const std::vector<unsigned char> cipher(packed.begin() + static_cast<std::ptrdiff_t>(offset),
                                             packed.end() - static_cast<std::ptrdiff_t>(tag.size()));
    const auto plain = gcm_decrypt(key, nonce, aad, cipher, tag);
    size_t plain_offset = 0;
    const uint64_t sequence_length = read_u64(plain, plain_offset);
    return parse_plain_metadata(plain, sequence_length);
}

void mask_sensitive_regions(int threads, const fs::path& input_path, const fs::path& public_pem,
                            const fs::path& output_dir, bool full_mask = false) {
    require(threads > 0, "thread count must be positive");
    const std::string sequence = read_dna_sequence(input_path);
    size_t matches = 0;
    Metadata metadata;
    std::vector<unsigned char> bitmap;
    if (full_mask) {
        bitmap.assign(sequence.size(), 1);
    } else {
        const bloom_filter filter = load_dictionary("dataBaseSrf.txt");
        bitmap = sensitive_bitmap(sequence, filter, threads, matches);
    }
    metadata.sequence_length = sequence.size();
    metadata.intervals = bitmap_to_intervals(bitmap);
    std::array<unsigned char, kSessionKeyLength> key{};
    require(RAND_bytes(key.data(), key.size()) == 1, "random session key generation failed");
    const DerivedKeys keys = derive_keys(key);
    require(RAND_bytes(metadata.ctr_iv.data(), metadata.ctr_iv.size()) == 1, "random AES-CTR IV generation failed");
    const auto stream = ctr_keystream(keys.mask, metadata.ctr_iv, sequence.size());
    const std::string masked = apply_mask(sequence, bitmap, stream, false);
    const std::vector<unsigned char> empty_aad;
    const auto metadata_file = pack_metadata(keys.meta, metadata, empty_aad);
    const auto wrapped_key = wrap_key(public_pem, key);
    const auto mac = compute_mac(keys.mac, metadata.ctr_iv, masked, metadata_file, wrapped_key, empty_aad);

    fs::create_directories(output_dir);
    write_binary(output_dir / "encry.data", {masked.begin(), masked.end()});
    write_binary(output_dir / "metadata.bin", metadata_file);
    write_binary(output_dir / "wrapped_key.bin", wrapped_key);
    write_binary(output_dir / "auth.tag", {mac.begin(), mac.end()});
    OPENSSL_cleanse(key.data(), key.size());
    if (full_mask) {
        std::cerr << "FullMask: marked all " << sequence.size() << " bases\n";
    } else {
        std::cerr << "PSSM: marked " << matches << " sensitive 50-mer windows\n";
    }
}

void unmask_sensitive_regions(int threads, const fs::path& masked_path, const fs::path& metadata_path,
                              const fs::path& wrapped_key_path, const fs::path& mac_path,
                              const fs::path& private_pem, const fs::path& output_path) {
    (void)threads;
    const std::string masked = read_dna_sequence(masked_path);
    const auto metadata_file = read_binary(metadata_path);
    const auto wrapped_key = read_binary(wrapped_key_path);
    const auto provided_mac = read_binary(mac_path);
    require(provided_mac.size() == kMacLength, "invalid HMAC tag length");
    auto key = unwrap_key(private_pem, wrapped_key);
    const DerivedKeys keys = derive_keys(key);
    // The CTR IV is authenticated inside GCM metadata; no plaintext is emitted until both checks pass.
    const std::vector<unsigned char> empty_aad;
    const Metadata metadata = unpack_metadata(keys.meta, metadata_file, empty_aad);
    const auto bitmap = intervals_to_bitmap(metadata, masked.size());
    const auto expected_mac = compute_mac(keys.mac, metadata.ctr_iv, masked, metadata_file, wrapped_key, empty_aad);
    require(CRYPTO_memcmp(provided_mac.data(), expected_mac.data(), expected_mac.size()) == 0,
            "masked sequence or archive authentication failed");
    const auto stream = ctr_keystream(keys.mask, metadata.ctr_iv, masked.size());
    const std::string recovered = apply_mask(masked, bitmap, stream, true);
    write_binary(output_path, {recovered.begin(), recovered.end()});
    OPENSSL_cleanse(key.data(), key.size());
}

void usage() {
    std::cerr << "Usage:\n"
              << "  prsec -c <threads> <input_dna> <recipient_public.pem> <output_dir>\n"
              << "  prsec -f <threads> <input_dna> <recipient_public.pem> <output_dir>\n"
              << "  prsec -d <threads> <masked_dna> <metadata.bin> <wrapped_key.bin> <auth.tag> <recipient_private.pem> <output_dna>\n";
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 6 && std::string(argv[1]) == "-c") {
            mask_sensitive_regions(std::stoi(argv[2]), argv[3], argv[4], argv[5]);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "-f") {
            mask_sensitive_regions(std::stoi(argv[2]), argv[3], argv[4], argv[5], true);
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "-d") {
            unmask_sensitive_regions(std::stoi(argv[2]), argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
            return 0;
        }
        usage();
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "prsec: " << error.what() << '\n';
        return 1;
    }
}
