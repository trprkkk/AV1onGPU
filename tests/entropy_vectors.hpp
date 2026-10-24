/*
 * AV1onGPU - golden entropy decoder vectors (generated, do not edit)
 *
 * Produced by decoding the payloads below with dav1d's `msac` (`dav1d_msac_decode_symbol_adapt_c`,
 * `dav1d_msac_decode_bool_c`, `dav1d_msac_decode_bool_equi_c`). dav1d and libaom's
 * `od_ec` are the two implementations AV1 streams are validated against, so
 * matching these arrays byte for byte is what "bit identical to the
 * reference decoders" means in practice.
 *
 * Each case stores the initial Q15 inverse CDF (N entries plus the adaptation
 * counter), the payload and the symbols dav1d returned. The counter entry of
 * the *final* table is kept as well, so adaptation is pinned too: a decoder
 * that decodes the right symbols but adapts differently fails the next frame.
 */
#ifndef AV1GPU_TESTS_ENTROPY_VECTORS_HPP
#define AV1GPU_TESTS_ENTROPY_VECTORS_HPP

#include <cstdint>

namespace entropy_vectors {

inline const uint16_t case0_cdf[3] = {24119, 0, 0};
inline const uint8_t case0_payload[24] = {0x27, 0x0d, 0x9a, 0xc9, 0x53, 0x6d, 0x59, 0x45, 0x78, 0x13, 0x27, 0x83, 0xa0, 0x64, 0x33, 0x28, 0x63, 0x99, 0x8c, 0x2a, 0xe0, 0xf9, 0xf9, 0xa6};
inline const unsigned case0_symbols[64] = {0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0};
inline const uint16_t case0_final_cdf[3] = {17249, 0, 32};
constexpr int case0_n = 2;
constexpr int case0_symbols_count = 64;

inline const uint16_t case1_cdf[4] = {26360, 20371, 0, 0};
inline const uint8_t case1_payload[24] = {0x4d, 0xd1, 0x44, 0x2d, 0xcf, 0x3a, 0xb8, 0xd8, 0x54, 0x21, 0x19, 0xe6, 0x9d, 0x13, 0x6f, 0x89, 0x38, 0xbb, 0xa5, 0xd3, 0x4c, 0xa5, 0x9b, 0x9f};
inline const unsigned case1_symbols[48] = {1, 2, 1, 2, 2, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 1, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 1, 0, 2, 0, 0, 2, 0, 1, 1, 0, 0, 1, 2, 2, 1, 2, 0, 2, 2, 2, 1, 1};
inline const uint16_t case1_final_cdf[4] = {23983, 15401, 0, 32};
constexpr int case1_n = 3;
constexpr int case1_symbols_count = 48;

inline const uint16_t case2_cdf[5] = {27481, 20252, 15392, 0, 0};
inline const uint8_t case2_payload[32] = {0x77, 0xed, 0x1e, 0x0a, 0x26, 0x71, 0x59, 0x8c, 0x5c, 0x6c, 0x8e, 0x37, 0x65, 0x81, 0xa9, 0x0b, 0x25, 0xdb, 0x0a, 0x5d, 0xc6, 0x86, 0x97, 0xa3, 0x35, 0x31, 0x45, 0x0e, 0x65, 0x02, 0x83, 0x7d};
inline const unsigned case2_symbols[64] = {2, 3, 0, 2, 2, 1, 3, 0, 0, 1, 0, 3, 3, 0, 0, 3, 0, 1, 3, 2, 1, 2, 2, 0, 3, 1, 1, 2, 2, 2, 0, 3, 3, 2, 2, 3, 3, 1, 0, 1, 2, 0, 3, 3, 3, 0, 1, 1, 3, 3, 0, 3, 0, 2, 1, 2, 2, 3, 3, 3, 0, 0, 1, 0};
inline const uint16_t case2_final_cdf[5] = {24808, 18477, 11977, 0, 32};
constexpr int case2_n = 4;
constexpr int case2_symbols_count = 64;

inline const uint16_t case3_cdf[9] = {29784, 26721, 24463, 21462, 17884, 14355, 10950, 0, 0};
inline const uint8_t case3_payload[48] = {0x5e, 0x20, 0xf9, 0x4c, 0x17, 0x5d, 0x52, 0x54, 0x38, 0xd0, 0xa8, 0x61, 0x38, 0x57, 0x58, 0xab, 0x31, 0xa0, 0x0a, 0xd6, 0x02, 0x39, 0xff, 0xa5, 0x2a, 0x9d, 0x40, 0xbf, 0x03, 0xc8, 0xb6, 0xdf, 0x68, 0x5e, 0x86, 0x45, 0xe0, 0x1d, 0x17, 0xbf, 0x33, 0x7c, 0xe5, 0xd4, 0x63, 0x1b, 0x1e, 0xd3};
inline const unsigned case3_symbols[80] = {4, 2, 4, 3, 4, 5, 2, 4, 4, 0, 0, 7, 2, 3, 7, 2, 2, 2, 1, 4, 6, 2, 0, 7, 6, 7, 0, 4, 0, 7, 4, 1, 6, 2, 6, 7, 0, 7, 3, 6, 5, 6, 1, 5, 3, 1, 5, 1, 4, 6, 7, 1, 7, 1, 7, 7, 5, 7, 4, 1, 0, 6, 7, 3, 4, 1, 7, 5, 6, 4, 4, 7, 7, 7, 2, 2, 1, 7, 4, 3};
inline const uint16_t case3_final_cdf[9] = {29499, 26188, 22132, 19428, 13999, 11446, 8356, 0, 32};
constexpr int case3_n = 8;
constexpr int case3_symbols_count = 80;

inline const uint16_t case4_cdf[12] = {30124, 28017, 26310, 24492, 22025, 19294, 16581, 13712, 10902, 8956, 0, 0};
inline const uint8_t case4_payload[48] = {0x3b, 0x78, 0x1e, 0x92, 0x24, 0xeb, 0x1a, 0xd0, 0xb1, 0x42, 0x05, 0x67, 0xc0, 0x58, 0x78, 0xf7, 0x0d, 0x2c, 0x56, 0x12, 0x1a, 0x56, 0xb5, 0x62, 0x13, 0x60, 0x33, 0x29, 0xc3, 0x7e, 0x43, 0x9f, 0xc8, 0xea, 0x60, 0xe2, 0xcf, 0x6b, 0x6a, 0xcc, 0xf5, 0x76, 0x21, 0xab, 0x5a, 0x8b, 0x82, 0xe0};
inline const unsigned case4_symbols[64] = {3, 8, 6, 10, 10, 6, 2, 3, 8, 5, 6, 1, 3, 2, 3, 10, 7, 8, 3, 7, 4, 10, 6, 4, 3, 2, 0, 8, 6, 7, 4, 5, 2, 10, 6, 5, 5, 1, 3, 4, 1, 10, 1, 7, 1, 9, 1, 4, 5, 7, 1, 0, 7, 6, 10, 7, 10, 10, 10, 3, 7, 5, 0, 1};
inline const uint16_t case4_final_cdf[12] = {30931, 28053, 25825, 22001, 19590, 16800, 13260, 10018, 7413, 6483, 0, 32};
constexpr int case4_n = 11;
constexpr int case4_symbols_count = 64;

inline const uint16_t case5_cdf[17] = {31049, 29693, 28450, 26787, 25476, 23717, 22679, 20867, 19265, 17405, 15997, 14334, 12678, 10865, 9369, 0, 0};
inline const uint8_t case5_payload[64] = {0x35, 0x30, 0x0e, 0x6d, 0xb7, 0xa6, 0xe0, 0xb6, 0xfb, 0x76, 0x3f, 0x1b, 0x05, 0x85, 0xfc, 0x66, 0x83, 0xf6, 0x7d, 0xc2, 0x5a, 0x54, 0x43, 0x04, 0x83, 0x38, 0xd9, 0x53, 0xb7, 0xee, 0x63, 0x1e, 0x7e, 0xae, 0x10, 0x5f, 0x1a, 0xfc, 0xb8, 0x21, 0x5d, 0x5a, 0xb2, 0x0c, 0xe5, 0x5d, 0x7a, 0xb9, 0x82, 0x5a, 0x5d, 0xd1, 0x40, 0x0d, 0x94, 0x8c, 0x8b, 0x36, 0x88, 0xf6, 0xa3, 0xa6, 0x0d, 0x8c};
inline const unsigned case5_symbols[80] = {4, 13, 12, 1, 12, 9, 14, 11, 15, 14, 1, 15, 14, 14, 12, 15, 9, 5, 9, 12, 7, 15, 14, 0, 6, 1, 6, 11, 1, 15, 14, 11, 6, 4, 5, 2, 15, 11, 5, 8, 2, 6, 14, 4, 14, 1, 1, 14, 13, 14, 10, 4, 14, 12, 15, 10, 6, 12, 1, 4, 3, 4, 14, 8, 15, 8, 15, 14, 4, 1, 12, 11, 4, 9, 14, 12, 4, 1, 9, 4};
inline const uint16_t case5_final_cdf[17] = {31880, 28857, 28088, 27338, 24581, 23373, 21825, 20948, 19791, 17740, 16873, 14813, 11809, 10679, 5985, 0, 32};
constexpr int case5_n = 16;
constexpr int case5_symbols_count = 80;

inline const uint8_t bool_payload[40] = {0x21, 0x90, 0x24, 0x7b, 0x19, 0xb7, 0xdc, 0x75, 0xae, 0xd5, 0xff, 0x1b, 0xe5, 0x6d, 0x97, 0xcd, 0x97, 0x48, 0x8c, 0xb6, 0xfe, 0xf6, 0xf3, 0x8d, 0x2c, 0x09, 0x8d, 0x6f, 0x05, 0x4f, 0xf4, 0x6a, 0x08, 0x81, 0x8e, 0xd5, 0x1c, 0xa5, 0xb6, 0x48};
inline const unsigned bool_probabilities[64] = {7040, 32375, 177, 13769, 19629, 21841, 6611, 13228, 20366, 1402, 4954, 9762, 30523, 28801, 29558, 10045, 18296, 5220, 10165, 22266, 12675, 20396, 8660, 8289, 18536, 29895, 14408, 7934, 13976, 10677, 8531, 11921, 19480, 31052, 10402, 20588, 14409, 4688, 28442, 24768, 32170, 23677, 31457, 28159, 10563, 25366, 20536, 28418, 32550, 8175, 414, 29522, 13286, 6917, 23546, 13065, 31363, 9491, 4926, 13712, 28605, 2234, 3373, 10293};
inline const unsigned bool_bits[64] = {0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0};
inline const uint8_t equi_payload[24] = {0x35, 0x3a, 0x74, 0x0b, 0x14, 0xc8, 0xb7, 0xe3, 0x6d, 0x60, 0x5d, 0xef, 0x90, 0x95, 0xbe, 0x77, 0x81, 0x71, 0x0e, 0xc7, 0xc3, 0xd4, 0x60, 0x69};
inline const unsigned equi_bits[64] = {0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1};
inline const uint8_t literal_payload[24] = {0x91, 0x19, 0x2e, 0x06, 0x0c, 0x24, 0x59, 0xce, 0x25, 0x4d, 0xc0, 0xba, 0xb6, 0xc1, 0x98, 0xe8, 0xc5, 0xf4, 0x7a, 0x0a, 0xad, 0xb1, 0x91, 0x8c};
inline const int literal_widths[8] = {1, 2, 3, 5, 8, 4, 6, 2};
inline const unsigned literal_values[8] = {1, 0, 4, 9, 0, 14, 0, 2};
inline const uint8_t uniform_payload[24] = {0xdd, 0xe3, 0xf4, 0xbb, 0xe7, 0xa7, 0x57, 0x05, 0x1b, 0x26, 0x09, 0x8b, 0xd3, 0x8d, 0x05, 0xa6, 0x5d, 0x4d, 0xe6, 0xf5, 0x48, 0x84, 0xef, 0xc3};
inline const unsigned uniform_sizes[12] = {1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31};
inline const unsigned uniform_values[12] = {0, 1, 1, 3, 2, 6, 4, 6, 0, 11, 2, 25};

} /* namespace entropy_vectors */

#endif /* AV1GPU_TESTS_ENTROPY_VECTORS_HPP */
