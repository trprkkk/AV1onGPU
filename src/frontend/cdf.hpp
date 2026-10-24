/*
 * AV1onGPU - default probability tables
 *
 * A tile carries no probability tables: decoder and encoder start from the
 * defaults in the specification (section 9.3) and adapt them in lockstep as
 * symbols are decoded. This file holds those defaults, in the layout the tile
 * parser needs, and nothing else - there is no syntax here yet, but every
 * symbol the parser reads will come out of these tables, so they are worth
 * getting exactly right on their own.
 *
 * Two conventions matter and they are easy to get backwards:
 *
 *   Storage. A row of `n` cumulative frequencies becomes `n` *inverse*
 *   cumulative values plus the adaptation counter, which is what
 *   RangeDecoder::read_symbol consumes: entry i is the probability that the
 *   symbol is below i, scaled to 32768, so it decreases towards the last
 *   symbol, and the last symbol (which needs no stored probability, see
 *   range_decoder.hpp) is followed by the counter. The specification gives the
 *   tables in the cumulative form, so `init_*_cdf` does the 32768 - x
 *   conversion; keeping the source numbers in that form is what makes them
 *   comparable with the specification and with the reference decoders.
 *
 *   Ownership. The defaults are constants, but decoding mutates them, so the
 *   parser owns a CdfContext per frame (or per tile, when the frame is coded
 *   error resiliently) and copies the defaults into it exactly once.
 *
 * The tables are the normative defaults, cross-checked value for value against
 * dav1d's src/cdf.c (whose CDFn() macros are the same 32768 - x conversion)
 * and against the CDFs dav1d used while decoding a real SVT-AV1 key frame:
 * tests/test_entropy.cpp decodes the first symbol of that tile out of these
 * tables and requires dav1d's exact values.
 */
#ifndef AV1GPU_FRONTEND_CDF_HPP
#define AV1GPU_FRONTEND_CDF_HPP

#include <cstdint>

namespace av1gpu {
namespace frontend {

/* The block level of the block being partitioned - that is, of the *parent*
 * block, which is what selects the table. Levels AV1 uses for the partition
 * symbol; 4x4 blocks are never partitioned. */
enum BlockLevel {
  kBlockLevel128x128 = 0,
  kBlockLevel64x64 = 1,
  kBlockLevel32x32 = 2,
  kBlockLevel16x16 = 3,
  kBlockLevel8x8 = 4,
  kBlockLevelCount = 5,
};

/* Partition symbols per level are counted down from N_PARTITIONS: 128x128
 * blocks cannot be split into four 4:1 slivers, and 8x8 blocks only have the
 * four legacy partition types. These counts are the `nsymbols` argument: the
 * alphabet holds one more symbol than this. */
constexpr int kPartitionSymbols[kBlockLevelCount] = {7, 9, 9, 9, 3};
/* above / left partition context */
constexpr int kPartitionContexts = 4;
/* widest row, in entries: probabilities plus the counter */
constexpr int kPartitionCdfSize = 10;

struct PartitionCdf {
  uint16_t table[kBlockLevelCount][kPartitionContexts][kPartitionCdfSize];

  /* Symbol count of a level's alphabet. */
  static int symbols(int level) { return kPartitionSymbols[level] + 1; }
  uint16_t* row(int level, int context) {
    return table[level][context];
  }
};

/* skip is a two symbol alphabet with three contexts (the number of coded
 * neighbour blocks); it is read for every block, so it shares the tile's CDF
 * state. */
constexpr int kSkipContexts = 3;
struct SkipCdf {
  uint16_t table[kSkipContexts][2];
  uint16_t* row(int context) { return table[context]; }
};

/* The probability tables one frame starts from. The tile parser will grow this
 * as the syntax layers land; the defaults are shared by every frame that does
 * not inherit them from a reference frame. */
struct CdfContext {
  PartitionCdf partition;
  SkipCdf skip;
};

/* Copies the specification's defaults into `cdf`. */
void init_default_cdf(CdfContext* cdf);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_CDF_HPP */
