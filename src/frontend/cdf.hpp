/*
 * AV1onGPU - probability table context
 *
 * A tile carries no probability tables: decoder and encoder start from the
 * defaults in cdf_tables.hpp and adapt them in lockstep as symbols are
 * decoded. This file holds the context a decoder owns - one per frame, or one
 * per tile when the frame is coded error resiliently - and nothing else; the
 * tile syntax in tile_decoder.cpp is the only writer.
 *
 * Two conventions matter and they are easy to get backwards:
 *
 *   Storage. A row of `n` cumulative frequencies becomes `n` *inverse*
 *   cumulative values followed by the adaptation counter, which is what
 *   RangeDecoder::read_symbol consumes: entry i is the probability that the
 *   symbol is below i, scaled to 32768, so it decreases towards the last
 *   symbol, and the last symbol (which needs no stored probability, see
 *   range_decoder.hpp) is followed by the counter. The defaults are stored that
 *   way already, so a table can be used by the range decoder with no
 *   conversion at all.
 *
 *   Ownership. Decoding mutates the tables, and the mutation is part of the
 *   bitstream: two decoders that adapt differently diverge on the next symbol
 *   of the same table. So each frame gets its own copy, and a frame that
 *   refreshes its context from a reference frame copies the whole context.
 *
 * Sizes and names follow the specification's table set. The comments name the
 * context each dimension indexes. Rows are padded, as in the reference
 * decoders: `read_symbol` only looks at the first nsymbols + 1 entries, so the
 * padding is never read.
 */
#ifndef AV1GPU_FRONTEND_CDF_HPP
#define AV1GPU_FRONTEND_CDF_HPP

#include <cstdint>

#include "frontend/cdf_tables.hpp"

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
constexpr int kPartitionCdfSize = 16;

/* The 13 directional/smooth intra modes, then CFL for chroma and the four
 * derived modes the specification lists after it. */
constexpr int kIntraModes = 13;
constexpr int kUvIntraModes = 14;

struct PartitionCdf {
  uint16_t table[kBlockLevelCount][kPartitionContexts][kPartitionCdfSize];

  static int symbols(int level) { return kPartitionSymbols[level] + 1; }
  uint16_t* row(int level, int context) { return table[level][context]; }
  const uint16_t* row(int level, int context) const { return table[level][context]; }
};

/* skip is a two symbol alphabet with three contexts (the number of coded
 * neighbour blocks). */
constexpr int kSkipContexts = 3;
struct SkipCdf {
  uint16_t table[kSkipContexts][2];
  uint16_t* row(int context) { return table[context]; }
  const uint16_t* row(int context) const { return table[context]; }
};

/* Coefficient probabilities: one row per transform size class (the reference
 * decoders use t_dim->ctx, five classes: 4x4, 8x8, 16x16, 32x32, 64x64),
 * plane (luma/chroma) and whatever context the symbol needs. */
struct CoefCdf {
  /* end of block position, indexed by [chroma][tx class of the transform] for
   * the 2D classes and [chroma] only for the 1D ones */
  uint16_t eob_bin_16[2][2][8];
  uint16_t eob_bin_32[2][2][8];
  uint16_t eob_bin_64[2][2][8];
  uint16_t eob_bin_128[2][2][8];
  uint16_t eob_bin_256[2][2][16];
  uint16_t eob_bin_512[2][16];
  uint16_t eob_bin_1024[2][16];
  /* the token of the last coefficient, then of the ones before it */
  uint16_t eob_base_tok[5][2][4][4];
  uint16_t base_tok[5][2][41][4];
  uint16_t br_tok[4][2][21][4];
  /* the coefficient's bit above the base token's range */
  uint16_t eob_hi_bit[5][2][9][2];
  /* "does this transform block have any coefficient at all" */
  uint16_t skip[5][13][2];
  uint16_t dc_sign[2][3][2];
};

/* The prediction syntax tables a key frame needs. Intra frames and inter
 * frames share most of these; the ones only inter frames use (reference,
 * compound, motion mode ...) are added with the inter syntax. */
struct ModeCdf {
  /* key frame y mode: [above][left] contexts of the neighbouring modes */
  uint16_t kfym[5][5][16];
  uint16_t partition[5][4][16];
  uint16_t skip[3][2];
  uint16_t intrabc[2];
  /* segment id / temporal prediction, only present when segmentation is on */
  uint16_t seg_id[3][8];
  uint16_t seg_pred[3][2];
  /* per superblock quantiser and loop filter deltas */
  uint16_t delta_q[4];
  uint16_t delta_lf[5][4];
  /* chroma mode: [cfl allowed][luma mode] */
  uint16_t uv_mode[2][13][16];
  /* angle delta for the seven directional modes */
  uint16_t angle_delta[8][8];
  uint16_t filter_intra[8];
  uint16_t use_filter_intra[22][2];
  /* palette */
  uint16_t pal_sz[2][7][8];
  uint16_t pal_y[7][3][2];
  uint16_t pal_uv[2][2];
  /* chroma from luma */
  uint16_t cfl_sign[8];
  uint16_t cfl_alpha[6][16];
  /* transform type, intra: [transform size class][luma mode] */
  uint16_t txtp_intra1[2][13][8];
  uint16_t txtp_intra2[3][13][8];
  /* transform size, transform partition */
  uint16_t txsz[4][3][4];
  uint16_t txpart[7][3][2];
};

/* The probability tables one frame starts from. A frame that inherits its
 * context (`primary_ref_frame`) copies the corresponding context of that
 * reference frame instead of calling init_default_cdf. */
struct CdfContext {
  PartitionCdf partition;
  SkipCdf skip;
  ModeCdf mode;
  CoefCdf coef;
};

/* Copies the specification's defaults into `cdf`. */
void init_default_cdf(CdfContext* cdf);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_CDF_HPP */
