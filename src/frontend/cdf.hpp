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

#include "frontend/block_size.hpp"
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

/* The partition table's first index is the *level*, and level 0 is the largest
 * block: row 0 is the 128x128 table of the annex, row 4 the 8x8 one, which is
 * the opposite order to the annex's own listing (it prints the 8x8 table
 * first). Rows are read with `kPartitionSymbols[level]` symbols, fewer at the
 * larger levels because 128x128 blocks cannot use the 4:1 splits and 8x8
 * blocks only have the four legacy types. */
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
  /* The token of the last coefficient, then of the ones before it.
   *
   * The specification sizes the base token table at 42 contexts but the syntax
   * never reaches the last one (the largest context any coefficient can select
   * is 40), and the reference decoders therefore carry 41 rows; the 42nd row of
   * the annex is a value no stream can read. The same holds for the extended
   * levels: the annex has a slice for each of the five transform classes, but
   * the largest class codes its levels with the Golomb escape alone, so only
   * four slices are ever read. */
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
  /* the colour map of a palette block: [plane][palette size - 2][colour
   * context], one probability per colour past the first, so a row for a
   * palette of N colours holds N - 1 of them */
  uint16_t pal_color[2][7][5][8];
  /* chroma from luma */
  uint16_t cfl_sign[8];
  uint16_t cfl_alpha[6][16];
  /* transform type, intra: [transform size class][luma mode] */
  uint16_t txtp_intra1[2][13][8];
  uint16_t txtp_intra2[3][13][8];
  /* transform size, transform partition */
  uint16_t txsz[4][3][4];
  uint16_t txpart[7][3][2];
  /* loop restoration: the switchable frame's filter symbol and the two
   * single-filter frames' on/off symbols (5.11.46). They are read once per
   * restoration unit and adapt like any other table. */
  uint16_t restore_switchable[3];
  uint16_t restore_wiener[2];
  uint16_t restore_sgrproj[2];
};

/* The tables only an inter frame reads: reference selection (single, compound,
 * unidirectional), skip mode, motion vectors with their classes and fractional
 * parts, the dynamic reference list, motion mode, the interp filter, compound
 * blending and inter/intra. They come from the specification's annex through
 * tools/gen_cdf_tables.py, in the same inverse layout and the same padding as
 * the intra tables.
 *
 * The comments name the indices each table is selected by, following the CDF
 * selection rules in section 8.3 of the specification; where a dimension is not
 * a context but the decoded value itself (the compound reference tree, the
 * motion vector class) the note says so. */
/* The specification's MV_CONTEXTS: an inter block and an intra block copy
 * block code their motion vector against separate copies of every table. */
constexpr int kMvContexts = 2;

struct InterCdf {
  /* [size group][13]: the intra mode of an intra block inside an inter frame
   * (a key frame codes the same symbol with its own neighbour contexts) */
  uint16_t y_mode[4][16];
  /* [inter neighbours][2]: is the block inter predicted at all */
  uint16_t is_inter[4][2];
  /* [neighbouring skip modes][2] */
  uint16_t skip_mode[3][2];
  /* [neighbouring references][2]: single or compound prediction */
  uint16_t comp_mode[5][2];
  /* [neighbouring references][2]: compound or unidirectional compound */
  uint16_t comp_ref_type[5][2];
  /* [reference counts][step of the reference tree][2] */
  uint16_t uni_comp_ref[3][3][2];
  uint16_t comp_ref[3][3][2];
  uint16_t comp_bwd_ref[3][2][2];
  uint16_t single_ref[3][6][2];
  /* [compound mode context][compound mode] */
  uint16_t compound_mode[8][8];
  uint16_t compound_idx[6][2];
  uint16_t comp_group_idx[6][2];
  /* Blend of two predictors: [block size][distance weighted or wedge].
   *
   * Two symbols, not four: whether a compound block averages, blends by
   * distance, wedges or uses a difference mask is coded as three decisions
   * (average against the rest, then wedge against difference), and this is the
   * last of them. */
  uint16_t compound_type[kBlockSizeCount][2];
  uint16_t wedge_index[kBlockSizeCount][16];
  /* inter/intra: [size group][2], then the intra mode of the pair */
  uint16_t inter_intra[3][2];
  uint16_t inter_intra_mode[3][4];
  uint16_t wedge_inter_intra[kBlockSizeCount][2];
  /* motion mode: [block size][simple, obmc, warped] */
  uint16_t use_obmc[kBlockSizeCount][2];
  uint16_t motion_mode[kBlockSizeCount][3];
  /* [filter context][eight tap, smooth, sharp] */
  uint16_t interp_filter[16][3];
  /* motion vectors: which components are coded, then per component the class,
   * the fractional part and - for class 0 - the integer bit */
  /* The two components (vertical first) keep separate copies of every motion
   * vector table, because adapting one must not change the other: the
   * specification initialises both from the same default rows, so the tables
   * start equal but drift apart as soon as a vector is coded. The defaults are
   * one row per symbol, so the copies below are broadcast at init time.
   *
   * The leading index is the motion vector *context* (MvCtx): 0 for an inter
   * block, 1 for an intra block copy block, whose vector is coded against its
   * own copy of the tables (7.11.3.7's MvCtx). Both start from the same
   * defaults, which is why a frame that only ever codes one of the two
   * contexts decodes identically either way - and why forgetting the second
   * context desynchronises the very first intra block copy block. */
  uint16_t mv_joint[kMvContexts][4];
  uint16_t mv_sign[kMvContexts][2][2];
  uint16_t mv_class[kMvContexts][2][16];
  uint16_t mv_class0_bit[kMvContexts][2][2];
  uint16_t mv_class0_fr[kMvContexts][2][2][4];
  uint16_t mv_class0_hp[kMvContexts][2][2];
  uint16_t mv_hp[kMvContexts][2][2];
  uint16_t mv_bit[kMvContexts][10][2][2];
  uint16_t mv_fr[kMvContexts][2][4];
  /* the dynamic reference list: new or predicted motion vector */
  uint16_t new_mv[6][2];
  uint16_t zero_mv[2][2];
  uint16_t ref_mv[6][2];
  uint16_t drl_mode[3][2];
  /* transform type, inter: the two classes and the set for 64x64 transforms */
  uint16_t txtp_inter1[2][16];
  uint16_t txtp_inter2[16];
  uint16_t txtp_inter3[4][2];
};

/* The probability tables one frame starts from. A frame that inherits its
 * context (`primary_ref_frame`) copies the corresponding context of that
 * reference frame instead of calling init_default_cdf. */
struct CdfContext {
  PartitionCdf partition;
  SkipCdf skip;
  ModeCdf mode;
  InterCdf inter;
  CoefCdf coef;
};

/* Copies the specification's defaults into `cdf`.
 *
 * `base_q_idx` selects the coefficient table set: AV1 ships four, split at
 * quantisers 20, 60 and 120, because the shape of a coefficient block changes
 * a lot with the quantiser (at 20 almost everything is coded, at 120 almost
 * nothing is). The other tables - partition, modes, motion - have a single
 * default set. */
void init_default_cdf(CdfContext* cdf, int base_q_idx);

/* Zeroes every table's adaptation counter, leaving the probabilities in place.
 *
 * This is the other half of handing a context to a later frame: the
 * specification's load_cdfs() says that once the tables are loaded "the last
 * entry in each array, representing the symbol count for that context, is set
 * to 0". The counter is what makes the adaptation slow down; a decoder that
 * kept the previous frame's counters would adapt more slowly than the encoder
 * expected, its probabilities would drift, and the two would disagree on a
 * symbol some frames later - with no symbol individually decoded wrong. */
void reset_cdf_counts(CdfContext* cdf);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_CDF_HPP */
