/*
 * AV1onGPU - tile syntax
 *
 * Everything up to here read the frame *header* with a bit reader: it says how
 * the frame is laid out. This is where the picture is actually described, one
 * superblock at a time, with the range decoder: which blocks the frame is cut
 * into, how each one is predicted, which transform sizes it uses and (in the
 * next layer) which coefficients it carries.
 *
 * The structure is a recursion, not a loop: a block is either coded as a whole
 * or split into two or four, and a split recurses at a smaller block level.
 * What makes it subtle is that the split decision itself depends on the
 * neighbours, so the decoder carries a per-4x4 map of the block sizes, skip
 * flags and transform sizes already decoded. Those maps are the frame state -
 * they are what "context" means for the syntax, and getting them wrong shows up
 * as a wrong symbol thousands of blocks later rather than immediately.
 *
 * This layer is deliberately separate from the range decoder and from the
 * headers: it consumes symbols and produces a block tree. Turning that tree
 * into prediction and residuals is the next layer, and the engine below
 * consumes the result.
 *
 * Status: intra frames - key frames and intra-only frames, which share the
 * syntax - are parsed and decoded completely. An inter frame is parsed far
 * enough to be compared against the reference decoder block by block: the skip
 * flags, the intra/inter decision, the reference tree, the motion vector
 * candidates (spatial *and* projected, the latter from the motion vector grid
 * the reference frames saved) and residual, the motion mode and the subpel
 * filter. The parts still missing (variable transform size, skip mode, compound
 * prediction, overlap and warp) stop the tile with a reason instead of
 * guessing, because a guess consumes the wrong number of symbols and
 * desynchronises the range decoder.
 */
#ifndef AV1GPU_FRONTEND_TILE_DECODER_HPP
#define AV1GPU_FRONTEND_TILE_DECODER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "frontend/block_size.hpp"
#include "frontend/cdf.hpp"
#include "frontend/headers.hpp"
#include "frontend/tx_size.hpp"

namespace av1gpu {
namespace frontend {

/* AV1's intra modes, in specification order. `kModeChromaFromLuma` is only ever
 * a chroma mode; the luma alphabet stops at kModePaeth. */
enum IntraMode {
  kModeDc = 0,
  kModeVertical,
  kModeHorizontal,
  kModeDiagonalDownLeft,
  kModeDiagonalDownRight,
  kModeVerticalRight,
  kModeHorizontalDown,
  kModeHorizontalUp,
  kModeVerticalLeft,
  kModeSmooth,
  kModeSmoothVertical,
  kModeSmoothHorizontal,
  kModePaeth,
  kModeLumaCount,
  kModeChromaFromLuma = kModeLumaCount,
  kModeChromaCount,
};

/* AV1's reference frame types, in specification order. Zero is the intra
 * "reference", which is also what the context maps store for a block that was
 * not inter predicted. */
enum RefFrameType {
  kRefIntra = 0,
  kRefLast,
  kRefLast2,
  kRefLast3,
  kRefGolden,
  kRefBwdref,
  kRefAltref2,
  kRefAltref,
  kRefFrameTypeCount,
};

/* The seven entries of `ref_frame_idx`, in specification order: they name the
 * reference *slot* each reference type points at for this frame. */
constexpr int kRefIdxLast = 0;
constexpr int kRefIdxLast2 = 1;
constexpr int kRefIdxLast3 = 2;
constexpr int kRefIdxGolden = 3;
constexpr int kRefIdxBwdref = 4;
constexpr int kRefIdxAltref2 = 5;
constexpr int kRefIdxAltref = 6;

/* Inter prediction modes. The single-reference modes come first, in the order
 * the specification's near/global/new syntax reads them; the compound ones
 * follow, in the order of the compound mode symbol. */
/* YMode for a block that is inter predicted, numbered the way the reference
 * decoders number it: the four single reference modes first, then the eight
 * compound combinations in the order the compound_mode symbol codes them. The
 * ordering of the compound half is the specification's, which is why it is not
 * simply the cross product: NEAREST_NEW and NEW_NEAREST are adjacent because
 * they are one symbol apart in the bitstream. */
enum InterPredMode {
  kModeNearestMv = 0,
  kModeNearMv = 1,
  kModeGlobalMv = 2,
  kModeNewMv = 3,
  kModeNearestNearestMv = 4,
  kModeNearNearMv = 5,
  kModeNearestNewMv = 6,
  kModeNewNearestMv = 7,
  kModeNearNewMv = 8,
  kModeNewNearMv = 9,
  kModeGlobalGlobalMv = 10,
  kModeNewNewMv = 11,
  kInterPredModeCount,
};

/* The two halves of a compound block's YMode, as a table indexed by the
 * compound_mode symbol (YMode - NEAREST_NEARESTMV). A single reference block
 * uses the same table with symbol 0 for its one vector, which is not how the
 * bitstream selects it but is how the vector is derived. */
struct CompoundModePair {
  int mode[2];
};

constexpr CompoundModePair kCompoundModePairs[8] = {
    {{kModeNearestMv, kModeNearestMv}},
    {{kModeNearMv, kModeNearMv}},
    {{kModeNewMv, kModeNearestMv}},
    {{kModeNearestMv, kModeNewMv}},
    {{kModeNearMv, kModeNewMv}},
    {{kModeNewMv, kModeNearMv}},
    {{kModeGlobalMv, kModeGlobalMv}},
    {{kModeNewMv, kModeNewMv}},
};

/* How the two predictions of a compound block are combined. The order is the
 * reference decoders' (it puts the two plain averages next to each other so a
 * comparison means "both are averages"), and the values are what the context
 * derivations test. */
enum CompoundType {
  kCompoundNone = 0,
  kCompoundWeightedAverage = 1, /* distance weighted, the jnt_comp symbol */
  kCompoundAverage = 2,
  kCompoundDiffwtd = 3,         /* masked: the difference weighted mask */
  kCompoundWedge = 4,           /* masked: a wedge mask */
};

/* Motion modes, in specification order. */
enum MotionMode {
  kMotionTranslation = 0,
  kMotionObmc,
  kMotionWarped,
};

/* One transform block: a rectangle of coefficients as the bitstream coded it.
 *
 * The parser produces levels, not pixels: the coefficient values in scan order,
 * dequantised later, plus enough description (plane, position, size, transform
 * type) for the inverse transform to place and shape them. `level_offset` is
 * where this block's levels start in TileDecodeResult::levels, so the residual
 * stage can walk them without re-parsing.
 *
 * `eob` is the number of coefficients coded, which for a sparse block is far
 * smaller than the transform area: everything else is zero and costs nothing.
 */
struct TileTransform {
  int plane = 0;        /* 0 = luma, 1 = Cb, 2 = Cr */
  TxSize size;          /* 4x4 unless the block says otherwise */
  int x = 0;            /* top left corner, in pixels */
  int y = 0;
  int qindex = 0;       /* the block's quantiser, which sets the dequant step */
  int tx_type = 0;      /* 2D transform combination, see coef_tables */
  int eob = 0;
  size_t level_offset = 0;  /* index of this block's first level in `levels` */
};

/* One decoded block: the syntax, not the pixels. Coordinates and sizes are in
 * 4x4 units on the frame's mi grid, which is the grid every other stage of the
 * decoder uses. */
struct TileBlock {
  int mi_row = 0;
  int mi_col = 0;
  BlockSize size = kBlock4x4;
  PartitionType partition = kPartitionNone;
  int block_level = 0;

  bool skip = false;
  int y_mode = kModeDc;
  int uv_mode = kModeDc;
  int y_angle = 0;  /* YMode rotated by this many 3 degree steps */
  int uv_angle = 0;
  int cfl_alpha[2] = {0, 0};
  bool filter_intra = false;
  int filter_intra_mode = 0;
  int palette_size[2] = {0, 0};
  int segment_id = 0;
  /* DeltaLFs[..][0..3]: the loop filter's delta, accumulated per superblock by
   * the delta_lf syntax (zero when the frame does not use it). */
  int delta_lf[4] = {0, 0, 0, 0};

  /* Inter mode info. `is_inter` is the specification's `intra == 0`: a block
   * of an inter frame may still be intra predicted. `ref_frame` holds
   * RefFrameType values, `mv` is in 1/8 luma sample units relative to the
   * block's position (the specification's Mv, before the per-plane scaling),
   * and `interp_filter` is 0 regular, 1 smooth, 2 sharp, 3 bilinear per
   * direction (a frame with dual filters codes one per list). */
  bool is_inter = false;
  bool skip_mode = false;
  int ref_frame[2] = {kRefIntra, kRefIntra};
  int mv[2][2] = {{0, 0}, {0, 0}};
  int inter_mode = kModeNearestMv;
  int motion_mode = kMotionTranslation;
  int interp_filter[2] = {0, 0};
  bool interintra = false;
  int interintra_mode = 0;
  /* an inter/intra block whose blend uses a wedge mask instead of a straight
   * average; the wedge's index is `wedge_index`, shared with compound wedges */
  bool interintra_wedge = false;
  /* Compound prediction: the mode picks one vector per reference, and
   * `compound_type` says how the two predictions are blended. A masked blend
   * (wedge or difference weighted) also carries the mask: the wedge's index and
   * the mask sign, or for a difference weighted mask the sign alone. */
  int compound_type = kCompoundNone;
  int wedge_index = 0;
  int mask_sign = 0;
  int drl_index = 0;

  bool has_chroma = true;
  bool have_left = false;
  bool have_above = false;

  /* The luma transform the block starts with. The full transform tree (and the
   * per-plane sizes) comes with the residual layer. */
  TxSize tx_size;
  int qindex = 0;

  /* The residual: one entry per transform block this block carries, in coding
   * order, and the transform tree they describe. */
  std::vector<TileTransform> transforms;
};

/* What one tile produced. */
struct TileDecodeResult {
  std::vector<TileBlock> blocks;

  /* Where parsing stopped, when it did. `stop_reason` is empty when the tile
   * was decoded to its end. */
  std::string stop_reason;
  bool complete = false;

  /* The residuals of the whole tile, flattened: each transform block's levels,
   * in scan order, signed, at their coded magnitude. The engine dequantises. */
  std::vector<int16_t> levels;

  /* Counters for the tile as a whole. */
  int transform_blocks = 0;
  size_t coefficients = 0;

  /* Blocks that were parsed but carry residuals the parser could not read. */
  int blocks_needing_residuals = 0;

  /* Blocks whose motion mode is overlapped block motion compensation or local
   * warped motion. Their syntax is complete, but the reconstruction stage
   * still predicts them as plain translation. */
  int obmc_blocks = 0;
  int warped_blocks = 0;

  /* cdef_idx per 64x64 block, row major over the frame, exactly as
   * read_cdef() fills it: -1 where no block that was not skipped claimed the
   * block, which leaves those blocks to the copy path of the CDEF process. */
  int cdef_cols = 0;
  int cdef_rows = 0;
  std::vector<int8_t> cdef_idx;

  /* The frame's motion vector grid, saved for the frames that refer back to
   * this one: their temporal candidate scan projects it onto itself. Null for
   * an intra frame, which has no motion vectors to store. */
  std::shared_ptr<const RefMvs> ref_mvs;

  /* Range decoder position at the end, for error reporting. */
  size_t bytes_used = 0;
  bool overrun = false;
};

/* Decodes the block tree and mode info of one intra tile.
 *
 * `data`/`size` are the tile's bytes (see TileExtent). `frame` must be the
 * parsed frame header, `seq` the sequence header in force. On success
 * `out->blocks` holds every block in coding order, which is also scan order.
 *
 * Returns Ok when the tile was parsed to its end, Unsupported when it stopped
 * at syntax this layer does not read yet (with a reason and the block it
 * stopped at), Invalid when the bitstream disagrees with itself. */
ParseStatus decode_intra_tile(const SequenceHeader& seq, const FrameHeader& frame,
                              const uint8_t* data, size_t size, TileDecodeResult* out,
                              std::string* error);

/* Decodes the block tree and mode info of one tile of any frame type.
 *
 * `refs` is the reference frame store after the frames this one refers to have
 * been decoded: it resolves `ref_frame_idx` to a slot, tells whether that slot
 * holds an intra frame (which is what decides if temporal motion vector
 * candidates exist at all) and carries the motion vector grid a reference frame
 * saved, which the candidate scan projects onto this frame. It may be null for a
 * key frame. */
ParseStatus decode_tile(const SequenceHeader& seq, const FrameHeader& frame,
                        const RefPool* refs, const uint8_t* data, size_t size,
                        TileDecodeResult* out, std::string* error);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_TILE_DECODER_HPP */
