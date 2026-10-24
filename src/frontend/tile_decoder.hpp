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
 * Status: key frames (and intra-only frames, which use the same syntax) are
 * parsed. Inter frames need reference frames, motion vectors and the compound
 * syntax, which is a separate body of work; they are reported as unsupported
 * rather than guessed at. Coefficient blocks are not read yet either: parsing
 * stops at the first block that carries residuals and says so, at the exact
 * block and byte, which is what the next layer starts from.
 */
#ifndef AV1GPU_FRONTEND_TILE_DECODER_HPP
#define AV1GPU_FRONTEND_TILE_DECODER_HPP

#include <cstddef>
#include <cstdint>
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
  int palette_size[2] = {0, 0};
  int segment_id = 0;

  bool has_chroma = true;
  bool have_left = false;
  bool have_above = false;

  /* The luma transform the block starts with. The full transform tree (and the
   * per-plane sizes) comes with the residual layer. */
  TxSize tx_size;
  int qindex = 0;
};

/* What one tile produced. */
struct TileDecodeResult {
  std::vector<TileBlock> blocks;

  /* Where parsing stopped, when it did. `stop_reason` is empty when the tile
   * was decoded to its end. */
  std::string stop_reason;
  bool complete = false;

  /* Blocks that were parsed but carry residuals the parser cannot read yet. */
  int blocks_needing_residuals = 0;

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

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_TILE_DECODER_HPP */
