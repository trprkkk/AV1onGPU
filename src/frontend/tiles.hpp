/*
 * AV1onGPU - tile group parsing
 *
 * After the frame header the payload stops being plain bits and becomes a
 * sequence of independently coded tiles. This file finds where each tile starts
 * and ends; it does not decode any of them. That split matters because the
 * tile payload is arithmetic coded and each tile has its own range decoder
 * state: getting the boundaries wrong silently decodes garbage, so they are
 * parsed, checked against the payload size, and reported as explicit byte
 * ranges plus the luma rectangle each tile covers.
 *
 * The tile grid itself was parsed with the frame header (`TileInfo`); the parts
 * that live here are the two fields the specification puts in the tile group
 * instead:
 *
 *   tile_start_and_end_present_flag   only when the frame has more than one tile
 *   tile_start, tile_end              f(TileColsLog2 + TileRowsLog2) each
 *   tile_size_minus_1                 Le(TileSizeBytes), for every tile but the
 *                                     last of the group, which runs to the end of
 *                                     the payload
 *
 * The tile sizes are *not* redundant with the grid: a tile group may cover a
 * subset of the tiles, and the last tile of the group always takes the rest of
 * the OBU, so a parser that derived sizes from the grid would be wrong for any
 * stream that splits tiles across several tile group OBUs.
 */
#ifndef AV1GPU_FRONTEND_TILES_HPP
#define AV1GPU_FRONTEND_TILES_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "frontend/headers.hpp"

namespace av1gpu {
namespace frontend {

/* One coded tile: the bytes to feed to a range decoder, and the part of the
 * picture it reconstructs. Coordinates and sizes are in luma samples, in the
 * coded (pre-superres) grid. */
struct TileExtent {
  int index = 0;  /* col + row * tiles.cols */
  int col = 0;
  int row = 0;
  size_t offset = 0; /* byte offset inside the payload passed to the parser */
  size_t size = 0;
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
};

struct TileGroup {
  int start = 0; /* index of the first tile in this group */
  int end = -1;  /* index of the last tile in this group, inclusive */
  std::vector<TileExtent> tiles;
};

/* Parses the tile group that starts at `frame.tile_data_offset` in `data`.
 *
 * For an OBU_FRAME the tile group shares the OBU with the frame header, so
 * `data`/`size` are the whole OBU payload; for an OBU_TILE_GROUP they are that
 * OBU's payload and `frame.tile_data_offset` is 0. Either way the last tile of
 * the group runs to the end of `size`. */
ParseStatus parse_tile_group(const SequenceHeader& seq, const FrameHeader& frame,
                             const uint8_t* data, size_t size, TileGroup* out,
                             std::string* error);

/* True when the frame's tile grid was parsed and the tile extents in `group`
 * cover each of the frame's tiles exactly once. Used by the tests and by
 * av1gpu_probe; the real decoder relies on parse_tile_group for the same
 * property. */
bool tile_group_is_complete(const FrameHeader& frame, const TileGroup& group);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_TILES_HPP */
