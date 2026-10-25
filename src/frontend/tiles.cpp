#include "frontend/tiles.hpp"

#include <algorithm>

#include "frontend/bitreader.hpp"

namespace av1gpu {
namespace frontend {
namespace {

ParseStatus invalid(const char* message, std::string* error) {
  if (error) *error = message;
  return ParseStatus::Invalid;
}

/* The luma rectangle covered by tile (col, row). The grid is stored in super
 * blocks, so the last tile of a row or column is clipped to the frame. */
void tile_rect(const FrameHeader& frame, const SequenceHeader& seq, int col, int row,
               TileExtent* extent) {
  const TileInfo& tiles = frame.tiles;
  const int sb_size = seq.use_128x128_superblock ? 128 : 64;
  const int col_start = tiles.col_start_sb[static_cast<size_t>(col)];
  const int col_end = tiles.col_start_sb[static_cast<size_t>(col) + 1];
  const int row_start = tiles.row_start_sb[static_cast<size_t>(row)];
  const int row_end = tiles.row_start_sb[static_cast<size_t>(row) + 1];
  extent->x = col_start * sb_size;
  extent->y = row_start * sb_size;
  extent->w = std::min(frame.width, col_end * sb_size) - extent->x;
  extent->h = std::min(frame.height, row_end * sb_size) - extent->y;
}

} /* namespace */

ParseStatus parse_tile_group(const SequenceHeader& seq, const FrameHeader& frame,
                             const uint8_t* data, size_t size, TileGroup* out,
                             std::string* error) {
  if (error) error->clear();
  if (!out) return invalid("parse_tile_group: no output", error);
  *out = TileGroup();

  const TileInfo& tiles = frame.tiles;
  if (tiles.cols <= 0 || tiles.rows <= 0 ||
      tiles.col_start_sb.size() != static_cast<size_t>(tiles.cols) + 1 ||
      tiles.row_start_sb.size() != static_cast<size_t>(tiles.rows) + 1) {
    return invalid("tile group: the frame header has no usable tile grid", error);
  }
  if (frame.tile_data_offset > size) {
    return invalid("tile group: tile data starts past the end of the payload", error);
  }
  const int total_tiles = tiles.cols * tiles.rows;

  BitReader r(data, size);
  r.skip(frame.tile_data_offset * 8);

  int tile_start = 0;
  int tile_end = total_tiles - 1;
  if (total_tiles > 1 && r.read_bit()) {
    /* the group covers a subset of the frame's tiles */
    const int bits = tiles.log2_cols + tiles.log2_rows;
    tile_start = static_cast<int>(r.read(bits));
    tile_end = static_cast<int>(r.read(bits));
    if (r.overrun()) return invalid("tile group: truncated start/end fields", error);
    if (tile_start > tile_end || tile_end >= total_tiles) {
      return invalid("tile group: tile range out of bounds", error);
    }
  }
  /* tile_group_obu() byte aligns before the size fields */
  r.align_to_byte();
  if (r.overrun()) return invalid("tile group: truncated header", error);

  /* Each tile but the last is preceded by its own size field: the sizes are
   * interleaved with the tile data, not collected in front of it, because the
   * decoder consumes each tile as soon as its header has been read. */
  size_t offset = r.bytes_read();
  for (int index = tile_start; index <= tile_end; ++index) {
    size_t tile_size = 0;
    if (index == tile_end) {
      /* the last tile of the group always runs to the end of the payload */
      if (offset > size) return invalid("tile group: no tile payload", error);
      tile_size = size - offset;
    } else {
      if (offset + static_cast<size_t>(tiles.tile_size_bytes) > size) {
        return invalid("tile group: truncated tile size", error);
      }
      uint32_t minus_one = 0;
      for (int i = 0; i < tiles.tile_size_bytes; ++i) {
        minus_one |= static_cast<uint32_t>(data[offset + static_cast<size_t>(i)]) << (8 * i);
      }
      offset += static_cast<size_t>(tiles.tile_size_bytes);
      tile_size = static_cast<size_t>(minus_one) + 1;
    }
    if (tile_size > size - offset) {
      return invalid("tile group: tile runs past the end of the payload", error);
    }
    TileExtent extent;
    extent.index = index;
    extent.col = index % tiles.cols;
    extent.row = index / tiles.cols;
    extent.offset = offset;
    extent.size = tile_size;
    tile_rect(frame, seq, extent.col, extent.row, &extent);
    out->tiles.push_back(extent);
    offset += tile_size;
  }
  out->start = tile_start;
  out->end = tile_end;
  return ParseStatus::Ok;
}

bool tile_group_is_complete(const FrameHeader& frame, const TileGroup& group) {
  const int total = frame.tiles.cols * frame.tiles.rows;
  if (total <= 0 || group.start != 0 || group.end != total - 1 ||
      group.end - group.start + 1 != static_cast<int>(group.tiles.size())) {
    return false;
  }
  std::vector<bool> seen(static_cast<size_t>(total), false);
  for (const TileExtent& tile : group.tiles) {
    if (tile.index < 0 || tile.index >= total) return false;
    if (seen[static_cast<size_t>(tile.index)]) return false;
    seen[static_cast<size_t>(tile.index)] = true;
    /* the union of the tiles has to be the frame, so an empty tile means the
     * grid and the geometry disagree */
    if (tile.w <= 0 || tile.h <= 0) return false;
  }
  for (int i = group.start; i <= group.end; ++i) {
    if (!seen[static_cast<size_t>(i)]) return false;
  }
  return true;
}

} /* namespace frontend */
} /* namespace av1gpu */
