/*
 * AV1onGPU - block sizes and partitions
 *
 * Blocks are measured in 4x4 "mi" units throughout the decoder: the tile grid,
 * the neighbour contexts and the transform sizes all live on that grid, and a
 * block size is just its width and height in those units. Keeping one unit for
 * everything is what makes the context bookkeeping (which neighbour is above,
 * which 4x4 positions a block covers) arithmetic instead of a set of
 * conversions.
 *
 * The block size numbering follows the specification, which is not raster
 * order: the square and 2:1 sizes come first, the 4:1 sizes last. The numbers
 * matter because the CDF tables and the syntax element alphabets are indexed by
 * them.
 */
#ifndef AV1GPU_FRONTEND_BLOCK_SIZE_HPP
#define AV1GPU_FRONTEND_BLOCK_SIZE_HPP

#include <cstdint>

namespace av1gpu {
namespace frontend {

/* Specification order. */
enum BlockSize {
  kBlock4x4 = 0,
  kBlock4x8,
  kBlock8x4,
  kBlock8x8,
  kBlock8x16,
  kBlock16x8,
  kBlock16x16,
  kBlock16x32,
  kBlock32x16,
  kBlock32x32,
  kBlock32x64,
  kBlock64x32,
  kBlock64x64,
  kBlock64x128,
  kBlock128x64,
  kBlock128x128,
  kBlock4x16,
  kBlock16x4,
  kBlock8x32,
  kBlock32x8,
  kBlock16x64,
  kBlock64x16,
  kBlockSizeCount,
};

/* Mi_Width_Log2 / Mi_Height_Log2: log2 of the block's width and height in 4x4
 * units. */
constexpr int8_t kBlockWidthLog2[kBlockSizeCount] = {0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3,
                                                     4, 4, 4, 5, 5, 0, 2, 1, 3, 2, 4};
constexpr int8_t kBlockHeightLog2[kBlockSizeCount] = {0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4,
                                                      3, 4, 5, 4, 5, 2, 0, 3, 1, 4, 2};

/* Partition types, in specification order, which is also the order of the CDF. */
enum PartitionType {
  kPartitionNone = 0,
  kPartitionHorizontal, /* two blocks, stacked */
  kPartitionVertical,   /* two blocks, side by side */
  kPartitionSplit,      /* four blocks */
  kPartitionHorizontalA,
  kPartitionHorizontalB,
  kPartitionVerticalA,
  kPartitionVerticalB,
  kPartitionHorizontal4,
  kPartitionVertical4,
  kPartitionTypeCount,
};

inline int block_width4(BlockSize size) { return 1 << kBlockWidthLog2[size]; }
inline int block_height4(BlockSize size) { return 1 << kBlockHeightLog2[size]; }

/* log2 of a power of two, or -1. */
inline int log2_of(int value) {
  for (int i = 0; i < 8; ++i) {
    if ((1 << i) == value) return i;
  }
  return -1;
}

/* The block size with this width and height in 4x4 units, or kBlockSizeCount
 * when the pair is not a block size. */
inline BlockSize block_size_from_dims(int w4, int h4) {
  const int lw = log2_of(w4), lh = log2_of(h4);
  if (lw < 0 || lh < 0) return kBlockSizeCount;
  for (int i = 0; i < kBlockSizeCount; ++i) {
    if (kBlockWidthLog2[i] == lw && kBlockHeightLog2[i] == lh) return static_cast<BlockSize>(i);
  }
  return kBlockSizeCount;
}

/* The block level a square block partitions at: 128x128 is level 0 and 8x8 is
 * level 4, the order the partition CDF tables are stored in. 4x4 blocks are
 * never partitioned. */
constexpr int kBlockLevels = 5;
inline int block_level_of(BlockSize size) {
  const int longer = kBlockWidthLog2[size] > kBlockHeightLog2[size] ? kBlockWidthLog2[size]
                                                                   : kBlockHeightLog2[size];
  return 5 - longer;
}

/* Square block size of a level: 0 = 128x128, 4 = 8x8. */
inline BlockSize square_block_size(int level) {
  static const BlockSize table[kBlockLevels] = {kBlock128x128, kBlock64x64, kBlock32x32,
                                                kBlock16x16, kBlock8x8};
  const int clamped = level < 0 ? 0 : (level >= kBlockLevels ? kBlockLevels - 1 : level);
  return table[clamped];
}

/* Partition_Subsize: the size of the sub-blocks a partition produces. Both
 * dimensions are plain halvings (or quarterings), which is why this does not
 * need the specification's 10 x 22 table. */
inline BlockSize partition_subsize(PartitionType partition, BlockSize parent) {
  int w4 = block_width4(parent);
  int h4 = block_height4(parent);
  switch (partition) {
    case kPartitionNone:
      break;
    case kPartitionHorizontal:
      h4 /= 2;
      break;
    case kPartitionVertical:
      w4 /= 2;
      break;
    case kPartitionHorizontal4:
      h4 /= 4;
      break;
    case kPartitionVertical4:
      w4 /= 4;
      break;
    default: /* SPLIT, and the four T shapes, use the quadrant size */
      w4 /= 2;
      h4 /= 2;
      break;
  }
  return block_size_from_dims(w4, h4);
}

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_BLOCK_SIZE_HPP */
