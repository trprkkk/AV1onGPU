#include "frontend/cdf.hpp"

namespace av1gpu {
namespace frontend {
namespace {

/* Default cumulative frequencies from the specification's table for the
 * partition symbol, indexed by the block level of the parent block and by the
 * above/left context. The values are cumulative and increase towards 32768,
 * exactly as the specification prints them; a row with fewer symbols than
 * kPartitionCdfSize leaves the unused tail at zero, which is never read because
 * the row's length is kPartitionSymbols[level]. */
constexpr uint16_t kPartitionDefault[kBlockLevelCount][kPartitionContexts][9] = {
    {/* 128x128 -> 64x64: eight partition types */
     {27899, 28219, 28529, 32484, 32539, 32619, 32639},
     {6607, 6990, 8268, 32060, 32219, 32338, 32371},
     {5429, 6676, 7122, 32027, 32227, 32531, 32582},
     {711, 966, 1172, 32448, 32538, 32617, 32664}},
    {/* 64x64 -> 32x32 */
     {20137, 21547, 23078, 29566, 29837, 30261, 30524, 30892, 31724},
     {6732, 7490, 9497, 27944, 28250, 28515, 28969, 29630, 30104},
     {5945, 7663, 8348, 28683, 29117, 29749, 30064, 30298, 32238},
     {870, 1212, 1487, 31198, 31394, 31574, 31743, 31881, 32332}},
    {/* 32x32 -> 16x16 */
     {18462, 20920, 23124, 27647, 28227, 29049, 29519, 30178, 31544},
     {7689, 9060, 12056, 24992, 25660, 26182, 26951, 28041, 29052},
     {6015, 9009, 10062, 24544, 25409, 26545, 27071, 27526, 32047},
     {1394, 2208, 2796, 28614, 29061, 29466, 29840, 30185, 31899}},
    {/* 16x16 -> 8x8 */
     {15597, 20929, 24571, 26706, 27664, 28821, 29601, 30571, 31902},
     {7925, 11043, 16785, 22470, 23971, 25043, 26651, 28701, 29834},
     {5414, 13269, 15111, 20488, 22360, 24500, 25537, 26336, 32117},
     {2662, 6362, 8614, 20860, 23053, 24778, 26436, 27829, 31171}},
    {/* 8x8 -> 4x4: only the four legacy partition types */
     {19132, 25510, 30392},
     {13928, 19855, 28540},
     {12522, 23679, 28629},
     {9896, 18783, 25853}},
};

/* skip, a two symbol alphabet: the probability that the block is not skipped,
 * by the number of coded neighbours. */
constexpr uint16_t kSkipDefault[kSkipContexts] = {31671, 16515, 4576};

} /* namespace */

void init_default_cdf(CdfContext* cdf) {
  if (!cdf) return;
  for (int level = 0; level < kBlockLevelCount; ++level) {
    for (int context = 0; context < kPartitionContexts; ++context) {
      uint16_t* row = cdf->partition.row(level, context);
      const int symbols = kPartitionSymbols[level];
      for (int i = 0; i < symbols; ++i) {
        row[i] = static_cast<uint16_t>(32768 - kPartitionDefault[level][context][i]);
      }
      /* the entry after the probabilities is the adaptation counter; the
       * unused tail stays zero so a stray lookup reads a probability of 0 */
      for (int i = symbols; i < kPartitionCdfSize; ++i) row[i] = 0;
    }
  }
  for (int context = 0; context < kSkipContexts; ++context) {
    cdf->skip.row(context)[0] = static_cast<uint16_t>(32768 - kSkipDefault[context]);
    cdf->skip.row(context)[1] = 0; /* counter */
  }
}

} /* namespace frontend */
} /* namespace av1gpu */
