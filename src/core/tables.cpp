/*
 * AV1onGPU - shared tables
 *
 * What lives here is the host side data the backends share: the subpel filter
 * weights, the film grain generator, and the colour transforms. The transform
 * and prediction tables are not here: they are normative, so they are generated
 * from the specification (tools/gen_kernel_tables.py) and compiled into the
 * kernel header, which every backend includes. Keeping one copy means the CPU
 * reference path and the GPU kernels cannot drift apart.
 */
#include "core/backend.hpp"

#include <algorithm>
#include <cstring>

namespace av1gpu {
namespace internal {
namespace tables {
namespace {

/* ---- subpel filters ---------------------------------------------------- */

/* Subpel_Filters[6][16][8] from the specification (section 7.11.3.5).
 *
 * Rows 0..3 are the four filter classes a block can name - regular, smooth,
 * sharp, bilinear - and are indexed by AV1G_FILT_*. Rows 4 and 5 are the four
 * tap versions of the regular and smooth filters that a prediction block only
 * four samples wide or high uses. All six rows are stored with eight taps so
 * the filter loops stay uniform: the four tap rows carry zeros at both ends,
 * which puts their taps at the same sample offsets the specification uses.
 *
 * The weights are Q7 (each row sums to 128) and there are sixteen phases per
 * row, one per 1/16 sample. The fractional part of a motion vector is
 * therefore *not* the index: a vector stored in 1/8 samples has even phases in
 * luma and every phase in chroma (the chroma vector is halved). Phase 0 is a
 * straight sample copy.
 *
 * These are the specification's numbers, not an approximation of them: an
 * interpolated prediction is only bit exact against a reference decoder if the
 * weights, the rounding and the phase resolution all match. */
constexpr int kFilterRows = 6;
constexpr int kFilterPhases = 16;
constexpr int kFilterTaps = 8;

const int kSubpelFilters[kFilterRows][kFilterPhases][kFilterTaps] = {
    { /* regular (EIGHTTAP) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 2, -6, 126, 8, -2, 0, 0},
        {0, 2, -10, 122, 18, -4, 0, 0},
        {0, 2, -12, 116, 28, -8, 2, 0},
        {0, 2, -14, 110, 38, -10, 2, 0},
        {0, 2, -14, 102, 48, -12, 2, 0},
        {0, 2, -16, 94, 58, -12, 2, 0},
        {0, 2, -14, 84, 66, -12, 2, 0},
        {0, 2, -14, 76, 76, -14, 2, 0},
        {0, 2, -12, 66, 84, -14, 2, 0},
        {0, 2, -12, 58, 94, -16, 2, 0},
        {0, 2, -12, 48, 102, -14, 2, 0},
        {0, 2, -10, 38, 110, -14, 2, 0},
        {0, 2, -8, 28, 116, -12, 2, 0},
        {0, 0, -4, 18, 122, -10, 2, 0},
        {0, 0, -2, 8, 126, -6, 2, 0},
    },
    { /* smooth (EIGHTTAP_SMOOTH) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 2, 28, 62, 34, 2, 0, 0},
        {0, 0, 26, 62, 36, 4, 0, 0},
        {0, 0, 22, 62, 40, 4, 0, 0},
        {0, 0, 20, 60, 42, 6, 0, 0},
        {0, 0, 18, 58, 44, 8, 0, 0},
        {0, 0, 16, 56, 46, 10, 0, 0},
        {0, -2, 16, 54, 48, 12, 0, 0},
        {0, -2, 14, 52, 52, 14, -2, 0},
        {0, 0, 12, 48, 54, 16, -2, 0},
        {0, 0, 10, 46, 56, 16, 0, 0},
        {0, 0, 8, 44, 58, 18, 0, 0},
        {0, 0, 6, 42, 60, 20, 0, 0},
        {0, 0, 4, 40, 62, 22, 0, 0},
        {0, 0, 4, 36, 62, 26, 0, 0},
        {0, 0, 2, 34, 62, 28, 2, 0},
    },
    { /* sharp (EIGHTTAP_SHARP) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {-2, 2, -6, 126, 8, -2, 2, 0},
        {-2, 6, -12, 124, 16, -6, 4, -2},
        {-2, 8, -18, 120, 26, -10, 6, -2},
        {-4, 10, -22, 116, 38, -14, 6, -2},
        {-4, 10, -22, 108, 48, -18, 8, -2},
        {-4, 10, -24, 100, 60, -20, 8, -2},
        {-4, 10, -24, 90, 70, -22, 10, -2},
        {-4, 12, -24, 80, 80, -24, 12, -4},
        {-2, 10, -22, 70, 90, -24, 10, -4},
        {-2, 8, -20, 60, 100, -24, 10, -4},
        {-2, 8, -18, 48, 108, -22, 10, -4},
        {-2, 6, -14, 38, 116, -22, 10, -4},
        {-2, 6, -10, 26, 120, -18, 8, -2},
        {-2, 4, -6, 16, 124, -12, 6, -2},
        {0, 2, -2, 8, 126, -6, 2, -2},
    },
    { /* bilinear (BILINEAR) */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, 0, 120, 8, 0, 0, 0},
        {0, 0, 0, 112, 16, 0, 0, 0},
        {0, 0, 0, 104, 24, 0, 0, 0},
        {0, 0, 0, 96, 32, 0, 0, 0},
        {0, 0, 0, 88, 40, 0, 0, 0},
        {0, 0, 0, 80, 48, 0, 0, 0},
        {0, 0, 0, 72, 56, 0, 0, 0},
        {0, 0, 0, 64, 64, 0, 0, 0},
        {0, 0, 0, 56, 72, 0, 0, 0},
        {0, 0, 0, 48, 80, 0, 0, 0},
        {0, 0, 0, 40, 88, 0, 0, 0},
        {0, 0, 0, 32, 96, 0, 0, 0},
        {0, 0, 0, 24, 104, 0, 0, 0},
        {0, 0, 0, 16, 112, 0, 0, 0},
        {0, 0, 0, 8, 120, 0, 0, 0},
    },
    { /* four tap regular, for blocks 4 wide or high */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, -4, 126, 8, -2, 0, 0},
        {0, 0, -8, 122, 18, -4, 0, 0},
        {0, 0, -10, 116, 28, -6, 0, 0},
        {0, 0, -12, 110, 38, -8, 0, 0},
        {0, 0, -12, 102, 48, -10, 0, 0},
        {0, 0, -14, 94, 58, -10, 0, 0},
        {0, 0, -12, 84, 66, -10, 0, 0},
        {0, 0, -12, 76, 76, -12, 0, 0},
        {0, 0, -10, 66, 84, -12, 0, 0},
        {0, 0, -10, 58, 94, -14, 0, 0},
        {0, 0, -10, 48, 102, -12, 0, 0},
        {0, 0, -8, 38, 110, -12, 0, 0},
        {0, 0, -6, 28, 116, -10, 0, 0},
        {0, 0, -4, 18, 122, -8, 0, 0},
        {0, 0, -2, 8, 126, -4, 0, 0},
    },
    { /* four tap smooth, for blocks 4 wide or high */
        {0, 0, 0, 128, 0, 0, 0, 0},
        {0, 0, 30, 62, 34, 2, 0, 0},
        {0, 0, 26, 62, 36, 4, 0, 0},
        {0, 0, 22, 62, 40, 4, 0, 0},
        {0, 0, 20, 60, 42, 6, 0, 0},
        {0, 0, 18, 58, 44, 8, 0, 0},
        {0, 0, 16, 56, 46, 10, 0, 0},
        {0, 0, 14, 54, 48, 12, 0, 0},
        {0, 0, 12, 52, 52, 12, 0, 0},
        {0, 0, 12, 48, 54, 14, 0, 0},
        {0, 0, 10, 46, 56, 16, 0, 0},
        {0, 0, 8, 44, 58, 18, 0, 0},
        {0, 0, 6, 42, 60, 20, 0, 0},
        {0, 0, 4, 40, 62, 22, 0, 0},
        {0, 0, 4, 36, 62, 26, 0, 0},
        {0, 0, 2, 34, 62, 30, 0, 0},
    },
};


/* Warped_Filters: the 8 tap kernel the specification's block warp process
 * filters with. Unlike Subpel_Filters this one is indexed by an *offset* rather
 * than a phase: WARPEDPIXEL_PREC_SHIFTS (64) is the 1.0 phase, and the two
 * neighbouring phases on each side are extra rows, 193 in all, because a warp
 * reads a sheared row whose phase can drift by a whole sample from the sample
 * it serves. The taps are Q7 like the subpel table.
 *
 * It is not a resampled Subpel_Filters: the 1.0 row is {0,0,0,127,1,0,0,0},
 * not {0,0,0,128,0,0,0,0}, and the reference decoders use the same table, so a
 * warp that copied a sample would be wrong. */
const int kWarpedFilters[kWarpFilterRows][8] = {
    {   0,   0, 127,   1,   0,   0,   0,   0}, {   0,  -1, 127,   2,   0,   0,   0,   0},
    {   1,  -3, 127,   4,  -1,   0,   0,   0}, {   1,  -4, 126,   6,  -2,   1,   0,   0},
    {   1,  -5, 126,   8,  -3,   1,   0,   0}, {   1,  -6, 125,  11,  -4,   1,   0,   0},
    {   1,  -7, 124,  13,  -4,   1,   0,   0}, {   2,  -8, 123,  15,  -5,   1,   0,   0},
    {   2,  -9, 122,  18,  -6,   1,   0,   0}, {   2, -10, 121,  20,  -6,   1,   0,   0},
    {   2, -11, 120,  22,  -7,   2,   0,   0}, {   2, -12, 119,  25,  -8,   2,   0,   0},
    {   3, -13, 117,  27,  -8,   2,   0,   0}, {   3, -13, 116,  29,  -9,   2,   0,   0},
    {   3, -14, 114,  32, -10,   3,   0,   0}, {   3, -15, 113,  35, -10,   2,   0,   0},
    {   3, -15, 111,  37, -11,   3,   0,   0}, {   3, -16, 109,  40, -11,   3,   0,   0},
    {   3, -16, 108,  42, -12,   3,   0,   0}, {   4, -17, 106,  45, -13,   3,   0,   0},
    {   4, -17, 104,  47, -13,   3,   0,   0}, {   4, -17, 102,  50, -14,   3,   0,   0},
    {   4, -17, 100,  52, -14,   3,   0,   0}, {   4, -18,  98,  55, -15,   4,   0,   0},
    {   4, -18,  96,  58, -15,   3,   0,   0}, {   4, -18,  94,  60, -16,   4,   0,   0},
    {   4, -18,  91,  63, -16,   4,   0,   0}, {   4, -18,  89,  65, -16,   4,   0,   0},
    {   4, -18,  87,  68, -17,   4,   0,   0}, {   4, -18,  85,  70, -17,   4,   0,   0},
    {   4, -18,  82,  73, -17,   4,   0,   0}, {   4, -18,  80,  75, -17,   4,   0,   0},
    {   4, -18,  78,  78, -18,   4,   0,   0}, {   4, -17,  75,  80, -18,   4,   0,   0},
    {   4, -17,  73,  82, -18,   4,   0,   0}, {   4, -17,  70,  85, -18,   4,   0,   0},
    {   4, -17,  68,  87, -18,   4,   0,   0}, {   4, -16,  65,  89, -18,   4,   0,   0},
    {   4, -16,  63,  91, -18,   4,   0,   0}, {   4, -16,  60,  94, -18,   4,   0,   0},
    {   3, -15,  58,  96, -18,   4,   0,   0}, {   4, -15,  55,  98, -18,   4,   0,   0},
    {   3, -14,  52, 100, -17,   4,   0,   0}, {   3, -14,  50, 102, -17,   4,   0,   0},
    {   3, -13,  47, 104, -17,   4,   0,   0}, {   3, -13,  45, 106, -17,   4,   0,   0},
    {   3, -12,  42, 108, -16,   3,   0,   0}, {   3, -11,  40, 109, -16,   3,   0,   0},
    {   3, -11,  37, 111, -15,   3,   0,   0}, {   2, -10,  35, 113, -15,   3,   0,   0},
    {   3, -10,  32, 114, -14,   3,   0,   0}, {   2,  -9,  29, 116, -13,   3,   0,   0},
    {   2,  -8,  27, 117, -13,   3,   0,   0}, {   2,  -8,  25, 119, -12,   2,   0,   0},
    {   2,  -7,  22, 120, -11,   2,   0,   0}, {   1,  -6,  20, 121, -10,   2,   0,   0},
    {   1,  -6,  18, 122,  -9,   2,   0,   0}, {   1,  -5,  15, 123,  -8,   2,   0,   0},
    {   1,  -4,  13, 124,  -7,   1,   0,   0}, {   1,  -4,  11, 125,  -6,   1,   0,   0},
    {   1,  -3,   8, 126,  -5,   1,   0,   0}, {   1,  -2,   6, 126,  -4,   1,   0,   0},
    {   0,  -1,   4, 127,  -3,   1,   0,   0}, {   0,   0,   2, 127,  -1,   0,   0,   0},
    {   0,   0,   0, 127,   1,   0,   0,   0}, {   0,   0,  -1, 127,   2,   0,   0,   0},
    {   0,   1,  -3, 127,   4,  -2,   1,   0}, {   0,   1,  -5, 127,   6,  -2,   1,   0},
    {   0,   2,  -6, 126,   8,  -3,   1,   0}, {  -1,   2,  -7, 126,  11,  -4,   2,  -1},
    {  -1,   3,  -8, 125,  13,  -5,   2,  -1}, {  -1,   3, -10, 124,  16,  -6,   3,  -1},
    {  -1,   4, -11, 123,  18,  -7,   3,  -1}, {  -1,   4, -12, 122,  20,  -7,   3,  -1},
    {  -1,   4, -13, 121,  23,  -8,   3,  -1}, {  -2,   5, -14, 120,  25,  -9,   4,  -1},
    {  -1,   5, -15, 119,  27, -10,   4,  -1}, {  -1,   5, -16, 118,  30, -11,   4,  -1},
    {  -2,   6, -17, 116,  33, -12,   5,  -1}, {  -2,   6, -17, 114,  35, -12,   5,  -1},
    {  -2,   6, -18, 113,  38, -13,   5,  -1}, {  -2,   7, -19, 111,  41, -14,   6,  -2},
    {  -2,   7, -19, 110,  43, -15,   6,  -2}, {  -2,   7, -20, 108,  46, -15,   6,  -2},
    {  -2,   7, -20, 106,  49, -16,   6,  -2}, {  -2,   7, -21, 104,  51, -16,   7,  -2},
    {  -2,   7, -21, 102,  54, -17,   7,  -2}, {  -2,   8, -21, 100,  56, -18,   7,  -2},
    {  -2,   8, -22,  98,  59, -18,   7,  -2}, {  -2,   8, -22,  96,  62, -19,   7,  -2},
    {  -2,   8, -22,  94,  64, -19,   7,  -2}, {  -2,   8, -22,  91,  67, -20,   8,  -2},
    {  -2,   8, -22,  89,  69, -20,   8,  -2}, {  -2,   8, -22,  87,  72, -21,   8,  -2},
    {  -2,   8, -21,  84,  74, -21,   8,  -2}, {  -2,   8, -22,  82,  77, -21,   8,  -2},
    {  -2,   8, -21,  79,  79, -21,   8,  -2}, {  -2,   8, -21,  77,  82, -22,   8,  -2},
    {  -2,   8, -21,  74,  84, -21,   8,  -2}, {  -2,   8, -21,  72,  87, -22,   8,  -2},
    {  -2,   8, -20,  69,  89, -22,   8,  -2}, {  -2,   8, -20,  67,  91, -22,   8,  -2},
    {  -2,   7, -19,  64,  94, -22,   8,  -2}, {  -2,   7, -19,  62,  96, -22,   8,  -2},
    {  -2,   7, -18,  59,  98, -22,   8,  -2}, {  -2,   7, -18,  56, 100, -21,   8,  -2},
    {  -2,   7, -17,  54, 102, -21,   7,  -2}, {  -2,   7, -16,  51, 104, -21,   7,  -2},
    {  -2,   6, -16,  49, 106, -20,   7,  -2}, {  -2,   6, -15,  46, 108, -20,   7,  -2},
    {  -2,   6, -15,  43, 110, -19,   7,  -2}, {  -2,   6, -14,  41, 111, -19,   7,  -2},
    {  -1,   5, -13,  38, 113, -18,   6,  -2}, {  -1,   5, -12,  35, 114, -17,   6,  -2},
    {  -1,   5, -12,  33, 116, -17,   6,  -2}, {  -1,   4, -11,  30, 118, -16,   5,  -1},
    {  -1,   4, -10,  27, 119, -15,   5,  -1}, {  -1,   4,  -9,  25, 120, -14,   5,  -2},
    {  -1,   3,  -8,  23, 121, -13,   4,  -1}, {  -1,   3,  -7,  20, 122, -12,   4,  -1},
    {  -1,   3,  -7,  18, 123, -11,   4,  -1}, {  -1,   3,  -6,  16, 124, -10,   3,  -1},
    {  -1,   2,  -5,  13, 125,  -8,   3,  -1}, {  -1,   2,  -4,  11, 126,  -7,   2,  -1},
    {   0,   1,  -3,   8, 126,  -6,   2,   0}, {   0,   1,  -2,   6, 127,  -5,   1,   0},
    {   0,   1,  -2,   4, 127,  -3,   1,   0}, {   0,   0,   0,   2, 127,  -1,   0,   0},
    {   0,   0,   0,   1, 127,   0,   0,   0}, {   0,   0,   0,  -1, 127,   2,   0,   0},
    {   0,   0,   1,  -3, 127,   4,  -1,   0}, {   0,   0,   1,  -4, 126,   6,  -2,   1},
    {   0,   0,   1,  -5, 126,   8,  -3,   1}, {   0,   0,   1,  -6, 125,  11,  -4,   1},
    {   0,   0,   1,  -7, 124,  13,  -4,   1}, {   0,   0,   2,  -8, 123,  15,  -5,   1},
    {   0,   0,   2,  -9, 122,  18,  -6,   1}, {   0,   0,   2, -10, 121,  20,  -6,   1},
    {   0,   0,   2, -11, 120,  22,  -7,   2}, {   0,   0,   2, -12, 119,  25,  -8,   2},
    {   0,   0,   3, -13, 117,  27,  -8,   2}, {   0,   0,   3, -13, 116,  29,  -9,   2},
    {   0,   0,   3, -14, 114,  32, -10,   3}, {   0,   0,   3, -15, 113,  35, -10,   2},
    {   0,   0,   3, -15, 111,  37, -11,   3}, {   0,   0,   3, -16, 109,  40, -11,   3},
    {   0,   0,   3, -16, 108,  42, -12,   3}, {   0,   0,   4, -17, 106,  45, -13,   3},
    {   0,   0,   4, -17, 104,  47, -13,   3}, {   0,   0,   4, -17, 102,  50, -14,   3},
    {   0,   0,   4, -17, 100,  52, -14,   3}, {   0,   0,   4, -18,  98,  55, -15,   4},
    {   0,   0,   4, -18,  96,  58, -15,   3}, {   0,   0,   4, -18,  94,  60, -16,   4},
    {   0,   0,   4, -18,  91,  63, -16,   4}, {   0,   0,   4, -18,  89,  65, -16,   4},
    {   0,   0,   4, -18,  87,  68, -17,   4}, {   0,   0,   4, -18,  85,  70, -17,   4},
    {   0,   0,   4, -18,  82,  73, -17,   4}, {   0,   0,   4, -18,  80,  75, -17,   4},
    {   0,   0,   4, -18,  78,  78, -18,   4}, {   0,   0,   4, -17,  75,  80, -18,   4},
    {   0,   0,   4, -17,  73,  82, -18,   4}, {   0,   0,   4, -17,  70,  85, -18,   4},
    {   0,   0,   4, -17,  68,  87, -18,   4}, {   0,   0,   4, -16,  65,  89, -18,   4},
    {   0,   0,   4, -16,  63,  91, -18,   4}, {   0,   0,   4, -16,  60,  94, -18,   4},
    {   0,   0,   3, -15,  58,  96, -18,   4}, {   0,   0,   4, -15,  55,  98, -18,   4},
    {   0,   0,   3, -14,  52, 100, -17,   4}, {   0,   0,   3, -14,  50, 102, -17,   4},
    {   0,   0,   3, -13,  47, 104, -17,   4}, {   0,   0,   3, -13,  45, 106, -17,   4},
    {   0,   0,   3, -12,  42, 108, -16,   3}, {   0,   0,   3, -11,  40, 109, -16,   3},
    {   0,   0,   3, -11,  37, 111, -15,   3}, {   0,   0,   2, -10,  35, 113, -15,   3},
    {   0,   0,   3, -10,  32, 114, -14,   3}, {   0,   0,   2,  -9,  29, 116, -13,   3},
    {   0,   0,   2,  -8,  27, 117, -13,   3}, {   0,   0,   2,  -8,  25, 119, -12,   2},
    {   0,   0,   2,  -7,  22, 120, -11,   2}, {   0,   0,   1,  -6,  20, 121, -10,   2},
    {   0,   0,   1,  -6,  18, 122,  -9,   2}, {   0,   0,   1,  -5,  15, 123,  -8,   2},
    {   0,   0,   1,  -4,  13, 124,  -7,   1}, {   0,   0,   1,  -4,  11, 125,  -6,   1},
    {   0,   0,   1,  -3,   8, 126,  -5,   1}, {   0,   0,   1,  -2,   6, 126,  -4,   1},
    {   0,   0,   0,  -1,   4, 127,  -3,   1}, {   0,   0,   0,   0,   2, 127,  -1,   0},
    {   0,   0,   0,   0,   2, 127,  -1,   0},
};


constexpr int kUpscaleShifts = 64;

/* Upscale_Filter: the eight tap kernel the superres upscaling process resamples
 * a coded frame with (specification 7.15). One row per 1/64 of a sample, so the
 * table has 64 phases like the subpel ones, and every row sums to 128 - it is
 * Q7 in tap units, and the process rounds by FILTER_BITS (7) after the sum.
 *
 * The rows are appended to the shared filter table right after the warp kernel
 * (AV1G_UPSCALE_FILTER_OFFSET), so one uploaded table still serves every filter
 * a prediction or a post filter needs. */
const int kUpscaleFilter[64][8] = {
    {    0,    0,    0,  128,    0,    0,    0,    0 }, {    0,    0,   -1,  128,    2,   -1,    0,    0 },
    {    0,    1,   -3,  127,    4,   -2,    1,    0 }, {    0,    1,   -4,  127,    6,   -3,    1,    0 },
    {    0,    2,   -6,  126,    8,   -3,    1,    0 }, {    0,    2,   -7,  125,   11,   -4,    1,    0 },
    {   -1,    2,   -8,  125,   13,   -5,    2,    0 }, {   -1,    3,   -9,  124,   15,   -6,    2,    0 },
    {   -1,    3,  -10,  123,   18,   -6,    2,   -1 }, {   -1,    3,  -11,  122,   20,   -7,    3,   -1 },
    {   -1,    4,  -12,  121,   22,   -8,    3,   -1 }, {   -1,    4,  -13,  120,   25,   -9,    3,   -1 },
    {   -1,    4,  -14,  118,   28,   -9,    3,   -1 }, {   -1,    4,  -15,  117,   30,  -10,    4,   -1 },
    {   -1,    5,  -16,  116,   32,  -11,    4,   -1 }, {   -1,    5,  -16,  114,   35,  -12,    4,   -1 },
    {   -1,    5,  -17,  112,   38,  -12,    4,   -1 }, {   -1,    5,  -18,  111,   40,  -13,    5,   -1 },
    {   -1,    5,  -18,  109,   43,  -14,    5,   -1 }, {   -1,    6,  -19,  107,   45,  -14,    5,   -1 },
    {   -1,    6,  -19,  105,   48,  -15,    5,   -1 }, {   -1,    6,  -19,  103,   51,  -16,    5,   -1 },
    {   -1,    6,  -20,  101,   53,  -16,    6,   -1 }, {   -1,    6,  -20,   99,   56,  -17,    6,   -1 },
    {   -1,    6,  -20,   97,   58,  -17,    6,   -1 }, {   -1,    6,  -20,   95,   61,  -18,    6,   -1 },
    {   -2,    7,  -20,   93,   64,  -18,    6,   -2 }, {   -2,    7,  -20,   91,   66,  -19,    6,   -1 },
    {   -2,    7,  -20,   88,   69,  -19,    6,   -1 }, {   -2,    7,  -20,   86,   71,  -19,    6,   -1 },
    {   -2,    7,  -20,   84,   74,  -20,    7,   -2 }, {   -2,    7,  -20,   81,   76,  -20,    7,   -1 },
    {   -2,    7,  -20,   79,   79,  -20,    7,   -2 }, {   -1,    7,  -20,   76,   81,  -20,    7,   -2 },
    {   -2,    7,  -20,   74,   84,  -20,    7,   -2 }, {   -1,    6,  -19,   71,   86,  -20,    7,   -2 },
    {   -1,    6,  -19,   69,   88,  -20,    7,   -2 }, {   -1,    6,  -19,   66,   91,  -20,    7,   -2 },
    {   -2,    6,  -18,   64,   93,  -20,    7,   -2 }, {   -1,    6,  -18,   61,   95,  -20,    6,   -1 },
    {   -1,    6,  -17,   58,   97,  -20,    6,   -1 }, {   -1,    6,  -17,   56,   99,  -20,    6,   -1 },
    {   -1,    6,  -16,   53,  101,  -20,    6,   -1 }, {   -1,    5,  -16,   51,  103,  -19,    6,   -1 },
    {   -1,    5,  -15,   48,  105,  -19,    6,   -1 }, {   -1,    5,  -14,   45,  107,  -19,    6,   -1 },
    {   -1,    5,  -14,   43,  109,  -18,    5,   -1 }, {   -1,    5,  -13,   40,  111,  -18,    5,   -1 },
    {   -1,    4,  -12,   38,  112,  -17,    5,   -1 }, {   -1,    4,  -12,   35,  114,  -16,    5,   -1 },
    {   -1,    4,  -11,   32,  116,  -16,    5,   -1 }, {   -1,    4,  -10,   30,  117,  -15,    4,   -1 },
    {   -1,    3,   -9,   28,  118,  -14,    4,   -1 }, {   -1,    3,   -9,   25,  120,  -13,    4,   -1 },
    {   -1,    3,   -8,   22,  121,  -12,    4,   -1 }, {   -1,    3,   -7,   20,  122,  -11,    3,   -1 },
    {   -1,    2,   -6,   18,  123,  -10,    3,   -1 }, {    0,    2,   -6,   15,  124,   -9,    3,   -1 },
    {    0,    2,   -5,   13,  125,   -8,    2,   -1 }, {    0,    1,   -4,   11,  125,   -7,    2,    0 },
    {    0,    1,   -3,    8,  126,   -6,    2,    0 }, {    0,    1,   -3,    6,  127,   -4,    1,    0 },
    {    0,    1,   -2,    4,  127,   -3,    1,    0 }, {    0,    0,   -1,    2,  128,   -1,    0,    0 },
};

struct FilterTable {
  std::vector<int> data;
  FilterTable() {
    data.resize(static_cast<size_t>(kFilterRows) * kFilterPhases * kFilterTaps +
                static_cast<size_t>(kWarpFilterRows) * 8 + static_cast<size_t>(kUpscaleShifts) * 8);
    for (int row = 0; row < kFilterRows; ++row)
      for (int phase = 0; phase < kFilterPhases; ++phase)
        for (int tap = 0; tap < kFilterTaps; ++tap)
          data[(static_cast<size_t>(row) * kFilterPhases + phase) * kFilterTaps + tap] =
              kSubpelFilters[row][phase][tap];
    /* the warp kernel follows the subpel rows: the kernels index it at
     * AV1G_WARP_FILTER_OFFSET, so this order is an interface, not a detail */
    for (int row = 0; row < kWarpFilterRows; ++row)
      for (int tap = 0; tap < 8; ++tap)
        data[static_cast<size_t>(kFilterRows) * kFilterPhases * kFilterTaps +
             static_cast<size_t>(row) * 8 + tap] = kWarpedFilters[row][tap];
    /* the superres upscale kernel follows the warp rows, at the offset the
     * kernels index it at (AV1G_UPSCALE_FILTER_OFFSET) */
    const size_t upscale_off = static_cast<size_t>(kFilterRows) * kFilterPhases * kFilterTaps +
                               static_cast<size_t>(kWarpFilterRows) * 8;
    for (int phase = 0; phase < kUpscaleShifts; ++phase)
      for (int tap = 0; tap < 8; ++tap)
        data[upscale_off + static_cast<size_t>(phase) * 8 + tap] = kUpscaleFilter[phase][tap];
  }
};

} /* namespace */

/* Transform_Row_Shift, keyed by the transform size.
 *
 * The specification's table is
 *
 *   Transform_Row_Shift[ TX_SIZES_ALL ] = {
 *     0, 1, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2
 *   }
 *
 * indexed by the specification's Tx_Size numbers, in this order: 4x4, 8x8,
 * 16x16, 32x32, 64x64, 4x8, 8x4, 8x16, 16x8, 16x32, 32x16, 32x64, 64x32,
 * 4x16, 16x4, 8x32, 32x8, 16x64, 64x16. Two things in it surprise: the shift
 * stops growing at 32x32 (larger transforms are coded with the same scale),
 * and the tall thin sizes 4x16/16x4 shift by 1 while 8x32/32x8 shift by 2 -
 * it is not a function of the width alone, so it has to be the table, not a
 * formula. */
int tx_row_shift(int w, int h) {
  static const struct {
    int w, h, shift;
  } kTable[] = {
      {4, 4, 0},   {8, 8, 1},   {16, 16, 2}, {32, 32, 2}, {64, 64, 2},
      {4, 8, 0},   {8, 4, 0},   {8, 16, 1},  {16, 8, 1},  {16, 32, 1},
      {32, 16, 1}, {32, 64, 1}, {64, 32, 1}, {4, 16, 1},  {16, 4, 1},
      {8, 32, 2},  {32, 8, 2},  {16, 64, 2}, {64, 16, 2},
  };
  for (const auto& e : kTable)
    if (e.w == w && e.h == h) return e.shift;
  return 0;
}

const std::vector<int>& filters() {
  static const FilterTable table;
  return table.data;
}

/* ------------------------------------------------------------------------- */
/* setup shear (7.11.3.6) and resolve divisor (7.11.3.7)                     */
/* ------------------------------------------------------------------------- */

namespace {

/* Div_Lut[DIV_LUT_NUM]: 1/x in Q14 for x in [1, 2), which turns the affine
 * model's division into a multiply and a shift. */
constexpr int kDivLutBits = 8;
constexpr int kDivLutPrecBits = 14;
const int kDivLut[257] = {
    16384, 16320, 16257, 16194, 16132, 16070, 16009, 15948, 15888, 15828, 15768,
    15709, 15650, 15592, 15534, 15477, 15420, 15364, 15308, 15252, 15197, 15142,
    15087, 15033, 14980, 14926, 14873, 14821, 14769, 14717, 14665, 14614, 14564,
    14513, 14463, 14413, 14364, 14315, 14266, 14218, 14170, 14122, 14075, 14028,
    13981, 13935, 13888, 13843, 13797, 13752, 13707, 13662, 13618, 13574, 13530,
    13487, 13443, 13400, 13358, 13315, 13273, 13231, 13190, 13148, 13107, 13066,
    13026, 12985, 12945, 12906, 12866, 12827, 12788, 12749, 12710, 12672, 12633,
    12596, 12558, 12520, 12483, 12446, 12409, 12373, 12336, 12300, 12264, 12228,
    12193, 12157, 12122, 12087, 12053, 12018, 11984, 11950, 11916, 11882, 11848,
    11815, 11782, 11749, 11716, 11683, 11651, 11619, 11586, 11555, 11523, 11491,
    11460, 11429, 11398, 11367, 11336, 11305, 11275, 11245, 11215, 11185, 11155,
    11125, 11096, 11067, 11038, 11009, 10980, 10951, 10923, 10894, 10866, 10838,
    10810, 10782, 10755, 10727, 10700, 10673, 10645, 10618, 10592, 10565, 10538,
    10512, 10486, 10460, 10434, 10408, 10382, 10356, 10331, 10305, 10280, 10255,
    10230, 10205, 10180, 10156, 10131, 10107, 10082, 10058, 10034, 10010, 9986,
    9963,  9939,  9916,  9892,  9869,  9846,  9823,  9800,  9777,  9754,  9732,
    9709,  9687,  9664,  9642,  9620,  9598,  9576,  9554,  9533,  9511,  9489,
    9468,  9447,  9425,  9404,  9383,  9362,  9341,  9321,  9300,  9279,  9259,
    9239,  9218,  9198,  9178,  9158,  9138,  9118,  9098,  9079,  9059,  9039,
    9020,  9001,  8981,  8962,  8943,  8924,  8905,  8886,  8867,  8849,  8830,
    8812,  8793,  8775,  8756,  8738,  8720,  8702,  8684,  8666,  8648,  8630,
    8613,  8595,  8577,  8560,  8542,  8525,  8508,  8490,  8473,  8456,  8439,
    8422,  8405,  8389,  8372,  8355,  8339,  8322,  8306,  8289,  8273,  8257,
    8240,  8224,  8208,  8192,
};

int floor_log2(int v) {
  int n = 0;
  while ((1 << (n + 1)) <= v) ++n;
  return n;
}

/* Round2Signed: the specification rounds towards the *larger* value, so a
 * negative input rounds towards zero rather than down. An arithmetic shift
 * rounds towards minus infinity, so the sign has to be split out. */
int round2_signed(int v, int n) {
  const int round = (1 << n) >> 1;
  return v >= 0 ? (v + round) >> n : -((-v + round) >> n);
}

int64_t round2s64(int64_t v, int n) {
  const int64_t round = static_cast<int64_t>(1) << (n - 1);
  return v >= 0 ? (v + round) >> n : -((-v + round) >> n);
}

void resolve_divisor(int d, int* shift, int* factor) {
  const int abs_d = d < 0 ? -d : d;
  const int n = floor_log2(abs_d);
  const int e = abs_d - (1 << n);
  const int f = n > kDivLutBits ? (e + (1 << ((n - kDivLutBits) - 1))) >> (n - kDivLutBits)
                                : e << (kDivLutBits - n);
  *shift = n + kDivLutPrecBits;
  *factor = d < 0 ? -kDivLut[f] : kDivLut[f];
}

} /* namespace */

bool gm_setup_shear(const int32_t* params, int* alpha, int* beta, int* gamma, int* delta) {
  constexpr int kPrecBits = 16;
  constexpr int kReduceBits = 6;
  constexpr int kClip = 32767;
  const int alpha0 = std::min(kClip, std::max(-kClip - 1, params[2] - (1 << kPrecBits)));
  const int beta0 = std::min(kClip, std::max(-kClip - 1, params[3]));
  int div_shift = 0, div_factor = 0;
  resolve_divisor(params[2], &div_shift, &div_factor);
  const int64_t v = static_cast<int64_t>(params[4]) << kPrecBits;
  /* Round2Signed(v * divFactor, divShift) in 64 bit: v is already shifted up by
   * 16 and the divisor factor is Q14, so the product does not fit 32 */
  const int64_t vd = v * div_factor;
  const int gamma0 =
      std::min(kClip, std::max(-kClip - 1, static_cast<int>(round2s64(vd, div_shift))));
  const int64_t w = static_cast<int64_t>(params[3]) * params[4];
  const int64_t wd = w * div_factor;
  const int delta0 = std::min(kClip,
                              std::max(-kClip - 1, static_cast<int>(params[5] -
                                                                    round2s64(wd, div_shift) -
                                                                    (1 << kPrecBits))));
  *alpha = round2_signed(alpha0, kReduceBits) << kReduceBits;
  *beta = round2_signed(beta0, kReduceBits) << kReduceBits;
  *gamma = round2_signed(gamma0, kReduceBits) << kReduceBits;
  *delta = round2_signed(delta0, kReduceBits) << kReduceBits;
  if (4 * std::abs(*alpha) + 7 * std::abs(*beta) >= (1 << kPrecBits)) return false;
  if (4 * std::abs(*gamma) + 4 * std::abs(*delta) >= (1 << kPrecBits)) return false;
  return true;
}

/* ------------------------------------------------------------------------- */
/* film grain template                                                       */
/* ------------------------------------------------------------------------- */

namespace {

/* xorshift, matching the distribution the AV1 film grain model expects (zero
 * mean, roughly gaussian tails). */
uint32_t rng_state = 1u;

uint32_t rng_next() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

} /* namespace */

void grain_template(int seed, int stride, int rows, std::vector<signed char>* out) {
  out->assign(static_cast<size_t>(stride) * rows, 0);
  rng_state = static_cast<uint32_t>(seed) * 2654435761u + 12345u;
  if (rng_state == 0) rng_state = 1;
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < stride; ++x) {
      /* sum of two uniforms gives a triangular, zero mean distribution */
      const int a = static_cast<int>(rng_next() % 129) - 64;
      const int b = static_cast<int>(rng_next() % 129) - 64;
      int v = (a + b) / 2;
      if (v < -64) v = -64;
      if (v > 63) v = 63;
      (*out)[static_cast<size_t>(y) * stride + x] = static_cast<signed char>(v);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* colour                                                                    */
/* ------------------------------------------------------------------------- */

void color_matrix(ColorMatrix matrix, ColorRange range, int bit_depth, int m[8],
                  int o[3]) {
  /* Y'CbCr -> R'G'B', Q16, samples already normalised to 8 bit range by the
   * kernel.  The offsets fold in the 16 / 128 / 0 nominal levels. */
  const int shift = bit_depth - 8;
  const int y_off = (range == ColorRange::Limited) ? (16 << shift) : 0;
  const int c_off = (range == ColorRange::Limited) ? (128 << shift)
                                                  : (1 << (bit_depth - 1));
  const int half = 1 << 15;
  switch (matrix) {
    case ColorMatrix::Bt2020: {
      m[0] = 76293;   /* 1.1644 * Y            */
      m[1] = 109969;  /* 1.6781 * (V - 128)    */
      m[2] = 76293;
      m[3] = -12255;  /* -0.1870 * (U - 128)   */
      m[4] = -42598;  /* -0.6500 * (V - 128)   */
      m[5] = 76293;
      m[6] = 121634;  /* 1.8556 * (U - 128)    */
      o[0] = half - m[0] * y_off - m[1] * c_off;
      o[1] = half - m[2] * y_off - m[3] * c_off - m[4] * c_off;
      o[2] = half - m[5] * y_off - m[6] * c_off;
      break;
    }
    case ColorMatrix::Identity: {
      /* GBR: the three planes are already R, G and B */
      m[0] = 65536;
      m[1] = 0;
      m[2] = 65536;
      m[3] = 0;
      m[4] = 0;
      m[5] = 65536;
      m[6] = 0;
      o[0] = 0;
      o[1] = 0;
      o[2] = 0;
      break;
    }
    case ColorMatrix::Bt709:
    default: {
      m[0] = 76293;   /* 1.1644 * Y            */
      m[1] = 117504;  /* 1.7930 * (V - 128)    */
      m[2] = 76293;
      m[3] = -13926;  /* -0.2125 * (U - 128)   */
      m[4] = -34937;  /* -0.5331 * (V - 128)   */
      m[5] = 76293;
      m[6] = 138412;  /* 2.1124 * (U - 128)    */
      o[0] = half - m[0] * y_off - m[1] * c_off;
      o[1] = half - m[2] * y_off - m[3] * c_off - m[4] * c_off;
      o[2] = half - m[5] * y_off - m[6] * c_off;
      break;
    }
  }
}

void frame_to_rgb(const Frame& frame, ColorMatrix matrix, ColorRange range,
                  std::vector<uint8_t>* rgb) {
  rgb->resize(static_cast<size_t>(frame.width) * frame.height * 3);
  int m[8];
  int o[3];
  /* the nominal-range offsets are folded into o[] by color_matrix(), exactly
   * like the kernel gets them */
  color_matrix(matrix, range, frame.bit_depth, m, o);

  for (int y = 0; y < frame.height; ++y) {
    const uint16_t* yr = frame.plane(0) + static_cast<size_t>(y) * frame.stride[0];
    const uint16_t* ur = frame.plane(1) + static_cast<size_t>(y >> frame.ss_y) * frame.stride[1];
    const uint16_t* vr = frame.plane(2) + static_cast<size_t>(y >> frame.ss_y) * frame.stride[2];
    uint8_t* row = rgb->data() + static_cast<size_t>(y) * frame.width * 3;
    for (int x = 0; x < frame.width; ++x) {
      const int yy = static_cast<int>(yr[x]);
      const int uu = static_cast<int>(ur[x >> frame.ss_x]);
      const int vv = static_cast<int>(vr[x >> frame.ss_x]);
      int r = (m[0] * yy + m[1] * vv + o[0]) >> 16;
      int g = (m[2] * yy + m[3] * uu + m[4] * vv + o[1]) >> 16;
      int b = (m[5] * yy + m[6] * uu + o[2]) >> 16;
      row[3 * x + 0] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
      row[3 * x + 1] = static_cast<uint8_t>(g < 0 ? 0 : (g > 255 ? 255 : g));
      row[3 * x + 2] = static_cast<uint8_t>(b < 0 ? 0 : (b > 255 ? 255 : b));
    }
  }
}

/* ---- masked compound: the wedge codebook ------------------------------- */

namespace {

/* Wedge_Codebook[3][16][3] from the specification (section 7.11.3.11): the
 * master mask direction and the offset, in eighths of the block size, that
 * places the wedge for one block shape. The three rows are the tall shapes, the
 * wide ones and the square ones - the wedge table is indexed by shape rather
 * than by size, so an 8x16 and a 16x32 block share a row. */
struct WedgeCode {
  int dir;
  int xoff;
  int yoff;
};

const WedgeCode kWedgeCodebook[3][16] = {
    {{AV1G_WEDGE_OBLIQUE27, 4, 4}, {AV1G_WEDGE_OBLIQUE63, 4, 4},
     {AV1G_WEDGE_OBLIQUE117, 4, 4}, {AV1G_WEDGE_OBLIQUE153, 4, 4},
     {AV1G_WEDGE_HORIZONTAL, 4, 2}, {AV1G_WEDGE_HORIZONTAL, 4, 4},
     {AV1G_WEDGE_HORIZONTAL, 4, 6}, {AV1G_WEDGE_VERTICAL, 4, 4},
     {AV1G_WEDGE_OBLIQUE27, 4, 2}, {AV1G_WEDGE_OBLIQUE27, 4, 6},
     {AV1G_WEDGE_OBLIQUE153, 4, 2}, {AV1G_WEDGE_OBLIQUE153, 4, 6},
     {AV1G_WEDGE_OBLIQUE63, 2, 4}, {AV1G_WEDGE_OBLIQUE63, 6, 4},
     {AV1G_WEDGE_OBLIQUE117, 2, 4}, {AV1G_WEDGE_OBLIQUE117, 6, 4}},
    {{AV1G_WEDGE_OBLIQUE27, 4, 4}, {AV1G_WEDGE_OBLIQUE63, 4, 4},
     {AV1G_WEDGE_OBLIQUE117, 4, 4}, {AV1G_WEDGE_OBLIQUE153, 4, 4},
     {AV1G_WEDGE_VERTICAL, 2, 4}, {AV1G_WEDGE_VERTICAL, 4, 4},
     {AV1G_WEDGE_VERTICAL, 6, 4}, {AV1G_WEDGE_HORIZONTAL, 4, 4},
     {AV1G_WEDGE_OBLIQUE27, 4, 2}, {AV1G_WEDGE_OBLIQUE27, 4, 6},
     {AV1G_WEDGE_OBLIQUE153, 4, 2}, {AV1G_WEDGE_OBLIQUE153, 4, 6},
     {AV1G_WEDGE_OBLIQUE63, 2, 4}, {AV1G_WEDGE_OBLIQUE63, 6, 4},
     {AV1G_WEDGE_OBLIQUE117, 2, 4}, {AV1G_WEDGE_OBLIQUE117, 6, 4}},
    {{AV1G_WEDGE_OBLIQUE27, 4, 4}, {AV1G_WEDGE_OBLIQUE63, 4, 4},
     {AV1G_WEDGE_OBLIQUE117, 4, 4}, {AV1G_WEDGE_OBLIQUE153, 4, 4},
     {AV1G_WEDGE_HORIZONTAL, 4, 2}, {AV1G_WEDGE_HORIZONTAL, 4, 6},
     {AV1G_WEDGE_VERTICAL, 2, 4}, {AV1G_WEDGE_VERTICAL, 6, 4},
     {AV1G_WEDGE_OBLIQUE27, 4, 2}, {AV1G_WEDGE_OBLIQUE27, 4, 6},
     {AV1G_WEDGE_OBLIQUE153, 4, 2}, {AV1G_WEDGE_OBLIQUE153, 4, 6},
     {AV1G_WEDGE_OBLIQUE63, 2, 4}, {AV1G_WEDGE_OBLIQUE63, 6, 4},
     {AV1G_WEDGE_OBLIQUE117, 2, 4}, {AV1G_WEDGE_OBLIQUE117, 6, 4}},
};

/* Wedge_Bits[bsize] > 0: the shapes that carry a wedge at all. Everything
 * larger than 32x32 uses the difference weighted mask instead, which is why the
 * master mask only ever has to cover 32 samples. */
bool wedge_shape(int w, int h, int* shape) {
  if (w < 8 || h < 8 || w > 32 || h > 32) return false;
  *shape = h > w ? 0 : (h < w ? 1 : 2);
  return true;
}

} /* namespace */

bool wedge_entry(int block_w, int block_h, int wedge_index, int* dir, int* xoff, int* yoff,
                 int* flip) {
  int shape = 0;
  if (wedge_index < 0 || wedge_index >= 16 || !wedge_shape(block_w, block_h, &shape))
    return false;
  const WedgeCode& c = kWedgeCodebook[shape][wedge_index];
  *dir = c.dir;
  *xoff = 32 - ((c.xoff * block_w) >> 3);
  *yoff = 32 - ((c.yoff * block_h) >> 3);
  /* flipSign: the specification stores both signs of every wedge and picks one
   * by the sign of the mask's mean. Only the first row and the first column are
   * summed, which is enough: the wedge is monotone along both. */
  int sum = 0;
  for (int i = 0; i < block_w; ++i) sum += av1g_wedge_master(c.dir, *yoff, *xoff + i);
  for (int i = 1; i < block_h; ++i) sum += av1g_wedge_master(c.dir, *yoff + i, *xoff);
  const int n = block_w + block_h - 1;
  const int avg = (sum + n / 2) / n;
  *flip = avg < 32 ? 1 : 0;
  return true;
}

} /* namespace tables */
} /* namespace internal */
} /* namespace av1gpu */
