/*
 * AV1onGPU - normative tables the device kernels need (generated, do not edit)
 *
 * Generated from the AV1 specification by tools/gen_kernel_tables.py. These are
 * part of the bitstream format, not tuning constants: the inverse transforms
 * and the directional predictors are only bit exact if every value matches.
 *
 * `AV1G_TBL` places them in constant memory on the device (OpenCL and CUDA both
 * need the storage class spelled out) and makes them plain constants on the
 * host, so the header compiles in all three places it is used.
 */
#ifndef AV1GPU_KERNEL_TABLES_H
#define AV1GPU_KERNEL_TABLES_H

#if defined(__OPENCL_VERSION__)
#define AV1G_TBL __constant
#elif defined(__CUDACC__)
#define AV1G_TBL __device__ __constant__
#else
#define AV1G_TBL static const
#endif

/* cos128(angle) = round(4096 * cos(angle * pi / 128)), for 0..64 degrees. */
AV1G_TBL int av1g_tbl_cos128[65] = {
    4096, 4095, 4091, 4085, 4076, 4065, 4052, 4036,
    4017, 3996, 3973, 3948, 3920, 3889, 3857, 3822,
    3784, 3745, 3703, 3659, 3612, 3564, 3513, 3461,
    3406, 3349, 3290, 3229, 3166, 3102, 3035, 2967,
    2896, 2824, 2751, 2675, 2598, 2520, 2440, 2359,
    2276, 2191, 2106, 2019, 1931, 1842, 1751, 1660,
    1567, 1474, 1380, 1285, 1189, 1092, 995, 897,
    799, 700, 601, 501, 401, 301, 201, 101,
    0,
};

/* Mode_To_Angle: the prediction angle of each intra mode, in degrees. */
AV1G_TBL int av1g_tbl_mode_to_angle[13] = {
    0, 90, 180, 45, 135, 113, 157, 203, 67, 0, 0, 0, 0,
};

/* Dr_Intra_Derivative: the horizontal step a directional predictor takes per
 * row, in 1/64 sample units. */
AV1G_TBL int av1g_tbl_dr_deriv[90] = {
    0, 0, 0, 1023, 0, 0, 547, 0, 0, 372,
    0, 0, 0, 0, 273, 0, 0, 215, 0, 0,
    178, 0, 0, 151, 0, 0, 132, 0, 0, 116,
    0, 0, 102, 0, 0, 0, 90, 0, 0, 80,
    0, 0, 71, 0, 0, 64, 0, 0, 57, 0,
    0, 51, 0, 0, 45, 0, 0, 0, 40, 0,
    0, 35, 0, 0, 31, 0, 0, 27, 0, 0,
    23, 0, 0, 19, 0, 0, 15, 0, 0, 0,
    0, 11, 0, 0, 7, 0, 0, 3, 0, 0,
};

/* Sm_Weights_Tx_4x4: the interpolation ramp of the smooth predictors. */
AV1G_TBL int av1g_tbl_sm_w4[4] = {
    255, 149, 85, 64,
};

/* Sm_Weights_Tx_8x8: the interpolation ramp of the smooth predictors. */
AV1G_TBL int av1g_tbl_sm_w8[8] = {
    255, 197, 146, 105, 73, 50, 37, 32,
};

/* Sm_Weights_Tx_16x16: the interpolation ramp of the smooth predictors. */
AV1G_TBL int av1g_tbl_sm_w16[16] = {
    255, 225, 196, 170, 145, 123, 102, 84, 68, 54,
    43, 33, 26, 20, 17, 16,
};

/* Sm_Weights_Tx_32x32: the interpolation ramp of the smooth predictors. */
AV1G_TBL int av1g_tbl_sm_w32[32] = {
    255, 240, 225, 210, 196, 182, 169, 157, 145, 133,
    122, 111, 101, 92, 83, 74, 66, 59, 52, 45,
    39, 34, 29, 25, 21, 17, 14, 12, 10, 9,
    8, 8,
};

/* Sm_Weights_Tx_64x64: the interpolation ramp of the smooth predictors. */
AV1G_TBL int av1g_tbl_sm_w64[64] = {
    255, 248, 240, 233, 225, 218, 210, 203, 196, 189,
    182, 176, 169, 163, 156, 150, 144, 138, 133, 127,
    121, 116, 111, 106, 101, 96, 91, 86, 82, 77,
    73, 69, 65, 61, 57, 54, 50, 47, 44, 41,
    38, 35, 32, 29, 27, 25, 22, 20, 18, 16,
    15, 13, 12, 10, 9, 8, 7, 6, 6, 5,
    5, 4, 4, 4,
};

/* Intra_Edge_Kernel: the three edge smoothing filters, by strength. */
AV1G_TBL int av1g_tbl_edge_kernel[3][5] = {
    {0, 4, 8, 4, 0},
    {0, 5, 6, 5, 0},
    {2, 4, 4, 4, 2},
};

/* Intra_Filter_Taps: five modes, eight 4x2 tap sets, seven taps each. */
AV1G_TBL int av1g_tbl_filter_intra[5][8][7] = {
    {
        {-6, 10, 0, 0, 0, 12, 0},
        {-5, 2, 10, 0, 0, 9, 0},
        {-3, 1, 1, 10, 0, 7, 0},
        {-3, 1, 1, 2, 10, 5, 0},
        {-4, 6, 0, 0, 0, 2, 12},
        {-3, 2, 6, 0, 0, 2, 9},
        {-3, 2, 2, 6, 0, 2, 7},
        {-3, 1, 2, 2, 6, 3, 5},
    },
    {
        {-10, 16, 0, 0, 0, 10, 0},
        {-6, 0, 16, 0, 0, 6, 0},
        {-4, 0, 0, 16, 0, 4, 0},
        {-2, 0, 0, 0, 16, 2, 0},
        {-10, 16, 0, 0, 0, 0, 10},
        {-6, 0, 16, 0, 0, 0, 6},
        {-4, 0, 0, 16, 0, 0, 4},
        {-2, 0, 0, 0, 16, 0, 2},
    },
    {
        {-8, 8, 0, 0, 0, 16, 0},
        {-8, 0, 8, 0, 0, 16, 0},
        {-8, 0, 0, 8, 0, 16, 0},
        {-8, 0, 0, 0, 8, 16, 0},
        {-4, 4, 0, 0, 0, 0, 16},
        {-4, 0, 4, 0, 0, 0, 16},
        {-4, 0, 0, 4, 0, 0, 16},
        {-4, 0, 0, 0, 4, 0, 16},
    },
    {
        {-2, 8, 0, 0, 0, 10, 0},
        {-1, 3, 8, 0, 0, 6, 0},
        {-1, 2, 3, 8, 0, 4, 0},
        {0, 1, 2, 3, 8, 2, 0},
        {-1, 4, 0, 0, 0, 3, 10},
        {-1, 3, 4, 0, 0, 4, 6},
        {-1, 2, 3, 4, 0, 4, 4},
        {-1, 2, 2, 3, 4, 3, 3},
    },
    {
        {-12, 14, 0, 0, 0, 14, 0},
        {-10, 0, 14, 0, 0, 12, 0},
        {-9, 0, 0, 14, 0, 11, 0},
        {-8, 0, 0, 0, 14, 10, 0},
        {-10, 12, 0, 0, 0, 0, 14},
        {-9, 1, 12, 0, 0, 0, 12},
        {-8, 0, 0, 12, 0, 1, 11},
        {-7, 0, 0, 1, 12, 1, 9},
    },
};

/* The four sine constants the 4 point inverse ADST is built from. */
#define AV1G_SINPI_1_9 1321
#define AV1G_SINPI_2_9 2482
#define AV1G_SINPI_3_9 3344
#define AV1G_SINPI_4_9 3803

#endif /* AV1GPU_KERNEL_TABLES_H */
