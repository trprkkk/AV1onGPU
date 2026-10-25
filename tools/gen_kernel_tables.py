"""Emit kernels/av1gpu_tables.h from the AV1 specification's constant tables.

The device kernels need a handful of normative tables: the cosine table the
inverse transforms are built from, the mode to angle map, the directional
derivative table, the smooth prediction weights, the intra edge filter kernel,
the filter intra taps, the inter/intra blend taper and the film grain noise. They are read out of the specification markdown so
that not a single value is retyped.

    python3 tools/gen_kernel_tables.py /tmp/av1-spec-master
"""
import re
import sys

SPEC = sys.argv[1] if len(sys.argv) > 1 else '/tmp/av1-spec-master'
text = (open(SPEC + '/08.decoding.process.md').read() +
        open(SPEC + '/10.additional.tables.md').read())


def numbers_after(pattern, count, flags=re.S):
    m = re.search(pattern, text, flags)
    if not m:
        raise SystemExit('table not found: ' + pattern)
    values = [int(v) for v in re.findall(r'-?\d+', m.group(1))]
    if len(values) < count:
        raise SystemExit('%s: found %d values, want %d' % (pattern, len(values), count))
    return values[:count]


out = []
w = out.append

w('''/*
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
''')

# ---- cos128 ---------------------------------------------------------------
cos = numbers_after(r'Cos128_Lookup\[ 65 \] = \{(.*?)\n\}', 65)
w('/* cos128(angle) = round(4096 * cos(angle * pi / 128)), for 0..64 degrees. */')
w('AV1G_TBL int av1g_tbl_cos128[65] = {')
for i in range(0, 65, 8):
    w('    ' + ', '.join(str(v) for v in cos[i:i + 8]) + ',')
w('};')
w('')

# ---- mode to angle --------------------------------------------------------
angles = numbers_after(r'Mode_To_Angle\[ INTRA_MODES \] = \{(.*?)\}', 13)
w('/* Mode_To_Angle: the prediction angle of each intra mode, in degrees. */')
w('AV1G_TBL int av1g_tbl_mode_to_angle[13] = {')
w('    ' + ', '.join(str(v) for v in angles) + ',')
w('};')
w('')

# ---- dr intra derivative --------------------------------------------------
deriv = numbers_after(r'Dr_Intra_Derivative\[ 90 \] = \{(.*?)\n\}', 90)
w('/* Dr_Intra_Derivative: the horizontal step a directional predictor takes per\n'
  ' * row, in 1/64 sample units. */')
w('AV1G_TBL int av1g_tbl_dr_deriv[90] = {')
for i in range(0, 90, 10):
    w('    ' + ', '.join(str(v) for v in deriv[i:i + 10]) + ',')
w('};')
w('')

# ---- smooth weights -------------------------------------------------------
for size in (4, 8, 16, 32, 64):
    values = numbers_after(r'Sm_Weights_Tx_%dx%d\s*\[\s*%d\s*\] = \{(.*?)\}' % (size, size, size),
                           size)
    w('/* Sm_Weights_Tx_%dx%d: the interpolation ramp of the smooth predictors. */' % (size, size))
    w('AV1G_TBL int av1g_tbl_sm_w%d[%d] = {' % (size, size))
    for i in range(0, size, 10):
        w('    ' + ', '.join(str(v) for v in values[i:i + 10]) + ',')
    w('};')
    w('')

# ---- intra edge kernel ----------------------------------------------------
w('/* Intra_Edge_Kernel: the three edge smoothing filters, by strength. */')
w('AV1G_TBL int av1g_tbl_edge_kernel[3][5] = {')
w('    {0, 4, 8, 4, 0},')
w('    {0, 5, 6, 5, 0},')
w('    {2, 4, 4, 4, 2},')
w('};')
w('')

# ---- filter intra taps ----------------------------------------------------
m = re.search(r'Intra_Filter_Taps\[ INTRA_FILTER_MODES \]\[ 8 \]\[ 7 \] = \{(.*?)\n~~~~~',
              text, re.S)
if not m:
    raise SystemExit('Intra_Filter_Taps not found')
rows = re.findall(r'\{\s*(-?\d[^{}]*)\}', m.group(1))
values = [[int(v) for v in re.findall(r'-?\d+', row)] for row in rows]
if len(values) != 40:
    raise SystemExit('Intra_Filter_Taps has %d rows, want 40' % len(values))
if any(len(row) != 7 for row in values):
    raise SystemExit('Intra_Filter_Taps rows are not 7 taps')
w('/* Intra_Filter_Taps: five modes, eight 4x2 tap sets, seven taps each. */')
w('AV1G_TBL int av1g_tbl_filter_intra[5][8][7] = {')
for mode in range(5):
    w('    {')
    for tap in range(8):
        row = values[mode * 8 + tap]
        w('        {' + ', '.join(str(v) for v in row) + '},')
    w('    },')
w('};')
w('')

# ---- inter/intra blend taper ----------------------------------------------
ii = numbers_after(r'Ii_Weights_1d\[MAX_SB_SIZE\] = \{(.*?)\n\}', 128)
w('/* Ii_Weights_1d (7.11.3.13): the taper of an inter/intra blend, indexed by the')
w(' * distance from the edge the intra half is predicted from, scaled by the block')
w(' * size. 60 at the edge and 1 far from it, so the intra prediction fades out')
w(' * across the block instead of meeting the inter one at a hard line. */')
w('AV1G_TBL int av1g_tbl_ii_weights[128] = {')
for i in range(0, 128, 16):
    w('    ' + ', '.join(str(v) for v in ii[i:i + 16]) + ',')
w('};')
w('')

# ---- gaussian sequence ----------------------------------------------------
gauss = numbers_after(r'Gaussian_Sequence\[ 2048 \] = \{(.*?)\n~~~~~', 2048)
w('/* Gaussian_Sequence (10.5): the white noise list the film grain generator')
w(' * draws from, zero mean with a standard deviation of about 512, clipped to')
w(' * [-2048, 2047] and rounded to a multiple of four. */')
w('AV1G_TBL int av1g_tbl_gaussian_sequence[2048] = {')
for i in range(0, 2048, 12):
    w('    ' + ', '.join('%5d' % v for v in gauss[i:i + 12]) + ',')
w('};')
w('')

# ---- sinpi constants ------------------------------------------------------
w('/* The four sine constants the 4 point inverse ADST is built from. */')
w('#define AV1G_SINPI_1_9 1321')
w('#define AV1G_SINPI_2_9 2482')
w('#define AV1G_SINPI_3_9 3344')
w('#define AV1G_SINPI_4_9 3803')
w('')
w('#endif /* AV1GPU_KERNEL_TABLES_H */')

path = 'kernels/av1gpu_tables.h'
open(path, 'w').write('\n'.join(out) + '\n')
print('wrote', path, len(out), 'lines')
