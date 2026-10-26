#!/usr/bin/env python3
"""AV1onGPU - OpenCL kernel self test.

Compiles the real kernel sources (``kernels/av1gpu_kernels.h`` +
``kernels/av1_opencl.cl``) with an actual OpenCL C compiler and runs a handful of
jobs, checking the results against values computed here in Python.

The C++ OpenCL backend needs an ICD loader, which some containers and CI images
do not have - a pure software implementation does work though:

    pip install pyopencl pocl-binary-distribution
    python3 tests/opencl_selftest.py
    python3 tests/opencl_selftest.py --list-devices   # what a machine has
    python3 tests/opencl_selftest.py --device 1       # the second one, not the first

On a machine that also has a software OpenCL implementation next to the real
card, the first device is not always the card: ``--list-devices`` names them and
``--device`` picks one, so the kernels are compiled by the driver that matters.

The job-struct layouts and the shared tables are not duplicated here: a small
probe, compiled from ``src/core/tables.cpp`` with the project's own compiler,
prints them, so the test always exercises the numbers the backends upload.

Exit codes: 0 all checks passed, 1 failure, 77 OpenCL not available (skip).
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
KERNELS = os.path.join(ROOT, "kernels")

PROBE = r"""
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>
#include "core/backend.hpp"
#include "av1gpu_kernels.h"

static void emit(const std::vector<int>& v, const char* name) {
  std::printf("{\"table\":\"%s\",\"data\":[", name);
  for (size_t i = 0; i < v.size(); ++i)
    std::printf("%s%d", i ? "," : "", v[i]);
  std::printf("]}\n");
}
/* the host build of the kernel that the device has to agree with, sample for
 * sample: the whole point of this project is that both are the same code */
static void emit_plane(const char* what, const uint16_t* plane, int stride, int x,
                       int y, int w, int h) {
  std::printf("{\"host\":\"%s\",\"data\":[", what);
  for (int i = 0; i < h; ++i)
    for (int j = 0; j < w; ++j)
      std::printf("%s%d", i || j ? "," : "",
                  (int)plane[(size_t)(y + i) * stride + x + j]);
  std::printf("]}\n");
}
#define F(T, f) std::printf("{\"struct\":\"%s\",\"field\":\"%s\",\"offset\":%zu}\n", #T, #f, offsetof(T, f))
#define FA(T, f, n) \
  for (int i = 0; i < (n); ++i) \
    std::printf("{\"struct\":\"%s\",\"field\":\"%s%d\",\"offset\":%zu}\n", #T, #f, i, \
                offsetof(T, f) + (size_t)i * sizeof(((T*)0)->f[0]))
/* an array of ints, one entry per element */
#define FA2(T, f, m) \
  for (int j = 0; j < (m); ++j) \
    std::printf("{\"struct\":\"%s\",\"field\":\"%s_%d\",\"offset\":%zu}\n", #T, #f, j, \
                offsetof(T, f) + (size_t)j * sizeof(((T*)0)->f[0]))
/* an array of structs of plain ints: the mirror needs one int per member, so the
 * probe walks the array as a flat int view and names them f<i>_<j> */
#define FS(T, f, n, E) \
  for (int i = 0; i < (n); ++i) \
    for (int j = 0; j < (int)(sizeof(E) / sizeof(int)); ++j) \
      std::printf("{\"struct\":\"%s\",\"field\":\"%s%d_%d\",\"offset\":%zu}\n", #T, #f, i, j, \
                  offsetof(T, f) + (size_t)i * sizeof(E) + (size_t)j * sizeof(int))
/* a two dimensional int array */
#define FM(T, f, n, m) \
  for (int i = 0; i < (n); ++i) \
    for (int j = 0; j < (m); ++j) \
      std::printf("{\"struct\":\"%s\",\"field\":\"%s%d_%d\",\"offset\":%zu}\n", #T, #f, i, j, \
                  offsetof(T, f) + ((size_t)i * (m) + (size_t)j) * sizeof(((T*)0)->f[0][0]))
#define S(T) std::printf("{\"struct\":\"%s\",\"size\":%zu}\n", #T, sizeof(T))

int main() {
  using namespace av1gpu::internal;

  /* ------------------------------------------------------------- tables */
  emit(tables::filters(), "filters");

  /* ------------------------------------------------- inverse transform */
  /* one DC coefficient in an 8x8 DCT_DCT: the same job the test runs on the
   * device, with the result the host build of the kernel produces */
  {
    const int stride = 32, size = 32;
    std::vector<int> coef(8 * 8 + 8, 0);
    coef[0] = 8 * 8;
    std::vector<int> tmp(8 * 8 + 8, 0);
    std::vector<uint16_t> plane((size_t)stride * size, 0);
    Av1gItxJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 8 * stride + 8;
    job.dst_stride = stride;
    job.w = 8;
    job.h = 8;
    job.tx_type = AV1G_TX_DCT_DCT;
    job.eob = 8 * 8;
    job.row_shift = tables::tx_row_shift(8, 8);
    job.bitdepth = 8;
    for (int r = 0; r < 8; ++r) av1g_itx_row(&job, coef.data(), tmp.data(), r);
    for (int c = 0; c < 8; ++c)
      av1g_itx_col_add(&job, tmp.data(), plane.data(), c, 255);
    emit_plane("itx", plane.data(), stride, 8, 8, 8, 8);
  }

  /* ------------------------------------------------------ intra, DC only */
  {
    const int stride = 32, size = 32;
    std::vector<uint16_t> plane((size_t)stride * size, 0);
    for (int x = 0; x < 8; ++x) plane[7 * stride + 8 + x] = 100;
    plane[7 * stride + 7] = 100;
    for (int y = 0; y < 8; ++y) plane[(8 + y) * stride + 7] = 200;
    Av1gIntraJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 8 * stride + 8;
    job.dst_stride = stride;
    job.x = 8;
    job.y = 8;
    job.w = 8;
    job.h = 8;
    job.mode = AV1G_PRED_DC;
    job.bitdepth = 8;
    job.have_left = 1;
    job.have_above = 1;
    job.filter_type = -1;
    job.max_x = stride - 1;
    job.max_y = size - 1;
    av1g_intra_block(&job, plane.data(), nullptr, 255);
    emit_plane("intra", plane.data(), stride, 8, 8, 8, 8);
  }

  /* --------------------------------------------------------- intra, palette */
  /* a palette block ignores its neighbours: every sample is a lookup into the
   * job's colour table through the coded colour map */
  {
    const int stride = 32, size = 32;
    std::vector<uint16_t> plane((size_t)stride * size, 0);
    unsigned char map[4 * 4] = {0, 1, 2, 1,
                                2, 0, 1, 2,
                                1, 2, 0, 1,
                                2, 1, 2, 0};
    Av1gIntraJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 8 * stride + 8;
    job.dst_stride = stride;
    job.x = 8;
    job.y = 8;
    job.w = 4;
    job.h = 4;
    job.bitdepth = 8;
    job.palette = 1;
    job.palette_size = 3;
    job.palette_color[0] = 12;
    job.palette_color[1] = 200;
    job.palette_color[2] = 77;
    job.palette_off = 0;
    job.palette_stride = 4;
    av1g_intra_block(&job, plane.data(), map, 255);
    emit_plane("palette", plane.data(), stride, 8, 8, 4, 4);
  }

  /* ---------------------------------------------------------------- cdef */
  /* a block of noise, filtered with the frame header's strengths: the kernel
   * has to find the direction itself, exactly as a decoder does */
  {
    const int stride = 32, size = 32;
    std::vector<uint16_t> src((size_t)stride * size, 0), dst;
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < stride; ++x)
        src[(size_t)y * stride + x] = (uint16_t)(128 + (((x * 5 + y * 11) * 7 + 3) % 17) - 8);
    dst = src;
    Av1gCdefJob job;
    std::memset(&job, 0, sizeof(job));
    job.x = 8;
    job.y = 8;
    job.src_off = 8 * stride + 8;
    job.src_stride = stride;
    job.dst_off = job.src_off;
    job.dst_stride = stride;
    job.luma_off = job.src_off;
    job.luma_stride = stride;
    job.pri_strength = 8;
    job.sec_strength = 0;
    job.damping = 3;
    job.dir = -1;
    job.bitdepth = 8;
    job.plane_w = stride;
    job.plane_h = size;
    av1g_cdef_block(&job, src.data(), dst.data(), src.data(), 255);
    emit_plane("cdef", dst.data(), stride, 8, 8, 8, 8);
  }

  /* --------------------------------------------------- loop restoration */
  /* Two snapshots: the CDEF output and the deblocked picture before it. The top
   * rows of the second are the two lines a stripe may borrow, so a device that
   * reads the wrong pointer disagrees on the rows at the top of the stripe. */
  {
    const int stride = 32, size = 32;
    std::vector<uint16_t> cdef((size_t)stride * size, 0), curr;
    for (int y = 0; y < size; ++y)
      for (int x = 0; x < stride; ++x)
        cdef[(size_t)y * stride + x] = (uint16_t)(90 + ((x * 5 + y * 3) % 23));
    curr = cdef;
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < stride; ++x) curr[(size_t)y * stride + x] = 200;
    Av1gLrJob job;
    std::memset(&job, 0, sizeof(job));
    job.dst_off = 8 * stride + 8;
    job.dst_stride = stride;
    job.cdef_stride = stride;
    job.curr_stride = stride;
    job.x = 8;
    job.y = 8;
    job.w = 8;
    job.h = 8;
    job.plane_w = stride;
    job.plane_h = size;
    job.stripe_start_y = 8;
    job.stripe_end_y = 23;
    job.bitdepth = 8;
    /* Wiener: the vertical taps reach two rows above, so the first output rows
     * mix the two pictures */
    job.type = AV1G_LR_WIENER;
    job.vfilter[0] = 3;
    job.vfilter[1] = -2;
    job.vfilter[2] = 5;
    job.hfilter[0] = 1;
    job.hfilter[1] = 6;
    job.hfilter[2] = -3;
    {
      std::vector<uint16_t> plane = cdef;
      av1g_lr_block(&job, cdef.data(), curr.data(), plane.data(), 255);
      emit_plane("lr_wiener", plane.data(), stride, 8, 8, 8, 8);
    }
    /* self guided: one pass with a radius, one without (which then uses the
     * sample itself), and the two projection weights */
    job.type = AV1G_LR_SGRPROJ;
    job.sgr_radius[0] = 2;
    job.sgr_eps[0] = 12;
    job.sgr_radius[1] = 0;
    job.sgr_eps[1] = 0;
    job.sgr_xqd[0] = 95;
    job.sgr_xqd[1] = 33;
    {
      std::vector<uint16_t> plane = cdef;
      av1g_lr_block(&job, cdef.data(), curr.data(), plane.data(), 255);
      emit_plane("lr_sgr", plane.data(), stride, 8, 8, 8, 8);
    }
    /* the same job at 10 bit, where the box filter's sums are scaled back to
     * 8 bit instead of being used as they are */
    {
      std::vector<uint16_t> cdef10(cdef.size(), 0), curr10(curr.size(), 0);
      for (size_t i = 0; i < cdef.size(); ++i) {
        cdef10[i] = (uint16_t)(cdef[i] * 4);
        curr10[i] = (uint16_t)(curr[i] * 4);
      }
      job.bitdepth = 10;
      std::vector<uint16_t> plane = cdef10;
      av1g_lr_block(&job, cdef10.data(), curr10.data(), plane.data(), 1023);
      emit_plane("lr_sgr10", plane.data(), stride, 8, 8, 8, 8);
    }
  }

  /* ------------------------------------------------------------ upscale */
  {
    const int src_stride = 24, dst_stride = 32, rows = 8;
    std::vector<uint16_t> src((size_t)src_stride * rows, 0);
    for (int y = 0; y < rows; ++y)
      for (int x = 0; x < src_stride; ++x)
        src[(size_t)y * src_stride + x] = (uint16_t)(40 + ((x * 7 + y * 11) % 61));
    std::vector<uint16_t> dst((size_t)dst_stride * rows, 0);
    Av1gUpscaleJob job;
    std::memset(&job, 0, sizeof(job));
    job.src_stride = src_stride;
    job.dst_stride = dst_stride;
    job.rows = rows;
    job.upscaled_w = dst_stride;
    job.max_x = src_stride - 1;
    job.bitdepth = 8;
    av1g_superres_params(src_stride, dst_stride, &job.step_x, &job.initial_subpel_x);
    av1g_upscale_plane(&job, src.data(), dst.data(), tables::filters().data(), 255);
    emit_plane("upscale", dst.data(), dst_stride, 0, 0, 8, 8);
    emit({job.step_x, job.initial_subpel_x, rows, dst_stride}, "upscale_params");
  }

  /* ------------------------------------------------------------- layouts */
  S(Av1gItxJob);
  F(Av1gItxJob, coef_off); F(Av1gItxJob, tmp_off); F(Av1gItxJob, dst_off);
  F(Av1gItxJob, dst_stride); F(Av1gItxJob, w); F(Av1gItxJob, h);
  F(Av1gItxJob, tx_type); F(Av1gItxJob, eob); F(Av1gItxJob, row_shift);
  F(Av1gItxJob, lossless); F(Av1gItxJob, bitdepth);
  S(Av1gIntraJob);
  F(Av1gIntraJob, dst_off); F(Av1gIntraJob, dst_stride); F(Av1gIntraJob, x);
  F(Av1gIntraJob, y); F(Av1gIntraJob, w); F(Av1gIntraJob, h);
  F(Av1gIntraJob, mode); F(Av1gIntraJob, angle); F(Av1gIntraJob, bitdepth);
  F(Av1gIntraJob, have_left); F(Av1gIntraJob, have_above);
  F(Av1gIntraJob, have_above_right); F(Av1gIntraJob, have_below_left);
  F(Av1gIntraJob, filter_type); F(Av1gIntraJob, filter_intra);
  F(Av1gIntraJob, filter_intra_mode); F(Av1gIntraJob, cfl_alpha);
  F(Av1gIntraJob, luma_off); F(Av1gIntraJob, luma_stride);
  F(Av1gIntraJob, luma_w); F(Av1gIntraJob, luma_h); F(Av1gIntraJob, ss_x);
  F(Av1gIntraJob, ss_y); F(Av1gIntraJob, max_x); F(Av1gIntraJob, max_y);
  F(Av1gIntraJob, palette); F(Av1gIntraJob, palette_size);
  FA(Av1gIntraJob, palette_color, 8);
  F(Av1gIntraJob, palette_off); F(Av1gIntraJob, palette_stride);
  S(Av1gInterJob);
  F(Av1gInterJob, dst_off); F(Av1gInterJob, dst_stride); F(Av1gInterJob, plane);
  F(Av1gInterJob, w);
  F(Av1gInterJob, h); F(Av1gInterJob, ref0_off); F(Av1gInterJob, ref0_stride);
  F(Av1gInterJob, ref1_off); F(Av1gInterJob, ref1_stride); F(Av1gInterJob, ref0_slot);
  F(Av1gInterJob, ref1_slot); F(Av1gInterJob, ref_x); F(Av1gInterJob, ref_y);
  F(Av1gInterJob, ref_w); F(Av1gInterJob, ref_h); F(Av1gInterJob, border);
  F(Av1gInterJob, mv0_x); F(Av1gInterJob, mv0_y); F(Av1gInterJob, mv1_x);
  F(Av1gInterJob, mv1_y); F(Av1gInterJob, filt); F(Av1gInterJob, compound);
  F(Av1gInterJob, w0); F(Av1gInterJob, w1); F(Av1gInterJob, bitdepth);
  F(Av1gInterJob, sub_x); F(Av1gInterJob, sub_y); F(Av1gInterJob, scale_x);
  F(Av1gInterJob, scale_y); F(Av1gInterJob, filt_v);
  F(Av1gInterJob, mask_kind); F(Av1gInterJob, mask_sign);
  F(Av1gInterJob, wedge_dir); F(Av1gInterJob, wedge_xoff); F(Av1gInterJob, wedge_yoff);
  F(Av1gInterJob, wedge_flip);
  F(Av1gInterJob, luma_ref0_off); F(Av1gInterJob, luma_ref1_off);
  F(Av1gInterJob, luma_stride); F(Av1gInterJob, luma_x); F(Av1gInterJob, luma_y);
  F(Av1gInterJob, luma_ref_w); F(Av1gInterJob, luma_ref_h);
  F(Av1gInterJob, obmc_count);
  FS(Av1gInterJob, obmc, 8, Av1gObmcCandidate);
  FA2(Av1gInterJob, warp, 2);
  FM(Av1gInterJob, warp_params, 2, 6);
  FM(Av1gInterJob, warp_abcd, 2, 4);
  FA2(Av1gInterJob, mask_warp, 2);
  F(Av1gInterJob, interintra); F(Av1gInterJob, interintra_mode);
  F(Av1gInterJob, interintra_wedge); F(Av1gInterJob, have_left);
  F(Av1gInterJob, have_above); F(Av1gInterJob, max_x); F(Av1gInterJob, max_y);
  S(Av1gDeblockJob);
  F(Av1gDeblockJob, p_off); F(Av1gDeblockJob, stride); F(Av1gDeblockJob, step);
  F(Av1gDeblockJob, len); F(Av1gDeblockJob, level); F(Av1gDeblockJob, sharpness);
  F(Av1gDeblockJob, filter_size); F(Av1gDeblockJob, is_chroma);
  F(Av1gDeblockJob, bitdepth);
  S(Av1gCdefJob);
  F(Av1gCdefJob, x); F(Av1gCdefJob, y);
  F(Av1gCdefJob, src_off); F(Av1gCdefJob, src_stride); F(Av1gCdefJob, dst_off);
  F(Av1gCdefJob, dst_stride); F(Av1gCdefJob, luma_off); F(Av1gCdefJob, luma_stride);
  F(Av1gCdefJob, pri_strength); F(Av1gCdefJob, sec_strength);
  F(Av1gCdefJob, damping); F(Av1gCdefJob, dir); F(Av1gCdefJob, uv);
  F(Av1gCdefJob, ss_x); F(Av1gCdefJob, ss_y); F(Av1gCdefJob, bitdepth);
  F(Av1gCdefJob, plane_w); F(Av1gCdefJob, plane_h);
  S(Av1gUpscaleJob);
  F(Av1gUpscaleJob, src_off); F(Av1gUpscaleJob, dst_off);
  F(Av1gUpscaleJob, src_stride); F(Av1gUpscaleJob, dst_stride);
  F(Av1gUpscaleJob, y0); F(Av1gUpscaleJob, rows);
  F(Av1gUpscaleJob, upscaled_w); F(Av1gUpscaleJob, max_x);
  F(Av1gUpscaleJob, step_x); F(Av1gUpscaleJob, initial_subpel_x);
  F(Av1gUpscaleJob, bitdepth);
  S(Av1gLrJob);
  F(Av1gLrJob, plane); F(Av1gLrJob, dst_off); F(Av1gLrJob, dst_stride);
  F(Av1gLrJob, cdef_off); F(Av1gLrJob, cdef_stride);
  F(Av1gLrJob, curr_off); F(Av1gLrJob, curr_stride);
  F(Av1gLrJob, x); F(Av1gLrJob, y); F(Av1gLrJob, w); F(Av1gLrJob, h);
  F(Av1gLrJob, plane_w); F(Av1gLrJob, plane_h);
  F(Av1gLrJob, stripe_start_y); F(Av1gLrJob, stripe_end_y);
  F(Av1gLrJob, type);
  FA(Av1gLrJob, vfilter, 3); FA(Av1gLrJob, hfilter, 3);
  FA(Av1gLrJob, sgr_radius, 2); FA(Av1gLrJob, sgr_eps, 2);
  FA(Av1gLrJob, sgr_xqd, 2);
  F(Av1gLrJob, bitdepth);
  S(Av1gGrainJob);
  F(Av1gGrainJob, dst_off); F(Av1gGrainJob, dst_stride);
  F(Av1gGrainJob, noise_off); F(Av1gGrainJob, noise_stride);
  F(Av1gGrainJob, x); F(Av1gGrainJob, y); F(Av1gGrainJob, w); F(Av1gGrainJob, h);
  F(Av1gGrainJob, plane); F(Av1gGrainJob, ss_x); F(Av1gGrainJob, ss_y);
  F(Av1gGrainJob, bitdepth); F(Av1gGrainJob, scaling_shift);
  F(Av1gGrainJob, num_y_points); F(Av1gGrainJob, num_cb_points); F(Av1gGrainJob, num_cr_points);
  F(Av1gGrainJob, chroma_scaling_from_luma);
  F(Av1gGrainJob, cb_mult); F(Av1gGrainJob, cb_luma_mult); F(Av1gGrainJob, cb_offset);
  F(Av1gGrainJob, cr_mult); F(Av1gGrainJob, cr_luma_mult); F(Av1gGrainJob, cr_offset);
  F(Av1gGrainJob, min_value); F(Av1gGrainJob, max_value);
  F(Av1gGrainJob, luma_off); F(Av1gGrainJob, luma_stride); F(Av1gGrainJob, luma_w);
  S(Av1gColorJob);
  F(Av1gColorJob, dst_off); F(Av1gColorJob, dst_stride); F(Av1gColorJob, w);
  F(Av1gColorJob, h); F(Av1gColorJob, y_off); F(Av1gColorJob, y_stride);
  F(Av1gColorJob, u_off); F(Av1gColorJob, v_off); F(Av1gColorJob, c_stride);
  F(Av1gColorJob, ss_x); F(Av1gColorJob, ss_y); FA(Av1gColorJob, m, 8);
  FA(Av1gColorJob, o, 3); F(Av1gColorJob, bitdepth);
  return 0;
}
"""


def run_probe():
    """Compiles and runs the layout/table probe, returning its parsed output."""
    cxx = os.environ.get("CXX", "c++")
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "probe.cpp")
        exe = os.path.join(tmp, "probe")
        with open(src, "w", encoding="utf-8") as handle:
            handle.write(PROBE)
        subprocess.run([cxx, "-std=c++17", "-O1", "-I", os.path.join(ROOT, "include"),
                        "-I", os.path.join(ROOT, "src"), "-I", KERNELS, src,
                        os.path.join(ROOT, "src", "core", "tables.cpp"), "-o", exe],
                       check=True)
        out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout

    tables = {}
    sizes = {}
    fields = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if "table" in entry:
            tables[entry["table"]] = entry["data"]
        elif "host" in entry:
            tables["host." + entry["host"]] = entry["data"]
        elif "size" in entry:
            sizes[entry["struct"]] = entry["size"]
        else:
            fields.setdefault(entry["struct"], []).append((entry["offset"], entry["field"]))

    layouts = {}
    for name, entries in fields.items():
        entries.sort()
        ordered = [field for _, field in entries]
        # every member of every job struct is a plain int
        cls = type(name, (ctypes.Structure,),
                   {"_fields_": [(f, ctypes.c_int) for f in ordered]})
        assert ctypes.sizeof(cls) == sizes[name], (name, ctypes.sizeof(cls), sizes[name])
        for offset, field in entries:
            assert getattr(cls, field).offset == offset, (name, field)
        layouts[name] = cls
    return layouts, tables


def build_source():
    """The device source, assembled exactly as cmake/embed.cmake does it.

    The device compiler gets one translation unit, so the quoted includes next to
    the header (the generated table header) are inlined rather than resolved by
    an include path the library does not have either.
    """
    names = sorted(os.listdir(KERNELS))
    with open(os.path.join(KERNELS, "av1gpu_kernels.h"), "r", encoding="utf-8") as handle:
        header = handle.read()
    inlined = ""
    for name in names:
        line = '#include "%s"' % name
        if name.endswith(".h") and line in header:
            with open(os.path.join(KERNELS, name), "r", encoding="utf-8") as handle:
                inlined += handle.read() + "\n"
            header = header.replace(line, "")
    with open(os.path.join(KERNELS, "av1_opencl.cl"), "r", encoding="utf-8") as handle:
        body = handle.read()
    return inlined + header + "\n" + body.replace('#include "av1gpu_kernels.h"', "")


def devices():
    """All OpenCL devices, or an empty list when there is no usable runtime.

    A container without an ICD loader makes clGetPlatformIDs raise instead of
    returning zero platforms, which is a "not here" and not a test failure.
    """
    try:
        import pyopencl as cl
    except ImportError:
        return []
    try:
        return [d for p in cl.get_platforms() for d in p.get_devices()]
    except cl.LogicError:
        return []


def main(argv=None):
    parser = argparse.ArgumentParser(description="compile and run the OpenCL kernels")
    parser.add_argument("--device", type=int, default=0,
                        help="index into --list-devices (default 0)")
    parser.add_argument("--list-devices", action="store_true",
                        help="print every OpenCL device and exit")
    args = parser.parse_args(argv)

    try:
        import numpy as np
        import pyopencl as cl
    except ImportError as exc:
        print("OpenCL self test skipped: %s" % exc)
        return 77

    found = devices()
    if not found:
        print("OpenCL self test skipped: no OpenCL platform")
        return 77
    if args.list_devices:
        for index, entry in enumerate(found):
            platform = entry.platform
            print("[%d] %s (%s) on %s" % (index, entry.name, entry.version,
                                          platform.name))
        return 0
    if args.device < 0 or args.device >= len(found):
        print("device %d does not exist: %d OpenCL device(s) found"
              % (args.device, len(found)))
        for index, entry in enumerate(found):
            print("  [%d] %s" % (index, entry.name))
        return 1
    device = found[args.device]
    print("device: %s (%s)" % (device.name, device.version))

    source = build_source()
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    try:
        program = cl.Program(context, source).build(["-cl-std=CL2.0"])
    except Exception as exc:  # noqa: BLE001
        print("BUILD FAILED\n%s" % exc)
        return 1
    print("kernel source compiled (%d bytes)" % len(source))

    layouts, tables = run_probe()
    ItxJob = layouts["Av1gItxJob"]
    IntraJob = layouts["Av1gIntraJob"]
    InterJob = layouts["Av1gInterJob"]
    DeblockJob = layouts["Av1gDeblockJob"]
    CdefJob = layouts["Av1gCdefJob"]
    UpscaleJob = layouts["Av1gUpscaleJob"]
    LrJob = layouts["Av1gLrJob"]
    GrainJob = layouts["Av1gGrainJob"]
    ColorJob = layouts["Av1gColorJob"]

    mf = cl.mem_flags
    failures = []

    def check(name, got, want=lambda: None, ok=None):
        passed = ok if ok is not None else (got == want)
        if passed:
            print("  ok   %-42s %s" % (name, got))
        else:
            print("  FAIL %-42s got %s want %s" % (name, got, want))
            failures.append(name)

    def upload(array):
        return cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=np.ascontiguousarray(array))

    def download(buf, shape, dtype=np.uint16):
        back = np.zeros(shape, dtype=dtype)
        cl.enqueue_copy(queue, back, buf).wait()
        return back

    def launch(kernel, handler, elements, *args):
        """Enqueues like the backend: the global size is rounded up."""
        local = 64
        global_size = ((elements + local - 1) // local) * local
        handler(queue, (global_size,), (local,), *args)
        return elements

    def prefix(counts):
        """njobs+1 prefix sums, padded like the backends do."""
        table = [0] * (len(counts) + 2)
        total = 0
        for i, count in enumerate(counts):
            table[i] = total
            total += count
        table[len(counts)] = total
        table[len(counts) + 1] = total
        return np.array(table, dtype=np.int32), total

    filter_buf = upload(np.array(tables["filters"], dtype=np.int32))

    # ------------------------------------------------------------------ itx
    print("inverse transform")
    stride, size, n = 32, 32, 8
    plane = np.zeros(stride * size, dtype=np.uint16)
    coef = np.zeros(n * n + 8, dtype=np.int32)
    coef[0] = 8 * n
    job = ItxJob()
    job.dst_off = 8 * stride + 8
    job.dst_stride = stride
    job.w = n
    job.h = n
    job.tx_type = 0          # AV1G_TX_DCT_DCT
    job.eob = n * n
    job.row_shift = 2        # Transform_Row_Shift for an 8x8 transform
    job.bitdepth = 8
    row_pref, rows = prefix([n])      # one work item per transform row
    col_pref, cols = prefix([n])      # one work item per transform column
    job_buf = upload(np.frombuffer(job, dtype=np.uint8).copy())
    row_pref_buf = upload(row_pref)
    col_pref_buf = upload(col_pref)
    coef_buf = upload(coef)
    tmp_buf = cl.Buffer(context, mf.READ_WRITE, coef.nbytes)
    plane_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=plane)
    njobs = np.int32(1)
    launch(None, program.av1g_k_itx_rows, rows, job_buf, row_pref_buf, njobs,
           np.int32(rows), coef_buf, tmp_buf)
    launch(None, program.av1g_k_itx_cols, cols, job_buf, col_pref_buf, njobs,
           np.int32(cols), tmp_buf, plane_buf, np.int32(255))
    queue.finish()
    out = download(plane_buf, plane.shape)
    block = out.reshape(size, stride)[8:16, 8:16].astype(int)
    # the host build of the same kernel is the reference: the device has to
    # agree with it sample for sample, not just on average
    want = np.array(tables["host.itx"], dtype=int).reshape(8, 8)
    check("itx matches the host kernel", bool((block == want).all()), True)
    check("itx DC only is flat", int(block.max() - block.min()), 0)

    # ---------------------------------------------------------------- intra
    print("intra prediction")
    plane = np.zeros(stride * size, dtype=np.uint16)
    view = plane.reshape(size, stride)
    view[7, 8:16] = 100  # row above the block
    view[7, 7] = 100     # corner (above-left)
    view[8:16, 7] = 200  # column left of the block
    ij = IntraJob()
    ij.dst_off = 8 * stride + 8
    ij.dst_stride = stride
    ij.x = 8
    ij.y = 8
    ij.w = 8
    ij.h = 8
    ij.mode = 0  # AV1G_PRED_DC
    ij.bitdepth = 8
    ij.have_left = 1
    ij.have_above = 1
    ij.filter_type = -1   # no edge filtering
    ij.luma_stride = stride
    ij.ss_x = 1
    ij.ss_y = 1
    ij.max_x = stride - 1
    ij.max_y = size - 1
    ijob_buf = upload(np.frombuffer(ij, dtype=np.uint8).copy())
    plane_buf = upload(plane)
    ipref, itotal = prefix([1])
    ipref_buf = upload(ipref)
    palette_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                            hostbuf=np.zeros(8, dtype=np.uint8))
    launch(None, program.av1g_k_intra, itotal, ijob_buf, np.int32(1), plane_buf,
           palette_buf, np.int32(255))
    queue.finish()
    out = download(plane_buf, plane.shape)
    region = out.reshape(size, stride)[8:16, 8:16].astype(int)
    want = np.array(tables["host.intra"], dtype=int).reshape(8, 8)
    check("intra matches the host kernel", bool((region == want).all()), True)
    check("intra DC is flat", int(region.max() - region.min()), 0)
    check("intra DC value", int(region[0, 0]), (100 * 8 + 200 * 8 + 8) // 16)

    # --------------------------------------------------------- intra, palette
    print("palette prediction")
    plane = np.zeros(stride * size, dtype=np.uint16)
    # a 4x4 block at (8, 8) with a 4x4 colour map and three colours: the kernel
    # has to read the map from the palette buffer and look each entry up
    colour_map = np.array([0, 1, 2, 1,
                           2, 0, 1, 2,
                           1, 2, 0, 1,
                           2, 1, 2, 0], dtype=np.uint8)
    pj = IntraJob()
    pj.dst_off = 8 * stride + 8
    pj.dst_stride = stride
    pj.x = 8
    pj.y = 8
    pj.w = 4
    pj.h = 4
    pj.bitdepth = 8
    pj.palette = 1
    pj.palette_size = 3
    pj.palette_color0 = 12
    pj.palette_color1 = 200
    pj.palette_color2 = 77
    pj.palette_off = 0
    pj.palette_stride = 4
    pjob_buf = upload(np.frombuffer(pj, dtype=np.uint8).copy())
    ppal_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR,
                         hostbuf=np.ascontiguousarray(colour_map))
    pplane_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=plane)
    launch(None, program.av1g_k_intra, 1, pjob_buf, np.int32(1), pplane_buf,
           ppal_buf, np.int32(255))
    queue.finish()
    out = download(pplane_buf, plane.shape)
    region = out.reshape(size, stride)[8:12, 8:12].astype(int)
    want = np.array(tables["host.palette"], dtype=int).reshape(4, 4)
    check("palette matches the host kernel", bool((region == want).all()), True)
    check("palette uses every colour", len(set(region.flatten().tolist())), 3)

    # ---------------------------------------------------------------- inter
    print("inter prediction")
    ref = np.zeros(stride * size, dtype=np.uint16)
    for y in range(size):
        for x in range(stride):
            ref[y * stride + x] = (x * 3 + y * 5) & 0xFF
    inter = InterJob()
    inter.w = 8
    inter.h = 8
    inter.dst_stride = stride
    inter.ref0_stride = stride
    inter.ref1_stride = stride
    inter.ref_x = 16
    inter.ref_y = 16
    inter.ref_w = size
    inter.ref_h = size
    inter.filt = 0
    inter.bitdepth = 8
    inter_buf = upload(np.frombuffer(inter, dtype=np.uint8).copy())
    ipref, itotal = prefix([64])
    ipref_buf = upload(ipref)
    ref_buf = upload(ref)
    dst_buf = cl.Buffer(context, mf.READ_WRITE, ref.nbytes)
    launch(None, program.av1g_k_inter, itotal, inter_buf, ipref_buf,
           np.int32(1), np.int32(itotal), ref_buf, ref_buf, dst_buf,
           filter_buf, np.int32(255))
    queue.finish()
    out = download(dst_buf, ref.shape)
    copied = out.reshape(size, stride)[:8, :8].astype(int)
    expected = ref.reshape(size, stride)[16:24, 16:24].astype(int)
    check("inter integer MV copies the patch",
          int(np.abs(copied - expected).max()), 0)

    # -------------------------------------------------------------- deblock
    print("loop filters")
    flat = np.full(stride * size, 90, dtype=np.uint16)
    db = DeblockJob()
    db.p_off = 8 * stride + 6  # p1; p0 = p1+step, q0 = p1+2*step
    db.stride = 1
    db.step = stride
    db.len = 8
    db.level = 32
    db.filter_size = 8
    db.bitdepth = 8
    dbuf = upload(np.frombuffer(db, dtype=np.uint8).copy())
    dpref, dtotal = prefix([db.len // 4])
    dpref_buf = upload(dpref)
    flat_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                         hostbuf=flat)
    launch(None, program.av1g_k_deblock, dtotal, dbuf, dpref_buf, np.int32(1),
           np.int32(dtotal), flat_buf, np.int32(255))
    queue.finish()
    out = download(flat_buf, flat.shape)
    check("deblock leaves flat areas alone", bool((out == flat).all()), True)

    # A real edge, long enough for several groups: the kernel is launched once
    # per group of four samples and the group helper wants a sample offset, so
    # an edge that is more than four samples long catches an index mix-up.
    step_of = 1
    edge = np.full(stride * size, 24, dtype=np.uint16)
    picture = edge.reshape(size, stride)
    for row in range(8, 16):
        picture[row, 6] = 40   # p1
        picture[row, 7] = 40   # p0
        picture[row, 8] = 48   # q0
        picture[row, 9] = 48   # q1
    db = DeblockJob()
    db.p_off = 8 * stride + 6
    db.stride = step_of          # steps across the edge
    db.step = stride             # runs along it
    db.len = 8                   # two groups, rows 8..15
    db.level = 32                # limit 9, blimit 27, thresh 1
    db.sharpness = 0
    db.filter_size = 4           # narrow filter only
    db.bitdepth = 8
    dbuf = upload(np.frombuffer(db, dtype=np.uint8).copy())
    dpref, dtotal = prefix([db.len // 4])
    dpref_buf = upload(dpref)
    edge_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=edge)
    launch(None, program.av1g_k_deblock, dtotal, dbuf, dpref_buf, np.int32(1),
           np.int32(dtotal), edge_buf, np.int32(255))
    queue.finish()
    out = download(edge_buf, edge.shape).reshape(size, stride)
    # level 32, sharpness 0 gives limit 9, blimit 27 and thresh 1, so the step
    # edge passes the mask but is not flat; the narrow filter then moves p1/p0
    # up and q0/q1 down by the same amount on *every* row of the edge, which
    # catches both a group/sample index mix-up and a wrongly shared delta.
    cols = out[8:16, 4:12].astype(int)
    check("deblock filters every row of a long edge",
          bool((cols == np.array([24, 24, 42, 43, 45, 46, 24, 24])).all()), True)

    # The plane edge may cut an edge short: the last group then filters fewer
    # than four samples and must not touch the samples past the edge.
    short = np.full(stride * size, 24, dtype=np.uint16)
    spic = short.reshape(size, stride)
    for row in range(8, 16):
        spic[row, 6] = 40
        spic[row, 7] = 40
        spic[row, 8] = 48
        spic[row, 9] = 48
    db = DeblockJob()
    db.p_off = 8 * stride + 6
    db.stride = step_of
    db.step = stride
    db.len = 6                   # one full group plus two samples
    db.level = 32
    db.sharpness = 0
    db.filter_size = 4
    db.bitdepth = 8
    dbuf = upload(np.frombuffer(db, dtype=np.uint8).copy())
    dpref, dtotal = prefix([(db.len + 3) // 4])
    dpref_buf = upload(dpref)
    short_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                          hostbuf=short)
    launch(None, program.av1g_k_deblock, dtotal, dbuf, dpref_buf, np.int32(1),
           np.int32(dtotal), short_buf, np.int32(255))
    queue.finish()
    out = download(short_buf, short.shape).reshape(size, stride)
    check("deblock stops at the end of a short edge",
          bool((out[8:14, 7] == 43).all() and (out[14:16, 7] == 40).all() and
               (out[8:14, 8] == 45).all() and (out[14:16, 8] == 48).all()), True)
    untouched = (out[:, :6] == 24).all() and (out[:, 10:] == 24).all()
    untouched = untouched and (out[:8, 6:10] == 24).all()
    untouched = untouched and (out[16:, 6:10] == 24).all()
    # rows 14 and 15 are past the end of the edge, so they keep the input values
    untouched = untouched and (out[14:16, 6:10] == np.array([40, 40, 48, 48])).all()
    # the filter reaches two samples either side of the edge, and only the rows
    # inside the (short) edge move
    untouched = untouched and (out[8:14, 6] == 42).all() and (out[8:14, 9] == 46).all()
    check("deblock leaves the rest of the picture alone", bool(untouched), True)

    cd = CdefJob()
    cd.x = 8
    cd.y = 8
    cd.src_off = 8 * stride + 8
    cd.src_stride = stride
    cd.dst_off = cd.src_off
    cd.dst_stride = stride
    cd.luma_off = cd.src_off
    cd.luma_stride = stride
    cd.pri_strength = 8
    cd.sec_strength = 0
    cd.damping = 3
    cd.dir = -1              # find the direction, as a decoder does
    cd.bitdepth = 8
    cd.plane_w = stride
    cd.plane_h = size
    cbuf = upload(np.frombuffer(cd, dtype=np.uint8).copy())
    cpref, ctotal = prefix([1])
    cpref_buf = upload(cpref)
    # a noisy block: a single step through the block is symmetric, so the taps on
    # either side of it cancel and CDEF leaves it exactly as it is (that is the
    # point of the constraint); noise does move samples
    edge = np.empty(stride * size, dtype=np.uint16)
    for y in range(size):
        for x in range(stride):
            edge[y * stride + x] = 128 + (((x * 5 + y * 11) * 7 + 3) % 17) - 8
    src_buf = upload(edge)
    dst_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=edge)
    launch(None, program.av1g_k_cdef, ctotal, cbuf, cpref_buf, np.int32(1),
           np.int32(ctotal), src_buf, dst_buf, np.int32(255))
    queue.finish()
    out = download(dst_buf, edge.shape)
    cdef_block = out.reshape(size, stride)[8:16, 8:16].astype(int)
    want = np.array(tables["host.cdef"], dtype=int).reshape(8, 8)
    check("cdef matches the host kernel", bool((cdef_block == want).all()), True)
    # and the block really was filtered, so the check above is not vacuous
    check("cdef moved samples next to the edge",
          int(np.abs(cdef_block - edge.reshape(size, stride)[8:16, 8:16]).max()) > 0,
          True)

    # ------------------------------------------------------ loop restoration
    # Two snapshots and one output picture: inside the stripe the kernel reads
    # the CDEF output, in the two lines above it the deblocked picture. The
    # reference is the host build of the same kernel, so any disagreement -
    # swapped snapshots included - shows up sample for sample.
    print("loop restoration")
    cdef_plane = np.zeros(stride * size, dtype=np.uint16)
    for y in range(size):
        for x in range(stride):
            cdef_plane[y * stride + x] = 90 + ((x * 5 + y * 3) % 23)
    curr_plane = cdef_plane.copy()
    curr_plane[:8 * stride] = 200

    def lr_job(bitdepth):
        job = LrJob()
        job.dst_off = 8 * stride + 8
        job.dst_stride = stride
        job.cdef_stride = stride
        job.curr_stride = stride
        job.x = 8
        job.y = 8
        job.w = 8
        job.h = 8
        job.plane_w = stride
        job.plane_h = size
        job.stripe_start_y = 8
        job.stripe_end_y = 23
        job.bitdepth = bitdepth
        return job

    def run_lr(job, cdef_src, curr_src, maxv):
        dst = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=cdef_src)
        jb = upload(np.frombuffer(job, dtype=np.uint8).copy())
        launch(None, program.av1g_k_lr, lr_total, jb, lr_pref_buf, np.int32(1),
               np.int32(lr_total), upload(cdef_src), upload(curr_src), dst,
               np.int32(maxv))
        queue.finish()
        return download(dst, cdef_src.shape)

    lr_pref, lr_total = prefix([64])
    lr_pref_buf = upload(lr_pref)

    wiener = lr_job(8)
    wiener.type = 1          # AV1G_LR_WIENER
    wiener.vfilter0, wiener.vfilter1, wiener.vfilter2 = 3, -2, 5
    wiener.hfilter0, wiener.hfilter1, wiener.hfilter2 = 1, 6, -3
    out = run_lr(wiener, cdef_plane, curr_plane, 255)
    region = out.reshape(size, stride)[8:16, 8:16].astype(int)
    want = np.array(tables["host.lr_wiener"], dtype=int).reshape(8, 8)
    check("wiener matches the host kernel", bool((region == want).all()), True)
    check("wiener changes the picture",
          int(np.abs(region - cdef_plane.reshape(size, stride)[8:16, 8:16]).max()) > 0,
          True)
    # The three rows at the top of the job are the only ones whose vertical taps
    # reach above the stripe, and those taps read the pre-CDEF picture: with the
    # snapshots made equal those rows are the only ones that change, which fails
    # if the two pointers are swapped or if one of them is ignored.
    flat_run = run_lr(wiener, cdef_plane, cdef_plane, 255)
    flat_region = flat_run.reshape(size, stride)[8:16, 8:16].astype(int)
    check("wiener reads the pre-CDEF picture above the stripe",
          bool((flat_region[:3] != region[:3]).any() and
               (flat_region[3:] == region[3:]).all()), True)

    sgr = lr_job(8)
    sgr.type = 2             # AV1G_LR_SGRPROJ
    sgr.sgr_radius0, sgr.sgr_eps0 = 2, 12
    sgr.sgr_radius1, sgr.sgr_eps1 = 0, 0   # the second pass then uses the input
    sgr.sgr_xqd0, sgr.sgr_xqd1 = 95, 33
    out = run_lr(sgr, cdef_plane, curr_plane, 255)
    region = out.reshape(size, stride)[8:16, 8:16].astype(int)
    want = np.array(tables["host.lr_sgr"], dtype=int).reshape(8, 8)
    check("self guided matches the host kernel", bool((region == want).all()), True)
    check("self guided changes the picture",
          int(np.abs(region - cdef_plane.reshape(size, stride)[8:16, 8:16]).max()) > 0,
          True)

    # 10 bit, where the box filter scales its sums back to 8 bit before they
    # become a variance: at 8 bit those shifts are zero and this check would
    # pass even if the device dropped them
    deep = lr_job(10)
    deep.type = 2
    deep.sgr_radius0, deep.sgr_eps0 = 2, 12
    deep.sgr_radius1, deep.sgr_eps1 = 0, 0
    deep.sgr_xqd0, deep.sgr_xqd1 = 95, 33
    out = run_lr(deep, (cdef_plane * 4).astype(np.uint16),
                 (curr_plane * 4).astype(np.uint16), 1023)
    region = out.reshape(size, stride)[8:16, 8:16].astype(int)
    want = np.array(tables["host.lr_sgr10"], dtype=int).reshape(8, 8)
    check("10 bit self guided matches the host kernel",
          bool((region == want).all()), True)

    # ------------------------------------------------------ superres upscale
    # The source and the target are different geometries, so every output
    # sample resamples its own eight taps: the reference is the host kernel.
    print("superres upscale")
    step_x, initial_subpel_x, up_rows, up_w = tables["upscale_params"]
    src_stride = 24
    src = np.zeros(src_stride * up_rows, dtype=np.uint16)
    for y in range(up_rows):
        for x in range(src_stride):
            src[y * src_stride + x] = 40 + ((x * 7 + y * 11) % 61)
    up = UpscaleJob()
    up.src_stride = src_stride
    up.dst_stride = up_w
    up.rows = up_rows
    up.upscaled_w = up_w
    up.max_x = src_stride - 1
    up.bitdepth = 8
    up.step_x = step_x
    up.initial_subpel_x = initial_subpel_x
    ubuf = upload(np.frombuffer(up, dtype=np.uint8).copy())
    upref, utotal = prefix([up_w * up_rows])
    upref_buf = upload(upref)
    src_buf = upload(src)
    dst_buf = cl.Buffer(context, mf.READ_WRITE, up_w * up_rows * 2)
    launch(None, program.av1g_k_upscale, utotal, ubuf, upref_buf, np.int32(1),
           np.int32(utotal), src_buf, dst_buf, filter_buf, np.int32(255))
    queue.finish()
    out = download(dst_buf, (up_w * up_rows,)).reshape(up_rows, up_w)
    region = out[:8, :8].astype(int)
    want = np.array(tables["host.upscale"], dtype=int).reshape(8, 8)
    check("upscale matches the host kernel", bool((region == want).all()), True)
    check("upscale moved samples",
          int(np.abs(region - src.reshape(up_rows, src_stride)[:8, :8]).max()) > 0,
          True)

    # ------------------------------------------------------------ film grain
    # The noise image is built on the host; the kernel scales one noise sample by
    # the plane's curve at that sample's value and adds it.
    print("film grain")
    noise = np.full(64 * 64, 40, dtype=np.int16)
    g = GrainJob()
    g.dst_off = 8 * stride + 8
    g.dst_stride = stride
    g.noise_off = 0
    g.noise_stride = 64
    g.x = 0
    g.y = 0
    g.w = 8
    g.h = 8
    g.plane = 0
    g.bitdepth = 8
    g.scaling_shift = 8
    g.num_y_points = 1
    g.min_value = 0
    g.max_value = 255
    g.luma_w = 8
    gbuf = upload(np.frombuffer(g, dtype=np.uint8).copy())
    gpref, gtotal = prefix([64])
    gpref_buf = upload(gpref)
    noise_buf = upload(noise)
    flat_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=flat)
    launch(None, program.av1g_k_grain, gtotal, gbuf, gpref_buf, np.int32(1),
           np.int32(gtotal), flat_buf, flat_buf, noise_buf, upload(np.zeros(768, dtype=np.int32)),
           np.int32(255))
    queue.finish()
    out = download(flat_buf, flat.shape)
    check("grain with a zero scaling table is a no-op", bool((out == flat).all()), True)

    # a curve of 256 is a gain of one: the noise (40) is added as it stands
    launch(None, program.av1g_k_grain, gtotal, gbuf, gpref_buf, np.int32(1),
           np.int32(gtotal), flat_buf, flat_buf, noise_buf,
           upload(np.full(768, 256, dtype=np.int32)), np.int32(255))
    queue.finish()
    out = download(flat_buf, flat.shape)
    got = out.reshape(size, stride)[8:16, 8:16].astype(int)
    want = flat.reshape(size, stride)[8:16, 8:16].astype(int) + 40
    check("grain adds the noise the curve scales it by", bool((got == want).all()), True)

    # ---------------------------------------------------------------- colour
    print("colour conversion")
    width = height = 8
    y_plane = np.full(width * height, 235, dtype=np.uint16)  # limited-range white
    u_plane = np.full(width * height, 128, dtype=np.uint16)
    v_plane = np.full(width * height, 128, dtype=np.uint16)
    colour = ColorJob()
    colour.w = width
    colour.h = height
    colour.dst_stride = width
    colour.y_stride = width
    colour.c_stride = width
    colour.bitdepth = 8
    m = [76293, 117504, 76293, -13926, -34937, 76293, 138412, 0]
    y_off, c_off = 16, 128  # BT.709 limited range
    half = 1 << 15
    for i, value in enumerate(m):
        setattr(colour, "m%d" % i, value)
    setattr(colour, "o0", half - m[0] * y_off - m[1] * c_off)
    setattr(colour, "o1", half - m[2] * y_off - m[3] * c_off - m[4] * c_off)
    setattr(colour, "o2", half - m[5] * y_off - m[6] * c_off)
    cbuf = upload(np.frombuffer(colour, dtype=np.uint8).copy())
    cpref, ctotal = prefix([width * height])
    cpref_buf = upload(cpref)
    yb = upload(y_plane)
    ub = upload(u_plane)
    vb = upload(v_plane)
    rgb_buf = cl.Buffer(context, mf.READ_WRITE, width * height * 3)
    launch(None, program.av1g_k_color, ctotal, cbuf, cpref_buf, np.int32(1),
           np.int32(ctotal), yb, ub, vb, rgb_buf)
    queue.finish()
    rgb = download(rgb_buf, width * height * 3, dtype=np.uint8)
    check("limited-range white is (near) 255", int(min(rgb[0], rgb[1], rgb[2])),
          ok=bool(min(rgb[0], rgb[1], rgb[2]) >= 253))

    print("")
    if failures:
        print("%d check(s) FAILED: %s" % (len(failures), ", ".join(failures)))
        return 1
    print("all OpenCL kernel checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
