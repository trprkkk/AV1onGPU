#!/usr/bin/env python3
"""AV1onGPU - OpenCL kernel self test.

Compiles the real kernel sources (``kernels/av1gpu_kernels.h`` +
``kernels/av1_opencl.cl``) with an actual OpenCL C compiler and runs a handful of
jobs, checking the results against values computed here in Python.

The C++ OpenCL backend needs an ICD loader, which some containers and CI images
do not have - a pure software implementation does work though:

    pip install pyopencl pocl-binary-distribution
    python3 tests/opencl_selftest.py

The job-struct layouts and the shared tables are not duplicated here: a small
probe, compiled from ``src/core/tables.cpp`` with the project's own compiler,
prints them, so the test always exercises the numbers the backends upload.

Exit codes: 0 all checks passed, 1 failure, 77 OpenCL not available (skip).
"""

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
#include <vector>
#include "core/backend.hpp"
#include "av1gpu_kernels.h"

static void emit(const std::vector<int>& v, const char* name) {
  std::printf("{\"table\":\"%s\",\"data\":[", name);
  for (size_t i = 0; i < v.size(); ++i)
    std::printf("%s%d", i ? "," : "", v[i]);
  std::printf("]}\n");
}
#define F(T, f) std::printf("{\"struct\":\"%s\",\"field\":\"%s\",\"offset\":%zu}\n", #T, #f, offsetof(T, f))
#define FA(T, f, n) \
  for (int i = 0; i < (n); ++i) \
    std::printf("{\"struct\":\"%s\",\"field\":\"%s%d\",\"offset\":%zu}\n", #T, #f, i, \
                offsetof(T, f) + (size_t)i * sizeof(((T*)0)->f[0]))
#define S(T) std::printf("{\"struct\":\"%s\",\"size\":%zu}\n", #T, sizeof(T))

int main() {
  using namespace av1gpu::internal;
  for (int n = 4; n <= 64; n *= 2)
    std::printf("{\"basis_dct\":%d,\"offset\":%d}\n", n, tables::basis_dct(n));
  for (int n = 4; n <= 16; n *= 2)
    std::printf("{\"basis_adst\":%d,\"offset\":%d}\n", n, tables::basis_adst(n));
  for (int n = 4; n <= 64; n *= 2)
    std::printf("{\"basis_idtx\":%d,\"offset\":%d}\n", n, tables::basis_identity(n));
  emit(tables::basis(), "basis");
  emit(tables::filters(), "filters");

  S(Av1gItxJob);
  F(Av1gItxJob, coef_off); F(Av1gItxJob, tmp_off); F(Av1gItxJob, dst_off);
  F(Av1gItxJob, dst_stride); F(Av1gItxJob, w); F(Av1gItxJob, h);
  F(Av1gItxJob, row_basis); F(Av1gItxJob, col_basis); F(Av1gItxJob, eob);
  F(Av1gItxJob, shift);
  S(Av1gIntraAux);
  F(Av1gIntraAux, dc); F(Av1gIntraAux, ac_dc); F(Av1gIntraAux, avg);
  S(Av1gIntraJob);
  F(Av1gIntraJob, dst_off); F(Av1gIntraJob, dst_stride); F(Av1gIntraJob, w);
  F(Av1gIntraJob, h); F(Av1gIntraJob, mode); F(Av1gIntraJob, bitdepth);
  F(Av1gIntraJob, have_left); F(Av1gIntraJob, have_above); F(Av1gIntraJob, cfl_alpha);
  F(Av1gIntraJob, luma_off); F(Av1gIntraJob, luma_stride); F(Av1gIntraJob, ss_x);
  F(Av1gIntraJob, ss_y);
  S(Av1gInterJob);
  F(Av1gInterJob, dst_off); F(Av1gInterJob, dst_stride); F(Av1gInterJob, w);
  F(Av1gInterJob, h); F(Av1gInterJob, ref0_off); F(Av1gInterJob, ref0_stride);
  F(Av1gInterJob, ref1_off); F(Av1gInterJob, ref1_stride); F(Av1gInterJob, ref0_slot);
  F(Av1gInterJob, ref1_slot); F(Av1gInterJob, ref_x); F(Av1gInterJob, ref_y);
  F(Av1gInterJob, ref_w); F(Av1gInterJob, ref_h); F(Av1gInterJob, border);
  F(Av1gInterJob, mv0_x); F(Av1gInterJob, mv0_y); F(Av1gInterJob, mv1_x);
  F(Av1gInterJob, mv1_y); F(Av1gInterJob, filt); F(Av1gInterJob, compound);
  F(Av1gInterJob, w0); F(Av1gInterJob, w1); F(Av1gInterJob, bitdepth);
  S(Av1gDeblockJob);
  F(Av1gDeblockJob, p_off); F(Av1gDeblockJob, stride); F(Av1gDeblockJob, step);
  F(Av1gDeblockJob, len); F(Av1gDeblockJob, level); F(Av1gDeblockJob, sharpness);
  F(Av1gDeblockJob, filter_len); F(Av1gDeblockJob, is_chroma);
  F(Av1gDeblockJob, bitdepth);
  S(Av1gCdefJob);
  F(Av1gCdefJob, src_off); F(Av1gCdefJob, src_stride); F(Av1gCdefJob, dst_off);
  F(Av1gCdefJob, dst_stride); F(Av1gCdefJob, pri_strength); F(Av1gCdefJob, sec_strength);
  F(Av1gCdefJob, damping); F(Av1gCdefJob, dir); F(Av1gCdefJob, bitdepth);
  F(Av1gCdefJob, plane_w); F(Av1gCdefJob, plane_h);
  S(Av1gGrainJob);
  F(Av1gGrainJob, dst_off); F(Av1gGrainJob, dst_stride); F(Av1gGrainJob, x);
  F(Av1gGrainJob, y); F(Av1gGrainJob, w); F(Av1gGrainJob, h);
  F(Av1gGrainJob, bitdepth); F(Av1gGrainJob, grain_off); F(Av1gGrainJob, grain_stride);
  F(Av1gGrainJob, grain_h); F(Av1gGrainJob, lut_off); F(Av1gGrainJob, overlap);
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
    offsets = {}
    sizes = {}
    fields = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if "table" in entry:
            tables[entry["table"]] = entry["data"]
        elif "basis_dct" in entry:
            offsets[("dct", entry["basis_dct"])] = entry["offset"]
        elif "basis_adst" in entry:
            offsets[("adst", entry["basis_adst"])] = entry["offset"]
        elif "basis_idtx" in entry:
            offsets[("idtx", entry["basis_idtx"])] = entry["offset"]
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
    return layouts, offsets, tables


def build_source():
    with open(os.path.join(KERNELS, "av1gpu_kernels.h"), "r", encoding="utf-8") as handle:
        header = handle.read()
    with open(os.path.join(KERNELS, "av1_opencl.cl"), "r", encoding="utf-8") as handle:
        body = handle.read()
    return header + "\n" + body.replace('#include "av1gpu_kernels.h"', "")


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


def main():
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
    devices_ = found
    device = devices_[0]
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

    layouts, basis_offsets, tables = run_probe()
    ItxJob = layouts["Av1gItxJob"]
    IntraJob = layouts["Av1gIntraJob"]
    IntraAux = layouts["Av1gIntraAux"]
    InterJob = layouts["Av1gInterJob"]
    DeblockJob = layouts["Av1gDeblockJob"]
    CdefJob = layouts["Av1gCdefJob"]
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

    basis_buf = upload(np.array(tables["basis"], dtype=np.int32))
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
    job.row_basis = basis_offsets[("dct", n)]
    job.col_basis = basis_offsets[("dct", n)]
    job.eob = n * n
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
           np.int32(rows), coef_buf, tmp_buf, basis_buf)
    launch(None, program.av1g_k_itx_cols, cols, job_buf, col_pref_buf, njobs,
           np.int32(cols), tmp_buf, plane_buf, basis_buf, np.int32(255))
    queue.finish()
    out = download(plane_buf, plane.shape)
    block = out.reshape(size, stride)[8:16, 8:16]
    check("itx DC only is flat", int(block.max() - block.min()), 0)
    # the 2D DC basis of an orthonormal DCT is 1/n
    check("itx DC value", int(block[0, 0]), coef[0] // n)

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
    ij.w = 8
    ij.h = 8
    ij.mode = 0  # AV1G_PRED_DC
    ij.bitdepth = 8
    ij.have_left = 1
    ij.have_above = 1
    ij.luma_stride = stride
    ij.ss_x = 1
    ij.ss_y = 1
    ijob_buf = upload(np.frombuffer(ij, dtype=np.uint8).copy())
    aux_buf = cl.Buffer(context, mf.READ_WRITE, ctypes.sizeof(IntraAux) + 64)
    plane_buf = upload(plane)
    program.av1g_k_intra_aux(queue, (64,), (64,), ijob_buf, np.int32(1),
                             plane_buf, aux_buf)
    ipref, itotal = prefix([64])
    ipref_buf = upload(ipref)
    launch(None, program.av1g_k_intra, itotal, ijob_buf, ipref_buf,
           np.int32(1), np.int32(itotal), plane_buf, aux_buf, np.int32(255))
    queue.finish()
    out = download(plane_buf, plane.shape)
    region = out.reshape(size, stride)[8:16, 8:16]
    check("intra DC is flat", int(region.max() - region.min()), 0)
    check("intra DC value", int(region[0, 0]), (100 * 8 + 200 * 8 + 8) // 16)

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
    db.filter_len = 8
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
    db.filter_len = 4            # narrow filter only
    db.bitdepth = 8
    dbuf = upload(np.frombuffer(db, dtype=np.uint8).copy())
    dpref, dtotal = prefix([db.len // 4])
    dpref_buf = upload(dpref)
    edge_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=edge)
    launch(None, program.av1g_k_deblock, dtotal, dbuf, dpref_buf, np.int32(1),
           np.int32(dtotal), edge_buf, np.int32(255))
    queue.finish()
    out = download(edge_buf, edge.shape).reshape(size, stride)
    # f = -8 for this step edge (same formula as the host helper), so p0 and q0
    # move one closed-form step towards each other on *every* row of the edge.
    touched = out[8:16, 7].astype(int).tolist() + out[8:16, 8].astype(int).tolist()
    check("deblock filters every row of a long edge",
          bool(len(set(out[8:16, 7].astype(int).tolist())) == 1 and
               len(set(out[8:16, 8].astype(int).tolist())) == 1 and
               out[8, 7] == 48 and out[8, 8] == 40), True)
    untouched = (out[:, :6] == 24).all() and (out[:, 10:] == 24).all()
    untouched = untouched and (out[:8, 6:10] == 24).all() and (out[16:, 6:10] == 24).all()
    # only p0/q0 move with the narrow filter, and only on the rows of the edge
    untouched = untouched and (out[8:16, 6] == 40).all() and (out[8:16, 9] == 48).all()
    check("deblock leaves the rest of the picture alone", bool(untouched), True)

    cd = CdefJob()
    cd.src_off = 8 * stride + 8
    cd.src_stride = stride
    cd.dst_off = 8 * stride + 8
    cd.dst_stride = stride
    cd.pri_strength = 4
    cd.sec_strength = 2
    cd.dir = 1
    cd.bitdepth = 8
    cd.plane_w = size
    cd.plane_h = size
    cbuf = upload(np.frombuffer(cd, dtype=np.uint8).copy())
    cpref, ctotal = prefix([64])
    cpref_buf = upload(cpref)
    src_buf = upload(flat)
    # cdef only writes the 8x8 block, so start the destination from the same
    # flat picture the source has
    dst_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                        hostbuf=flat)
    launch(None, program.av1g_k_cdef, ctotal, cbuf, cpref_buf, np.int32(1),
           np.int32(ctotal), src_buf, dst_buf, np.int32(255))
    queue.finish()
    out = download(dst_buf, flat.shape)
    cdef_block = out.reshape(size, stride)[8:16, 8:16]
    check("cdef leaves flat areas alone", bool((cdef_block == 90).all()), True)

    grain = np.full(64 * 64, 40, dtype=np.int8)
    g = GrainJob()
    g.dst_off = 8 * stride + 8
    g.dst_stride = stride
    g.w = 8
    g.h = 8
    g.bitdepth = 8
    g.grain_stride = 64
    g.grain_h = 64
    gbuf = upload(np.frombuffer(g, dtype=np.uint8).copy())
    gpref, gtotal = prefix([64])
    gpref_buf = upload(gpref)
    grain_buf = upload(grain)
    lut_buf = upload(np.zeros(64, dtype=np.int32))
    flat_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR,
                         hostbuf=flat)
    launch(None, program.av1g_k_grain, gtotal, gbuf, gpref_buf, np.int32(1),
           np.int32(gtotal), flat_buf, grain_buf, lut_buf, np.int32(255))
    queue.finish()
    out = download(flat_buf, flat.shape)
    check("grain with a zero gain LUT is a no-op", bool((out == flat).all()), True)

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
