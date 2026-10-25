#!/usr/bin/env python3
"""AV1onGPU - end to end validation of the OpenCL decode path.

``opencl_selftest.py`` runs one job of each kernel in isolation.  This test
validates a *whole frame*: it asks ``av1gpu_dump`` for the real decode plans the
engine produces (job arrays, prefix tables, wave order, coefficients, loop
filter setup) and replays them through the actual OpenCL kernels with pyopencl,
in the same order and with the same element counts the OpenCL backend uses.  The
frame that comes back has to be bit identical to the CPU reference decode the
same tool wrote next to the plan.

That is as close as a machine without an ICD loader can get to running the
OpenCL backend: the kernels, the memory layout of the jobs and the scheduling
are the production ones, and the comparison is exact.

    pip install pyopencl pocl-binary-distribution
    python3 tests/opencl_pipeline.py
    python3 tests/opencl_pipeline.py --frames 6 --qstep 20 --no-grain

Exit codes: 0 all frames match, 1 mismatch, 77 OpenCL or the dumper is missing.
"""

import argparse
import ctypes
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import opencl_selftest as selftest  # noqa: E402  (probe, kernel source, helpers)

MAGIC = b"AV1GPLAN"
FORMAT_VERSION = 1
REFERENCE_SLOTS = 6      # kReferenceSlots in the backends
CDEF_DIM = 8             # AV1G_CDEF_DIM


# --------------------------------------------------------------------------
# plan file
# --------------------------------------------------------------------------

class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def raw(self, size):
        chunk = self.data[self.pos:self.pos + size]
        if len(chunk) != size:
            raise ValueError("truncated plan file")
        self.pos += size
        return chunk

    def i32(self):
        return struct.unpack("<i", self.raw(4))[0]

    def i64(self):
        return struct.unpack("<q", self.raw(8))[0]

    def ints(self, count):
        return list(struct.unpack("<%di" % count, self.raw(4 * count)))

    def jobs(self, cls, count):
        """Reads `count` POD job structs and returns them as ctypes objects."""
        size = ctypes.sizeof(cls)
        blob = self.raw(size * count)
        out = []
        for i in range(count):
            out.append(cls.from_buffer_copy(blob[i * size:(i + 1) * size]))
        return out


def read_plan(path, layouts):
    with open(path, "rb") as handle:
        r = Reader(handle.read())
    if r.raw(8) != MAGIC:
        raise ValueError("%s is not a plan file" % path)
    version = r.i32()
    if version != FORMAT_VERSION:
        raise ValueError("unsupported plan version %d" % version)

    plan = {"version": version}
    (plan["width"], plan["height"], plan["bit_depth"], plan["ss_x"],
     plan["ss_y"]) = (r.i32() for _ in range(5))
    plan["maxv"] = r.i32()
    plan["blocks"] = r.i32()
    plan["plane_w"] = r.ints(3)
    plan["plane_h"] = r.ints(3)
    plan["stride"] = r.ints(3)
    plan["plane_off"] = r.ints(3)
    plan["buffer_size"] = r.i64()
    plan["ref_slot"] = r.ints(2)
    plan["grain_stride"] = r.i32()
    plan["grain_h"] = r.i32()
    plan["grain_lut"] = r.ints(64)
    plan["coef_count"] = r.i32()
    plan["residual_count"] = r.i32()
    plan["grain_count"] = r.i32()

    plan["waves"] = []
    for _ in range(r.i32()):
        wave = {}
        for key in ("intra", "inter", "itx", "residual"):
            wave[key] = r.jobs(layouts["Av1g" + {"intra": "IntraJob", "inter": "InterJob",
                                                "itx": "ItxJob", "residual": "ItxJob"}[key]],
                               r.i32())
        plan["waves"].append(wave)
    for key, name in (("deblock_v", "DeblockJob"), ("deblock_h", "DeblockJob"),
                      ("cdef", "CdefJob"), ("grain", "GrainJob")):
        plan[key] = r.jobs(layouts["Av1g" + name], r.i32())

    plan["coefficients"] = r.ints(plan["coef_count"])
    plan["residual"] = r.ints(plan["residual_count"])
    plan["grain_template"] = list(struct.unpack(
        "<%db" % plan["grain_count"], r.raw(plan["grain_count"])))
    return plan


# --------------------------------------------------------------------------
# replay
# --------------------------------------------------------------------------

def pack(jobs):
    """Serialises a job list the way the backends upload it."""
    return b"".join(bytes(memoryview(job)) for job in jobs)


class Replay:
    """One OpenCL context replaying plans the way the backend submits them."""

    def __init__(self, cl, np, context, queue, program):
        self.cl, self.np = cl, np
        self.context, self.queue, self.program = context, queue, program

    def buffer(self, data, readonly=True):
        np, cl, mf = self.np, self.cl, self.cl.mem_flags
        array = np.ascontiguousarray(data)
        flags = (mf.READ_ONLY if readonly else mf.READ_WRITE) | mf.COPY_HOST_PTR
        return cl.Buffer(self.context, flags, hostbuf=array)

    def jobs(self, jobs, counts):
        """Uploads a job array plus its prefix table, like upload_jobs()."""
        np = self.np
        job_buf = self.buffer(np.frombuffer(pack(jobs), dtype=np.uint8))
        table = [0] * (len(counts) + 2)
        total = 0
        for i, count in enumerate(counts):
            table[i] = total
            total += count
        table[len(counts)] = table[len(counts) + 1] = total
        return job_buf, self.buffer(np.array(table, dtype=np.int32)), total

    def prefix(self, counts):
        np = self.np
        table = [0] * (len(counts) + 2)
        total = 0
        for i, count in enumerate(counts):
            table[i] = total
            total += count
        table[len(counts)] = table[len(counts) + 1] = total
        return self.buffer(np.array(table, dtype=np.int32)), total

    def launch(self, name, elements, args):
        if elements <= 0:
            return
        kernel = self.cl.Kernel(self.program, name)
        kernel.set_args(*args)
        local = 64
        global_size = ((elements + local - 1) // local) * local
        self.cl.enqueue_nd_range_kernel(self.queue, kernel, (global_size,), (local,))

    def decode(self, plan, filter_buf, ref_pool):
        """Decodes one frame; returns a device buffer holding the result."""
        cl, np, mf = self.cl, self.np, self.cl.mem_flags
        maxv = plan["maxv"]
        size = plan["buffer_size"]

        coef = np.array(plan["coefficients"] + [0] * 64, dtype=np.int32)
        residual = np.array(plan["residual"] + [0] * 64, dtype=np.int32)
        coef_buf = self.buffer(coef)
        residual_buf = self.buffer(residual)
        # tmp shares the coefficient offsets (itx.tmp_off == itx.coef_off)
        tmp_buf = self.buffer(np.zeros(max(len(coef), len(residual)), dtype=np.int32),
                              readonly=False)
        frame = self.buffer(np.zeros(size, dtype=np.uint16), readonly=False)
        palette_buf = self.buffer(np.zeros(8, dtype=np.uint8))

        for wave in plan["waves"]:
            # ---- inter prediction (reads the reference pool) --------------
            if wave["inter"]:
                jobs = wave["inter"]
                for job in jobs:
                    # the backend folds the reference slot into the offsets so
                    # every job can share one flat reference buffer
                    if job.ref0_slot >= 0:
                        job.ref0_off += job.ref0_slot * size
                    if job.ref1_slot >= 0:
                        job.ref1_off += job.ref1_slot * size
                job_buf, pref, total = self.jobs(jobs, [j.w * j.h for j in jobs])
                self.launch("av1g_k_inter", total,
                            (job_buf, pref, np.int32(len(jobs)), np.int32(total),
                             ref_pool, ref_pool, frame, filter_buf, np.int32(maxv)))

            # ---- intra prediction: one work item per block -----------------
            if wave["intra"]:
                jobs = wave["intra"]
                job_buf, pref, total = self.jobs(jobs, [1] * len(jobs))
                # Palette blocks index a per frame colour map buffer. The dump
                # does not carry that map yet, so a palette block would read the
                # wrong entries; the frames replayed here code none. The kernel
                # side is covered by opencl_selftest.py, and the whole path by
                # decoding tests/data/palette_sc.ivf with --backend opencl on a
                # machine that has an ICD loader.
                self.launch("av1g_k_intra", len(jobs),
                            (job_buf, np.int32(len(jobs)), frame, palette_buf,
                             np.int32(maxv)))

            # ---- inverse transform: rows then columns ----------------------
            if wave["itx"]:
                jobs = wave["itx"]
                job_buf, row_pref, rows = self.jobs(jobs, [j.h for j in jobs])
                col_pref, cols = self.prefix([j.w for j in jobs])
                self.launch("av1g_k_itx_rows", rows,
                            (job_buf, row_pref, np.int32(len(jobs)), np.int32(rows),
                             coef_buf, tmp_buf))
                self.launch("av1g_k_itx_cols", cols,
                            (job_buf, col_pref, np.int32(len(jobs)), np.int32(cols),
                             tmp_buf, frame, np.int32(maxv)))

            # ---- raw residual ----------------------------------------------
            if wave["residual"]:
                jobs = wave["residual"]
                job_buf, pref, total = self.jobs(jobs, [j.w * j.h for j in jobs])
                self.launch("av1g_k_add_residual", total,
                            (job_buf, pref, np.int32(len(jobs)), np.int32(total),
                             residual_buf, frame, np.int32(maxv)))

        # ---- loop filter: vertical edges, then horizontal ones -------------
        for key in ("deblock_v", "deblock_h"):
            jobs = plan[key]
            if not jobs:
                continue
            job_buf, pref, total = self.jobs(jobs, [j.len // 4 for j in jobs])
            self.launch("av1g_k_deblock", total,
                        (job_buf, pref, np.int32(len(jobs)), np.int32(total), frame,
                         np.int32(maxv)))

        # ---- CDEF works from a snapshot and writes the frame ---------------
        if plan["cdef"]:
            jobs = plan["cdef"]
            # one work item per block: the direction, the strength it earns and
            # the copy for a block with no strength are all per block
            job_buf, pref, total = self.jobs(jobs, [1] * len(jobs))
            snapshot = self.buffer(np.zeros(size, dtype=np.uint16), readonly=False)
            cl.enqueue_copy(self.queue, snapshot, frame)
            self.launch("av1g_k_cdef", total,
                        (job_buf, pref, np.int32(len(jobs)), np.int32(total), snapshot,
                         frame, np.int32(maxv)))

        # ---- film grain ----------------------------------------------------
        if plan["grain"]:
            jobs = plan["grain"]
            job_buf, pref, total = self.jobs(jobs, [j.w * j.h for j in jobs])
            grain_buf = self.buffer(np.array(plan["grain_template"], dtype=np.int8))
            lut_buf = self.buffer(np.array(plan["grain_lut"], dtype=np.int32))
            self.launch("av1g_k_grain", total,
                        (job_buf, pref, np.int32(len(jobs)), np.int32(total), frame,
                         grain_buf, lut_buf, np.int32(maxv)))

        return frame

    def read(self, frame_buf, size):
        np = self.np
        out = np.zeros(size, dtype=np.uint16)
        self.cl.enqueue_copy(self.queue, out, frame_buf).wait()
        return out


def compare(plan, decoded, expected, index):
    """Compares the decoded buffer with the flat reference planes."""
    import numpy as np  # noqa: PLC0415  (only reachable when pyopencl imported)

    diff_total = 0
    pos = 0
    worst = 0
    parts = []
    for p in range(3):
        w, h, stride, off = (plan["plane_w"][p], plan["plane_h"][p],
                             plan["stride"][p], plan["plane_off"][p])
        got = decoded[off:off + stride * h].reshape(h, stride)[:, :w].reshape(-1)
        want = expected[pos:pos + w * h]
        pos += w * h
        delta = np.abs(got.astype(np.int32) - want.astype(np.int32))
        bad = int((delta != 0).sum())
        if bad:
            parts.append("plane %d: %d/%d samples, max %d" %
                         (p, bad, delta.size, int(delta.max())))
        diff_total += bad
        worst = max(worst, int(delta.max()) if bad else 0)
    total = int(sum(plan["plane_w"][p] * plan["plane_h"][p] for p in range(3)))
    if diff_total == 0:
        print("frame %d: %d samples, exact" % (index, total))
        return []
    print("frame %d: %d samples, %d differ (max %d): %s" %
          (index, total, diff_total, worst, "; ".join(parts)))
    return ["frame %d" % index]


def read_manifest(path):
    """Parses manifest.txt: frame -> (reference slots, slot to store it in)."""
    manifest = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            # frame <n> refs <slot0> <slot1> out <slot>
            fields = line.split()
            if not fields:
                continue
            manifest[int(fields[1])] = (int(fields[3]), int(fields[4]), int(fields[6]))
    return manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=int, default=4,
                        help="frames to validate (the plans themselves decide the "
                             "wave counts, so more frames means more inter refs)")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--qstep", type=int, default=6)
    parser.add_argument("--no-cdef", action="store_true", help="decode without CDEF")
    parser.add_argument("--no-grain", action="store_true", help="decode without grain")
    parser.add_argument("--dumper", default=None, help="path to av1gpu_dump")
    return parser.parse_args(argv)


def main():
    try:
        import numpy as np
        import pyopencl as cl
    except ImportError as exc:
        print("OpenCL pipeline test skipped: %s" % exc)
        return 77

    found = selftest.devices()
    if not found:
        print("OpenCL pipeline test skipped: no OpenCL platform")
        return 77

    args = parse_args()
    dumper = args.dumper or os.path.join(ROOT, "build", "bin", "av1gpu_dump")
    if not os.path.exists(dumper):
        print("OpenCL pipeline test skipped: %s is not built" % dumper)
        return 77

    device = found[0]
    print("device: %s (%s)" % (device.name, device.version))
    context = cl.Context([device])
    queue = cl.CommandQueue(context)
    program = cl.Program(context, selftest.build_source()).build(["-cl-std=CL2.0"])

    layouts, tables = selftest.run_probe()
    replay = Replay(cl, np, context, queue, program)
    mf = cl.mem_flags

    filter_buf = replay.buffer(np.array(tables["filters"], dtype=np.int32))

    frames = args.frames
    work = tempfile.mkdtemp(prefix="av1gpu-plans-")
    try:
        command = [dumper, "--width", str(args.width), "--height", str(args.height),
                   "--frames", str(frames), "--qstep", str(args.qstep),
                   "--output", work]
        if not args.no_cdef:
            command.append("--cdef")
        if not args.no_grain:
            command.append("--grain")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print("av1gpu_dump failed:\n%s" % result.stderr)
            return 1
        for line in result.stdout.strip().splitlines():
            print("  %s" % line)

        plans = [read_plan(os.path.join(work, "frame_%d.plan" % f), layouts)
                 for f in range(frames)]
        manifest = read_manifest(os.path.join(work, "manifest.txt"))
        # one flat pool, `REFERENCE_SLOTS` frames deep, exactly like the backend
        size = plans[0]["buffer_size"]
        if any(plan["buffer_size"] != size for plan in plans):
            print("plans disagree on the frame buffer size")
            return 1
        ref_pool = cl.Buffer(context, mf.READ_WRITE, size * REFERENCE_SLOTS * 2)
        cl.enqueue_fill_buffer(queue, ref_pool, np.uint16(0), 0,
                               size * REFERENCE_SLOTS * 2)

        failures = []
        filled = set()
        for index, plan in enumerate(plans):
            refs, _, out_slot = manifest[index]
            # the backend keeps the reference pool on the device, so a plan can
            # only be replayed once the frames it points at are there
            missing = [slot for slot in (refs,) if slot >= 0 and slot not in filled]
            if missing:
                failures.append("frame %d: reference slot %s not decoded yet" %
                                (index, missing))
                break

            decoded_buf = replay.decode(plan, filter_buf, ref_pool)
            decoded = replay.read(decoded_buf, size)
            expected = np.fromfile(os.path.join(work, "frame_%d.yuv16" % index),
                                   dtype=np.uint16)
            failures += compare(plan, decoded, expected, index)

            # this frame becomes a reference for the frames that follow it, in
            # the pool slot the encoder assigned to it
            cl.enqueue_copy(queue, ref_pool, decoded_buf,
                            dst_offset=out_slot * size * 2, byte_count=size * 2).wait()
            filled.add(out_slot)

        print("")
        if failures:
            print("%d frame(s) FAILED: %s" % (len(failures), ", ".join(failures)))
            return 1
        print("all %d frames match the CPU reference decode bit for bit" % frames)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
