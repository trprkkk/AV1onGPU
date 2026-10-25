#!/usr/bin/env python3
"""Mutation smoke test: a stream the decoder cannot trust must not crash it.

The AV1 parser is the only part of this library that reads bytes it did not
produce, so it is the part where a corrupt stream must be an error and not an
out of bounds write. This test takes the fixtures, applies a handful of single
byte mutations at deterministic offsets, decodes each of them with the decoder
under test, and requires that

  * the process comes back with an exit status rather than a signal,
  * it does not print a sanitizer report, and
  * it does not run past a timeout.

Each mutation is decoded twice: once for the pixels, and once for the decode
plan the backend builds (`--dump-plans`, one frame), because that second path
reads the parsed block list through the block geometry, the job bounds and the
wave schedule.

With the sanitizer build (`-DAV1GPU_SANITIZE=ON`) the second condition is what
catches memory errors; in a normal build this is a crash test. Neither is a
conformance check: the mutated streams are not AV1, so whatever they decode to
(or fail to decode) is fine.

    python3 tests/fuzz_decode.py --decoder build/bin/av1gpu_decode

Exit status is 0 when every mutation was handled, 1 otherwise, and each failure
prints the fixture, the byte and the value to patch so it can be reproduced with
one run of the decoder.
"""

import argparse
import os
import random
import shutil
import subprocess
import sys
import tempfile

# The one mutation behind this test: byte 166 of palette_sc.ivf (the key frame's
# tile bytes) makes a block merge two eight colour neighbour palettes, which is
# the widest a palette cache can get. The cache used to be eight bytes on the
# stack; ASan catches the write past it in frame 1.
KNOWN = [("palette_sc.ivf", 166, 0x7C)]

BAD_OUTPUT = ("AddressSanitizer", "LeakSanitizer", "runtime error:",
              "UndefinedBehaviorSanitizer", "SEGV")


def decode(decoder, path, frames, env, extra=()):
    command = [decoder, "--backend", "cpu", "--frames", str(frames),
               "-o", os.devnull, path] + list(extra)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=env, timeout=120)
    except subprocess.TimeoutExpired:
        return "timed out after 120 s"
    stderr = result.stderr.decode("utf-8", "replace")
    if result.returncode < 0:
        return "killed by signal %d" % -result.returncode
    for pattern in BAD_OUTPUT:
        if pattern in stderr:
            lines = [line for line in stderr.splitlines()
                     if pattern in line or "SUMMARY" in line]
            return "sanitizer: %s" % (lines[0] if lines else pattern)
    if result.returncode not in (0, 1):
        return "exit status %d" % result.returncode
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", required=True, help="av1gpu_decode to test")
    parser.add_argument("--fixtures", default=None,
                        help="directory of .ivf fixtures (default: next to this script)")
    parser.add_argument("--per-file", type=int, default=2,
                        help="mutations per fixture, on top of the known one (default 2)")
    parser.add_argument("--frames", type=int, default=3,
                        help="temporal units to decode per mutation (default 3)")
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()

    fixtures = args.fixtures or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    if not os.path.isdir(fixtures):
        print("fuzz_decode: no fixture directory at %s" % fixtures)
        return 77
    streams = sorted(name for name in os.listdir(fixtures) if name.endswith(".ivf"))
    if not streams:
        print("fuzz_decode: no fixtures in %s" % fixtures)
        return 77
    if not os.path.exists(args.decoder):
        print("fuzz_decode: no decoder at %s" % args.decoder)
        return 77

    env = dict(os.environ)
    env.setdefault("ASAN_OPTIONS", "detect_leaks=0")
    env.setdefault("UBSAN_OPTIONS", "print_stacktrace=1")

    # every case: (fixture, offset, value, description)
    cases = []
    for name, offset, value in KNOWN:
        if name in streams:
            cases.append((name, offset, value, "known palette cache merge"))
    rng = random.Random(args.seed)
    for name in streams:
        path = os.path.join(fixtures, name)
        size = os.path.getsize(path)
        with open(path, "rb") as handle:
            data = handle.read()
        for _ in range(args.per_file):
            # the IVF header is 32 bytes and each frame record 12, so offsets past
            # 44 land in OBU bytes
            offset = rng.randrange(44, max(45, size))
            value = data[offset] ^ (1 << rng.randrange(8))
            if value == data[offset]:
                value ^= 0x01
            cases.append((name, offset, value, "bit flip"))

    failures = []
    for index, (name, offset, value, why) in enumerate(cases):
        path = os.path.join(fixtures, name)
        with open(path, "rb") as handle:
            data = bytearray(handle.read())
        if offset >= len(data):
            continue
        old = data[offset]
        data[offset] = value
        with tempfile.NamedTemporaryFile(suffix=".ivf", delete=False) as handle:
            handle.write(data)
            mutated = handle.name
        plan_dir = tempfile.mkdtemp(prefix="av1gpu-plans-")
        try:
            problem = decode(args.decoder, mutated, args.frames, env)
            if problem is None:
                # the plan the backend builds covers the picture a second time:
                # block geometry, job bounds, the wave schedule and the writer
                # all read the parsed block list, so a corrupt block list has to
                # come out of that path too - with one frame, to stay quick.
                problem = decode(args.decoder, mutated, 1, env,
                                 ["--dump-plans", plan_dir])
                if problem:
                    problem = "dump-plans: " + problem
        finally:
            os.unlink(mutated)
            shutil.rmtree(plan_dir, ignore_errors=True)
        if problem:
            failures.append((name, offset, old, value, why, problem))
            print("FAIL %s byte %d: 0x%02X -> 0x%02X (%s): %s"
                  % (name, offset, old, value, why, problem))
        if (index + 1) % 10 == 0:
            print("%d/%d mutations, %d failure(s)" % (index + 1, len(cases), len(failures)),
                  flush=True)

    print("fuzz_decode: %d mutation(s) over %d fixture(s), %d failure(s)"
          % (len(cases), len(streams), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
