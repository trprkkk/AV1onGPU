#!/usr/bin/env python3
"""AV1onGPU - verification on a machine that actually has a GPU.

Everything else in this tree is designed to be checkable without a device, and
that is deliberate: the kernels are the same C that the CPU backend runs, and
`tests/opencl_selftest.py` / `tests/opencl_pipeline.py` compile and replay them
through a software OpenCL implementation. What those layers cannot show is
whether a *real* driver accepts this backend at all: device queries, buffer
creation, the job layouts the kernels declare, launch shapes, and whether the CPU
stays idle while the GPU works.

This script is the one command to run on such a machine:

    python3 tests/gpu_verify.py                    # build/ next to this repo
    python3 tests/gpu_verify.py --fixtures ~/av1   # plus your own streams
    python3 tests/gpu_verify.py --backends opencl  # one backend only

It checks, for every GPU backend it can initialise:

  1. synthetic parity  - `av1gpu_bench --all`: the GPU result has to be bit
     identical to the CPU reference decode, and the CPU load and ms/frame are
     reported so the speed claims can be read off.
  2. real streams      - `av1gpu_decode --backend <gpu> --require-device` against
     the reference decoder's frame hashes (`tests/data/*.hash`) or a YUV
     reference (`<stream>_ref.yuv`), so a whole front end plus reconstruction runs
     on the device and every frame has to match bit for bit.

`--require-device` matters: without it a backend that fails to initialise falls
back to the CPU reference and the run would "pass" while proving nothing. This
script fails instead, and it also fails if the device it got is not the backend it
asked for.

Exit codes: 0 everything verified, 1 a mismatch or a backend failure, 77 no GPU
backend could be initialised (ctest reports that as a skip).
"""

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SKIP = 77

GPU_KINDS = ("opencl", "cuda")


def run(cmd, verbose=False):
    if verbose:
        print("  $ " + " ".join(cmd))
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except FileNotFoundError:
        return None, "", "%s: not found" % cmd[0]
    except subprocess.TimeoutExpired:
        return None, "", "timed out"
    return done.returncode, done.stdout, done.stderr


def find_tools(build):
    tools = {}
    for name in ("av1gpu_info", "av1gpu_bench", "av1gpu_decode"):
        for candidate in (os.path.join(build, "bin", name),
                          os.path.join(build, name),
                          os.path.join(ROOT, "build", "bin", name)):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                tools[name] = candidate
                break
    return tools


def enumerate_devices(info_path):
    """Returns [(kind, index, name)] from av1gpu_info's device list."""
    code, out, err = run([info_path])
    if code != 0:
        return [], (err or out).strip().splitlines()[-1:] or ["av1gpu_info failed"]
    devices = []
    for line in out.splitlines():
        match = re.match(r"\s*\[(\w+)(?:\s+(\d+))?\]\s+(.*)$", line)
        if match:
            kind = match.group(1)
            index = int(match.group(2) or 0)
            devices.append((kind, index, match.group(3).strip()))
    return devices, []


def bench(tools, frames, width, height, qstep, verbose):
    """Runs av1gpu_bench --all and returns ({kind: row}, parity, ok)."""
    cmd = [tools["av1gpu_bench"], "--all", "--frames", str(frames),
           "--width", str(width), "--height", str(height), "--qstep", str(qstep)]
    code, out, err = run(cmd, verbose)
    if code is None:
        return {}, None, False
    rows = {}
    parity = None
    for line in out.splitlines():
        if line.startswith("parity:"):
            parity = line.split(":", 1)[1].strip()
            continue
        match = re.match(r"^(\w+)\s{2,}(\S.*)$", line)
        if not match:
            continue
        kind = match.group(1)
        if kind not in ("cpu", "opencl", "cuda"):
            continue
        fields = match.group(2).split()
        if len(fields) < 7:
            # "opencl     device workgroup limit too small"
            rows[kind] = {"error": match.group(2).strip()}
            continue
        try:
            rows[kind] = {
                "device": " ".join(fields[:-5]),
                "ms": float(fields[-5]),
                "fps": float(fields[-4]),
                "cpu": float(fields[-3]),
                "psnr": float(fields[-2]),
                "launch": int(fields[-1]),
            }
        except ValueError:
            rows[kind] = {"error": match.group(2).strip()}
    return rows, parity, code == 0


def parse_decode(out):
    """Pulls the backend line, the match count and the summed frame times."""
    backend = None
    matched = total = None
    ms = 0.0
    frames = None
    differences = 0
    for line in out.splitlines():
        if re.match(r"^frame \d+: hash 0x", line):
            differences += 1
        if line.startswith("backend: "):
            backend = line[len("backend: "):].strip()
        match = re.match(r"^(\d+) of (\d+) decoded frame\(s\) (?:bit exact|match the "
                         r"recorded hashes)", line)
        if match:
            matched, total = int(match.group(1)), int(match.group(2))
        match = re.match(r"^(\d+) frame\(s\) decoded$", line)
        if match:
            frames = int(match.group(1))
        match = re.search(r"([\d.]+) ms", line)
        if line.startswith("frame ") and match:
            ms += float(match.group(1))
    return {"backend": backend, "matched": matched, "total": total, "ms": ms,
            "frames": frames, "differences": differences}


def collect_streams(fixtures_dirs):
    """[(name, ivf, reference, kind)] where reference is a .hash or a .yuv.

    Streams are keyed by file name and a directory given with --fixtures wins over
    tests/data, so a copy of a fixture pointed at a different reference replaces
    the repository's one instead of being listed twice.
    """
    found = {}

    def add(directory, name):
        path = os.path.join(directory, name)
        stem = path[:-4]
        if os.path.isfile(stem + ".hash"):
            found[name] = (path, stem + ".hash")
        elif os.path.isfile(stem + "_ref.yuv"):
            found[name] = (path, stem + "_ref.yuv")

    data = os.path.join(ROOT, "tests", "data")
    if os.path.isdir(data):
        for name in sorted(os.listdir(data)):
            if name.endswith(".ivf"):
                add(data, name)
    for directory in fixtures_dirs:
        if not os.path.isdir(directory):
            print("  warning: %s is not a directory, skipped" % directory)
            continue
        for name in sorted(os.listdir(directory)):
            if name.endswith(".ivf"):
                add(directory, name)
    streams = []
    for name in sorted(found):
        path, reference = found[name]
        kind = "hash" if reference.endswith(".hash") else "yuv"
        streams.append((name[:-4], path, reference, kind))
    return streams


def main():
    parser = argparse.ArgumentParser(description="verify the GPU backends on a real device")
    parser.add_argument("--build", default=os.path.join(ROOT, "build"),
                        help="build directory with bin/ (default: build/)")
    parser.add_argument("--backends", default="",
                        help="comma separated GPU backends to verify (default: every "
                             "GPU backend av1gpu_info reports)")
    parser.add_argument("--device", type=int, default=0, help="device index (default 0)")
    parser.add_argument("--fixtures", action="append", default=[],
                        help="extra directory of .ivf streams (with .hash or _ref.yuv)")
    parser.add_argument("--stream-frames", type=int, default=0,
                        help="temporal units per stream (default 0 = the whole "
                             "stream); a limit compares the frames it decodes")
    parser.add_argument("--bench-frames", type=int, default=20, help="bench frames (default 20)")
    parser.add_argument("--bench-size", default="1280x720", help="bench size (default 1280x720)")
    parser.add_argument("--qstep", type=int, default=8, help="bench quantiser (default 8)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    tools = find_tools(args.build)
    missing = [name for name in ("av1gpu_info", "av1gpu_bench", "av1gpu_decode")
               if name not in tools]
    if missing:
        print("gpu_verify: missing tools: %s" % ", ".join(missing))
        print("  build them with -DAV1GPU_BUILD_TOOLS=ON, or pass --build DIR")
        return 2

    print("AV1onGPU GPU verification")
    print("  build: %s" % os.path.dirname(os.path.dirname(tools["av1gpu_info"])))

    devices, errors = enumerate_devices(tools["av1gpu_info"])
    present = [kind for kind, _, _ in devices if kind in GPU_KINDS]
    # By default verify whatever this machine has, so a machine with only OpenCL
    # is not failed because of the CUDA it was never going to have. Naming a
    # backend explicitly makes it a requirement.
    requested = [kind.strip() for kind in args.backends.split(",") if kind.strip()]
    missing = [kind for kind in requested if kind not in present]
    wanted = requested if requested else list(present)
    if not wanted:
        wanted = [kind for kind in requested]
    usable = [d for d in devices if d[0] in wanted]
    print("  devices:")
    if not devices:
        print("    (none)")
    for kind, index, name in devices:
        label = kind if kind == "cpu" else "%s%d" % (kind, index)
        print("    [%s] %s" % (label, name))
    for line in errors:
        print("    ! %s" % line)
    if not usable:
        print("\ngpu_verify: no %s device could be initialised - skipped"
              % "/".join(wanted or GPU_KINDS))
        print("  a machine with a GPU and its OpenCL (or CUDA) driver is needed; "
              "use tests/opencl_selftest.py for the driverless kernel checks")
        return SKIP

    failures = []
    for kind in missing:
        failures.append("%s was requested but no such device is present" % kind)

    # ---------------------------------------------------------------- synthetic
    width, height = (int(value) for value in args.bench_size.lower().split("x"))
    print("\n1. synthetic clip: %dx%d, %d frames, qstep %d (av1gpu_bench --all)"
          % (width, height, args.bench_frames, args.qstep))
    rows, parity, parity_ok = bench(tools, args.bench_frames, width, height, args.qstep,
                                    args.verbose)
    print("   %-8s %-30s %9s %8s %7s %8s %7s  %s"
          % ("backend", "device", "ms/frame", "fps", "cpu %", "psnr Y", "launch", "status"))
    cpu = rows.get("cpu")
    for kind in ("cpu",) + tuple(wanted):
        row = rows.get(kind)
        if row is None:
            continue
        if "error" in row:
            print("   %-8s %s" % (kind, row["error"]))
            if kind != "cpu":
                failures.append("%s backend failed in the benchmark: %s" % (kind, row["error"]))
            continue
        status = "reference" if kind == "cpu" else (
            "bit identical" if parity_ok and parity == "all backends bit identical"
            else "MISMATCH")
        print("   %-8s %-30s %9.2f %8.1f %7.1f %8.2f %7d  %s"
              % (kind, row["device"][:30], row["ms"], row["fps"], row["cpu"], row["psnr"],
                 row["launch"], status))
        if kind != "cpu" and status != "bit identical":
            failures.append("%s benchmark output is not bit identical to the CPU reference"
                            % kind)
    if parity is None:
        failures.append("the benchmark printed no parity line")
    elif parity != "all backends bit identical" and len(rows) > 1:
        failures.append("benchmark parity: %s" % parity)
    for kind in wanted:
        row = rows.get(kind)
        if row and "error" not in row and cpu and "error" not in cpu and row["ms"] > 0:
            print("   %s: %.2fx the CPU backend's ms/frame, CPU load %.1f%% vs %.1f%%"
                  % (kind, cpu["ms"] / row["ms"], row["cpu"], cpu["cpu"]))

    # ------------------------------------------------------------------ streams
    streams = collect_streams(args.fixtures)
    if not streams:
        print("\n2. real streams: none found (tests/data has no .ivf, no --fixtures)")
    else:
        scope = ("every temporal unit" if args.stream_frames == 0
                 else "up to %d temporal units each" % args.stream_frames)
        print("\n2. real streams: %d stream(s), %s" % (len(streams), scope))
        print("   %-14s %-8s %-30s %10s %8s  %s"
              % ("stream", "backend", "device", "frames", "ms/frame", "status"))
        for name, ivf, reference, ref_kind in streams:
            for kind in ("cpu",) + tuple(wanted):
                cmd = [tools["av1gpu_decode"], "--backend", kind, "--device", str(args.device)]
                if args.stream_frames > 0:
                    cmd += ["--frames", str(args.stream_frames)]
                cmd.append(ivf)
                if kind != "cpu":
                    cmd.insert(1, "--require-device")
                if ref_kind == "hash":
                    cmd += ["--hash-check", reference]
                else:
                    cmd += ["--output", os.devnull, "--compare", reference]
                code, out, err = run(cmd, args.verbose)
                info = parse_decode(out)
                got = info["matched"]
                total = info["total"]
                frames = info["frames"]
                if total:
                    shown = "%d/%d" % (got, total)
                elif frames is not None:
                    shown = "%d" % frames
                else:
                    shown = "-"
                ms = (info["ms"] / total) if total else 0.0
                if args.stream_frames > 0:
                    # a limited run consumes only the head of the hash list, so
                    # the exit code is non zero by design; the pictures still
                    # have to match, which is what `differences` counts
                    ok = info["differences"] == 0 and got is not None and got > 0
                else:
                    ok = code == 0 and total is not None and got == total
                if not ok:
                    status = "FAILED"
                    detail = [line for line in out.splitlines()
                              if line.startswith("frame ") and "hash 0x" in line]
                    if not detail:
                        detail = (err or out).strip().splitlines()
                    if detail:
                        status += ": " + detail[0][:80]
                    failures.append("%s on %s: %s" % (name, kind, status))
                else:
                    status = "bit exact" if kind != "cpu" else "reference"
                print("   %-14s %-8s %-30s %10s %8.2f  %s"
                      % (name, kind, (info["backend"] or "-")[:30], shown, ms, status))

    # ------------------------------------------------------------------ summary
    print()
    if failures:
        print("gpu_verify: %d failure(s)" % len(failures))
        for line in failures:
            print("  - %s" % line)
        return 1
    print("gpu_verify: all checks passed on %s"
          % ", ".join("%s%d" % (kind, index) for kind, index, _ in usable))
    return 0


if __name__ == "__main__":
    sys.exit(main())
