#!/usr/bin/env python3
"""Build the bit depth and chroma fixtures: encoder clips of a generated ramp.

Most of the suite is 8 bit 4:2:0, and the deeper and wider paths are not the
same code: the deblocking thresholds are scaled by `1 << (BitDepth - 8)`, the
reconstruction has more headroom at 10 and 12 bit, the film grain range is
`1 << (BitDepth - 1)` either side of zero, and a 4:4:4 frame's chroma planes are
as large as its luma one (which is where `frame_to_yuv8` used to write past its
buffer). A fixture that only ever produces 8 bit 4:2:0 samples cannot tell any of
those apart from the 8 bit behaviour.

The clip is generated here rather than taken from anywhere, so the fixtures can
be rebuilt from nothing: a slow luma ramp with a little noise on it, and flat
chroma (noise on the chroma planes in 4:4:4, where they are big enough to carry
it). Twelve frames at 320x240.

    python3 tests/data/make_aom_clip.py out.ivf [--pix-fmt 420p|444p]
                                     [--bit-depth 8|10|12] [--frames N]

The 10 bit 4:2:0 stream is the one with `--require-unfiltered`: libaom codes its
first frame's *loop filter level as zero* while still enabling the per-reference
deltas, and `decode_frame_wrapup` only runs the loop filter when a luma level was
coded, so a frame that codes 0 and 0 is left unfiltered no matter what the deltas
say. A decoder that applies them anyway filters edges a reference decoder leaves
alone - a few thousand samples off by one or two, which then drifts through the
references. The other two fixtures do not need that property, so they do not ask
for it.

The encoder is the libaom build inside the ffmpeg binary that `imageio-ffmpeg`
installs (`imageio_ffmpeg.get_ffmpeg_exe()`); pass --ffmpeg if yours lives
somewhere else. Encoding is deterministic for a given libaom build, so the .ivf
comes out the same every time.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT = 320, 240
CRF = "40"
CPU_USED = "8"

# short name -> (bit depth, subsampling, the ffmpeg pixel format)
FORMATS = {
    "420p": (8, "420p", "yuv420p"),
    "420p10": (10, "420p", "yuv420p10le"),
    "420p12": (12, "420p", "yuv420p12le"),
    "444p": (8, "444p", "yuv444p"),
    "444p10": (10, "444p", "yuv444p10le"),
}


def source_clip(bit_depth, subsampling, frames):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    dtype = "<u2" if bit_depth > 8 else np.uint8
    peak = (1 << bit_depth) - 1
    rng = np.random.default_rng(11)
    out = bytearray()
    for frame in range(frames):
        planes = []
        luma = np.zeros((HEIGHT, WIDTH), np.int32)
        for row in range(HEIGHT):
            # a ramp that moves 18 levels per frame across and 10 down, so every
            # frame is a little brighter than the one before it
            across = (2 * (np.arange(WIDTH) - WIDTH // 2) * 18) // WIDTH
            down = (2 * (row - HEIGHT // 2) * 10) // HEIGHT
            luma[row] = 120 + across + down + frame * 3
        luma += rng.integers(-4, 5, size=(HEIGHT, WIDTH))
        planes.append(luma)
        if subsampling == "420p":
            width, height = WIDTH // 2, HEIGHT // 2
        else:
            width, height = WIDTH, HEIGHT
        for _ in range(2):
            # flat chroma keeps the 4:2:0 streams simple enough that libaom
            # codes no loop filter level at all
            spread = 2 if subsampling == "420p" else 8
            planes.append(128 + rng.integers(-spread, spread + 1, size=(height, width)))
        for plane in planes:
            out += np.clip(plane, 0, peak).astype(dtype).tobytes()
    return bytes(out)


def encode(ffmpeg, pix_fmt, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", "1x1",
        "-pix_fmt", pix_fmt, "-f", "ivf", ivf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s" % result.stderr[-2000:])


def header_positions(probe, path):
    """The probe's AV1GPU_HDRPOS lines, as (seq fields, [frame fields])."""
    env = dict(os.environ, AV1GPU_HDRPOS="1")
    result = subprocess.run([probe, path], capture_output=True, text=True, env=env)
    seq, frames = None, []
    for line in result.stderr.splitlines():
        parts = line.split()
        if not parts or parts[0] != "HDRPOS":
            continue
        fields = {}
        for item in parts[2:]:
            key, _, value = item.partition("=")
            fields[key] = int(value)
        if parts[1] == "seq":
            seq = fields
        else:
            frames.append(fields)
    if seq is None:
        raise SystemExit("the probe printed no sequence header positions")
    return seq, frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--pix-fmt", default="420p10", choices=sorted(FORMATS),
                        help="default 420p10, the 10 bit fixture")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--require-unfiltered", action="store_true",
                        help="refuse a stream whose frames all code a luma level")
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--probe", default=os.path.join("build", "bin", "av1gpu_probe"))
    parser.add_argument("--keep-raw", default="", help="keep the raw clip at this path")
    args = parser.parse_args()

    if not args.ffmpeg:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError:
            raise SystemExit("pass --ffmpeg, or install imageio-ffmpeg")
        args.ffmpeg = get_ffmpeg_exe()

    bit_depth, subsampling, ffmpeg_format = FORMATS[args.pix_fmt]
    raw_path = args.keep_raw or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), ".aom_clip_source.yuv")
    with open(raw_path, "wb") as handle:
        handle.write(source_clip(bit_depth, subsampling, args.frames))
    try:
        encode(args.ffmpeg, ffmpeg_format, raw_path, args.output, args.frames)
    finally:
        if not args.keep_raw:
            os.unlink(raw_path)

    seq, frames = header_positions(args.probe, args.output)
    if seq.get("bitdepth") != bit_depth:
        raise SystemExit("%s: the stream is not %d bit: %r" % (args.output, bit_depth, seq))
    if len(frames) < args.frames:
        raise SystemExit("%s: only %d frame headers, expected %d" %
                         (args.output, len(frames), args.frames))
    quiet = [index for index, frame in enumerate(frames)
             if frame.get("lf_y0", 0) == 0 and frame.get("lf_y1", 0) == 0]
    if args.require_unfiltered and not any(frames[index].get("deltas", 0) for index in quiet):
        raise SystemExit("%s: no frame codes a zero luma level with the reference "
                         "deltas enabled, so the stream does not exercise the rule "
                         "this fixture exists for" % args.output)
    print("%s: %s, %d bit, %d frames, %d bytes, %s" %
          (args.output, subsampling, bit_depth, len(frames),
           os.path.getsize(args.output),
           ", ".join("frame %d unfiltered" % index for index in quiet) or
           "every frame filtered"))


if __name__ == "__main__":
    sys.exit(main())
