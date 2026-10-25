#!/usr/bin/env python3
"""Build the odd sized fixture: a frame that is *not* a multiple of eight.

Every other fixture in this directory is a multiple of eight in both
directions, which is the size where three different roundings agree - and they
do not have to:

  * `MiCols = 2 * ((FrameWidth + 7) >> 3)` (7.4.4) rounds the frame up to eight
    samples, so a 130 wide frame codes 34 mi columns (136 samples). Rounding to
    four instead gives 33, and the block loop of the last superblock column then
    reads the wrong number of blocks: the whole frame decodes as garbage.
  * the loop filters and the intra prediction read samples *outside* the
    picture but inside that grid (7.15 copies whole 8x8 blocks, 7.17.6 borrows
    the two lines across a stripe border, and a block at the last row is
    predicted from the left column below it), so the prediction dependencies
    and the filter extents are the grid ones, not the visible ones.
  * the picture itself is 130x100, i.e. a partial stripe at the bottom
    (100 = 64 + 36) and two samples past 128 in every row.

    python3 tests/data/make_odd_size.py tests/data/odd_size.ivf
    python3 tests/data/make_odd_size.py /tmp/odd.ivf --frames 8

Six frames of 130x100 at `-crf 32`, key frame first, so five of them are inter
frames and the last row band of every frame is predicted from the row band above
it. The content is a panning ramp with hard edged bars and a little noise, which
keeps deblocking and CDEF busy at the frame edge instead of leaving them at
level zero. The encoder is the libaom build inside the ffmpeg binary that
`imageio-ffmpeg` installs (`--ffmpeg` selects another); encoding is
deterministic for a given build, so `odd_size.ivf` comes out byte identical
every time and the hashes in `odd_size.hash` can be recorded.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 130, 100, 6
CPU_USED = "5"
CRF = "32"


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    cw, ch = WIDTH // 2, HEIGHT // 2
    yyc, xxc = np.mgrid[0:ch, 0:cw].astype(np.float64)
    u = np.clip(128 + 20 * np.sin(xxc / 5.0), 0, 255).astype(np.uint8)
    v = np.clip(128 + 20 * np.cos(yyc / 6.0), 0, 255).astype(np.uint8)
    out = bytearray()
    for frame in range(frames):
        # the ramp slides a frame's worth of phase and every frame carries its
        # own noise, so an inter frame codes both motion and a residual instead
        # of a copy of the frame before it
        luma = 110 + 40 * np.sin(xx / 9.0 + frame) + rng.normal(0, 6, (HEIGHT, WIDTH))
        out += np.clip(luma, 0, 255).astype(np.uint8).tobytes()
        out += u.tobytes() + v.tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", "1x1",
        "-enable-cdef", "1", "-enable-restoration", "1",
        "-pix_fmt", "yuv420p", "-f", "ivf", ivf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s" % result.stderr[-2000:])


def check_geometry(probe, path):
    """The probe's header lines, which have to show the size and the filters."""
    result = subprocess.run([probe, path], capture_output=True, text=True)
    out = result.stdout
    line = next((l.strip() for l in out.splitlines() if "deblock" in l), "")
    if "%dx%d" % (WIDTH, HEIGHT) not in out:
        raise SystemExit("%s: the probe did not report %dx%d" % (path, WIDTH, HEIGHT))
    if not line:
        raise SystemExit("%s: the probe reported no loop filter line" % path)
    return line


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--frames", type=int, default=FRAMES)
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

    raw_path = args.keep_raw or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), ".odd_size_source.yuv")
    with open(raw_path, "wb") as handle:
        handle.write(source_clip(args.frames))
    try:
        encode(args.ffmpeg, raw_path, args.output, args.frames)
    finally:
        if not args.keep_raw:
            os.unlink(raw_path)

    line = check_geometry(args.probe, args.output)
    print("%s: yuv420p, %dx%d, %d frames, %d bytes" %
          (args.output, WIDTH, HEIGHT, args.frames, os.path.getsize(args.output)))
    print("  %s" % line)


if __name__ == "__main__":
    sys.exit(main())
