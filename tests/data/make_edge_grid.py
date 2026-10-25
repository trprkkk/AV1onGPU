#!/usr/bin/env python3
"""Build the coded grid fixture: a frame whose planes end inside their grid.

Two roundings matter at the edge of a picture and they are not the same one:

  * blocks are coded on an eight sample grid - `MiCols = 2 * ((FrameWidth + 7) >>
    3)` (7.4.4) - so a 130x66 frame has 34 mi columns and 9 mi rows, i.e. a 136
    by 72 sample block grid;
  * the loop filter (7.14) walks every plane up to *its own* four sample grid:
    the plane rounded up to four samples, which for a 4:2:0 frame is 132x68 for
    luma and 68x36 for chroma. So the last two luma rows and the last three
    chroma rows are outside the picture but inside the grid the filter covers,
    and 7.15's CDEF then copies those rows - and 7.17.6's restoration borrows
    them across a stripe border - so the picture's bottom edge depends on them.

`odd_size.ivf` (130x100) covers the first rounding; its height is a multiple of
four in every plane, so it cannot tell a decoder that filters the plane from one
that filters the plane rounded up to four. This fixture can: 130x66 is 66 = 4 *
16 + 2 luma rows and 33 = 4 * 8 + 1 chroma rows, so both planes end inside their
four sample grid, and a decoder that stops at the visible edge leaves the samples
CDEF reads at level zero - a handful of samples in the last rows come out one
level off, which is exactly what this fixture was added for.

    python3 tests/data/make_edge_grid.py tests/data/edge_grid.ivf

Four frames at `-crf 40`, key frame first, so three of them are inter frames and
the bottom rows code a residual rather than a copy. The content is a smoothed
random texture with fresh noise on every frame plus hard edged panels pinned to
the bottom and right edges, which is what keeps the loop filter, CDEF and
restoration at a non-zero level right where the grids disagree - a smoother clip
codes the last rows at level zero and then the two roundings agree again. All
four frames carry restoration, and it is wiener, sgrproj and none across the
four, on luma and chroma both.

The encoder is the libaom build inside the ffmpeg binary that `imageio-ffmpeg`
installs (`--ffmpeg` selects another); encoding is deterministic for a given
build, so `edge_grid.ivf` comes out byte identical every time and the hashes in
`edge_grid.hash` can be recorded at all.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 130, 66, 4
CPU_USED = "4"
CRF = "40"


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(1)
    cw, ch = WIDTH // 2, HEIGHT // 2
    out = bytearray()
    for frame in range(frames):
        yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
        # a smoothed random texture with noise on it: hard enough that the
        # quantiser leaves a residual in the inter frames, and the smoothing
        # leaves gradients for the self guided filter
        tex = rng.integers(0, 256, (HEIGHT, WIDTH)).astype(np.float64)
        tex = (tex * 3 + np.roll(tex, 1, 1) * 2 + np.roll(tex, -1, 1)) / 7
        luma = tex + rng.normal(0, 8, (HEIGHT, WIDTH)) + 2 * frame
        # hard edges pinned to the bottom rows and the right columns, so the
        # last 4x4 groups of the frame are where the deblocking really happens
        luma[HEIGHT - 8:HEIGHT, :] = 50 + 40 * frame
        luma[HEIGHT - 2:HEIGHT, :] = 220 - 30 * frame
        luma[:, WIDTH - 8:WIDTH] = 40 + 30 * frame
        luma[:, WIDTH - 2:WIDTH] = 210 - 20 * frame
        out += np.clip(luma, 0, 255).astype(np.uint8).tobytes()

        yyc, xxc = np.mgrid[0:ch, 0:cw].astype(np.float64)
        u = 128 + 45 * np.sin((xxc + 2 * frame) / 4.0) + rng.normal(0, 6, (ch, cw))
        v = 128 + 45 * np.cos((yyc + 2 * frame) / 4.0) + rng.normal(0, 6, (ch, cw))
        # same idea on chroma, whose plane ends one row inside its grid
        u[ch - 4:ch, :] = 60 + 40 * frame
        v[ch - 4:ch, :] = 200 - 40 * frame
        out += np.clip(u, 0, 255).astype(np.uint8).tobytes()
        out += np.clip(v, 0, 255).astype(np.uint8).tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", "1x1",
        "-enable-restoration", "1", "-pix_fmt", "yuv420p", "-f", "ivf", ivf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s" % result.stderr[-2000:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--ffmpeg", default=None,
                        help="encoder binary, default the one imageio-ffmpeg ships")
    args = parser.parse_args()

    ffmpeg = args.ffmpeg
    if ffmpeg is None:
        try:
            import imageio_ffmpeg
        except ImportError:
            raise SystemExit("need imageio-ffmpeg, or pass --ffmpeg")
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    raw_path = args.output + ".yuv"
    with open(raw_path, "wb") as handle:
        handle.write(source_clip(args.frames))
    encode(ffmpeg, raw_path, args.output, args.frames)
    os.unlink(raw_path)
    print("%s: %d bytes" % (args.output, os.path.getsize(args.output)))


if __name__ == "__main__":
    main()
