#!/usr/bin/env python3
"""Build the second multi-tile fixture: a 4x2 grid on a frame that is not a
multiple of the superblock size.

`tiles.ivf` is four 2x2 tiles on a 256x160 frame, i.e. equal widths and a tile
row boundary at exactly half the picture. It exercises the per tile state and
the two `is_inside` questions, but not the *geometry*: the tile grid of 5.9.1
distributes the superblock columns over the tile columns with `tile_log2`, which
for a width that is not a multiple of the tile width leaves the last column
wider, and `MiRowStarts`/`MiColStarts` clamp the last tile to the frame's mi
grid - which is rounded up to eight samples, not to the frame size.

This clip is 320x188 cut into 4x2 tiles. 320 is five superblock columns and
libaom gives the four tile columns 64, 64, 64 and 128 luma samples; 188 is two
superblock rows and 24 mi rows (192 luma samples), so the two tile rows are 64
and 124 samples high and the last tile row ends at the *coded* extent, two
samples past the picture. The content pans and zooms over a smoothed random
texture with fresh noise on every frame - global motion, so libaom codes local
warped motion - with hard edged panels on the right and bottom edges and a dark
band just above the tile row boundary, where the encoder spends 4x4 blocks (the
size whose chroma is coded the sub-8x8 way).

    python3 tests/data/make_tile_grid_clip.py tests/data/tiles_grid.ivf

The encoder is the libaom build inside the ffmpeg binary `imageio-ffmpeg`
installs (`--ffmpeg` selects another). `-tiles 4x2` is ffmpeg's AV1 tile
option; libaom needs row-mt on to honour more than one tile row, which ffmpeg
turns on with the option. Encoding is deterministic for a given build, so
`tiles_grid.ivf` comes out byte identical every time and `tiles_grid.hash` can
be recorded at all.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 320, 188, 4
TILES = "4x2"
CPU_USED = "4"
CRF = "38"


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(11)
    cw, ch = WIDTH // 2, HEIGHT // 2
    # One texture plane, larger than the picture, which the frames pan over:
    # global motion is what makes the encoder code local warped motion.
    bw, bh = WIDTH * 2, HEIGHT * 2
    big = rng.integers(0, 256, (bh, bw)).astype(np.float64)
    big = (big * 3 + np.roll(big, 1, 1) * 2 + np.roll(big, -1, 1)) / 7

    out = bytearray()
    for frame in range(frames):
        shift = 3 * frame
        ys = np.clip(((np.arange(HEIGHT) - HEIGHT / 2) + HEIGHT / 2 + shift).astype(int),
                     0, bh - 1)
        xs = np.clip(((np.arange(WIDTH) - WIDTH / 2) + WIDTH / 2 + shift).astype(int),
                     0, bw - 1)
        luma = big[np.ix_(ys, xs)] + rng.normal(0, 5, (HEIGHT, WIDTH))
        # hard edges on the right and the bottom, so the loop filter, CDEF and
        # restoration all have something to do there as well as at the frame edge
        luma[HEIGHT - 10:HEIGHT, :] = 40 + 30 * frame
        luma[:, WIDTH - 10:WIDTH] = 200 - 20 * frame
        # a dark band just above the tile row boundary at y = 128, where the
        # encoder spends 4x4 blocks and therefore codes chroma the sub-8x8 way
        luma[132:140, :] = 20 + 5 * frame
        out += np.clip(luma, 0, 255).astype(np.uint8).tobytes()

        yy, xx = np.mgrid[0:ch, 0:cw]
        u = 128 + 40 * np.sin(xx / 5.0 + 0.3 * frame) + rng.normal(0, 5, (ch, cw))
        v = 128 + 40 * np.cos(yy / 5.0 + 0.2 * frame) + rng.normal(0, 5, (ch, cw))
        out += np.clip(u, 0, 255).astype(np.uint8).tobytes()
        out += np.clip(v, 0, 255).astype(np.uint8).tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", TILES,
        "-pix_fmt", "yuv420p", "-f", "ivf", ivf_path,
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
