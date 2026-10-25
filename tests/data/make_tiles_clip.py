#!/usr/bin/env python3
"""Build the multi-tile fixture: a frame cut into four independently coded tiles.

Everything up to this fixture was one tile per frame, so the parts of the
specification that only exist because a frame may be cut up were never
exercised:

  * a tile starts with its own range decoder, its own copy of the frame's CDF
    tables and cleared above/left contexts (5.11.2), and the tables the frame
    hands on are the ones of the single tile ``context_update_tile_id`` names
    (5.11.1's frame_end_update_cdf);
  * ``is_inside`` bounds every neighbour question - AvailU/AvailL, the
    above-right and below-left BlockDecoded samples, the sub-8x8 chroma
    variants (AvailUChroma/AvailLChroma), the motion vector candidate scan and
    the local warp's sample scan - to the tile, not to the frame. So a block on
    a tile edge predicts from fewer neighbours than the same block inside a
    tile, even though the frame's reconstructed samples are right there.

The clip is 256x160 cut into 2x2 tiles: the grid divides into 128 + 128
samples across and 128 + 32 down, i.e. the tile row boundary is not at the
frame's middle, the bottom tiles are only 32 samples high, and both boundaries
land in the middle of the content. Four frames, the
first a key frame, so three of them code inter prediction (motion vectors,
OBMC and warped motion) across the tile edges. The content is a smoothed random
texture that the frames pan and zoom over - global motion, which is what makes
libaom code local warped motion - with fresh noise on every frame, hard edged
panels on the right and bottom edges, a dark band across the tile row boundary
and gradients on chroma, which keeps the loop filter, CDEF and restoration at a
non-zero level.

The two sub-8x8 chroma rules are what makes the fixture worth its 64 KB: with
4x4 luma blocks (which this content codes in the dark noisy areas) the chroma
of a block one 4x4 wide is shared with its neighbour, so AvailUChroma and
AvailLChroma reach two mi units out - and on a tile edge that is a sample the
tile does not own.

    python3 tests/data/make_tiles_clip.py tests/data/tiles.ivf

The encoder is the libaom build inside the ffmpeg binary `imageio-ffmpeg`
installs (`--ffmpeg` selects another). `-tiles 2x2` is ffmpeg's AV1 tile
option; libaom needs row-mt on to honour more than one tile row. Encoding is
deterministic for a given build, so `tiles.ivf` comes out byte identical every
time and the hashes in `tiles.hash` can be recorded at all.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 256, 160, 4
TILES = "2x2"
CPU_USED = "3"
CRF = "28"


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(4)
    cw, ch = WIDTH // 2, HEIGHT // 2
    # One texture plane, larger than the picture, which the frames pan and zoom
    # over: global motion is what makes the encoder code local warped motion,
    # whose sample scan is one of the places ``is_inside`` has to mean the tile.
    bw, bh = WIDTH * 2, HEIGHT * 2
    big = rng.integers(0, 256, (bh, bw)).astype(np.float64)
    big = (big * 3 + np.roll(big, 1, 1) * 2 + np.roll(big, -1, 1)) / 7
    yy, xx = np.mgrid[0:ch * 2, 0:cw * 2].astype(np.float64)
    ubig = 128 + 50 * np.sin(xx / 6.0)
    vbig = 128 + 50 * np.cos(yy / 6.0)

    out = bytearray()
    for frame in range(frames):
        shift = 2 * frame
        zoom = 1.0 + 0.02 * frame
        ys = np.clip(((np.arange(HEIGHT) - HEIGHT / 2) / zoom + HEIGHT / 2 + shift).astype(int),
                     0, bh - 1)
        xs = np.clip(((np.arange(WIDTH) - WIDTH / 2) / zoom + WIDTH / 2 + shift).astype(int),
                     0, bw - 1)
        luma = big[np.ix_(ys, xs)] + rng.normal(0, 4, (HEIGHT, WIDTH))
        # hard edges, so the loop filter, CDEF and restoration all have
        # something to do on the tile boundaries as well as on the frame edges
        luma[HEIGHT - 8:HEIGHT, :] = 50 + 40 * frame
        luma[HEIGHT - 2:HEIGHT, :] = 220 - 30 * frame
        luma[:, WIDTH - 8:WIDTH] = 40 + 30 * frame
        luma[:, WIDTH - 2:WIDTH] = 210 - 20 * frame
        # a dark band across the tile row boundary, where the encoder spends
        # 4x4 blocks and therefore codes chroma the sub-8x8 way
        luma[124:132, :] = 24 + 6 * frame
        out += np.clip(luma, 0, 255).astype(np.uint8).tobytes()

        yc = np.clip(((np.arange(ch) - ch / 2) / zoom + ch / 2 + shift / 2).astype(int),
                     0, ch * 2 - 1)
        xc = np.clip(((np.arange(cw) - cw / 2) / zoom + cw / 2 + shift / 2).astype(int),
                     0, cw * 2 - 1)
        u = ubig[np.ix_(yc, xc)] + rng.normal(0, 5, (ch, cw))
        v = vbig[np.ix_(yc, xc)] + rng.normal(0, 5, (ch, cw))
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
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", TILES,
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
