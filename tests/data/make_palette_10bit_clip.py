#!/usr/bin/env python3
"""Build the deep colour palette fixture: a 10 bit stream that codes palettes.

Palette colours are samples, so the syntax reads them with ``BitDepth`` bits
(5.11.5 ``palette_colors_y`` / ``palette_colors_u`` / ``palette_colors_v``) and
the delta width follows the sample range. Every palette fixture before this one
was 8 bit, where a colour fits in a byte, which is exactly the width an
implementation is tempted to store it in: the range decoder stays in step
(eight bits are read from the top of the same range coder either way), the
palette sizes and the colour maps come out right, and only the samples are
wrong. `palette_sc` and the 10 bit fixtures each looked fine on their own; the
two together did not.

The bug this pins was a ``uint8_t`` colour array: a 10 bit entry of 880 arrived
as 112 (``880 & 0xff``) and the picture came back looking like a prediction
failure - the whole luma plane off by tens of levels while the chroma was exact,
because chroma happened to use colours below 256. So the clip codes palettes
whose colours are above 255 in every plane, at a quality high enough that the
palette is what the picture is made of.

The content is deliberately flat and few coloured - panels of one colour on a
flat background, with hard edges - because that is what makes libaom choose
palette coding instead of a transform block. ``-cpu-used 5 -crf 32 -g 2`` keeps
it small; two frames, the second inter, so a frame that predicts from the first
also runs through the palette cache.

    python3 tests/data/make_palette_10bit_clip.py tests/data/palette_10bit.ivf

The encoder is the libaom build inside the ffmpeg binary `imageio-ffmpeg`
installs (`--ffmpeg` selects another). Encoding is deterministic for a given
build, so `palette_10bit.ivf` comes out byte identical every time and the hashes
in `palette_10bit.hash` can be recorded at all.
"""

import argparse
import os
import subprocess

WIDTH, HEIGHT, FRAMES = 64, 64, 2
CPU_USED = "5"
CRF = "32"
BIT_DEPTH = 10


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    scale = 1 << (BIT_DEPTH - 8)
    peak = (1 << BIT_DEPTH) - 1
    cw, ch = WIDTH // 2, HEIGHT // 2

    out = bytearray()
    for frame in range(frames):
        luma = np.full((HEIGHT, WIDTH), 28.0)
        for k in range(6):
            x0 = 4 + k * (WIDTH // 8) + 2 * frame
            y0 = 2 + (k % 3) * (HEIGHT // 4)
            luma[y0:y0 + HEIGHT // 4, x0:x0 + WIDTH // 6] = (60 + 90 * k) % 256
        # rows and columns of a second colour: a palette block two samples wide
        # still has to reproduce both, and the two hard edges next to each other
        # are what the encoder cannot code with a smooth predictor
        luma[0::17, :] = 220 - 20 * frame
        luma[:, 0::23] = 30 + 40 * frame
        out += np.clip(luma * scale, 0, peak).astype(np.uint16).tobytes()

        yy, xx = np.mgrid[0:ch, 0:cw]
        for phase in (0.0, 0.7):
            chroma = np.full((ch, cw), 90.0 + 20 * frame)
            chroma[3::5, :] = 200 - 10 * frame
            chroma[:, 2::7] = 140 + 15 * frame
            chroma = chroma + 8 * np.sin(xx / 5.0 + phase)
            out += np.clip(chroma * scale, 0, peak).astype(np.uint16).tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(frames),
        "-pix_fmt", "yuv420p10le", "-f", "ivf", ivf_path,
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
