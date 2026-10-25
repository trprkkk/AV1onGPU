#!/usr/bin/env python3
"""Build the lossless fixture: a clip the encoder codes at `base_q_idx = 0`.

A frame that codes no quantiser at all is the one case where the whole in-loop
filter chain is switched off *by the bitstream*: 5.9.2 leaves `CodedLossless`
set, `AllLossless` follows from it (the picture is not superres'ed), and the
frame header then codes neither `loop_filter_params` levels nor
`cdef_params` strengths nor any restoration unit - `uncompressed_header` skips
all three. A decoder that reads those fields anyway desynchronises the header
immediately, and one that then runs deblock, CDEF or restoration on the picture
changes it, which is exactly what a lossless stream cannot afford.

    python3 tests/data/make_lossless_clip.py tests/data/lossless.ivf
    python3 tests/data/make_lossless_clip.py /tmp/lossless.ivf --keep-raw /tmp/ll.yuv

The content is the one residual coding is for: flat panels next to a hard
diagonal edge, a checkerboard finer than one sample per pixel in places, and a
little noise so that "lossless" also has to reproduce *coeffients* rather than
copy samples - and all of it pans three samples a frame, so three of the four
frames are inter frames. The encoder is the libaom build inside the ffmpeg
binary `imageio-ffmpeg` installs (`--ffmpeg` selects another) and encoding is
deterministic for a given build: the fixture comes out byte identical every
time, which is why the hashes in `lossless.hash` can be recorded at all.

`-enable-intrabc 0` is not cosmetic. With intra block copy on, the frame header
would set `allow_intrabc`, and that flag *also* skips the three filter parameter
sets (5.9.2) - the same header shape by a different route. Keeping it off is
what makes this fixture the one that pins down the lossless path itself; the
`intrabc` fixture covers the block copy path.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 160, 120, 4
CPU_USED = "6"


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(11)
    # the canvas is wider than a frame and every frame takes the next three
    # columns of it, so the picture pans
    canvas_w = WIDTH + 3 * frames
    yy, xx = np.mgrid[0:HEIGHT, 0:canvas_w].astype(np.float64)
    luma = np.full((HEIGHT, canvas_w), 118.0)
    # flat panels on a hard diagonal edge
    luma[(xx + yy) % 97 < 48] = 32.0
    luma[(xx - 2 * yy) % 71 < 30] = 214.0
    # a checkerboard that is one sample per pixel in places, so the residual
    # coding has to switch to the smallest transform
    luma[((xx // 1 + yy // 1) % 2 == 0) & (xx >= 96) & (yy >= 64)] = 240.0
    luma += rng.normal(0, 3.0, (HEIGHT, canvas_w))
    cw, ch = WIDTH // 2, HEIGHT // 2
    yyc, xxc = np.mgrid[0:ch, 0:cw + 3 * frames // 2].astype(np.float64)
    u = 128 + 46 * np.sin(xxc / 7.0) * np.cos(yyc / 9.0) + rng.normal(0, 3.0, yyc.shape)
    v = 128 + 38 * np.cos(xxc / 5.0) + rng.normal(0, 3.0, yyc.shape)

    out = bytearray()
    for frame in range(frames):
        # ... and every frame carries its own noise, so an inter frame cannot
        # get away with a pure block copy: it has to code a residual
        plane = np.clip(luma[:, 3 * frame:3 * frame + WIDTH] +
                        rng.normal(0, 2.0, (HEIGHT, WIDTH)), 0, 255)
        out += plane.astype(np.uint8).tobytes()
        for chroma in (u, v):
            out += np.clip(chroma[:, frame:frame + cw], 0, 255).astype(np.uint8).tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", "0", "-lossless", "1",
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", "1x1",
        "-enable-intrabc", "0", "-pix_fmt", "yuv420p", "-f", "ivf", ivf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s" % result.stderr[-2000:])


def check_lossless(probe, path):
    """The probe's header line, which has to show the lossless header shape."""
    result = subprocess.run([probe, path], capture_output=True, text=True)
    lines = [line for line in result.stdout.splitlines() if "deblock" in line]
    if not lines:
        raise SystemExit("%s: the probe reported no loop filter line" % path)
    line = lines[0]
    # 5.9.2: all_lossless skips the CDEF strengths, and the three parameter sets
    # leave the levels at zero
    if "y strengths" in line or "deblock 0/0" not in line:
        raise SystemExit("%s: not a lossless frame header: %s" % (path, line.strip()))
    if "restoration no" not in result.stdout:
        raise SystemExit("%s: the sequence header allows restoration" % path)
    if "base_q_idx 0" not in result.stdout:
        raise SystemExit("%s: the first frame is not at quantiser zero" % path)
    return line.strip()


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
        os.path.dirname(os.path.abspath(args.output)), ".lossless_source.yuv")
    with open(raw_path, "wb") as handle:
        handle.write(source_clip(args.frames))
    try:
        encode(args.ffmpeg, raw_path, args.output, args.frames)
    finally:
        if not args.keep_raw:
            os.unlink(raw_path)

    line = check_lossless(args.probe, args.output)
    print("%s: yuv420p, %d frames, %d bytes" %
          (args.output, args.frames, os.path.getsize(args.output)))
    print("  %s" % line)


if __name__ == "__main__":
    sys.exit(main())
