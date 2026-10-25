#!/usr/bin/env python3
"""Build the loop restoration fixtures: a clip both filters are chosen for.

Restoration is a rate/distortion decision, so a fixture that carries it has to
contain the content each filter exists for: the self guided filter pays off on
smooth gradients with a little noise on them, the Wiener filter on hard edged
panels. This clip alternates the two every four frames and pans both three
samples a frame, so the inter frames code residual *and* restoration, and the
chroma planes carry the same mix - which is what makes the encoder pick a filter
for them as well.

    python3 tests/data/make_lr_clip.py tests/data/lr.ivf
    python3 tests/data/make_lr_clip.py tests/data/lr_444.ivf --pix-fmt 444
    python3 tests/data/make_lr_clip.py tests/data/lr_422.ivf --pix-fmt 422

The three numbers in the name of a fixture are the only thing that changes
between them: the 4:2:0 one is `lr.ivf`, the 4:4:4 and 4:2:2 ones exist because
restoration on a plane that is not subsampled the way 4:2:0 is does not take the
same code path (the chroma unit grid halves only for 4:2:0, and a chroma sample
of a 4:4:4 frame still has to take its difference weighted mask from the *luma*
prediction). The same clip is coded at 10 and 12 bit by
`make_lr_depth.py` from the same generator, because the filters' bit depth
shifts are a no-op at 8 bit.

The encoder is the libaom build inside the ffmpeg binary that `imageio-ffmpeg`
installs (`--ffmpeg` selects another), and encoding is deterministic for a given
libaom build: `lr.ivf` comes out byte identical every time, which is why the
hashes in `lr*.hash` can be recorded at all.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 320, 240, 8
CRF = "36"
CPU_USED = "4"

# short name -> (ffmpeg pixel format, x subsampling, y subsampling)
FORMATS = {
    "420": ("yuv420p", 2, 2),
    "422": ("yuv422p", 2, 1),
    "444": ("yuv444p", 1, 1),
}


def source_clip(ss_x, ss_y, frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(5)
    # the generator works on a canvas four samples wider than a frame and every
    # frame takes the next three columns of it, so the picture pans
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH + 4 * frames].astype(np.float64)
    smooth = 110 + 35 * np.sin(xx / 47.0) * np.cos(yy / 31.0)
    tex = rng.integers(0, 256, (HEIGHT, WIDTH + 4 * frames)).astype(np.float64)
    tex = (tex * 3 + np.roll(tex, 1, 1) * 2 + np.roll(tex, -1, 1)) / 7 - 128
    noise = rng.normal(0, 7, (HEIGHT, WIDTH + 4 * frames))
    cw, ch = WIDTH // ss_x, HEIGHT // ss_y
    yyc, xxc = np.mgrid[0:ch, 0:cw + 4 * frames // ss_x].astype(np.float64)
    u = 128 + 50 * np.sin(xxc / 9.0) * np.cos(yyc / 13.0)
    u += rng.normal(0, 4, (ch, cw + 4 * frames // ss_x))
    v = 128 + 40 * np.cos(xxc / 11.0)
    v += rng.normal(0, 4, (ch, cw + 4 * frames // ss_x))

    out = bytearray()
    for frame in range(frames):
        smooth_frame = frame < frames // 2
        luma = smooth + noise if smooth_frame else 130 + tex
        luma = np.clip(luma[:, 3 * frame:3 * frame + WIDTH] + 2 * frame, 0, 255)
        if not smooth_frame:
            # hard edged panels, only on the frames the Wiener filter is for
            for k in range(6):
                x0 = 10 + (k % 3) * 100 + 3 * frame
                y0 = 15 + (k // 3) * 115 + 2 * frame
                luma[y0:y0 + 55, x0:x0 + 70] = (40 + 35 * k) % 256
        out += luma.astype(np.uint8).tobytes()
        for plane in (u, v):
            out += np.clip(plane[:, frame:frame + cw], 0, 255).astype(np.uint8).tobytes()
    return bytes(out)


def encode(ffmpeg, pix_fmt, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", pix_fmt,
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(frames), "-tiles", "1x1",
        "-enable-restoration", "1", "-pix_fmt", pix_fmt, "-f", "ivf", ivf_path,
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit("ffmpeg failed:\n%s" % result.stderr[-2000:])


def probe_restoration(probe, path):
    """The probe's `restoration` lines, one per frame."""
    result = subprocess.run([probe, path], capture_output=True, text=True)
    lines = [line.strip() for line in result.stdout.splitlines()
             if "restoration" in line and "unit" in line]
    if not lines:
        raise SystemExit("%s: the probe reported no restoration at all" % path)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("--pix-fmt", default="420", choices=sorted(FORMATS),
                        help="default 420, the lr.ivf fixture")
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

    ffmpeg_format, ss_x, ss_y = FORMATS[args.pix_fmt]
    raw_path = args.keep_raw or os.path.join(
        os.path.dirname(os.path.abspath(args.output)), ".lr_clip_source.yuv")
    with open(raw_path, "wb") as handle:
        handle.write(source_clip(ss_x, ss_y, args.frames))
    try:
        encode(args.ffmpeg, ffmpeg_format, raw_path, args.output, args.frames)
    finally:
        if not args.keep_raw:
            os.unlink(raw_path)

    lines = probe_restoration(args.probe, args.output)
    used = " ".join(lines)
    print("%s: %s, %d frames, %d bytes" %
          (args.output, ffmpeg_format, args.frames, os.path.getsize(args.output)))
    for line in lines[:2]:
        print("  %s" % line)
    for name in ("wiener", "sgrproj"):
        if name not in used:
            raise SystemExit("%s: no frame codes %s, so the fixture does not cover "
                             "that filter" % (args.output, name))


if __name__ == "__main__":
    sys.exit(main())
