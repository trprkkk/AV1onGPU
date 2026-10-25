#!/usr/bin/env python3
"""Build the 10 bit fixture: a smooth clip, encoded with libaom at 10 bit.

The suite was eight bit only, and the 10 bit paths are not the same code: the
deblocking thresholds are scaled by `1 << (BitDepth - 8)`, the film grain range
is +/-512 instead of +/-127, and the reconstruction arithmetic has two more bits
of headroom. A fixture that decodes only 8 bit samples cannot tell any of that
apart from the 8 bit behaviour, so this script writes one.

The clip is generated here rather than taken from anywhere, so the fixture can be
rebuilt from nothing:

  * a slow luma ramp with a little noise on it, and flat chroma - flat enough
    that libaom codes the first frame's *loop filter level as zero* while still
    enabling the per-reference deltas, and

  * twelve frames of it at 320x240.

That first frame is the reason the clip is this simple. `decode_frame_wrapup`
only runs the loop filter when a luma level was coded, so a frame that codes
0 and 0 is left unfiltered no matter what the deltas say; a decoder that applies
them anyway produces a picture that is subtly wrong (a few thousand samples off
by one or two) and nothing else in the suite notices. The script checks that the
stream it produced still has that property before it writes anything:

    python3 tests/data/make_aom10.py out.ivf [--ffmpeg path] [--probe path]

The encoder is the libaom build inside the ffmpeg binary that
`imageio-ffmpeg` installs (`imageio_ffmpeg.get_ffmpeg_exe()`); pass --ffmpeg if
yours lives somewhere else. Encoding is deterministic for a given libaom build,
so the .ivf comes out the same every time.
"""

import argparse
import os
import subprocess
import sys

WIDTH, HEIGHT, FRAMES = 320, 240, 12
CRF = "40"
CPU_USED = "8"


def source_clip():
    """The 10 bit 4:2:0 clip, as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    rng = np.random.default_rng(11)
    out = bytearray()
    for frame in range(FRAMES):
        y = np.zeros((HEIGHT, WIDTH), np.int32)
        for row in range(HEIGHT):
            # a ramp that moves 18 levels per frame across and 10 down, so every
            # frame is a little brighter than the one before it
            across = (2 * (np.arange(WIDTH) - WIDTH // 2) * 18) // WIDTH
            down = (2 * (row - HEIGHT // 2) * 10) // HEIGHT
            y[row] = 120 + across + down + frame * 3
        y += rng.integers(-4, 5, size=(HEIGHT, WIDTH))
        chroma = []
        for _ in range(2):
            plane = 128 + rng.integers(-2, 3, size=(HEIGHT // 2, WIDTH // 2))
            chroma.append(plane)
        for plane in [y] + chroma:
            out += np.clip(plane, 0, 1023).astype("<u2").tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", str(FRAMES), "-tiles", "1x1",
        "-pix_fmt", "yuv420p10le", "-f", "ivf", ivf_path,
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

    raw_path = args.keep_raw or os.path.join(os.path.dirname(os.path.abspath(args.output)),
                                             ".aom10_source.yuv")
    with open(raw_path, "wb") as handle:
        handle.write(source_clip())
    try:
        encode(args.ffmpeg, raw_path, args.output)
    finally:
        if not args.keep_raw:
            os.unlink(raw_path)

    seq, frames = header_positions(args.probe, args.output)
    if seq.get("bitdepth") != 10:
        raise SystemExit("the stream is not 10 bit: %r" % (seq,))
    if len(frames) < FRAMES:
        raise SystemExit("only %d frame headers, expected %d" % (len(frames), FRAMES))
    quiet = [index for index, frame in enumerate(frames)
             if frame.get("lf_y0", 0) == 0 and frame.get("lf_y1", 0) == 0]
    if not quiet or not any(frames[index].get("deltas", 0) for index in quiet):
        raise SystemExit("no frame codes a zero luma level with the reference "
                         "deltas enabled, so the stream does not exercise the "
                         "rule this fixture exists for")
    size = os.path.getsize(args.output)
    print("%s: %d frames, %d bytes, %s" %
          (args.output, len(frames), size,
           ", ".join("frame %d unfiltered" % index for index in quiet)))


if __name__ == "__main__":
    sys.exit(main())
