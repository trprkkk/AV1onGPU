#!/usr/bin/env python3
"""Build the error resilient fixture: inter frames that are also frame-id numbered.

``-error-resilience 1`` makes libaom code ``error_resilient_mode = 1`` on the
inter frames, which changes the frame header in ways nothing else in the
fixture set reaches:

  * ``primary_ref_frame`` is not coded at all (5.9.2 sets it to
    PRIMARY_REF_NONE, so every frame starts from the default CDF tables);
  * ``frame_id_numbers_present_flag`` is set, so each reference is named by a
    ``delta_frame_id`` as well as a slot index, and the decoder has to derive the
    id the slot must hold (``expectedFrameId``) and refuse to predict from a slot
    that holds something else;
  * every slot's order hint is re-stated and a slot that disagrees is emptied.

The 5.9.2 loop that reads the reference names is *interleaved*: for each of the
seven references, the three bit index comes first and then that reference's
``delta_frame_id``. Reading the seven indices and then the seven ids consumes
the same number of bits, so the rest of the header still parses and a stream
whose references all point at the same slot still decodes - which is why the
bug this pins survived every earlier fixture. This clip's second inter frame
predicts from two different slots with two different ids, and the pairs the
decoder built were half from one reference and half from another: the header
then failed its own conformance check ("a reference frame id does not match the
frame its slot holds") on frame 2.

The content is panels on a flat background with hard edges and a per frame pan,
which is what makes the encoder keep several slots alive in a four frame group.
``-g 4 -lag-in-frames 0`` keeps the group small (four frames, no altref), and
``-cpu-used 5 -crf 32`` keeps the fixture a few kilobytes.

    python3 tests/data/make_error_resilient_clip.py tests/data/error_resilient.ivf

The encoder is the libaom build inside the ffmpeg binary `imageio-ffmpeg`
installs (`--ffmpeg` selects another). Encoding is deterministic for a given
build, so `error_resilient.ivf` comes out byte identical every time and the
hashes in `error_resilient.hash` can be recorded at all.
"""

import argparse
import os
import subprocess

WIDTH, HEIGHT, FRAMES = 320, 180, 4
CPU_USED = "5"
CRF = "32"
GOP = "4"


def source_clip(frames=FRAMES):
    """The clip as the bytes ffmpeg's rawvideo demuxer wants."""
    import numpy as np

    cw, ch = WIDTH // 2, HEIGHT // 2

    out = bytearray()
    for frame in range(frames):
        luma = np.full((HEIGHT, WIDTH), 28.0)
        for k in range(7):
            x0 = 10 + k * (WIDTH // 8) + 2 * frame
            y0 = 8 + (k % 3) * (HEIGHT // 3)
            luma[y0:y0 + HEIGHT // 4, x0:x0 + WIDTH // 6] = (50 + 30 * k) % 256
        # single sample wide rows and columns: hard edges the predictor cannot
        # code, so the frames differ enough for the encoder to keep more than
        # one reference alive
        luma[0::17, :] = 220 - 5 * frame
        luma[:, 0::23] = 30 + 10 * frame
        out += np.clip(luma, 0, 255).astype(np.uint8).tobytes()

        yy, xx = np.mgrid[0:ch, 0:cw]
        for phase, fn in ((0.0, np.sin), (0.7, np.cos)):
            chroma = 128 + 60 * fn(xx / 5.0 + phase) + 3 * frame
            out += np.clip(chroma, 0, 255).astype(np.uint8).tobytes()
    return bytes(out)


def encode(ffmpeg, raw_path, ivf_path, frames):
    command = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p",
        "-s", "%dx%d" % (WIDTH, HEIGHT), "-r", "30", "-i", raw_path,
        "-c:v", "libaom-av1", "-cpu-used", CPU_USED, "-crf", CRF,
        "-lag-in-frames", "0", "-g", GOP, "-error-resilience", "1",
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
