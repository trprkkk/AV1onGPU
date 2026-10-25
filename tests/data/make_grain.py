#!/usr/bin/env python3
"""Add film grain to an ordinary AV1 stream, bit for bit.

No encoder reachable from here can signal film grain: libaom only exposes it
through aomenc's own `--film-grain-test` switch (ffmpeg's libaom wrapper does not
pass it through), and the aom sources are not downloadable in this sandbox. So,
like the superres fixture, this one is *patched* rather than encoded:

  * the sequence header is told that `film_grain_params_present` is 1, and
  * every frame header gains the syntax of `film_grain_params()` (5.9.30).

Nothing else moves. The film grain parameters are the last field of a frame
header, so the patch appends them at its end, pads the header back to a byte
boundary and leaves the tile data untouched - the picture the tiles decode is
exactly the one the encoder wrote, and the grain is the only difference from the
base stream. The parameters themselves come from `GRAIN` below: a scaling curve
for luma, chroma points, a second order auto-regressive filter and both chroma
coupling coefficients, i.e. every part of the synthesis filter that can be
switched on.

Where the fields sit is not guessed: `av1gpu_probe` prints the bit positions
under AV1GPU_HDRPOS and this script reads that trace, so the header syntax stays
in one place - the parser - instead of a second copy here.

    python3 tests/data/make_grain.py in.ivf out.ivf [--probe path] [--seed 1234]

The base stream must not use `show_existing_frame` (its grain parameters would
have to be inherited instead of coded) and must be 4:2:0, 8 bit.
"""

import argparse
import os
import sys

from make_superres import (OBU_FRAME, OBU_FRAME_HEADER, OBU_SEQUENCE_HEADER, UNSET,
                           bits_of, build_obu, bytes_of, header_positions, parse_obus,
                           read_ivf, write_ivf)

INTER_FRAME = 1

# The film grain parameters every frame codes. The scaling curves are lists of
# (sample value, grain strength) points, strictly increasing in value; the
# auto-regressive coefficients are the *signed* values, i.e. what the syntax
# codes minus 128.
GRAIN = {
    "y": [(0, 44), (64, 34), (128, 18), (214, 6)],
    "cb": [(0, 28), (255, 10)],
    "cr": [(0, 28), (255, 10)],
    "chroma_scaling_from_luma": False,
    "grain_scaling_minus_8": 0,
    "ar_coeff_lag": 2,
    # 2 * lag * (lag + 1) = 12 luma coefficients, 13 chroma ones: the last
    # chroma coefficient sits at the centre position and reads the luma grain
    "ar_coeffs_y": [4, -3, 5, -2, 3, -4, 2, -1, 1, 3, -2, 2],
    "ar_coeffs_cb": [2, -1, 3, -2, 2, -1, 1, 0, 1, 2, -1, 1, 60],
    "ar_coeffs_cr": [2, -1, 3, -2, 2, -1, 1, 0, 1, 2, -1, 1, 60],
    "ar_coeff_shift_minus_6": 1,
    "grain_scale_shift": 0,
    "cb_mult": 128, "cb_luma_mult": 192, "cb_offset": 256,
    "cr_mult": 128, "cr_luma_mult": 192, "cr_offset": 256,
    "overlap_flag": True,
    "clip_to_restricted_range": True,
}


def u(value, bits):
    """A fixed width unsigned syntax element, as a list of bit characters."""
    if value < 0 or value >= (1 << bits):
        raise SystemExit(f"{value} does not fit in {bits} bits")
    return list(f"{value:0{bits}b}")


def film_grain_params(frame_type, seed, seq):
    """film_grain_params() (5.9.30), with update_grain = 1."""
    mono = seq["mono"]
    bits = []
    bits += u(1, 1)               # apply_grain
    bits += u(seed & 0xFFFF, 16)  # grain_seed
    if frame_type == INTER_FRAME:
        bits += u(1, 1)           # update_grain: the parameters are coded here
    bits += u(len(GRAIN["y"]), 4)
    for value, scaling in GRAIN["y"]:
        bits += u(value, 8) + u(scaling, 8)
    if not mono:
        bits += u(1 if GRAIN["chroma_scaling_from_luma"] else 0, 1)
    chroma = not mono and not GRAIN["chroma_scaling_from_luma"] and not (
        seq["ss_x"] and seq["ss_y"] and not GRAIN["y"])
    if chroma:
        bits += u(len(GRAIN["cb"]), 4)
        for value, scaling in GRAIN["cb"]:
            bits += u(value, 8) + u(scaling, 8)
        bits += u(len(GRAIN["cr"]), 4)
        for value, scaling in GRAIN["cr"]:
            bits += u(value, 8) + u(scaling, 8)
    bits += u(GRAIN["grain_scaling_minus_8"], 2)
    bits += u(GRAIN["ar_coeff_lag"], 2)
    lag = GRAIN["ar_coeff_lag"]
    num_pos_luma = 2 * lag * (lag + 1)
    num_pos_chroma = num_pos_luma + (1 if GRAIN["y"] else 0)
    if GRAIN["y"]:
        for value in GRAIN["ar_coeffs_y"][:num_pos_luma]:
            bits += u(value + 128, 8)
    if chroma and (GRAIN["chroma_scaling_from_luma"] or GRAIN["cb"]):
        for value in GRAIN["ar_coeffs_cb"][:num_pos_chroma]:
            bits += u(value + 128, 8)
    if chroma and (GRAIN["chroma_scaling_from_luma"] or GRAIN["cr"]):
        for value in GRAIN["ar_coeffs_cr"][:num_pos_chroma]:
            bits += u(value + 128, 8)
    bits += u(GRAIN["ar_coeff_shift_minus_6"], 2)
    bits += u(GRAIN["grain_scale_shift"], 2)
    if chroma and GRAIN["cb"]:
        bits += u(GRAIN["cb_mult"], 8) + u(GRAIN["cb_luma_mult"], 8) + u(GRAIN["cb_offset"], 9)
    if chroma and GRAIN["cr"]:
        bits += u(GRAIN["cr_mult"], 8) + u(GRAIN["cr_luma_mult"], 8) + u(GRAIN["cr_offset"], 9)
    bits += u(1 if GRAIN["overlap_flag"] else 0, 1)
    bits += u(1 if GRAIN["clip_to_restricted_range"] else 0, 1)
    return bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--probe", default=os.path.join("build", "bin", "av1gpu_probe"))
    ap.add_argument("--seed", type=int, default=1234,
                    help="grain_seed of the first frame; later frames step it")
    args = ap.parse_args()

    seq, frame_positions = header_positions(args.probe, args.input)
    if seq["mono"]:
        raise SystemExit("monochrome streams are not supported by this fixture")
    header, frames = read_ivf(args.input)

    patched_frames = []
    frame_index = 0
    seq_done = False
    for stamp, payload in frames:
        out_obus = []
        for kind, body in parse_obus(payload):
            if kind == OBU_SEQUENCE_HEADER:
                bits = list(bits_of(body))
                bits[seq["film_grain"]] = "1"
                body = bytes_of("".join(bits))
                seq_done = True
            elif kind in (OBU_FRAME, OBU_FRAME_HEADER):
                bits = bits_of(body)
                if bits[0] == "1":
                    raise SystemExit("the stream uses show_existing_frame, which this "
                                     "fixture cannot patch")
                pos = frame_positions[frame_index]
                if pos["type"] not in (0, INTER_FRAME):
                    raise SystemExit(f"key/intra-only frame {frame_index}: the fixture only "
                                     f"patches key frames and inter frames")
                # append the parameters at the end of the header, then realign:
                # the padding between the last field and the tile data is byte
                # alignment, which the patch has to redo from where it now ends
                ticket = list(bits[:pos["header_bits"]])
                ticket += film_grain_params(pos["type"], args.seed + 31 * frame_index, seq)
                while len(ticket) % 8:
                    ticket.append("0")
                body = bytes_of("".join(ticket)) + body[pos["tile_data_offset"]:]
                frame_index += 1
            out_obus.append(build_obu(kind, body))
        patched_frames.append((stamp, b"".join(out_obus)))

    if not seq_done:
        raise SystemExit("the stream has no sequence header")
    if frame_index != len(frame_positions):
        raise SystemExit(f"patched {frame_index} frame headers, the probe reported "
                         f"{len(frame_positions)}")
    write_ivf(args.output, header, patched_frames)
    print(f"wrote {args.output}: {len(patched_frames)} temporal units, grain seed "
          f"{args.seed} (+{31} per frame)")


if __name__ == "__main__":
    sys.exit(main())
