#!/usr/bin/env python3
"""Turn an ordinary AV1 stream into a superres one, bit for bit.

No encoder reachable from here can signal superres: libaom only exposes
`superres-mode` through aomenc's own command line (ffmpeg's `-aom-params` does
not take it), and the aom sources are not downloadable in this sandbox. So the
fixture is *patched* rather than encoded:

  * the sequence header is told the picture is displayed wider than it is coded
    and that the sequence may use superres, and
  * every frame header gains the four bits of `superres_params()` (use_superres
    plus the three bit denominator) that then say so.

The tile data is not touched, and that is what keeps the result a valid stream:
the width the encoder coded is exactly the `FrameWidth` the patched
`UpscaledWidth` and denominator work out to, since `FrameWidth =
(UpscaledWidth * 8 + SuperresDenom / 2) / SuperresDenom`. With a denominator of
12 that display width is 1.5x the coded width.

One bit is *removed* rather than added. `allow_intrabc` is only coded when
`UpscaledWidth == FrameWidth`, so a frame that uses superres has no such field:
a decoder that reads one consumes the first bit of the next field instead. The
patch has to drop it for the stream to stay parseable.

Where the fields sit is not guessed: `av1gpu_probe` prints it under
AV1GPU_HDRPOS (in bits, counted from the start of the OBU payload) and this
script reads that trace. That keeps the header syntax in one place - the
parser - instead of maintaining a second copy of it here.

    python3 tests/data/make_superres.py in.ivf out.ivf [--probe path] [--width 480]

The stream has to be coded narrower than the width asked for; the script refuses
anything else (it cannot create picture content, only relabel it).
"""

import argparse
import os
import subprocess
import sys

OBU_SEQUENCE_HEADER = 1
OBU_FRAME_HEADER = 3
OBU_FRAME = 6
SUPERRES_DENOMINATOR = 12
UNSET = 2 ** 64 - 1


def read_ivf(path):
    """Returns (header_bytes, [(timestamp, payload_bytes)])."""
    with open(path, "rb") as handle:
        data = handle.read()
    if data[:4] != b"DKIF":
        raise SystemExit(f"{path}: not an IVF file")
    header_size = int.from_bytes(data[6:8], "little")
    header = data[:header_size]
    frames = []
    pos = header_size
    while pos + 12 <= len(data):
        size = int.from_bytes(data[pos:pos + 4], "little")
        stamp = data[pos + 4:pos + 12]
        payload = data[pos + 12:pos + 12 + size]
        if len(payload) != size:
            raise SystemExit(f"{path}: truncated frame at {pos}")
        frames.append((stamp, payload))
        pos += 12 + size
    return header, frames


def write_ivf(path, header, frames):
    with open(path, "wb") as handle:
        handle.write(header)
        for stamp, payload in frames:
            handle.write(len(payload).to_bytes(4, "little"))
            handle.write(stamp)
            handle.write(payload)


def parse_obus(payload):
    """Splits one temporal unit's payload into (type, bytes) OBUs."""
    obus = []
    pos = 0
    while pos < len(payload):
        head = payload[pos]
        pos += 1
        kind = (head >> 3) & 0xF
        has_extension = (head >> 2) & 1
        has_size = (head >> 1) & 1
        if has_extension:
            pos += 1
        if not has_size:
            obus.append((kind, payload[pos:]))
            break
        size = 0
        shift = 0
        while True:
            byte = payload[pos]
            pos += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                break
        obus.append((kind, payload[pos:pos + size]))
        pos += size
    return obus


def build_obu(kind, body):
    head = (kind << 3) | 0x02  # has_size
    size = bytearray()
    value = len(body)
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            size.append(byte | 0x80)
        else:
            size.append(byte)
            break
    return bytes([head]) + bytes(size) + body


def bits_of(data):
    return "".join(f"{byte:08b}" for byte in data)


def bytes_of(bits):
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def header_positions(probe, path):
    """Runs the probe with AV1GPU_HDRPOS and returns (seq, [frames])."""
    env = dict(os.environ, AV1GPU_HDRPOS="1")
    result = subprocess.run([probe, path], capture_output=True, text=True, env=env)
    seq = None
    frames = []
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
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--probe", default=os.path.join("build", "bin", "av1gpu_probe"))
    ap.add_argument("--width", type=int, default=0,
                    help="the display width to claim (default 1.5x the coded width)")
    args = ap.parse_args()

    seq, frame_positions = header_positions(args.probe, args.input)
    header, frames = read_ivf(args.input)

    coded_width = None
    target = args.width
    patched_frames = []
    frame_index = 0
    seq_done = False

    for stamp, payload in frames:
        out_obus = []
        for kind, body in parse_obus(payload):
            if kind == OBU_SEQUENCE_HEADER:
                bits = list(bits_of(body))
                width_bits = seq["width_bits"]
                coded_width = int("".join(bits[seq["max_frame_width"]:
                                                seq["max_frame_width"] + width_bits]), 2) + 1
                if not target:
                    target = (coded_width * 3) // 2
                expected = (target * 8 + SUPERRES_DENOMINATOR // 2) // SUPERRES_DENOMINATOR
                if expected != coded_width:
                    raise SystemExit(
                        f"denominator {SUPERRES_DENOMINATOR} on a {target} wide display codes "
                        f"{expected} wide, but the tiles code {coded_width}: pass a --width that "
                        f"works out to the coded width")
                here = f"{target - 1:0{width_bits}b}"
                bits[seq["max_frame_width"]:seq["max_frame_width"] + width_bits] = list(here)
                bits[seq["enable_superres"]] = "1"
                body = bytes_of("".join(bits))
                seq_done = True
            elif kind in (OBU_FRAME, OBU_FRAME_HEADER):
                bits = bits_of(body)
                if bits[0] == "1":  # show_existing_frame: no header to patch
                    out_obus.append(build_obu(kind, body))
                    continue
                pos = frame_positions[frame_index]
                frame_index += 1
                tail = body[pos["tile_data_offset"]:]
                # The header is truncated at its last field, not at the tile
                # data: the padding in between is byte alignment, and the
                # patched header has to be realigned from where it now ends.
                ticket = list(bits[:pos["header_bits"]])
                if pos["size_w"] != UNSET:
                    ticket[pos["size_w"]:pos["size_w"] + seq["width_bits"]] = list(
                        f"{target - 1:0{seq['width_bits']}b}")
                # superres_params(): use_superres = 1, coded_denom = denom - 9
                ticket[pos["use_superres"]:pos["use_superres"]] = list(
                    "1" + f"{SUPERRES_DENOMINATOR - 9:03b}")
                if pos["intrabc"] != UNSET:
                    # allow_intrabc is only coded when UpscaledWidth ==
                    # FrameWidth, so a scaled frame has no such field. The four
                    # bits inserted above all sit before it.
                    del ticket[pos["intrabc"] + 4]
                while len(ticket) % 8:
                    ticket.append("0")
                body = bytes_of("".join(ticket)) + tail
            out_obus.append(build_obu(kind, body))
        patched_frames.append((stamp, b"".join(out_obus)))

    if not seq_done:
        raise SystemExit("the stream has no sequence header")
    if frame_index != len(frame_positions):
        raise SystemExit(f"patched {frame_index} frame headers, the probe reported "
                         f"{len(frame_positions)}")
    write_ivf(args.output, header, patched_frames)
    print(f"wrote {args.output}: {coded_width} coded, {target} displayed, "
          f"{len(patched_frames)} temporal units")


if __name__ == "__main__":
    sys.exit(main())
