# Test streams

`tile_trace.hpp` is not a stream but the first 480 range coder events dav1d
logged while decoding that stream's key frame, together with the CDFs it used
and the range coder state it reached; `tests/test_entropy.cpp` replays them.
See `docs/ACCURACY.md` for how it was captured.

`lw_t2.ivf` and `lw_t3.ivf` are 30 frame 320x240 streams that code local warped
motion (`motion_mode = LOCALWARP`) and OBMC, the two inter tools that no smaller
fixture exercises. They were encoded with libaom at `--cpu-used=2` and `-3`,
`--crf=30 --lag-in-frames=25 --enable-restoration=0 --tiles=1x1`, and are kept
here rather than next to the other fixtures because they are checked for *pixels*
and not only for syntax: `tests/data/lw_t?.hash` holds one FNV-1a 64 hash per
shown frame of the reference decoder's output, and the `stream_lw_t2` /
`stream_lw_t3` tests decode the streams with `av1gpu_decode --hash-check` and
compare. A YUV reference would be 3.4 MB per stream; the hash lists are 2 KB.

To regenerate a hash list, decode the stream, compare it with dav1d, and only
then record our own output:

```
./build/bin/dav1d -i lw_t2.ivf --muxer yuv -o reference.yuv
./build/bin/av1gpu_decode lw_t2.ivf -o ours.yuv --compare reference.yuv
./build/bin/av1gpu_decode lw_t2.ivf --hash-out lw_t2.hash
```

`palette_sc.ivf` is the screen content fixture: 10 frames of 320x240, 6.7 KB,
encoded with libaom at `-cpu-used 4 -crf 30 --lag-in-frames 0 -g 10` and
`-aom-params "tune-content=screen:enable-restoration=0:tile-columns=0:tile-rows=0:enable-intrabc=0"`.
Every intra block that wants a palette codes one, on one or both planes, with 2,
3 and 4 colours and reuse of the neighbour palette cache; intrabc is switched off
because intrabc blocks are refused by name. `palette_sc.hash` holds the per frame
hashes and `stream_palette_sc` checks them, so it is the only fixture that
exercises the palette syntax - without it the palette code could be arbitrarily
wrong and nothing would fail.

The clip itself is generated, so it can be rebuilt from nothing:

```python
import numpy as np

rng = np.random.default_rng(7)
W, H, N = 320, 240, 10
out = bytearray()
for f in range(N):
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (28, 30, 34)                      # flat background
    for k in range(6):                         # flat colour panels
        x0, y0 = 20 + k * 45 + (3 * f if k % 2 else 0), 15 + (k % 3) * 60
        img[y0:y0 + 50, x0:x0 + 70] = ((60 + 40 * k) % 256, (200 - 20 * k) % 256,
                                       (120 + 10 * k) % 256)
    img[0::17, :, 0] = img[1::17, :, 0] = 220  # thin lines
    img[:, 0::23, 2] = img[:, 1::23, 2] = 230
    for gy in range(8):                        # text-like glyph runs
        for gx in range(40):
            if rng.random() < 0.45:
                bx, by = 8 + gx * 7, 30 + gy * 25
                img[by:by + 9, bx:bx + 5] = 255
    y = img[:, :, 0]
    u = np.full((H // 2, W // 2), 128, np.uint8)
    v = np.full((H // 2, W // 2), 128, np.uint8)
    u[:, :W // 4] = 96 if f % 2 == 0 else 150  # flat chroma so palettes appear
    u[:, W // 4:W // 2] = 170
    v[:, W // 2:] = 90
    v[:, :W // 4] = 200
    out += y.tobytes() + u.tobytes() + v.tobytes()
open('src.yuv', 'wb').write(bytes(out))
```

`inter_intra.ivf` is the inter/intra compound fixture: 110 frames of 320x240
and 19 KB, cut from a longer encode of the same synthetic clip as `lw_t2`/`lw_t3`
(SVT-AV1, `preset 1`, `crf 30`, `-g 8`,
`svtav1-params "enable-restoration=0"`, muxed to IVF with `av`, no re-encode of
the kept frames - the file is a prefix of the longer one, so the encoder's
decisions are the ones it really made). It is the only fixture that codes
inter/intra compound blocks: 23 of them, with `interintra_mode` in its V, H and
SMOOTH variants and 5 of them as `wedge_interintra` (wedge indices 4, 8 and 12),
on 16x16, 32x32, 16x32, 32x16 and 16x8 blocks, in frames 1 to 109.
`inter_intra.hash` holds the per frame hashes and `stream_inter_intra` checks
them, and the same stream is replayed through the device kernels by
`opencl_inter_intra`. The DC mode (a flat weight of 32) does not appear in any
encoding of this clip, so it is covered by `test_kernels`' inter/intra case
instead, together with the four mask rules and the chroma sampling of the mask.

The clip generator was as follows (all 160 frames; the fixture keeps the first
110):

```python
import numpy as np

W, H, N = 320, 240, 160
yy, xx = np.mgrid[0:H, 0:W]
base = 80 + xx * 0.35 + yy * 0.2
for k in range(6):                            # a moving texture
    base += np.sin(xx * (0.017 + 0.009 * k) + yy * (0.02 + 0.013 * k) + k) * (6 - k)
out = bytearray()
for f in range(N):
    img = base.copy()
    x0 = 20 + (f * 2) % 240                    # a bright bar that moves
    img[40:100, x0:x0 + 48] = 210
    y0 = 120 + int(18 * np.sin(f * 0.35))      # a dark box that slides
    img[y0:y0 + 40, 200:260] = 30
    for i in range(30):                        # a ramp
        img[170 + i, 90 + i:130 + i] = 150
    y = np.clip(img, 0, 255).astype(np.uint8)
    u = np.full((H // 2, W // 2), 128, np.uint8)
    v = np.full((H // 2, W // 2), 128, np.uint8)
    u[20:50, x0 // 2:(x0 + 48) // 2] = 100     # chroma that follows the bar
    v[20:50, x0 // 2:(x0 + 48) // 2] = 160
    out += y.tobytes() + u.tobytes() + v.tobytes()
open('src160.yuv', 'wb').write(bytes(out))
```

The truncated file is the first 110 frame records of the 160 frame one; every
frame of it decodes bit exactly against dav1d, and so does the whole stream
through the CPU backend and through the OpenCL kernels.

`svt_av1_320x240.ivf` is a real AV1 stream, not a hand written header: six frames
of a 320x240 gradient at 8 bit 4:2:0, encoded with SVT-AV1 (crf 40, preset 10)
and muxed into IVF. The front-end tests parse it end to end, so the parser is
checked against bytes produced by an encoder rather than only against the same
bit writer that produced the test data.

To regenerate it (needs a Python environment with `av` >= 14, which brings both
SVT-AV1 and libdav1d):

```python
import av, numpy as np

w, h, n = 320, 240, 6
rng = np.random.default_rng(7)
out = av.open('svt_av1_320x240.ivf', 'w', format='ivf')
st = out.add_stream('libsvtav1', rate=30)
st.width, st.height, st.pix_fmt = w, h, 'yuv420p'
st.options = {'crf': '40', 'preset': '10'}
for i in range(n):
    yy = (np.add.outer(np.arange(h), np.arange(w)) * 3 + i * 11) % 256
    yy = np.clip(yy + rng.integers(-8, 8, (h, w)), 0, 255).astype(np.uint8)
    frame = av.VideoFrame(w, h, 'yuv420p')
    frame.planes[0].update(yy.tobytes())
    frame.planes[1].update(np.full((h // 2, w // 2), 128, np.uint8).tobytes())
    frame.planes[2].update(np.full((h // 2, w // 2), 128, np.uint8).tobytes())
    for packet in st.encode(frame):
        out.mux(packet)
for packet in st.encode(None):
    out.mux(packet)
out.close()
```

The expected values in `tests/test_frontend.cpp` (quantiser, deblock levels, tile
offsets, order hints) were cross-checked against FFmpeg's `trace_headers` bitstream
filter, which prints every syntax element with its bit position.
