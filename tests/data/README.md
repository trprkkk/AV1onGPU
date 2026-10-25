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
there so that the encoder spends its screen content tools on palettes. The
`intrabc.ivf` fixture below is the one that codes block copies.
`palette_sc.hash` holds the per frame hashes and `stream_palette_sc` checks them,
so it is the only fixture that exercises the palette syntax - without it the
palette code could be arbitrarily wrong and nothing would fail.

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

`intrabc.ivf` is the intra block copy fixture: 16 frames of 320x240, 6.0 KB,
encoded with libaom at `-cpu-used 2 -crf 24 --lag-in-frames 0 -g 1` and
`-aom-params "tune-content=screen:enable-intrabc=1:enable-restoration=0:tile-columns=0:tile-rows=0:enable-palette=1:enable-cdef=1"`.
`-g 1` makes every frame a key frame, because a block copy block only exists in an
intra frame - with a longer key frame interval the stream had exactly one block
copy in twelve frames. All intra, it codes **239** of them: 204 whose vector comes
off a neighbour's candidate stack and 35 that fall back to the "one superblock up"
offset, on blocks from 8x32 to 32x32, ending up on four distinct displacements so
the residual and the 7.11.5 clipping both run. It also codes palettes, which is
fine - this stream is about the block copies. Note that a block copy is an *inter*
block in an *intra* frame, so `--enable-cdef=1` here is deliberate: the encoder
signals CDEF off for a frame that uses block copy, and that is the header path
`allow_intrabc` takes.

`intrabc.hash` holds the per frame hashes, `stream_intrabc` checks them and
`opencl_intrabc` replays the same frames through the device kernels, where the
block copy's job has a negative reference slot and therefore reads the picture
being decoded rather than a pool slot.

The clip is generated, so it can be rebuilt from nothing (the moving widget and
the scrolling stripes are what make a block copy cheaper than intra prediction):

```python
import numpy as np

W, H, N = 320, 240, 16
def make(t):
    img = np.full((H, W), 40, np.uint8)
    img[16:112, 16:144] = 200                    # flat panels
    img[128:224, 16:144] = 120
    img[16:112, 160:304] = 90
    img[128:224, 160:304] = 230
    off = (t * 4) % 64                           # text-like stripes that scroll
    for i in range(8):
        y = 24 + i * 8
        for x in range(160 + off, 304, 12):
            img[y:y + 4, x:x + 6] = 250
    wx = 32 + (t * 3) % 80                       # a widget that slides
    img[160:200, wx:wx + 40] = 20
    img[168:192, wx + 8:wx + 32] = 255
    img[232:240, 300:320] = 255                  # a static corner marker
    return img
with open('sc2.yuv', 'wb') as f:
    for t in range(N):
        luma = make(t)
        chroma = luma.reshape(120, 2, 160, 2).mean(axis=(1, 3)).astype(np.uint8)
        f.write(luma.tobytes())
        f.write(chroma.tobytes())
        f.write(chroma.tobytes())
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

`seg_altq.ivf` is the segmentation fixture: 30 frames of 320x240, 30 KB,
encoded with libaom at `-cpu-used 4 -crf 40 --aq-mode 3 --lag-in-frames 0 -g 30`
and `-aom-params "enable-restoration=0:tile-columns=0:tile-rows=0"`. `aq-mode 3`
makes the encoder move whole superblocks between segments, and a segment's only
feature is its quantiser: `base_q_idx` is 160 and segment 1 asks for `ALT_Q` -65
while segment 2 asks for -80, so the two classes code at quantiser 95 and 80.
That means the offsets need nine bits (`su(1 + bits)`), which is exactly the
width a parser gets wrong by one. Every frame carries `update_map` and
`temporal_update`, so the segment id of a block is either predicted from the
previous frame's map or coded as a distance from its neighbours, and 207 of the
stream's 2,586 blocks end up in segment 2, 15 in segment 1.
`seg_altq.hash` holds the per frame hashes and `stream_seg_altq` checks them;
`opencl_seg_altq` replays the same frames through the device kernels. Every frame
is bit exact against dav1d.

The clip is generated, so it can be rebuilt from nothing:

```python
import numpy as np

W, H, N = 320, 240, 30
rng = np.random.default_rng(11)
yy, xx = np.mgrid[0:H, 0:W]
base = 60 + xx * 0.30 + yy * 0.25 + rng.integers(-18, 18, (H, W))   # textured
out = bytearray()
for f in range(N):
    img = base.copy()
    x0 = 20 + (f * 4) % 260                      # a bright bar that sweeps
    img[30:120, x0:x0 + 56] += 60
    x1 = 60 + (f * 3) % 200                      # a dark box that slides
    img[130:220, x1:x1 + 70] -= 45
    y = np.clip(img, 0, 255).astype(np.uint8)
    u = np.full((H // 2, W // 2), 128, np.uint8)
    v = np.full((H // 2, W // 2), 128, np.uint8)
    u[:, :80] = 96                               # flat chroma, two halves
    u[:, 80:] = 170
    out += y.tobytes() + u.tobytes() + v.tobytes()
open('seg_src.yuv', 'wb').write(bytes(out))
```

`superres.ivf` is the superres fixture: 30 frames, 320 columns coded and 480
displayed (1.5x), 30 KB. No encoder reachable from here signals superres -
ffmpeg's `libaom-av1` wrapper does not pass `superres-mode` through and the aom
sources are not downloadable in this sandbox - so the fixture is an ordinary
stream whose *headers* are patched: the sequence header is told the picture is
displayed wider than it is coded and that superres is allowed, and every frame
header gains the four bits of `superres_params()` (`use_superres` plus the three
bit denominator) that then say 480/12. Nothing else changes, and the result is a
valid stream because the coded width the base encoder chose is exactly the
`FrameWidth` that `UpscaledWidth = 480` and `SuperresDenom = 12` work out to:
`FrameWidth = (UpscaledWidth * 8 + SuperresDenom / 2) / SuperresDenom = 320`.

One field is *dropped* rather than added. `allow_intrabc` is only coded when
`UpscaledWidth == FrameWidth`, so a frame that uses superres has no such bit: a
decoder that reads one consumes the first bit of the next field instead and the
header ends up in the wrong place. `make_superres.py` deletes it and realigns
the patched header to the byte boundary the tile data starts on.

```
python3 tests/data/make_superres.py plain.ivf superres.ivf --width 480
```

The base is any 30 frame 320x240 stream (this one is an encode of the
`seg_altq` clip below); the script reads the field positions out of
`av1gpu_probe`'s `AV1GPU_HDRPOS` trace instead of keeping a second copy of the
header syntax, so it stays right when the parser changes. `superres.hash` holds
the per frame hashes of *this* copy and `stream_superres` checks them, while
`opencl_superres` replays the same frames through the device kernels. Every frame
is bit exact against dav1d, and the decode is a real superres decode: the tiles
reconstruct 320 columns, CDEF runs on those, and the upscale kernel then produces
the 480 column picture (see `docs/ACCURACY.md`).

`grain.ivf` is the film grain fixture: 30 frames of 320x240, 32 KB, the
`seg_altq` stream with `film_grain_params()` patched into every frame header. No
encoder here signals grain (libaom's `film-grain` option is not reachable through
ffmpeg's wrapper) so the parameters are written by hand: four luma curve points,
two points each for Cb and Cr, `ar_coeff_lag` 2 with a full set of coefficients
(chroma ones included, so the luma correlation at the centre position runs),
overlap blending on and clipping to the restricted range on. `grain_seed` is 1234
for the first frame and 31 higher for each following one, so every frame's noise
is a different image.

```
python3 tests/data/make_grain.py seg_altq.ivf grain.ivf --seed 1234
```

The script reads the field positions from `av1gpu_probe`'s `AV1GPU_HDRPOS` trace
and re-aligns the patched header to the byte boundary the tile data starts on, so
it does not keep a second copy of the header syntax. `--profile` switches on one
part of the synthesis at a time (`luma`, `luma_ar`, `luma_overlap`, `luma_curve`,
`luma_clip`, `chroma_only`, `chroma_csl`), which is how the implementation was
bisected against dav1d; each profile is bit exact on its own. `grain.hash` holds
the per frame hashes and `stream_grain` checks them, while `opencl_grain` replays
the same frames through the device kernels. Every sample of every frame is equal
to dav1d's (see `docs/ACCURACY.md`).

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
