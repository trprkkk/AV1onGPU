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

A stream deeper than 8 bit is compared with `--yuv16` on both sides: dav1d's
yuv muxer writes 16 bit little endian for those, `av1gpu_decode --yuv16` writes
the samples unchanged, and the hash is taken over those bytes - so a 10 bit
stream is checked sample for sample instead of after a shift.

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

`grain_inherit.ivf` is the same idea for the *inherited* parameters: the
`palette_sc` stream with every inter frame from index 3 on coding
`update_grain = 0` and naming a reference slot instead of a parameter set, which
is what an encoder does whenever the grain settings do not change (5.9.30) and
the only way to reach `load_grain_params()` in 7.18.3.1. The frame keeps its own
`grain_seed`, so its noise is still its own; the curve, the AR coefficients and
the chroma coupling are the slot's copy. Ten frames, 6.7 KB, and `grain_inherit.hash`
holds the per frame hashes.

```
python3 tests/data/make_grain.py palette_sc.ivf grain_inherit.ivf --seed 900 \
    --inherit-from 3
```

The coded index has to be one of the frame's own reference slots - the
specification makes that a conformance requirement and dav1d fails the frame if
it is violated - so `--inherit-from` takes the slot from the probe's trace (the
frame's LAST reference) instead of assuming one; `--inherit-slot` overrides it.
`stream_grain_inherit` checks the hashes. Every sample of every frame is equal to
dav1d's.

`lr.ivf` is the loop restoration fixture: 8 frames of 320x240, 82 KB, the only
stream in this directory whose frame headers and units code 7.17. It is an
ordinary libaom encode of a generated clip, so nothing had to be patched:

```
python3 clip.py > lr_clip.yuv          # the snippet below
ffmpeg -f rawvideo -pix_fmt yuv420p -s 320x240 -r 30 -i lr_clip.yuv \
    -c:v libaom-av1 -cpu-used 4 -crf 36 -lag-in-frames 0 -g 8 -tiles 1x1 \
    -enable-restoration 1 -f ivf lr.ivf
```

The clip alternates two kinds of luma content every four frames - a smooth ramp
with fine noise, which the encoder restores with the self guided filter, and a
fine texture with hard edged panels, which it restores with Wiener - and pans
both three pixels a frame, so the inter frames code residual and restoration.
The result is one stream that covers the whole syntax: three frames code the
type per unit (`SWITCHABLE` on luma, frames 4, 5 and 7), the other five code one
filter for the plane, and over the eight frames the units are 7x Wiener / 1x SGR
on luma and 12x Wiener / 4x SGR on the two chroma planes, with ten units that are
coded `RESTORE_NONE` inside a frame that restores. `lr.hash` holds the per frame
hashes and `stream_lr` checks them, while `opencl_lr` replays the same frames
through the device kernels - both are bit exact, and dav1d and libaom agree with
each other on the file.

`lr_12bit.ivf` is the same again at 12 bit (79 KB, `-pix_fmt yuv420p12le`), the
one depth where the Wiener filter's rounding constants change
(`InterRound0 = 5`, `InterRound1 = 9` instead of 3 and 11) and where the box
filter's bit depth shifts are at their widest. `lr_12bit.hash` +
`stream_lr_12bit` and `opencl_lr_12bit` are the 16 bit checks, both bit exact.

`lr_10bit.ivf` is the same clip coded at 10 bit (78 KB), with the same
`-enable-restoration 1` and `-pix_fmt yuv420p10le`. It exists because the self
guided filter scales its local sums by the bit depth
(`a = Round2(a, 2 * (BitDepth - 8))`, `d = Round2(b, BitDepth - 8)`) and at 8 bit
both shifts are zero - so an implementation that forgets them, or that builds the
offset `B` from the rounded `d` instead of the raw `b`, is bit exact on the 8 bit
fixture and wrong on this one. `lr_10bit.hash` holds the per frame hashes of the
16 bit samples (`--yuv16` on both sides, see above) and `stream_lr_10bit` checks
them; `opencl_lr_10bit` replays the frames through the device kernels. Both are
bit exact, against dav1d and through the kernels.

`tests/data/make_lr_clip.py` is that generator with the encoder wrapped around
it: run as `python3 tests/data/make_lr_clip.py tests/data/lr.ivf` it writes this
stream back byte for byte, `--pix-fmt 444` / `--pix-fmt 422` build the two
fixtures below from the same content, and `--bit-depth 10` / `--bit-depth 12`
build the deeper ones - the generator's 8 bit samples shifted up, so the picture
is the same picture at every depth. All five come out byte identical to the
files here.

`lr_444.ivf` and `lr_422.ivf` are that clip coded at 4:4:4 (196 KB, `-pix_fmt
yuv444p`, profile 1) and 4:2:2 (105 KB, `-pix_fmt yuv422p`, profile 2), with the
same encoder settings. They exist because restoration on a chroma plane that is
*not* subsampled the 4:2:0 way is a different path: the chroma unit grid only
halves for 4:2:0 (`lr_uv_shift` in the plan), so at 4:4:4 the chroma units are as
large as the luma ones - six per plane at 320x240 - and at 4:2:2 only their
horizontal step halves (two per plane) - and a 4:4:4 frame's chroma samples still
take their difference weighted mask from the *luma* prediction even though one
chroma sample covers exactly one luma sample. Both streams code both filters on
chroma: across the eight frames the 4:4:4 chroma planes hold 78 Wiener and 6 SGR
units, the 4:2:2 ones 22 Wiener and 1 SGR, and three of the 4:4:4 frames code the
chroma type per unit (`switchable`) instead of one filter for the plane. `av1gpu_probe`
prints that breakdown as a `restoration units:` line.

The 4:4:4 one caught exactly that mask bug: the decoder decided "rebuild the mask
from the luma predictions" from the subsampling flags, which is right for 4:2:0
and wrong at 4:4:4, where it built the mask from the chroma prediction under it.
`Av1gInterJob` carries the plane now (plan format 8) and the decision follows it.
`lr_444.hash` / `lr_422.hash` hold the per frame hashes, `stream_lr_444` /
`stream_lr_422` check them and `opencl_lr_444` / `opencl_lr_422` replay the same
frames through the device kernels; dav1d and ffmpeg agree with each other and
with us on both files, sample for sample.

```python
import sys
import numpy as np

rng = np.random.default_rng(5)
W, H, N = 320, 240, 8
yy, xx = np.mgrid[0:H, 0:W + 4 * N].astype(np.float64)
smooth = 110 + 35 * np.sin(xx / 47.0) * np.cos(yy / 31.0)
tex = rng.integers(0, 256, (H, W + 4 * N)).astype(np.float64)
tex = (tex * 3 + np.roll(tex, 1, 1) * 2 + np.roll(tex, -1, 1)) / 7 - 128
noise = rng.normal(0, 7, (H, W + 4 * N))
yyc, xxc = np.mgrid[0:H // 2, 0:W // 2 + 2 * N].astype(np.float64)
u = 128 + 50 * np.sin(xxc / 9.0) * np.cos(yyc / 13.0) + rng.normal(0, 4, (H // 2, W // 2 + 2 * N))
v = 128 + 40 * np.cos(xxc / 11.0) + rng.normal(0, 4, (H // 2, W // 2 + 2 * N))
out = bytearray()
for f in range(N):
    smooth_frame = f < N // 2
    lum = smooth + noise if smooth_frame else 130 + tex
    img = np.clip(lum[:, 3 * f:3 * f + W] + 2 * f, 0, 255).astype(np.uint8)
    if not smooth_frame:
        for k in range(6):
            x0, y0 = 10 + (k % 3) * 100 + 3 * f, 15 + (k // 3) * 115 + 2 * f
            img[y0:y0 + 55, x0:x0 + 70] = (40 + 35 * k) % 256
    sys.stdout.buffer.write(img.tobytes())
    sys.stdout.buffer.write(np.clip(u[:, f:f + W // 2], 0, 255).astype(np.uint8).tobytes())
    sys.stdout.buffer.write(np.clip(v[:, f:f + W // 2], 0, 255).astype(np.uint8).tobytes())
```

`lossless.ivf` is the lossless fixture: four frames of 160x120, 58 KB, coded at
`-lossless 1 -crf 0 -cpu-used 6 -lag-in-frames 0 -g 4 -tiles 1x1` with
`-enable-intrabc 0`. It exists because a frame that codes no quantiser at all
(`CodedLossless`, and `AllLossless` follows from it - the picture is not
superres'ed) has a *different frame header*: 5.9.2 skips `loop_filter_params`,
`cdef_params` and the restoration units entirely, so a parser that reads them
anyway desynchronises immediately and a decoder that then runs deblock, CDEF or
restoration on the picture changes samples a lossless stream promises are exact.
`-enable-intrabc 0` is deliberate: `allow_intrabc` skips the same three parameter
sets by the other route, and this fixture is meant to pin down the lossless one -
`intrabc.ivf` covers the block copy path. The content is flat panels on a hard
diagonal edge, a checkerboard finer than one sample per pixel, noise on every
frame (so the three inter frames code residual instead of a block copy) and a
three sample pan per frame.

`lossless.hash` holds the per frame hashes; `stream_lossless` checks them and
`opencl_lossless` replays the same plans through the device kernels. dav1d and
ffmpeg agree with us on every sample. `tests/data/make_lossless_clip.py` writes
this stream back byte for byte (`python3 tests/data/make_lossless_clip.py
tests/data/lossless.ivf`), and it refuses to finish unless the frame header it
produced really is the lossless one - no CDEF strengths, deblock levels at zero,
no restoration.

`odd_size.ivf` is the only fixture whose picture is not a multiple of eight in
either direction: six frames of 130x100, 16 KB, `-cpu-used 5 -crf 32
-lag-in-frames 0 -g 6 -tiles 1x1` with CDEF and restoration enabled. Both
dimensions matter. Rounding the frame up to a *multiple of four* mi columns
instead of eight (`MiCols = 2 * ((FrameWidth + 7) >> 3)`, 7.4.4) gives one mi
column too few at 130 wide, and the block loop of the last superblock column then
reads a different number of blocks than the encoder wrote - the frame decodes as
garbage, which nothing noticed while every fixture was a multiple of eight. The
height of 100 is a partial stripe (64 + 36) and leaves every frame predicted from
row bands whose neighbours are outside the picture but inside the coded grid,
which is what the wave scheduler's dependency tracking now covers as well; CDEF
copies whole 8x8 blocks and filters against a halo that reaches into that border.
`odd_size.hash` holds the per frame hashes, `stream_odd_size` checks them and
`opencl_odd_size` replays the same plans through the device kernels; dav1d and
ffmpeg agree with us on every sample.

`tests/data/make_odd_size.py` writes the stream back byte for byte
(`python3 tests/data/make_odd_size.py tests/data/odd_size.ivf`).

`edge_grid.ivf` is the fixture for the *third* rounding at the edge of a picture:
the loop filter (7.14) walks each plane up to the plane size rounded up to
*four* samples, which is neither the picture size nor the eight sample block
grid. Four frames of 130x66, 17 KB, `-cpu-used 4 -crf 40 -lag-in-frames 0 -g 4
-tiles 1x1` with CDEF and restoration enabled, so luma ends two rows and chroma
one row inside that grid (68 luma rows, 36 chroma rows) and the deringer and the
restoration filter read the difference. The content is a smoothed random texture
with fresh noise on every frame plus hard edged panels pinned to the bottom and
right edges - a smoother clip codes those last rows at level zero and then the
three roundings agree again, which is why `odd_size.ivf` (130x100, a multiple of
four in every plane) passes either way. `edge_grid.hash` holds the per frame
hashes, `stream_edge_grid` checks them and `opencl_edge_grid` replays the same
plans through the device kernels; dav1d and ffmpeg agree with us on every sample.

`tests/data/make_edge_grid.py` writes the stream back byte for byte
(`python3 tests/data/make_edge_grid.py tests/data/edge_grid.ivf`).

`tiles.ivf` is the only stream whose frames are cut into more than one tile: four
frames of 256x160, 72 KB, `-cpu-used 3 -crf 28 -lag-in-frames 0 -g 4 -tiles 2x2`
(libaom needs row-mt on, which ffmpeg's `-tiles` turns on, to honour more than
one tile row). The grid divides 256 into 128 + 128 samples across and 160 into
128 + 32 down, so the tile row boundary is not at the frame's middle and the
bottom two tiles are only 32 samples high - every one of the four tiles has a
neighbour from another tile on two sides. Four frames, the first a key frame, so
three code inter prediction across the tile edges.

It exists for the two things a single tile frame cannot show. First, what starts
over at a tile (5.11.2): its own range decoder over its own bytes, its own copy
of the frame's CDF tables, cleared above/left contexts, `CurrentQIndex` back to
`BaseQIdx`, `DeltaLF` zero and the restoration predictors back to their mid
points - while `MiSizes`, the references, the motion field, the segment ids, the
CDEF index grid and the restoration units stay the frame's, and the tile group
must cover all of them. Second, `is_inside` meaning the tile rather than the
frame, which two places had wrong: `AvailUChroma`/`AvailLChroma` for a block one
4x4 wide in a subsampled frame (they reach two mi units out, into the
neighbouring luma block's chroma) and `find_warp_samples`' `add_sample` for a
locally warped block on a tile edge. Putting either back leaves 18,447 and 3,676
of the frame's 245,760 bytes one level off against dav1d - and nothing
desynchronises, so both decode plausibly and are only visible by comparison.

The content is what makes the encoder produce the shapes the bugs need: a
smoothed random texture that the four frames pan and zoom over, i.e. global
motion, which is what makes libaom code local warped motion (699 blocks of the
four frames), fresh noise on every frame, hard edged panels on the right and
bottom edges so the loop filter, CDEF and restoration work there, a dark band
across the tile row boundary in the noisy area where the encoder spends 4x4
blocks - the size whose chroma is coded the sub-8x8 way - and gradients on
chroma.

`tiles.hash` holds the per frame hashes, `stream_tiles` checks them and
`opencl_tiles` replays the same plans through the device kernels; dav1d and
ffmpeg agree with us on every sample.

`tests/data/make_tiles_clip.py` writes the stream back byte for byte
(`python3 tests/data/make_tiles_clip.py tests/data/tiles.ivf`).

The three `aom_*` fixtures are the ones that are not 8 bit 4:2:0: twelve frames
of a 320x240 ramp each, encoded with libaom through ffmpeg's `libaom-av1` at
`-crf 40 -cpu-used 8 -lag-in-frames 0`.

| fixture | depth | chroma | bytes | why |
| --- | --- | --- | --- | --- |
| `aom_10bit_320x240.ivf` | 10 | 4:2:0 | 681 | 10 bit, and its first frame codes no loop filter level |
| `aom_444_10bit_320x240.ivf` | 10 | 4:4:4 | 8585 | chroma planes as large as luma |
| `aom_12bit_320x240.ivf` | 12 | 4:2:0 | 547 | the deepest path, `1 << (BitDepth - 8)` = 16 |

They exist because the deeper and wider paths are not the 8 bit ones: the
deblocking thresholds scale with `1 << (BitDepth - 8)`, the reconstruction has
more headroom, the grain range is wider, and a 4:4:4 frame's chroma planes are
the same size as its luma one - which is where `frame_to_yuv8` used to write past
the end of its buffer, a crash that nothing in the suite could reach before there
was a 4:4:4 fixture.

`aom_10bit_320x240.ivf`'s first frame is also the reason its clip is as flat as
it is: libaom codes a *zero* loop filter level there while still enabling the
per-reference deltas, and `decode_frame_wrapup` only runs the loop filter when a
luma level was coded, so a decoder that applies the deltas anyway filters edges a
reference decoder leaves alone (a few thousand samples, off by one or two). The
clips are generated, so all three streams can be rebuilt from nothing:

```
python3 tests/data/make_aom_clip.py tests/data/aom_10bit_320x240.ivf \
        --pix-fmt 420p10 --require-unfiltered
python3 tests/data/make_aom_clip.py tests/data/aom_444_10bit_320x240.ivf --pix-fmt 444p10
python3 tests/data/make_aom_clip.py tests/data/aom_12bit_320x240.ivf --pix-fmt 420p12
```

The script writes the clip, encodes it with the ffmpeg that `imageio-ffmpeg`
installs (`--ffmpeg` selects another), and with `--require-unfiltered` refuses to
finish unless the stream still codes a zero luma level with the deltas on - the
property that fixture exists for. `aom_*.hash` holds the per frame hashes of the
*samples*: `stream_aom_*` decodes with `--yuv16` and `--hash-check`, and
`opencl_aom_*` replays the same plans through the device kernels. Every sample of
every frame is equal to dav1d's, whose yuv muxer writes 16 bit little endian for
streams this deep.

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
