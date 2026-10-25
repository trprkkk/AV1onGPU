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
