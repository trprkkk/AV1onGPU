# Test streams

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
