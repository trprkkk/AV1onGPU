# av1gpu_dshow - the DirectShow AV1 decoder filter

`av1gpu_dshow.dll` is a DirectShow source decoder for AV1: an input pin that
takes the AV1 elementary stream a splitter delivers (`MEDIASUBTYPE_AV1` /
`'AV01'`, one temporal unit per media sample) and an output pin that delivers
NV12. The decode runs on the av1gpu library's CUDA or OpenCL backend, so a GPU
without AV1 fixed-function hardware decodes AV1 while the CPU only parses and
submits. Bit exactness against the reference decoder is what the test suite
checks (see below).

Build (MSVC, x64):

```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target av1gpu_dshow
```

The DLL lands in `build/bin/Release/av1gpu_dshow.dll`.

## Using it in a player

### MPC-BE / MPC-HC

1. Options -> (Advanced) External Filters -> Add filter... -> Browse, and pick
   `av1gpu_dshow.dll` directly. Registration is not needed; the player loads
   the DLL by path.
2. Select the filter and set its merit to *Prefer* (MPC-BE) or tick *Prefer*
   (MPC-HC), so the splitter's AV1 output connects to it instead of to
   Microsoft's AV1 decoder (MFT) or LAV's software decoder.
3. With the internal LAV Video decoder still enabled, make sure the external
   filter is preferred over it for AV1; LAV keeps decoding everything else.

### PotPlayer

Filter Control (F5 -> Video -> ... or right click -> Filters) -> Add external
filter -> browse to the DLL -> set to *Prefer*.

### Registered system-wide (optional)

```
regsvr32 /n /i:user av1gpu_dshow.dll     registers, per user (no admin needed)
regsvr32 av1gpu_dshow.dll                registers for the whole machine (admin)
regsvr32 /u /n /i:user av1gpu_dshow.dll  unregisters
```

`DllRegisterServer` writes the COM class and a FilterMapper2 entry under
`HKCR`; when that is access denied (no admin) it falls back to
`HKCU\Software\Classes`, so a per-user registration works without elevation.
After registration the filter shows up by name in every player's filter list
and in GraphStudioNext / GraphEdit (`DirectShow Filters -> AV1onGPU AV1
Decoder`, CLSID `{B7E0E1C2-5A3D-4F6B-9C48-2A1D7E5F3B60}`).

### Graph builders

The output pin proposes NV12 only after the input pin is connected, because
the output geometry comes from the input's media type - the same order
`IGraphBuilder::Render` connects decoders in. Nothing else is special.

### VLC

VLC has no "external DirectShow filter" mechanism on Windows, so using this
decoder there means a VLC module that loads the decode core directly (the
`dsfilter/decode_core.*` files are deliberately library-only for that) - a
build of VLC with a small custom video decoder module, not a settings change.
Until that module exists, VLC keeps its own AV1 decoder (dav1d/libaom).

### Browsers

Chrome, Edge and Firefox on Windows only decode through their own pipelines
(Microsoft's AV1 MFT, or built-in dav1d) and cannot be pointed at an external
DirectShow filter, so this decoder does not hook into a browser as it is.
The realistic paths are the ones `docs/ROADMAP.md` section 6 tracks: a
VA-API-style driver shim for the browsers' hardware decode interface, or a
transcode proxy that re-encodes to a codec the GPU *does* have fixed function
hardware for (NVENC H.264/HEVC) and feeds the browser a normal media stream.
The library side of both is ready (device-side NV12 packing exists); the
integration is future work.

## v1 limitations

- **NV12 output only.** Frames whose sequence header says anything but 4:2:0
  (4:2:2, 4:4:4) or more than 8 bits per sample are dropped, with one message
  logged per stream. All the test fixtures that are 4:2:0 8 bit decode bit
  exact.
- **The input media type must carry dimensions** (a `VIDEOINFOHEADER` or
  `VIDEOINFOHEADER2` format block). Splitters that offer a partial type
  without a format block cannot connect.
- **Synchronous decode.** `IMemInputPin::Receive` decodes the temporal unit
  and delivers every frame it shows before returning, one allocator buffer at
  a time. Simple and zero latency; throughput is the GPU's problem (it is not
  the bottleneck).
- A seek (`BeginFlush`/`EndFlush` + `NewSegment`) resets the decoder state;
  the next key frame rebuilds the reference pictures. Reference frames from
  before the seek are not re-fetched (there is no index of them), so a seek
  target should be a key frame or the first frames after it - what splitters
  hand out after a flush anyway.

## Diagnostics

Set the environment variable `AV1GPU_DS_DEBUG=1` and run the player from a
shell; the filter logs connections, rejections, the backend it picked and
frame drops through `OutputDebugStringA` (visible in Sysinternals DbgView or
any attached debugger).

## Tests

- `test_ds_core` - the decode core without DirectShow: whole temporal units,
  units cut into 1/7/64 byte chunks (the incremental OBU parser), a reset and
  re-decode (the seek path), and - when a GPU is present - the backend the
  filter really picks against the CPU reference, byte for byte.
- `av1gpu_dsgraph` - the real COM surface: the DLL loaded the way a player
  does (`LoadLibrary` + `DllGetClassObject`), a fake splitter source, media
  type agreement, the allocator handshake, every fixture frame counted at a
  minimal sink, plus QueryAccept rejections, EndOfStream, a flush + NewSegment
  round trip and a Stop/Pause cycle.

```
ctest --test-dir build -C Release -R "ds_core|dsgraph" --output-on-failure
```
