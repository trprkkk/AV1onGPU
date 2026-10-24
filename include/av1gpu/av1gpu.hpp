/*
 * AV1onGPU - public API
 *
 * A GPU accelerated AV1 reconstruction engine for GPUs that have no AV1 fixed
 * function decoder: everything an AV1 decoder needs after entropy decoding
 * (intra/inter prediction, inverse transform, deblocking, CDEF, film grain and
 * YUV->RGB) runs on compute kernels (OpenCL 2.0 or CUDA), with a bit exact CPU
 * reference implementation of the same maths.
 *
 * The library is deliberately small and dependency free. The caller does the
 * bitstream parsing (libaom, dav1d, an in-house parser ...) and hands over the
 * decoded *syntax* of a frame: block partition, prediction modes, motion
 * vectors, dequantised coefficients and filter parameters. That split is what
 * makes the project usable on old hardware - nothing on the GPU side needs to
 * understand an AV1 bitstream, and the same syntax drives every backend.
 *
 *   std::string error;
 *   std::unique_ptr<av1gpu::Decoder> dec = av1gpu::Decoder::create({}, &error);
 *   av1gpu::Frame out;
 *   int slot = dec->decode_reference(syntax, &out, &error);
 *
 * Copyright (c) 2026 AV1onGPU contributors. BSD-3-Clause.
 */
#ifndef AV1GPU_AV1GPU_HPP
#define AV1GPU_AV1GPU_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace av1gpu {

/* ------------------------------------------------------------------ video */

enum class ColorMatrix : int {
  Bt601 = 0,
  Bt709 = 1,
  Bt2020 = 2,
  Identity = 3,
};

enum class ColorRange : int {
  Limited = 0,
  Full = 1,
};

enum class BackendKind : int {
  Auto = 0,
  Cpu = 1,
  OpenCL = 2,
  CUDA = 3,
};

/* "cpu" / "opencl" / "cuda" / "auto" */
const char* to_string(BackendKind kind);
BackendKind backend_kind_from_string(const std::string& name);

struct FrameGeometry {
  int width = 0;
  int height = 0;
  int bit_depth = 8;
  int ss_x = 1;
  int ss_y = 1;
};

/* ------------------------------------------------------------------ frame */

/* One decoded picture: three 16 bit planes with 32 sample aligned strides.
 *
 * Samples are stored 16 bits wide whatever the bit depth, because every stage
 * of the pipeline (prediction, residual add, loop filters) works on the same
 * intermediate precision and the device kernels are compiled for one sample
 * type. */
class Frame {
 public:
  Frame() = default;
  Frame(int width, int height, int bit_depth = 8, int ss_x = 1, int ss_y = 1) {
    allocate(width, height, bit_depth, ss_x, ss_y);
  }

  /* (Re)allocates the planes and clears them to zero. */
  void allocate(int width, int height, int bit_depth, int ss_x, int ss_y);

  /* True when the frame already has this geometry and enough storage. */
  bool matches(int width, int height, int bit_depth, int ss_x, int ss_y) const;

  void fill(uint16_t value);

  int plane_width(int plane) const;
  int plane_height(int plane) const;
  int max_value() const { return (1 << bit_depth) - 1; }
  bool empty() const { return width <= 0 || height <= 0; }

  uint16_t* plane(int p) { return storage[p].data(); }
  const uint16_t* plane(int p) const { return storage[p].data(); }

  /* Public on purpose: the backends copy rows straight into these, and the
   * tests compare pixels without going through accessors. */
  int width = 0;
  int height = 0;
  int bit_depth = 8;
  int ss_x = 1;
  int ss_y = 1;
  int stride[3] = {0, 0, 0};
  std::vector<uint16_t> storage[3];
};

/* ----------------------------------------------------------------- syntax */

/* Prediction modes, numbered exactly like the AV1G_PRED_* constants the
 * kernels switch on. */
enum class PredictionMode : int {
  Dc = 0,
  Vertical = 1,
  Horizontal = 2,
  Paeth = 3,
  Smooth = 4,
  SmoothVertical = 5,
  SmoothHorizontal = 6,
  ChromaFromLuma = 7,
};

/* Subpel filter classes, numbered like AV1G_FILT_*. */
enum class FilterType : int {
  Regular = 0,
  Smooth = 1,
  Sharp = 2,
  Bilinear = 3,
};

/* Compound prediction, numbered like AV1G_COMPOUND_*. */
enum class CompoundType : int {
  None = 0,
  Average = 1,
  Distance = 2,
};

/* One transform block inside a BlockSyntax.  Coefficients live in the frame's
 * shared pool (FrameSyntax::coefficients) at `coef_offset`, in raster order. */
struct TransformBlock {
  int x = 0; /* position inside the prediction block */
  int y = 0;
  int w = 0;
  int h = 0;
  int coef_offset = 0;
  int eob = 0; /* 0 = the whole block is zero, skip the transform */
  bool adst_horizontal = false;
  bool adst_vertical = false;
};

/* One prediction block of one plane. Chroma blocks are addressed in their own
 * (subsampled) sample grid. */
struct BlockSyntax {
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int plane = 0;
  int bit_depth = 8;

  bool is_intra = true;

  /* intra */
  PredictionMode mode = PredictionMode::Dc;
  bool have_left = true;
  bool have_above = true;
  int cfl_alpha = 0; /* Q3, only used by ChromaFromLuma */

  /* inter */
  int ref[2] = {-1, -1};  /* indices into FrameSyntax::reference_slots */
  int mv[2][2] = {{0, 0}, {0, 0}}; /* 1/8 pel, relative to the block */
  FilterType filter = FilterType::Regular;
  CompoundType compound = CompoundType::None;
  int blend[2] = {32, 32}; /* Q6 weights for distance weighted compound */

  /* When set, the transform blocks carry raw residual samples
   * (FrameSyntax::residual) instead of an inverse transform. Used for block
   * sizes that have no transform, and by tests that want to inject a known
   * residual. */
  bool residual_provided = false;

  std::vector<TransformBlock> transforms;
};

/* One deblocking edge segment. `x`/`y` address the sample on the far side of
 * the edge (q0); the segment itself runs along the edge for `length` samples,
 * a multiple of 4. */
struct DeblockEdge {
  int plane = 0;
  int x = 0;
  int y = 0;
  bool vertical = true;
  int length = 0;
  int level = 0;
  int filter_len = 8; /* 4, 6 or 8 */
};

/* One 8x8 CDEF block (luma grid; chroma blocks are selected per plane). */
struct CdefBlock {
  int plane = 0;
  int x = 0;
  int y = 0;
  int pri_strength = 0;
  int sec_strength = 0;
  int damping = 3;
  int dir = 0;
};

/* Film grain parameters. The template is generated from the seed by the
 * engine, in the same pseudo random sequence the AV1 spec uses. */
struct GrainSyntax {
  int seed = 0;
  int overlap = 1;
  int template_size = 73;
  int lut[32] = {0}; /* Q6 gain per luma bucket */
};

struct ColorSyntax {
  ColorMatrix matrix = ColorMatrix::Bt709;
  ColorRange range = ColorRange::Limited;
};

/* Everything the decoder needs to reconstruct one frame. */
struct FrameSyntax {
  int width = 0;
  int height = 0;
  int bit_depth = 8;
  int ss_x = 1;
  int ss_y = 1;
  int base_q_idx = 0;
  int sharpness = 0;

  /* Short term reference slots the inter blocks index into. */
  std::vector<int> reference_slots;

  std::vector<BlockSyntax> blocks;

  /* Shared pools indexed by TransformBlock::coef_offset. */
  std::vector<int32_t> coefficients;
  std::vector<int32_t> residual;

  std::vector<DeblockEdge> deblock;
  std::vector<CdefBlock> cdef;

  bool apply_grain = false;
  GrainSyntax grain;
  ColorSyntax color;
};

/* ------------------------------------------------------------------ stats */

/* Timings of the last decoded frame, in milliseconds. The split matters for
 * this project: on a working GPU setup `reconstruct_ms` and `filter_ms` are
 * device time while the host only pays for the submission. */
struct DecodeStats {
  int blocks = 0;
  int waves = 0;
  int kernel_launches = 0;
  double reconstruct_ms = 0.0;
  double filter_ms = 0.0;
  double readback_ms = 0.0;
  double total_ms = 0.0;
};

/* ----------------------------------------------------------------- device */

struct DeviceInfo {
  BackendKind kind = BackendKind::Cpu;
  std::string name;
  std::string vendor;
  std::string version;
  std::string detail;
  bool available = false;
  bool unified_memory = false;
  uint64_t global_mem = 0;
  int compute_units = 0;
};

struct BackendOptions {
  BackendKind kind = BackendKind::Auto;
  int device_index = 0;
  /* Fall back to the CPU reference when the requested GPU backend cannot be
   * initialised. Set to false to make a missing GPU a hard error. */
  bool allow_cpu_fallback = true;
  /* Size of the dependency grid cell, in luma samples, used to schedule the
   * wavefront. Bigger cells serialise more (and therefore launch more, smaller
   * waves); 16 matches the smallest block the engine is typically fed, so a
   * block rarely shares a cell with an independent neighbour. */
  int dependency_cell = 16;
  /* Append device/driver/compiler diagnostics to error strings. */
  bool verbose = false;
};

/* Every device this build can use, the CPU reference first. */
std::vector<DeviceInfo> enumerate_devices();

/* Human readable summary of what this build supports. */
std::string build_info();

/* ---------------------------------------------------------------- metrics */

struct Psnr {
  double y = 0.0;
  double u = 0.0;
  double v = 0.0;
};

/* Per plane PSNR of two frames of the same geometry (99 dB when identical). */
Psnr frame_psnr(const Frame& a, const Frame& b);

/* Packs a frame into planar YUV420p / 4:4:4 at 8 bits per sample. */
void frame_to_yuv8(const Frame& frame, std::vector<uint8_t>* out);

/* ------------------------------------------------------------------ decode */

class Decoder {
 public:
  /* Picks a backend. With BackendKind::Auto the GPU backends are tried in
   * order and the CPU reference is the last resort. Returns null and fills
   * `error` when nothing usable is available. */
  static std::unique_ptr<Decoder> create(const BackendOptions& options,
                                         std::string* error = nullptr);

  virtual ~Decoder();

  Decoder(const Decoder&) = delete;
  Decoder& operator=(const Decoder&) = delete;

  virtual const DeviceInfo& device() const = 0;
  virtual const BackendOptions& options() const = 0;
  virtual const DecodeStats& stats() const = 0;

  /* Decodes a frame and leaves it resident on the device: the returned slot
   * can be used as a reference by later frames without a host round trip.
   * Returns -1 on failure. */
  virtual int decode_reference(const FrameSyntax& syntax, Frame* out,
                               std::string* error = nullptr) = 0;

  /* Convenience wrapper: decode and drop the device copy. */
  virtual bool decode(const FrameSyntax& syntax, Frame* out,
                      std::string* error = nullptr);

  virtual void release_reference(int slot) = 0;

  /* YUV -> RGB24 with the same arithmetic the kernels use. */
  virtual bool to_rgb(int slot, ColorMatrix matrix, ColorRange range,
                      std::vector<uint8_t>* rgb, std::string* error = nullptr) = 0;

  virtual void reset() = 0;

 protected:
  Decoder() = default;
};

/* Same as Decoder::create; kept for call sites that want to read as "give me
 * the best one". */
std::unique_ptr<Decoder> create_best_decoder(const BackendOptions& options,
                                             std::string* error = nullptr);

} /* namespace av1gpu */

#endif /* AV1GPU_AV1GPU_HPP */
