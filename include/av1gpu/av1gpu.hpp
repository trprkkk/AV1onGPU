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

/* Prediction modes, numbered exactly like the AV1G_PRED_* constants the kernels
 * switch on (which are the specification's own numbering, plus chroma from luma
 * as 13). The eight directional modes are contiguous so that the mode to angle
 * table can be indexed directly. */
enum class PredictionMode : int {
  Dc = 0,
  Vertical = 1,
  Horizontal = 2,
  D45 = 3,
  D135 = 4,
  D113 = 5,
  D157 = 6,
  D203 = 7,
  D67 = 8,
  Smooth = 9,
  SmoothVertical = 10,
  SmoothHorizontal = 11,
  Paeth = 12,
  ChromaFromLuma = 13,
};

/* Transform types, numbered like Tx_Type in the specification. */
enum class TransformType : int {
  DctDct = 0,
  AdstDct = 1,
  DctAdst = 2,
  AdstAdst = 3,
  FlipadstDct = 4,
  DctFlipadst = 5,
  FlipadstFlipadst = 6,
  AdstFlipadst = 7,
  FlipadstAdst = 8,
  Idtx = 9,
  VDct = 10,
  HDct = 11,
  VAdst = 12,
  HAdst = 13,
  VFlipadst = 14,
  HFlipadst = 15,
};

/* The filter intra (recursive) prediction modes, numbered like
 * Intra_Filter_Mode in the specification. */
enum class FilterIntraMode : int {
  Dc = 0,
  V = 1,
  H = 2,
  D157 = 3,
  Paeth = 4,
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
  /* the two predictions are blended with the weights the frame distances give
   * (BlockSyntax::blend) */
  Distance = 2,
  /* a masked blend: a wedge shape, or weights that follow how far the two
   * predictions are apart (BlockSyntax::wedge_index and ::mask_sign) */
  Wedge = 3,
  DiffWtd = 4,
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
  TransformType tx_type = TransformType::DctDct;
};

/* One overlapped block motion compensation strip of an inter block: the top
 * (or left) strip of the block predicted from a neighbouring block's motion
 * vector and blended in with a tapered mask. `x`/`y` are relative to the
 * block's top left sample in the plane, because a strip is a region of the
 * block, not a block of its own. */
struct ObmcCandidate {
  int pass = 0;   /* 0: a strip from above, 1: from the left */
  int x = 0;
  int y = 0;
  int w = 0;
  int h = 0;
  int mv[2] = {0, 0};   /* the neighbour's vector, 1/8 luma samples */
  FilterType filter = FilterType::Regular;   /* the neighbour's filters */
  FilterType filter_v = FilterType::Regular;
  int ref = -1;         /* index into FrameSyntax::reference_slots */
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
  /* Intra block copy (7.11.5): an `is_inter` block that predicts from the frame
   * being decoded, at an integer offset. It has no reference frame at all
   * (`ref[0]` is -1, i.e. the frame itself), always uses the bilinear filter,
   * and its intra neighbours are the ones its *source* rectangle has - so it is
   * scheduled like an intra block and reads whole reconstructed samples. */
  bool intrabc = false;

  /* intra */
  PredictionMode mode = PredictionMode::Dc;
  bool have_left = true;
  bool have_above = true;
  /* Availability of the samples immediately above the block's top right
   * corner and immediately left of its bottom left corner. The specification
   * answers this with BlockDecoded, a per-4x4 map of what has already been
   * reconstructed, which is *not* the same as "inside the frame": at a
   * superblock edge the samples exist but have not been decoded yet.
   * -1 means "derive it from the wavefront schedule" (see build_frame_plan). */
  int have_above_right = -1;
  int have_below_left = -1;
  int angle_delta = 0; /* -3..3, in ANGLE_STEP = 3 degree steps */
  /* The specification's filterType: 1 when a neighbouring block was predicted
   * with a smooth mode, which switches the intra edge filter's strength table.
   * Set by whoever builds the frame; it cannot be derived from the block. */
  int filter_type = 0;
  bool use_filter_intra = false;
  FilterIntraMode filter_intra_mode = FilterIntraMode::Dc;
  int cfl_alpha = 0; /* Q3, only used by ChromaFromLuma */

  /* Lossless coding of this block (segment feature or coded lossless): the
   * transform becomes a Walsh-Hadamard and both shifts are dropped. */
  bool lossless = false;

  /* Palette prediction (screen content): the block's samples are chosen from
   * `palette_size` colours by a coded colour map, instead of being predicted
   * from the neighbours. `palette` holds this plane's colours, `palette_off`
   * the corner of this transform block inside FrameSyntax::palette_map, whose
   * rows are `palette_stride` bytes - the coding block's own width. A palette
   * block still carries residual if the bitstream codes one. */
  bool use_palette = false;
  int palette_size = 0;
  int palette[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int palette_off = 0;
  int palette_stride = 0;

  /* inter */
  int ref[2] = {-1, -1};  /* indices into FrameSyntax::reference_slots */
  /* Motion vectors in 1/8 *luma* samples, exactly as the bitstream codes them
   * (the specification's Mv). Chroma vectors are not stored scaled: the kernel
   * halves them per subsampled axis, which is also where the 1/16 subpel phase
   * comes from. */
  int mv[2][2] = {{0, 0}, {0, 0}};
  /* The subpel filter class per direction. `filter` is the horizontal one, the
   * specification's interp_filter[1]; interp_filter[0] is the vertical one, so
   * the two are stored the way the kernel uses them rather than in bitstream
   * order. A frame with dual_filter off sets both to the same class. */
  FilterType filter = FilterType::Regular;
  FilterType filter_v = FilterType::Regular;
  /* 0 translation, 1 OBMC, 2 local warped motion (the specification's
   * MotionMode), kept for diagnostics and dumps. */
  int motion_mode = 0;
  CompoundType compound = CompoundType::None;
  /* Distance weighted compound weights, Q4 and summing to 16 (the
   * specification's FwdWeight/BckWeight). */
  int blend[2] = {8, 8};
  /* Masked compound: the wedge shape (0..15, an index into the wedge codebook)
   * and the mask sign - which half of the wedge, or which way the difference
   * weighted mask leans. */
  int wedge_index = 0;
  int mask_sign = 0;
  /* Inter/intra compound (7.11.4): this block's inter prediction is blended
   * with an intra prediction of the same region, predicted from the
   * *reconstructed* neighbours of the frame being decoded. `interintra_mode` is
   * the specification's interintra_mode (0 DC, 1 V, 2 H, 3 smooth); with
   * `interintra_wedge` the blend uses the wedge `wedge_index` of the codebook
   * instead of the mode's own taper, and the mask sign is always zero. Such a
   * block has one reference, so `compound` stays None. */
  bool interintra = false;
  int interintra_mode = 0;
  bool interintra_wedge = false;
  /* Global warped motion: this reference is predicted by projecting the
   * sample's position through the frame's affine model instead of displacing
   * it by the block's vector (`warp` per reference list). The parameters are
   * the frame header's global motion model at WARPEDMODEL_PREC_BITS precision;
   * a compound block can warp both halves, each with its own model. */
  bool warp[2] = {false, false};
  int32_t warp_params[2][6] = {{0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0}};
  /* The same decision taken for the block's *luma* region. A chroma region is
   * half the size, so it can fall under the specification's 8x8 floor and be
   * predicted from its vector while the luma region above it is warped. The
   * difference weighted mask is built on the luma grid (7.11.3.12 only runs for
   * plane 0 and 7.11.3.14 averages that array for the chroma samples), so a
   * chroma job still needs the luma decision to reproduce the mask. */
  bool mask_warp[2] = {false, false};
  /* Overlapped block motion compensation strips, in the order the
   * specification blends them: the ones from above first, then from the left. */
  std::vector<ObmcCandidate> obmc;

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
  /* The specification's filterSize: the maximum number of taps the filter may
   * use (4, 8 or 16). The filter itself picks 4, 6 or 8 taps per side from it
   * and from the plane, so this is not the tap count. */
  int filter_size = 8;
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

/* FrameRestorationType (5.9.20). */
enum RestorationType {
  kRestoreNone = 0,
  kRestoreSwitchable = 1,
  kRestoreWiener = 2,
  kRestoreSgrproj = 3,
};

/* The constants of the loop restoration process (7.17). They are here rather
 * than in the kernels because the front end needs the coefficient bounds while
 * it decodes the unit syntax. */
constexpr int kWienerTaps = 3;              /* coded coefficients per pass  */
constexpr int kSgrParamsBits = 4;           /* Sgr_Params index bits        */
constexpr int kSgrProjPrjBits = 7;          /* SGRPROJ_PRJ_BITS             */
constexpr int kSgrProjRstBits = 4;          /* SGRPROJ_RST_BITS             */
constexpr int kSgrProjMtableBits = 20;      /* SGRPROJ_MTABLE_BITS          */
constexpr int kSgrProjSgrBits = 8;          /* SGRPROJ_SGR_BITS             */
constexpr int kSgrProjRecipBits = 12;       /* SGRPROJ_RECIP_BITS           */
constexpr int kSgrProjSubexpK = 4;          /* the subexp k of the weights  */
/* Wiener_Taps_Min / Max / K: the bounds and the subexponential k of each coded
 * coefficient. */
constexpr int kWienerTapsMin[3] = {-5, -23, -17};
constexpr int kWienerTapsMax[3] = {10, 8, 46};
constexpr int kWienerTapsK[3] = {1, 2, 3};
/* Sgrproj_Xqd_Min / Max: the bounds of the two projection weights. */
constexpr int kSgrXqdMin[2] = {-96, -32};
constexpr int kSgrXqdMax[2] = {31, 95};
/* Wiener_Taps_Mid / Sgrproj_Xqd_Mid: what a tile's unit syntax predicts from
 * before it has read one. */
constexpr int kWienerTapsMid[3] = {3, -7, 15};
constexpr int kSgrXqdMid[2] = {-32, 31};
/* Sgr_Params: (radius, epsilon) per pass, sixteen sets. */
constexpr int kSgrParams[16][4] = {
    {2, 12, 1, 4},  {2, 15, 1, 6},   {2, 18, 1, 8},   {2, 21, 1, 9},
    {2, 24, 1, 10}, {2, 29, 1, 11},  {2, 36, 1, 12},  {2, 45, 1, 13},
    {2, 56, 1, 14}, {2, 68, 1, 15},  {0, 0, 1, 5},    {0, 0, 1, 8},
    {0, 0, 1, 11},  {0, 0, 1, 14},   {2, 30, 0, 0},   {2, 75, 0, 0},
};

/* One loop restoration unit (5.11.46): the filter it selected and the
 * parameters it coded. The units are stored densely, one entry per unit of the
 * plane, so the process can index them the way the specification's
 * LrType[plane][unitRow][unitCol] is indexed.
 *
 * `wiener` holds the coefficients as the syntax codes them - three per pass,
 * the vertical pass first - and `sgr_set`, `sgr_xqd` the self guided table
 * index and its two projection weights. */
struct RestorationUnit {
  /* 0 none, 1 Wiener, 2 self guided. A frame whose FrameRestorationType is
   * RESTORE_SWITCHABLE stores the unit's own choice here; one that codes a
   * single filter stores that filter or none. */
  int type = 0;
  int wiener[2][3] = {{0, 0, 0}, {0, 0, 0}};
  int sgr_set = 0;
  int sgr_xqd[2] = {0, 0};
};

/* film_grain_params() (5.9.30): the parameters of the film grain synthesis
 * filter (7.18.3). Everything the filter needs is here, so a frame carries its
 * own copy - and a reference slot keeps the one its picture was decoded with,
 * because a later frame may re-use it (update_grain == 0).
 *
 * `ar_coeffs_*` hold the signed coefficients, i.e. the coded value minus 128. */
struct FilmGrainParams {
  bool apply = false;
  int seed = 0;
  /* The scaling curve: a list of (sample value, grain strength) points, up to
   * sixteen of them, strictly increasing in value. */
  int num_y_points = 0;
  int point_y_value[16] = {0};
  int point_y_scaling[16] = {0};
  bool chroma_scaling_from_luma = false;
  int num_cb_points = 0;
  int point_cb_value[16] = {0};
  int point_cb_scaling[16] = {0};
  int num_cr_points = 0;
  int point_cr_value[16] = {0};
  int point_cr_scaling[16] = {0};
  int grain_scaling_minus_8 = 0;
  /* The auto-regressive filter that shapes the noise: a lag of 0 to 3 samples
   * and one coefficient per position (plus, for chroma, one that reads the
   * luma grain). */
  int ar_coeff_lag = 0;
  int ar_coeffs_y[24] = {0};
  int ar_coeffs_cb[25] = {0};
  int ar_coeffs_cr[25] = {0};
  int ar_coeff_shift_minus_6 = 0;
  int grain_scale_shift = 0;
  /* How the chroma strength follows the luma and chroma values. */
  int cb_mult = 0;
  int cb_luma_mult = 0;
  int cb_offset = 0;
  int cr_mult = 0;
  int cr_luma_mult = 0;
  int cr_offset = 0;
  bool overlap_flag = false;
  bool clip_to_restricted_range = false;
};

struct ColorSyntax {
  ColorMatrix matrix = ColorMatrix::Bt709;
  ColorRange range = ColorRange::Limited;
};

/* Everything the decoder needs to reconstruct one frame. */
struct FrameSyntax {
  /* The display size: what the frame decodes to and what an output frame holds.
   * A superres frame codes its tiles at `coded_width`, which is narrower, and
   * the upscaling process resamples the result to `width` before the output and
   * before the picture goes into the reference pool. */
  int width = 0;
  int coded_width = 0; /* 0: the frame is not superres, `width` is what it codes */
  int height = 0;
  int bit_depth = 8;
  int ss_x = 1;
  int ss_y = 1;
  int base_q_idx = 0;
  int sharpness = 0;
  /* The sequence header's enable_intra_edge_filter: when it is off, intra
   * prediction uses the raw edge samples. */
  bool enable_intra_edge_filter = true;

  /* Short term reference slots the inter blocks index into. */
  std::vector<int> reference_slots;
  /* The display size of each reference type (indexed by RefFrameType - 1), as
   * the reference pool has it stored. Reference scaling needs it: a superres
   * frame is coded narrower than the references it predicts from, so a motion
   * vector addresses a larger picture than the one being reconstructed. Zero
   * means "the same size as this frame", which is what a stream without
   * superres always has. */
  int reference_width[7] = {0, 0, 0, 0, 0, 0, 0};
  int reference_height[7] = {0, 0, 0, 0, 0, 0, 0};

  std::vector<BlockSyntax> blocks;

  /* Shared pools indexed by TransformBlock::coef_offset. */
  std::vector<int32_t> coefficients;
  std::vector<int32_t> residual;
  /* Palette colour maps of every palette block in the frame, one byte per
   * sample, indexed by BlockSyntax::palette_off. Empty unless the frame uses
   * screen content tools. */
  std::vector<uint8_t> palette_map;

  std::vector<DeblockEdge> deblock;
  std::vector<CdefBlock> cdef;

  /* Loop restoration (7.17). `restoration_type` is the frame's
   * FrameRestorationType per plane, and the units of each plane are stored row
   * major with `restoration_unit_cols[plane]` entries per row. */
  int restoration_type[3] = {0, 0, 0};
  int restoration_unit_log2[3] = {0, 0, 0};
  int restoration_unit_cols[3] = {0, 0, 0};
  std::vector<RestorationUnit> restoration[3];

  /* Film grain: `grain.apply` is the syntax's apply_grain. */
  FilmGrainParams grain;
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

/* Packs a frame into planar YUV420p / 4:4:4 at 8 bits per sample. A stream
 * deeper than 8 bits is shifted down; the samples keep their own plane size. */
void frame_to_yuv8(const Frame& frame, std::vector<uint8_t>* out);

/* The same packing with the samples *as they are*, little endian 16 bit, so a
 * 10 or 12 bit stream can be compared bit for bit instead of after a shift. */
void frame_to_yuv16(const Frame& frame, std::vector<uint8_t>* out);

/* Packs a 4:2:0 frame into NV12: `height` rows of luma, then one row of
 * interleaved Cb/Cr pairs per two luma rows, every row `pitch` bytes wide. That
 * is the layout a VA-API surface has - its pitch is not the picture width, which
 * is why it is a parameter - and it is what a compositor imports without a copy.
 * Bytes between the picture and the pitch are written as zero. NV12 is an 8 bit
 * format, so a deeper stream is shifted down like frame_to_yuv8 does. Returns
 * false when the frame cannot be packed: it is not 4:2:0, `pitch` is odd, or
 * `pitch` is narrower than a chroma row. */
bool frame_to_nv12(const Frame& frame, size_t pitch, std::vector<uint8_t>* out);

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

  /* Like decode_reference, but stores the picture into `slot`, which a
   * pipelined caller reserved earlier through reserve_reference. A negative
   * `slot` decodes and drops the device copy; -1 is returned either way. */
  virtual int decode_reference_reserved(const FrameSyntax& syntax, int slot, Frame* out,
                                        std::string* error = nullptr) {
    (void)syntax;
    (void)slot;
    (void)out;
    if (error) *error = "reserved-slot decode is not supported";
    return -1;
  }

  /* Reserves a reference slot before its frame is decoded: a caller that plans
   * frames ahead of their decode (a pipeline) can name the slot in the plan of
   * the *next* frame without waiting for this one's kernels. The next
   * decode_reference call on this decoder uses the returned slot; -1 means the
   * backend cannot reserve (the caller then decodes synchronously). */
  virtual int reserve_reference() { return -1; }

  /* The frame planner (the wavefront scheduler and the job arrays) is the
   * CPU-heavy half of a decode, and it does not touch the backend. A pipelined
   * caller builds the plan on its own thread - `plan` is an opaque handle owned
   * by the decoder until it is handed to decode_reference_planned or dropped
   * with discard_frame_plan - and decodes it on another, so the plan of frame
   * N+1 overlaps the kernels of frame N. Returns false and fills `error` on
   * failure; backends that do not support it leave the default, which fails. */
  virtual bool plan_frame(const FrameSyntax& syntax, void** plan,
                          std::string* error = nullptr) {
    (void)syntax;
    (void)plan;
    if (error) *error = "pre-planning is not supported";
    return false;
  }

  /* Decodes a plan built by plan_frame into `slot` (a reserved reference, or a
   * negative slot to decode and drop). The plan is consumed either way. */
  virtual int decode_reference_planned(void* plan, int slot, Frame* out,
                                       std::string* error = nullptr) {
    (void)plan;
    (void)slot;
    (void)out;
    if (error) *error = "pre-planned decode is not supported";
    return -1;
  }

  /* Drops a plan that will never be decoded. Null is ignored. */
  virtual void discard_frame_plan(void* plan) { (void)plan; }

  /* True when plan_frame/decode_reference_planned are implemented (the engine
   * is; a backend wrapper that decodes directly is not). A pipeline only
   * pre-plans against a decoder that says yes. */
  virtual bool supports_preplanning() const { return false; }

  /* Convenience wrapper: decode and drop the device copy. */
  virtual bool decode(const FrameSyntax& syntax, Frame* out,
                      std::string* error = nullptr);

  virtual void release_reference(int slot) = 0;

  /* YUV -> RGB24 with the same arithmetic the kernels use. */
  virtual bool to_rgb(int slot, ColorMatrix matrix, ColorRange range,
                      std::vector<uint8_t>* rgb, std::string* error = nullptr) = 0;

  /* The same picture as NV12, packed *on the device*, into rows `pitch` bytes
   * wide - the last step of a zero copy path, so the planes never come back to
   * the host. Same buffer size and padding rules as frame_to_nv12, and the same
   * refusal for a frame that is not 4:2:0. */
  virtual bool to_nv12(int slot, size_t pitch, std::vector<uint8_t>* nv12,
                       std::string* error = nullptr) = 0;

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
