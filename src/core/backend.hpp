/*
 * AV1onGPU - internal backend interface
 *
 * The engine (src/core/engine.cpp) turns a FrameSyntax into a FramePlan: flat,
 * POD job arrays that describe every unit of work in the frame. A backend is
 * then free to execute that plan any way it likes (loops on the CPU, OpenCL
 * kernels, CUDA kernels) but must produce exactly the same pixels, because all
 * of them run the same arithmetic from kernels/av1gpu_kernels.h.
 */
#ifndef AV1GPU_CORE_BACKEND_HPP
#define AV1GPU_CORE_BACKEND_HPP

#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "av1gpu_kernels.h"

namespace av1gpu {
namespace internal {

struct FrameGeometry {
  int width = 0;
  int height = 0;
  int bit_depth = 8;
  int ss_x = 1;
  int ss_y = 1;
};

/* Geometry plus the strides the backend actually uses.  A frame lives in one
 * flat buffer (the three planes back to back), so every job carries a plain
 * sample offset and a backend needs a single allocation per frame. */
struct PlanLayout {
  /* The *coded* geometry: superres codes a frame narrower than it displays it,
   * and every reconstruction job works on the narrow grid. */
  FrameGeometry frame;
  int plane_w[3] = {0, 0, 0};  /* visible coded samples */
  int plane_h[3] = {0, 0, 0};
  /* The *display* geometry: what the upscaling process produces, and what the
   * output frame and the reference pool hold. Equal to the coded size for a
   * stream without superres. */
  int out_width = 0;
  int out_height = 0;
  int out_w[3] = {0, 0, 0};
  int out_h[3] = {0, 0, 0};
  int stride[3] = {0, 0, 0};   /* row pitch, including the edge padding */
  int plane_off[3] = {0, 0, 0};
  size_t buffer_size = 0;

  /* True when the frame is coded narrower than it is displayed, i.e. when the
   * upscaling process has to run before the output and the reference store. */
  bool upscaled() const { return out_width != plane_w[0]; }
};

/* Edge padding, in luma samples.
 *
 * AV1 codes blocks on a 4x4 grid that is rounded up to the frame size, so the
 * blocks along the right and bottom edges extend past the visible picture: a
 * 32x32 block whose top left corner is the last 4x4 of the frame is coded and
 * predicted in full, and only the part inside the picture is ever displayed.
 * Every reference decoder therefore allocates its planes with a border (libaom
 * calls it the border, dav1d allocates aligned pictures), and the prediction
 * kernels here write the full block without clipping - all three of them,
 * identically, which is what keeps them bit exact against each other.
 *
 * 64 samples covers the largest block AV1 codes (a 64x64 superblock with a
 * 64x64 block; 128 for 128x128 superblocks, which this decoder does not
 * enable), and a chroma plane inherits it divided by the subsampling, so a
 * chroma block can never leave the padded area either. */
constexpr int kPlanePadding = 64;

/* The host half of the film grain synthesis filter (7.18.3): the generated
 * grain, the noise image cut from it, and the three scaling tables.
 *
 * The synthesis is only *partly* parallel: the auto-regressive filter and the
 * block stitching are sequential, so they run on the host once per frame, and
 * what reaches the kernels is the finished noise image plus the per sample
 * blend. That keeps the GPU path a pure map over samples - and, because the
 * host is the only place the sequential part lives, all three backends agree by
 * construction. */
struct GrainNoise {
  /* The parameters the noise was built from, in the kernels' layout. */
  Av1gGrainParams params;
  /* The three noise images, back to back, one 16 bit sample each (the grain
   * range is +/-128 at 8 bit, so a deeper stream needs more than a byte): the plane
   * offsets and row pitches the jobs index them with. */
  std::vector<short> noise;
  int noise_off[3] = {0, 0, 0};
  int noise_stride[3] = {0, 0, 0};
  /* ScalingLut[ plane ][ 8 bit sample value ], the curve each plane's noise is
   * scaled by. */
  int lut[AV1G_GRAIN_LUT_ENTRIES] = {0};
  /* The range the blended result is clipped to: the restricted range when
   * clip_to_restricted_range is set, the full range otherwise. */
  int min_value = 0;
  int max_luma = 255;
  int max_chroma = 255;
};

/* Builds the grain and the noise image for a frame whose *display* size is
 * `luma_w` x `luma_h` (superres included: the grain is added after the
 * upscaling). A frame whose parameters give chroma no grain builds no chroma
 * noise. */
void build_grain(const FilmGrainParams& params, int bit_depth, int ss_x, int ss_y,
                 bool identity_matrix, int luma_w, int luma_h, GrainNoise* out);

/* Appends the grain jobs of one plane: `dst_off`/`dst_stride` address the plane
 * the jobs write, `luma_off`/`luma_stride` the luma samples they read (from the
 * same buffer; the CPU host path passes them relative to the luma plane). */
void append_grain_jobs(const GrainNoise& grain, int plane, int plane_w, int plane_h, int dst_off,
                       int dst_stride, int luma_off, int luma_stride, int luma_w, int ss_x,
                       int ss_y, std::vector<Av1gGrainJob>* chroma,
                       std::vector<Av1gGrainJob>* luma);

/* Builds the plane layout the backends share: visible sample counts, padded
 * strides and plane offsets into one flat sample buffer. `stride_align` is the
 * row alignment the backend wants (32 for the CPU, 64 for the GPU ones).
 *
 * `geometry` is the *display* geometry; `coded_width` is the width the tile
 * data codes, which a superres frame codes narrower than it displays. The
 * buffer is sized for the display geometry, so one allocation holds the coded
 * picture and the upscaled one - the upscaler needs both. */
PlanLayout make_plan_layout(const FrameGeometry& geometry, int coded_width, int stride_align);

/* All work of one dependency level. Within a wave no block reads another
 * block's reconstructed samples, so the jobs can run in any order or in
 * parallel. */
struct WaveJobs {
  std::vector<Av1gIntraJob> intra;
  std::vector<Av1gInterJob> inter;
  std::vector<Av1gItxJob> itx;
  std::vector<Av1gItxJob> residual;

  bool empty() const {
    return intra.empty() && inter.empty() && itx.empty() && residual.empty();
  }
};

struct FramePlan {
  PlanLayout layout;
  int ref_slot[2] = {-1, -1};
  int maxv = 255;
  int blocks = 0;

  /* Coefficient pool, indexed by Av1gItxJob::coef_off. */
  std::vector<int32_t> coefficients;
  std::vector<int32_t> residual;
  /* Palette colour maps of the frame's palette blocks, indexed by
   * Av1gIntraJob::palette_off. Empty unless the frame codes a palette. */
  std::vector<uint8_t> palette_map;
  /* Size of the inverse transform scratch buffer, indexed by
   * Av1gItxJob::tmp_off. A job needs w * h ints, which is not the same as its
   * coefficient area (that one is clipped to 32 samples wide for a 64 point
   * transform, because only the low 32x32 quarter is coded). */
  size_t tmp_size = 0;

  std::vector<WaveJobs> waves;

  /* Deblocking runs in two ordered passes: all vertical edges, then all
   * horizontal ones. Edges of one direction never overlap, so a pass is
   * parallel, but a vertical edge and a horizontal edge cross, so the second
   * pass has to see the result of the first (this is also the order the AV1
   * specification uses). */
  std::vector<Av1gDeblockJob> deblock_v;
  std::vector<Av1gDeblockJob> deblock_h;
  std::vector<Av1gCdefJob> cdef;
  /* Superres: the rows of each plane resampled from the coded width to the
   * display width. Runs on the CDEF output and before loop restoration and the
   * film grain, which is why the jobs below are built on the display
   * geometry. */
  std::vector<Av1gUpscaleJob> upscale;
  /* Loop restoration: one job per restoration unit per stripe it touches, so
   * every job's samples come from a single stripe. It restores the *upscaled*
   * CDEF output (7.17), so it runs after the upscaling and before the grain. */
  std::vector<Av1gLrJob> restoration;
  /* Film grain, in two passes: the chroma planes read the luma the picture had
   * before any grain was added, so they run first, and the luma samples they
   * blend against are the ones a later pass is about to change. */
  std::vector<Av1gGrainJob> grain_chroma;
  std::vector<Av1gGrainJob> grain_luma;

  /* The film grain noise image and scaling tables the jobs above index. */
  GrainNoise grain_noise;
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual const DeviceInfo& info() const = 0;
  virtual const BackendOptions& options() const = 0;

  /* Allocates (or reuses) the working buffers for a geometry and reports the
   * layout that job offsets must be built against. */
  /* `geometry` is the display geometry; `coded_width` the width the tiles
   * code, which is smaller for a superres frame. */
  virtual PlanLayout prepare(const FrameGeometry& geometry, int coded_width) = 0;

  /* Uploads a host frame into the reference pool; returns the slot id. */
  virtual int upload_reference(const Frame& frame, std::string* error) = 0;
  virtual void release_reference(int slot) = 0;
  virtual int reference_capacity() const = 0;

  /* Executes the plan. When `out` is non null the planes are read back into it.
   * When `keep_reference` is true the result is stored in a reference slot and
   * its id is written to `out_slot`. */
  virtual bool decode(const FramePlan& plan, bool keep_reference, Frame* out,
                      int* out_slot, DecodeStats* stats, std::string* error) = 0;

  virtual bool to_rgb(int slot, ColorMatrix matrix, ColorRange range,
                      std::vector<uint8_t>* rgb, std::string* error) = 0;

  virtual bool to_nv12(int slot, size_t pitch, std::vector<uint8_t>* nv12,
                       std::string* error) = 0;

  virtual void reset() = 0;
};

/* Bytes an NV12 buffer with this pitch holds for a frame of this geometry, or 0
 * when the frame cannot be packed: NV12 is 4:2:0, so `ss_x`/`ss_y` must be 1, and
 * the pitch must be even and at least as wide as a chroma row (two bytes per
 * chroma sample). Every backend validates with this, and frame_to_nv12 uses it as
 * the size of its output. */
size_t nv12_buffer_size(int width, int height, int ss_x, int ss_y, size_t pitch);

std::unique_ptr<Backend> create_cpu_backend(const BackendOptions& options,
                                            std::string* error);
std::unique_ptr<Backend> create_opencl_backend(const BackendOptions& options,
                                               std::string* error);
std::unique_ptr<Backend> create_cuda_backend(const BackendOptions& options,
                                             std::string* error);

std::vector<DeviceInfo> list_opencl_devices();
std::vector<DeviceInfo> list_cuda_devices();

/* Turns decode syntax into the flat job plan a backend executes. `layout`
 * comes from Backend::prepare() for the frame geometry, so the job offsets
 * match the buffers that backend allocated. */
bool build_frame_plan(const FrameSyntax& syntax, const PlanLayout& layout,
                      const BackendOptions& options, int reference_capacity,
                      FramePlan* plan, std::string* error);

/* Runs deblock / CDEF (and, by default, film grain) on a host frame, exactly as
 * the decoder does. Used by encoders (including the synthetic one) so that their
 * reference frames do not drift. `grain` adds the film grain at the end;
 * the reference frame store needs the picture *before* it (7.20), so an encoder
 * calls this with `grain = false` and then apply_grain_host() on the display
 * copy. */
void apply_post_filters_host(Frame* frame, const FrameSyntax& syntax, bool grain = true);

/* The film grain synthesis filter alone (7.18.3), on a host frame. */
void apply_grain_host(Frame* frame, const FrameSyntax& syntax);

/* ------------------------------------------------------------------------- */
/* tables shared by every backend                                            */
/* ------------------------------------------------------------------------- */

namespace tables {

/* Transform_Row_Shift for a transform of this size: the right shift between
 * the row and column passes of the inverse transform. */
int tx_row_shift(int w, int h);

/* The specification's Subpel_Filters: AV1G_FILT_ROWS rows (the four filter
 * classes and their two four tap variants) x 16 phases (1/16 pel) x 8 taps,
 * Q7, laid out row major. */
const std::vector<int>& filters();

/* The specification's block warp process filters with Warped_Filters, appended
 * to the same table: AV1G_WARP_FILTER_OFFSET is where the 193 rows of 8 taps
 * start, so one uploaded table serves both. */
constexpr int kWarpFilterRows = 193;

/* setup shear (7.11.3.6, with the resolve divisor of 7.11.3.7): the two
 * shearing operations an affine model is applied as. `params` is the
 * specification's warp matrix at WARPEDMODEL_PREC_BITS precision. Returns the
 * specification's warpValid: a model whose shear is too steep to be applied as
 * two passes is not warped at all, the block falls back to its motion vector. */
bool gm_setup_shear(const int32_t* params, int* alpha, int* beta, int* gamma, int* delta);

/* The wedge codebook entry for a block of `block_w` x `block_h` *luma* samples
 * and a wedge index: the master mask direction, the mask origin and the sign
 * the block's mask is flipped with (the specification's flipSign). Returns
 * false for the shapes that carry no wedge. */
bool wedge_entry(int block_w, int block_h, int wedge_index, int* dir, int* xoff, int* yoff,
                 int* flip);

/* Q16 colour matrix coefficients for (matrix, range). */
void color_matrix(ColorMatrix matrix, ColorRange range, int bit_depth, int m[8],
                  int o[3]);

/* YUV -> RGB for the host path (identical arithmetic to the kernel). */
void frame_to_rgb(const Frame& frame, ColorMatrix matrix, ColorRange range,
                  std::vector<uint8_t>* rgb);

} /* namespace tables */
} /* namespace internal */
} /* namespace av1gpu */

#endif /* AV1GPU_CORE_BACKEND_HPP */
