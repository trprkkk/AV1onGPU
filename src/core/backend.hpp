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
  FrameGeometry frame;
  int plane_w[3] = {0, 0, 0};  /* visible samples */
  int plane_h[3] = {0, 0, 0};
  int stride[3] = {0, 0, 0};   /* row pitch, including the edge padding */
  int plane_off[3] = {0, 0, 0};
  size_t buffer_size = 0;
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

/* Builds the plane layout the backends share: visible sample counts, padded
 * strides and plane offsets into one flat sample buffer. `stride_align` is the
 * row alignment the backend wants (32 for the CPU, 64 for the GPU ones). */
PlanLayout make_plan_layout(const FrameGeometry& geometry, int stride_align);

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
  std::vector<Av1gGrainJob> grain;

  /* Film grain template (host generated, uploaded once per frame). */
  std::vector<signed char> grain_template;
  int grain_stride = 0;
  int grain_h = 0;
  int grain_lut[64] = {0};
};

class Backend {
 public:
  virtual ~Backend() = default;

  virtual const DeviceInfo& info() const = 0;
  virtual const BackendOptions& options() const = 0;

  /* Allocates (or reuses) the working buffers for a geometry and reports the
   * layout that job offsets must be built against. */
  virtual PlanLayout prepare(const FrameGeometry& geometry) = 0;

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

  virtual void reset() = 0;
};

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

/* Runs deblock / CDEF / film grain on a host frame, exactly as the decoder
 * does. Used by encoders (including the synthetic one) so that their reference
 * frames do not drift. */
void apply_post_filters_host(Frame* frame, const FrameSyntax& syntax);

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

/* Generates the film grain template for a seed, as the AV1 spec's
 * pseudo random noise generator does. */
void grain_template(int seed, int stride, int rows, std::vector<signed char>* out);

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
