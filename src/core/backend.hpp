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
  int plane_w[3] = {0, 0, 0};
  int plane_h[3] = {0, 0, 0};
  int stride[3] = {0, 0, 0};
  int plane_off[3] = {0, 0, 0};
  size_t buffer_size = 0;
};

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

/* Q12 orthonormal basis matrices, laid out row major, one block per size. */
const std::vector<int>& basis();
int basis_dct(int n);
int basis_adst(int n);
int basis_identity(int n);

/* Q7 subpel filters: AV1G_FILT_COUNT filters x 8 phases x 8 taps. */
const std::vector<int>& filters();

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
