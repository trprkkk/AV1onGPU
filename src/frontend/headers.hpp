/*
 * AV1onGPU - sequence and frame header syntax
 *
 * Parses the *uncompressed* headers of an AV1 bitstream: everything the spec
 * codes with literal bits before the tile payload. This is the first half of
 * the front-end: it turns bytes into the parameters a decoder needs (geometry,
 * bit depth, quantiser, loop filter levels, tile layout), and it stops exactly
 * where the arithmetic coded tile data begins.
 *
 * Supported today
 *   - full sequence header, including operating points and timing info,
 *   - KEY, INTRA_ONLY, INTER and SWITCH frame headers, complete up to the tile
 *     data, plus show_existing_frame,
 *   - tile layout, quantiser, segmentation, delta Q / delta LF, deblocking,
 *     CDEF, transform set, reference and skip mode, global motion, film grain
 *     presence.
 *
 * Reference frames: inter frame sizes and skip mode depend on the frames a
 * stream has decoded before, so `parse_frame_header` takes the `RefPool` the
 * caller maintains and `refresh_ref_pool()` applies each frame's refresh flags.
 *
 * Deliberately rejected with a clear message (never silently misparsed)
 *   - superres, intra block copy, restoration filters, film grain *data*,
 *     short reference signalling and frame ids.
 *
 * The field order and the derived values follow the AV1 specification; where a
 * value is derived from a reference frame (loop filter deltas, segmentation
 * data, film grain of a non-updating frame) the parser says so instead of
 * guessing.
 */
#ifndef AV1GPU_FRONTEND_HEADERS_HPP
#define AV1GPU_FRONTEND_HEADERS_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "frontend/cdf.hpp"

namespace av1gpu {
namespace frontend {

enum class FrameType : int {
  Key = 0,
  Inter = 1,
  IntraOnly = 2,
  Switch = 3,
};

const char* frame_type_name(FrameType type);

/* Sentinel for `primary_ref_frame`: the frame does not use a reference for its
 * CDF state. Matches PRIMARY_REF_NONE in the specification. */
constexpr int kPrimaryRefNone = 7;

struct OperatingPoint {
  bool present = false;
  int idc = 0;
  int major_level = 0;
  int minor_level = 0;
  bool tier = false;
  bool decoder_model_param_present = false;
  uint32_t decoder_buffer_delay = 0;
  uint32_t encoder_buffer_delay = 0;
  bool low_delay_mode = false;
  bool display_model_param_present = false;
  int initial_display_delay = 10;
};

struct TimingInfo {
  bool present = false;
  uint32_t num_units_in_display_tick = 0;
  uint32_t time_scale = 0;
  bool equal_picture_interval = false;
  uint32_t num_ticks_per_picture = 0;
  bool decoder_model_info_present = false;
  int buffer_delay_length = 0;
  uint32_t num_units_in_decoding_tick = 0;
  int buffer_removal_delay_length = 0;
  int frame_presentation_delay_length = 0;
  bool display_model_info_present = false;
};

struct SequenceHeader {
  int profile = 0;
  bool still_picture = false;
  bool reduced_still_picture_header = false;

  TimingInfo timing;
  std::vector<OperatingPoint> operating_points;

  int width_n_bits = 0;
  int height_n_bits = 0;
  int max_frame_width = 0;
  int max_frame_height = 0;

  bool frame_id_numbers_present = false;
  int delta_frame_id_n_bits = 0;
  int frame_id_n_bits = 0;

  bool use_128x128_superblock = false;
  bool enable_filter_intra = false;
  bool enable_intra_edge_filter = false;
  bool enable_interintra_compound = false;
  bool enable_masked_compound = false;
  bool enable_warped_motion = false;
  bool enable_dual_filter = false;
  bool enable_order_hint = false;
  bool enable_jnt_comp = false;
  bool enable_ref_frame_mvs = false;
  int order_hint_n_bits = 0;

  /* 0 = off, 1 = on, 2 = selected per frame (SELECT_SCREEN_CONTENT_TOOLS). */
  int seq_force_screen_content_tools = 2;
  int seq_force_integer_mv = 2;

  bool enable_superres = false;
  bool enable_cdef = false;
  bool enable_restoration = false;

  int bit_depth = 8;
  bool monochrome = false;
  bool color_description_present = false;
  int color_primaries = 2;            /* CP_UNSPECIFIED */
  int transfer_characteristics = 2;   /* TC_UNSPECIFIED */
  int matrix_coefficients = 2;        /* MC_UNSPECIFIED */
  bool color_range = false;
  int ss_x = 1;
  int ss_y = 1;
  int chroma_sample_position = 0;
  bool separate_uv_delta_q = false;
  bool film_grain_params_present = false;

  bool is_420() const { return ss_x == 1 && ss_y == 1; }
};

struct TileInfo {
  bool uniform = false;
  int log2_cols = 0;
  int log2_rows = 0;
  int cols = 0;
  int rows = 0;
  std::vector<int> col_start_sb;
  std::vector<int> row_start_sb;
  int context_update_tile_id = 0;
  int tile_size_bytes = 1;
};

struct LoopFilter {
  int level_y[2] = {0, 0};
  int level_u = 0;
  int level_v = 0;
  int sharpness = 0;
  bool mode_ref_delta_enabled = false;
  bool mode_ref_delta_update = false;
  int ref_deltas[8] = {1, 0, 0, 0, -1, 0, -1, -1}; /* spec defaults */
  int mode_deltas[2] = {0, 0};
};

struct CdefParams {
  int damping = 3;
  int bits = 0;
  int y_strength[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int uv_strength[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

/* Segmentation feature list, in specification order, with the width of each
 * coded value and whether it carries a sign bit. */
constexpr int kSegFeatureCount = 8;
constexpr int kMaxSegments = 8;

struct SegmentationParams {
  bool enabled = false;
  bool update_map = false;
  bool temporal_update = false;
  bool update_data = false;
  bool feature_enabled[kMaxSegments][kSegFeatureCount] = {};
  int feature_data[kMaxSegments][kSegFeatureCount] = {};
};

/* Global motion models, in specification order. */
enum GmType {
  kGmIdentity = 0,
  kGmTranslation = 1,
  kGmRotZoom = 2,
  kGmAffine = 3,
};

/* WARPEDMODEL_PREC_BITS: the precision the global motion parameters are stored
 * at. The diagonal entries of the identity model are 1 << 16, so an unscaled,
 * untranslated frame has no motion at all. */
constexpr int kWarpmodelPrecBits = 16;
constexpr int32_t kGmIdentityParam = 1 << kWarpmodelPrecBits;

/* A global motion model: the six warped-motion parameters, in the order
 * (a, b, c, d, e, f) of the specification's warp matrix.
 *
 *   x' = (a x + b y + c) / 2^16
 *   y' = (d x + e y + f) / 2^16
 *
 * The parameters are stored in this coded form rather than as a matrix so that
 * re-reading the same model from a reference frame gives bit identical
 * arithmetic: `is_global` codes the *difference* from the reference frame's
 * model, so a decoder that normalised them would drift. */
struct GlobalMotion {
  int type = kGmIdentity;
  int32_t params[6] = {0, 0, kGmIdentityParam, 0, 0, kGmIdentityParam};

  bool identity() const { return type == kGmIdentity; }
  bool translation_only() const { return type == kGmTranslation; }
};

/* The specification's RefMvTemporalBlock: what one 8x8 cell of a frame
 * contributes to the temporal motion vector candidates of a later frame.
 *
 * `ref` is the reference *type* the vector points at, and zero means the cell
 * contributed nothing - the block was intra predicted, or its vector pointed
 * forwards in time, in which case it cannot be projected onto a future frame.
 * The two reference decoders use the invalid motion vector value as the marker;
 * a zero reference type is the same information without the sentinel. */
struct TemporalMv {
  int16_t x = 0;
  int16_t y = 0;
  uint8_t ref = 0;

  bool valid() const { return ref != 0; }
};

/* A frame's motion vector grid, one entry per 8x8 cell, saved when the frame is
 * decoded (`load_tmvs`/`save_tmvs` in the reference decoders). A later frame
 * projects these vectors onto its own grid and scans the result next to the
 * spatial candidates, which is what `use_ref_frame_mvs` buys. */
struct RefMvs {
  int cols = 0; /* in 8x8 cells */
  int rows = 0;
  std::vector<TemporalMv> cells;

  void resize(int new_cols, int new_rows) {
    cols = new_cols;
    rows = new_rows;
    cells.assign(static_cast<size_t>(new_cols) * new_rows, TemporalMv());
  }
  TemporalMv& at(int col, int row) {
    return cells[static_cast<size_t>(row) * cols + col];
  }
  const TemporalMv& at(int col, int row) const {
    return cells[static_cast<size_t>(row) * cols + col];
  }
};

/* mv_projection, one component: scale a stored motion vector by the ratio of
 * two order hint distances, `num / den`, both in [-31, 31] and `den` non-zero.
 *
 * The scaling is done in 1/64 steps (the table is 2^14 / den rounded, and the
 * result is shifted back by 14), rounded away from zero, and clipped to the
 * range a projectable vector can span (2^14 - 1). This is the specification's
 * section 7.9.3 arithmetic, bit for bit: a decoder that rounds differently gets
 * a different candidate and then a different dynamic reference list index. */
int project_mv_component(int mv, int num, int den);

/* One entry of the reference frame store: enough for the bitstream front-end to
 * resolve inter frame sizes and skip-mode decisions. */
struct RefSlot {
  bool valid = false;
  int width = 0;
  int height = 0;
  int order_hint = 0;
  FrameType type = FrameType::Key;
  /* The frame's global motion model per reference type, which the next frame
   * reads to decode its own model differentially. */
  GlobalMotion global_motion[8];
  /* The order hints of the seven frames *this* frame referenced. Projecting a
   * stored vector needs both ends of its trajectory, so the next frame reads
   * them together with the grid itself. */
  int ref_poc[7] = {0, 0, 0, 0, 0, 0, 0};
  /* The motion vectors this frame saved, null for an intra frame and for a
   * frame whose tile was not decoded. Shared because one frame can refresh
   * several slots. */
  std::shared_ptr<const RefMvs> ref_mvs;
  /* The probability tables as they stood at the end of this frame. A later
   * frame whose primary_ref_frame names this slot starts from them instead of
   * the defaults, which is how AV1 keeps a whole inter frame's symbols cheap. */
  std::shared_ptr<const CdfContext> cdf;
};

/* The eight AV1 reference slots. `parse_frame_header` reads them to resolve
 * inter frame sizes and skip-mode decisions; `refresh_ref_pool()` applies a
 * parsed frame's refresh_frame_flags. */
struct RefPool {
  RefSlot slots[8];
};

struct FrameHeader {
  bool show_existing_frame = false;
  int existing_frame_idx = 0;

  FrameType type = FrameType::Key;
  bool show_frame = false;
  bool showable_frame = false;
  bool error_resilient_mode = false;
  bool disable_cdf_update = false;
  bool allow_screen_content_tools = false;
  bool force_integer_mv = false;
  bool allow_intrabc = false;
  bool frame_id_present = false;
  int frame_id = 0;

  bool frame_size_override = false;
  int width = 0;   /* upscaled width: what the decoder produces */
  int height = 0;
  int render_width = 0;
  int render_height = 0;
  int frame_width = 0; /* coded width before superres upscaling */
  bool superres = false;
  int superres_denominator = 8;

  bool order_hint_present = false;
  int order_hint = 0;
  int primary_ref_frame = kPrimaryRefNone;
  int refresh_frame_flags = 0;
  bool refresh_frame_context = false;
  int frame_presentation_delay = 0;

  /* Inter frame syntax. `ref_frame_idx` is resolved through the reference pool;
   * `size_from_reference` is set when the frame inherits its size from
   * reference frame 0 instead of coding width/height. */
  bool size_from_reference = false;
  int ref_frame_idx[7] = {0, 0, 0, 0, 0, 0, 0};
  int ref_order_hint[7] = {0, 0, 0, 0, 0, 0, 0};
  bool allow_high_precision_mv = false;
  bool is_filter_switchable = false;
  int interpolation_filter = 0;
  bool is_motion_mode_switchable = false;
  bool use_ref_frame_mvs = false;
  bool reference_select = false;
  bool allow_warped_motion = false;
  bool skip_mode_allowed = false;
  bool skip_mode_present = false;
  int skip_mode_frame[2] = {0, 0};
  bool is_global[8] = {false, false, false, false, false, false, false, false};
  /* Global motion, one model per reference type; identity for the reference
   * types the frame does not code a model for. */
  GlobalMotion global_motion[8];

  TileInfo tiles;

  int base_q_idx = 0;
  int delta_q_y_dc = 0;
  int delta_q_u_dc = 0;
  int delta_q_u_ac = 0;
  int delta_q_v_dc = 0;
  int delta_q_v_ac = 0;
  bool using_qmatrix = false;
  int qm_y = 0;
  int qm_u = 0;
  int qm_v = 0;

  SegmentationParams segmentation;
  bool segmentation_enabled = false;
  bool delta_q_present = false;
  int delta_q_res = 0;
  bool delta_lf_present = false;
  int delta_lf_res = 0;
  bool delta_lf_multi = false;

  bool coded_lossless = false;
  bool all_lossless = false;

  LoopFilter loop_filter;
  CdefParams cdef;
  int restoration_type[3] = {0, 0, 0};
  int restoration_unit_size[2] = {0, 0};

  bool tx_mode_select = false;
  bool reduced_tx_set = false;

  bool film_grain_present = false;
  int film_grain_seed = 0;

  /* Byte offset into the OBU payload where the tile data starts. */
  size_t tile_data_offset = 0;
};

/* Applies `frame.refresh_frame_flags` to the reference pool. `ref_mvs` is the
 * motion vector grid the frame's tile produced, if it was decoded; the slots the
 * frame refreshes keep it for the frames that project from them. `cdf` is the
 * frame's adapted probability context, which a later frame may inherit. */
void refresh_ref_pool(const FrameHeader& frame, RefPool* pool,
                      const std::shared_ptr<const RefMvs>& ref_mvs = nullptr,
                      const std::shared_ptr<const CdfContext>& cdf = nullptr);

enum class ParseStatus {
  Ok = 0,
  Unsupported = 1, /* the syntax was understood, the feature is not implemented */
  Invalid = 2,     /* the bitstream is broken or not AV1 */
};

const char* parse_status_name(ParseStatus status);

/* Parses one sequence header OBU payload. */
ParseStatus parse_sequence_header(const uint8_t* data, size_t size, SequenceHeader* out,
                                  std::string* error);

/* Parses one frame header OBU payload (or the header part of a frame OBU).
 * `seq` must be the sequence header in force. On success `out->tile_data_offset`
 * is the byte offset of the tile payload inside `data`. */
ParseStatus parse_frame_header(const SequenceHeader& seq, const uint8_t* data, size_t size,
                               FrameHeader* out, std::string* error,
                               const RefPool* refs = nullptr);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_HEADERS_HPP */
