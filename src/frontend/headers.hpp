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
 *   - KEY and INTRA_ONLY frame headers, complete up to the tile data,
 *   - tile layout, quantiser, deblocking, CDEF, transform set and film grain
 *     presence.
 *
 * Deliberately rejected with a clear message (never silently misparsed)
 *   - inter and switch frames (their reference management belongs with the
 *     tile decoder that is not written yet),
 *   - show_existing_frame, segmentation, delta Q / delta LF, superres,
 *     intra block copy, restoration filters and film grain *data*.
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
#include <string>
#include <vector>

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
                               FrameHeader* out, std::string* error);

} /* namespace frontend */
} /* namespace av1gpu */

#endif /* AV1GPU_FRONTEND_HEADERS_HPP */
