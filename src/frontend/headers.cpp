#include "frontend/headers.hpp"

#include <algorithm>

#include "frontend/bitreader.hpp"

namespace av1gpu {
namespace frontend {
namespace {

ParseStatus invalid(const std::string& message, std::string* error) {
  if (error) *error = message;
  return ParseStatus::Invalid;
}

ParseStatus unsupported(const std::string& message, std::string* error) {
  if (error) *error = message;
  return ParseStatus::Unsupported;
}

/* Smallest k with (size << k) >= target, the tile grid helper from the spec. */
int tile_log2(int size, int target) {
  int k = 0;
  while ((size << k) < target && k < 31) ++k;
  return k;
}

/* trailing_bits(): a one bit, zeros up to the byte boundary, then zero bytes to
 * the end of the OBU. Anything else means the parser lost sync. */
ParseStatus check_trailing_bits(BitReader* reader, std::string* error) {
  if (reader->overrun()) return invalid("header runs past the end of the OBU", error);
  if (!reader->read_bit()) {
    return invalid("missing trailing one bit at bit " +
                       std::to_string(reader->bits_read() - 1) +
                       " (parsing lost sync)",
                   error);
  }
  if (reader->overrun()) return invalid("header runs past the end of the OBU", error);
  reader->align_to_byte();
  while (reader->bits_left() > 0) {
    if (reader->read_bit()) {
      return invalid("non-zero padding after the sequence header", error);
    }
  }
  return ParseStatus::Ok;
}

/* Width of each segmentation feature value, in specification order. */
constexpr int kSegFeatureBits[kSegFeatureCount] = {8, 6, 6, 6, 6, 3, 1, 1};

/* get_relative_dist(): sign extended distance between two order hints. */
int get_relative_dist(const SequenceHeader& seq, int a, int b) {
  if (!seq.enable_order_hint) return 0;
  int diff = a - b;
  const int m = 1 << (seq.order_hint_n_bits - 1);
  return (diff & (m - 1)) - (diff & m);
}

/* decode_subexp(): AV1's subexponential code, used for the global motion
 * parameters. It is a bit reader construct (not arithmetic coded), so it lives
 * here rather than in the range decoder. */
int read_subexp(BitReader* r, int num_symbols) {
  int i = 0;
  int mk = 0;
  const int k = 3;
  for (;;) {
    const int b2 = i ? k + i - 1 : k;
    const int a = 1 << b2;
    if (num_symbols <= mk + 3 * a) {
      return static_cast<int>(r->read_uniform(static_cast<uint32_t>(num_symbols - mk))) + mk;
    }
    if (r->read_bit()) {
      ++i;
      mk += a;
    } else {
      return static_cast<int>(r->read(b2)) + mk;
    }
  }
}

/* inverse_recenter(): maps the uniformly coded value `v` back around the
 * reference value `r`, so that values near the reference cost few bits. */
int inverse_recenter(int r, int v) {
  if (v > 2 * r) return v;
  if (v & 1) return r - ((v + 1) >> 1);
  return r + (v >> 1);
}

int decode_unsigned_subexp_with_ref(int mx, int r, BitReader* reader) {
  const int v = read_subexp(reader, mx);
  if ((r << 1) <= mx) return inverse_recenter(r, v);
  return mx - 1 - inverse_recenter(mx - 1 - r, v);
}

int decode_signed_subexp_with_ref(int low, int high, int r, BitReader* reader) {
  return decode_unsigned_subexp_with_ref(high - low, r - low, reader) + low;
}

/* read_global_param(): one warp model coefficient, coded as a subexponential
 * difference from the coefficient the reference frame used.
 *
 * The coefficient is stored at WARPEDMODEL_PREC_BITS of precision, but coded at
 * a coarser one, so the low bits of the reference value are what the coder
 * predicts from and the decoded value is re-quantised on the way in. `idx % 3
 * == 2` marks the translation column, which is coded around 1.0 (the identity
 * model's diagonal) rather than around 0. */
void read_global_param(const FrameHeader& h, BitReader* r, GlobalMotion* gm, int idx,
                       const GlobalMotion& prev) {
  int abs_bits = 12; /* GM_ABS_ALPHA_BITS */
  int prec_bits = 15; /* GM_ALPHA_PREC_BITS */
  if (idx < 2) {
    if (gm->type == kGmTranslation) {
      abs_bits = 9 - (h.allow_high_precision_mv ? 0 : 1);  /* GM_ABS_TRANS_ONLY_BITS */
      prec_bits = 3 - (h.allow_high_precision_mv ? 0 : 1); /* GM_TRANS_ONLY_PREC_BITS */
    } else {
      abs_bits = 12; /* GM_ABS_TRANS_BITS */
      prec_bits = 6; /* GM_TRANS_PREC_BITS */
    }
  }
  const int prec_diff = kWarpmodelPrecBits - prec_bits;
  const int round = (idx % 3) == 2 ? kGmIdentityParam : 0;
  const int sub = (idx % 3) == 2 ? (1 << prec_bits) : 0;
  const int mx = 1 << abs_bits;
  const int ref = (prev.params[idx] >> prec_diff) - sub;
  gm->params[idx] = (decode_signed_subexp_with_ref(-mx, mx + 1, ref, r) << prec_diff) + round;
}

/* global_motion_params(): one model per reference type, coded as a difference
 * from the model the referenced frame used. Reference types the frame does not
 * code a model for keep the identity model. */
void read_global_motion_params(const SequenceHeader& seq, FrameHeader& h, BitReader* r,
                               const RefPool* refs) {
  (void)seq;
  for (int ref = 0; ref < 8; ++ref) h.global_motion[ref] = GlobalMotion();

  for (int ref = 1; ref < 8; ++ref) {
    GlobalMotion& gm = h.global_motion[ref];
    /* PrevGmParams: the model the reference frame itself used, which is what
     * the coded difference is relative to. */
    GlobalMotion prev;
    if (refs) prev = refs->slots[h.ref_frame_idx[ref - 1]].global_motion[ref];

    h.is_global[ref] = r->read_bit();
    if (!h.is_global[ref]) {
      gm.type = kGmIdentity;
      continue;
    }
    if (r->read_bit()) {
      gm.type = kGmRotZoom;
    } else {
      gm.type = r->read_bit() ? kGmTranslation : kGmAffine;
    }
    if (gm.type >= kGmRotZoom) {
      read_global_param(h, r, &gm, 2, prev);
      read_global_param(h, r, &gm, 3, prev);
      if (gm.type == kGmAffine) {
        read_global_param(h, r, &gm, 4, prev);
        read_global_param(h, r, &gm, 5, prev);
      } else {
        gm.params[4] = -gm.params[3];
        gm.params[5] = gm.params[2];
      }
    }
    if (gm.type >= kGmTranslation) {
      read_global_param(h, r, &gm, 0, prev);
      read_global_param(h, r, &gm, 1, prev);
    }
  }
}

/* skip_mode_params(): AV1 only allows skip mode when the reference list holds a
 * frame that is both in the past and in the future of this one. */
void compute_skip_mode(const SequenceHeader& seq, FrameHeader& h, BitReader* r) {
  h.skip_mode_allowed = false;
  h.skip_mode_present = false;
  h.skip_mode_frame[0] = 0;
  h.skip_mode_frame[1] = 0;
  if (h.error_resilient_mode || !seq.enable_order_hint) return;

  int forward_idx = -1;
  int backward_idx = -1;
  int forward_hint = 0;
  int backward_hint = 0;
  for (int i = 0; i < 7; ++i) {
    const int ref_hint = h.ref_order_hint[i];
    const int dist = get_relative_dist(seq, ref_hint, h.order_hint);
    if (dist < 0) {
      if (forward_idx < 0 || get_relative_dist(seq, ref_hint, forward_hint) > 0) {
        forward_idx = i;
        forward_hint = ref_hint;
      }
    } else if (dist > 0) {
      if (backward_idx < 0 || get_relative_dist(seq, ref_hint, backward_hint) < 0) {
        backward_idx = i;
        backward_hint = ref_hint;
      }
    }
  }

  if (forward_idx >= 0 && backward_idx >= 0) {
    h.skip_mode_allowed = true;
    if (get_relative_dist(seq, forward_hint, backward_hint) < 0) {
      h.skip_mode_frame[0] = forward_idx + 1;
      h.skip_mode_frame[1] = backward_idx + 1;
    } else {
      h.skip_mode_frame[0] = backward_idx + 1;
      h.skip_mode_frame[1] = forward_idx + 1;
    }
  } else if (forward_idx >= 0) {
    if (forward_idx > 0) {
      h.skip_mode_allowed = true;
      h.skip_mode_frame[0] = h.skip_mode_frame[1] = forward_idx + 1;
    }
  } else if (backward_idx >= 0) {
    if (backward_idx > 0) {
      h.skip_mode_allowed = true;
      h.skip_mode_frame[0] = h.skip_mode_frame[1] = backward_idx + 1;
    }
  }

  if (h.skip_mode_allowed) h.skip_mode_present = r->read_bit();
}

} /* namespace */

void refresh_ref_pool(const FrameHeader& frame, RefPool* pool) {
  if (!pool || frame.show_existing_frame) return;
  for (int i = 0; i < 8; ++i) {
    if (frame.refresh_frame_flags & (1 << i)) {
      RefSlot& slot = pool->slots[i];
      slot.valid = true;
      slot.width = frame.width;
      slot.height = frame.height;
      slot.order_hint = frame.order_hint;
      slot.type = frame.type;
      for (int ref = 0; ref < 8; ++ref) slot.global_motion[ref] = frame.global_motion[ref];
    }
  }
}

const char* frame_type_name(FrameType type) {
  switch (type) {
    case FrameType::Key: return "key";
    case FrameType::Inter: return "inter";
    case FrameType::IntraOnly: return "intra_only";
    case FrameType::Switch: return "switch";
  }
  return "?";
}

const char* parse_status_name(ParseStatus status) {
  switch (status) {
    case ParseStatus::Ok: return "ok";
    case ParseStatus::Unsupported: return "unsupported";
    case ParseStatus::Invalid: return "invalid";
  }
  return "?";
}

ParseStatus parse_sequence_header(const uint8_t* data, size_t size, SequenceHeader* out,
                                  std::string* error) {
  if (!data || size == 0) return invalid("empty sequence header", error);

  SequenceHeader seq;
  BitReader r(data, size);

  seq.profile = static_cast<int>(r.read(3));
  if (seq.profile > 2) {
    return invalid("reserved seq_profile " + std::to_string(seq.profile), error);
  }
  seq.still_picture = r.read_bit();
  seq.reduced_still_picture_header = r.read_bit();
  if (seq.reduced_still_picture_header && !seq.still_picture) {
    return invalid("reduced_still_picture_header requires still_picture", error);
  }

  if (seq.reduced_still_picture_header) {
    OperatingPoint op;
    op.present = true;
    op.major_level = static_cast<int>(r.read(3));
    op.minor_level = static_cast<int>(r.read(2));
    op.initial_display_delay = 10;
    seq.operating_points.push_back(op);
  } else {
    TimingInfo& timing = seq.timing;
    timing.present = r.read_bit();
    if (timing.present) {
      timing.num_units_in_display_tick = r.read(32);
      timing.time_scale = r.read(32);
      timing.equal_picture_interval = r.read_bit();
      if (timing.equal_picture_interval) {
        const uint32_t ticks = r.read_uvlc();
        if (ticks == 0xffffffffu) {
          return invalid("invalid num_ticks_per_picture_minus_1", error);
        }
        timing.num_ticks_per_picture = ticks + 1;
      }
      timing.decoder_model_info_present = r.read_bit();
      if (timing.decoder_model_info_present) {
        timing.buffer_delay_length = static_cast<int>(r.read(5)) + 1;
        timing.num_units_in_decoding_tick = r.read(32);
        timing.buffer_removal_delay_length = static_cast<int>(r.read(5)) + 1;
        timing.frame_presentation_delay_length = static_cast<int>(r.read(5)) + 1;
        if (timing.num_units_in_decoding_tick == 0) {
          return invalid("num_units_in_decoding_tick is zero", error);
        }
      }
      if (timing.num_units_in_display_tick == 0 || timing.time_scale == 0) {
        return invalid("timing info with a zero tick or time scale", error);
      }
    }
    timing.display_model_info_present = r.read_bit();

    const int num_operating_points = static_cast<int>(r.read(5)) + 1;
    seq.operating_points.resize(static_cast<size_t>(num_operating_points));
    for (int i = 0; i < num_operating_points; ++i) {
      OperatingPoint& op = seq.operating_points[static_cast<size_t>(i)];
      op.present = true;
      op.idc = static_cast<int>(r.read(12));
      if (op.idc != 0 && ((op.idc & 0xff) == 0 || (op.idc & 0xf00) == 0)) {
        return invalid("operating point " + std::to_string(i) + " has an invalid idc",
                       error);
      }
      op.major_level = 2 + static_cast<int>(r.read(3));
      op.minor_level = static_cast<int>(r.read(2));
      if (op.major_level > 3) op.tier = r.read_bit();
      if (timing.decoder_model_info_present) {
        op.decoder_model_param_present = r.read_bit();
        if (op.decoder_model_param_present) {
          op.decoder_buffer_delay = r.read(timing.buffer_delay_length);
          op.encoder_buffer_delay = r.read(timing.buffer_delay_length);
          op.low_delay_mode = r.read_bit();
        }
      }
      if (timing.display_model_info_present) {
        op.display_model_param_present = r.read_bit();
      }
      if (op.display_model_param_present) {
        op.initial_display_delay = static_cast<int>(r.read(4)) + 1;
      } else {
        op.initial_display_delay = 10;
      }
    }
  }

  seq.width_n_bits = static_cast<int>(r.read(4)) + 1;
  seq.height_n_bits = static_cast<int>(r.read(4)) + 1;
  seq.max_frame_width = static_cast<int>(r.read(seq.width_n_bits)) + 1;
  seq.max_frame_height = static_cast<int>(r.read(seq.height_n_bits)) + 1;

  if (!seq.reduced_still_picture_header) {
    seq.frame_id_numbers_present = r.read_bit();
    if (seq.frame_id_numbers_present) {
      seq.delta_frame_id_n_bits = static_cast<int>(r.read(4)) + 2;
      seq.frame_id_n_bits = static_cast<int>(r.read(3)) + seq.delta_frame_id_n_bits + 1;
    }
  }

  seq.use_128x128_superblock = r.read_bit();
  seq.enable_filter_intra = r.read_bit();
  seq.enable_intra_edge_filter = r.read_bit();
  if (seq.reduced_still_picture_header) {
    seq.seq_force_screen_content_tools = 2;
    seq.seq_force_integer_mv = 2;
  } else {
    seq.enable_interintra_compound = r.read_bit();
    seq.enable_masked_compound = r.read_bit();
    seq.enable_warped_motion = r.read_bit();
    seq.enable_dual_filter = r.read_bit();
    seq.enable_order_hint = r.read_bit();
    if (seq.enable_order_hint) {
      seq.enable_jnt_comp = r.read_bit();
      seq.enable_ref_frame_mvs = r.read_bit();
    }
    if (r.read_bit()) { /* seq_choose_screen_content_tools */
      seq.seq_force_screen_content_tools = 2;
    } else {
      seq.seq_force_screen_content_tools = static_cast<int>(r.read_bit());
    }
    if (seq.seq_force_screen_content_tools > 0) {
      if (r.read_bit()) { /* seq_choose_integer_mv */
        seq.seq_force_integer_mv = 2;
      } else {
        seq.seq_force_integer_mv = static_cast<int>(r.read_bit());
      }
    } else {
      seq.seq_force_integer_mv = 2;
    }
    if (seq.enable_order_hint) {
      seq.order_hint_n_bits = static_cast<int>(r.read(3)) + 1;
    }
  }

  seq.enable_superres = r.read_bit();
  seq.enable_cdef = r.read_bit();
  seq.enable_restoration = r.read_bit();

  /* colour_config() */
  const bool high_bitdepth = r.read_bit();
  if (seq.profile == 2 && high_bitdepth) {
    seq.bit_depth = r.read_bit() ? 12 : 10;
  } else {
    seq.bit_depth = high_bitdepth ? 10 : 8;
  }
  if (seq.profile != 1) seq.monochrome = r.read_bit();

  seq.color_description_present = r.read_bit();
  if (seq.color_description_present) {
    seq.color_primaries = static_cast<int>(r.read(8));
    seq.transfer_characteristics = static_cast<int>(r.read(8));
    seq.matrix_coefficients = static_cast<int>(r.read(8));
  }

  const int kCpBt709 = 1, kTcSrgb = 13, kMcIdentity = 0;
  if (seq.monochrome) {
    seq.color_range = r.read_bit();
    seq.ss_x = seq.ss_y = 1;
    seq.chroma_sample_position = 0;
  } else if (seq.color_primaries == kCpBt709 &&
             seq.transfer_characteristics == kTcSrgb &&
             seq.matrix_coefficients == kMcIdentity) {
    seq.color_range = true;
    seq.ss_x = seq.ss_y = 0;
    if (!(seq.profile == 1 || (seq.profile == 2 && seq.bit_depth == 12))) {
      return invalid("identity matrix needs seq_profile 1 or 2 with 12 bit", error);
    }
  } else {
    seq.color_range = r.read_bit();
    switch (seq.profile) {
      case 0:
        seq.ss_x = seq.ss_y = 1;
        break;
      case 1:
        seq.ss_x = seq.ss_y = 0;
        break;
      default: /* profile 2: 12 bit is signalled, 8/10 bit is 4:2:2 */
        if (seq.bit_depth == 12) {
          seq.ss_x = static_cast<int>(r.read_bit());
          seq.ss_y = seq.ss_x ? static_cast<int>(r.read_bit()) : 0;
        } else {
          seq.ss_x = 1;
          seq.ss_y = 0;
        }
        break;
    }
    if (seq.ss_x == 1 && seq.ss_y == 1) {
      seq.chroma_sample_position = static_cast<int>(r.read(2));
    }
  }

  if (!seq.monochrome) seq.separate_uv_delta_q = r.read_bit();
  seq.film_grain_params_present = r.read_bit();

  const ParseStatus status = check_trailing_bits(&r, error);
  if (status != ParseStatus::Ok) return status;

  *out = seq;
  return ParseStatus::Ok;
}

ParseStatus parse_frame_header(const SequenceHeader& seq, const uint8_t* data, size_t size,
                               FrameHeader* out, std::string* error,
                               const RefPool* refs) {
  if (!data || size == 0) return invalid("empty frame header", error);
  const int sb128 = seq.use_128x128_superblock ? 1 : 0;

  FrameHeader h;
  BitReader r(data, size);

  if (!seq.reduced_still_picture_header) {
    h.show_existing_frame = r.read_bit();
  }
  if (h.show_existing_frame) {
    h.existing_frame_idx = static_cast<int>(r.read(3));
    if (seq.timing.decoder_model_info_present && !seq.timing.equal_picture_interval) {
      h.frame_presentation_delay = static_cast<int>(
          r.read(seq.timing.frame_presentation_delay_length));
    }
    if (seq.frame_id_numbers_present) {
      h.frame_id_present = true;
      h.frame_id = static_cast<int>(r.read(seq.frame_id_n_bits));
    }
    if (!refs) {
      *out = h;
      return unsupported("show_existing_frame needs the reference frame pool", error);
    }
    const RefSlot& slot = refs->slots[h.existing_frame_idx];
    if (!slot.valid) {
      *out = h;
      return invalid("show_existing_frame refers to an empty reference slot", error);
    }
    /* The frame is a repeat of a stored one: report its geometry so callers can
     * display it without decoding anything. */
    h.type = slot.type;
    h.show_frame = true;
    h.showable_frame = true;
    h.width = slot.width;
    h.height = slot.height;
    h.frame_width = slot.width;
    h.render_width = slot.width;
    h.render_height = slot.height;
    h.order_hint_present = seq.enable_order_hint;
    h.order_hint = slot.order_hint;
    *out = h;
    return ParseStatus::Ok;
  }

  if (seq.reduced_still_picture_header) {
    h.type = FrameType::Key;
    h.show_frame = true;
  } else {
    h.type = static_cast<FrameType>(r.read(2));
    h.show_frame = r.read_bit();
  }
  if (h.show_frame) {
    if (seq.timing.decoder_model_info_present && !seq.timing.equal_picture_interval) {
      h.frame_presentation_delay = static_cast<int>(
          r.read(seq.timing.frame_presentation_delay_length));
    }
    h.showable_frame = h.type != FrameType::Key;
  } else {
    h.showable_frame = r.read_bit();
  }

  const bool key_or_intra = h.type == FrameType::Key || h.type == FrameType::IntraOnly;
  h.error_resilient_mode =
      (h.type == FrameType::Key && h.show_frame) || h.type == FrameType::Switch ||
      seq.reduced_still_picture_header || r.read_bit();

  h.disable_cdf_update = r.read_bit();
  if (seq.seq_force_screen_content_tools == 2) {
    h.allow_screen_content_tools = r.read_bit();
  } else {
    h.allow_screen_content_tools = seq.seq_force_screen_content_tools == 1;
  }
  if (h.allow_screen_content_tools) {
    if (seq.seq_force_integer_mv == 2) {
      h.force_integer_mv = r.read_bit();
    } else {
      h.force_integer_mv = seq.seq_force_integer_mv == 1;
    }
  }

  if (seq.frame_id_numbers_present) {
    h.frame_id_present = true;
    h.frame_id = static_cast<int>(r.read(seq.frame_id_n_bits));
  }

  if (!seq.reduced_still_picture_header) {
    h.frame_size_override = h.type == FrameType::Switch ? true : r.read_bit();
  }
  if (seq.enable_order_hint) {
    h.order_hint_present = true;
    h.order_hint = static_cast<int>(r.read(seq.order_hint_n_bits));
  }
  h.primary_ref_frame = (!h.error_resilient_mode &&
                         (h.type == FrameType::Inter || h.type == FrameType::Switch))
                            ? static_cast<int>(r.read(3))
                            : kPrimaryRefNone;

  if (seq.timing.decoder_model_info_present) {
    const bool buffer_removal_time_present = r.read_bit();
    if (buffer_removal_time_present) {
      for (size_t i = 0; i < seq.operating_points.size(); ++i) {
        const OperatingPoint& op = seq.operating_points[i];
        if (op.decoder_model_param_present) {
          r.read(seq.timing.buffer_removal_delay_length);
        }
      }
    }
  }

  h.refresh_frame_flags = (h.type == FrameType::Key && h.show_frame)
                              ? 0xff
                              : static_cast<int>(r.read(8));
  if (h.refresh_frame_flags != 0xff && h.error_resilient_mode && seq.enable_order_hint) {
    for (int i = 0; i < 8; ++i) r.read(seq.order_hint_n_bits);
  }

  /* frame_refs(): which of the eight slots this frame predicts from. */
  if (!key_or_intra) {
    if (!refs) {
      *out = h;
      return unsupported("inter frames need the reference frame pool", error);
    }
    if (r.read_bit()) {
      *out = h;
      return unsupported("frame_refs_short_signaling is not implemented", error);
    }
    for (int i = 0; i < 7; ++i) h.ref_frame_idx[i] = static_cast<int>(r.read(3));
    if (seq.frame_id_numbers_present) {
      *out = h;
      return unsupported("frame ids in inter frame references are not implemented", error);
    }
    for (int i = 0; i < 7; ++i) {
      h.ref_order_hint[i] = refs->slots[h.ref_frame_idx[i]].order_hint;
    }
  }

  /* frame_size(): inter frames with size override off inherit the size of the
   * reference frame they were predicted from. */
  if (h.frame_size_override) {
    h.frame_width = static_cast<int>(r.read(seq.width_n_bits)) + 1;
    h.height = static_cast<int>(r.read(seq.height_n_bits)) + 1;
  } else if (!key_or_intra) {
    const RefSlot& slot = refs->slots[h.ref_frame_idx[0]];
    if (!slot.valid) {
      *out = h;
      return invalid("inter frame refers to an empty reference slot", error);
    }
    h.size_from_reference = true;
    h.frame_width = slot.width;
    h.height = slot.height;
  } else {
    h.frame_width = seq.max_frame_width;
    h.height = seq.max_frame_height;
  }
  h.width = h.frame_width;
  if (seq.enable_superres) {
    h.superres = r.read_bit();
    if (h.superres) {
      h.superres_denominator = 9 + static_cast<int>(r.read(3));
      const int denominator = h.superres_denominator;
      h.width = std::max((h.frame_width * 8 + (denominator >> 1)) / denominator,
                         std::min(16, h.frame_width));
    }
  }
  const bool have_render_size = r.read_bit();
  if (have_render_size) {
    h.render_width = static_cast<int>(r.read(16)) + 1;
    h.render_height = static_cast<int>(r.read(16)) + 1;
  } else {
    h.render_width = h.width;
    h.render_height = h.height;
  }
  if (h.allow_screen_content_tools && !h.superres) {
    h.allow_intrabc = r.read_bit();
  }
  if (h.allow_intrabc) {
    *out = h;
    return unsupported("intra block copy (allow_intrabc) is not implemented", error);
  }

  if (!key_or_intra) {
    h.allow_high_precision_mv = h.force_integer_mv ? false : r.read_bit();
    if (!seq.enable_dual_filter) {
      h.is_filter_switchable = r.read_bit();
      if (h.is_filter_switchable) {
        h.interpolation_filter = 3; /* SWITCHABLE */
      } else {
        h.interpolation_filter = static_cast<int>(r.read(2));
      }
    } else {
      h.interpolation_filter = static_cast<int>(r.read(2));
    }
    h.is_motion_mode_switchable = r.read_bit();
    if (!h.error_resilient_mode && seq.enable_ref_frame_mvs) {
      h.use_ref_frame_mvs = r.read_bit();
    }
  }

  if (!seq.reduced_still_picture_header && !h.disable_cdf_update) {
    h.refresh_frame_context = !r.read_bit();
  }

  /* ---- tile_info() ------------------------------------------------------- */
  TileInfo& t = h.tiles;
  t.uniform = r.read_bit();
  const int sbsz_min1 = (64 << sb128) - 1;
  const int sbsz_log2 = 6 + sb128;
  const int sbw = (h.width + sbsz_min1) >> sbsz_log2; /* upscaled width */
  const int sbh = (h.height + sbsz_min1) >> sbsz_log2;
  const int max_tile_width_sb = 4096 >> sbsz_log2;
  const int max_tile_area_sb = (4096 * 2304) >> (2 * sbsz_log2);
  const int min_log2_cols = tile_log2(max_tile_width_sb, sbw);
  const int max_log2_cols = tile_log2(1, std::min(sbw, 64));
  const int max_log2_rows = tile_log2(1, std::min(sbh, 64));
  int min_log2_rows = 0;
  const int min_log2_tiles =
      std::max(tile_log2(max_tile_area_sb, sbw * sbh), min_log2_cols);

  if (t.uniform) {
    for (t.log2_cols = min_log2_cols; t.log2_cols < max_log2_cols;) {
      if (!r.read_bit()) break;
      ++t.log2_cols;
    }
    const int tile_width_sb = 1 + ((sbw - 1) >> t.log2_cols);
    for (int sbx = 0; sbx < sbw; sbx += tile_width_sb)
      t.col_start_sb.push_back(sbx);
    t.cols = static_cast<int>(t.col_start_sb.size());
    min_log2_rows = std::max(min_log2_tiles - t.log2_cols, 0);
    for (t.log2_rows = min_log2_rows; t.log2_rows < max_log2_rows;) {
      if (!r.read_bit()) break;
      ++t.log2_rows;
    }
    const int tile_height_sb = 1 + ((sbh - 1) >> t.log2_rows);
    for (int sby = 0; sby < sbh; sby += tile_height_sb)
      t.row_start_sb.push_back(sby);
    t.rows = static_cast<int>(t.row_start_sb.size());
  } else {
    int widest_tile = 0;
    int max_tile_area = sbw * sbh;
    for (int sbx = 0; sbx < sbw && t.cols < 64; ++t.cols) {
      const int tile_width_sb = std::min(sbw - sbx, max_tile_width_sb);
      const int tile_width = tile_width_sb > 1
                                 ? 1 + static_cast<int>(r.read_uniform(
                                           static_cast<uint32_t>(tile_width_sb)))
                                 : 1;
      t.col_start_sb.push_back(sbx);
      sbx += tile_width;
      widest_tile = std::max(widest_tile, tile_width);
    }
    t.log2_cols = tile_log2(1, t.cols);
    if (min_log2_tiles) max_tile_area >>= min_log2_tiles + 1;
    const int max_tile_height_sb = std::max(max_tile_area / widest_tile, 1);
    for (int sby = 0; sby < sbh && t.rows < 64; ++t.rows) {
      const int tile_height_sb = std::min(sbh - sby, max_tile_height_sb);
      const int tile_height = tile_height_sb > 1
                                  ? 1 + static_cast<int>(r.read_uniform(
                                            static_cast<uint32_t>(tile_height_sb)))
                                  : 1;
      t.row_start_sb.push_back(sby);
      sby += tile_height;
    }
    t.log2_rows = tile_log2(1, t.rows);
  }
  t.col_start_sb.push_back(sbw);
  t.row_start_sb.push_back(sbh);
  if (t.log2_cols || t.log2_rows) {
    t.context_update_tile_id = static_cast<int>(r.read(t.log2_cols + t.log2_rows));
    if (t.context_update_tile_id >= t.cols * t.rows) {
      return invalid("tile context update id out of range", error);
    }
    t.tile_size_bytes = static_cast<int>(r.read(2)) + 1;
  }

  /* ---- quantisation ------------------------------------------------------ */
  h.base_q_idx = static_cast<int>(r.read(8));
  if (r.read_bit()) h.delta_q_y_dc = r.read_signed(7);
  if (!seq.monochrome) {
    const bool diff_uv_delta = seq.separate_uv_delta_q ? r.read_bit() : false;
    if (r.read_bit()) h.delta_q_u_dc = r.read_signed(7);
    if (r.read_bit()) h.delta_q_u_ac = r.read_signed(7);
    if (diff_uv_delta) {
      if (r.read_bit()) h.delta_q_v_dc = r.read_signed(7);
      if (r.read_bit()) h.delta_q_v_ac = r.read_signed(7);
    } else {
      h.delta_q_v_dc = h.delta_q_u_dc;
      h.delta_q_v_ac = h.delta_q_u_ac;
    }
  }
  h.using_qmatrix = r.read_bit();
  if (h.using_qmatrix) {
    h.qm_y = static_cast<int>(r.read(4));
    h.qm_u = static_cast<int>(r.read(4));
    h.qm_v = seq.separate_uv_delta_q ? static_cast<int>(r.read(4)) : h.qm_u;
  }

  /* ---- segmentation ------------------------------------------------------ */
  SegmentationParams& seg = h.segmentation;
  seg.enabled = r.read_bit();
  h.segmentation_enabled = seg.enabled;
  if (seg.enabled) {
    if (h.primary_ref_frame == kPrimaryRefNone) {
      seg.update_map = r.read_bit();
      if (seg.update_map) {
        seg.temporal_update = r.read_bit();
        seg.update_data = r.read_bit();
      }
    } else {
      seg.update_data = r.read_bit();
      if (seg.update_data) {
        seg.update_map = r.read_bit();
        if (seg.update_map) seg.temporal_update = r.read_bit();
      }
    }
    if (seg.update_data) {
      for (int i = 0; i < kMaxSegments; ++i) {
        for (int j = 0; j < kSegFeatureCount; ++j) {
          seg.feature_enabled[i][j] = r.read_bit();
          if (seg.feature_enabled[i][j]) {
            seg.feature_data[i][j] = r.read_signed(kSegFeatureBits[j]);
          }
        }
      }
    }
  }

  /* ---- delta q / delta lf ------------------------------------------------ */
  if (h.base_q_idx > 0) {
    h.delta_q_present = r.read_bit();
    if (h.delta_q_present) {
      h.delta_q_res = static_cast<int>(r.read(2));
      if (!h.allow_intrabc) {
        h.delta_lf_present = r.read_bit();
        if (h.delta_lf_present) {
          h.delta_lf_res = static_cast<int>(r.read(2));
          h.delta_lf_multi = r.read_bit();
        }
      }
    }
  }

  const bool delta_lossless = h.delta_q_y_dc == 0 && h.delta_q_u_dc == 0 &&
                              h.delta_q_u_ac == 0 && h.delta_q_v_dc == 0 &&
                              h.delta_q_v_ac == 0;
  h.coded_lossless = h.base_q_idx == 0 && delta_lossless;
  h.all_lossless = h.coded_lossless;

  /* ---- loop filter ------------------------------------------------------- */
  LoopFilter& lf = h.loop_filter;
  if (h.all_lossless || h.allow_intrabc) {
    lf.mode_ref_delta_enabled = true;
    lf.mode_ref_delta_update = true;
  } else {
    lf.level_y[0] = static_cast<int>(r.read(6));
    lf.level_y[1] = static_cast<int>(r.read(6));
    if ((lf.level_y[0] || lf.level_y[1]) && !seq.monochrome) {
      lf.level_u = static_cast<int>(r.read(6));
      lf.level_v = static_cast<int>(r.read(6));
    }
    lf.sharpness = static_cast<int>(r.read(3));
    lf.mode_ref_delta_enabled = r.read_bit();
    if (lf.mode_ref_delta_enabled) {
      lf.mode_ref_delta_update = r.read_bit();
      if (lf.mode_ref_delta_update) {
        for (int i = 0; i < 8; ++i) {
          if (r.read_bit()) lf.ref_deltas[i] = r.read_signed(7);
        }
        for (int i = 0; i < 2; ++i) {
          if (r.read_bit()) lf.mode_deltas[i] = r.read_signed(7);
        }
      }
    }
  }

  /* ---- CDEF -------------------------------------------------------------- */
  if (!h.all_lossless && seq.enable_cdef && !h.allow_intrabc) {
    h.cdef.damping = static_cast<int>(r.read(2)) + 3;
    h.cdef.bits = static_cast<int>(r.read(2));
    const int strengths = 1 << h.cdef.bits;
    for (int i = 0; i < strengths; ++i) {
      h.cdef.y_strength[i] = static_cast<int>(r.read(6));
      if (!seq.monochrome) h.cdef.uv_strength[i] = static_cast<int>(r.read(6));
    }
  }

  /* ---- restoration ------------------------------------------------------- */
  if ((!h.all_lossless || h.superres) && seq.enable_restoration && !h.allow_intrabc) {
    h.restoration_type[0] = static_cast<int>(r.read(2));
    if (!seq.monochrome) {
      h.restoration_type[1] = static_cast<int>(r.read(2));
      h.restoration_type[2] = static_cast<int>(r.read(2));
    }
    if (h.restoration_type[0] || h.restoration_type[1] || h.restoration_type[2]) {
      h.restoration_unit_size[0] = 6 + sb128;
      if (r.read_bit()) {
        ++h.restoration_unit_size[0];
        if (!sb128) h.restoration_unit_size[0] += static_cast<int>(r.read_bit());
      }
      h.restoration_unit_size[1] = h.restoration_unit_size[0];
      if ((h.restoration_type[1] || h.restoration_type[2]) && seq.ss_x == 1 &&
          seq.ss_y == 1) {
        h.restoration_unit_size[1] -= static_cast<int>(r.read_bit());
      }
      *out = h;
      return unsupported("restoration filters (Wiener / self guided) are not implemented",
                         error);
    }
  }

  /* ---- transform mode ---------------------------------------------------- */
  if (!h.all_lossless) h.tx_mode_select = r.read_bit();

  /* ---- reference mode, skip mode and warped motion ----------------------- */
  if (key_or_intra) {
    h.reference_select = false;
    h.skip_mode_allowed = false;
    h.allow_warped_motion = false;
  } else {
    h.reference_select = r.read_bit();
    compute_skip_mode(seq, h, &r);
    if (h.error_resilient_mode || !seq.enable_warped_motion) {
      h.allow_warped_motion = false;
    } else {
      h.allow_warped_motion = r.read_bit();
    }
  }
  h.reduced_tx_set = r.read_bit();

  /* ---- global motion ----------------------------------------------------- */
  if (!key_or_intra) read_global_motion_params(seq, h, &r, refs);

  /* ---- film grain -------------------------------------------------------- */
  if (seq.film_grain_params_present && (h.show_frame || h.showable_frame)) {
    h.film_grain_present = r.read_bit();
    if (h.film_grain_present) {
      h.film_grain_seed = static_cast<int>(r.read(16));
      *out = h;
      return unsupported("film grain parameters are parsed, applying them from a real "
                         "bitstream is not implemented",
                         error);
    }
  }

  /* The header ends on a byte boundary; in a FRAME OBU the tile payload starts
   * right after it, so the padding bits are not required to be zero here. */
  if (r.overrun()) {
    *out = h;
    return invalid("frame header runs past the end of the OBU", error);
  }
  r.align_to_byte();
  h.tile_data_offset = r.bytes_read();

  /* Frame level sanity: geometry and tile grid have to agree. */
  if (h.width <= 0 || h.height <= 0) return invalid("frame has no area", error);
  if (h.tiles.cols <= 0 || h.tiles.rows <= 0) {
    return invalid("empty tile grid", error);
  }
  if (h.tiles.col_start_sb.back() != sbw || h.tiles.row_start_sb.back() != sbh) {
    return invalid("tile grid does not cover the frame", error);
  }
  if (h.superres) {
    *out = h;
    return unsupported("superres (coded width below display width) is not implemented",
                       error);
  }

  *out = h;
  return ParseStatus::Ok;
}

} /* namespace frontend */
} /* namespace av1gpu */
