#include "frontend/headers.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

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

/* Width of each segmentation feature value, in specification order, whether it
 * carries a sign, and the limit it is clipped to. A signed value is coded as
 * su(1 + bits), so it costs one bit more than the table says. */
constexpr int kSegFeatureBits[kSegFeatureCount] = {8, 6, 6, 6, 6, 3, 0, 0};
constexpr bool kSegFeatureSigned[kSegFeatureCount] = {true, true, true, true, true,
                                                      false, false, false};
/* SEG_LVL_SKIP and SEG_LVL_GLOBALMV carry no value: enabling them is the whole
 * signal, so they read zero bits and their limit is 0. */
constexpr int kSegFeatureMax[kSegFeatureCount] = {255, 63, 63, 63, 63, 7, 0, 0};

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
  /* the parameter is signed, so the scaling to the model precision is a
   * multiply rather than a left shift */
  const int decoded = decode_signed_subexp_with_ref(-mx, mx + 1, ref, r);
  gm->params[idx] = decoded * (1 << prec_diff) + round;
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
    /* PrevGmParams: the model the *primary reference frame* used, which is
     * what the coded difference is relative to. It is the frame named by
     * primary_ref_frame, not the reference this model describes: every model
     * of the frame is coded against the same previous frame's models. Without
     * a primary reference the defaults stand, which is what an identity model
     * holds. */
    GlobalMotion prev;
    if (refs && h.primary_ref_frame != kPrimaryRefNone) {
      const int slot = h.ref_frame_idx[h.primary_ref_frame];
      if (slot >= 0 && slot < 8) {
        prev = refs->slots[slot].global_motion[ref];
      }
    }

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

/* skip_mode_params(): skip mode predicts a block from the *pair* of references
 * that straddle this frame in display order, so it needs two of them - one in
 * the past, one in the future. When every reference is in the past it takes the
 * two nearest ones instead, which still gives a pair whose common motion is
 * worth reusing. A frame with a single usable reference allows nothing: there
 * would be no pair to name. */
void compute_skip_mode(const SequenceHeader& seq, FrameHeader& h, BitReader* r) {
  h.skip_mode_allowed = false;
  h.skip_mode_present = false;
  h.skip_mode_frame[0] = 0;
  h.skip_mode_frame[1] = 0;
  /* a frame that does not switch between single and compound prediction never
   * codes a skip mode frame pair */
  if (!h.reference_select || !seq.enable_order_hint) return;

  /* the nearest reference in the past and the nearest one in the future */
  int forward_idx = -1;
  int backward_idx = -1;
  int forward_hint = 0;
  for (int i = 0; i < 7; ++i) {
    const int ref_hint = h.ref_order_hint[i];
    const int dist = get_relative_dist(seq, ref_hint, h.order_hint);
    if (dist < 0) {
      if (forward_idx < 0 || get_relative_dist(seq, ref_hint, forward_hint) > 0) {
        forward_idx = i;
        forward_hint = ref_hint;
      }
    } else if (dist > 0) {
      if (backward_idx < 0 || get_relative_dist(seq, ref_hint, h.ref_order_hint[backward_idx]) < 0) {
        backward_idx = i;
      }
    }
  }

  if (forward_idx < 0) return;
  if (backward_idx >= 0) {
    h.skip_mode_allowed = true;
    h.skip_mode_frame[0] = std::min(forward_idx, backward_idx) + 1;
    h.skip_mode_frame[1] = std::max(forward_idx, backward_idx) + 1;
    h.skip_mode_present = r->read_bit();
    return;
  }

  /* no future reference: the second nearest past one completes the pair */
  int second_idx = -1;
  int second_hint = 0;
  for (int i = 0; i < 7; ++i) {
    const int ref_hint = h.ref_order_hint[i];
    if (get_relative_dist(seq, ref_hint, forward_hint) < 0) {
      if (second_idx < 0 || get_relative_dist(seq, ref_hint, second_hint) > 0) {
        second_idx = i;
        second_hint = ref_hint;
      }
    }
  }
  if (second_idx < 0) return;
  h.skip_mode_allowed = true;
  h.skip_mode_frame[0] = std::min(forward_idx, second_idx) + 1;
  h.skip_mode_frame[1] = std::max(forward_idx, second_idx) + 1;
  h.skip_mode_present = r->read_bit();
}


/* AV1GPU_HDRPOS=1 prints, in bits, where a few fields sit in the payload they
 * were read from. It exists to build test streams: a fixture that has to say
 * "this frame is superres" has to be *patched* rather than encoded (no encoder
 * available here can produce one), and this turns that into "write these bits"
 * instead of a second copy of the header syntax in a script. */
bool hdrpos_enabled() {
  static const bool on = std::getenv("AV1GPU_HDRPOS") != nullptr;
  return on;
}

} /* namespace */

/* get_relative_dist(): sign extended distance between two order hints. */
int get_relative_dist(const SequenceHeader& seq, int a, int b) {
  if (!seq.enable_order_hint) return 0;
  int diff = a - b;
  const int m = 1 << (seq.order_hint_n_bits - 1);
  return (diff & (m - 1)) - (diff & m);
}

int project_mv_component(int mv, int num, int den) {
  /* 2^14 / den, rounded, for the denominators the projection can produce */
  static const int32_t kDivMult[32] = {
      0,    16384, 8192, 5461, 4096, 3276, 2730, 2340, 2048, 1820, 1638,
      1489, 1365,  1260, 1170, 1092, 1024, 963,  910,  862,  819,  780,
      744,  712,   682,  655,  630,  606,  585,  564,  546,  528};
  if (den <= 0 || den >= 32) return 0;
  /* 64 bit, so that the product cannot overflow for any input a stored vector
   * can hold (the reference decoders cap them at 4095, which fits in 32 bits,
   * but a wider intermediate costs nothing and removes the question) */
  const int64_t scaled = static_cast<int64_t>(mv) * num * kDivMult[den];
  /* add the rounding term and the sign bit so that negative values round away
   * from zero as well */
  const int64_t rounded = (scaled + 8192 + (scaled >> 63)) >> 14;
  return std::max(-0x3fff, std::min(0x3fff, static_cast<int>(rounded)));
}

void refresh_ref_pool(const FrameHeader& frame, RefPool* pool,
                      const std::shared_ptr<const RefMvs>& ref_mvs,
                      const std::shared_ptr<const CdfContext>& cdf,
                      const std::shared_ptr<const SegmentationMap>& seg_map) {
  if (!pool || frame.show_existing_frame) return;
  /* An intra frame codes no motion vectors, so it clears the grid of every slot
   * it refreshes: a later frame must not project the motion vectors of a frame
   * that is no longer in that slot. */
  const bool stores_mvs = frame.type == FrameType::Inter || frame.type == FrameType::Switch;
  for (int i = 0; i < 8; ++i) {
    if (frame.refresh_frame_flags & (1 << i)) {
      RefSlot& slot = pool->slots[i];
      slot.valid = true;
      slot.upscaled_width = frame.upscaled_width;
      slot.width = frame.width;
      slot.height = frame.height;
      slot.render_width = frame.render_width;
      slot.render_height = frame.render_height;
      slot.order_hint = frame.order_hint;
      slot.frame_id = frame.frame_id_present ? frame.frame_id : -1;
      slot.type = frame.type;
      for (int ref = 0; ref < 8; ++ref) slot.global_motion[ref] = frame.global_motion[ref];
      for (int ref = 0; ref < 7; ++ref) slot.ref_poc[ref] = frame.ref_order_hint[ref];
      slot.ref_mvs = stores_mvs ? ref_mvs : nullptr;
      slot.cdf = cdf;
      slot.seg_map = seg_map;
      slot.segmentation = frame.segmentation;
      /* save_grain_params(): a later frame with update_grain == 0 reads these */
      slot.grain = frame.grain;
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
  const size_t pos_max_frame_width = r.bits_read();
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

  const size_t pos_enable_superres = r.bits_read();
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
  const size_t pos_film_grain_present = r.bits_read();
  seq.film_grain_params_present = r.read_bit();

  const ParseStatus status = check_trailing_bits(&r, error);
  if (status != ParseStatus::Ok) return status;

  *out = seq;
  if (hdrpos_enabled()) {
    std::fprintf(stderr,
                 "HDRPOS seq max_frame_width=%zu max_frame_height=%zu enable_superres=%zu "
                 "width_bits=%d film_grain=%zu mono=%d ss_x=%d ss_y=%d bitdepth=%d\n",
                 pos_max_frame_width, pos_max_frame_width + static_cast<size_t>(seq.width_n_bits),
                 pos_enable_superres, seq.width_n_bits, pos_film_grain_present,
                 seq.monochrome ? 1 : 0, seq.ss_x, seq.ss_y, seq.bit_depth);
  }
  return ParseStatus::Ok;
}

/* set_frame_refs(): 7.8. A frame may code just LAST_FRAME and GOLDEN_FRAME and
 * let the decoder fill in the other five from the order hints the slots carry:
 * the latest and earliest *backward* references (in the future relative to this
 * frame) become ALTREF and BWDREF, the next closest backward one ALTREF2, and
 * the remaining forward references go to LAST2, LAST3 and whatever is left in
 * anti-chronological order. Everything is compared on a shifted hint scale
 * where this frame sits at 1 << (OrderHintBits - 1), so "before" and "after"
 * reduce to comparing integers.
 *
 * The specification notes that ties are common - several slots can carry the
 * same order hint - and that the reference decoder breaks them by slot index.
 * The scans below do the same by visiting slots in order and only replacing
 * the current best on a strict improvement. */
void set_frame_refs(const SequenceHeader& seq, FrameHeader* h, const RefPool& pool) {
  /* Indices into ref_frame_idx, i.e. the reference type minus one. */
  const int kLastIdx = 0, kLast2Idx = 1, kLast3Idx = 2, kGoldenIdx = 3;
  const int kBwdrefIdx = 4, kAltref2Idx = 5, kAltrefIdx = 6;
  const int refs_per_frame = 7;
  for (int i = 0; i < refs_per_frame; ++i) h->ref_frame_idx[i] = -1;
  /* the two the frame actually codes */
  const int last_idx = h->last_frame_idx;
  const int gold_idx = h->gold_frame_idx;
  if (last_idx < 0 || last_idx > 7 || gold_idx < 0 || gold_idx > 7) return;
  h->ref_frame_idx[kLastIdx] = last_idx;
  h->ref_frame_idx[kGoldenIdx] = gold_idx;

  bool used[8] = {false, false, false, false, false, false, false, false};
  used[last_idx] = true;
  used[gold_idx] = true;

  const int cur = 1 << (seq.order_hint_n_bits - 1);
  int shifted[8];
  for (int i = 0; i < 8; ++i) {
    const int hint = pool.slots[i].valid ? pool.slots[i].order_hint : 0;
    shifted[i] = cur + get_relative_dist(seq, hint, h->order_hint);
  }

  /* the frame with the highest output order at or after this one */
  auto latest_backward = [&]() {
    int ref = -1, best = 0;
    for (int i = 0; i < 8; ++i) {
      if (!pool.slots[i].valid || used[i]) continue;
      const int hint = shifted[i];
      if (hint < cur) continue;
      if (ref < 0 || hint >= best) {
        ref = i;
        best = hint;
      }
    }
    return ref;
  };
  /* ... and the one with the lowest */
  auto earliest_backward = [&]() {
    int ref = -1, best = 0;
    for (int i = 0; i < 8; ++i) {
      if (!pool.slots[i].valid || used[i]) continue;
      const int hint = shifted[i];
      if (hint < cur) continue;
      if (ref < 0 || hint < best) {
        ref = i;
        best = hint;
      }
    }
    return ref;
  };
  /* the latest frame strictly before this one */
  auto latest_forward = [&]() {
    int ref = -1, best = 0;
    for (int i = 0; i < 8; ++i) {
      if (!pool.slots[i].valid || used[i]) continue;
      const int hint = shifted[i];
      if (hint >= cur) continue;
      if (ref < 0 || hint >= best) {
        ref = i;
        best = hint;
      }
    }
    return ref;
  };

  int ref = latest_backward();
  if (ref >= 0) {
    h->ref_frame_idx[kAltrefIdx] = ref;
    used[ref] = true;
  }
  ref = earliest_backward();
  if (ref >= 0) {
    h->ref_frame_idx[kBwdrefIdx] = ref;
    used[ref] = true;
  }
  ref = earliest_backward();
  if (ref >= 0) {
    h->ref_frame_idx[kAltref2Idx] = ref;
    used[ref] = true;
  }

  /* Ref_Frame_List, as indices: the two the frame did not code, then BWDREF,
   * ALTREF2 and ALTREF in case the backward scans found nothing for them and
   * they still need a forward frame. */
  static const int kRemaining[] = {kLast2Idx, kLast3Idx, kBwdrefIdx, kAltref2Idx, kAltrefIdx};
  for (int frame : kRemaining) {
    if (h->ref_frame_idx[frame] >= 0) continue;
    ref = latest_forward();
    if (ref >= 0) {
      h->ref_frame_idx[frame] = ref;
      used[ref] = true;
    }
  }

  /* Anything still unset takes the frame with the lowest output order overall -
   * a pool this narrow only happens in a broken stream, but the specification
   * names a value for every slot, so there is no "undefined" case. */
  int fallback = -1, earliest = 0;
  for (int i = 0; i < 8; ++i) {
    if (!pool.slots[i].valid) continue;
    if (fallback < 0 || shifted[i] < earliest) {
      fallback = i;
      earliest = shifted[i];
    }
  }
  for (int i = 0; i < refs_per_frame; ++i) {
    if (h->ref_frame_idx[i] < 0) h->ref_frame_idx[i] = fallback;
  }
}

/* film_grain_params() (5.9.30) after apply_grain, grain_seed and the
 * update_grain decision: the pointwise scaling curves, the auto-regressive
 * filter that shapes the noise, and how it is added back. */
ParseStatus read_film_grain_params(const SequenceHeader& seq, FilmGrainParams* g, BitReader* r,
                                   std::string* error) {
  g->num_y_points = static_cast<int>(r->read(4));
  for (int i = 0; i < g->num_y_points; ++i) {
    g->point_y_value[i] = static_cast<int>(r->read(8));
    /* the values are coded in increasing order and the scaling curve is
     * interpolated between them, so a decreasing pair is an error */
    if (i > 0 && g->point_y_value[i] <= g->point_y_value[i - 1]) {
      return invalid("film grain luma points are not strictly increasing", error);
    }
    g->point_y_scaling[i] = static_cast<int>(r->read(8));
  }
  const bool mono = seq.monochrome;
  g->chroma_scaling_from_luma = !mono && r->read_bit() != 0;
  /* With 4:2:0 and no luma points the chroma grain would have nothing to be
   * correlated with, so the syntax does not code any chroma points at all. */
  if (mono || g->chroma_scaling_from_luma || (seq.ss_x && seq.ss_y && g->num_y_points == 0)) {
    g->num_cb_points = 0;
    g->num_cr_points = 0;
  } else {
    g->num_cb_points = static_cast<int>(r->read(4));
    for (int i = 0; i < g->num_cb_points; ++i) {
      g->point_cb_value[i] = static_cast<int>(r->read(8));
      g->point_cb_scaling[i] = static_cast<int>(r->read(8));
    }
    g->num_cr_points = static_cast<int>(r->read(4));
    for (int i = 0; i < g->num_cr_points; ++i) {
      g->point_cr_value[i] = static_cast<int>(r->read(8));
      g->point_cr_scaling[i] = static_cast<int>(r->read(8));
    }
  }
  g->grain_scaling_minus_8 = static_cast<int>(r->read(2));
  g->ar_coeff_lag = static_cast<int>(r->read(2));
  const int num_pos_luma = 2 * g->ar_coeff_lag * (g->ar_coeff_lag + 1);
  int num_pos_chroma = num_pos_luma;
  if (g->num_y_points) {
    /* one coefficient per chroma position that reads the *luma* grain follows
     * the luma ones, so the chroma filter has one position more */
    num_pos_chroma = num_pos_luma + 1;
    for (int i = 0; i < num_pos_luma; ++i) g->ar_coeffs_y[i] = static_cast<int>(r->read(8)) - 128;
  }
  if (g->chroma_scaling_from_luma || g->num_cb_points) {
    for (int i = 0; i < num_pos_chroma; ++i) {
      g->ar_coeffs_cb[i] = static_cast<int>(r->read(8)) - 128;
    }
  }
  if (g->chroma_scaling_from_luma || g->num_cr_points) {
    for (int i = 0; i < num_pos_chroma; ++i) {
      g->ar_coeffs_cr[i] = static_cast<int>(r->read(8)) - 128;
    }
  }
  g->ar_coeff_shift_minus_6 = static_cast<int>(r->read(2));
  g->grain_scale_shift = static_cast<int>(r->read(2));
  if (g->num_cb_points) {
    g->cb_mult = static_cast<int>(r->read(8));
    g->cb_luma_mult = static_cast<int>(r->read(8));
    g->cb_offset = static_cast<int>(r->read(9));
  }
  if (g->num_cr_points) {
    g->cr_mult = static_cast<int>(r->read(8));
    g->cr_luma_mult = static_cast<int>(r->read(8));
    g->cr_offset = static_cast<int>(r->read(9));
  }
  g->overlap_flag = r->read_bit() != 0;
  g->clip_to_restricted_range = r->read_bit() != 0;
  return ParseStatus::Ok;
}

ParseStatus parse_frame_header(const SequenceHeader& seq, const uint8_t* data, size_t size,
                               FrameHeader* out, std::string* error, RefPool* refs) {
  if (!data || size == 0) return invalid("empty frame header", error);
  const int sb128 = seq.use_128x128_superblock ? 1 : 0;

  FrameHeader h;
  BitReader r(data, size);
  /* Where the superres related fields sit, for AV1GPU_HDRPOS (see above). */
  size_t pos_frame_size_width = SIZE_MAX;
  size_t pos_use_superres = SIZE_MAX;
  size_t pos_allow_intrabc = SIZE_MAX;

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
      /* display_frame_id: the id the frame being shown was stored with. It is
       * not an id of this "frame" - a shown frame has none of its own. */
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
    if (h.frame_id_present && slot.frame_id >= 0 && slot.frame_id != h.frame_id) {
      *out = h;
      return invalid("show_existing_frame names a frame id the slot does not hold", error);
    }
    /* The frame is a repeat of a stored one: report its geometry so callers can
     * display it without decoding anything. */
    h.type = slot.type;
    h.show_frame = true;
    h.showable_frame = true;
    h.width = slot.width;
    h.height = slot.height;
    h.upscaled_width = slot.upscaled_width > 0 ? slot.upscaled_width : slot.width;
    h.render_width = slot.render_width > 0 ? slot.render_width : h.upscaled_width;
    h.render_height = slot.render_height > 0 ? slot.render_height : slot.height;
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
  /* A shown key frame starts a new coded video sequence, so every reference the
   * decoder was holding belongs to the sequence that just ended: 5.9.2 clears
   * RefValid[] here, before the refresh flags at the end of the header fill all
   * eight slots again with this frame. */
  if (h.type == FrameType::Key && h.show_frame && refs) {
    for (int i = 0; i < 8; ++i) {
      refs->slots[i].valid = false;
      refs->slots[i].frame_id = -1;
      refs->slots[i].order_hint = 0; /* RefOrderHint[i] = 0 as well */
    }
  }
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
  /* 5.9.2 sets it again right after: "if (FrameIsIntra) force_integer_mv = 1".
   * An intra frame has no reference frame to interpolate, and its one motion
   * vector syntax - intra block copy - is defined in whole samples, so the flag
   * is not so much read as overridden. It changes no syntax element on an intra
   * frame (allow_high_precision_mv, the only other reader, is coded on inter
   * frames only), but it is the frame property the reconstruction and the
   * vector precision rules look at. */
  if (key_or_intra) h.force_integer_mv = true;

  if (seq.frame_id_numbers_present) {
    h.frame_id_present = true;
    h.frame_id = static_cast<int>(r.read(seq.frame_id_n_bits));
    /* mark_ref_frames(): a slot whose id has fallen outside the window the
     * delta_frame_id field can express can no longer be named, so it is marked
     * empty. That is how a decoder notices a lost reference frame instead of
     * predicting from the wrong picture. The window is 1 << deltaFrameIdLen
     * wide and wraps around the id space, so the test has two branches
     * depending on which side of the wrap this frame's id sits. */
    if (refs) {
      const int span = 1 << seq.delta_frame_id_n_bits;
      const int limit = 1 << seq.frame_id_n_bits;
      const int id = h.frame_id;
      for (int i = 0; i < 8; ++i) {
        RefSlot& slot = refs->slots[i];
        if (!slot.valid || slot.frame_id < 0) continue;
        const bool stale =
            id > span ? (slot.frame_id > id || slot.frame_id < id - span)
                      : (slot.frame_id > id && slot.frame_id < limit + id - span);
        if (stale) {
          slot.valid = false;
          slot.frame_id = -1;
        }
      }
    }
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
  /* An error resilient frame re-states the order hint of every slot, and a slot
   * that disagrees with what the decoder holds is marked empty: the encoder is
   * telling it that its idea of the pool is stale. The outer test is the
   * specification's "this is not a fresh intra restart", which needs no
   * correction. */
  const bool reorder_hints =
      (!key_or_intra || h.refresh_frame_flags != 0xff) && h.error_resilient_mode &&
      seq.enable_order_hint;
  if (reorder_hints) {
    for (int i = 0; i < 8; ++i) {
      const int hint = static_cast<int>(r.read(seq.order_hint_n_bits));
      if (!refs) continue;
      if (refs->slots[i].order_hint != hint) {
        refs->slots[i].valid = false;
        refs->slots[i].frame_id = -1;
      }
    }
  }

  /* frame_refs(): which of the eight slots this frame predicts from. */
  if (!key_or_intra) {
    if (!refs) {
      *out = h;
      return unsupported("inter frames need the reference frame pool", error);
    }
    if (r.read_bit()) {
      /* frame_refs_short_signaling: only the two slots the encoder committed to
       * are coded, the other five come from the order hints (set_frame_refs). */
      if (!seq.enable_order_hint) {
        *out = h;
        return invalid("frame_refs_short_signaling needs order hints", error);
      }
      h.last_frame_idx = static_cast<int>(r.read(3));
      h.gold_frame_idx = static_cast<int>(r.read(3));
      set_frame_refs(seq, &h, *refs);
    } else {
      for (int i = 0; i < 7; ++i) h.ref_frame_idx[i] = static_cast<int>(r.read(3));
    }
    /* With frame ids the header names, for each reference, how far back its id
     * has to be: a decoder that cannot match the value knows the slot does not
     * hold the frame the encoder meant and says so instead of predicting from
     * the wrong picture. */
    if (seq.frame_id_numbers_present) {
      const int id_len = seq.frame_id_n_bits;
      const int n = seq.delta_frame_id_n_bits;
      for (int i = 0; i < 7; ++i) {
        const int delta = static_cast<int>(r.read(n)) + 1;
        const int expected = (h.frame_id + (1 << id_len) - delta) % (1 << id_len);
        const int slot = h.ref_frame_idx[i];
        if (slot < 0 || slot > 7) {
          *out = h;
          return invalid("a reference frame index is out of range", error);
        }
        if (refs->slots[slot].frame_id != expected) {
          *out = h;
          return invalid("a reference frame id does not match the frame its slot holds",
                         error);
        }
      }
    }
    /* 7.8 requires every reference this frame codes to name a frame the
     * decoder holds. Saying so here, once, is what keeps a stale slot from
     * being predicted from - the order hints below and everything the tile
     * decoder reads depend on it. */
    for (int i = 0; i < 7; ++i) {
      const int slot = h.ref_frame_idx[i];
      if (slot < 0 || slot > 7) {
        *out = h;
        return invalid("a reference frame index is out of range", error);
      }
      if (!refs->slots[slot].valid) {
        *out = h;
        return invalid("an inter frame refers to an empty reference slot", error);
      }
    }
    for (int i = 0; i < 7; ++i) {
      h.ref_order_hint[i] = refs->slots[h.ref_frame_idx[i]].order_hint;
    }
  }

  /* frame_size() / frame_size_with_refs() (5.9.4, 5.9.5): an inter frame with
   * the override on may take its geometry from a reference it names; otherwise
   * it codes a size or, with the override off, is the sequence's maximum. */
  bool size_from_ref = false;
  if (!key_or_intra && h.frame_size_override && !h.error_resilient_mode) {
    /* frame_size_with_refs(): one found_ref bit per reference, in reference
     * order, and the first one that is set names the frame this one copies its
     * whole geometry from - the bits after it are not coded at all. An encoder
     * that scales a frame relative to its reference says so this way. */
    int found = -1;
    for (int i = 0; i < 7; ++i) {
      if (!r.read_bit()) continue;
      found = i;
      break;
    }
    if (found >= 0) {
      const RefSlot& slot = refs->slots[h.ref_frame_idx[found]];
      if (!slot.valid) {
        *out = h;
        return invalid("frame size taken from an empty reference slot", error);
      }
      h.size_from_reference = true;
      /* found_ref copies the reference's *upscaled* width: a frame that takes
       * its size from a reference takes the display size, and decides its own
       * superres denom below */
      h.upscaled_width = slot.upscaled_width > 0 ? slot.upscaled_width : slot.width;
      h.height = slot.height;
      h.render_width = slot.render_width;
      h.render_height = slot.render_height;
      size_from_ref = true;
    }
  }
  if (!size_from_ref) {
    /* frame_size(): with the override the frame codes its own size, without it
     * the frame is the sequence's maximum size. It is *not* inherited from a
     * reference without found_ref - a smaller reference does not make a frame
     * smaller. */
    if (h.frame_size_override) {
      pos_frame_size_width = r.bits_read();
      h.upscaled_width = static_cast<int>(r.read(seq.width_n_bits)) + 1;
      h.height = static_cast<int>(r.read(seq.height_n_bits)) + 1;
    } else {
      h.upscaled_width = seq.max_frame_width;
      h.height = seq.max_frame_height;
    }
  }
  h.width = h.upscaled_width;
  /* superres_params() is read on both paths: a frame that took its size from a
   * reference still decides whether to scale that size down. */
  /* the position is recorded even when the sequence has no superres: that is
   * where a patched stream has to insert the field (see AV1GPU_HDRPOS) */
  pos_use_superres = r.bits_read();
  if (seq.enable_superres) {
    h.superres = r.read_bit();
    if (h.superres) {
      h.superres_denominator = 9 + static_cast<int>(r.read(3));
      const int denominator = h.superres_denominator;
      /* FrameWidth = max((UpscaledWidth * 8 + SuperresDenom / 2) / SuperresDenom,
       *                 min(16, UpscaledWidth)): eight eighths of the display width
       * per denominator eighth, never below 16 samples. */
      h.width = std::max((h.upscaled_width * 8 + (denominator >> 1)) / denominator,
                         std::min(16, h.upscaled_width));
    }
  }
  /* render_size() is not coded when the size came from a reference: the render
   * size was inherited with it. */
  if (!size_from_ref) {
    const bool have_render_size = r.read_bit();
    if (have_render_size) {
      h.render_width = static_cast<int>(r.read(16)) + 1;
      h.render_height = static_cast<int>(r.read(16)) + 1;
    } else {
      /* with no render size coded the frame renders at its *display* size */
      h.render_width = h.upscaled_width;
      h.render_height = h.height;
    }
  }
  /* allow_intrabc is only present on an intra frame: the specification reads
   * it inside `if (FrameIsIntra)`, and UpscaledWidth == FrameWidth rules out
   * superres. Reading it on an inter frame shifts every following field. */
  pos_allow_intrabc = r.bits_read();
  if (key_or_intra && h.allow_screen_content_tools && !h.superres) {
    h.allow_intrabc = r.read_bit();
  } else {
    pos_allow_intrabc = SIZE_MAX;
  }
  /* allow_intrabc only says the *frame* may use intra block copy: the flag is
   * per block (use_intrabc), and an encoder sets it for the whole frame as soon
   * as it wants the option, whether or not a block takes it. A frame that
   * allows it also codes no CDEF, no restoration and no restore/delta LF
   * levels (see below) - all of which follow from this flag rather than from
   * the blocks, so it changes the header even when no block copies. */

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
    /* A frame that resets the context (a key frame, or one that names no
     * primary reference) does not code the three flags at all: it updates its
     * map, does not predict it temporally, and carries its own feature data. */
    if (h.primary_ref_frame == kPrimaryRefNone) {
      seg.update_map = true;
      seg.temporal_update = false;
      seg.update_data = true;
    } else {
      seg.update_map = r.read_bit();
      if (seg.update_map) seg.temporal_update = r.read_bit();
      seg.update_data = r.read_bit();
    }
    if (seg.update_data) {
      for (int i = 0; i < kMaxSegments; ++i) {
        for (int j = 0; j < kSegFeatureCount; ++j) {
          seg.feature_enabled[i][j] = r.read_bit();
          if (seg.feature_enabled[i][j]) {
            const int limit = kSegFeatureMax[j];
            if (kSegFeatureSigned[j]) {
              /* su(1 + bits): the sign is one bit of the value */
              const int value = r.read_signed(kSegFeatureBits[j] + 1);
              seg.feature_data[i][j] = std::max(-limit, std::min(limit, value));
            } else {
              const int value = static_cast<int>(r.read(kSegFeatureBits[j]));
              seg.feature_data[i][j] = std::max(0, std::min(limit, value));
            }
          }
        }
      }
    } else if (refs && h.primary_ref_frame != kPrimaryRefNone) {
      /* segmentation_update_data == 0: the features of this frame are the ones
       * the primary reference was decoded with, so the header carries none. */
      const int slot = h.ref_frame_idx[h.primary_ref_frame];
      if (slot >= 0 && slot < 8 && refs->slots[slot].valid) {
        const SegmentationParams& from = refs->slots[slot].segmentation;
        for (int i = 0; i < kMaxSegments; ++i) {
          for (int j = 0; j < kSegFeatureCount; ++j) {
            seg.feature_enabled[i][j] = from.feature_enabled[i][j];
            seg.feature_data[i][j] = from.feature_data[i][j];
          }
        }
      }
    }
  }

  /* 5.11.9 derives three things from the features that are read here: the
   * highest segment that has any feature at all (the segment id syntax only
   * codes values up to it), whether any segment carries a feature that changes
   * syntax read before the skip flag (SEG_LVL_REF_FRAME and up), and which
   * segments are lossless. */
  for (int i = 0; i < kMaxSegments; ++i) {
    for (int j = 0; j < kSegFeatureCount; ++j) {
      if (!seg.feature_enabled[i][j]) continue;
      seg.last_active_seg_id = i;
      if (j >= kSegRefFrame) seg.seg_id_pre_skip = true;
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

  /* LosslessArray: a *segment* is lossless when its quantiser - the base one
   * plus its SEG_LVL_ALT_Q, ignoring the superblock delta - is zero and no
   * delta Q of the frame is set. CodedLossless is then true only when every
   * segment is, because a frame with one lossless segment still codes
   * coefficients for the others. */
  const bool delta_lossless = h.delta_q_y_dc == 0 && h.delta_q_u_dc == 0 &&
                              h.delta_q_u_ac == 0 && h.delta_q_v_dc == 0 &&
                              h.delta_q_v_ac == 0;
  h.coded_lossless = true;
  for (int i = 0; i < kMaxSegments; ++i) {
    int qindex = h.base_q_idx;
    if (segment_feature_active(seg, i, kSegAltQ)) qindex += seg.feature_data[i][kSegAltQ];
    const int clipped = std::max(0, std::min(255, qindex));
    seg.lossless[i] = clipped == 0 && delta_lossless;
    if (!seg.lossless[i]) h.coded_lossless = false;
  }
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
      /* LoopRestorationSize[0] = RESTORATION_TILESIZE_MAX >> (2 - lr_unit_shift),
       * i.e. 64, 128 or 256 samples across, and the chroma size is the same or
       * half of it. Everything downstream only needs the base two logarithm. */
      int log2_size = 6 + sb128;
      if (r.read_bit()) {
        ++log2_size;
        if (!sb128) log2_size += static_cast<int>(r.read_bit());
      }
      h.restoration_unit_log2[0] = log2_size;
      h.restoration_unit_log2[1] = log2_size;
      h.restoration_unit_log2[2] = log2_size;
      if ((h.restoration_type[1] || h.restoration_type[2]) && seq.ss_x == 1 &&
          seq.ss_y == 1) {
        /* lr_uv_shift: the chroma units are the same size as the luma ones or
         * half of them, and the bit says which */
        if (r.read_bit()) --h.restoration_unit_log2[1];
        h.restoration_unit_log2[2] = h.restoration_unit_log2[1];
      }
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
    FilmGrainParams& g = h.grain;
    g.apply = r.read_bit();
    if (g.apply) {
      g.seed = static_cast<int>(r.read(16));
      bool update_grain = true;
      if (h.type == FrameType::Inter) update_grain = r.read_bit();
      if (!update_grain) {
        /* The frame re-uses a reference's grain parameters and codes only its
         * own seed (5.9.30): the rest comes from the slot it names, the way
         * that slot was stored. */
        const int idx = static_cast<int>(r.read(3));
        if (!refs || !refs->slots[idx].valid) {
          *out = h;
          return invalid("film grain parameters taken from an empty reference slot", error);
        }
        const int seed = g.seed;
        g = refs->slots[idx].grain;
        g.apply = true;
        g.seed = seed;
      } else {
        ParseStatus status = read_film_grain_params(seq, &g, &r, error);
        if (status != ParseStatus::Ok) {
          *out = h;
          return status;
        }
      }
    }
  }

  /* The header ends on a byte boundary; in a FRAME OBU the tile payload starts
   * right after it, so the padding bits are not required to be zero here. */
  if (r.overrun()) {
    *out = h;
    return invalid("frame header runs past the end of the OBU", error);
  }
  /* the last bit of the header proper, before the alignment padding: a patched
   * stream has to know it to realign (see AV1GPU_HDRPOS) */
  const size_t header_bits = r.bits_read();
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

  *out = h;
  if (hdrpos_enabled()) {
    std::fprintf(stderr,
                 "HDRPOS frame type=%d show=%d override=%d found=%d size_w=%zu "
                 "use_superres=%zu intrabc=%zu header_bits=%zu tile_data_offset=%zu "
                 "upscaled=%d coded=%d superres=%d denom=%d "
                 "ref0=%d ref1=%d ref2=%d ref3=%d ref4=%d ref5=%d ref6=%d "
                 "lf_y0=%d lf_y1=%d lf_u=%d lf_v=%d sharp=%d deltas=%d\n",
                 static_cast<int>(h.type), h.show_frame ? 1 : 0, h.frame_size_override ? 1 : 0,
                 h.size_from_reference ? 1 : 0, pos_frame_size_width, pos_use_superres,
                 pos_allow_intrabc, header_bits, h.tile_data_offset,
                 h.upscaled_width, h.width, h.superres ? 1 : 0, h.superres_denominator,
                 h.type == FrameType::Inter ? h.ref_frame_idx[0] : -1,
                 h.type == FrameType::Inter ? h.ref_frame_idx[1] : -1,
                 h.type == FrameType::Inter ? h.ref_frame_idx[2] : -1,
                 h.type == FrameType::Inter ? h.ref_frame_idx[3] : -1,
                 h.type == FrameType::Inter ? h.ref_frame_idx[4] : -1,
                 h.type == FrameType::Inter ? h.ref_frame_idx[5] : -1,
                 h.type == FrameType::Inter ? h.ref_frame_idx[6] : -1,
                 h.loop_filter.level_y[0], h.loop_filter.level_y[1], h.loop_filter.level_u,
                 h.loop_filter.level_v, h.loop_filter.sharpness,
                 h.loop_filter.mode_ref_delta_enabled ? 1 : 0);
  }
  return ParseStatus::Ok;
}

} /* namespace frontend */
} /* namespace av1gpu */
