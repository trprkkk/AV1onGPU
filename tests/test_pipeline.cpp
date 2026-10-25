/*
 * AV1onGPU - end to end pipeline tests
 *
 * Decodes a small synthetic clip with every available backend and checks that
 *   - the output matches the source picture (PSNR),
 *   - the GPU backends are bit identical to the CPU reference,
 *   - the wavefront scheduler is independent of its cell size,
 *   - reference frames survive the zero copy path.
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "av1gpu/av1gpu.hpp"
#include "av1gpu/synthetic.hpp"
#include "test_common.hpp"

using namespace av1gpu;

namespace {

struct Clip {
  SyntheticConfig cfg;
  std::vector<Frame> source;
  std::vector<FrameSyntax> syntax;
  std::vector<Frame> recon;
};

Clip make_clip(int frames, int qstep, bool cdef, bool grain, int w = 128, int h = 96) {
  Clip clip;
  clip.cfg.width = w;
  clip.cfg.height = h;
  clip.cfg.block_size = 16;
  clip.cfg.transform_size = 8;
  clip.cfg.qstep = qstep;
  clip.cfg.deblock_level = 12;
  clip.cfg.cdef = cdef;
  clip.cfg.grain = grain;
  clip.cfg.search_radius = 6;
  Frame ref;
  clip.source.resize(frames);
  clip.syntax.resize(frames);
  clip.recon.resize(frames);
  for (int f = 0; f < frames; ++f) {
    /* recon is the picture a decoder outputs (grain included), stored the one it
     * keeps as a reference (before the grain) */
    Frame stored;
    FrameSyntax syntax = synthetic_encode(clip.cfg, f, f == 0 ? nullptr : &ref, &clip.source[f],
                                          &clip.recon[f], &stored);
    clip.syntax[f] = syntax;
    ref = stored;
  }
  return clip;
}

/* A superres frame that also restores. No stream in the fixture set can code
 * this combination - making an LR stream superres changes the unit grid (7.17.1
 * sizes the units on the *upscaled* picture), so it cannot be reached by
 * patching a header - which is why it is built here: it is the only case the
 * backend has to keep a *second* upscaled picture (the pre-CDEF snapshot, which
 * a unit reads across a stripe border) and the only path in this tree that runs
 * 7.15 and 7.17 back to back.
 *
 * The coded picture stays at the clip's size and only the display size grows, so
 * the blocks the synthetic encoder wrote remain valid. The luma units get
 * `vfilter`/`hfilter`, the three coded coefficients of the two directions (each
 * gives seven taps, symmetric around 128 minus twice their sum). All zero is the
 * identity filter, which the stage has to reproduce the picture through;
 * anything else is a filter the specification defines exactly, so the test can
 * compute what it owes and compare. The unit counts follow 5.9.20, so the same
 * helper builds the units of a picture whose size is not a multiple of the unit
 * or of 64. */
/* count_units_in_frame() (5.9.20): the number of units a frame of `size`
 * samples holds, rounding to the nearest unit so a sliver of less than half a
 * unit at the far edge is not coded. */
int count_units(int unit_size, int size) {
  return std::max((size + (unit_size >> 1)) / unit_size, 1);
}

/* The units of one plane: the count is the specification's, and every unit gets
 * the same filter, so a wrong grid shows up as a wrong sample rather than as a
 * unit that happens to be skipped. */
void set_restoration_plane(FrameSyntax* syntax, int plane, int type, int unit_log2,
                           int plane_w, int plane_h, const RestorationUnit& unit) {
  const int unit_size = 1 << unit_log2;
  const int cols = count_units(unit_size, plane_w);
  const int rows = count_units(unit_size, plane_h);
  syntax->restoration_type[plane] = type;
  syntax->restoration_unit_log2[plane] = unit_log2;
  syntax->restoration_unit_cols[plane] = cols;
  syntax->restoration[plane].assign(static_cast<size_t>(cols) * rows, unit);
}

Clip make_superres_lr_clip(const Clip& base, int display_width, const int* vfilter,
                           const int* hfilter, int unit_log2 = 6) {
  Clip clip = base;
  const int height = base.cfg.height;
  for (FrameSyntax& syntax : clip.syntax) {
    syntax.width = display_width;
    syntax.coded_width = base.cfg.width;

    RestorationUnit wiener;
    wiener.type = kRestoreWiener;
    for (int i = 0; i < 3; ++i) {
      wiener.wiener[0][i] = vfilter[i];
      wiener.wiener[1][i] = hfilter[i];
    }
    set_restoration_plane(&syntax, 0, kRestoreWiener, unit_log2, display_width, height, wiener);

    /* chroma: the self guided filter, on units half the size - 4:2:0 is the one
     * layout whose chroma unit log2 drops by one (5.9.20), and these clips are
     * all 4:2:0 */
    RestorationUnit sgr;
    sgr.type = kRestoreSgrproj;
    sgr.sgr_set = 0; /* radius 2/eps 12, radius 1/eps 4 */
    sgr.sgr_xqd[0] = 0;
    sgr.sgr_xqd[1] = 95;
    for (int plane = 1; plane < 3; ++plane)
      set_restoration_plane(&syntax, plane, kRestoreSgrproj, unit_log2 - 1, display_width / 2,
                            height / 2, sgr);
  }
  return clip;
}

/* the same clip with restoration switched off, for the comparisons below */
Clip without_restoration(const Clip& clip) {
  Clip plain = clip;
  for (FrameSyntax& syntax : plain.syntax) {
    for (int plane = 0; plane < 3; ++plane) {
      syntax.restoration_type[plane] = kRestoreNone;
      syntax.restoration[plane].clear();
    }
  }
  return plain;
}

struct Run {
  std::vector<Frame> frames;
  double ms = 0;
  bool ok = false;
  std::string error;
};

Run decode_clip(const BackendOptions& options, const Clip& clip) {
  Run run;
  std::string error;
  std::unique_ptr<Decoder> decoder = Decoder::create(options, &error);
  if (!decoder) {
    run.error = error;
    return run;
  }
  run.frames.resize(clip.syntax.size());
  int slot = -1;
  for (size_t f = 0; f < clip.syntax.size(); ++f) {
    FrameSyntax syntax = clip.syntax[f];
    if (slot >= 0) {
      syntax.reference_slots.clear();
      syntax.reference_slots.push_back(slot);
    }
    const int next = decoder->decode_reference(syntax, &run.frames[f], &error);
    if (next < 0) {
      run.error = error;
      return run;
    }
    if (slot >= 0) decoder->release_reference(slot);
    slot = next;
    run.ms += decoder->stats().total_ms;
  }
  if (slot >= 0) decoder->release_reference(slot);
  run.ok = true;
  return run;
}

bool identical(const Frame& a, const Frame& b) {
  if (a.width != b.width || a.height != b.height) return false;
  for (int p = 0; p < 3; ++p) {
    const int w = a.plane_width(p);
    const int h = a.plane_height(p);
    for (int y = 0; y < h; ++y) {
      const uint16_t* ra = a.plane(p) + static_cast<size_t>(y) * a.stride[p];
      const uint16_t* rb = b.plane(p) + static_cast<size_t>(y) * b.stride[p];
      if (std::memcmp(ra, rb, static_cast<size_t>(w) * sizeof(uint16_t)) != 0) return false;
    }
  }
  return true;
}

/* The stripe a luma row belongs to (7.17.1): 64 rows per stripe, the first one
 * starting eight rows above the frame. */
void luma_stripe(int y, int* start, int* end) {
  const int stripe = (y + 8) / 64;
  *start = -8 + stripe * 64;
  *end = *start + 63;
}

/* 7.17.4 with 7.17.6's choice of picture: a row inside the stripe comes from the
 * CDEF output, a row outside it from the deblocked picture, and the third row
 * outside is a copy of the second - a filter whose taps reach across the border
 * has to see the clamped row, not the one it asked for. Sections below pass the
 * same picture twice when CDEF is off (the two are then equal and only the clamp
 * matters), and the CDEF output plus a decode of the same clip *without* CDEF
 * when it is on. */
int wiener_sample_striped(const Frame& cdef_src, const Frame& pre_src, int x, int y,
                          const int* vfilter, const int* hfilter, int bitdepth, int stripe_start,
                          int stripe_end) {
  const int plane = 0;
  const int w = cdef_src.plane_width(plane);
  const int h = cdef_src.plane_height(plane);
  const int iround0 = bitdepth == 12 ? 5 : 3;
  const int iround1 = bitdepth == 12 ? 9 : 11;
  const int offset = 1 << (bitdepth + 7 /* FILTER_BITS */ - iround0 - 1);
  const int limit = (1 << (bitdepth + 1 + 7 - iround0)) - 1;
  int vtap[7], htap[7];
  for (int t = 0; t < 3; ++t) {
    vtap[t] = vtap[6 - t] = vfilter[t];
    htap[t] = htap[6 - t] = hfilter[t];
  }
  vtap[3] = 128 - 2 * (vfilter[0] + vfilter[1] + vfilter[2]);
  htap[3] = 128 - 2 * (hfilter[0] + hfilter[1] + hfilter[2]);
  auto sample = [&](int xx, int yy) {
    xx = std::min(std::max(xx, 0), w - 1);
    yy = std::min(std::max(yy, 0), h - 1);
    if (yy < stripe_start) {
      yy = std::max(yy, stripe_start - 2);
      return static_cast<int>(pre_src.plane(plane)[static_cast<size_t>(yy) * pre_src.stride[plane] +
                                                   xx]);
    }
    if (yy > stripe_end) {
      yy = std::min(yy, stripe_end + 2);
      return static_cast<int>(pre_src.plane(plane)[static_cast<size_t>(yy) * pre_src.stride[plane] +
                                                   xx]);
    }
    return static_cast<int>(cdef_src.plane(plane)[static_cast<size_t>(yy) * cdef_src.stride[plane] +
                                                  xx]);
  };
  int inter[7];
  for (int t = 0; t < 7; ++t) {
    int sum = 0;
    for (int k = 0; k < 7; ++k) sum += htap[k] * sample(x + k - 3, y + t - 3);
    const int v = (sum + (1 << (iround0 - 1))) >> iround0;
    inter[t] = std::min(std::max(v, -offset), limit - offset);
  }
  int sum = 0;
  for (int t = 0; t < 7; ++t) sum += vtap[t] * inter[t];
  const int v = (sum + (1 << (iround1 - 1))) >> iround1;
  return std::min(std::max(v, 0), (1 << bitdepth) - 1);
}

/* Give every luma unit its own filter, so that a sample filtered by the wrong
 * unit comes out wrong: two units that filter alike cannot show a grid mistake. */
void vary_luma_filters(Clip* clip, int unit_log2) {
  for (FrameSyntax& syntax : clip->syntax) {
    const int cols = syntax.restoration_unit_cols[0];
    for (size_t i = 0; i < syntax.restoration[0].size(); ++i) {
      RestorationUnit& unit = syntax.restoration[0][i];
      const int row = static_cast<int>(i) / std::max(cols, 1);
      const int col = static_cast<int>(i) % std::max(cols, 1);
      (void)unit_log2;
      /* taps at +-3 and +-1 vertically (the +-3 one reaches across a stripe
       * border), and a different set again for every column horizontally */
      unit.wiener[0][0] = 2 + 2 * row; /* +-3 */
      unit.wiener[0][1] = 0;           /* +-2 */
      unit.wiener[0][2] = 3;           /* +-1 */
      unit.wiener[1][0] = 1 + 2 * col; /* +-3 */
      unit.wiener[1][1] = -2;          /* +-2 */
      unit.wiener[1][2] = 3;           /* +-1 */
    }
  }
}

/* The taps of the unit a luma sample belongs to (7.17.1): the column is the
 * sample divided by the unit width, the row is the sample offset by the same
 * eight luma samples the stripes are shifted by, both clamped to the last unit.
 * A grid built on the wrong size is a different answer for some sample. */
void unit_taps(const FrameSyntax& syntax, int x, int y, int* vfilter, int* hfilter) {
  const int unit_size = 1 << syntax.restoration_unit_log2[0];
  const int cols = std::max(syntax.restoration_unit_cols[0], 1);
  const int rows = std::max((syntax.height + (unit_size >> 1)) / unit_size, 1);
  const int col = std::min(cols - 1, x / unit_size);
  const int row = std::min(rows - 1, (y + 8) / unit_size);
  const size_t index = static_cast<size_t>(row) * cols + col;
  const RestorationUnit& unit = syntax.restoration[0][std::min(index,
                                                             syntax.restoration[0].size() - 1)];
  for (int i = 0; i < 3; ++i) {
    vfilter[i] = unit.wiener[0][i];
    hfilter[i] = unit.wiener[1][i];
  }
}

} /* namespace */

int main() {
  std::printf("AV1onGPU pipeline tests\n");

  /* ---- 1. the synthetic encoder and decoder agree ------------------------ */
  std::printf("synthetic clip\n");
  const Clip clip = make_clip(3, 4, true, true);
  CHECK(!clip.syntax[0].blocks.empty());

  {
    BackendOptions options;
    options.kind = BackendKind::Cpu;
    const Run cpu = decode_clip(options, clip);
    CHECK(cpu.ok);
    if (!cpu.ok) {
      std::printf("  cpu decode failed: %s\n", cpu.error.c_str());
      return testing::finish("test_pipeline");
    }
    for (size_t f = 0; f < clip.source.size(); ++f) {
      const Psnr psnr = frame_psnr(cpu.frames[f], clip.recon[f]);
      std::printf("  frame %zu: %.2f dB (vs encoder recon) %.2f ms\n", f, psnr.y, cpu.ms);
      CHECK(psnr.y > 40.0);
      /* the encoder's own reconstruction and the decoder must agree exactly */
      CHECK(identical(cpu.frames[f], clip.recon[f]));
    }
    /* PSNR against the source picture, which is what a user would judge */
    const Psnr source_psnr = frame_psnr(cpu.frames[0], clip.source[0]);
    std::printf("  frame 0 source PSNR: Y %.2f U %.2f V %.2f dB\n", source_psnr.y,
                source_psnr.u, source_psnr.v);
    CHECK(source_psnr.y > 25.0);
  }

  /* ---- 2. the scheduler must be reproducible ------------------------------ */
  {
    /* The wavefront is built from exact sample ownership, so two decodes of the
     * same syntax have to produce the same picture, the same number of waves
     * and no leftover samples: an uninitialised or unowned sample would show up
     * as a difference between the two runs (or as a mismatch against the
     * encoder's reconstruction, checked above). */
    BackendOptions options;
    options.kind = BackendKind::Cpu;
    const Run ra = decode_clip(options, clip);
    const Run rb = decode_clip(options, clip);
    CHECK(ra.ok && rb.ok);
    if (ra.ok && rb.ok)
      for (size_t f = 0; f < clip.source.size(); ++f) CHECK(identical(ra.frames[f], rb.frames[f]));
  }

  /* ---- 3. GPU backends, when present ------------------------------------ */
  {
    BackendOptions cpu_options;
    cpu_options.kind = BackendKind::Cpu;
    cpu_options.allow_cpu_fallback = false;
    const Run cpu = decode_clip(cpu_options, clip);

    const BackendKind kinds[] = {BackendKind::OpenCL, BackendKind::CUDA};
    for (BackendKind kind : kinds) {
      BackendOptions options;
      options.kind = kind;
      options.allow_cpu_fallback = false;
      std::string error;
      std::unique_ptr<Decoder> probe = Decoder::create(options, &error);
      if (!probe || probe->device().kind != kind) {
        std::printf("%s: not available (%s)\n", to_string(kind),
                    error.empty() ? "fallback only" : error.c_str());
        continue;
      }
      std::printf("%s: %s\n", to_string(kind), probe->device().name.c_str());
      probe.reset();
      const Run gpu = decode_clip(options, clip);
      CHECK(gpu.ok);
      if (!gpu.ok) {
        std::printf("  decode failed: %s\n", gpu.error.c_str());
        continue;
      }
      for (size_t f = 0; f < clip.source.size(); ++f) {
        const bool same = identical(gpu.frames[f], cpu.frames[f]);
        CHECK(same);
        if (!same) {
          const Psnr diff = frame_psnr(gpu.frames[f], cpu.frames[f]);
          std::printf("  frame %zu differs from the CPU reference (%.2f dB)\n", f, diff.y);
        }
      }
      std::printf("  cpu %.2f ms, gpu %.2f ms per clip\n", cpu.ms, gpu.ms);
    }
  }

  /* ---- 4. superres and loop restoration together -------------------------- */
  std::printf("superres + loop restoration\n");
  {
    /* one frame, so the second one does not predict from a reference filter
     * this section changed, and no CDEF, so the picture the stage restores and
     * the one it borrows across a stripe border are the same and the expected
     * samples can be computed here */
    const int vfilter[3] = {0, 0, 4};
    const int hfilter[3] = {1, -2, 3};
    const Clip base = make_clip(1, 4, /*cdef=*/false, /*grain=*/false, 112, 96);
    const Clip restored = make_superres_lr_clip(base, 128, vfilter, hfilter);
    const Clip plain = without_restoration(restored);

    BackendOptions cpu_options;
    cpu_options.kind = BackendKind::Cpu;
    const Run cpu = decode_clip(cpu_options, restored);
    CHECK(cpu.ok);
    if (!cpu.ok) {
      std::printf("  superres + LR decode failed: %s\n", cpu.error.c_str());
    } else {
      const Frame& out = cpu.frames[0];
      CHECK(out.width == 128 && out.height == 96);
      CHECK(out.plane_width(0) == 128);
      CHECK(out.plane_width(1) == 64 && out.plane_height(1) == 48);
      /* every sample of every plane is a real sample, not a read of an
       * uninitialised corner of a buffer shaped for the coded geometry */
      const int maxv = (1 << restored.syntax[0].bit_depth) - 1;
      bool in_range = true;
      for (int plane = 0; plane < 3; ++plane)
        for (int y = 0; y < out.plane_height(plane); ++y)
          for (int x = 0; x < out.plane_width(plane); ++x) {
            const int v = out.plane(plane)[static_cast<size_t>(y) * out.stride[plane] + x];
            if (v < 0 || v > maxv) in_range = false;
          }
      CHECK(in_range);

      const Run plain_cpu = decode_clip(cpu_options, plain);
      CHECK(plain_cpu.ok);
      if (plain_cpu.ok) {
        CHECK(!identical(cpu.frames[0], plain_cpu.frames[0]));
        /* the restored luma has to be what 7.17.4 says a symmetric seven tap
         * filter on the upscaled picture owes - over the whole of it, including
         * the columns past the coded width (112) and the rows across the stripe
         * border, which a unit reads from the second upscaled picture */
        int worst = 0;
        int wrong = 0;
        for (int y = 0; y < out.plane_height(0); ++y)
          for (int x = 0; x < out.plane_width(0); ++x) {
            const int got = out.plane(0)[static_cast<size_t>(y) * out.stride[0] + x];
            int stripe_start = 0, stripe_end = 0;
            luma_stripe(y, &stripe_start, &stripe_end);
            const int want =
                wiener_sample_striped(plain_cpu.frames[0], plain_cpu.frames[0], x, y, vfilter,
                                      hfilter, restored.syntax[0].bit_depth, stripe_start,
                                      stripe_end);
            const int d = std::abs(got - want);
            if (d) ++wrong;
            if (d > worst) worst = d;
          }
        std::printf("  luma vs the reference: %d of %d samples wrong, worst %d\n", wrong,
                    out.plane_width(0) * out.plane_height(0), worst);
        CHECK(wrong == 0);
      }

      /* The same picture with an identity filter in every luma unit: the stage
       * then has to be a bit exact no-op over the whole frame, chroma included
       * (a self guided unit is one on this ramp as well, where both box filters
       * return the sample itself). A job that writes through the wrong stride or
       * at a doubled offset cannot put these samples back; the unit grid is what
       * the reference comparison above pins. */
      const int zero[3] = {0, 0, 0};
      const Clip ident = make_superres_lr_clip(base, 128, zero, zero);
      const Clip ident_plain = without_restoration(ident);
      const Run ident_cpu = decode_clip(cpu_options, ident);
      const Run ident_plain_cpu = decode_clip(cpu_options, ident_plain);
      CHECK(ident_cpu.ok && ident_plain_cpu.ok);
      if (ident_cpu.ok && ident_plain_cpu.ok)
        CHECK(identical(ident_cpu.frames[0], ident_plain_cpu.frames[0]));

      /* deterministic: the stage reads two snapshots, so a second run of the
       * same syntax has to produce the same picture */
      const Run again = decode_clip(cpu_options, restored);
      CHECK(again.ok && identical(again.frames[0], cpu.frames[0]));

      /* and the OpenCL backend, when this machine has a device, has to agree
       * with the CPU reference on this combination too */
      BackendOptions ocl_options;
      ocl_options.kind = BackendKind::OpenCL;
      ocl_options.allow_cpu_fallback = false;
      std::string error;
      std::unique_ptr<Decoder> probe = Decoder::create(ocl_options, &error);
      if (probe && probe->device().kind == BackendKind::OpenCL) {
        std::printf("  opencl: %s\n", probe->device().name.c_str());
        probe.reset();
        const Run ocl = decode_clip(ocl_options, restored);
        CHECK(ocl.ok);
        if (ocl.ok)
          for (size_t f = 0; f < restored.syntax.size(); ++f)
            CHECK(identical(ocl.frames[f], cpu.frames[f]));
      } else {
        std::printf("  opencl: not available (%s)\n",
                    error.empty() ? "fallback only" : error.c_str());
      }
    }
  }

  /* ---- 5. the restoration geometry --------------------------------------- */
  std::printf("restoration geometry\n");
  {
    /* Stripes are 64 rows and the first one starts eight rows above the frame,
     * units are counted with a rounding rule, and the last unit of a row takes
     * the remainder - so a picture that is not a multiple of the unit, or not a
     * multiple of 64 rows, exercises clamps and partial stripes that a round
     * 128 x 96 picture never reaches. Each case is compared against the
     * specification's filter applied to the whole decoded picture, which pins
     * the stripe and unit grid of every one of them. */
    struct Geometry {
      int coded_width;
      int display_width;
      int height;
      int unit_log2;
      const char* what;
    };
    const Geometry geometries[] = {
        {112, 128, 96, 6, "superres, 2 x 2 units"},  /* the section above */
        {128, 128, 100, 6, "height 100: 64 + 36"},   /* a partial second stripe */
        {128, 128, 68, 6, "height 68: 64 + 4"},      /* four rows under a stripe */
        {120, 120, 68, 6, "width 120: 1.875 units"}, /* the last unit takes 56 */
        /* one 128 sample unit on a 160 wide picture: the count rounds down, so
         * the unit's own width is not the plane's and only the clamp reaches the
         * right hand samples */
        {160, 160, 96, 7, "1 unit, 160x96"},
        {96, 96, 72, 6, "2 units, the last 32 wide"},
        {32, 32, 24, 6, "smaller than one unit"}, /* every count clamps to 1 */
        {16, 16, 16, 6, "16 x 16"},
    };
    const int vfilter[3] = {0, 0, 4};
    const int hfilter[3] = {1, -2, 3};
    BackendOptions cpu_options;
    cpu_options.kind = BackendKind::Cpu;
    for (const Geometry& geometry : geometries) {
      const Clip base = make_clip(1, 4, /*cdef=*/false, /*grain=*/false, geometry.coded_width,
                                  geometry.height);
      Clip restored = make_superres_lr_clip(base, geometry.display_width, vfilter, hfilter,
                                            geometry.unit_log2);
      vary_luma_filters(&restored, geometry.unit_log2);
      const Clip plain = without_restoration(restored);
      const Run cpu = decode_clip(cpu_options, restored);
      const Run plain_cpu = decode_clip(cpu_options, plain);
      CHECK(cpu.ok && plain_cpu.ok);
      if (!cpu.ok || !plain_cpu.ok) {
        std::printf("  %s: decode failed: %s%s\n", geometry.what, cpu.error.c_str(),
                    plain_cpu.error.c_str());
        continue;
      }
      const Frame& out = cpu.frames[0];
      CHECK(out.width == geometry.display_width && out.height == geometry.height);
      const int bitdepth = restored.syntax[0].bit_depth;
      int wrong = 0;
      int worst = 0;
      int changed = 0;
      for (int y = 0; y < out.plane_height(0); ++y)
        for (int x = 0; x < out.plane_width(0); ++x) {
          const int got = out.plane(0)[static_cast<size_t>(y) * out.stride[0] + x];
          int unit_v[3], unit_h[3];
          unit_taps(restored.syntax[0], x, y, unit_v, unit_h);
          int stripe_start = 0, stripe_end = 0;
          luma_stripe(y, &stripe_start, &stripe_end);
          const int want = wiener_sample_striped(plain_cpu.frames[0], plain_cpu.frames[0], x, y,
                                                 unit_v, unit_h, bitdepth, stripe_start,
                                                 stripe_end);
          const int d = std::abs(got - want);
          if (d) ++wrong;
          if (d > worst) worst = d;
          if (got != plain_cpu.frames[0].plane(0)[static_cast<size_t>(y) * out.stride[0] + x])
            ++changed;
        }
      std::printf("  %-28s %3dx%-3d %d wrong, worst %d, %d samples restored\n", geometry.what,
                  out.width, out.height, wrong, worst, changed);
      CHECK(wrong == 0);
      /* the filter really ran: a case where nothing changes could not tell a
       * wrong grid from a right one */
      CHECK(changed > 0);
    }
  }

  /* ---- 6. the two pictures a unit reads ----------------------------------- */
  std::printf("restoration across a stripe border\n");
  {
    /* 7.17.6: inside its stripe a unit filters the CDEF output, but the two
     * lines either side of the stripe's border come from the picture before
     * CDEF - which is why the backend keeps a second snapshot, and why a
     * superres frame has to upscale it too. The test can check that rule
     * because the same clip decoded with CDEF switched off *is* the deblocked
     * picture: the expected sample of a row in the stripe is filtered from one
     * picture, a row outside it from the other. */
    const int vfilter[3] = {0, 0, 4};
    const int hfilter[3] = {1, -2, 3};
    Clip base = make_clip(1, 4, /*cdef=*/true, /*grain=*/false, 112, 96);
    Clip restored = make_superres_lr_clip(base, 128, vfilter, hfilter, 6);
    vary_luma_filters(&restored, 6);
    Clip with_cdef = without_restoration(restored);
    Clip pre_cdef = without_restoration(restored);
    for (FrameSyntax& syntax : pre_cdef.syntax) syntax.cdef.clear();

    BackendOptions cpu_options;
    cpu_options.kind = BackendKind::Cpu;
    const Run cdef_run = decode_clip(cpu_options, with_cdef);
    const Run pre_run = decode_clip(cpu_options, pre_cdef);
    const Run lr_run = decode_clip(cpu_options, restored);
    CHECK(cdef_run.ok && pre_run.ok && lr_run.ok);
    if (!cdef_run.ok || !pre_run.ok || !lr_run.ok) {
      std::printf("  decode failed: %s%s%s\n", cdef_run.error.c_str(), pre_run.error.c_str(),
                  lr_run.error.c_str());
    } else {
      /* the two pictures have to differ, or nothing below says anything */
      CHECK(!identical(cdef_run.frames[0], pre_run.frames[0]));
      const Frame& out = lr_run.frames[0];
      const int bitdepth = restored.syntax[0].bit_depth;
      /* the stripes are 64 luma rows, the first one starting eight rows above
       * the frame (7.17.1) */
      int wrong = 0;
      int worst = 0;
      int across = 0;
      for (int y = 0; y < out.plane_height(0); ++y) {
        const int stripe = (y + 8) / 64;
        const int stripe_start = -8 + stripe * 64;
        const int stripe_end = stripe_start + 63;
        /* a sample whose filter reaches across the border is the one that can
         * tell the snapshots apart */
        const bool reaches = y - 3 < stripe_start || y + 3 > stripe_end;
        for (int x = 0; x < out.plane_width(0); ++x) {
          const int got = out.plane(0)[static_cast<size_t>(y) * out.stride[0] + x];
          int unit_v[3], unit_h[3];
          unit_taps(restored.syntax[0], x, y, unit_v, unit_h);
          const int want = wiener_sample_striped(cdef_run.frames[0], pre_run.frames[0], x, y,
                                                 unit_v, unit_h, bitdepth, stripe_start,
                                                 stripe_end);
          const int d = std::abs(got - want);
          if (d) ++wrong;
          if (d > worst) worst = d;
          if (reaches) ++across;
        }
      }
      /* is the stripe border really where the two pictures meet? The border
       * rows are the ones where using the wrong picture changes the answer. */
      int border_diff = 0;
      for (int y = 0; y < out.plane_height(0); ++y) {
        const int stripe = (y + 8) / 64;
        const int stripe_start = -8 + stripe * 64;
        if (y - 3 >= stripe_start && y + 3 <= stripe_start + 63) continue;
        for (int x = 0; x < out.plane_width(0); ++x)
          if (pre_run.frames[0].plane(0)[static_cast<size_t>(y) * pre_run.frames[0].stride[0] + x] !=
              cdef_run.frames[0].plane(0)[static_cast<size_t>(y) * cdef_run.frames[0].stride[0] + x])
            ++border_diff;
      }
      std::printf("  %d wrong of %d samples, worst %d (%d reach a border, %d of them differ "
                  "between the two pictures)\n",
                  wrong, out.plane_width(0) * out.plane_height(0), worst, across, border_diff);
      CHECK(wrong == 0);
      CHECK(border_diff > 0);
    }
  }

  return testing::finish("test_pipeline");
}
