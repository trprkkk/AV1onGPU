#include "frontend/cdf.hpp"

#include "frontend/cdf_tables_inter.hpp"

#include <cstring>
#include <type_traits>

namespace av1gpu {
namespace frontend {
namespace {

/* The generated tables and the context have the same shape field for field;
 * the copies below are checked by a static assertion so a typo in either side
 * cannot silently truncate a table. */
template <typename A, typename B>
void copy_table(A& dst, const B& src) {
  static_assert(sizeof(dst) == sizeof(src), "CDF table shape mismatch");
  std::memcpy(&dst, &src, sizeof(dst));
}

/* A default table that the specification copies once per motion vector
 * component: the copies start equal and then adapt independently. The copies
 * are interleaved per row - a table of ten rows becomes ten pairs of rows, not
 * ten rows followed by ten more - because the decoder indexes the result as
 * [row][component]; copying the table as one block would put row 2 where row 1
 * of component 0 is expected. */
template <int N, typename A, typename B>
void broadcast_table(A& dst, const B& src) {
  static_assert(sizeof(dst) == sizeof(src) * static_cast<size_t>(N),
                "CDF table broadcast shape mismatch");
  using Elem = std::remove_reference_t<decltype(src[0])>;
  constexpr size_t kRowBytes = std::is_array_v<Elem> ? sizeof(Elem) : sizeof(src);
  constexpr size_t kRows = sizeof(src) / kRowBytes;
  const char* from = reinterpret_cast<const char*>(&src);
  char* to = reinterpret_cast<char*>(&dst);
  for (size_t row = 0; row < kRows; ++row)
    for (int n = 0; n < N; ++n)
      std::memcpy(to + (row * N + n) * kRowBytes, from + row * kRowBytes, kRowBytes);
}

} /* namespace */

void init_default_cdf(CdfContext* cdf, int base_q_idx) {
  if (!cdf) return;
  namespace t = cdf_tables;

  /* the coefficient tables come in four quantiser classes */
  const int qcat = (base_q_idx > 20) + (base_q_idx > 60) + (base_q_idx > 120);

  copy_table(cdf->partition.table, t::partition);
  copy_table(cdf->skip.table, t::skip);

  copy_table(cdf->mode.kfym, t::kfym);
  copy_table(cdf->mode.partition, t::partition);
  copy_table(cdf->mode.skip, t::skip);
  copy_table(cdf->mode.intrabc, t::intrabc);
  copy_table(cdf->mode.seg_id, t::seg_id);
  copy_table(cdf->mode.seg_pred, t::seg_pred);
  copy_table(cdf->mode.delta_q, t::delta_q);
  copy_table(cdf->mode.delta_lf, t::delta_lf);
  copy_table(cdf->mode.uv_mode, t::uv_mode);
  copy_table(cdf->mode.angle_delta, t::angle_delta);
  copy_table(cdf->mode.filter_intra, t::filter_intra);
  copy_table(cdf->mode.use_filter_intra, t::use_filter_intra);
  copy_table(cdf->mode.pal_sz, t::pal_sz);
  copy_table(cdf->mode.pal_y, t::pal_y);
  copy_table(cdf->mode.pal_uv, t::pal_uv);
  copy_table(cdf->mode.cfl_sign, t::cfl_sign);
  copy_table(cdf->mode.cfl_alpha, t::cfl_alpha);
  copy_table(cdf->mode.txtp_intra1, t::txtp_intra1);
  copy_table(cdf->mode.txtp_intra2, t::txtp_intra2);
  copy_table(cdf->mode.txsz, t::txsz);
  copy_table(cdf->mode.txpart, t::txpart);

  namespace it = cdf_tables_inter;
  copy_table(cdf->inter.y_mode, it::y_mode);
  copy_table(cdf->inter.is_inter, it::is_inter);
  copy_table(cdf->inter.skip_mode, it::skip_mode);
  copy_table(cdf->inter.comp_mode, it::comp_mode);
  copy_table(cdf->inter.comp_ref_type, it::comp_ref_type);
  copy_table(cdf->inter.uni_comp_ref, it::uni_comp_ref);
  copy_table(cdf->inter.comp_ref, it::comp_ref);
  copy_table(cdf->inter.comp_bwd_ref, it::comp_bwd_ref);
  copy_table(cdf->inter.single_ref, it::single_ref);
  copy_table(cdf->inter.compound_mode, it::compound_mode);
  copy_table(cdf->inter.compound_idx, it::compound_idx);
  copy_table(cdf->inter.comp_group_idx, it::comp_group_idx);
  copy_table(cdf->inter.compound_type, it::compound_type);
  copy_table(cdf->inter.wedge_index, it::wedge_index);
  copy_table(cdf->inter.inter_intra, it::inter_intra);
  copy_table(cdf->inter.inter_intra_mode, it::inter_intra_mode);
  copy_table(cdf->inter.wedge_inter_intra, it::wedge_inter_intra);
  copy_table(cdf->inter.use_obmc, it::use_obmc);
  copy_table(cdf->inter.motion_mode, it::motion_mode);
  copy_table(cdf->inter.interp_filter, it::interp_filter);
  copy_table(cdf->inter.mv_joint, it::mv_joint);
  broadcast_table<2>(cdf->inter.mv_sign, it::mv_sign);
  copy_table(cdf->inter.mv_class, it::mv_class);
  broadcast_table<2>(cdf->inter.mv_class0_bit, it::mv_class0_bit);
  copy_table(cdf->inter.mv_class0_fr, it::mv_class0_fr);
  broadcast_table<2>(cdf->inter.mv_class0_hp, it::mv_class0_hp);
  broadcast_table<2>(cdf->inter.mv_hp, it::mv_hp);
  broadcast_table<2>(cdf->inter.mv_bit, it::mv_bit);
  copy_table(cdf->inter.mv_fr, it::mv_fr);
  copy_table(cdf->inter.new_mv, it::new_mv);
  copy_table(cdf->inter.zero_mv, it::zero_mv);
  copy_table(cdf->inter.ref_mv, it::ref_mv);
  copy_table(cdf->inter.drl_mode, it::drl_mode);
  copy_table(cdf->inter.txtp_inter1, it::tx_type_inter1);
  copy_table(cdf->inter.txtp_inter2, it::tx_type_inter2);
  copy_table(cdf->inter.txtp_inter3, it::tx_type_inter3);

  copy_table(cdf->coef.eob_bin_16, t::eob_bin_16[qcat]);
  copy_table(cdf->coef.eob_bin_32, t::eob_bin_32[qcat]);
  copy_table(cdf->coef.eob_bin_64, t::eob_bin_64[qcat]);
  copy_table(cdf->coef.eob_bin_128, t::eob_bin_128[qcat]);
  copy_table(cdf->coef.eob_bin_256, t::eob_bin_256[qcat]);
  copy_table(cdf->coef.eob_bin_512, t::eob_bin_512[qcat]);
  copy_table(cdf->coef.eob_bin_1024, t::eob_bin_1024[qcat]);
  copy_table(cdf->coef.eob_base_tok, t::eob_base_tok[qcat]);
  copy_table(cdf->coef.base_tok, t::base_tok[qcat]);
  copy_table(cdf->coef.br_tok, t::br_tok[qcat]);
  copy_table(cdf->coef.eob_hi_bit, t::eob_hi_bit[qcat]);
  copy_table(cdf->coef.skip, t::coef_skip[qcat]);
  copy_table(cdf->coef.dc_sign, t::dc_sign[qcat]);
}

} /* namespace frontend */
} /* namespace av1gpu */
