#include "frontend/cdf.hpp"

#include <cstring>

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
