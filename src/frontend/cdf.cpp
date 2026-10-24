#include "frontend/cdf.hpp"

#include "frontend/cdf_tables_inter.hpp"

#include <algorithm>
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

namespace {

/* The counter of a row sits just past its probabilities, at the index the
 * decoder passes as `nsymbols` to read_symbol. */
template <size_t N>
void reset_row(uint16_t (&row)[N], int nsyms) {
  if (nsyms >= 0 && static_cast<size_t>(nsyms) < N) row[nsyms] = 0;
}

template <typename T, size_t A, size_t B>
void reset_2d(T (&table)[A][B], int nsyms) {
  for (size_t i = 0; i < A; ++i) reset_row(table[i], nsyms);
}

template <typename T, size_t A, size_t B, size_t C>
void reset_3d(T (&table)[A][B][C], int nsyms) {
  for (size_t i = 0; i < A; ++i) reset_2d(table[i], nsyms);
}

template <typename T, size_t A, size_t B, size_t C, size_t D>
void reset_4d(T (&table)[A][B][C][D], int nsyms) {
  for (size_t i = 0; i < A; ++i) reset_3d(table[i], nsyms);
}

} /* namespace */

void reset_cdf_counts(CdfContext* cdf) {
  if (!cdf) return;
  ModeCdf& m = cdf->mode;
  InterCdf& in = cdf->inter;
  CoefCdf& co = cdf->coef;

  /* coefficient tables: the end of block tables carry their own symbol counts,
   * which grow with the transform size */
  reset_3d(co.eob_bin_16, 4);
  reset_3d(co.eob_bin_32, 5);
  reset_3d(co.eob_bin_64, 6);
  reset_3d(co.eob_bin_128, 7);
  reset_3d(co.eob_bin_256, 8);
  reset_2d(co.eob_bin_512, 9);
  reset_2d(co.eob_bin_1024, 10);
  reset_4d(co.eob_base_tok, 2);
  reset_4d(co.base_tok, 3);
  reset_4d(co.br_tok, 3);
  reset_4d(co.eob_hi_bit, 1);
  reset_3d(co.skip, 1);
  reset_3d(co.dc_sign, 1);

  /* mode tables: the chroma mode table has one symbol more when chroma from
   * luma is available, and the transform size table narrows near the top of
   * the tree, so both are reset row by row */
  for (int cfl = 0; cfl < 2; ++cfl)
    for (int mode = 0; mode < 13; ++mode)
      reset_row(m.uv_mode[cfl][mode], kUvIntraModes - 1 - (cfl ? 0 : 1));
  /* the partition and skip tables live outside ModeCdf: they are the two an
   * intra frame reads as often as an inter frame */
  for (int level = 0; level < kBlockLevelCount; ++level)
    reset_2d(cdf->partition.table[level], kPartitionSymbols[level]);
  reset_2d(m.cfl_alpha, 15);
  reset_3d(m.txtp_intra1, 6);
  reset_3d(m.txtp_intra2, 4);
  for (int cls = 0; cls < 4; ++cls)
    reset_2d(m.txsz[cls], std::min(cls + 1, 2));
  reset_row(m.cfl_sign, 7);
  reset_2d(m.angle_delta, 6);
  reset_row(m.filter_intra, 4);
  reset_2d(m.seg_id, 7);
  reset_2d(m.use_filter_intra, 1);
  reset_3d(m.txpart, 1);
  reset_2d(cdf->skip.table, 1);
  reset_row(m.delta_q, 3);
  reset_2d(m.delta_lf, 3);

  /* inter tables */
  reset_2d(in.y_mode, 12);
  reset_2d(in.is_inter, 1);
  reset_2d(in.skip_mode, 1);
  reset_2d(in.comp_mode, 1);
  reset_2d(in.comp_ref_type, 1);
  reset_3d(in.uni_comp_ref, 1);
  reset_3d(in.comp_ref, 1);
  reset_3d(in.comp_bwd_ref, 1);
  reset_3d(in.single_ref, 1);
  reset_2d(in.compound_mode, 7);
  reset_2d(in.compound_idx, 1);
  reset_2d(in.comp_group_idx, 1);
  reset_2d(in.compound_type, 1);
  reset_2d(in.wedge_index, 15);
  reset_2d(in.inter_intra, 1);
  reset_2d(in.inter_intra_mode, 3);
  reset_2d(in.wedge_inter_intra, 1);
  reset_2d(in.use_obmc, 1);
  reset_2d(in.motion_mode, 2);
  reset_2d(in.interp_filter, 2);
  reset_row(in.mv_joint, 3);
  reset_2d(in.mv_sign, 1);
  reset_2d(in.mv_class, 10);
  reset_2d(in.mv_class0_bit, 1);
  reset_3d(in.mv_class0_fr, 3);
  reset_2d(in.mv_class0_hp, 1);
  reset_2d(in.mv_hp, 1);
  reset_3d(in.mv_bit, 1);
  reset_2d(in.mv_fr, 3);
  reset_2d(in.new_mv, 1);
  reset_2d(in.zero_mv, 1);
  reset_2d(in.ref_mv, 1);
  reset_2d(in.drl_mode, 1);
  reset_2d(in.txtp_inter1, 15);
  reset_row(in.txtp_inter2, 11);
  reset_2d(in.txtp_inter3, 1);
}

} /* namespace frontend */
} /* namespace av1gpu */
