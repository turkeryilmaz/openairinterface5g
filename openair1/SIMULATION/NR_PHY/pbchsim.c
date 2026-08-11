/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include "common/config/config_userapi.h"
#include "common/utils/LOG/log.h"
#include "common/utils/load_module_shlib.h"
#include "common/ran_context.h" 
#include "common/utils/nr/nr_common.h"
#include "PHY/types.h"
#include "PHY/defs_nr_common.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/defs_gNB.h"
#include "PHY/MODULATION/modulation_eNB.h"
#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/INIT/nr_phy_init.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_REFSIG/pss_nr.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_UE_ESTIMATION/nr_estimation.h"
#include "PHY/phy_vars.h"
#include "SCHED_NR/sched_nr.h"
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "openair1/SIMULATION/RF/rf.h"
#include "openair1/SIMULATION/NR_PHY/nr_unitary_defs.h"
#include "openair1/PHY/MODULATION/nr_modulation.h"
#include <executables/softmodem-common.h>
#include <executables/nr-uesoftmodem.h>
#include "nfapi/oai_integration/vendor_ext.h"
//#define DEBUG_NR_PBCHSIM

PHY_VARS_gNB *gNB;
PHY_VARS_NR_UE *UE;
RAN_CONTEXT_t RC;
int64_t uplink_frequency_offset[MAX_NUM_CCs][4];

double cpuf;

// needed for some functions
openair0_config_t openair0_cfg_g[MAX_CARDS] = {};

uint8_t const nr_rv_round_map[4] = {0, 2, 3, 1};

void inc_ref_sched_response(int _)
{
  LOG_E(PHY, "fatal\n");
  exit(1);
}
void deref_sched_response(int _)
{
  LOG_E(PHY, "fatal\n");
  exit(1);
}
static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void) {
  return &softmodem_params;
}

nrUE_params_t nrUE_params={0};

nrUE_params_t *get_nrUE_params(void) {
  return &nrUE_params;
}

void init_downlink_harq_status(NR_DL_UE_HARQ_t *dl_harq) {}
NR_IF_Module_t *NR_IF_Module_init(int Mod_id) { return (NULL); }
nfapi_mode_t nfapi_getmode(void) { return NFAPI_MODE_UNKNOWN; }

void nr_fill_rx_indication(fapi_nr_rx_indication_t *rx_ind,
                           uint8_t pdu_type,
                           PHY_VARS_NR_UE *ue,
                           int cw_idx,
                           int harq_pid,
                           NR_UE_DLSCH_t *dlsch,
                           const UE_nr_rxtx_proc_t *proc,
                           void *typeSpecific)
{
}

typedef struct {
  int nid_cell;
  bool valid_pbch;
  int insert_sample;
  double gain;
  double cfo_hz;
} pbchsim_ssb_component_t;

typedef struct {
  pbchsim_ssb_component_t component[2];
  int num_components;
  int target_nid_cell;
  bool expect_detected;
  int expected_nid_cell;
  int expected_insert_sample;
  double expected_cfo_hz;
  unsigned int ofdm_offset_divisor;
} pbchsim_capture_case_t;

typedef struct {
  int nid_cell;
  int ssb_start_subcarrier;
  int half_frame_bit;
  int ssb_index;
  int symbol_offset;
  double freq_offset;
  int adjust_rxgain;
  int init_sync_frame;
} pbchsim_sync_state_t;

static pbchsim_sync_state_t pbchsim_get_sync_state(const PHY_VARS_NR_UE *ue)
{
  return (pbchsim_sync_state_t){.nid_cell = ue->frame_parms.Nid_cell,
                                .ssb_start_subcarrier = ue->frame_parms.ssb_start_subcarrier,
                                .half_frame_bit = ue->frame_parms.half_frame_bit,
                                .ssb_index = ue->frame_parms.ssb_index,
                                .symbol_offset = ue->symbol_offset,
                                .freq_offset = ue->common_vars.freq_offset,
                                .adjust_rxgain = ue->adjust_rxgain,
                                .init_sync_frame = ue->init_sync_frame};
}

static bool pbchsim_sync_state_equal(const pbchsim_sync_state_t *a, const pbchsim_sync_state_t *b)
{
  return a->nid_cell == b->nid_cell && a->ssb_start_subcarrier == b->ssb_start_subcarrier && a->half_frame_bit == b->half_frame_bit
         && a->ssb_index == b->ssb_index && a->symbol_offset == b->symbol_offset && a->freq_offset == b->freq_offset
         && a->adjust_rxgain == b->adjust_rxgain && a->init_sync_frame == b->init_sync_frame;
}

static int64_t pbchsim_floor_divide(int64_t dividend, int64_t divisor)
{
  AssertFatal(divisor > 0, "Invalid divisor %ld\n", divisor);
  int64_t quotient = dividend / divisor;
  if (dividend % divisor < 0)
    quotient--;
  return quotient;
}

static int pbchsim_generate_ssb(PHY_VARS_gNB *gNB,
                                c16_t *slot_time,
                                c16_t *ssb_time,
                                int ssb_subcarrier_offset,
                                int nid_cell,
                                bool valid_pbch)
{
  NR_DL_FRAME_PARMS *fp = &gNB->frame_parms;
  const int original_config_nid = gNB->gNB_config.cell_config.phy_cell_id.value;
  const int original_frame_nid = fp->Nid_cell;
  const int original_ssb_start_subcarrier = fp->ssb_start_subcarrier;
  const int ssb_index = 0;
  const int start_symbol = nr_get_ssb_start_symbol(fp, ssb_index);
  const int slot = start_symbol / fp->symbols_per_slot;
  const int slot_symbol = start_symbol % fp->symbols_per_slot;
  const int sc_offset = fp->freq_range == FR1 ? ssb_subcarrier_offset << fp->numerology_index : ssb_subcarrier_offset;
  const int prb_offset = fp->freq_range == FR1 ? gNB->gNB_config.ssb_table.ssb_offset_point_a.value << fp->numerology_index
                                               : gNB->gNB_config.ssb_table.ssb_offset_point_a.value << (fp->numerology_index - 2);
  nfapi_nr_dl_tti_ssb_pdu ssb_pdu = {0};
  ssb_pdu.ssb_pdu_rel15.bchPayload = 0x55dd33;
  ssb_pdu.ssb_pdu_rel15.SsbBlockIndex = ssb_index;
  ssb_pdu.ssb_pdu_rel15.SsbSubcarrierOffset = sc_offset;
  ssb_pdu.ssb_pdu_rel15.ssbOffsetPointA = prb_offset;

  gNB->gNB_config.cell_config.phy_cell_id.value = nid_cell;
  fp->Nid_cell = nid_cell;
  memset(gNB->common_vars.txdataF[0], 0, fp->samples_per_slot_wCP * sizeof(*gNB->common_vars.txdataF[0]));
  nr_common_signal_procedures(gNB, 0, slot, &ssb_pdu);

  if (!valid_pbch) {
    const int invalid_nid_cell = nid_cell >= 4 ? nid_cell - 4 : nid_cell + 4;
    const int n_hf = slot < (fp->slots_per_frame >> 1) ? 0 : 1;
    const int hf = fp->Lmax == 4 ? n_hf : 0;
    gNB->gNB_config.cell_config.phy_cell_id.value = invalid_nid_cell;
    fp->Nid_cell = invalid_nid_cell;
    nr_generate_pbch_dmrs(nr_gold_pbch(fp->Lmax, invalid_nid_cell, hf, ssb_index & 7),
                          gNB->common_vars.txdataF[0],
                          gNB->TX_AMP,
                          slot_symbol,
                          &gNB->gNB_config,
                          fp);
    nr_generate_pbch(gNB, &ssb_pdu, gNB->common_vars.txdataF[0], slot_symbol, n_hf, 0, &gNB->gNB_config, fp);
  }

  __attribute__((aligned(64))) c16_t fft_in[fp->ofdm_symbol_size * fp->symbols_per_slot];
  memset(fft_in, 0, sizeof(fft_in));
  apply_nr_rotation_TX(fp, gNB->common_vars.txdataF[0], true, fp->symbol_rotation[0], slot, fp->N_RB_DL, 0, fp->symbols_per_slot);
  fft_shift(gNB->common_vars.txdataF[0], fp->ofdm_symbol_size, fp->N_RB_DL, fft_in, fp->ofdm_symbol_size, 0, fp->symbols_per_slot);
  memset(slot_time, 0, fp->samples_per_slot_wCP * sizeof(*slot_time));
  PHY_ofdm_mod((int *)fft_in, (int *)slot_time, fp->ofdm_symbol_size, 1, fp->nb_prefix_samples0, CYCLIC_PREFIX);
  PHY_ofdm_mod((int *)fft_in + fp->ofdm_symbol_size,
               (int *)&slot_time[fp->nb_prefix_samples0 + fp->ofdm_symbol_size],
               fp->ofdm_symbol_size,
               fp->symbols_per_slot - 1,
               fp->nb_prefix_samples,
               CYCLIC_PREFIX);

  const int ssb_start = get_samples_symbol_timestamp(fp, slot, slot_symbol);
  const int ssb_length = get_samples_symbol_duration(fp, slot, slot_symbol, NR_N_SYMBOLS_SSB);
  memcpy(ssb_time, &slot_time[ssb_start], ssb_length * sizeof(*ssb_time));

  gNB->gNB_config.cell_config.phy_cell_id.value = original_config_nid;
  fp->Nid_cell = original_frame_nid;
  fp->ssb_start_subcarrier = original_ssb_start_subcarrier;
  return ssb_length;
}

static void pbchsim_accumulate_component(double *capture_re,
                                         double *capture_im,
                                         int capture_samples,
                                         const c16_t *ssb_time,
                                         int ssb_samples,
                                         const pbchsim_ssb_component_t *component,
                                         double sampling_rate_hz)
{
  const int first_source_sample = max(0, -component->insert_sample);
  const int last_source_sample = min(ssb_samples, capture_samples - component->insert_sample);
  for (int source_sample = first_source_sample; source_sample < last_source_sample; source_sample++) {
    const int destination_sample = component->insert_sample + source_sample;
    const double phase = 2.0 * M_PI * component->cfo_hz * destination_sample / sampling_rate_hz;
    const double re = ssb_time[source_sample].r;
    const double im = ssb_time[source_sample].i;
    capture_re[destination_sample] += component->gain * (re * cos(phase) - im * sin(phase));
    capture_im[destination_sample] += component->gain * (re * sin(phase) + im * cos(phase));
  }
}

static void pbchsim_quantize_capture(c16_t *capture, const double *capture_re, const double *capture_im, int capture_samples)
{
  double peak = 0.0;
  for (int sample = 0; sample < capture_samples; sample++) {
    peak = max(peak, fabs(capture_re[sample]));
    peak = max(peak, fabs(capture_im[sample]));
  }
  const double scale = peak > 20000.0 ? 20000.0 / peak : 1.0;
  for (int sample = 0; sample < capture_samples; sample++) {
    capture[sample].r = (int16_t)lround(scale * capture_re[sample]);
    capture[sample].i = (int16_t)lround(scale * capture_im[sample]);
  }
}

static bool pbchsim_configure_capture_case(const char *name,
                                           const NR_DL_FRAME_PARMS *fp,
                                           int native_ssb_start,
                                           int ssb_samples,
                                           pbchsim_capture_case_t *test_case)
{
  const int frame_samples = fp->samples_per_frame;
  const int later_ssb_start = native_ssb_start + 2 * ssb_samples;
  const int boundary_ssb_start = (frame_samples - ssb_samples / 2) & ~3;
  const int exact_end_ssb_start = 2 * frame_samples - ssb_samples;
  const int same_nid2_target = 12; // PCI 0 and 12 share NID2 0.
  *test_case = (pbchsim_capture_case_t){.target_nid_cell = -1, .ofdm_offset_divisor = 8};

  if (!strcmp(name, "positive-cfo")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, native_ssb_start, 1.0, 1300.0};
    test_case->num_components = 1;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = 0;
    test_case->expected_insert_sample = native_ssb_start;
    test_case->expected_cfo_hz = 1300.0;
  } else if (!strcmp(name, "negative-cfo")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, native_ssb_start, 1.0, -1300.0};
    test_case->num_components = 1;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = 0;
    test_case->expected_insert_sample = native_ssb_start;
    test_case->expected_cfo_hz = -1300.0;
  } else if (!strcmp(name, "internal-boundary")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, boundary_ssb_start, 1.0, 900.0};
    test_case->num_components = 1;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = 0;
    test_case->expected_insert_sample = boundary_ssb_start;
    test_case->expected_cfo_hz = 900.0;
  } else if (!strcmp(name, "same-pci-retry")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, false, native_ssb_start, 1.0, 800.0};
    test_case->component[1] = (pbchsim_ssb_component_t){0, true, later_ssb_start, 0.75, 800.0};
    test_case->num_components = 2;
    test_case->target_nid_cell = 0;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = 0;
    test_case->expected_insert_sample = later_ssb_start;
    test_case->expected_cfo_hz = 800.0;
  } else if (!strcmp(name, "opposite-target-late")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, native_ssb_start, 1.0, 1300.0};
    test_case->component[1] = (pbchsim_ssb_component_t){same_nid2_target, true, later_ssb_start, 0.75, -1300.0};
    test_case->num_components = 2;
    test_case->target_nid_cell = same_nid2_target;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = same_nid2_target;
    test_case->expected_insert_sample = later_ssb_start;
    test_case->expected_cfo_hz = -1300.0;
  } else if (!strcmp(name, "opposite-target-early")) {
    test_case->component[0] = (pbchsim_ssb_component_t){same_nid2_target, true, native_ssb_start, 0.75, -1300.0};
    test_case->component[1] = (pbchsim_ssb_component_t){0, true, later_ssb_start, 1.0, 1300.0};
    test_case->num_components = 2;
    test_case->target_nid_cell = same_nid2_target;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = same_nid2_target;
    test_case->expected_insert_sample = native_ssb_start;
    test_case->expected_cfo_hz = -1300.0;
  } else if (!strcmp(name, "target-absent")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, native_ssb_start, 1.0, 1300.0};
    test_case->num_components = 1;
    test_case->target_nid_cell = 3;
    test_case->expect_detected = false;
  } else if (!strcmp(name, "exact-end")) {
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, exact_end_ssb_start, 1.0, 500.0};
    test_case->num_components = 1;
    test_case->target_nid_cell = 0;
    test_case->expect_detected = true;
    test_case->expected_nid_cell = 0;
    test_case->expected_insert_sample = exact_end_ssb_start;
    test_case->expected_cfo_hz = 500.0;
  } else if (!strcmp(name, "incomplete-end")) {
    const int missing_samples = fp->nb_prefix_samples / test_case->ofdm_offset_divisor;
    test_case->component[0] = (pbchsim_ssb_component_t){0, true, exact_end_ssb_start + missing_samples, 1.0, 500.0};
    test_case->num_components = 1;
    test_case->target_nid_cell = 0;
    test_case->expect_detected = false;
  } else {
    return false;
  }

  for (int component = 0; component < test_case->num_components; component++) {
    const int pss_position = test_case->component[component].insert_sample + fp->nb_prefix_samples;
    AssertFatal(pss_position % 4 == 0, "Case %s has unaligned PSS position %d\n", name, pss_position);
  }
  return true;
}

static int pbchsim_run_pss_candidate_case(const NR_DL_FRAME_PARMS *fp)
{
#ifndef NR_PSS_SEARCH_MAX_CANDIDATES
  printf("SSB capture regression pss-candidates: UNSUPPORTED (multi-peak API unavailable)\n");
  return 2;
#else
  const int sequence_samples = fp->ofdm_symbol_size;
  const int capture_samples = 12 * sequence_samples;
  c16_t *capture = malloc16_clear(capture_samples * sizeof(*capture));
  c16_t *snapshot = malloc16(capture_samples * sizeof(*snapshot));
  AssertFatal(capture != NULL && snapshot != NULL, "Could not allocate PSS candidate fixture\n");
  __attribute__((aligned(32))) c16_t pss_time[NUMBER_PSS_SEQUENCE][sequence_samples];
  for (int nid2 = 0; nid2 < NUMBER_PSS_SEQUENCE; nid2++)
    generate_pss_nr_time(sequence_samples, fp->first_carrier_offset, nid2, fp->ssb_start_subcarrier, pss_time[nid2]);
  const int first_position = sequence_samples;
  const int second_position = 6 * sequence_samples;
  memcpy(&capture[first_position], pss_time[0], sequence_samples * sizeof(*capture));
  memcpy(&capture[second_position], pss_time[0], sequence_samples * sizeof(*capture));
  memcpy(snapshot, capture, capture_samples * sizeof(*snapshot));
  c16_t *rxdata[1] = {capture};
  const pss_search_t search = {.rxdata = rxdata,
                               .nb_antennas_rx = 1,
                               .rxdata_length = capture_samples,
                               .ofdm_symbol_size = sequence_samples,
                               .nb_prefix_samples = fp->nb_prefix_samples,
                               .subcarrier_spacing = fp->subcarrier_spacing,
                               .fo_flag = false,
                               .target_Nid_cell = -1,
                               .pssTime = (c16_t *)pss_time};
  pss_detection_result_t candidates[NR_PSS_SEARCH_MAX_CANDIDATES] = {0};
  bool full_truncated = false;
  const size_t candidate_count = pss_search_time_nr_candidates(&search, candidates, NR_PSS_SEARCH_MAX_CANDIDATES, &full_truncated);
  int first_index = -1;
  int second_index = -1;
  for (size_t candidate = 0; candidate < candidate_count; candidate++) {
    if (candidates[candidate].nid2 == 0 && candidates[candidate].pos == first_position)
      first_index = candidate;
    if (candidates[candidate].nid2 == 0 && candidates[candidate].pos == second_position)
      second_index = candidate;
  }

  pss_search_t targeted_search = search;
  targeted_search.target_Nid_cell = 0;
  pss_detection_result_t targeted_candidates[NR_PSS_SEARCH_MAX_CANDIDATES] = {0};
  bool targeted_truncated = false;
  const size_t targeted_count =
      pss_search_time_nr_candidates(&targeted_search, targeted_candidates, NR_PSS_SEARCH_MAX_CANDIDATES, &targeted_truncated);
  int targeted_first_index = -1;
  int targeted_second_index = -1;
  for (size_t candidate = 0; candidate < targeted_count; candidate++) {
    if (targeted_candidates[candidate].pos == first_position)
      targeted_first_index = candidate;
    if (targeted_candidates[candidate].pos == second_position)
      targeted_second_index = candidate;
  }

  pss_detection_result_t one_candidate = {0};
  bool one_truncated = false;
  const size_t one_count = pss_search_time_nr_candidates(&targeted_search, &one_candidate, 1, &one_truncated);
  const bool input_unchanged = memcmp(capture, snapshot, capture_samples * sizeof(*capture)) == 0;
  const bool passed = candidate_count >= 2 && !full_truncated && first_index >= 0 && second_index > first_index && one_count == 1
                      && targeted_count >= 2 && targeted_count < NR_PSS_SEARCH_MAX_CANDIDATES && !targeted_truncated
                      && targeted_first_index >= 0 && targeted_second_index > targeted_first_index && one_truncated
                      && one_candidate.nid2 == 0 && one_candidate.pos == first_position && input_unchanged;
  printf(
      "SSB capture regression pss-candidates: count=%zu positions=%d,%d targeted=%zu targeted_positions=%d,%d "
      "cap1=%d truncated=%d immutable=%d: %s\n",
      candidate_count,
      first_index,
      second_index,
      targeted_count,
      targeted_first_index,
      targeted_second_index,
      one_candidate.pos,
      one_truncated,
      input_unchanged,
      passed ? "PASS" : "FAIL");
  free(snapshot);
  free(capture);
  return passed ? 0 : 1;
#endif
}

static int pbchsim_run_capture_case(const char *name,
                                    PHY_VARS_gNB *gNB,
                                    PHY_VARS_NR_UE *ue,
                                    c16_t *slot_time,
                                    int ssb_subcarrier_offset)
{
  NR_DL_FRAME_PARMS *fp = &ue->frame_parms;
  const int capture_samples = 2 * fp->samples_per_frame;
  const int ssb_index = 0;
  const int start_symbol = nr_get_ssb_start_symbol(fp, ssb_index);
  const int slot = start_symbol / fp->symbols_per_slot;
  const int slot_symbol = start_symbol % fp->symbols_per_slot;
  const int native_ssb_start = get_samples_slot_timestamp(fp, slot) + get_samples_symbol_timestamp(fp, slot, slot_symbol);
  const int expected_ssb_samples = get_samples_symbol_duration(fp, slot, slot_symbol, NR_N_SYMBOLS_SSB);
  pbchsim_capture_case_t test_case;
  if (!pbchsim_configure_capture_case(name, fp, native_ssb_start, expected_ssb_samples, &test_case)) {
    printf("Unknown SSB capture regression case: %s\n", name);
    return 2;
  }

  double *capture_re = calloc(capture_samples, sizeof(*capture_re));
  double *capture_im = calloc(capture_samples, sizeof(*capture_im));
  c16_t *ssb_time = malloc16_clear(expected_ssb_samples * sizeof(*ssb_time));
  c16_t *snapshot = malloc16(capture_samples * sizeof(*snapshot));
  AssertFatal(capture_re != NULL && capture_im != NULL && ssb_time != NULL && snapshot != NULL,
              "Could not allocate SSB capture fixture\n");

  for (int component = 0; component < test_case.num_components; component++) {
    const int generated_samples = pbchsim_generate_ssb(gNB,
                                                       slot_time,
                                                       ssb_time,
                                                       ssb_subcarrier_offset,
                                                       test_case.component[component].nid_cell,
                                                       test_case.component[component].valid_pbch);
    AssertFatal(generated_samples == expected_ssb_samples,
                "Generated SSB length %d differs from expected %d\n",
                generated_samples,
                expected_ssb_samples);
    pbchsim_accumulate_component(capture_re,
                                 capture_im,
                                 capture_samples,
                                 ssb_time,
                                 generated_samples,
                                 &test_case.component[component],
                                 fp->samples_per_subframe * 1000.0);
  }
  pbchsim_quantize_capture(ue->common_vars.rxdata[0], capture_re, capture_im, capture_samples);
  memcpy(snapshot, ue->common_vars.rxdata[0], capture_samples * sizeof(*snapshot));

  const int gscn_ssb_start_subcarrier = gNB->frame_parms.ssb_start_subcarrier;
  ue->target_Nid_cell = test_case.target_nid_cell;
  ue->UE_fo_compensation = 1;
  ue->initial_fo = 0;
  ue->frame_parms.ofdm_offset_divisor = test_case.ofdm_offset_divisor;
  ue->frame_parms.Nid_cell = 1000;
  ue->frame_parms.ssb_start_subcarrier = 1234;
  ue->frame_parms.half_frame_bit = 1;
  ue->frame_parms.ssb_index = 63;
  ue->symbol_offset = 200;
  ue->common_vars.freq_offset = 12345.0;
  ue->adjust_rxgain = 77;
  ue->init_sync_frame = 99;
  const pbchsim_sync_state_t state_before = pbchsim_get_sync_state(ue);

  UE_nr_rxtx_proc_t proc = {0};
  nr_gscn_info_t gscn_info[MAX_GSCN_BAND] = {0};
  gscn_info[0].ssbFirstSC = gscn_ssb_start_subcarrier;
  const nr_initial_sync_t result = nr_initial_sync(&proc, ue, 2, gscn_info, 1);
  const pbchsim_sync_state_t state_after = pbchsim_get_sync_state(ue);
  const bool input_unchanged = memcmp(ue->common_vars.rxdata[0], snapshot, capture_samples * sizeof(*snapshot)) == 0;

  bool passed = result.cell_detected == test_case.expect_detected && input_unchanged;
  if (test_case.expect_detected) {
    const int64_t sync_delta = (int64_t)test_case.expected_insert_sample - native_ssb_start;
    const int64_t frame_delta = pbchsim_floor_divide(sync_delta, fp->samples_per_frame);
    const int expected_rx_offset = sync_delta - frame_delta * fp->samples_per_frame;
    const int expected_frame_id = test_case.expected_insert_sample / fp->samples_per_frame;
    const int expected_init_sync_frame = 1 - frame_delta;
    const double cfo_error_hz = fabs(state_after.freq_offset - test_case.expected_cfo_hz);
    passed = passed && result.rx_offset == expected_rx_offset && result.frame_id == expected_frame_id
             && state_after.nid_cell == test_case.expected_nid_cell && state_after.ssb_start_subcarrier == gscn_ssb_start_subcarrier
             && state_after.half_frame_bit == 0 && state_after.ssb_index == 0 && state_after.symbol_offset == start_symbol
             && state_after.init_sync_frame == expected_init_sync_frame && state_after.adjust_rxgain != state_before.adjust_rxgain
             && cfo_error_hz <= 200.0;
    printf("SSB capture regression %s: detected=%d pci=%d frame=%d rx_offset=%d cfo=%.0f error=%.0f immutable=%d: %s\n",
           name,
           result.cell_detected,
           state_after.nid_cell,
           result.frame_id,
           result.rx_offset,
           state_after.freq_offset,
           cfo_error_hz,
           input_unchanged,
           passed ? "PASS" : "FAIL");
  } else {
    passed = passed && result.rx_offset == 0 && result.frame_id == 0 && pbchsim_sync_state_equal(&state_before, &state_after);
    printf("SSB capture regression %s: detected=%d state_unchanged=%d immutable=%d: %s\n",
           name,
           result.cell_detected,
           pbchsim_sync_state_equal(&state_before, &state_after),
           input_unchanged,
           passed ? "PASS" : "FAIL");
  }

  free(snapshot);
  free(ssb_time);
  free(capture_im);
  free(capture_re);
  return passed ? 0 : 1;
}

configmodule_interface_t *uniqCfg = NULL;
int main(int argc, char **argv)
{
  stop = false;
  __attribute__((unused)) struct sigaction oldaction;
  sigaction(SIGINT, &sigint_action, &oldaction);

  int i,aa,start_symbol;
  double sigma2, sigma2_dB=10,SNR,snr0=-2.0,snr1=2.0;
  double cfo=0;
  uint8_t snr1set=0;
  c16_t **txdata;
  double **s_re,**s_im,**r_re,**r_im;
  //double iqim = 0.0;
  double ip =0.0;
  //unsigned char pbch_pdu[6];
  //  int sync_pos, sync_pos_slot;
  //  FILE *rx_frame_file;
  FILE *output_fd = NULL;
  //uint8_t write_output_file=0;
  //int result;
  //int freq_offset;
  //  int subframe_offset;
  //  char fname[40], vname[40];
  int trial,n_trials=1,n_errors=0,n_errors_payload=0;
  int ret_test = 1;
  uint8_t transmission_mode = 1,n_tx=1,n_rx=1;
  uint16_t Nid_cell=0;
  uint64_t SSB_positions=0x01;
  int ssb_subcarrier_offset = 0;
  int ssb_scan_threads = 0;
  const char *ssb_capture_regression = NULL;

  channel_desc_t *gNB2UE;

  //uint8_t extended_prefix_flag=0;
  //int8_t interf1=-21,interf2=-21;

  FILE *input_fd=NULL,*pbch_file_fd=NULL;

  //uint32_t nsymb,tx_lev,tx_lev1 = 0,tx_lev2 = 0;
  //char input_val_str[50],input_val_str2[50];
  //uint8_t frame_mod4,num_pdcch_symbols = 0;
  //double pbch_sinr;
  //int pbch_tx_ant;

  SCM_t channel_model=AWGN;//Rayleigh1_anticorr;


  int N_RB_DL=273,mu=1;

  //unsigned char frame_type = 0;
  unsigned char pbch_phase = 0;

  int frame=0;
  int frame_length_complex_samples;
  __attribute__((unused))
  int frame_length_complex_samples_no_prefix;
  NR_DL_FRAME_PARMS *frame_parms;

  int ret;
  int run_initial_sync=0;

  int loglvl=OAILOG_WARNING;

  float target_error_rate = 0.01;

  cpuf = get_cpu_freq_GHz();

  if ((uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY)) == 0) {
    exit_fun("[NR_PBCHSIM] Error, configuration module init failed\n");
  }

  int c;
  while ((c = getopt(argc, argv, "--:O:c:F:g:hIL:m:M:n:N:o:P:Q:R:s:S:x:y:z:")) != -1) {
    /* ignore long options starting with '--', option '-O' and their arguments that are handled by configmodule */
    /* with this opstring getopt returns 1 for non-option arguments, refer to 'man 3 getopt' */
    if (c == 1 || c == '-' || c == 'O')
      continue;

    printf("handling optarg %c\n",c);
    switch (c) {

    case 'c':
      ssb_subcarrier_offset = atoi(optarg);
      break;

    /*case 'f':
      write_output_file=1;
      output_fd = fopen(optarg,"w");

      if (output_fd==NULL) {
        printf("Error opening %s\n",optarg);
        exit(-1);
      }

      break;*/

    /*case 'd':
      frame_type = 1;
      break;*/

    case 'F':
      input_fd = fopen(optarg,"r");
      if (input_fd==NULL) {
        printf("Problem with filename %s. Exiting.\n", optarg);
        exit(-1);
      }
      break;

    case 'g':
      switch((char)*optarg) {
      case 'A':
        channel_model=SCM_A;
        break;

      case 'B':
        channel_model=SCM_B;
        break;

      case 'C':
        channel_model=SCM_C;
        break;

      case 'D':
        channel_model=SCM_D;
        break;

      case 'E':
        channel_model=EPA;
        break;

      case 'F':
        channel_model=EVA;
        break;

      case 'G':
        channel_model=ETU;
        break;

      default:
        printf("Unsupported channel model! Exiting.\n");
        exit(-1);
      }

      break;

    /*
    case 'i':
      interf1=atoi(optarg);
      break;
    */

    case 'I':
      run_initial_sync=1;
      target_error_rate=0.1;
      break;

    /*
    case 'j':
      interf2=atoi(optarg);
      break;*/

    case 'L':
      loglvl = atoi(optarg);
      break;

    case 'm':
      mu = atoi(optarg);
      break;

    case 'M':
      SSB_positions = atoi(optarg);
      break;

    case 'n':
      n_trials = atoi(optarg);
      break;

    case 'N':
      Nid_cell = atoi(optarg);
      break;

    case 'o':
      cfo = atof(optarg);
#ifdef DEBUG_NR_PBCHSIM
      printf("Setting CFO to %f Hz\n",cfo);
#endif
      break;

    /*case 'p':
      extended_prefix_flag=1;
      break;*/

    case 'P':
      pbch_phase = atoi(optarg);
      if (pbch_phase>3)
        printf("Illegal PBCH phase (0-3) got %d\n",pbch_phase);
      break;

    case 'Q':
      ssb_capture_regression = optarg;
      run_initial_sync = 1;
      break;

    case 'R':
      N_RB_DL = atoi(optarg);
      break;

    case 's':
      snr0 = atof(optarg);
#ifdef DEBUG_NR_PBCHSIM
      printf("Setting SNR0 to %f\n",snr0);
#endif
      break;

    case 'S':
      snr1 = atof(optarg);
      snr1set=1;
#ifdef DEBUG_NR_PBCHSIM
      printf("Setting SNR1 to %f\n",snr1);
#endif
      break;

      /*
      case 't':
      Td= atof(optarg);
      break;
      */

    case 'x':
      transmission_mode=atoi(optarg);

      if ((transmission_mode!=1) && (transmission_mode!=2) && (transmission_mode!=6)) {
        printf("Unsupported transmission mode %d. Exiting.\n",transmission_mode);
        exit(-1);
      }

      break;

    case 'y':
      n_tx=atoi(optarg);
      if ((n_tx==0) || (n_tx>2)) {
        printf("Unsupported number of TX antennas %d. Exiting.\n", n_tx);
        exit(-1);
      }
      break;

    case 'z':
      n_rx=atoi(optarg);
      if ((n_rx==0) || (n_rx>2)) {
        printf("Unsupported number of RX antennas %d. Exiting.\n", n_rx);
        exit(-1);
      }
      break;

    default:
    case 'h':
      printf(
          "OAI_RNGSEED=xxx ./%s -F input_filename -g channel_mod -h(elp) -I(nitial sync) -L log_lvl -n n_frames -M SSBs -n frames "
          "-N cell_id -o FO -P phase -Q case -R RBs -s snr0 -S snr1 -x transmission_mode -y TXant -z RXant\n",
          argv[0]);
      //printf("-A Interpolation_filname Run with Abstraction to generate Scatter plot using interpolation polynomial in file\n");
      printf("-c SSB subcarrier offset\n");
      //printf("-C Generate Calibration information for Abstraction (effective SNR adjustment to remove Pe bias w.r.t. AWGN)\n");
      //printf("-d Use TDD\n");
      //printf("-f Output filename (.txt format) for Pe/SNR results\n");
      printf("-F Input filename (.txt format) for RX conformance testing\n");
      printf("-g [A,B,C,D,E,F,G] Use 3GPP SCM (A,B,C,D) or 36-101 (E-EPA,F-EVA,G-ETU) models (ignores delay spread and Ricean factor)\n");
      printf("-h This message\n");
      //printf("-i Relative strength of first intefering eNB (in dB) - cell_id mod 3 = 1\n");
      printf("-I run initial sync with target error rate 0.1\n");
      //printf("-j Relative strength of second intefering eNB (in dB) - cell_id mod 3 = 2\n");
      printf("-L set the log level (-1 disable, 0 error, 1 warning, 2 info, 3 debug, 4 trace)\n");
      printf("-m Numerology index\n");
      printf("-M Multiple SSB positions in burst\n");
      printf("-n Number of frames to simulate\n");
      printf("-N Nid_cell\n");
      printf("-o Carrier frequency offset in Hz\n");
      //printf("-O oversampling factor (1,2,4,8,16)\n");
      //printf("-p Use extended prefix mode\n");
      printf("-P PBCH phase, allowed values 0-3\n");
      printf("-Q Run one deterministic linear-capture initial-sync regression case\n");
      printf("-R N_RB_DL\n");
      printf("-s Starting SNR, runs from SNR0 to SNR0 + 10 dB if not -S given. If -n 1, then just SNR is simulated\n");
      printf("-S Ending SNR, runs from SNR0 to SNR1\n");
      //printf("-t Delay spread for multipath channel\n");
      printf("-x Transmission mode (1,2,6 for the moment)\n");
      printf("-y Number of TX antennas used in eNB\n");
      printf("-z Number of RX antennas used in UE\n");
      exit (-1);
      break;
    }
  }

  if (ssb_capture_regression != NULL) {
    AssertFatal(input_fd == NULL, "The SSB capture regression does not accept an input file\n");
    AssertFatal(mu == 0 && N_RB_DL == 25, "The SSB capture regression requires -m0 -R25\n");
    AssertFatal(n_tx == 1 && n_rx == 1, "The SSB capture regression requires one TX and one RX antenna\n");
  }

  randominit();

  logInit();
  set_glog(loglvl);

  if (snr1set==0)
    snr1 = snr0+10;

  printf("Initializing gNodeB for mu %d, N_RB_DL %d\n",mu,N_RB_DL);


  RC.gNB = (PHY_VARS_gNB**) malloc(sizeof(PHY_VARS_gNB *));
  RC.gNB[0] = malloc16_clear(sizeof(*(RC.gNB[0])));
  gNB = RC.gNB[0];
  gNB->ofdm_offset_divisor = UINT_MAX;
  frame_parms = &gNB->frame_parms; //to be initialized I suppose (maybe not necessary for PBCH)
  frame_parms->nb_antennas_tx = n_tx;
  frame_parms->nb_antennas_rx = n_rx;
  frame_parms->nb_antenna_ports_gNB = n_tx;
  frame_parms->N_RB_DL = N_RB_DL;
  frame_parms->Nid_cell = Nid_cell;
  frame_parms->ssb_type = nr_ssb_type_C;
  frame_parms->freq_range = mu<2 ? FR1 : FR2;

  nr_phy_config_request_sim(gNB, N_RB_DL, N_RB_DL, mu, Nid_cell, SSB_positions);
  // TDD configuration
  gNB->gNB_config.tdd_table.tdd_period.value = 6;
  do_tdd_config_sim(gNB, mu);

  phy_init_nr_gNB(gNB);
  frame_parms->ssb_start_subcarrier = 12 * gNB->gNB_config.ssb_table.ssb_offset_point_a.value + ssb_subcarrier_offset;
  initFloatingCoresTpool(ssb_scan_threads, &nrUE_params.Tpool, false, "UE-tpool");

  int n_hf = 0;
  int cyclic_prefix_type = NFAPI_CP_NORMAL;

  double fs=0, eps;
  double scs = 30000;
  double txbw, rxbw;
  get_samplerate_and_bw(mu, N_RB_DL, frame_parms->threequarter_fs, &fs, &txbw, &rxbw);

  // cfo with respect to sub-carrier spacing
  eps = cfo/scs;

  // computation of integer and fractional FO to compare with estimation results
  int IFO;
  if(eps!=0.0){
	printf("Introducing a CFO of %lf relative to SCS of %d kHz\n",eps,(int)(scs/1000));
	if (eps>0)	
  	  IFO=(int)(eps+0.5);
	else
	  IFO=(int)(eps-0.5);
	printf("FFO = %lf; IFO = %d\n",eps-IFO,IFO);
  }

  gNB2UE = new_channel_desc_scm(n_tx, n_rx, channel_model, fs, 0, txbw, 300e-9, 0.0, CORR_LEVEL_LOW, 0, 0, 0, 0);

  if (gNB2UE==NULL) {
	printf("Problem generating channel model. Exiting.\n");
    exit(-1);
  }

  frame_length_complex_samples = frame_parms->samples_per_subframe*NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  frame_length_complex_samples_no_prefix = frame_parms->samples_per_subframe_wCP;

  s_re = malloc(2*sizeof(double*));
  s_im = malloc(2*sizeof(double*));
  r_re = malloc(2*sizeof(double*));
  r_im = malloc(2*sizeof(double*));
  txdata = calloc(2, sizeof(c16_t*));

  for (i=0; i<2; i++) {


    s_re[i] = malloc16_clear(frame_length_complex_samples*sizeof(double));
    s_im[i] = malloc16_clear(frame_length_complex_samples*sizeof(double));
    r_re[i] = malloc16_clear(frame_length_complex_samples*sizeof(double));
    r_im[i] = malloc16_clear(frame_length_complex_samples*sizeof(double));
    printf("Allocating %d samples for txdata\n",frame_length_complex_samples);
    txdata[i] = malloc16_clear(frame_length_complex_samples * sizeof(c16_t));
  }

  if (pbch_file_fd!=NULL) {
    load_pbch_desc(pbch_file_fd);
  }


  //configure UE

  UE = malloc16_clear(sizeof(*UE));
  memcpy(&UE->frame_parms,frame_parms,sizeof(UE->frame_parms));
  //phy_init_nr_top(UE); //called from init_nr_ue_signal
  if (run_initial_sync==1)  UE->is_synchronized = 0;
  else                      UE->is_synchronized = 1;
                      
  if(eps!=0.0)
	UE->UE_fo_compensation = 1; // if a frequency offset is set then perform fo estimation and compensation

  if (init_nr_ue_signal(UE, 1) != 0) {
    printf("Error at UE NR initialisation\n");
    exit(-1);
  }

  // generate signal
  const uint32_t rxdataF_sz = UE->frame_parms.samples_per_slot_wCP;
  __attribute__ ((aligned(32))) c16_t rxdataF[UE->frame_parms.nb_antennas_rx][rxdataF_sz];
  nfapi_nr_dl_tti_ssb_pdu ssb_pdu[64] = {0};
  if (ssb_capture_regression != NULL) {
    ret_test = !strcmp(ssb_capture_regression, "pss-candidates")
                   ? pbchsim_run_pss_candidate_case(&UE->frame_parms)
                   : pbchsim_run_capture_case(ssb_capture_regression, gNB, UE, txdata[0], ssb_subcarrier_offset);
    goto cleanup;
  }

  if (input_fd==NULL) {

    for (i=0; i<frame_parms->Lmax; i++) {
      if((SSB_positions >> i) & 0x01) {

        const int sc_offset = frame_parms->freq_range == FR1 ? ssb_subcarrier_offset<<mu : ssb_subcarrier_offset;
        const int prb_offset = frame_parms->freq_range == FR1 ? gNB->gNB_config.ssb_table.ssb_offset_point_a.value<<mu : gNB->gNB_config.ssb_table.ssb_offset_point_a.value << (mu - 2);
        ssb_pdu[i].ssb_pdu_rel15.bchPayload = 0x55dd33;
        ssb_pdu[i].ssb_pdu_rel15.SsbBlockIndex = i;
        ssb_pdu[i].ssb_pdu_rel15.SsbSubcarrierOffset = sc_offset;
        ssb_pdu[i].ssb_pdu_rel15.ssbOffsetPointA = prb_offset;

        start_symbol = nr_get_ssb_start_symbol(frame_parms,i);
        int slot = start_symbol/14;

        for (aa=0; aa<gNB->frame_parms.nb_antennas_tx; aa++)
          memset(gNB->common_vars.txdataF[aa], 0, frame_parms->samples_per_slot_wCP * sizeof(int32_t));

        nr_common_signal_procedures (gNB,frame,slot, &ssb_pdu[i]);

        int samp = get_samples_slot_timestamp(frame_parms, slot);
        for (aa = 0; aa < gNB->frame_parms.nb_antennas_tx; aa++) {
          c16_t fft_in_buff[frame_parms->ofdm_symbol_size * frame_parms->symbols_per_slot] __attribute__((aligned(64)));
          memset(fft_in_buff, 0, sizeof(fft_in_buff));
          if (cyclic_prefix_type == 1) {
            apply_nr_rotation_TX(frame_parms,
                                 gNB->common_vars.txdataF[aa],
                                 true,
                                 frame_parms->symbol_rotation[0],
                                 slot,
                                 frame_parms->N_RB_DL,
                                 0,
                                 12);

            fft_shift(gNB->common_vars.txdataF[aa],
                      frame_parms->ofdm_symbol_size,
                      frame_parms->N_RB_DL,
                      fft_in_buff,
                      frame_parms->ofdm_symbol_size,
                      0,
                      12);

            PHY_ofdm_mod((int *)fft_in_buff,
                         (int *)&txdata[aa][samp],
                         frame_parms->ofdm_symbol_size,
                         12,
                         frame_parms->nb_prefix_samples,
                         CYCLIC_PREFIX);
          } else {
            apply_nr_rotation_TX(frame_parms,
                                 gNB->common_vars.txdataF[aa],
                                 true,
                                 frame_parms->symbol_rotation[0],
                                 slot,
                                 frame_parms->N_RB_DL,
                                 0,
                                 14);

            fft_shift(gNB->common_vars.txdataF[aa],
                      frame_parms->ofdm_symbol_size,
                      frame_parms->N_RB_DL,
                      fft_in_buff,
                      frame_parms->ofdm_symbol_size,
                      0,
                      14);

            PHY_ofdm_mod((int *)fft_in_buff,
                         (int *)&txdata[aa][samp],
                         frame_parms->ofdm_symbol_size,
                         1,
                         frame_parms->nb_prefix_samples0,
                         CYCLIC_PREFIX);

            PHY_ofdm_mod((int *)fft_in_buff + frame_parms->ofdm_symbol_size,
                         (int *)&txdata[aa][samp + frame_parms->nb_prefix_samples0 + frame_parms->ofdm_symbol_size],
                         frame_parms->ofdm_symbol_size,
                         13,
                         frame_parms->nb_prefix_samples,
                         CYCLIC_PREFIX);
          }
        }
      }
    }
    LOG_M("txsigF0.m", "txsF0", gNB->common_vars.txdataF[0], frame_parms->samples_per_slot_wCP, 1, 1);
    if (gNB->frame_parms.nb_antennas_tx > 1)
      LOG_M("txsigF1.m", "txsF1", gNB->common_vars.txdataF[1], frame_parms->samples_per_slot_wCP, 1, 1);

  } else {
    printf("Reading %d samples from file to antenna buffer %d\n",frame_length_complex_samples,0);
    UE->UE_fo_compensation = 1; // perform fo compensation when samples from file are used
    if (fread(txdata[0],
        sizeof(int32_t),
        frame_length_complex_samples,
        input_fd) != frame_length_complex_samples) {
      printf("error reading from file\n");
      //exit(-1);
    }
  }

  LOG_M("txsig0.m","txs0", txdata[0],frame_length_complex_samples,1,1);
  if (gNB->frame_parms.nb_antennas_tx>1)
    LOG_M("txsig1.m","txs1", txdata[1],frame_length_complex_samples,1,1);

  if (output_fd) 
    fwrite(txdata[0],sizeof(int32_t),frame_length_complex_samples,output_fd);

  /*int txlev = signal_energy(&txdata[0][5*frame_parms->ofdm_symbol_size + 4*frame_parms->nb_prefix_samples + frame_parms->nb_prefix_samples0],
		  	  	  	  	    frame_parms->ofdm_symbol_size + frame_parms->nb_prefix_samples);
  printf("txlev %d (%f)\n",txlev,10*log10(txlev));*/

  
  for (SNR = snr0; SNR < snr1 && !stop; SNR+=.2) {

    n_errors = 0;
    n_errors_payload = 0;

    for (trial = 0; trial < n_trials && !stop; trial++) {

      for (i=0; i<frame_length_complex_samples; i++) {
        for (aa=0; aa<frame_parms->nb_antennas_tx; aa++) {
          r_re[aa][i] = (double)txdata[aa][i].r;
          r_im[aa][i] = (double)txdata[aa][i].i;
        }
      }

      // multipath channel
      //multipath_channel(gNB2UE,s_re,s_im,r_re,r_im,frame_length_complex_samples,0);
      
      //AWGN
      sigma2_dB = 20*log10((double)AMP/4)-SNR;
      sigma2 = pow(10,sigma2_dB/10);
      //printf("sigma2 %f (%f dB), tx_lev %f (%f dB)\n",sigma2,sigma2_dB,txlev,10*log10((double)txlev));

      if(eps!=0.0)
        rf_rx(r_re,  // real part of txdata
           r_im,  // imag part of txdata
           NULL,  // interference real part
           NULL, // interference imag part
           0,  // interference power
           frame_parms->nb_antennas_rx,  // number of rx antennas
           frame_length_complex_samples,  // number of samples in frame
           1.0e9/fs,   //sampling time (ns)
           cfo,	// frequency offset in Hz
           0.0, // drift (not implemented)
           0.0, // noise figure (not implemented)
           0.0, // rx gain in dB ?
           200, // 3rd order non-linearity in dB ?
           &ip, // initial phase
           30.0e3,  // phase noise cutoff in kHz
           -500.0, // phase noise amplitude in dBc
           0.0,  // IQ imbalance (dB),
	   0.0); // IQ phase imbalance (rad)

      for (i=0; i<frame_length_complex_samples; i++) {
        for (aa=0; aa<frame_parms->nb_antennas_rx; aa++) {
          UE->common_vars.rxdata[aa][i].r = (short)(r_re[aa][i] + sqrt(sigma2 / 2) * gaussdouble(0.0, 1.0));
          UE->common_vars.rxdata[aa][i].i = (short)(r_im[aa][i] + sqrt(sigma2 / 2) * gaussdouble(0.0, 1.0));
        }
      }

      if (n_trials==1) {
        LOG_M("rxsig0.m", "rxs0", UE->common_vars.rxdata[0], frame_parms->samples_per_frame, 1, 1);
        if (gNB->frame_parms.nb_antennas_tx > 1)
          LOG_M("rxsig1.m", "rxs1", UE->common_vars.rxdata[1], frame_parms->samples_per_frame, 1, 1);
      }
      if (UE->is_synchronized == 0) {
        UE_nr_rxtx_proc_t proc = {0};
        nr_gscn_info_t gscnInfo[MAX_GSCN_BAND] = {0};
        const int numGscn = 1;
        gscnInfo[0].ssbFirstSC = frame_parms->ssb_start_subcarrier;
        nr_initial_sync_t ret = nr_initial_sync(&proc, UE, 1, gscnInfo, numGscn);
        printf("nr_initial_sync1 returns %s\n", ret.cell_detected ? "cell detected" : "cell not detected");
        if (!ret.cell_detected)
          n_errors++;
      }
      else {
        UE_nr_rxtx_proc_t proc={0};

        uint8_t ssb_index = 0;
        while (!((SSB_positions >> ssb_index) & 0x01))
          ssb_index++; // to select the first transmitted ssb
        UE->symbol_offset = nr_get_ssb_start_symbol(frame_parms, ssb_index);

        int ssb_slot = (UE->symbol_offset/14)+(n_hf*(frame_parms->slots_per_frame>>1));
        proc.nr_slot_rx = ssb_slot;
        proc.gNB_id = 0;
        int16_t pbch_e_rx[NR_POLAR_PBCH_E];
        uint8_t log2_maxh = 0;
        for (int i = UE->symbol_offset + 1; i < UE->symbol_offset + 4; i++) {
          nr_slot_fep(UE,
                      frame_parms,
                      proc.nr_slot_rx,
                      i % frame_parms->symbols_per_slot,
                      rxdataF,
                      link_type_dl,
                      0,
                      UE->common_vars.rxdata);
          __attribute__((aligned(32))) struct complex16 rxdataF_symb[frame_parms->nb_antennas_rx][frame_parms->ofdm_symbol_size];
          __attribute__((aligned(32))) struct complex16 dl_ch_estimates[frame_parms->nb_antennas_rx][frame_parms->ofdm_symbol_size];

          for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
            memcpy(rxdataF_symb[aarx],
                   &rxdataF[0][i * frame_parms->ofdm_symbol_size],
                   sizeof(c16_t) * frame_parms->ofdm_symbol_size);
            nr_pbch_channel_estimation(frame_parms,
                                       &UE->SL_UE_PHY_PARAMS,
                                       dl_ch_estimates[aarx],
                                       &proc,
                                       i - (UE->symbol_offset + 1),
                                       ssb_index % 8,
                                       n_hf,
                                       frame_parms->ssb_start_subcarrier,
                                       rxdataF_symb[aarx],
                                       false,
                                       frame_parms->Nid_cell);
          }
          nr_generate_pbch_llr(UE,
                               &proc,
                               frame_parms,
                               i - UE->symbol_offset,
                               ssb_index % 8,
                               Nid_cell,
                               frame_parms->ssb_start_subcarrier,
                               rxdataF_symb,
                               dl_ch_estimates,
                               pbch_e_rx,
                               &log2_maxh);
        }
        fapiPbch_t result;
        int ret_ssb_idx;
        int ret_symbol_offset;
        ret = nr_pbch_decode(UE,
                             frame_parms,
                             &proc,
                             ssb_index % 8,
                             Nid_cell,
                             pbch_e_rx,
                             &n_hf,
                             &ret_ssb_idx,
                             &ret_symbol_offset,
                             &result);

        if (ret == 0) {
          uint32_t xtra_byte = nr_pbch_extra_byte_generation(frame,
                                                             n_hf,
                                                             ssb_index,
                                                             gNB->gNB_config.ssb_table.ssb_subcarrier_offset.value,
                                                             frame_parms->Lmax);
          int payload_ret = (result.xtra_byte == xtra_byte);
          nfapi_nr_dl_tti_ssb_pdu_rel15_t *pdu = &ssb_pdu[ssb_index].ssb_pdu_rel15;
          for (int i = 0; i < 3; i++)
            payload_ret += (result.decoded_output[i] == ((pdu->bchPayload >> (8 * i)) & 0xff));
          // printf("ret %d\n", payload_ret);
          if (payload_ret != 4)
            n_errors_payload++;
        }

        if (ret != 0)
          n_errors++;
      }
    } //noise trials
    printf("SNR %f: trials %d, n_errors_crc = %d, n_errors_payload %d\n", SNR,n_trials,n_errors,n_errors_payload);

    if (((float)n_errors/(float)n_trials <= target_error_rate) && (n_errors_payload==0)) {
      printf("PBCH test OK\n");
      ret_test = 0;
      break;
    }
      
    if (n_trials==1)
      break;

  } // NSR

cleanup:
  free_channel_desc_scm(gNB2UE);

  int nb_slots_to_set = (1 << mu) * NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  for (int i = 0; i < nb_slots_to_set; ++i)
    free(gNB->gNB_config.tdd_table.max_tdd_periodicity_list[i].max_num_of_symbol_per_slot_list);
  free(gNB->gNB_config.tdd_table.max_tdd_periodicity_list);

  phy_free_nr_gNB(gNB);
  free(RC.gNB[0]);
  free(RC.gNB);

  term_nr_ue_signal(UE);
  free(UE);

  for (i=0; i<2; i++) {
    free(s_re[i]);
    free(s_im[i]);
    free(r_re[i]);
    free(r_im[i]);
    free(txdata[i]);
  }

  free(s_re);
  free(s_im);
  free(r_re);
  free(r_im);
  free(txdata);

  if (output_fd)
    fclose(output_fd);

  if (input_fd)
    fclose(input_fd);

  loader_reset();
  logTerm();

  return ret_test;

}
