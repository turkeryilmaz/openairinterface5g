/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/*!
 * \brief Top-level routines for generating and decoding the PRACH physical channel V15.4 2018-12
 */

#include "PHY/defs_gNB.h"
#include "SCHED_NR/sched_nr.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "openair1/PHY/NR_TRANSPORT/nr_prach.h"
#include "common/utils/LOG/flight_recorder.h"
#include <math.h>

typedef struct {
  int reps;
  int Ncp;
  int dftlen;
  int N_ZC;
  int k;
  dft_size_idx_t dftsize;
  int sample_offset_slot;
} prach_ru_params_t;

static prach_ru_params_t get_prach_ru_params(prach_item_t *p,
                                             int prachStartSymbol,
                                             NR_DL_FRAME_PARMS *fp)
{
  prach_ru_params_t par = {0};
  const int sum = fp->ofdm_symbol_size + fp->nb_prefix_samples;
  const int sum0 = fp->ofdm_symbol_size + fp->nb_prefix_samples0;
  if (prachStartSymbol == 0) {
    par.sample_offset_slot = 0;
  } else if (fp->slots_per_subframe == 1) {
    if (prachStartSymbol <= 7)
      par.sample_offset_slot = sum * (prachStartSymbol - 1) + sum0;
    else
      par.sample_offset_slot = sum * (prachStartSymbol - 2) + sum0 * 2;
  } else {
    if (!(p->slot % (fp->slots_per_subframe / 2)))
      par.sample_offset_slot = sum * (prachStartSymbol - 1) + sum0;
    else
      par.sample_offset_slot = sum * prachStartSymbol;
  }

  int mu = p->numerology_index;

  if (p->prach_sequence_length == 0) {
    switch (p->pdu.prach_format) {
      case 0:
        par.reps = 1;
        par.Ncp = 3168;
        par.dftlen = 24576;
        break;

      case 1:
        par.reps = 2;
        par.Ncp = 21024;
        par.dftlen = 24576;
        break;

      case 2:
        par.reps = 4;
        par.Ncp = 4688;
        par.dftlen = 24576;
        break;

      case 3:
        par.reps = 4;
        par.Ncp = 3168;
        par.dftlen = 6144;
        break;

      default:
        AssertFatal(1 == 0, "Illegal prach format %d for length 839\n", p->pdu.prach_format);
        break;
    }
  } else {
    switch (p->pdu.prach_format) {
      case 4: // A1
        par.reps = 2;
        par.Ncp = 288 >> mu;
        break;

      case 5: // A2
        par.reps = 4;
        par.Ncp = 576 >> mu;
        break;

      case 6: // A3
        par.reps = 6;
        par.Ncp = 864 >> mu;
        break;

      case 7: // B1
        par.reps = 2;
        par.Ncp = 216 >> mu;
        break;

      case 8: // B4
        par.reps = 12;
        par.Ncp = 936 >> mu;
        break;

      case 9: // C0
        par.reps = 1;
        par.Ncp = 1240 >> mu;
        break;

      case 10: // C2
        par.reps = 4;
        par.Ncp = 2048 >> mu;
        break;

      default:
        AssertFatal(1 == 0, "unknown prach format %x\n", p->pdu.prach_format);
        break;
    }
    par.dftlen = 2048 >> mu;
  }

  if (p->numerology_index == 0) {
    if (prachStartSymbol == 0 || prachStartSymbol == 7)
      par.Ncp += 16;
  } else {
    if (p->slot % (fp->slots_per_subframe / 2) == 0 && prachStartSymbol == 0)
      par.Ncp += 16;
  }

  switch(fp->samples_per_subframe) {
  case 7680:
    // 5 MHz @ 7.68 Ms/s
    par.Ncp >>= 2;
    par.dftlen >>= 2;
    break;

  case 15360:
    // 10, 15 MHz @ 15.36 Ms/s
    par.Ncp >>= 1;
    par.dftlen >>= 1;
    break;

  case 23040:
    // 20 MHz @ 23.04 Ms/s
    par.Ncp = (par.Ncp * 3) / 4;
    par.dftlen = (par.dftlen * 3) / 4;
    break;

  case 30720:
    // 20, 25, 30 MHz @ 30.72 Ms/s
    break;

  case 46080:
    // 40 MHz @ 46.08 Ms/s
    par.Ncp = (par.Ncp*3)/2;
    par.dftlen = (par.dftlen*3)/2;
    break;

  case 61440:
    // 40, 50, 60 MHz @ 61.44 Ms/s
    par.Ncp <<= 1;
    par.dftlen <<= 1;
    break;

  case 92160:
    // 50, 60, 70, 80, 90 MHz @ 92.16 Ms/s
    par.Ncp *= 3;
    par.dftlen *= 3;
    break;

  case 122880:
    // 70, 80, 90, 100 MHz @ 122.88 Ms/s
    par.Ncp <<= 2;
    par.dftlen <<= 2;
    break;

  case 184320:
    // 100 MHz @ 184.32 Ms/s
    par.Ncp = par.Ncp*6;
    par.dftlen = par.dftlen*6;
    break;

  case 245760:
    // 200 MHz @ 245.76 Ms/s
    par.Ncp <<= 3;
    par.dftlen <<= 3;
    break;

  default:
    AssertFatal(1==0,"sample rate %f MHz not supported for numerology %d\n", fp->samples_per_subframe / 1000.0, mu);
  }

  par.dftsize = get_dft(par.dftlen);
  par.N_ZC = (p->prach_sequence_length == 0) ? 839 : 139;

  const unsigned int K = get_prach_K(p->prach_sequence_length, p->pdu.prach_format, p->numerology_index, p->mu);
  const uint8_t kbar = get_PRACH_k_bar(p->mu, p->numerology_index);

  int n_ra_prb = p->msg1_frequencystart;
  int k                   = (12*n_ra_prb) - 6*fp->N_RB_UL;

  if (k<0) k+=(fp->ofdm_symbol_size);
  
  k*=K;
  k+=kbar;
  par.k = k;

  return par;
}

void init_nr_prach(PHY_VARS_gNB *gNB)
{
  int num_prach = 16;
  bool ret;
  ret = spsc_q_alloc(&gNB->prach_ru_queue, num_prach, sizeof(prach_item_t));
  DevAssert(ret);
  ret = spsc_q_alloc(&gNB->prach_l1rx_queue, num_prach, sizeof(prach_item_t));
  DevAssert(ret);
}

void reset_nr_prach(PHY_VARS_gNB *gNB)
{
  prach_item_t pending;
  // Teardown runs after the producers stop; queued entries still own buffers.
  while (spsc_q_get(&gNB->prach_ru_queue, &pending, sizeof(pending)))
    free_nr_prach_entry(&pending);
  while (spsc_q_get(&gNB->prach_l1rx_queue, &pending, sizeof(pending)))
    free_nr_prach_entry(&pending);
  spsc_q_free(&gNB->prach_ru_queue);
  spsc_q_free(&gNB->prach_l1rx_queue);
}

void free_nr_prach_entry(prach_item_t *p)
{
  free(p->prach_buf);
}

static bool drop_old_prach(const void *data, void *user)
{
  const prach_item_t *p = data;
  const fsn_t *now = user;
  // account for long PRACH over more than 1 slot
  const fsn_t t = {p->frame, p->slot + p->num_slots - 1, now->mu};
  bool drop = fsn_in_the_past(t, *now);
  if (drop)
    LOG_E(NR_PHY, "%4d.%2d PRACH job is in the past (%4d.%2d)\n", now->f, now->s, t.f, t.s);
  return drop;
}

static bool get_current_prach(const void *data, void *user)
{
  const prach_item_t *p = data;
  const fsn_t *now = user;
  // account for long PRACH over more than 1 slot
  const fsn_t t = {p->frame, p->slot + p->num_slots - 1, now->mu};
  return fsn_equal(t, *now);
}

bool get_next_nr_prach(spsc_q_t *q, const fsn_t *now, prach_item_t *p)
{
  prach_item_t stale;
  while (spsc_q_get_if(q, drop_old_prach, (void *)now, &stale, sizeof(stale)))
    free_nr_prach_entry(&stale);
  return spsc_q_get_if(q, get_current_prach, (void *)now, p, sizeof(*p));
}

void nr_schedule_rx_prach(PHY_VARS_gNB *gNB, int SFN, int Slot, nfapi_nr_prach_pdu_t *prach_pdu)
{
  const int fmt = prach_pdu->prach_format;
  const NR_DL_FRAME_PARMS *fp = &gNB->frame_parms;
  const nfapi_nr_prach_config_t *cfg = &gNB->gNB_config.prach_config;
  const nfapi_nr_num_prach_fd_occasions_t *occ = &cfg->num_prach_fd_occasions_list[prach_pdu->num_ra];
  const int num_rx_per_beam = gNB->frame_parms.nb_antennas_rx / gNB->common_vars.num_beams_period;
  prach_item_t prach = {
      .frame = SFN,
      .slot = Slot,
      .num_slots = fmt < 4 ? get_long_prach_dur(fmt, fp->numerology_index) : 1,
      .pdu = *prach_pdu,
      .rootSequenceIndex = occ->prach_root_sequence_index.value,
      .numrootSequenceIndex = occ->num_root_sequences.value,
      .msg1_frequencystart = occ->k1.value,
      .mu = cfg->prach_sub_c_spacing.value,
      .prach_sequence_length = cfg->prach_sequence_length.value,
      .restricted_set = cfg->restricted_set_config.value,
      .numerology_index = fp->numerology_index,
      .nb_rx = num_rx_per_beam,
      .Xu = gNB->X_u,
      .rx_prach = &gNB->rx_prach,
      .normalized_detection = gNB->prach_normalized_detection,
  };
  // Keep optional diagnostic spectra in the existing job's allocation/lifetime.
  const size_t buffer_bytes = sizeof(c16_t) * prach.nb_rx * NUMBER_OF_NR_RU_PRACH_OCCASIONS_MAX * NR_PRACH_SEQ_LEN_L;
  const bool capture_repetitions = flight_recorder_enabled() && prach.nb_rx == 1 && prach.prach_sequence_length == 0 && fmt < 4
                                   && prach_pdu->num_prach_ocas > 0
                                   && prach_pdu->num_prach_ocas <= NUMBER_OF_NR_RU_PRACH_OCCASIONS_MAX;
  const size_t repetition_bytes =
      capture_repetitions ? sizeof(c16_t) * prach_pdu->num_prach_ocas * NR_PRACH_MAX_LONG_REPETITIONS * NR_PRACH_SEQ_LEN_L : 0;
  prach.prach_buf = calloc_or_fail(1, buffer_bytes + repetition_bytes);
  prach.prach_repetitions = capture_repetitions ? (void *)((char *)prach.prach_buf + buffer_bytes) : NULL;
  const int num_beams = prach_pdu->beamforming.dig_bf_interface;
  int n_symb = get_nr_prach_duration(prach_pdu->prach_format);
  AssertFatal(num_beams < NFAPI_MAX_NUM_BG_IF, "impossible beams size %d\n", num_beams);
  for (int i = 0; i < num_beams; i++) {
    int fapi_beam_idx = prach_pdu->beamforming.prgs_list[0].dig_bf_interface_list[i].beam_idx;
    int start_symb = prach_pdu->prach_start_symbol + i * n_symb;
    int bitmap = SL_to_bitmap(start_symb, n_symb);
    if (gNB->common_vars.beam_id) {
      // TODO: Remove assumption of contiguous ports after DAS is properly handled in beamforming
      uint16_t ant_start = get_first_ant_idx(gNB->enable_analog_das,
                                             num_rx_per_beam,
                                             fapi_beam_idx,
                                             prach_pdu->param_v4.spatialStreamIndices[i * num_rx_per_beam]);
      beam_index_allocation(fapi_beam_idx,
                            ant_start,
                            num_rx_per_beam,
                            NR_SYMBOLS_PER_SLOT,
                            Slot,
                            bitmap,
                            gNB->frame_parms.nb_antennas_rx,
                            gNB->common_vars.beam_id);
      prach.ant_start = ant_start;
    }
  }
  bool found = spsc_q_put(&gNB->prach_ru_queue, &prach, sizeof(prach));
  if (!found) {
    LOG_W(NR_PHY, "%4d.%2d PRACH occ queue is full: dropping PRACH request\n", SFN, Slot);
    free_nr_prach_entry(&prach);
  }
}

static void rx_nr_prach_ru_internal_rep(prach_item_t *p,
                                        int ant_offset,
                                        int32_t **rxdata,
                                        NR_DL_FRAME_PARMS *fp,
                                        int N_TA_offset,
                                        int rep,
                                        const prach_ru_params_t *params,
                                        c16_t (*rxsigF)[NR_PRACH_SEQ_LEN_L],
                                        nr_prach_capture_layout_t *capture,
                                        c16_t *repetition)
{
  AssertFatal(rep >= 0 && rep < params->reps, "rep %d is out of range (reps = %d)\n", rep, params->reps);

  int slot2 = p->prach_sequence_length ? p->slot : p->slot;
  int sample_offset =
      get_samples_slot_timestamp(fp, slot2) + params->sample_offset_slot - N_TA_offset + params->Ncp + rep * params->dftlen;
  if (capture && rep == 0) {
    *capture = (nr_prach_capture_layout_t){.buffer_first = sample_offset,
                                           .ncp = params->Ncp,
                                           .dftlen = params->dftlen,
                                           .reps = params->reps,
                                           .n_ta_offset = N_TA_offset,
                                           .k = params->k,
                                           .first_antenna = ant_offset,
                                           .antenna_count = p->nb_rx};
  }

  for (int aa = 0; aa < p->nb_rx; aa++) {
    int idx = ant_offset + aa;
    c16_t *prach2 = (c16_t *)&rxdata[idx][sample_offset];

    // do DFT for the specific repetition
    c16_t tmp[params->dftlen] __attribute__((aligned(32)));
    dft(params->dftsize, (int16_t *)prach2, (int16_t *)tmp, 1);
    // Coherent combining of PRACH repetitions (assumes channel does not change, to be revisted for "long" PRACH)
    LOG_D(PHY, "Doing PRACH combining of repetition %d/%d N_ZC %d\n", rep, params->reps, params->N_ZC);
    int k2 = params->k;
    for (int j = 0; j < params->N_ZC; j++, k2++) {
      if (k2 == params->dftlen)
        k2 = 0;
      if (repetition) {
        repetition[j] = tmp[k2];
        const int sum_r = rxsigF[aa][j].r + tmp[k2].r;
        const int sum_i = rxsigF[aa][j].i + tmp[k2].i;
        if (capture)
          capture->coherent_wraps += sum_r < INT16_MIN || sum_r > INT16_MAX || sum_i < INT16_MIN || sum_i > INT16_MAX;
      }
      rxsigF[aa][j] = c16add(rxsigF[aa][j], tmp[k2]);
    }
  }
  if (repetition && capture)
    capture->repetition_mask |= 1U << rep;
}

static void rx_nr_prach_ru_internal(prach_item_t *p,
                                    int prachStartSymbol,
                                    int prachOccasion,
                                    int32_t **rxdata,
                                    NR_DL_FRAME_PARMS *fp,
                                    int N_TA_offset,
                                    bool das)
{
  prach_ru_params_t params = get_prach_ru_params(p, prachStartSymbol, fp);
  c16_t rxsigF_tmp[p->nb_rx][NR_PRACH_SEQ_LEN_L];
  memset(rxsigF_tmp, 0, sizeof(rxsigF_tmp));

  const uint8_t num_beams = p->pdu.beamforming.dig_bf_interface;
  // When more than one beams, then each occasion is on one beam
  int ant_offset = 0;
  if (num_beams > 1) {
    AssertFatal(prachOccasion < num_beams, "Num of PRACH Occasions must be same as number of beams in beamforming mode\n");
    ant_offset = prachOccasion * p->nb_rx;
  }

  // TODO: Remove assumption of contiguous ports after DAS is properly handled in beamforming
  uint16_t ant_start =
      get_first_ant_idx(das,
                        p->nb_rx,
                        p->pdu.beamforming.prgs_list[0].dig_bf_interface_list[0].beam_idx,
                        p->pdu.param_v4.numSpatialStreamIndices > 0 ? p->pdu.param_v4.spatialStreamIndices[ant_offset] : 0);

  nr_prach_capture_layout_t *capture =
      flight_recorder_enabled() && prachOccasion < NUMBER_OF_NR_RU_PRACH_OCCASIONS_MAX ? &p->capture[prachOccasion] : NULL;
  for (int rep = 0; rep < params.reps; rep++) {
    c16_t *snapshot = p->prach_repetitions && rep < NR_PRACH_MAX_LONG_REPETITIONS ? p->prach_repetitions[prachOccasion][rep] : NULL;
    rx_nr_prach_ru_internal_rep(p, ant_start, rxdata, fp, N_TA_offset, rep, &params, rxsigF_tmp, capture, snapshot);
  }
  if (capture)
    capture->valid = true;

  for (int aa = 0; aa < p->nb_rx; aa++) {
    memcpy(p->prach_buf[aa][prachOccasion], rxsigF_tmp[aa], sizeof(c16_t) * params.N_ZC);
  }
}

void rx_nr_prach_ru(prach_item_t *p, int32_t **rxdata, NR_DL_FRAME_PARMS *fp, int N_TA_offset, bool das)
{
  int N_dur = get_nr_prach_duration(p->pdu.prach_format);
  LOG_D(NR_PHY_RACH, "%d.%d try to decode %d occasions \n", p->frame, p->slot, p->pdu.num_prach_ocas);
  for (int prach_oc = 0; prach_oc < p->pdu.num_prach_ocas; prach_oc++) {
    int prachStartSymbol = p->pdu.prach_start_symbol + prach_oc * N_dur;
    // comment FK: the standard 38.211 section 5.3.2 has one extra term +14*N_RA_slot. This is because there prachStartSymbol is
    // given wrt to start of the 15kHz slot or 60kHz slot. Here we work slot based, so this function is anyway only called in slots
    // where there is PRACH. Its up to the MAC to schedule another PRACH PDU in the case there are there N_RA_slot \in {0,1}.
    rx_nr_prach_ru_internal(p, prachStartSymbol, prach_oc, rxdata, fp, N_TA_offset, das);
  }
}

void rx_nr_prach_ru_rep(prach_item_t *p,
                        int32_t **rxdata,
                        NR_DL_FRAME_PARMS *fp,
                        int N_TA_offset,
                        int rep,
                        int prachOccasion,
                        c16_t (*rxsigF)[NR_PRACH_SEQ_LEN_L])
{
  int N_dur = get_nr_prach_duration(p->pdu.prach_format);
  int prachStartSymbol = p->pdu.prach_start_symbol + prachOccasion * N_dur;
  prach_ru_params_t params = get_prach_ru_params(p, prachStartSymbol, fp);
  rx_nr_prach_ru_internal_rep(p, 0, rxdata, fp, N_TA_offset, rep, &params, rxsigF, NULL, NULL);
}

bool nr_prach_normalized_detected(const rx_prach_out_t *out, int threshold_tenth_db)
{
  if (!out->correlation_total || !out->correlation_bins || !out->correlation_antennas || threshold_tenth_db < 0)
    return false;
  // Preserve the receiver's first representable normalized power unit. Below
  // it, sparse quantization residue can have a large ratio without a usable peak.
  if (out->correlation_peak < (uint64_t)out->correlation_bins * out->correlation_antennas)
    return false;
  // Compare peak/mean without truncating the mean or multiplying integers.
  const double threshold = pow(10.0, threshold_tenth_db / 100.0);
  return (double)out->correlation_peak * out->correlation_bins > (double)out->correlation_total * threshold;
}

static void prach_correlate(const c16_t *root,
                            const c16_t *received,
                            int sequence_length,
                            int bins,
                            bool rounded,
                            c16_t *frequency,
                            c16_t *correlation)
{
  for (int i = 0; i < sequence_length; i++) {
    if (rounded) {
      // Arithmetic truncation biases low-level complex noise toward a DC peak.
      const int64_t real = (int64_t)root[i].r * received[i].r + (int64_t)root[i].i * received[i].i;
      const int64_t imag = (int64_t)root[i].r * received[i].i - (int64_t)root[i].i * received[i].r;
      const int64_t real_q = real >= 0 ? (real + 16384) / 32768 : -((-real + 16384) / 32768);
      const int64_t imag_q = imag >= 0 ? (imag + 16384) / 32768 : -((-imag + 16384) / 32768);
      frequency[i].r = max(INT16_MIN, min(INT16_MAX, real_q));
      frequency[i].i = max(INT16_MIN, min(INT16_MAX, imag_q));
    } else {
      frequency[i] = c16MulConjShift(root[i], received[i], 15);
    }
  }
  memset(frequency + sequence_length, 0, sizeof(*frequency) * (bins - sequence_length));
  idft(get_idft(bins), (int16_t *)frequency, (int16_t *)correlation, 1);
}

static void prach_repetition_evidence(const prach_item_t *in, int occasion, const rx_prach_out_t *out)
{
  if (!flight_recorder_enabled())
    return;
  const nr_prach_capture_layout_t *capture = &in->capture[occasion];
  const int64_t token = capture->token > 0 ? capture->token : INT64_MIN;
  const bool complete = in->prach_repetitions && in->nb_rx == 1 && capture->valid && capture->reps > 0
                        && capture->reps <= NR_PRACH_MAX_LONG_REPETITIONS && capture->repetition_mask == (1U << capture->reps) - 1;
  const int flags = (in->prach_repetitions != NULL) | (capture->valid << 1) | (complete << 2);
  flight_recorder_emit(FLIGHT_EVENT_GNB_PRACH_COMBINING,
                       token,
                       flags | ((int64_t)capture->repetition_mask << 16),
                       capture->valid ? capture->reps : INT64_MIN,
                       complete ? capture->coherent_wraps : INT64_MIN,
                       in->pdu.prach_format,
                       in->nb_rx);
  if (!complete)
    return;
  const int bins = out->correlation_bins;
  const int nzc = in->prach_sequence_length == 0 ? 839 : 139;
  c16_t frequency[bins] __attribute__((aligned(32)));
  c16_t correlation[bins] __attribute__((aligned(32)));
  uint64_t separate_total = 0;
  for (int rep = 0; rep < capture->reps; rep++) {
    uint64_t spectral_total = 0, spectral_peak = 0;
    const c16_t *received = in->prach_repetitions[occasion][rep];
    for (int i = 0; i < nzc; i++) {
      const int64_t real = received[i].r, imag = received[i].i;
      const uint64_t energy = real * real + imag * imag;
      spectral_total += energy;
      spectral_peak = max(spectral_peak, energy);
    }
    separate_total += spectral_total;
    flight_recorder_emit(FLIGHT_EVENT_GNB_PRACH_REPETITION_SPECTRUM, token, rep, spectral_total, spectral_peak, nzc, 1);
    // Spectral evidence remains available even if coherent cancellation leaves no candidate.
    if (!out->correlation_peak)
      continue;
    prach_correlate(in->Xu[out->correlation_root],
                    in->prach_repetitions[occasion][rep],
                    nzc,
                    bins,
                    in->normalized_detection,
                    frequency,
                    correlation);
    uint64_t peak = 0, total = 0;
    int peak_bin = 0;
    for (int i = 0; i < bins; i++) {
      const int64_t real = correlation[i].r, imag = correlation[i].i;
      const uint64_t energy = real * real + imag * imag;
      total += energy;
      if (energy > peak) {
        peak = energy;
        peak_bin = i;
      }
    }
    const c16_t candidate = correlation[out->correlation_bin];
    const uint32_t packed = (uint16_t)candidate.r | ((uint32_t)(uint16_t)candidate.i << 16);
    flight_recorder_emit(FLIGHT_EVENT_GNB_PRACH_REPETITION, token, rep, peak_bin, peak, total, packed);
  }
  uint64_t combined_total = 0;
  for (int i = 0; i < nzc; i++) {
    const int64_t real = in->prach_buf[0][occasion][i].r, imag = in->prach_buf[0][occasion][i].i;
    combined_total += real * real + imag * imag;
  }
  flight_recorder_emit(FLIGHT_EVENT_GNB_PRACH_COMBINED_SPECTRUM,
                       token,
                       combined_total,
                       separate_total,
                       capture->reps,
                       capture->coherent_wraps,
                       nzc);
}

rx_prach_out_t rx_nr_prach(const prach_item_t *in, int occasion)
{
  rx_prach_out_t out = {};
  uint16_t preamble_index0 = 0;
  uint16_t numshift = 0;
  int first_nonzero_root_idx = 0;
  bool new_dft = false;
  int log2_ifft_size = in->prach_sequence_length == 0 ? 10 : 8;

  const int nb_rx = in->nb_rx;
  const int NCS = in->pdu.num_cs;
  const int prach_fmt = in->pdu.prach_format;
  const int N_ZC = in->prach_sequence_length == 0 ? 839 : 139;
  const int dft_sz = N_ZC == 839 ? 1024 : 256;

  LOG_D(NR_PHY_RACH,
        "L1 PRACH RX: rooSequenceIndex %d, numRootSeqeuences %d, NCS %d, N_ZC %d, format %d \n",
        in->rootSequenceIndex,
        in->numrootSequenceIndex,
        NCS,
        N_ZC,
        prach_fmt);

  if (LOG_DEBUGFLAG(DEBUG_PRACH)) {
    if ((in->frame & 1023) < 20)
      LOG_D(PHY, "PRACH (gNB) : running rx_prach for slot %d, rootSequenceIndex %d\n", in->slot, in->rootSequenceIndex);
  }

  start_meas(in->rx_prach);

  const uint16_t *prach_root_sequence_map =
      in->prach_sequence_length == 0 ? prach_root_sequence_map_0_3 : prach_root_sequence_map_abc;

  // PDP is oversampled, e.g. 1024 sample instead of 839
  // Adapt the NCS (zero-correlation zones) with oversampling factor e.g. 1024/839
  int NCS2 = N_ZC == 839 ? (NCS << 10) / 839 : (NCS << 8) / 139;

  // An unrestricted NCS=0 preamble uses its own root and the full correlation window.
  if (NCS2 == 0)
    NCS2 = NCS == 0 && in->restricted_set == 0 ? dft_sz : N_ZC;

  int preamble_offset = 0, preamble_offset_old = 99;

  int16_t preamble_shift = 0;
  const bool collect_linear = in->normalized_detection || flight_recorder_enabled();
  uint64_t prach_ifft[dft_sz] __attribute__((aligned(32)));
  uint64_t correlation_total = 0;
  out.correlation_bins = dft_sz;
  out.correlation_antennas = nb_rx;
  for (int preamble_index = 0; preamble_index < 64; preamble_index++) {
    if (LOG_DEBUGFLAG(DEBUG_PRACH)) {
      int en = dB_fixed(signal_energy((int32_t *)in->prach_buf[0][occasion], N_ZC == 839 ? 840 : 140));
      if (en > 60)
        LOG_D(PHY, "frame %d, slot %d : Trying preamble %d \n", in->frame, in->slot, preamble_index);
    }
    if (in->restricted_set == 0) {
      // This is the relative offset in the root sequence table (5.7.2-4 from 36.211) for the given preamble index
      preamble_offset = ((NCS==0)? preamble_index : (preamble_index/(N_ZC/NCS)));

      if (preamble_offset != preamble_offset_old) {
        preamble_offset_old = preamble_offset;
        new_dft = true;
        // This is the \nu corresponding to the preamble index
        preamble_shift  = 0;
      } else {
        preamble_shift -= NCS;

        if (preamble_shift < 0)
          preamble_shift += N_ZC;
      }
    } else { // This is the high-speed case
      new_dft = false;
      uint16_t nr_du[NR_PRACH_SEQ_LEN_L - 1];
      nr_fill_du(N_ZC, prach_root_sequence_map, nr_du);
      // set preamble_offset to initial rootSequenceIndex and look if we need more root sequences for this
      // preamble index and find the corresponding cyclic shift
      // Check if all shifts for that root have been processed
      int n_shift_ra = 0, n_shift_ra_bar, d_start = 0;
      if (preamble_index0 == numshift) {
        bool not_found = true;
        new_dft = true;
        preamble_index0 -= numshift;
        while (not_found) {
          // current root depending on rootSequenceIndex
          int index = (in->rootSequenceIndex + preamble_offset) % N_ZC;
          int u = nr_du[prach_root_sequence_map[index]];
          uint16_t n_group_ra = 0;

          if (u < (N_ZC / 3) && u >= NCS) {
            n_shift_ra = u / NCS;
            d_start = (u << 1) + (n_shift_ra * NCS);
            n_group_ra = N_ZC / d_start;
            n_shift_ra_bar = max(0, (N_ZC - (u << 1) - (n_group_ra * d_start)) / N_ZC);
          } else if (u >= (N_ZC / 3) && u <= ((N_ZC - NCS) >> 1)) {
            n_shift_ra = (N_ZC - (u << 1)) / NCS;
            d_start = N_ZC - (u << 1) + (n_shift_ra * NCS);
            n_group_ra = u / d_start;
            n_shift_ra_bar = min(n_shift_ra, max(0, (u - (n_group_ra * d_start)) / NCS));
          } else {
            n_shift_ra = 0;
            n_shift_ra_bar = 0;
          }

          // This is the number of cyclic shifts for the current root u
          numshift = (n_shift_ra * n_group_ra) + n_shift_ra_bar;
          // skip to next root and recompute parameters if numshift==0
          numshift > 0 ? not_found = false : preamble_offset++;
        }
      }

      if (n_shift_ra>0)
        preamble_shift = -(d_start * (preamble_index0 / n_shift_ra)
                           + (preamble_index0 % n_shift_ra) * NCS); // minus because the channel is h(t -\tau + Cv)
      else
        preamble_shift = 0;

      if (preamble_shift < 0)
        preamble_shift+=N_ZC;

      preamble_index0++;

      if (preamble_index == 0)
        first_nonzero_root_idx = preamble_offset;
    }

    // Compute DFT of RX signal (conjugate in->rxsigF[occasion], results in conjugate output) for each new rootSequenceIndex
    if (LOG_DEBUGFLAG(DEBUG_PRACH)) {
      int en = dB_fixed(signal_energy((int32_t *)in->prach_buf[0][occasion], 840));
      if (en>60)
        LOG_D(PHY,
              "frame %d, slot %d : preamble index %d, NCS %d, N_ZC/NCS %d: offset %d, preamble shift %d , en %d)\n",
              in->frame,
              in->slot,
              preamble_index,
              NCS,
              N_ZC / NCS,
              preamble_offset,
              preamble_shift,
              en);
    }

    LOG_D(NR_PHY_RACH,
          "PRACH RX preamble_index %d, preamble_offset %d, preamb shift %d new dft %d\n",
          preamble_index,
          preamble_offset,
          preamble_shift,
          new_dft);

    if (new_dft) {
      new_dft = false;

      c16_t *Xu = in->Xu[preamble_offset - first_nonzero_root_idx];
      LOG_D(PHY,"PRACH RX new dft preamble_offset-first_nonzero_root_idx %d\n",preamble_offset-first_nonzero_root_idx);

      memset(prach_ifft, 0, sizeof(prach_ifft));
      if (LOG_DUMPFLAG(DEBUG_PRACH)) {
        LOG_M("prach_rxF0.m", "prach_rxF0", in->prach_buf[0][occasion], N_ZC, 1, 1);
        LOG_M("prach_rxF1.m", "prach_rxF1", in->prach_buf[1][occasion], 6144, 1, 1);
      }
      c16_t prachF[dft_sz] __attribute__((aligned(32)));
      for (int aa = 0; aa < nb_rx; aa++) {
        c16_t prach_ifft_tmp[dft_sz] __attribute__((aligned(32)));
        prach_correlate(Xu, in->prach_buf[aa][occasion], N_ZC, dft_sz, in->normalized_detection, prachF, prach_ifft_tmp);
        // compute energy and accumulate over receive antennas
        for (int i = 0; i < dft_sz; i++) {
          const int64_t real = prach_ifft_tmp[i].r;
          const int64_t imag = prach_ifft_tmp[i].i;
          prach_ifft[i] += real * real + imag * imag;
        }

        if (LOG_DUMPFLAG(DEBUG_PRACH)) {
          if (aa == 0)
            LOG_M("prach_rxF_comp0.m","prach_rxF_comp0", prachF, 1024, 1, 1);
          if (aa == 1)
            LOG_M("prach_rxF_comp1.m","prach_rxF_comp1", prachF, 1024, 1, 1);
        }

      } // antennas_rx

      // Keep the complete linear energy for this root, before legacy truncation.
      if (collect_linear) {
        correlation_total = 0;
        for (int i = 0; i < dft_sz; i++)
          correlation_total += prach_ifft[i];
      }

    } // new dft

    // check energy in nth time shift, for

    int preamble_shift2 = preamble_shift == 0 ? 0 : (preamble_shift << log2_ifft_size) / N_ZC;

    for (int i = 0; i < NCS2; i++) {
      const int peak_bin = preamble_shift2 + i;
      const uint64_t linear = prach_ifft[peak_bin];
      const uint32_t normalized = (linear >> log2_ifft_size) / nb_rx;
      const int levdB = normalized ? dB_fixed_times10(normalized) : -900;
      if (collect_linear && (linear > out.correlation_peak || (linear == out.correlation_peak && out.correlation_delay_raw > i))) {
        out.correlation_peak = linear;
        out.correlation_total = correlation_total;
        out.correlation_bin = peak_bin;
        out.correlation_root = preamble_offset - first_nonzero_root_idx;
        out.correlation_preamble = preamble_index;
        out.correlation_delay_raw = i;
      }
      if (levdB > out.max_preamble_energy || (levdB == out.max_preamble_energy && out.max_preamble_delay_raw > i)) {
        LOG_D(NR_PHY_RACH, "preamble_index %d, delay %d en %d dB > %d dB\n", preamble_index, i, levdB, out.max_preamble_energy);
        out.max_preamble_energy = levdB;
        out.max_preamble_delay_raw = i;
        out.max_preamble = preamble_index;
      }
    }
  } // preamble_index

  if (in->normalized_detection && out.correlation_peak) {
    out.max_preamble = out.correlation_preamble;
    out.max_preamble_delay_raw = out.correlation_delay_raw;
    const uint32_t normalized = (out.correlation_peak >> log2_ifft_size) / nb_rx;
    out.max_preamble_energy = normalized ? max(0, dB_fixed_times10(normalized)) : 0;
  }

  // The conversion from raw PRACH delay to TA value is done here.
  // It is normalized to the 30.72 Ms/s, considering the numerology, N_RB and the sampling rate
  // See table 6.3.3.1 -1 and -2 in 38211.

  // Format 0, 1, 2: 24576 samples @ 30.72 Ms/s, 98304 samples @ 122.88 Ms/s
  // By solving:
  // max_preamble_delay * ( (24576*(fs/30.72M)) / 1024 ) / fs = TA * 16 * 64 / 2^mu * Tc

  // Format 3: 6144 samples @ 30.72 Ms/s, 24576 samples @ 122.88 Ms/s
  // By solving:
  // max_preamble_delay * ( (6144*(fs/30.72M)) / 1024 ) / fs = TA * 16 * 64 / 2^mu * Tc

  // Format >3: 2048/2^mu samples @ 30.72 Ms/s, 2048/2^mu * 4 samples @ 122.88 Ms/s
  // By solving:
  // max_preamble_delay * ( (2048/2^mu*(fs/30.72M)) / 256 ) / fs = TA * 16 * 64 / 2^mu * Tc
  const uint32_t raw_delay = out.max_preamble_delay_raw;
  const int mu = in->numerology_index;
  if (in->prach_sequence_length == 0) {
    const uint32_t mu_scale = 1U << mu;

    if (prach_fmt == 0 || prach_fmt == 1 || prach_fmt == 2)
      out.max_preamble_delay = (raw_delay * 3U * mu_scale + 1U) / 2U;
    else if (prach_fmt == 3)
      out.max_preamble_delay = (raw_delay * 3U * mu_scale + 4U) / 8U;
  } else {
    out.max_preamble_delay = (raw_delay + 1U) / 2U;
  }

  stop_meas(in->rx_prach);
  prach_repetition_evidence(in, occasion, &out);
  return out;
}
