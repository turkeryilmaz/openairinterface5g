/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "radio/COMMON/radio_gain_device.h"
#include "common/utils/LOG/flight_recorder.h"
#include "common/utils/LOG/flight_monitor.h"
#include "PHY/defs_nr_common.h"
#include "PHY/impl_defs_nr.h"
#define _GNU_SOURCE // For pthread_setname_np
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include "executables/nr-ue-ru.h"
#include "executables/nr-ue-tx-deadline.h"
#include "executables/nr-uesoftmodem.h"
#include "PHY/INIT/nr_phy_init.h"
#include "LAYER2/NR_MAC_UE/mac_proto.h"
#include "RRC/NR_UE/rrc_proto.h"
#include "RRC/NR_UE/L2_interface_ue.h"
#include "SCHED_NR_UE/defs.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "executables/softmodem-common.h"
#include "radio/COMMON/common_lib.h"
#include "LAYER2/nr_pdcp/nr_pdcp_oai_api.h"
#include "LAYER2/nr_rlc/nr_rlc_oai_api.h"
#include "openair1/PHY/TOOLS/phy_scope_interface.h"
#include "instrumentation.h"
#include "common/utils/threadPool/notified_fifo.h"
#include "actor.h"
#include "position_interface.h"
#include "nr_phy_common.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "common/utils/time_manager/time_manager.h"
#include "log.h"

/*
 *  NR SLOT PROCESSING SEQUENCE
 *
 *  Processing occurs with following steps for connected mode:
 *
 *  - Rx samples for a slot are received,
 *  - PDCCH processing (including DCI extraction for downlink and uplink),
 *  - PDSCH processing (including transport blocks decoding),
 *  - PUCCH/PUSCH (transmission of acknowledgements, CSI, ... or data).
 *
 *  Time between reception of the slot and related transmission depends on UE processing performance.
 *  It is defined by the value command line parameter ue-capability-rx2tx.
 *
 *  In NR, network gives the duration between Rx slot and Tx slot in the DCI:
 *  - for reception of a PDSCH and its associated acknowledgment slot (with a PUCCH or a PUSCH),
 *  - for reception of an uplink grant and its associated PUSCH slot.
 *
 *  So duration between reception and it associated transmission depends on its transmission slot given in the DCI.
 *  ue-capability-rx2tx means the minimum duration but higher duration can be given by the network because UE can support it.
 *
 *                                                                                                    Slot k
 *                                                                                  -------+------------+--------
 *                Frame                                                                    | Tx samples |
 *                Subframe                                                                 |   buffer   |
 *                Slot n                                                            -------+------------+--------
 *       ------ +------------+--------                                                     |
 *              | Rx samples |                                                             |
 *              |   buffer   |                                                             |
 *       -------+------------+--------                                                     |
 *                           |                                                             |
 *                           V                                                             |
 *                           +------------+                                                |
 *                           |   PDCCH    |                                                |
 *                           | processing |                                                |
 *                           +------------+                                                |
 *                           |            |                                                |
 *                           |            v                                                |
 *                           |            +------------+                                   |
 *                           |            |   PDSCH    |                                   |
 *                           |            | processing | decoding result                   |
 *                           |            +------------+    -> ACK/NACK of PDSCH           |
 *                           |                         |                                   |
 *                           |                         v                                   |
 *                           |                         +-------------+------------+        |
 *                           |                         | PUCCH/PUSCH | Tx samples |        |
 *                           |                         |  processing | transfer   |        |
 *                           |                         +-------------+------------+        |
 *                           |                                                             |
 *                           |/___________________________________________________________\|
 *                            \  duration between reception and associated transmission   /
 *
 * Remark: processing is done slot by slot, it can be distribute on different threads which are executed in parallel.
 * This is an architecture optimization in order to cope with real time constraints.
 * By example, for LTE, subframe processing is spread over 4 different threads.
 *
 */

static void start_process_slot_tx(void *arg)
{
  notifiedFIFO_elt_t *newTx = arg;
  // A late DL completion must not enqueue work into a stopped UL actor.
  if (oai_exit || nrue_ru_failed()) {
    delNotifiedFIFO_elt(newTx);
    return;
  }
  nr_rxtx_thread_data_t *curMsgTx = NotifiedFifoData(newTx);
  int num_ul_actors = get_nrUE_params()->num_ul_actors;
  if (num_ul_actors > 0) {
    pushNotifiedFIFO(&curMsgTx->ul_actors[curMsgTx->proc.nr_slot_tx % num_ul_actors].fifo, newTx);
  } else {
    newTx->processingFunc(curMsgTx);
  }
}

static size_t dump_L1_UE_meas_stats(PHY_VARS_NR_UE *ue, char *output, size_t max_len)
{
  const char *begin = output;
  const char *end = output + max_len;
  for (int i = 0; i < MAX_CPU_STAT_TYPE; i++) {
    output += print_meas_log(&ue->phy_cpu_stats.cpu_time_stats[i],
                             ue->phy_cpu_stats.cpu_time_stats[i].meas_name,
                             NULL,
                             NULL,
                             output,
                             end - output);
  }
  return output - begin;
}

static void *nrL1_UE_stats_thread(void *param)
{
  PHY_VARS_NR_UE *ue = (PHY_VARS_NR_UE *) param;
  const int max_len = 16384;
  char output[max_len];
  char filename[30];
  snprintf(filename, 29, "nrL1_UE_stats-%d.log", ue->Mod_id);
  filename[29] = 0;
  FILE *fd = fopen(filename, "w");
  AssertFatal(fd != NULL, "Cannot open %s\n", filename);

  while (!oai_exit) {
    sleep(1);
    const int len = dump_L1_UE_meas_stats(ue, output, max_len);
    AssertFatal(len < max_len, "exceeded length\n");
    fwrite(output, len + 1, 1, fd); // + 1 for terminating NULL byte
    fflush(fd);
    fseek(fd, 0, SEEK_SET);
  }
  fclose(fd);

  return NULL;
}

static int determine_N_TA_offset(PHY_VARS_NR_UE *ue) {
  if (ue->sl_mode == 2)
    return 0;
  else {
    int N_TA_offset = ue->nrUE_config.cell_config.N_TA_offset;
    if (N_TA_offset == -1) {
      return set_default_nta_offset(ue->frame_parms.freq_range, ue->frame_parms.samples_per_subframe);
    } else {
      // Return N_TA_offet in samples, as described in 38.211 4.1 and 4.3.1
      // T_c[s] =  1/(Δf_max x N_f) = 1 / (480 * 1000 * 4096)
      // N_TA_offset[s] = N_TA_offset x T_c
      // N_TA_offset[samples] = samples_per_second x N_TA_offset[s]
      // N_TA_offset[samples] = N_TA_offset x samples_per_subframe x 1000 x T_c
      return (N_TA_offset * ue->frame_parms.samples_per_subframe) / (4096 * 480);
    }
  }
}

void init_nr_ue_vars(PHY_VARS_NR_UE *ue, uint8_t UE_id)
{
  int nb_connected_gNB = 1;

  ue->Mod_id      = UE_id;
  ue->if_inst     = nr_ue_if_module_init(UE_id);
  ue->dci_thres   = 0;
  ue->target_Nid_cell = -1;

  // initialize all signal buffers
  init_nr_ue_signal(ue, nb_connected_gNB);

  // intialize transport
  init_nr_ue_transport(ue);

  // Initialization of measurement variables
  init_phy_nr_measurements(ue);

  ue->ta_frame = -1;
  ue->ta_slot = -1;
  ue->rar_ta_command.ta_frame = -1;
  ue->rar_ta_command.ta_slot = -1;
  ue->tx_stream_timing_adjustment = 0;
}

/*!
 * It performs band scanning and synchonization.
 * \param arg is a pointer to a \ref PHY_VARS_NR_UE structure.
 */

typedef struct {
  c16_t **input;
  int input_sz;
  PHY_VARS_NR_UE *UE;
  UE_nr_rxtx_proc_t proc;
  nr_gscn_info_t gscnInfo[MAX_GSCN_BAND];
  int numGscn;
  int rx_offset;
  int frame_number;
  uint64_t bcch_generation;
} syncData_t;

static void UE_synch(void *arg) {
  syncData_t *syncD = (syncData_t *)arg;
  PHY_VARS_NR_UE *UE = syncD->UE;
  radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_ACQUISITION);
  UE->is_synchronized = 0;
  UE->managed_serving_pbch_failed = false;

  if (UE->target_Nid_cell != -1) {
    LOG_W(NR_PHY, "Starting re-sync detection for target Nid_cell %i\n", UE->target_Nid_cell);
  } else {
    LOG_W(NR_PHY, "Starting sync detection\n");
  }

  LOG_I(PHY, "[UE thread Synch] Running Initial Synch \n");

  uint64_t dl_carrier, ul_carrier;
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  nr_initial_sync_t ret = {0};
  if (UE->sl_mode == 2) {
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;
    dl_carrier = fp->sl_CarrierFreq;
    ul_carrier = fp->sl_CarrierFreq;
    ret = sl_nr_slss_search(UE, &syncD->proc, SL_NR_SSB_REPETITION_IN_FRAMES, syncD->input_sz, syncD->input);
  } else {
    nr_get_carrier_frequencies(UE, &dl_carrier, &ul_carrier);
    ret = nr_initial_sync(&syncD->proc, UE, syncD->input_sz, syncD->input, syncD->gscnInfo, syncD->numGscn);
  }

  if (ret.cell_detected) {
    syncD->rx_offset = ret.rx_offset;
    syncD->frame_number = ret.frame_number;
    const int freq_offset = UE->common_vars.freq_offset; // frequency offset computed with pss in initial sync
    const int64_t doubled_offset = (int64_t)ret.rx_offset * 2;
    const int hw_slot_offset = (doubled_offset / fp->samples_per_subframe * fp->slots_per_subframe)
                               + round((double)(doubled_offset % fp->samples_per_subframe) / fp->samples_per_slot0);

    if (!get_nrUE_params()->cont_fo_comp) {
      // rerun with new cell parameters and frequency-offset
      if (nrue_ru_set_freq(UE, ul_carrier, dl_carrier, freq_offset) != 0) {
        LOG_W(PHY, "Synchronization retune failed or was deferred; synchronization remains pending\n");
        return;
      }
    }
    UE->freq_offset = freq_offset - UE->dl_Doppler_shift;

    if (get_nrUE_params()->agc) {
      nrue_ru_adjust_rx_gain(UE, UE->adjust_rxgain);
    }

    flight_recorder_emit(FLIGHT_EVENT_UE_SYNC, UE->Mod_id, 1, hw_slot_offset, freq_offset, ret.rx_offset, 0);
    LOG_I(PHY, "Got synch: hw_slot_offset %d, carrier off %d Hz\n", hw_slot_offset, freq_offset);

    UE->is_synchronized = 1;
    radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_TRACKING);
  } else {
    flight_recorder_emit(FLIGHT_EVENT_UE_SYNC, UE->Mod_id, 0, 0, 0, 0, 0);
    radio_gain_device_observe_rx(&syncD->proc.rx_gain_context, 0, false, true, RADIO_RX_SOURCE_UE_SEARCH);
    int gain_change = 0;
    if (get_nrUE_params()->agc)
      gain_change = nrue_ru_adjust_rx_gain(UE, INCREASE_IN_RXGAIN);
    if (gain_change)
      LOG_I(PHY, "synch retry: Rx gain increased \n");
    else
      LOG_E(PHY, "synch Failed: \n");
  }
}

static uint num_dl_symbols_slot(const fapi_nr_config_request_t *cfg, int nr_slot)
{
  if (cfg->cell_config.frame_duplex_type == FDD)
    return NR_SYMBOLS_PER_SLOT;

  const fapi_nr_tdd_table_t *tdd_table = &cfg->tdd_table;
  if (tdd_table->max_tdd_periodicity_list == NULL) {
    LOG_E(NR_PHY, "No TDD table present in PHY config\n");
    return 0;
  }

  int rel_slot = nr_slot % tdd_table->tdd_period_in_slots;

  const fapi_nr_max_tdd_periodicity_t *current_slot = &tdd_table->max_tdd_periodicity_list[rel_slot];
  return current_slot->num_dl_ul_symbols_list.num_dl;
}

static int nr_ue_slot_select(const fapi_nr_config_request_t *cfg, int nr_slot)
{
  if (cfg->cell_config.frame_duplex_type == FDD)
    return NR_UPLINK_SLOT | NR_DOWNLINK_SLOT;

  const fapi_nr_tdd_table_t *tdd_table = &cfg->tdd_table;
  int rel_slot = nr_slot % tdd_table->tdd_period_in_slots;

  if (tdd_table->max_tdd_periodicity_list == NULL) // this happens before receiving TDD configuration
    return NR_DOWNLINK_SLOT;

  const fapi_nr_max_tdd_periodicity_t *current_slot = &tdd_table->max_tdd_periodicity_list[rel_slot];

  if (current_slot->num_dl_ul_symbols_list.num_dl == NR_SYMBOLS_PER_SLOT)
    return NR_DOWNLINK_SLOT;

  if (current_slot->num_dl_ul_symbols_list.num_ul == NR_SYMBOLS_PER_SLOT)
    return NR_UPLINK_SLOT;

  return NR_MIXED_SLOT;
}

/* Managed TX closes admission before the settings worker delivers SIGTERM.
 * Let queued actors unwind during that interval instead of aborting before the
 * coordinated shutdown can flush evidence. Other radio errors remain fatal. */
static inline bool radio_shutdown_cancelled(int result)
{
  return (result == -ESHUTDOWN && (oai_exit || nrue_ru_failed())) || radio_gain_device_tx_cancelled(result);
}

static bool RU_write(nr_rxtx_thread_data_t *rxtxD, bool sl_tx_action, bool prach_tx, c16_t **txp)
{
  int writeBlockSize = rxtxD->writeBlockSize;
  if (writeBlockSize == 0)
    return true;

  PHY_VARS_NR_UE *UE = rxtxD->UE;
  const fapi_nr_config_request_t *cfg = &UE->nrUE_config;
  const UE_nr_rxtx_proc_t *proc = &rxtxD->proc;

  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  if (UE->sl_mode == 2)
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;

  int slot = proc->nr_slot_tx;

  radio_tx_burst_flag_t flags = TX_BURST_INVALID;

  const int curr_slot_type = nr_ue_slot_select(cfg, slot);
  if (UE->received_config_request) {
    if (fp->frame_type == FDD || get_softmodem_params()->continuous_tx) {
      flags = TX_BURST_MIDDLE;
    // In case of Sidelink, USRP write needed only in case transmission
    // needs to be done in this slot and not based on tdd ULDL configuration.
    } else if (UE->sl_mode == 2) {
      if (sl_tx_action)
        flags = TX_BURST_START_AND_END;
    } else {
      int slots_frame = fp->slots_per_frame;
      if (curr_slot_type != NR_DOWNLINK_SLOT) {
        int next_slot = nr_ue_slot_select(cfg, (slot + 1) % slots_frame);
        int prev_slot = nr_ue_slot_select(cfg, (slot + slots_frame - 1) % slots_frame);
        if (prev_slot == NR_DOWNLINK_SLOT)
          flags = TX_BURST_START;
        else if (next_slot == NR_DOWNLINK_SLOT)
          flags = TX_BURST_END;
        else
          flags = TX_BURST_MIDDLE;
      }
    }
  }

  if (!IS_SOFTMODEM_RFSIM && flags != TX_BURST_INVALID) {
    const nr_ue_tx_deadline_t deadline = {.monotonic_ns = rxtxD->tx_deadline_monotonic_ns,
                                          .error_code = rxtxD->tx_deadline_error_code,
                                          .valid = rxtxD->tx_deadline_valid};
    struct timespec monotonic_time = {0};
    const int clock_status = deadline.valid ? clock_gettime(CLOCK_MONOTONIC, &monotonic_time) : -1;
    const int clock_error = deadline.valid && clock_status != 0 ? errno : deadline.error_code;
    const nr_ue_tx_deadline_check_t check =
        nr_ue_tx_deadline_check(&deadline, clock_status == 0 ? &monotonic_time : NULL, clock_error);
    if (flight_recorder_enabled()
        && flight_recorder_sample(FLIGHT_SAMPLE_TX_DEADLINE, proc->timestamp_tx, !check.valid || check.missed))
      flight_recorder_emit(FLIGHT_EVENT_UE_TX_DEADLINE,
                           UE->Mod_id,
                           proc->flight_sync_epoch,
                           proc->timestamp_tx,
                           check.valid ? check.monotonic_now_ns : INT64_MIN,
                           check.valid ? check.lateness_ns : INT64_MIN,
                           ((uint64_t)(uint32_t)check.error_code << 32) | (uint32_t)flags);
    if (!check.valid) {
      static _Atomic(uint64_t) deadline_error_rate_limit;
      if (nr_ue_tx_deadline_log_due(&deadline_error_rate_limit))
        LOG_E(PHY,
              "Cannot %s deadline for tx slot %d.%d: error %d (%s)\n",
              deadline.valid ? "check" : "construct",
              proc->frame_tx,
              proc->nr_slot_tx,
              check.error_code,
              strerror(check.error_code));
    } else if (check.missed) {
      static _Atomic(uint64_t) deadline_warning_rate_limit;
      if (nr_ue_tx_deadline_log_due(&deadline_warning_rate_limit))
        LOG_W(PHY,
              "Deadline missed for tx slot %d.%d (monotonic time %" PRIu64 " ns, deadline %" PRIu64 " ns, missed by %" PRId64
              " ns)\n",
              proc->frame_tx,
              proc->nr_slot_tx,
              check.monotonic_now_ns,
              deadline.monotonic_ns,
              check.lateness_ns);
    }
  }

  openair0_timestamp_t writeTimestamp = proc->timestamp_tx;
  // PRACH already fills its complete interval; normal TX retains the repeated-prefix extension.
  const int maxWriteBlockSize = get_samples_per_slot(proc->nr_slot_tx, fp);
  while (!prach_tx && writeBlockSize > maxWriteBlockSize) {
    const int dummyBlockSize = min(writeBlockSize - maxWriteBlockSize, maxWriteBlockSize);
    int tmp = nrue_ru_write_reorder(UE, writeTimestamp, (void **)txp, dummyBlockSize, fp->nb_antennas_tx, flags);
    if (radio_shutdown_cancelled(tmp)) {
      // The rejected fragment has already advanced the reorder cursor. Retire
      // the rest of this scheduled interval so later actors cannot leave a hole.
      nrue_ru_write_reorder_cancel(UE, writeTimestamp + dummyBlockSize, writeBlockSize - dummyBlockSize);
      return false;
    }
    AssertFatal(tmp == dummyBlockSize, "write samples to reorder function failed %d", tmp);

    writeTimestamp += dummyBlockSize;
    writeBlockSize -= dummyBlockSize;
  }

  // if mixed slot in TDD, set start of burst in first guard symbol.
  if (flags == TX_BURST_START && curr_slot_type == NR_MIXED_SLOT) {
    const uint num_dl_symbols = num_dl_symbols_slot(cfg, slot);
    const uint num_zero_samples = get_samples_symbol_timestamp(fp, slot, num_dl_symbols);
    int tmp = nrue_ru_write_reorder(UE, writeTimestamp, (void **)txp, num_zero_samples, fp->nb_antennas_tx, TX_BURST_INVALID);
    if (radio_shutdown_cancelled(tmp)) {
      if (writeBlockSize > num_zero_samples)
        nrue_ru_write_reorder_cancel(UE, writeTimestamp + num_zero_samples, writeBlockSize - num_zero_samples);
      return false;
    }
    AssertFatal(tmp == num_zero_samples, "write samples to reorder function failed %d", tmp);

    writeTimestamp += num_zero_samples;
    writeBlockSize -= num_zero_samples;
    for (int ant = 0; ant < fp->nb_antennas_tx; ant++)
      txp[ant] += num_zero_samples;
  }

  // pre-compensate UL frequency offset
  if (flags != TX_BURST_INVALID && get_nrUE_params()->cont_fo_comp) {
    double ul_freq_offset = -UE->freq_offset * ((double)fp->ul_CarrierFreq / (double)fp->dl_CarrierFreq);
    if (get_nrUE_params()->cont_fo_comp == 2) // different from LO frequency error compensation, Doppler UL pre-compensation has to be negative
      ul_freq_offset = -ul_freq_offset;
    else if (get_nrUE_params()->cont_fo_comp == 3) // do not consider residual DL FO for UL pre-compensation at all
      ul_freq_offset = 0;
    for (int i = 0; i < fp->nb_antennas_tx; i++)
      nr_fo_compensation(UE->ul_Doppler_shift + ul_freq_offset,
                         fp->samples_per_subframe,
                         writeTimestamp,
                         txp[i],
                         txp[i],
                         writeBlockSize);
  }

  int tmp = nrue_ru_write_reorder(UE, writeTimestamp, (void **)txp, writeBlockSize, fp->nb_antennas_tx, flags);
  if (radio_shutdown_cancelled(tmp))
    return false;
  AssertFatal(tmp == writeBlockSize, "write to reorder function failed %d", tmp);
  return true;
}

static void apply_ta_samples(PHY_VARS_NR_UE *ue, int frame, int slot, int command, bool is_rar, int samples)
{
  const int timing_advance_before = ue->timing_advance;
  const int64_t timing_advance = is_rar ? samples : (int64_t)timing_advance_before + samples;
  DevAssert(timing_advance >= INT_MIN && timing_advance <= INT_MAX);
  ue->timing_advance = timing_advance;
  // Emit actual application, including zero-delta commands; bit 16 identifies RAR.
  flight_recorder_emit(FLIGHT_EVENT_UE_TA_APPLIED,
                       ue->Mod_id,
                       frame * 1000 + slot,
                       (uint16_t)command | ((int64_t)is_rar << 16),
                       timing_advance_before,
                       ue->timing_advance,
                       ue->frame_parms.samples_per_subframe);
}

static void update_tx_timing(nr_rxtx_thread_data_t *rxtxD)
{
  PHY_VARS_NR_UE *ue = rxtxD->UE;
  UE_nr_rxtx_proc_t *proc = &rxtxD->proc;
  const NR_DL_FRAME_PARMS *fp = ue->sl_mode == 2 ? &ue->SL_UE_PHY_PARAMS.sl_frame_params : &ue->frame_parms;

  if (ue->sl_mode != 2) {
    const int next_slot = (proc->nr_slot_tx + 1) % fp->slots_per_frame;
    const int next_frame = (proc->frame_tx + (next_slot == 0)) % MAX_FRAME_NUMBER;
    const bool rar_due = ue->rar_ta_command.ta_frame == next_frame && ue->rar_ta_command.ta_slot == next_slot;
    const int rar_command = ue->rar_ta_command.ta_command;
    // Move the blank boundary before Msg3; relative updates retain their post-sizing order below.
    const int rar_delta = nr_ue_ta_command_samples(ue, next_slot, next_frame, true, NULL);
    if (rar_due)
      apply_ta_samples(ue, proc->frame_tx, proc->nr_slot_tx, rar_command, true, rar_delta);
  }

  const int stream_adjustment = ue->tx_stream_timing_adjustment;
  proc->tx_mac_stream_adjustment = stream_adjustment;
  proc->timestamp_tx -= stream_adjustment;
  const int requested_ta = ue->sl_mode == 2 ? 0 : ue->timing_advance;
  const int64_t write_size = (int64_t)rxtxD->writeBlockSize - ((int64_t)requested_ta - stream_adjustment);
  if (write_size < 0) {
    // Carry the unapplied change into later chunks without reversing the sample stream.
    const int64_t carried_adjustment = (int64_t)requested_ta + write_size;
    DevAssert(carried_adjustment >= INT_MIN && carried_adjustment <= INT_MAX);
    ue->tx_stream_timing_adjustment = carried_adjustment;
    rxtxD->writeBlockSize = 0;
  } else {
    DevAssert(write_size <= INT_MAX);
    ue->tx_stream_timing_adjustment = requested_ta;
    rxtxD->writeBlockSize = write_size;
  }

  // RX captured each relative update at its original application point, after sizing the current slot.
  if (ue->sl_mode != 2 && rxtxD->relative_ta_command >= 0)
    apply_ta_samples(ue, proc->frame_tx, proc->nr_slot_tx, rxtxD->relative_ta_command, false, rxtxD->relative_ta_delta);

  // The original RX anchor remains immutable; the deadline and evidence describe the final sample timestamp.
  const nr_ue_tx_deadline_anchor_t *anchor = &rxtxD->tx_deadline_anchor;
  const nr_ue_tx_deadline_t deadline =
      nr_ue_tx_deadline_compute(anchor, proc->timestamp_tx, fp->ofdm_symbol_size, fp->samples_per_subframe);
  rxtxD->tx_deadline_monotonic_ns = deadline.monotonic_ns;
  rxtxD->tx_deadline_error_code = deadline.error_code;
  rxtxD->tx_deadline_valid = deadline.valid;
  if (flight_recorder_enabled()
      && flight_recorder_sample(FLIGHT_SAMPLE_TX_SCHEDULE, proc->timestamp_tx, !deadline.valid || !anchor->valid)) {
    const int64_t tx_slot = ((int64_t)proc->hfn_tx * MAX_FRAME_NUMBER + proc->frame_tx) * fp->slots_per_frame + proc->nr_slot_tx;
    flight_recorder_emit(FLIGHT_EVENT_UE_TX_SCHEDULE,
                         ue->Mod_id,
                         proc->flight_sync_epoch,
                         tx_slot,
                         proc->timestamp_tx,
                         deadline.valid ? deadline.monotonic_ns : INT64_MIN,
                         rxtxD->writeBlockSize);
    flight_recorder_emit(FLIGHT_EVENT_UE_TX_ANCHOR,
                         ue->Mod_id,
                         proc->flight_sync_epoch,
                         proc->timestamp_tx,
                         anchor->valid ? anchor->radio_timestamp : INT64_MIN,
                         anchor->valid ? anchor->monotonic_ns : INT64_MIN,
                         ((uint64_t)(uint32_t)fp->ofdm_symbol_size << 32) | (uint32_t)fp->samples_per_subframe);
  }
}

void processSlotTX(void *arg)
{
  TracyCZone(ctx, true);
  if (oai_exit || nrue_ru_failed()) {
    TracyCZoneEnd(ctx);
    return;
  }
  nr_rxtx_thread_data_t *rxtxD = arg;
  const UE_nr_rxtx_proc_t *proc = &rxtxD->proc;
  PHY_VARS_NR_UE *UE = rxtxD->UE;
  nr_phy_data_tx_t phy_data = {0};
  bool sl_tx_action = false;
  bool prach_tx = false;
  bool tx_samples_valid = true;

  if (UE->if_inst)
    UE->if_inst->slot_indication(UE->Mod_id, true);

  // This prefix is ordered by the previous TX slot; publish immutable timing before releasing the next slot.
  update_tx_timing(rxtxD);

  LOG_D(PHY, "SlotTx %d.%d => slot type %d\n", proc->frame_tx, proc->nr_slot_tx, proc->tx_slot_type);

  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  const int tx_slot_buffer_factor = UE->sl_mode == 2 ? 1 : NR_UE_TX_SLOT_BUFFER_FACTOR;
  c16_t *txp[fp->nb_antennas_tx];
  for (int i = 0; i < fp->nb_antennas_tx; i++) {
    txp[i] = UE->common_vars.txData[i] + (size_t)tx_slot_buffer_factor * get_samples_slot_timestamp(fp, proc->nr_slot_tx);
  }

  if (proc->tx_slot_type == NR_UPLINK_SLOT || proc->tx_slot_type == NR_MIXED_SLOT) {
    if (UE->sl_mode == 2 && proc->tx_slot_type == NR_SIDELINK_SLOT) {
      // trigger L2 to run ue_sidelink_scheduler thru IF module
      if (UE->if_inst != NULL && UE->if_inst->sl_indication != NULL) {
        start_meas(&UE->ue_ul_indication_stats);
        nr_sidelink_indication_t sl_indication = {.module_id = UE->Mod_id,
                                                  .hfn_tx = proc->hfn_tx,
                                                  .frame_tx = proc->frame_tx,
                                                  .slot_tx = proc->nr_slot_tx,
                                                  .hfn_rx = proc->hfn_rx,
                                                  .frame_rx = proc->frame_rx,
                                                  .slot_rx = proc->nr_slot_rx,
                                                  .slot_type = SIDELINK_SLOT_TYPE_TX,
                                                  .phy_data = &phy_data};

        UE->if_inst->sl_indication(&sl_indication);
        stop_meas(&UE->ue_ul_indication_stats);
      }
      dynamic_barrier_join(rxtxD->next_barrier);

      if (phy_data.sl_tx_action) {

        AssertFatal((phy_data.sl_tx_action >= SL_NR_CONFIG_TYPE_TX_PSBCH &&
                     phy_data.sl_tx_action < SL_NR_CONFIG_TYPE_TX_MAXIMUM), "Incorrect SL TX Action Scheduled\n");

        phy_procedures_nrUE_SL_TX(UE, proc, &phy_data, txp);

        sl_tx_action = true;
      }

    } else {
      // trigger L2 to run ue_scheduler thru IF module
      // [TODO] mapping right after NR initial sync
      if (UE->if_inst != NULL && UE->if_inst->ul_indication != NULL) {
        start_meas(&UE->ue_ul_indication_stats);
        nr_uplink_indication_t ul_indication = {.module_id = UE->Mod_id,
                                                .frame = proc->frame_tx,
                                                .slot = proc->nr_slot_tx,
                                                .phy_data = &phy_data};

        UE->if_inst->ul_indication(&ul_indication);
        stop_meas(&UE->ue_ul_indication_stats);
      }
      phy_data.prach_handled = true;
      const NR_UE_PRACH *prach = UE->prach_vars[proc->gNB_id];
      if (prach->active || prach->tx_request_pending || prach->tx_waveform_active) {
        prach_tx = true;
        const int guard_samples =
            fp->frame_type == TDD && proc->tx_slot_type == NR_MIXED_SLOT
                ? get_samples_symbol_timestamp(fp, proc->nr_slot_tx, num_dl_symbols_slot(&UE->nrUE_config, proc->nr_slot_tx))
                : 0;
        tx_samples_valid = nr_ue_prach_tx(UE, proc, &phy_data, txp, rxtxD->writeBlockSize, guard_samples);
        dynamic_barrier_join(rxtxD->next_barrier);
      } else {
        dynamic_barrier_join(rxtxD->next_barrier);
        phy_procedures_nrUE_TX(UE, proc, &phy_data, txp);
      }
    }
  } else {
    const NR_UE_PRACH *prach = UE->prach_vars[proc->gNB_id];
    if (UE->sl_mode != 2 && (prach->active || prach->tx_request_pending || prach->tx_waveform_active)) {
      prach_tx = true;
      tx_samples_valid = nr_ue_prach_tx(UE, proc, &phy_data, txp, rxtxD->writeBlockSize, 0);
    }
    dynamic_barrier_join(rxtxD->next_barrier);
  }
  if (!tx_samples_valid) {
    // An unrepresentable interval cannot advance the reorder cursor.
    if (rxtxD->writeBlockSize > 0 && proc->timestamp_tx <= INT64_MAX - rxtxD->writeBlockSize)
      nrue_ru_write_reorder_cancel(UE, proc->timestamp_tx, rxtxD->writeBlockSize);
  } else if (RU_write(rxtxD, sl_tx_action, prach_tx, txp))
    flight_monitor_add(FLIGHT_MONITOR_UE_TX_COMPLETED, 1);
  TracyCZoneEnd(ctx);
}

static uint64_t get_carrier_frequency(const int N_RB, const int mu, const uint32_t pointA_freq_khz)
{
  const uint64_t bw = (NR_NB_SC_PER_RB * N_RB) * MU_SCS(mu);
  const uint64_t carrier_freq = (pointA_freq_khz + bw / 2) * 1000;
  return carrier_freq;
}

static int handle_sync_req_from_mac(PHY_VARS_NR_UE *UE, Actor_t *dl_actors, Actor_t *ul_actors)
{
  NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  // Start synchronization with a target gNB
  if (UE->synch_request.received_synch_request == 1) {
    // if upper layers signal BW scan we do as instructed by command line parameter
    // if upper layers disable BW scan we set it to false
    if (UE->synch_request.synch_req.ssb_bw_scan)
      UE->UE_scan_carrier = get_nrUE_params()->UE_scan_carrier;
    else
      UE->UE_scan_carrier = false;
    UE->target_Nid_cell = UE->synch_request.synch_req.target_Nid_cell;

    const fapi_nr_config_request_t *config = &UE->nrUE_config;
    const fapi_nr_ue_carrier_config_t *cfg = &config->carrier_config;
    if (cfg->dl_frequency != 0) {
      uint64_t dl_CarrierFreq = get_carrier_frequency(fp->N_RB_DL, fp->numerology_index, cfg->dl_frequency);
      uint64_t ul_CarrierFreq = get_carrier_frequency(fp->N_RB_UL, fp->numerology_index, cfg->uplink_frequency);
      if (dl_CarrierFreq != fp->dl_CarrierFreq || ul_CarrierFreq != fp->ul_CarrierFreq) {
        LOG_I(NR_PHY,
              "[UE %d] SYNC REQ: RF frequency change: dl %lu->%lu Hz, ul %lu->%lu Hz (from dl_frequency=%u kHz, "
              "target_Nid_cell=%d)\n",
              UE->Mod_id,
              fp->dl_CarrierFreq,
              dl_CarrierFreq,
              fp->ul_CarrierFreq,
              ul_CarrierFreq,
              cfg->dl_frequency,
              UE->target_Nid_cell);
        if (nrue_ru_set_freq(UE, ul_CarrierFreq, dl_CarrierFreq, 0) != 0)
          return 1;
        fp->dl_CarrierFreq = dl_CarrierFreq;
        fp->ul_CarrierFreq = ul_CarrierFreq;
        init_symbol_rotation(fp);
      }

      int ssb_start_subcarrier = nr_get_ssb_start_sc(fp->numerology_index,
                                                     config->ssb_table.ssb_offset_point_a,
                                                     config->ssb_table.ssb_subcarrier_offset,
                                                     fp->freq_range);
      // SSB location can change during for ex: handover on the target cell
      if (ssb_start_subcarrier != fp->ssb_start_subcarrier) {
        fp->ssb_start_subcarrier = ssb_start_subcarrier;
        LOG_I(NR_PHY, "SYNC REQ: SSB location changed:%d\n", fp->ssb_start_subcarrier);
      }
    }

    // Apply Doppler based on NTN-Config for target cell
    if (UE->nrUE_config.ntn_config.is_targetcell)
      apply_ntn_timing_advance_and_doppler(UE, fp, -1);
    // Apply NTN DL Doppler as initial FO
    UE->initial_fo = UE->dl_Doppler_shift;

    /* Clearing UE harq while DL actors are active causes race condition.
        So we let the current execution to complete here.*/
    for (int i = 0; i < get_nrUE_params()->num_dl_actors; i++) {
      flush_actor(&dl_actors[i]);
    }
    for (int i = 0; i < get_nrUE_params()->num_ul_actors; i++) {
      flush_actor(&ul_actors[i]);
    }

    clean_UE_harq(UE);
    radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_ACQUISITION);
    UE->is_synchronized = 0;
    UE->synch_request.received_synch_request = 0;
    return 0;
  }
  return 1;
}

static bool accept_sync_result(PHY_VARS_NR_UE *UE,
                               const NR_DL_FRAME_PARMS *fp,
                               const syncData_t *result,
                               int skipped_frames,
                               int *decoded_frame_rx,
                               int *initial_sync_offset)
{
  if (UE->sl_mode != SL_MODE2_SUPPORTED && !nr_rrc_bcch_is_current(UE->Mod_id, result->bcch_generation))
    UE->is_synchronized = 0;
  if (!UE->is_synchronized)
    return false;
  if (UE->sl_mode == SL_MODE2_SUPPORTED)
    *decoded_frame_rx = UE->SL_UE_PHY_PARAMS.sync_params.DFN;
  else
    *decoded_frame_rx = result->frame_number;
  LOG_A(PHY, "UE synchronized! decoded_frame_rx=%d skipped_frames=%d\n", *decoded_frame_rx, skipped_frames);
  const int64_t frame_samples = fp->samples_per_frame;
  const int64_t raw_offset = result->rx_offset;
  const int mib_frame = *decoded_frame_rx;
  int64_t frame_carry = 0;
  // A negative NR offset identifies a frame boundary behind the capture origin.
  // Align to the next reachable boundary and advance its SFN by the same whole frames.
  if (UE->sl_mode != SL_MODE2_SUPPORTED && raw_offset < 0)
    frame_carry = (-raw_offset + frame_samples - 1) / frame_samples;
  *initial_sync_offset = (int)(raw_offset + frame_carry * frame_samples);
  *decoded_frame_rx = (int)(((int64_t)*decoded_frame_rx + skipped_frames + frame_carry) % MAX_FRAME_NUMBER);
  flight_recorder_emit(FLIGHT_EVENT_UE_SYNC_ALIGNMENT,
                       UE->Mod_id,
                       raw_offset,
                       *initial_sync_offset,
                       frame_samples,
                       mib_frame,
                       ((uint64_t)(uint32_t)skipped_frames << 32) | (uint32_t)*decoded_frame_rx);
  return true;
}

static bool service_unsynchronized_control(PHY_VARS_NR_UE *UE, Actor_t *dl_actors, Actor_t *ul_actors)
{
  // A SYNC response is posted only after the actor has stopped using acquisition state.
  // Bound each pass so queue traffic or mutex contention cannot stop RX and RRC timer progress.
  enum { UNSYNC_RRC_MESSAGE_BUDGET = 8 };
  bool controls_ready = nr_ue_service_rrc_messages(UE->Mod_id, UNSYNC_RRC_MESSAGE_BUDGET);
  if (controls_ready) {
    handle_sync_req_from_mac(UE, dl_actors, ul_actors);
    NR_UE_MAC_INST_t *mac = get_mac_inst(UE->Mod_id);
    mutexlock(mac->if_mutex);
    controls_ready = !mac->pending_bcch_invalidation && mac->state != UE_DETACHING;
    mutexunlock(mac->if_mutex);
    controls_ready = controls_ready && !UE->synch_request.received_synch_request;
  }
  return controls_ready;
}

static int UE_dl_preprocessing(PHY_VARS_NR_UE *UE,
                               const UE_nr_rxtx_proc_t *proc,
                               int *tx_wait_for_dlsch,
                               nr_phy_data_t *phy_data,
                               bool *stats_printed,
                               int *relative_ta_delta,
                               int *relative_ta_command)
{
  TracyCZone(ctx, true);
  int sampleShift = INT_MAX;
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  if (UE->sl_mode == 2)
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;

  // process what RRC thread sent to MAC
  do {
    notifiedFIFO_elt_t *elt = pollNotifiedFIFO(&get_mac_inst(UE->Mod_id)->input_nf);
    if (!elt) {
      break;
    }
    process_msg_rcc_to_mac(NotifiedFifoData(elt), UE->Mod_id);
    delNotifiedFIFO_elt(elt);
  } while (true);

  if (UE->if_inst)
    UE->if_inst->slot_indication(UE->Mod_id, false);

  bool dl_slot = false;
  if (proc->rx_slot_type == NR_DOWNLINK_SLOT || proc->rx_slot_type == NR_MIXED_SLOT) {
    dl_slot = true;
    if(UE->if_inst != NULL && UE->if_inst->dl_indication != NULL) {
      nr_downlink_indication_t dl_indication = (nr_downlink_indication_t){
          .gNB_index = proc->gNB_id,
          .module_id = UE->Mod_id,
          .cc_id = UE->CC_id,
          .hfn = proc->hfn_rx,
          .frame = proc->frame_rx,
          .slot = proc->nr_slot_rx,
          .phy_data = phy_data,
      };
      UE->if_inst->dl_indication(&dl_indication);
    }

    sampleShift = pbch_processing(UE, proc, phy_data);
    pdcch_processing(UE, proc, phy_data);
    if (phy_data->dlsch[0].active
        && (phy_data->dlsch[0].rnti_type == TYPE_C_RNTI_ || phy_data->dlsch[0].rnti_type == TYPE_RA_RNTI_)) {
      // indicate to tx thread to wait for DLSCH decoding
      if (phy_data->dlsch_config.k1_feedback) {  // if feedback is 0 there is no HARQ associated with this DLSCH
        const int ack_nack_slot = (proc->nr_slot_rx + phy_data->dlsch_config.k1_feedback) % fp->slots_per_frame;
        tx_wait_for_dlsch[ack_nack_slot]++;
      }
    }
  }
  if (fp->frame_type == FDD || !dl_slot) {
    // good time to print statistics, we don't have to spend time  to decode DCI
    if (proc->frame_rx % 128 == 0) {
      if (*stats_printed == false) {
        print_ue_mac_stats(UE->Mod_id, proc->frame_rx, proc->nr_slot_rx);
        *stats_printed = true;
      }
    } else {
      *stats_printed = false;
    }
  }

  if (UE->sl_mode == 2) {
    if (proc->rx_slot_type == NR_SIDELINK_SLOT) {
      phy_data->sl_rx_action = 0;
      if (UE->if_inst != NULL && UE->if_inst->sl_indication != NULL) {
        nr_sidelink_indication_t sl_indication;
        nr_fill_sl_indication(&sl_indication, NULL, NULL, proc, UE, phy_data);
        UE->if_inst->sl_indication(&sl_indication);
      }

      if (phy_data->sl_rx_action) {

        AssertFatal((phy_data->sl_rx_action >= SL_NR_CONFIG_TYPE_RX_PSBCH &&
                     phy_data->sl_rx_action < SL_NR_CONFIG_TYPE_RX_MAXIMUM), "Incorrect SL RX Action Scheduled\n");

        sampleShift = psbch_pscch_processing(UE, proc, phy_data);
      }
    }
  } else {
    *relative_ta_delta = nr_ue_ta_command_samples(UE, proc->nr_slot_tx, proc->frame_tx, false, relative_ta_command);
  }

  TracyCZoneEnd(ctx);
  return sampleShift;
}

void UE_dl_processing(void *arg) {
  TracyCZone(ctx, true);;
  nr_rxtx_thread_data_t *rxtxD = (nr_rxtx_thread_data_t *) arg;
  UE_nr_rxtx_proc_t *proc = &rxtxD->proc;
  PHY_VARS_NR_UE    *UE   = rxtxD->UE;
  nr_phy_data_t *phy_data = &rxtxD->phy_data;

  if (!UE->sl_mode)
    pdsch_processing(UE, proc, phy_data);

  flight_monitor_add(FLIGHT_MONITOR_UE_DL_COMPLETED, 1);
  TracyCZoneEnd(ctx);
}

static bool dummyWrite(PHY_VARS_NR_UE *UE, openair0_timestamp_t timestamp, int writeBlockSize)
{
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  if (UE->sl_mode == 2)
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;

  c16_t *dummy_tx[fp->nb_antennas_tx];
  c16_t dummy_tx_data[writeBlockSize];
  memset(dummy_tx_data, 0, sizeof(dummy_tx_data));
  for (int i = 0; i < fp->nb_antennas_tx; i++)
    dummy_tx[i] = dummy_tx_data;

  int tmp = nrue_ru_write(UE, timestamp, (void **)dummy_tx, writeBlockSize, fp->nb_antennas_tx, 4);
  if (radio_shutdown_cancelled(tmp))
    return false;
  AssertFatal(writeBlockSize == tmp, "write to reorder function failed %d", tmp);
  return true;
}

static int compute_sync_size(PHY_VARS_NR_UE *UE)
{
  int sz = 0;
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  // two frames for initial sync
  int num_frames = 2;
  // In Sidelink worst case SL-SSB can be sent once in 16 frames
  if (UE->sl_mode == 2) {
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;
    num_frames = SL_NR_PSBCH_REPETITION_IN_FRAMES;
  }
  for (int slot_rx = 0; slot_rx < fp->slots_per_subframe; slot_rx++)
    sz += get_samples_per_slot(slot_rx, fp);
  sz *= num_frames * NR_NUMBER_OF_SUBFRAMES_PER_FRAME;
  return sz;
}

static bool readFrame(PHY_VARS_NR_UE *UE,
                      openair0_timestamp_t *timestamp,
                      int duration_rx_to_tx,
                      int sz,
                      c16_t **result,
                      radio_gain_sample_context_t *gain_context)
{
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  c16_t *rxp[fp->nb_antennas_rx];
  if (!result) {
    int sz = 0;
    for (int slot = 0; slot < fp->slots_per_subframe; slot++)
      sz = max(sz, get_samples_per_slot(slot, fp));
    rxp[0] = malloc16(sz * sizeof(**rxp));
    for (int i = 1; i < fp->nb_antennas_rx; i++)
      rxp[i] = rxp[0];
  } else {
    for (int i = 0; i < fp->nb_antennas_rx; i++)
      rxp[i] = result[i];
  }

  bool completed = false;
  openair0_timestamp_t first_timestamp = 0, end_timestamp = 0;
  for (int remain = sz; remain > 0;) {
    for (int slot_rx = 0; slot_rx < fp->slots_per_subframe; slot_rx++) {
      int readBlockSize = min(get_samples_per_slot(slot_rx, fp), sz);
      int tmp = nrue_ru_read(UE, timestamp, (void **)rxp, readBlockSize, fp->nb_antennas_rx, NULL);
      if (radio_shutdown_cancelled(tmp))
        goto done;
      AssertFatal(readBlockSize == tmp, "read rf board failed %d", tmp);
      UEscopeCopy(UE, ueTimeDomainSamplesBeforeSync, rxp[0], sizeof(c16_t), 1, readBlockSize, 0);
      if (remain == sz)
        first_timestamp = *timestamp;
      end_timestamp = *timestamp + tmp;
      if (result)
        for (int i = 0; i < fp->nb_antennas_rx; i++)
          rxp[i] += readBlockSize;
      remain -= readBlockSize;
      if (IS_SOFTMODEM_RFSIM) {
        int slot_tx = (slot_rx + duration_rx_to_tx) % fp->slots_per_frame;
        int writeBlockSize = get_samples_per_slot(slot_tx, fp);
        int ta = UE->timing_advance + UE->timing_advance_ntn;
        const openair0_timestamp_t writeTimestamp =
            *timestamp + get_samples_slot_duration(fp, slot_rx, duration_rx_to_tx) - UE->N_TA_offset - ta;
        if (!dummyWrite(UE, writeTimestamp, writeBlockSize))
          goto done;
      }
    }
  }
  completed = true;
done:
  if (completed && gain_context)
    *gain_context = nrue_ru_sample_context(UE, first_timestamp, end_timestamp);
  if (!result)
    free(rxp[0]);
  return completed;
}

static bool syncInFrame(PHY_VARS_NR_UE *UE, openair0_timestamp_t *timestamp, int duration_rx_to_tx, openair0_timestamp_t rx_offset)
{
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  if (UE->sl_mode == 2)
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;

  LOG_I(PHY, "Resynchronizing RX by %ld samples\n", rx_offset);

  int size = rx_offset;
  while (size > 0) {
    // Set a maximum transfer size. As we usually read/write single slots, we use the size of slot 0 as maximum here.
    const int unitTransfer = min(get_samples_per_slot(0, fp), size);
    const int res = nrue_ru_read(UE, timestamp, (void **)UE->common_vars.rxdata, unitTransfer, fp->nb_antennas_rx, NULL);
    if (radio_shutdown_cancelled(res))
      return false;
    DevAssert(unitTransfer == res);
    if (IS_SOFTMODEM_RFSIM) {
      int ta = UE->timing_advance + UE->timing_advance_ntn;
      const openair0_timestamp_t writeTimestamp =
          *timestamp + get_samples_slot_duration(fp, 0, duration_rx_to_tx) - UE->N_TA_offset - ta;
      if (!dummyWrite(UE, writeTimestamp, unitTransfer))
        return false;
    }
    size -= unitTransfer;
  }
  return true;
}

// Keeps RRC timers (T310/T311/T304/...) advancing while out of sync: one RRC tick per frame read from the radio.
static void out_of_sync_rrc_timer_tick(PHY_VARS_NR_UE *UE, int num_frames, int *frame, int *hfn)
{
  for (int i = 0; i < num_frames; i++) {
    nr_ue_rrc_timer_trigger(UE->Mod_id, *hfn, *frame, 0);
    *frame = (*frame + 1) % MAX_FRAME_NUMBER;
    if (*frame == 0)
      (*hfn)++;
  }
}

static inline int get_firstSymSamp(uint16_t slot, const NR_DL_FRAME_PARMS *fp)
{
  return get_samples_symbol_duration(fp, slot, 0, 1);
}

static inline int get_readBlockSize(uint16_t slot, const NR_DL_FRAME_PARMS *fp)
{
  int rem_samples = get_samples_per_slot(slot, fp) - get_firstSymSamp(slot, fp);
  int next_slot_first_symbol = 0;
  if (slot < (fp->slots_per_frame-1))
    next_slot_first_symbol = get_firstSymSamp(slot+1, fp);
  return rem_samples + next_slot_first_symbol;
}

void trs_freq_correction(PHY_VARS_NR_UE *ue, int cfo)
{
  if (abs(cfo) > TRS_CFO_THRESH) {
    LOG_A(PHY, "CFO estimated (%d) from TRS exceeded threshold (%d). Adjusting radio CF\n", cfo, TRS_CFO_THRESH);
    uint64_t dl_carrier;
    uint64_t ul_carrier;
    nr_get_carrier_frequencies(ue, &dl_carrier, &ul_carrier);
    const int requested_offset = ue->freq_offset + cfo;
    if (nrue_ru_set_freq(ue, ul_carrier, dl_carrier, requested_offset) == 0)
      ue->freq_offset = requested_offset;
  }
}

/* Parse --actor-affinity into cores[]. Returns number of cores. Empty/NULL -> 0. */
static int parse_actor_affinity(const char *params, int *cores, int max_cores)
{
  if (params == NULL || params[0] == '\0')
    return 0;

  char *cpy = strdup(params);
  AssertFatal(cpy != NULL, "Memory exhausted\n");
  int n = 0;
  char *saveptr = NULL;
  for (char *tok = strtok_r(cpy, ",", &saveptr); tok != NULL; tok = strtok_r(NULL, ",", &saveptr)) {
    AssertFatal(n < max_cores, "actor-affinity has more than %d entries\n", max_cores);
    cores[n++] = atoi(tok);
  }
  free(cpy);
  return n;
}

/* Assign one core from the shared pool to each DL actor, then each UL actor.
 * If a spare remains, pin SYNC too. Leave everyone at -1 when actor-affinity is unset.
 * The actors are UE_thread-owned stack arrays: they live exactly as long as the thread. */
static void init_ue_actors(Actor_t *sync_actor, Actor_t *dl_actors, Actor_t *ul_actors)
{
  const nrUE_params_t *p = get_nrUE_params();
  const int need = p->num_dl_actors + p->num_ul_actors;
  int cores[64];
  const int have = parse_actor_affinity(p->actor_affinity, cores, sizeof(cores) / sizeof(cores[0]));
  AssertFatal(have == 0 || have >= need,
              "actor-affinity has %d entries but needs at least num-dl-actors (%d) + num-ul-actors (%d) = %d\n",
              have,
              p->num_dl_actors,
              p->num_ul_actors,
              need);

  int idx = 0;
  const int sync_core = (have > need) ? cores[need] : -1;
  init_actor(sync_actor, "SYNC_", sync_core);

  for (int i = 0; i < p->num_dl_actors; i++) {
    const int core = (have > 0) ? cores[idx++] : -1;
    init_actor(&dl_actors[i], "DL_", core);
  }
  for (int i = 0; i < p->num_ul_actors; i++) {
    const int core = (have > 0) ? cores[idx++] : -1;
    init_actor(&ul_actors[i], "UL_", core);
  }
  if (have > 0)
    LOG_I(PHY,
          "Pinned actors from actor-affinity=%s (DL %d, UL %d, SYNC %d)%s\n",
          p->actor_affinity,
          p->num_dl_actors,
          p->num_ul_actors,
          sync_core,
          have > need + 1 ? " (extra cores unused)" : "");
}

static int nrue_consume_timing_shift(int measured_shift, bool holdover, double drift, double *remainder)
{
  if (!holdover) {
    // Fresh feedback includes P+I; keep its integer correction unchanged.
    *remainder = 0.0;
    return measured_shift;
  }

  // Carry fractional samples between consumed frames, not queued commands:
  // fresh PBCH feedback can replace a queued holdover command before use.
  const double requested = drift + *remainder;
  const double applied = round(requested);
  AssertFatal(isfinite(applied) && applied >= INT_MIN && applied <= INT_MAX, "Invalid timing shift %f\n", applied);
  *remainder = requested - applied;
  return (int)applied;
}

void *UE_thread(void *arg)
{
  //this thread should be over the processing thread to keep in real time
  PHY_VARS_NR_UE *UE = (PHY_VARS_NR_UE *)arg;
  const NR_DL_FRAME_PARMS *fp = &UE->frame_parms;
  //  int tx_enabled = 0;
  enum stream_status_e stream_status = STREAM_STATUS_UNSYNC;
  fapi_nr_config_request_t *cfg = &UE->nrUE_config;
  sl_nr_phy_config_request_t *sl_cfg = NULL;
  if (UE->sl_mode == 2) {
    fp = &UE->SL_UE_PHY_PARAMS.sl_frame_params;
    sl_cfg = &UE->SL_UE_PHY_PARAMS.sl_config;
  }

  UE->is_synchronized = 0;
  radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_ACQUISITION);
  InitSinLUT();

  notifiedFIFO_t nf;
  initNotifiedFIFO(&nf);

  notifiedFIFO_t freeBlocks;
  initNotifiedFIFO_nothreadSafe(&freeBlocks);

  const int num_dl_actors = get_nrUE_params()->num_dl_actors;
  const int num_ul_actors = get_nrUE_params()->num_ul_actors;
  Actor_t sync_actor;
  // zero-length VLAs are undefined (C11 6.7.6.2), hence the size-1 floor
  Actor_t dl_actors[num_dl_actors > 0 ? num_dl_actors : 1];
  Actor_t ul_actors[num_ul_actors > 0 ? num_ul_actors : 1];
  init_ue_actors(&sync_actor, dl_actors, ul_actors);

  const double ntn_init_time_drift = get_nrUE_params()->ntn_init_time_drift;
  if (get_nrUE_params()->time_sync_I)
    // ntn_init_time_drift is in µs/s, max_pos_acc * time_sync_I is in samples/frame
    UE->max_pos_acc = ntn_init_time_drift * 1e-6 * fp->samples_per_frame / get_nrUE_params()->time_sync_I;
  else
    UE->max_pos_acc = 0;

  bool ntn_targetcell = false;
  int ntn_koffset = 0;
  int duration_rx_to_tx = get_nrUE_params()->ue_capability_rx2tx;
  int timing_advance = UE->timing_advance_ntn;
  UE->N_TA_offset = determine_N_TA_offset(UE);

  bool syncRunning = false;
  notifiedFIFO_elt_t *sync_result = NULL;
  const int nb_slot_frame = fp->slots_per_frame;
  int absolute_slot = 0, decoded_frame_rx = MAX_FRAME_NUMBER - 1, skipped_frames = 0;
  uint64_t flight_sync_epoch = 0;
  int tx_wait_for_dlsch[NR_MAX_SLOTS_PER_FRAME];
  // out-of-sync RRC timer tick frame/hfn (see out_of_sync_rrc_timer_tick())
  int out_of_sync_rrc_tick_frame = 0, out_of_sync_rrc_tick_hfn = 0;
  bool out_of_sync_rrc_tick_seeded = false;

  for(int i = 0; i < NUM_PROCESS_SLOT_TX_BARRIERS; i++) {
    dynamic_barrier_init(&UE->process_slot_tx_barriers[i]);
  }
  int shiftForNextFrame = 0;
  bool timing_holdover_pending = false;
  double timing_holdover_drift = 0.0;
  double timing_holdover_remainder = 0.0;
  int intialSyncOffset = 0;
  openair0_timestamp_t sync_timestamp;
  bool stats_printed = false;

  if (get_softmodem_params()->sync_ref && UE->sl_mode == 2) {
    UE->is_synchronized = 1;
    radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_TRACKING);
  } else {
    //warm up the RF board
    openair0_timestamp_t tmp;
    for (int i = 0; i < 50; i++) {
      if (!readFrame(UE, &tmp, duration_rx_to_tx, compute_sync_size(UE), NULL, NULL))
        break;
    }
  }

  c16_t *sync_buf[fp->nb_antennas_rx];
  memset(sync_buf, 0, sizeof(sync_buf)); // mandatory for CI compile options

  while (!oai_exit) {
    if (syncRunning) {
      notifiedFIFO_elt_t *res = pollNotifiedFIFO(&nf);

      if (res) {
        syncRunning = false;
        for (int i = 0; i < fp->nb_antennas_rx; i++)
          free(sync_buf[i]);
        flight_monitor_add(FLIGHT_MONITOR_SEARCH_ATTEMPTS, 1);
        sync_result = res;
        stream_status = STREAM_STATUS_UNSYNC;
      } else {
        if (IS_SOFTMODEM_IQPLAYER || IS_SOFTMODEM_IQRECORDER) {
          /* For IQ recorder-player we force synchronization to happen in a fixed duration so that
             the replay runs in sync with recorded samples.
          */
          openair0_config_t *cfg0 = &openair0_cfg_g[UE->rf_map.card];
          const unsigned int sync_in_frames = cfg0->recplay_conf->u_f_sync;
          while (skipped_frames != sync_in_frames) {
            if (!readFrame(UE, &sync_timestamp, duration_rx_to_tx, compute_sync_size(UE), NULL, NULL))
              break;
            skipped_frames += 2;
            out_of_sync_rrc_timer_tick(UE, 2, &out_of_sync_rrc_tick_frame, &out_of_sync_rrc_tick_hfn);
          }
        } else {
          if (!readFrame(UE, &sync_timestamp, duration_rx_to_tx, compute_sync_size(UE), NULL, NULL))
            break;
          const int num_frames_read = (UE->sl_mode == 2) ? SL_NR_PSBCH_REPETITION_IN_FRAMES : 2;
          skipped_frames += num_frames_read;
          out_of_sync_rrc_timer_tick(UE, num_frames_read, &out_of_sync_rrc_tick_frame, &out_of_sync_rrc_tick_hfn);
        }
        continue;
      }
    }

    AssertFatal(!syncRunning, "At this point synchronization can't be running\n");

    if (sync_result || !UE->is_synchronized) {
      if (!out_of_sync_rrc_tick_seeded) {
        out_of_sync_rrc_tick_frame = (absolute_slot / nb_slot_frame) % MAX_FRAME_NUMBER;
        out_of_sync_rrc_tick_hfn = (absolute_slot / nb_slot_frame) / MAX_FRAME_NUMBER;
        out_of_sync_rrc_tick_seeded = true;
      }

      bool controls_ready = service_unsynchronized_control(UE, dl_actors, ul_actors);
      if (!controls_ready) {
        if (!readFrame(UE, &sync_timestamp, duration_rx_to_tx, compute_sync_size(UE), NULL, NULL))
          break;
        const int frames = UE->sl_mode == SL_MODE2_SUPPORTED ? SL_NR_PSBCH_REPETITION_IN_FRAMES : 2;
        skipped_frames += frames;
        out_of_sync_rrc_timer_tick(UE, frames, &out_of_sync_rrc_tick_frame, &out_of_sync_rrc_tick_hfn);
        continue;
      }

      if (sync_result) {
        if (accept_sync_result(UE, fp, NotifiedFifoData(sync_result), skipped_frames, &decoded_frame_rx, &intialSyncOffset)) {
          flight_monitor_add(FLIGHT_MONITOR_SYNC_SUCCESSES, 1);
          out_of_sync_rrc_tick_seeded = false;
        }
        delNotifiedFIFO_elt(sync_result);
        sync_result = NULL;
      }
    }

    if (!UE->is_synchronized) {
      int sz = compute_sync_size(UE);
      for (int i = 0; i < fp->nb_antennas_rx; i++)
        sync_buf[i] = malloc(sz * sizeof(**sync_buf));
      radio_gain_sample_context_t sync_gain;
      if (!readFrame(UE, &sync_timestamp, duration_rx_to_tx, sz, sync_buf, &sync_gain)) {
        for (int i = 0; i < fp->nb_antennas_rx; i++)
          free(sync_buf[i]);
        break;
      }
      notifiedFIFO_elt_t *Msg = newNotifiedFIFO_elt(sizeof(syncData_t), 0, &nf, UE_synch);
      syncData_t *syncMsg = (syncData_t *)NotifiedFifoData(Msg);
      *syncMsg = (syncData_t){.input = sync_buf, .input_sz = sz};
      if (UE->UE_scan_carrier) {
        // Get list of GSCN in this band for UE's bandwidth and center frequency.
        LOG_W(PHY, "UE set to scan all GSCN in current bandwidth\n");
        syncMsg->numGscn =
            get_scan_ssb_first_sc(fp->dl_CarrierFreq, fp->N_RB_DL, nrue_get_band(UE), fp->numerology_index, syncMsg->gscnInfo);
      } else {
        LOG_W(PHY, "SSB position provided\n");
        syncMsg->gscnInfo[0] = (nr_gscn_info_t){.ssbFirstSC = fp->ssb_start_subcarrier};
        syncMsg->numGscn = 1;
      }
      syncMsg->UE = UE;
      memset(&syncMsg->proc, 0, sizeof(syncMsg->proc));
      syncMsg->proc.rx_gain_context = sync_gain;
      skipped_frames = UE->sl_mode == 2 ? SL_NR_PSBCH_REPETITION_IN_FRAMES : 2; // the capture for decoding
      out_of_sync_rrc_timer_tick(UE, skipped_frames, &out_of_sync_rrc_tick_frame, &out_of_sync_rrc_tick_hfn);
      if (UE->sl_mode == SL_MODE2_SUPPORTED) {
        pushNotifiedFIFO(&sync_actor.fifo, Msg);
      } else if (!nr_ue_publish_sync_job(UE->Mod_id, &sync_actor.fifo, Msg, &syncMsg->bcch_generation)) {
        // A control published during capture must be serviced before another search starts.
        delNotifiedFIFO_elt(Msg);
        for (int i = 0; i < fp->nb_antennas_rx; i++)
          free(sync_buf[i]);
        continue;
      }
      syncRunning = true;
      continue;
    }

    if (stream_status == STREAM_STATUS_UNSYNC) {
      stream_status = STREAM_STATUS_SYNCING;
      if (!syncInFrame(UE, &sync_timestamp, duration_rx_to_tx, intialSyncOffset))
        break;
      nrue_ru_write_reorder_clear_context(UE);
      shiftForNextFrame = -(skipped_frames)*UE->max_pos_acc
                          * get_nrUE_params()->time_sync_I; // compensate for the time drift that happened during initial sync
      timing_holdover_pending = false;
      timing_holdover_remainder = 0.0;
      LOG_I(PHY, "max_pos_acc = %d, shiftForNextFrame = %d\n", UE->max_pos_acc, shiftForNextFrame);
      // read in first symbol
      int ret = nrue_ru_read(UE,
                             &sync_timestamp,
                             (void **)UE->common_vars.rxdata,
                             fp->ofdm_symbol_size + fp->nb_prefix_samples0,
                             fp->nb_antennas_rx,
                             NULL);
      if (radio_shutdown_cancelled(ret))
        break;
      AssertFatal(fp->ofdm_symbol_size + fp->nb_prefix_samples0 == ret, "read rf board failed %d", ret);
      flight_sync_epoch++;
      if (flight_recorder_enabled())
        flight_recorder_emit(FLIGHT_EVENT_UE_SYNC_EPOCH,
                             UE->Mod_id,
                             flight_sync_epoch,
                             sync_timestamp,
                             decoded_frame_rx,
                             UE->freq_offset,
                             fp->samples_per_frame);
      // we have the decoded frame index in the return of the synch process
      // and we shifted above to the first slot of next frame
      const int prev_frame_rx = (absolute_slot / nb_slot_frame) % MAX_FRAME_NUMBER;
      const int prev_hfn_rx = (absolute_slot / nb_slot_frame) / MAX_FRAME_NUMBER;
      int decoded_hfn_rx = prev_hfn_rx;
      if (decoded_frame_rx <= prev_frame_rx)
        decoded_hfn_rx++;
      // we do ++ first in the regular processing, so it will be begin of frame;
      absolute_slot = (decoded_hfn_rx * MAX_FRAME_NUMBER + decoded_frame_rx) * nb_slot_frame - 1;
      if (UE->sl_mode == 2) {
        // Set to the slot where the SL-SSB was decoded
        absolute_slot += UE->SL_UE_PHY_PARAMS.sync_params.slot_offset;
      }
      // With the correct frame and slot numbers, we can now fix the UL timing
      fix_ntn_epoch_hfn(UE, decoded_hfn_rx, decoded_frame_rx);
      if (UE->nrUE_config.ntn_config.params_changed) {
        apply_ntn_config(UE,
                         fp,
                         decoded_hfn_rx,
                         decoded_frame_rx,
                         0,
                         &duration_rx_to_tx,
                         &timing_advance,
                         &ntn_koffset,
                         &ntn_targetcell);
      } else {
        const int abs_subframe_tx = (absolute_slot + 1 + duration_rx_to_tx) / fp->slots_per_subframe;
        apply_ntn_timing_advance_and_doppler(UE, fp, abs_subframe_tx);
        ntn_targetcell = false;
      }
      UE->timing_advance = 0;
      UE->tx_stream_timing_adjustment = 0;
      UE->ta_frame = -1;
      UE->ta_slot = -1;
      UE->rar_ta_command.ta_frame = -1;
      UE->rar_ta_command.ta_slot = -1;
      for (int gNB_id = 0; gNB_id < UE->n_connected_gNB; gNB_id++) {
        UE->prach_vars[gNB_id]->active = false;
        UE->prach_vars[gNB_id]->tx_request_pending = false;
        UE->prach_vars[gNB_id]->tx_waveform_active = false;
      }
      // We have resynchronized, maybe after RF loss so we need to purge any existing context
      memset(tx_wait_for_dlsch, 0, sizeof(tx_wait_for_dlsch));
      for (int i = 0; i < NUM_PROCESS_SLOT_TX_BARRIERS; i++) {
        dynamic_barrier_reset(&UE->process_slot_tx_barriers[i]);
      }
      continue;
    }

    /* check if MAC has sent sync request */
    if (handle_sync_req_from_mac(UE, dl_actors, ul_actors) == 0)
      continue;

    // start of normal case, the UE is in sync
    absolute_slot++;
    TracyCFrameMark;

    // pretend we have 1 iq sample per slot
    // and so nb_slot_frame * 100 iq samples per second (1 frame being 10ms)
    time_manager_iq_samples(1, nb_slot_frame * 100);

    int slot_nr = absolute_slot % nb_slot_frame;
    nr_rxtx_thread_data_t curMsg = {0};
    curMsg.UE=UE;
    curMsg.proc.flight_sync_epoch = flight_sync_epoch;
    // update thread index for received subframe
    curMsg.proc.nr_slot_rx  = slot_nr;
    curMsg.proc.nr_slot_tx  = (absolute_slot + duration_rx_to_tx) % nb_slot_frame;
    curMsg.proc.frame_rx    = (absolute_slot / nb_slot_frame) % MAX_FRAME_NUMBER;
    curMsg.proc.frame_tx    = ((absolute_slot + duration_rx_to_tx) / nb_slot_frame) % MAX_FRAME_NUMBER;
    curMsg.proc.hfn_rx      = (absolute_slot / nb_slot_frame) / MAX_FRAME_NUMBER;
    curMsg.proc.hfn_tx      = ((absolute_slot + duration_rx_to_tx) / nb_slot_frame) / MAX_FRAME_NUMBER;
    if (UE->received_config_request) {
      if (UE->sl_mode) {
        curMsg.proc.rx_slot_type = sl_nr_ue_slot_select(sl_cfg, curMsg.proc.nr_slot_rx, TDD);
        curMsg.proc.tx_slot_type = sl_nr_ue_slot_select(sl_cfg, curMsg.proc.nr_slot_tx, TDD);
      } else {
        curMsg.proc.rx_slot_type = nr_ue_slot_select(cfg, curMsg.proc.nr_slot_rx);
        curMsg.proc.tx_slot_type = nr_ue_slot_select(cfg, curMsg.proc.nr_slot_tx);
      }
    }
    else {
      curMsg.proc.rx_slot_type = NR_DOWNLINK_SLOT;
      curMsg.proc.tx_slot_type = NR_DOWNLINK_SLOT;
    }

    int firstSymSamp = get_firstSymSamp(slot_nr, fp);
    c16_t *rxp[fp->nb_antennas_rx];
    for (int i = 0; i < fp->nb_antennas_rx; i++)
      rxp[i] = &UE->common_vars.rxdata[i][firstSymSamp + get_samples_slot_timestamp(fp, slot_nr)];

    int iq_shift_to_apply = 0;
    if (slot_nr == nb_slot_frame - 1) {
      // we shift of half of measured drift, at each beginning of frame for both rx and tx
      iq_shift_to_apply =
          nrue_consume_timing_shift(shiftForNextFrame, timing_holdover_pending, timing_holdover_drift, &timing_holdover_remainder);
      // autonomous timing advance calculation, which does not use SIB19 information
      if (ntn_koffset && get_nrUE_params()->autonomous_ta)
        UE->timing_advance_ntn -= 2 * iq_shift_to_apply;
      timing_holdover_pending = !UE->sl_mode;
      timing_holdover_drift = -(UE->max_pos_acc * get_nrUE_params()->time_sync_I);
      if (!timing_holdover_pending)
        shiftForNextFrame = -round(UE->max_pos_acc * get_nrUE_params()->time_sync_I);
    }

    // Calculate new TA based on SIB19 information for each subframe in NTN mode, if "autonomous_ta" is not enabled
    if (ntn_koffset && !ntn_targetcell && !get_nrUE_params()->autonomous_ta
        && (absolute_slot + duration_rx_to_tx) % fp->slots_per_subframe == 0) {
      const int abs_subframe_tx = (absolute_slot + duration_rx_to_tx) / fp->slots_per_subframe;
      apply_ntn_timing_advance_and_doppler(UE, fp, abs_subframe_tx);
    }

    const int readBlockSize = get_readBlockSize(slot_nr, fp) - iq_shift_to_apply;
    openair0_timestamp_t rx_timestamp;
    nr_ue_tx_deadline_anchor_t deadline_anchor = {.error_code = EINVAL};
    int tmp = nrue_ru_read(UE,
                           &rx_timestamp,
                           (void **)rxp,
                           readBlockSize,
                           fp->nb_antennas_rx,
                           slot_nr != nb_slot_frame - 1 ? &deadline_anchor : NULL);
    if (radio_shutdown_cancelled(tmp))
      break;
    AssertFatal(readBlockSize == tmp, "read to rf board failed %d", tmp);
    metadata meta = {.slot =  curMsg.proc.nr_slot_rx, .frame =  curMsg.proc.frame_rx};
    UEscopeCopyWithMetadata(UE, ueTimeDomainSamples, rxp[0] - firstSymSamp, sizeof(c16_t), 1, readBlockSize, 0, &meta);

    if(slot_nr == (nb_slot_frame - 1)) {
      // read in first symbol of next frame and adjust for timing drift
      int first_symbols = fp->ofdm_symbol_size + fp->nb_prefix_samples0; // first symbol of every frames

      if (first_symbols > 0) {
        openair0_timestamp_t ignore_timestamp;
        int tmp = nrue_ru_read(UE,
                               &ignore_timestamp,
                               (void **)UE->common_vars.rxdata,
                               first_symbols,
                               fp->nb_antennas_rx,
                               &deadline_anchor);
        if (radio_shutdown_cancelled(tmp))
          break;
        AssertFatal(first_symbols == tmp, "read to rf board failed %d", tmp);

      } else {
        LOG_E(PHY,"can't compensate: diff =%d\n", first_symbols);
      }
    }

    // MAC TA is applied by the ordered TX prefix; this cache contains only the producer's NTN timing.
    const openair0_timestamp_t writeTimestamp =
        rx_timestamp + get_samples_slot_duration(fp, slot_nr, duration_rx_to_tx) - firstSymSamp - UE->N_TA_offset - timing_advance;

    // Leave the raw length unclamped so opposing NTN and MAC adjustments can cancel in the TX prefix.
    int writeBlockSize = get_samples_per_slot((slot_nr + duration_rx_to_tx) % nb_slot_frame, fp) - iq_shift_to_apply;
    int new_timing_advance = UE->timing_advance_ntn;
    if (new_timing_advance != timing_advance) {
      writeBlockSize -= new_timing_advance - timing_advance;
      timing_advance = new_timing_advance;
    }
    int new_N_TA_offset = determine_N_TA_offset(UE);
    if (new_N_TA_offset != UE->N_TA_offset) {
      LOG_I(PHY, "N_TA_offset changed from %d to %d\n", UE->N_TA_offset, new_N_TA_offset);
      writeBlockSize -= new_N_TA_offset - UE->N_TA_offset;
      UE->N_TA_offset = new_N_TA_offset;
    }
    if (curMsg.proc.nr_slot_rx == 0)
      nr_ue_rrc_timer_trigger(UE->Mod_id, curMsg.proc.hfn_rx, curMsg.proc.frame_rx, curMsg.proc.gNB_id);

    /* Include the first symbol read with the preceding slot, before handing
     * immutable context to asynchronous PHY consumers. */
    curMsg.proc.rx_gain_context =
        nrue_ru_sample_context(UE, rx_timestamp - firstSymSamp, rx_timestamp - firstSymSamp + get_samples_per_slot(slot_nr, fp));

    // RX slot processing. We launch and forget.
    flight_monitor_add(FLIGHT_MONITOR_UE_SLOT_INPUTS, 1);
    notifiedFIFO_elt_t *newRx = newNotifiedFIFO_elt(sizeof(nr_rxtx_thread_data_t), curMsg.proc.nr_slot_tx, NULL, UE_dl_processing);
    nr_rxtx_thread_data_t *curMsgRx = (nr_rxtx_thread_data_t *)NotifiedFifoData(newRx);
    *curMsgRx = (nr_rxtx_thread_data_t){.proc = curMsg.proc, .UE = UE};
    int relative_ta_delta = 0;
    int relative_ta_command = -1;
    int ret = UE_dl_preprocessing(UE,
                                  &curMsgRx->proc,
                                  tx_wait_for_dlsch,
                                  &curMsgRx->phy_data,
                                  &stats_printed,
                                  &relative_ta_delta,
                                  &relative_ta_command);
    if (ret != INT_MAX) {
      shiftForNextFrame = ret;
      timing_holdover_pending = false;
    }
    if (num_dl_actors > 0) {
      pushNotifiedFIFO(&dl_actors[curMsg.proc.nr_slot_rx % num_dl_actors].fifo, newRx);
    } else {
      newRx->processingFunc(curMsgRx);
    }

    // apply new NTN timing information
    apply_ntn_config(UE,
                     fp,
                     curMsg.proc.hfn_rx,
                     curMsg.proc.frame_rx,
                     curMsg.proc.nr_slot_rx,
                     &duration_rx_to_tx,
                     &timing_advance,
                     &ntn_koffset,
                     &ntn_targetcell);

    // Start TX slot processing here. It runs in parallel with RX slot processing
    // in current code, DURATION_RX_TO_TX constant is the limit to get UL data to encode from a RX slot
    notifiedFIFO_elt_t *newTx = newNotifiedFIFO_elt(sizeof(nr_rxtx_thread_data_t), 0, 0, processSlotTX);
    nr_rxtx_thread_data_t *curMsgTx = NotifiedFifoData(newTx);
    memset(curMsgTx, 0, sizeof(*curMsgTx));
    curMsgTx->proc = curMsg.proc;
    curMsgTx->writeBlockSize = writeBlockSize;
    curMsgTx->relative_ta_delta = relative_ta_delta;
    curMsgTx->relative_ta_command = relative_ta_command;
    curMsgTx->proc.timestamp_tx = writeTimestamp;
    curMsgTx->UE = UE;
    curMsgTx->ul_actors = ul_actors;
    curMsgTx->tx_deadline_anchor = deadline_anchor;

    int slot = curMsgTx->proc.nr_slot_tx;
    int slot_and_frame = slot + curMsgTx->proc.frame_tx * nb_slot_frame;
    int next_tx_slot_and_frame = absolute_slot + duration_rx_to_tx + 1;
    int wait_for_prev_slot = stream_status == STREAM_STATUS_SYNCED ? 1 : 0;

    dynamic_barrier_t *next_barrier = &UE->process_slot_tx_barriers[next_tx_slot_and_frame % NUM_PROCESS_SLOT_TX_BARRIERS];
    curMsgTx->next_barrier = next_barrier;
    if (!dynamic_barrier_try_update(&UE->process_slot_tx_barriers[slot_and_frame % NUM_PROCESS_SLOT_TX_BARRIERS],
                                    tx_wait_for_dlsch[slot] + wait_for_prev_slot,
                                    start_process_slot_tx,
                                    newTx)) {
      delNotifiedFIFO_elt(newTx);
      nrue_ru_tx_barrier_failed(UE, curMsg.proc.frame_tx, slot);
      break;
    }
    stream_status = STREAM_STATUS_SYNCED;
    tx_wait_for_dlsch[slot] = 0;
  }
  if (sync_result)
    delNotifiedFIFO_elt(sync_result);
  LOG_W(NR_PHY, "UE main thread is ending\n");
  for (int i = 0; i < num_ul_actors; i++)
    shutdown_actor(&ul_actors[i]);
  for (int i = 0; i < num_dl_actors; i++)
    shutdown_actor(&dl_actors[i]);
  destroy_actor(&sync_actor);
  // All producers are joined; no callback can race with pending-argument cleanup.
  for (int i = 0; i < NUM_PROCESS_SLOT_TX_BARRIERS; i++)
    dynamic_barrier_reset(&UE->process_slot_tx_barriers[i]);
  return NULL;
}

void init_NR_UE(int nb_inst, char *uecap_file, char *reconfig_file, char *rbconfig_file, int rx2tx, int numerology)
{
  for (int instance_id = 0; instance_id < nb_inst; instance_id++) {
    NR_UE_RRC_INST_t* rrc = nr_rrc_init_ue(uecap_file, instance_id, get_nrUE_params()->nb_antennas_tx);
    NR_UE_MAC_INST_t *mac = nr_l2_init_ue(instance_id, numerology, rx2tx);

    nr_rrc_set_mac_queue(instance_id, &mac->input_nf);
    mac->if_module = nr_ue_if_module_init(instance_id);
    AssertFatal(mac->if_module, "can not initialize IF module\n");
    if (!IS_SA_MODE(get_softmodem_params()) && !get_softmodem_params()->sl_mode) {
      init_nsa_message(rrc, reconfig_file, rbconfig_file);
      nr_rlc_activate_srb0(mac->crnti, NULL, send_srb0_rrc);
    }
    //TODO: Move this call to RRC
    start_sidelink(instance_id);
  }
}

void init_NR_UE_threads(PHY_VARS_NR_UE *UE) {
  char thread_name[16];
  sprintf(thread_name, "UEthread_%d", UE->Mod_id);
  threadCreate(&UE->main_thread, UE_thread, (void *)UE, thread_name, -1, OAI_PRIORITY_RT_MAX);
  if (!IS_SOFTMODEM_NOSTATS) {
    sprintf(thread_name, "L1_UE_stats_%d", UE->Mod_id);
    threadCreate(&UE->stat_thread, nrL1_UE_stats_thread, UE, thread_name, -1, OAI_PRIORITY_RT_LOW);
  }
}
