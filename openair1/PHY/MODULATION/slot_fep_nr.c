/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "PHY/defs_nr_UE.h"
#include "PHY/defs_gNB.h"
#include "modulation_UE.h"
#include "nr_modulation.h"
#include "PHY/LTE_ESTIMATION/lte_estimation.h"
#include "PHY/NR_UE_ESTIMATION/nr_estimation.h"
#include "PHY/nr_phy_common/inc/nr_phy_common.h"
#include <common/utils/LOG/log.h>

/* rxdataF should be 16 bytes aligned */
void nr_symbol_fep(const NR_DL_FRAME_PARMS *frame_parms,
                   const int slot,
                   const unsigned char symbol,
                   const int link_type,
                   c16_t *rxdata[frame_parms->nb_antennas_rx],
                   c16_t *rxdataF[frame_parms->nb_antennas_rx],
                   time_stats_t* dft_stats)
{
  AssertFatal(symbol < frame_parms->symbols_per_slot,
              "slot_fep: symbol must be between 0 and %d\n",
              frame_parms->symbols_per_slot - 1);
  AssertFatal(slot < frame_parms->slots_per_frame, "slot_fep: Ns must be between 0 and %d\n", frame_parms->slots_per_frame - 1);

  dft_size_idx_t dftsize = get_dft(frame_parms->ofdm_symbol_size);
  for (unsigned char aa = 0; aa < frame_parms->nb_antennas_rx; aa++) {
    if (dft_stats) start_meas(dft_stats);
    dft(dftsize, (int16_t *)rxdata[aa], (int16_t *)rxdataF[aa], 1);
    if (dft_stats) stop_meas(dft_stats);

    const bool is_sl = (link_type == link_type_sl);
    apply_nr_rotation_symbol_RX(frame_parms,
                                rxdataF[aa],
                                frame_parms->symbol_rotation[link_type],
                                is_sl ? frame_parms->N_RB_SL : frame_parms->N_RB_DL,
                                slot,
                                symbol);
  }
}

int nr_slot_fep(PHY_VARS_NR_UE *ue,
                const NR_DL_FRAME_PARMS *frame_parms,
                unsigned int slot,
                unsigned int symbol,
                c16_t rxdataF[][frame_parms->samples_per_slot_wCP],
                enum nr_Link linktype,
                uint32_t sample_offset,
                c16_t **rxdata)
{
  AssertFatal(symbol < frame_parms->symbols_per_slot,
              "slot_fep: symbol must be between 0 and %d\n",
              frame_parms->symbols_per_slot - 1);
  AssertFatal(slot < frame_parms->slots_per_frame, "slot_fep: Ns must be between 0 and %d\n", frame_parms->slots_per_frame - 1);

  bool is_sl = (linktype == link_type_sl);
  bool is_synchronized = (ue) ? ue->is_synchronized : false;
  unsigned int nb_prefix_samples = frame_parms->nb_prefix_samples;
  unsigned int nb_prefix_samples0 = (is_synchronized || is_sl) ? frame_parms->nb_prefix_samples0 : nb_prefix_samples;

  // For Sidelink 16 frames worth of samples is processed to find SSB, for 5G-NR 2.
  const unsigned int total_samples = (is_sl) ? 16 * frame_parms->samples_per_frame : 2 * frame_parms->samples_per_frame;

  unsigned int rx_offset = frame_parms->get_samples_slot_timestamp(slot,frame_parms,0);
  const unsigned int abs_symbol = slot * frame_parms->symbols_per_slot + symbol;
  for (int idx_symb = slot * frame_parms->symbols_per_slot; idx_symb <= abs_symbol; idx_symb++)
    rx_offset += (idx_symb % (0x7 << frame_parms->numerology_index)) ? nb_prefix_samples : nb_prefix_samples0;
  rx_offset += frame_parms->ofdm_symbol_size * symbol;

  rx_offset += sample_offset;

  // use OFDM symbol from within 1/8th of the CP to avoid ISI
  rx_offset -= (nb_prefix_samples / frame_parms->ofdm_offset_divisor);

  LOG_D(PHY,
        "slot_fep: slot %d, symbol %d, nb_prefix_samples %u, nb_prefix_samples0 %u, rx_offset %u energy %d\n",
        slot,
        symbol,
        nb_prefix_samples,
        nb_prefix_samples0,
        rx_offset,
        dB_fixed(signal_energy((int32_t *)&rxdata[0][rx_offset], frame_parms->ofdm_symbol_size)));

  c16_t tmp_dft_in[frame_parms->nb_antennas_rx][frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
  c16_t *rxdata_symb_ptr[frame_parms->nb_antennas_rx];
  c16_t *rxdataF_symb_ptr[frame_parms->nb_antennas_rx];
  for (unsigned char aa = 0; aa < frame_parms->nb_antennas_rx; aa++) {
    rxdataF_symb_ptr[aa] = &rxdataF[aa][frame_parms->ofdm_symbol_size * symbol];
    // This happens only during initial sync
    if (rx_offset + frame_parms->ofdm_symbol_size > total_samples) {
      // we have to wrap on the end
      memcpy(&tmp_dft_in[aa][0], &rxdata[aa][rx_offset], (total_samples - rx_offset) * sizeof(int32_t));
      memcpy(&tmp_dft_in[aa][total_samples - rx_offset],
             &rxdata[aa][0],
             (frame_parms->ofdm_symbol_size - (total_samples - rx_offset)) * sizeof(int32_t));
      rxdata_symb_ptr[aa] = tmp_dft_in[aa];
    } else {
      rxdata_symb_ptr[aa] = &rxdata[aa][rx_offset];
    }

    if (ue && ue->cont_fo_comp) {
      start_meas_nr_ue_phy(ue, RX_FO_COMPENSATION_STATS);
      nr_fo_compensation(ue->freq_offset,
                         frame_parms->samples_per_subframe,
                         rx_offset,
                         rxdata_symb_ptr[aa],
                         tmp_dft_in[aa],
                         frame_parms->ofdm_symbol_size);
      stop_meas_nr_ue_phy(ue, RX_FO_COMPENSATION_STATS);
      rxdata_symb_ptr[aa] = tmp_dft_in[aa];
    }
  }
  time_stats_t* dft_stats = NULL;
  if (ue) dft_stats = &ue->phy_cpu_stats.cpu_time_stats[RX_DFT_STATS];
  nr_symbol_fep(frame_parms, slot, symbol, linktype, rxdata_symb_ptr, rxdataF_symb_ptr, dft_stats);
  return 0;
}

int nr_slot_fep_ul(NR_DL_FRAME_PARMS *frame_parms,
                   int32_t *rxdata,
                   int32_t *rxdataF,
                   unsigned char symbol,
                   unsigned char Ns,
                   int sample_offset)
{
  unsigned int nb_prefix_samples  = frame_parms->nb_prefix_samples;
  unsigned int nb_prefix_samples0 = frame_parms->nb_prefix_samples0;

  dft_size_idx_t dftsize = get_dft(frame_parms->ofdm_symbol_size);
  // This is for misalignment issues
  int32_t tmp_dft_in[8192] __attribute__ ((aligned (32)));

  // offset of first OFDM symbol
  unsigned int rxdata_offset = frame_parms->get_samples_slot_timestamp(Ns,frame_parms,0);
  unsigned int abs_symbol = Ns * frame_parms->symbols_per_slot + symbol;
  for (int idx_symb = Ns*frame_parms->symbols_per_slot; idx_symb <= abs_symbol; idx_symb++)
    rxdata_offset += (idx_symb%(0x7<<frame_parms->numerology_index)) ? nb_prefix_samples : nb_prefix_samples0;
  rxdata_offset += frame_parms->ofdm_symbol_size * symbol;

  // use OFDM symbol from within 1/8th of the CP to avoid ISI
  rxdata_offset -= (nb_prefix_samples / frame_parms->ofdm_offset_divisor);

  int16_t *rxdata_ptr;

  if(sample_offset > rxdata_offset) {

    memcpy((void *)&tmp_dft_in[0],
           (void *)&rxdata[frame_parms->samples_per_frame - sample_offset + rxdata_offset],
           (sample_offset - rxdata_offset) * sizeof(int32_t));
    memcpy((void *)&tmp_dft_in[sample_offset - rxdata_offset],
           (void *)&rxdata[0],
           (frame_parms->ofdm_symbol_size - sample_offset + rxdata_offset) * sizeof(int32_t));
    rxdata_ptr = (int16_t *)tmp_dft_in;

  } else if (((rxdata_offset - sample_offset) & 7) != 0) {

    // if input to dft is not 256-bit aligned
    memcpy((void *)&tmp_dft_in[0],
           (void *)&rxdata[rxdata_offset - sample_offset],
           (frame_parms->ofdm_symbol_size) * sizeof(int32_t));
    rxdata_ptr = (int16_t *)tmp_dft_in;

  } else {

    // use dft input from RX buffer directly
    rxdata_ptr = (int16_t *)&rxdata[rxdata_offset - sample_offset];

  }

  dft(dftsize,
      rxdata_ptr,
      (int16_t *)&rxdataF[symbol * frame_parms->ofdm_symbol_size],
      1);

  return 0;
}

void apply_nr_rotation_symbol_RX(const NR_DL_FRAME_PARMS *frame_parms,
                                 c16_t *rxdataF,
                                 const c16_t *rot,
                                 int nb_rb,
                                 int slot,
                                 int symbol)
{
  const int symb_offset = (slot % frame_parms->slots_per_subframe) * frame_parms->symbols_per_slot;

  c16_t rot2 = rot[symbol + symb_offset];
  rot2.i = -rot2.i;
  LOG_D(PHY,"slot %d, symb_offset %d rotating by %d.%d\n", slot, symb_offset, rot2.r, rot2.i);
  const c16_t *shift_rot = frame_parms->timeshift_symbol_rotation;
  c16_t *this_symbol = rxdataF;

  if (nb_rb & 1) {
    rotate_cpx_vector(this_symbol, &rot2, this_symbol, (nb_rb + 1) * 6, 15);
    rotate_cpx_vector(this_symbol + frame_parms->first_carrier_offset - 6,
                      &rot2,
                      this_symbol + frame_parms->first_carrier_offset - 6,
                      (nb_rb + 1) * 6,
                      15);
    mult_cpx_vector(this_symbol, shift_rot, this_symbol, (nb_rb + 1) * 6, 15);
    mult_cpx_vector(this_symbol + frame_parms->first_carrier_offset - 6,
                    shift_rot + frame_parms->first_carrier_offset - 6,
                    this_symbol + frame_parms->first_carrier_offset - 6,
                    (nb_rb + 1) * 6,
                    15);
  } else {
    rotate_cpx_vector(this_symbol, &rot2, this_symbol, nb_rb * 6, 15);
    rotate_cpx_vector(this_symbol + frame_parms->first_carrier_offset,
                      &rot2,
                      this_symbol + frame_parms->first_carrier_offset,
                      nb_rb * 6,
                      15);
    mult_cpx_vector(this_symbol, shift_rot, this_symbol, nb_rb * 6, 15);
    mult_cpx_vector(this_symbol + frame_parms->first_carrier_offset,
                    shift_rot + frame_parms->first_carrier_offset,
                    this_symbol + frame_parms->first_carrier_offset,
                    nb_rb * 6,
                    15);
  }
}
