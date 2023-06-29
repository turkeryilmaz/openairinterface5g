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

/*! \file PHY/NR_UE_TRANSPORT/nr_slsch.c
* \brief Top-level routines for transmission of the PUSCH TS 38.211 v 15.4.0
* \author Khalid Ahmed
* \date 2019
* \version 0.1
* \company Fraunhofer IIS
* \email: khalid.ahmed@iis.fraunhofer.de
* \note
* \warning
*/
#include <time.h>
#include <stdint.h>
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
//#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_common.h"
#include "common/utils/assertions.h"
#include "common/utils/nr/nr_common.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_UE_TRANSPORT/nr_sch_dmrs_sl.h"
#include "PHY/defs_nr_common.h"
#include "PHY/TOOLS/tools_defs.h"
#include "executables/nr-softmodem.h"
#include "executables/softmodem-common.h"
#include "PHY/NR_REFSIG/ul_ref_seq_nr.h"
#include <openair2/UTIL/OPT/opt.h>
//#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_UE_ESTIMATION/nr_estimation.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "nr_transport_proto_ue.h"


//#define DEBUG_PSSCH_MAPPING
//#define DEBUG_MAC_PDU
//#define DEBUG_DFT_IDFT

//extern int32_t uplink_counter;

void nr_pusch_codeword_scrambling_sl(uint8_t *in,
                                     uint32_t size,
                                     uint32_t SCI2_bits,
                                     uint16_t Nid,
                                     uint32_t* out)
{
  uint8_t reset = 1, c, j = 0;
  uint32_t x1, x2, s = 0;
  int m_ij = 0;
  x2 = (Nid << 15) + 1010; // 1010 is following the spec. 38.211, 8.3.1.1
  for (int i = 0; i < size; i++) {
    const uint8_t b_idx = i & 0x1f;
    if (b_idx == 0) {
      s = lte_gold_generic(&x1, &x2, reset);
      reset = 0;
      if (i)
        out++;
    }
    if (in[i] == NR_PSSCH_x) {
      *out ^= ((*out >> (b_idx - 2)) & 1) << b_idx;
      j++;
    } else {
      m_ij =  (i < SCI2_bits) ? j : SCI2_bits;
      c = (uint8_t)((s >> ((i - m_ij) % 32)) & 1);
      *out ^= ((in[i] + c) & 1) << b_idx;
    }
    //LOG_I(NR_PHY, "i %d b_idx %d in %d s 0x%08x out 0x%08x\n", i, b_idx, in[i], s, *out);
  }
}

void nr_slsch_layer_demapping(int16_t *llr_cw,
                              uint8_t Nl,
                              uint8_t mod_order,
                              uint32_t length,
                              int16_t **llr_layers)
{

  switch (Nl) {
    case 1:
      memcpy((void*)llr_cw, (void*)llr_layers[0], (length) * sizeof(int16_t));
      break;
    case 2:
    case 3:
    case 4:
      for (int i = 0; i < (length / Nl / mod_order); i++) {
        for (int l = 0; l < Nl; l++) {
          for (int m = 0; m < mod_order; m++) {
            llr_cw[i * Nl * mod_order + l * mod_order + m] = llr_layers[l][i * mod_order + m];
          }
        }
      }
      break;
  default:
  AssertFatal(0, "Not supported number of layers %d\n", Nl);
  }
}

void nr_pssch_data_control_multiplexing(uint8_t *in_slssh,
                                        uint8_t *in_sci2,
                                        uint32_t slssh_bits,
                                        uint32_t SCI2_bits,
                                        uint8_t Nl,
                                        uint8_t Q_SCI2,
                                        uint8_t* out)
{
  if (Nl == 1) {
    memcpy(out, in_sci2, SCI2_bits);
    memcpy(out + SCI2_bits, in_slssh, slssh_bits);
  } else if (Nl == 2) {
    uint32_t  M = SCI2_bits / Q_SCI2;
    uint8_t m = 0;
    for (int i = 0; i < M; i++) {
      for (int v = 0; v < Nl; v++) {
        for (int q = 0; q < Q_SCI2; q++) {
          if(v == 0)
            out[m] = in_sci2[i * Q_SCI2 + q];
          else
            out[m] = NR_PSSCH_x;
          m = m + 1;
        }
      }
    }
    memcpy(out + SCI2_bits * Nl, in_slssh, slssh_bits);
  }
}

void nr_ue_slsch_tx_procedures(PHY_VARS_NR_UE *txUE,
                               unsigned char harq_pid,
                               uint32_t frame,
                               uint8_t slot) {

  LOG_D(NR_PHY, "nr_ue_slsch_tx_procedures hard_id %d %d.%d\n", harq_pid, frame, slot);

  uint8_t nb_dmrs_re_per_rb;
  NR_DL_FRAME_PARMS *frame_parms = &txUE->frame_parms;
  c16_t **txdataF = txUE->common_vars.txdataF;

  NR_UE_ULSCH_t *slsch_ue = txUE->slsch[0];
  NR_UL_UE_HARQ_t *harq_process_ul_ue = slsch_ue->harq_processes[harq_pid];
  nfapi_nr_ue_pssch_pdu_t *pssch_pdu = &harq_process_ul_ue->pssch_pdu;

  uint8_t number_of_symbols = pssch_pdu->nr_of_symbols;
  uint16_t nb_rb            = pssch_pdu->rb_size;
  uint8_t Nl                = pssch_pdu->nrOfLayers;
  uint8_t mod_order         = pssch_pdu->qam_mod_order;
  uint8_t cdm_grps_no_data  = pssch_pdu->num_dmrs_cdm_grps_no_data;
  uint16_t dmrs_pos         = pssch_pdu->sl_dmrs_symb_pos;

  uint16_t length_dmrs = get_num_dmrs(dmrs_pos);
  nb_dmrs_re_per_rb = 6 * cdm_grps_no_data;

  /////////////////////////SLSCH data and SCI2 encoding/////////////////////////
  unsigned int G_slsch_bits = nr_get_G(nb_rb, number_of_symbols,
                            nb_dmrs_re_per_rb, length_dmrs, mod_order, Nl);

  if (nr_slsch_encoding(txUE, slsch_ue, frame_parms, harq_pid, G_slsch_bits) == -1)
    return;
  unsigned int G_SCI2_bits = harq_process_ul_ue->B_sci2;

  //////////////////SLSCH data and control multiplexing//////////////
  uint32_t M_SCI2_bits = G_SCI2_bits * Nl;
  uint32_t M_data_bits = G_slsch_bits;
  uint8_t  SCI2_mod_order = 2;

  nr_pssch_data_control_multiplexing(harq_process_ul_ue->f,
                                     harq_process_ul_ue->f_sci2,
                                     G_slsch_bits,
                                     harq_process_ul_ue->B_sci2,
                                     Nl,
                                     SCI2_mod_order,
                                     harq_process_ul_ue->f_multiplexed);

  /////////////////////////SLSCH scrambling/////////////////////////
  uint16_t Nidx = slsch_ue->Nidx;
  uint32_t scrambled_output[(harq_process_ul_ue->B_multiplexed >> 5) + 1];
  memset(scrambled_output, 0, ((harq_process_ul_ue->B_multiplexed >> 5) + 1) * sizeof(uint32_t));

  nr_pusch_codeword_scrambling_sl(harq_process_ul_ue->f_multiplexed,
                                  harq_process_ul_ue->B_multiplexed,
                                  M_SCI2_bits,
                                  Nidx,
                                  scrambled_output);
  #ifdef DEBUG_PSSCH_MAPPING
    char filename[40];
    sprintf(filename,"scramble_output.m");
    LOG_M(filename,"scramble_output",&scrambled_output,(harq_process_ul_ue->B_multiplexed >> 5) + 1, 1, 13);
  #endif
  /////////////////////////SLSCH modulation/////////////////////////

  int max_num_re = Nl * number_of_symbols * nb_rb * NR_NB_SC_PER_RB;
  int32_t d_mod[max_num_re] __attribute__ ((aligned(16)));

  // modulating for the 2nd-stage SCI bits
  nr_modulation(scrambled_output, // assume one codeword for the moment
                M_SCI2_bits,
                SCI2_mod_order,
                (int16_t *)d_mod);

  // modulating SL-SCH bits
  nr_modulation(scrambled_output + (M_SCI2_bits >> 5), // assume one codeword for the moment
                G_slsch_bits,
                mod_order,
                (int16_t *)(d_mod + M_SCI2_bits / SCI2_mod_order));

  /////////////////////////SLSCH layer mapping/////////////////////////

  int16_t **tx_layers = (int16_t **)malloc16_clear(Nl * sizeof(int16_t *));
  uint16_t num_sci2_symbs = (M_SCI2_bits << 1) / SCI2_mod_order;
  uint16_t num_data_symbs = (M_data_bits << 1) / mod_order;
  uint32_t num_sum_symbs = (num_sci2_symbs + num_data_symbs) >> 1;
  for (int nl = 0; nl < Nl; nl++)
    tx_layers[nl] = (int16_t *)malloc16_clear((num_sci2_symbs + num_data_symbs) * sizeof(int16_t));

  nr_ue_layer_mapping((int16_t *)d_mod, Nl, num_sum_symbs, tx_layers);

  /////////////////////////DMRS Modulation/////////////////////////

  nr_init_pssch_dmrs(txUE, Nidx);

  uint32_t **pssch_dmrs = txUE->nr_gold_pssch_dmrs[slot];


  ////////////////SLSCH Mapping to virtual resource blocks////////////////

  int16_t** tx_precoding = virtual_resource_mapping(frame_parms, pssch_pdu, G_SCI2_bits, SCI2_mod_order, tx_layers, pssch_dmrs);


  /////////SLSCH Mapping from virtual to physical resource blocks/////////

  physical_resource_mapping(frame_parms, pssch_pdu, tx_precoding, (uint32_t **)txdataF);

  NR_UL_UE_HARQ_t *harq_process_slsch = NULL;
  harq_process_slsch = slsch_ue->harq_processes[harq_pid];
  harq_process_slsch->status = SCH_IDLE;

  for (int nl = 0; nl < Nl; nl++) {
    free_and_zero(tx_layers[nl]);
    free_and_zero(tx_precoding[nl]);
  }
  free_and_zero(tx_layers);
  free_and_zero(tx_precoding);
  ////////////////////////OFDM modulation/////////////////////////////
  nr_ue_pssch_common_procedures(txUE, slot, &txUE->frame_parms, Nl, link_type_sl);
}

int16_t** virtual_resource_mapping(NR_DL_FRAME_PARMS *frame_parms,
                              nfapi_nr_ue_pssch_pdu_t *pssch_pdu,
                              unsigned int G_SCI2_bits,
                              uint8_t  SCI2_mod_order,
                              int16_t **tx_layers,
                              uint32_t **pssch_dmrs
                             ) {
  uint8_t mod_order = pssch_pdu->qam_mod_order;
  int start_symbol  = pssch_pdu->start_symbol_index;
  uint8_t Nl        = pssch_pdu->nrOfLayers;
  uint8_t number_of_symbols = pssch_pdu->nr_of_symbols;
  uint16_t start_rb         = pssch_pdu->rb_start;
  uint16_t bwp_start        = pssch_pdu->bwp_start;
  uint16_t sl_dmrs_symb_pos = pssch_pdu->sl_dmrs_symb_pos;
  uint16_t nb_rb            = pssch_pdu->rb_size;
  uint8_t dmrs_type         = pssch_pdu->dmrs_config_type;
  int16_t Wf[2], Wt[2];
  int sample_offsetF, l_prime[2], delta;
  uint16_t start_sc = frame_parms->first_carrier_offset + (start_rb + bwp_start) * NR_NB_SC_PER_RB;
  uint16_t n_dmrs = (bwp_start + start_rb + nb_rb) * ((dmrs_type == pusch_dmrs_type1) ? 6 : 4);
  int16_t mod_dmrs[n_dmrs << 1] __attribute((aligned(16)));
  if (start_sc >= frame_parms->ofdm_symbol_size)
    start_sc -= frame_parms->ofdm_symbol_size;

  l_prime[0] = 0; // single symbol ap 0
  pssch_pdu->dmrs_ports = (Nl == 2) ? 0b11 : 0b01;
  uint16_t M_SCI2_Layer = G_SCI2_bits / SCI2_mod_order;
  uint16_t nb_re_sci1 = NB_RB_SCI1 * NR_NB_SC_PER_RB; //NB_RB_SCI1 needs to be from parameter.

  int encoded_length = frame_parms->N_RB_SL * 14 * NR_NB_SC_PER_RB * mod_order * Nl;
  int16_t **tx_precoding = (int16_t **)malloc16_clear(Nl * sizeof(int16_t *));
  for (int nl = 0; nl < Nl; nl++)
    tx_precoding[nl] = (int16_t *)malloc16_clear((encoded_length << 1) * sizeof(int16_t));

  for (int nl = 0; nl < Nl; nl++) {
    uint8_t k_prime = 0;
    uint16_t m = M_SCI2_Layer;
    uint16_t m0 = 0;
    int dmrs_port = get_dmrs_port(nl, pssch_pdu->dmrs_ports);
    // DMRS params for this dmrs port
    get_Wt_sl(Wt, dmrs_port);
    get_Wf_sl(Wf, dmrs_port);
    delta = get_delta_sl(dmrs_port);

    for (int l = start_symbol; l < start_symbol + number_of_symbols; l++) {
      uint16_t k = (1 <= l && l <= 3) ? start_sc + nb_re_sci1 : start_sc;
      uint16_t n = 0;
      uint8_t is_dmrs_sym = 0;
      uint16_t dmrs_idx = 0;

      if ((sl_dmrs_symb_pos >> l) & 0x01) {
        is_dmrs_sym = 1;

        if (pssch_pdu->transform_precoding == transformPrecoder_disabled) {
            dmrs_idx = (bwp_start + start_rb) * 6;
          // TODO: performance improvement, we can skip the modulation of DMRS symbols outside the bandwidth part
          // Perform this on gold sequence, not required when SC FDMA operation is done,
          LOG_D(NR_PHY, "DMRS in symbol %d\n", l);
          nr_modulation(pssch_dmrs[l], n_dmrs * 2, DMRS_MOD_ORDER, mod_dmrs); // currently only codeword 0 is modulated. Qm = 2 as DMRS is QPSK modulated

          #ifdef DEBUG_PSSCH_MAPPING
          char filename[40];
          if (l == 4){
            sprintf(filename, "tx_dmrs_output_4.m");
            LOG_M(filename, "tx_dmrs_output", mod_dmrs, 1200, 1, 3);
          }
          #endif
        } else {
            dmrs_idx = 0;
        }
      }

      for (int i = k; i < nb_rb * NR_NB_SC_PER_RB; i++) {
        uint8_t is_dmrs = 0;

        sample_offsetF = l * frame_parms->ofdm_symbol_size + k;

        if (is_dmrs_sym) {
          if (k == ((start_sc + get_dmrs_freq_idx_ul(n, k_prime, delta, dmrs_type)) % frame_parms->ofdm_symbol_size))
            is_dmrs = 1;
        }

        if (is_dmrs == 1) {
          ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1] = (Wt[l_prime[0]] * Wf[k_prime] * AMP * mod_dmrs[dmrs_idx << 1]) >> 15;
          ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = (Wt[l_prime[0]] * Wf[k_prime] * AMP * mod_dmrs[(dmrs_idx << 1) + 1]) >> 15;

#ifdef DEBUG_PSSCH_MAPPING
          printf("DMRS: Layer: %d\t, dmrs_idx %d\t l %d \t k %d \t k_prime %d \t n %d \t dmrs: %d %d\n",
                 nl, dmrs_idx, l, k, k_prime, n, ((int16_t*)tx_precoding[nl])[(sample_offsetF) << 1],
                 ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1]);
#endif

          dmrs_idx++;
          k_prime++;
          k_prime &= 1;
          n += (k_prime) ? 0 : 1;

        } else if (!is_dmrs_sym) {
          ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1]       = ((int16_t *)tx_layers[nl])[m << 1];
          ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = ((int16_t *)tx_layers[nl])[(m << 1) + 1];

#ifdef DEBUG_PSSCH_MAPPING
          printf("DATA: layer %d\t m %d\t l %d \t k %d \t tx_precoding: %d %d\n",
                 nl, m, l, k, ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1],
                 ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1]);
#endif
          m++;
        } else if ((is_dmrs_sym) && (is_dmrs != 1)) {
          if (m0 < M_SCI2_Layer) {
            ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1]       = ((int16_t *)tx_layers[nl])[m0 << 1];
            ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = ((int16_t *)tx_layers[nl])[(m0 << 1) + 1];
            m0++;
          } else {
            ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1]       = ((int16_t *)tx_layers[nl])[m << 1];
            ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = ((int16_t *)tx_layers[nl])[(m << 1) + 1];
            m++;
          }
        } else {
          ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1]       = 0;
          ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = 0;
        }
        if (++k >= frame_parms->ofdm_symbol_size)
          k -= frame_parms->ofdm_symbol_size;
      } //for (i = 0; i < nb_rb * NR_NB_SC_PER_RB; i++)
    } //for (l = start_symbol; l < start_symbol + number_of_symbols; l++)
  } //for (nl = 0; nl < Nl; nl++)
  return tx_precoding;
}

void physical_resource_mapping(NR_DL_FRAME_PARMS *frame_parms,
                               nfapi_nr_ue_pssch_pdu_t *pssch_pdu,
                               int16_t** tx_precoding,
                               uint32_t **txdataF) {
  uint16_t nb_re_sci1 = NB_RB_SCI1 * NR_NB_SC_PER_RB; //NB_RB_SCI1 needs to be from parameter.
  uint16_t start_rb   = pssch_pdu->rb_start;
  uint16_t bwp_start  = pssch_pdu->bwp_start;
  uint16_t nb_rb      = pssch_pdu->rb_size;
  int start_symbol    = pssch_pdu->start_symbol_index;
  uint8_t number_of_symbols = pssch_pdu->nr_of_symbols;
  uint16_t start_sc = frame_parms->first_carrier_offset + (start_rb + bwp_start) * NR_NB_SC_PER_RB;
  for (int ap = 0; ap < frame_parms->nb_antennas_tx; ap++) {
    // Copying symbol 1 to symbol 0 (AGC)
    memcpy(&txdataF[ap][start_sc],
          &tx_precoding[ap][2 * (frame_parms->ofdm_symbol_size + start_sc)],
          NR_NB_SC_PER_RB * sizeof(int32_t));

    for (int l = start_symbol; l < start_symbol + number_of_symbols; l++) {
      uint16_t k;
      if (1 <= l && l <= 3) { // Assumption there are three SLCCH symbols
        k = start_sc + nb_re_sci1;
      } else {
        k = start_sc;
      }
      for (int rb = 0; rb < nb_rb; rb++) {
        //get pmi info
        uint8_t pmi = pssch_pdu->Tpmi;
        if (pmi == 0) {//unitary Precoding
          if (k + NR_NB_SC_PER_RB <= frame_parms->ofdm_symbol_size) { // RB does not cross DC
            if (ap < pssch_pdu->nrOfLayers) {
              if (1 <= l && l <= 3)
                memcpy(&txdataF[ap][l * frame_parms->ofdm_symbol_size  + k],
                      &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size + k)],
                      NR_NB_SC_PER_RB * sizeof(int32_t));
              else
                memcpy(&txdataF[ap][l * frame_parms->ofdm_symbol_size  + k],
                      &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size + k)],
                      NR_NB_SC_PER_RB * sizeof(int32_t));
            } else {
                memset(&txdataF[ap][l * frame_parms->ofdm_symbol_size + k],
                      0,
                      NR_NB_SC_PER_RB * sizeof(int32_t));
            }
          } else { // RB does cross DC
            int neg_length = frame_parms->ofdm_symbol_size - k;
            int pos_length = NR_NB_SC_PER_RB - neg_length;
            if (ap < pssch_pdu->nrOfLayers) {
              if (1 <= l && l <= 3)
                memcpy(&txdataF[ap][l * frame_parms->ofdm_symbol_size + k],
                      &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size + k)],
                      neg_length * sizeof(int32_t));
              else
                memcpy(&txdataF[ap][l * frame_parms->ofdm_symbol_size + k],
                      &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size + k)],
                      neg_length * sizeof(int32_t));
              memcpy(&txdataF[ap][l * frame_parms->ofdm_symbol_size],
                    &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size)],
                    pos_length * sizeof(int32_t));
            } else {
              memset(&txdataF[ap][l * frame_parms->ofdm_symbol_size + k],
                    0,
                    neg_length * sizeof(int32_t));
              memset(&txdataF[ap][l * frame_parms->ofdm_symbol_size],
                    0,
                    pos_length * sizeof(int32_t));
            }
          }
          k += NR_NB_SC_PER_RB;
          if (k >= frame_parms->ofdm_symbol_size) {
            k -= frame_parms->ofdm_symbol_size;
          }
        }
      } //RB loop
    } // symbol loop
  } // port loop
}

uint8_t nr_ue_pssch_common_procedures(PHY_VARS_NR_UE *UE,
                                      uint8_t slot,
                                      NR_DL_FRAME_PARMS *frame_parms,
                                      uint8_t n_antenna_ports,
                                      int link_type) {

  c16_t **txdata = UE->common_vars.txdata;
  c16_t **txdataF = UE->common_vars.txdataF;
  int tx_offset = frame_parms->get_samples_slot_timestamp(slot, frame_parms, 0);
  int symb_offset = (slot % frame_parms->slots_per_subframe) * frame_parms->symbols_per_slot;

  for(int ap = 0; ap < n_antenna_ports; ap++) {
    for (int s = 0; s < NR_NUMBER_OF_SYMBOLS_PER_SLOT; s++){
      c16_t *this_symbol = &txdataF[ap][frame_parms->ofdm_symbol_size * s];
      c16_t rot = frame_parms->symbol_rotation[link_type][s + symb_offset];
      LOG_D(NR_PHY, "offset is %d rotating txdataF symbol %d (%d) => (%d.%d)\n", tx_offset, s, s + symb_offset, rot.r, rot.i);
      if (frame_parms->N_RB_SL & 1) {
        rotate_cpx_vector(this_symbol, &rot, this_symbol,
                          (frame_parms->N_RB_SL + 1) * 6, 15);
        rotate_cpx_vector(this_symbol + frame_parms->first_carrier_offset - 6,
                          &rot,
                          this_symbol + frame_parms->first_carrier_offset - 6,
                          (frame_parms->N_RB_SL + 1) * 6, 15);
      } else {
        rotate_cpx_vector(this_symbol, &rot, this_symbol,
                          frame_parms->N_RB_SL * 6, 15);
        rotate_cpx_vector(this_symbol + frame_parms->first_carrier_offset,
                          &rot,
                          this_symbol + frame_parms->first_carrier_offset,
                          frame_parms->N_RB_SL * 6, 15);
      }
    }
  }

  for (int ap = 0; ap < n_antenna_ports; ap++) {
    if (frame_parms->Ncp == 1) { // extended cyclic prefix
      PHY_ofdm_mod((int *)txdataF[ap],
                   (int *)&txdata[ap][tx_offset],
                   frame_parms->ofdm_symbol_size,
                   NR_NUMBER_OF_SYMBOLS_PER_SLOT,
                   frame_parms->nb_prefix_samples,
                   CYCLIC_PREFIX);
    } else { // normal cyclic prefix
      nr_normal_prefix_mod(txdataF[ap],
                           &txdata[ap][tx_offset],
                           NR_NUMBER_OF_SYMBOLS_PER_SLOT,
                           frame_parms,
                           slot);
    }
  }

  return 0;
}

//----------------------------------------------------------------------------------------------
// QPSK
//----------------------------------------------------------------------------------------------
void nr_slsch_qpsk_llr(int32_t *rxdataF_comp,
                      int16_t  *slsch_llr,
                      uint32_t nb_re,
                      uint8_t  symbol)
{
  c16_t *rxF   = (c16_t *)rxdataF_comp;
  c16_t *llr32 = (c16_t *)slsch_llr;

  if (!llr32) {
    LOG_E(PHY,"nr_slsch_qpsk_llr: llr is null, symbol %d, llr32 = %p\n",symbol, llr32);
  }
  for (int i = 0; i < nb_re; i++) {
    //*llr32 = *rxF;
    llr32->r = rxF->r >> 3;
    llr32->i = rxF->i >> 3;
    rxF++;
    llr32++;
  }
}

//----------------------------------------------------------------------------------------------
// 16-QAM
//----------------------------------------------------------------------------------------------

void nr_slsch_16qam_llr(int32_t *rxdataF_comp,
                        int32_t *sl_ch_mag,
                        int16_t  *slsch_llr,
                        uint32_t nb_rb,
                        uint32_t nb_re,
                        uint8_t  symbol)

{

#if defined(__x86_64__) || defined(__i386__)
  __m256i *rxF = (__m256i*)rxdataF_comp;
  __m256i *ch_mag;
  __m256i llr256[2];
  register __m256i xmm0;
  uint32_t *llr32;
#elif defined(__arm__) || defined(__aarch64__)
  int16x8_t *rxF = (int16x8_t*)&rxdataF_comp;
  int16x8_t *ch_mag;
  int16x8_t xmm0;
  int16_t *llr16;
#endif


  int i;

  int off = ((nb_rb&1) == 1)? 4:0;

#if defined(__x86_64__) || defined(__i386__)
    llr32 = (uint32_t*)slsch_llr;
#elif defined(__arm__) || defined(__aarch64__)
    llr16 = (int16_t*)slsch_llr;
#endif

#if defined(__x86_64__) || defined(__i386__)
    ch_mag = (__m256i*)&sl_ch_mag[(symbol*(off+(nb_rb*12)))];
#elif defined(__arm__) || defined(__aarch64__)
  ch_mag = (int16x8_t*)&sl_ch_mag[(symbol*nb_rb*12)];
#endif
  unsigned char len_mod8 = nb_re&7;
  nb_re >>= 3;  // length in quad words (4 REs)
  nb_re += (len_mod8 == 0 ? 0 : 1);

  for (i=0; i<nb_re; i++) {
#if defined(__x86_64__) || defined(__i386)
    xmm0 = simde_mm256_abs_epi16(rxF[i]); // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
    xmm0 = simde_mm256_subs_epi16(ch_mag[i],xmm0); // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
 
    llr256[0] = simde_mm256_unpacklo_epi32(rxF[i],xmm0); // llr128[0] contains the llrs of the 1st,2nd,5th and 6th REs
    llr256[1] = simde_mm256_unpackhi_epi32(rxF[i],xmm0); // llr128[1] contains the llrs of the 3rd, 4th, 7th and 8th REs
    
    // 1st RE
    llr32[0] = simde_mm256_extract_epi32(llr256[0],0); // llr32[0] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[1] = simde_mm256_extract_epi32(llr256[0],1); // llr32[1] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 2nd RE
    llr32[2] = simde_mm256_extract_epi32(llr256[0],2); // llr32[2] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[3] = simde_mm256_extract_epi32(llr256[0],3); // llr32[3] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 3rd RE
    llr32[4] = simde_mm256_extract_epi32(llr256[1],0); // llr32[4] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[5] = simde_mm256_extract_epi32(llr256[1],1); // llr32[5] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 4th RE
    llr32[6] = simde_mm256_extract_epi32(llr256[1],2); // llr32[6] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[7] = simde_mm256_extract_epi32(llr256[1],3); // llr32[7] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 5th RE
    llr32[8] = simde_mm256_extract_epi32(llr256[0],4); // llr32[8] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[9] = simde_mm256_extract_epi32(llr256[0],5); // llr32[9] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 6th RE
    llr32[10] = simde_mm256_extract_epi32(llr256[0],6); // llr32[10] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[11] = simde_mm256_extract_epi32(llr256[0],7); // llr32[11] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 7th RE
    llr32[12] = simde_mm256_extract_epi32(llr256[1],4); // llr32[12] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[13] = simde_mm256_extract_epi32(llr256[1],5); // llr32[13] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    // 8th RE
    llr32[14] = simde_mm256_extract_epi32(llr256[1],6); // llr32[14] low 16 bits-> y_R        , high 16 bits-> y_I
    llr32[15] = simde_mm256_extract_epi32(llr256[1],7); // llr32[15] low 16 bits-> |h|-|y_R|^2, high 16 bits-> |h|-|y_I|^2

    llr32+=16;
#elif defined(__arm__) || defined(__aarch64__)
    xmm0 = vabsq_s16(rxF[i]);
    xmm0 = vqsubq_s16((*(__m128i*)&ones[0]),xmm0);

    llr16[0]  = vgetq_lane_s16(rxF[i],0);
    llr16[1]  = vgetq_lane_s16(rxF[i],1);
    llr16[2]  = vgetq_lane_s16(xmm0,0);
    llr16[3]  = vgetq_lane_s16(xmm0,1);
    llr16[4]  = vgetq_lane_s16(rxF[i],2);
    llr16[5]  = vgetq_lane_s16(rxF[i],3);
    llr16[6]  = vgetq_lane_s16(xmm0,2);
    llr16[7]  = vgetq_lane_s16(xmm0,3);
    llr16[8]  = vgetq_lane_s16(rxF[i],4);
    llr16[9]  = vgetq_lane_s16(rxF[i],5);
    llr16[10] = vgetq_lane_s16(xmm0,4);
    llr16[11] = vgetq_lane_s16(xmm0,5);
    llr16[12] = vgetq_lane_s16(rxF[i],6);
    llr16[13] = vgetq_lane_s16(rxF[i],6);
    llr16[14] = vgetq_lane_s16(xmm0,7);
    llr16[15] = vgetq_lane_s16(xmm0,7);
    llr16+=16;
#endif

  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif
}

//----------------------------------------------------------------------------------------------
// 64-QAM
//----------------------------------------------------------------------------------------------

void nr_slsch_64qam_llr(int32_t *rxdataF_comp,
                        int32_t *sl_ch_mag,
                        int32_t *sl_ch_magb,
                        int16_t  *slsch_llr,
                        uint32_t nb_rb,
                        uint32_t nb_re,
                        uint8_t  symbol)
{
  int off = ((nb_rb&1) == 1)? 4:0;

#if defined(__x86_64__) || defined(__i386__)
  __m256i *rxF = (__m256i*)rxdataF_comp;
  __m256i *ch_mag,*ch_magb;
  register __m256i xmm0,xmm1,xmm2;
#elif defined(__arm__) || defined(__aarch64__)
  int16x8_t *rxF = (int16x8_t*)&rxdataF_comp;
  int16x8_t *ch_mag,*ch_magb; // [hna] This should be uncommented once channel estimation is implemented
  int16x8_t xmm0,xmm1,xmm2;
#endif

  int i;

#if defined(__x86_64__) || defined(__i386__)
  ch_mag = (__m256i*)&sl_ch_mag[(symbol*(off+(nb_rb*12)))];
  ch_magb = (__m256i*)&sl_ch_magb[(symbol*(off+(nb_rb*12)))];
#elif defined(__arm__) || defined(__aarch64__)
  ch_mag = (int16x8_t*)&sl_ch_mag[(symbol*nb_rb*12)];
  ch_magb = (int16x8_t*)&sl_ch_magb[(symbol*nb_rb*12)];
#endif

  int len_mod8 = nb_re&7;
  nb_re    = nb_re>>3;  // length in quad words (4 REs)
  nb_re   += ((len_mod8 == 0) ? 0 : 1);

  for (i=0; i<nb_re; i++) {
    xmm0 = rxF[i];
#if defined(__x86_64__) || defined(__i386__)
    xmm1 = simde_mm256_abs_epi16(xmm0);
    xmm1 = simde_mm256_subs_epi16(ch_mag[i],xmm1);
    xmm2 = simde_mm256_abs_epi16(xmm1);
    xmm2 = simde_mm256_subs_epi16(ch_magb[i],xmm2);
#elif defined(__arm__) || defined(__aarch64__)
    xmm1 = vabsq_s16(xmm0);
    xmm1 = vsubq_s16(ch_mag[i],xmm1);
    xmm2 = vabsq_s16(xmm1);
    xmm2 = vsubq_s16(ch_magb[i],xmm2);
#endif
    
    // ---------------------------------------
    // 1st RE
    // ---------------------------------------
#if defined(__x86_64__) || defined(__i386__)
    slsch_llr[0] = simde_mm256_extract_epi16(xmm0,0);
    slsch_llr[1] = simde_mm256_extract_epi16(xmm0,1);
    slsch_llr[2] = simde_mm256_extract_epi16(xmm1,0);
    slsch_llr[3] = simde_mm256_extract_epi16(xmm1,1);
    slsch_llr[4] = simde_mm256_extract_epi16(xmm2,0);
    slsch_llr[5] = simde_mm256_extract_epi16(xmm2,1);
#elif defined(__arm__) || defined(__aarch64__)
    slsch_llr[0] = vgetq_lane_s16(xmm0,0);
    slsch_llr[1] = vgetq_lane_s16(xmm0,1);
    slsch_llr[2] = vgetq_lane_s16(xmm1,0);
    slsch_llr[3] = vgetq_lane_s16(xmm1,1);
    slsch_llr[4] = vgetq_lane_s16(xmm2,0);
    slsch_llr[5] = vgetq_lane_s16(xmm2,1);
#endif
    // ---------------------------------------

    slsch_llr+=6;
    
    // ---------------------------------------
    // 2nd RE
    // ---------------------------------------
#if defined(__x86_64__) || defined(__i386__)
    slsch_llr[0] = simde_mm256_extract_epi16(xmm0,2);
    slsch_llr[1] = simde_mm256_extract_epi16(xmm0,3);
    slsch_llr[2] = simde_mm256_extract_epi16(xmm1,2);
    slsch_llr[3] = simde_mm256_extract_epi16(xmm1,3);
    slsch_llr[4] = simde_mm256_extract_epi16(xmm2,2);
    slsch_llr[5] = simde_mm256_extract_epi16(xmm2,3);
#elif defined(__arm__) || defined(__aarch64__)
    slsch_llr[2] = vgetq_lane_s16(xmm0,2);
    slsch_llr[3] = vgetq_lane_s16(xmm0,3);
    slsch_llr[2] = vgetq_lane_s16(xmm1,2);
    slsch_llr[3] = vgetq_lane_s16(xmm1,3);
    slsch_llr[4] = vgetq_lane_s16(xmm2,2);
    slsch_llr[5] = vgetq_lane_s16(xmm2,3);
#endif
    // ---------------------------------------

    slsch_llr+=6;
    
    // ---------------------------------------
    // 3rd RE
    // ---------------------------------------
#if defined(__x86_64__) || defined(__i386__)
    slsch_llr[0] = simde_mm256_extract_epi16(xmm0,4);
    slsch_llr[1] = simde_mm256_extract_epi16(xmm0,5);
    slsch_llr[2] = simde_mm256_extract_epi16(xmm1,4);
    slsch_llr[3] = simde_mm256_extract_epi16(xmm1,5);
    slsch_llr[4] = simde_mm256_extract_epi16(xmm2,4);
    slsch_llr[5] = simde_mm256_extract_epi16(xmm2,5);
#elif defined(__arm__) || defined(__aarch64__)
    slsch_llr[0] = vgetq_lane_s16(xmm0,4);
    slsch_llr[1] = vgetq_lane_s16(xmm0,5);
    slsch_llr[2] = vgetq_lane_s16(xmm1,4);
    slsch_llr[3] = vgetq_lane_s16(xmm1,5);
    slsch_llr[4] = vgetq_lane_s16(xmm2,4);
    slsch_llr[5] = vgetq_lane_s16(xmm2,5);
#endif
    // ---------------------------------------

    slsch_llr+=6;
    
    // ---------------------------------------
    // 4th RE
    // ---------------------------------------
#if defined(__x86_64__) || defined(__i386__)
    slsch_llr[0] = simde_mm256_extract_epi16(xmm0,6);
    slsch_llr[1] = simde_mm256_extract_epi16(xmm0,7);
    slsch_llr[2] = simde_mm256_extract_epi16(xmm1,6);
    slsch_llr[3] = simde_mm256_extract_epi16(xmm1,7);
    slsch_llr[4] = simde_mm256_extract_epi16(xmm2,6);
    slsch_llr[5] = simde_mm256_extract_epi16(xmm2,7);
#elif defined(__arm__) || defined(__aarch64__)
    slsch_llr[0] = vgetq_lane_s16(xmm0,6);
    slsch_llr[1] = vgetq_lane_s16(xmm0,7);
    slsch_llr[2] = vgetq_lane_s16(xmm1,6);
    slsch_llr[3] = vgetq_lane_s16(xmm1,7);
    slsch_llr[4] = vgetq_lane_s16(xmm2,6);
    slsch_llr[5] = vgetq_lane_s16(xmm2,7);
#endif
    // ---------------------------------------

    slsch_llr+=6;
    slsch_llr[0] = simde_mm256_extract_epi16(xmm0,8);
    slsch_llr[1] = simde_mm256_extract_epi16(xmm0,9);
    slsch_llr[2] = simde_mm256_extract_epi16(xmm1,8);
    slsch_llr[3] = simde_mm256_extract_epi16(xmm1,9);
    slsch_llr[4] = simde_mm256_extract_epi16(xmm2,8);
    slsch_llr[5] = simde_mm256_extract_epi16(xmm2,9);

    slsch_llr[6] = simde_mm256_extract_epi16(xmm0,10);
    slsch_llr[7] = simde_mm256_extract_epi16(xmm0,11);
    slsch_llr[8] = simde_mm256_extract_epi16(xmm1,10);
    slsch_llr[9] = simde_mm256_extract_epi16(xmm1,11);
    slsch_llr[10] = simde_mm256_extract_epi16(xmm2,10);
    slsch_llr[11] = simde_mm256_extract_epi16(xmm2,11);

    slsch_llr[12] = simde_mm256_extract_epi16(xmm0,12);
    slsch_llr[13] = simde_mm256_extract_epi16(xmm0,13);
    slsch_llr[14] = simde_mm256_extract_epi16(xmm1,12);
    slsch_llr[15] = simde_mm256_extract_epi16(xmm1,13);
    slsch_llr[16] = simde_mm256_extract_epi16(xmm2,12);
    slsch_llr[17] = simde_mm256_extract_epi16(xmm2,13);

    slsch_llr[18] = simde_mm256_extract_epi16(xmm0,14);
    slsch_llr[19] = simde_mm256_extract_epi16(xmm0,15);
    slsch_llr[20] = simde_mm256_extract_epi16(xmm1,14);
    slsch_llr[21] = simde_mm256_extract_epi16(xmm1,15);
    slsch_llr[22] = simde_mm256_extract_epi16(xmm2,14);
    slsch_llr[23] = simde_mm256_extract_epi16(xmm2,15);

    slsch_llr+=24;
  }

#if defined(__x86_64__) || defined(__i386__)
  _mm_empty();
  _m_empty();
#endif
}

void nr_slsch_256qam_llr(int32_t *rxdataF_comp,
                         int32_t *sl_ch_mag,
                         int32_t *sl_ch_magb,
	                 int32_t *sl_ch_magc,
	                 int16_t  *slsch_llr,
	                 uint32_t nb_rb,
	                 uint32_t nb_re,
	                 uint8_t  symbol)
{
  int off = ((nb_rb&1) == 1)? 4:0;

  simde__m256i *rxF = (simde__m256i*)rxdataF_comp;
  simde__m256i *ch_mag,*ch_magb,*ch_magc;
  register simde__m256i xmm0,xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;
  simde__m256i *llr256=(simde__m256i*)slsch_llr;

  ch_mag  = (simde__m256i*)&sl_ch_mag[(symbol*(off+(nb_rb*12)))];
  ch_magb = (simde__m256i*)&sl_ch_magb[(symbol*(off+(nb_rb*12)))];
  ch_magc = (simde__m256i*)&sl_ch_magc[(symbol*(off+(nb_rb*12)))];
  int len_mod8 = nb_re&7;
  int nb_re256    = nb_re>>3;  // length in 256-bit words (8 REs)

  for (int i=0; i<nb_re256; i++) {
       xmm0 = simde_mm256_abs_epi16(rxF[i]); // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
       xmm0 = simde_mm256_subs_epi16(ch_mag[i],xmm0); // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
      //  xmmtmpD2 contains 16 LLRs
       xmm1 = simde_mm256_abs_epi16(xmm0);
       xmm1 = simde_mm256_subs_epi16(ch_magb[i],xmm1); // contains 16 LLRs
       xmm2 = simde_mm256_abs_epi16(xmm1);
       xmm2 = simde_mm256_subs_epi16(ch_magc[i],xmm2); // contains 16 LLRs
        // rxF[i] A0 A1 A2 A3 A4 A5 A6 A7 bits 7,6
        // xmm0   B0 B1 B2 B3 B4 B5 B6 B7 bits 5,4
        // xmm1   C0 C1 C2 C3 C4 C5 C6 C7 bits 3,2
        // xmm2   D0 D1 D2 D3 D4 D5 D6 D7 bits 1,0
       xmm3 = simde_mm256_unpacklo_epi32(rxF[i],xmm0); // A0 B0 A1 B1 A4 B4 A5 B5
       xmm4 = simde_mm256_unpackhi_epi32(rxF[i],xmm0); // A2 B2 A3 B3 A6 B6 A7 B7
       xmm5 = simde_mm256_unpacklo_epi32(xmm1,xmm2);   // C0 D0 C1 D1 C4 D4 C5 D5
       xmm6 = simde_mm256_unpackhi_epi32(xmm1,xmm2);   // C2 D2 C3 D3 C6 D6 C7 D7

       xmm0 = simde_mm256_unpacklo_epi64(xmm3,xmm5); // A0 B0 C0 D0 A4 B4 C4 D4
       xmm1 = simde_mm256_unpackhi_epi64(xmm3,xmm5); // A1 B1 C1 D1 A5 B5 C5 D5
       xmm2 = simde_mm256_unpacklo_epi64(xmm4,xmm6); // A2 B2 C2 D2 A6 B6 C6 D6
       xmm3 = simde_mm256_unpackhi_epi64(xmm4,xmm6); // A3 B3 C3 D3 A7 B7 C7 D7
       llr256[0] = simde_mm256_permute2x128_si256(xmm0, xmm1, 0x20); // A0 B0 C0 D0 A1 B1 C1 D1
       llr256[1] = simde_mm256_permute2x128_si256(xmm2, xmm3, 0x20); // A2 B2 C2 D2 A3 B3 C3 D3
       llr256[2] = simde_mm256_permute2x128_si256(xmm0, xmm1, 0x31); // A4 B4 C4 D4 A5 B5 C5 D5
       llr256[3] = simde_mm256_permute2x128_si256(xmm2, xmm3, 0x31); // A6 B6 C6 D6 A7 B7 C7 D7
       llr256+=4;

  }
  simde__m128i *llr128 = (simde__m128i*)llr256;
  if (len_mod8 >= 4) {
     int nb_re128 = nb_re>>2;
     simde__m128i xmm0,xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;
     simde__m128i *rxF = (simde__m128i*)rxdataF_comp;
     simde__m128i *ch_mag  = (simde__m128i*)&sl_ch_mag[(symbol*(off+(nb_rb*12)))];
     simde__m128i *ch_magb = (simde__m128i*)&sl_ch_magb[(symbol*(off+(nb_rb*12)))];
     simde__m128i *ch_magc = (simde__m128i*)&sl_ch_magc[(symbol*(off+(nb_rb*12)))];

     xmm0 = simde_mm_abs_epi16(rxF[nb_re128-1]); // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
     xmm0 = simde_mm_subs_epi16(ch_mag[nb_re128-1],xmm0); // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
      //  xmmtmpD2 contains 8 LLRs
     xmm1 = simde_mm_abs_epi16(xmm0);
     xmm1 = simde_mm_subs_epi16(ch_magb[nb_re128-1],xmm1); // contains 8 LLRs
     xmm2 = simde_mm_abs_epi16(xmm1);
     xmm2 = simde_mm_subs_epi16(ch_magc[nb_re128-1],xmm2); // contains 8 LLRs
     // rxF[i] A0 A1 A2 A3
     // xmm0   B0 B1 B2 B3
     // xmm1   C0 C1 C2 C3
     // xmm2   D0 D1 D2 D3
     xmm3 = simde_mm_unpacklo_epi32(rxF[nb_re128-1],xmm0); // A0 B0 A1 B1
     xmm4 = simde_mm_unpackhi_epi32(rxF[nb_re128-1],xmm0); // A2 B2 A3 B3
     xmm5 = simde_mm_unpacklo_epi32(xmm1,xmm2);   // C0 D0 C1 D1
     xmm6 = simde_mm_unpackhi_epi32(xmm1,xmm2);   // C2 D2 C3 D3

     llr128[0] = simde_mm_unpacklo_epi64(xmm3,xmm5); // A0 B0 C0 D0
     llr128[1] = simde_mm_unpackhi_epi64(xmm3,xmm5); // A1 B1 C1 D1
     llr128[2] = simde_mm_unpacklo_epi64(xmm4,xmm6); // A2 B2 C2 D2
     llr128[3] = simde_mm_unpackhi_epi64(xmm4,xmm6); // A3 B3 C3 D3
     llr128+=4;
  }
  if (len_mod8 == 6) {
     int nb_re64 = nb_re>>1;
     simde__m64 *llr64 = (simde__m64 *)llr128;
     simde__m64 xmm0,xmm1,xmm2;
     simde__m64 *rxF = (simde__m64*)rxdataF_comp;
     simde__m64 *ch_mag  = (simde__m64*)&sl_ch_mag[(symbol*(off+(nb_rb*12)))];
     simde__m64 *ch_magb = (simde__m64*)&sl_ch_magb[(symbol*(off+(nb_rb*12)))];
     simde__m64 *ch_magc = (simde__m64*)&sl_ch_magc[(symbol*(off+(nb_rb*12)))];

     xmm0 = simde_mm_abs_pi16(rxF[nb_re64-1]); // registers of even index in xmm0-> |y_R|, registers of odd index in xmm0-> |y_I|
     xmm0 = simde_mm_subs_pi16(ch_mag[nb_re-1],xmm0); // registers of even index in xmm0-> |y_R|-|h|^2, registers of odd index in xmm0-> |y_I|-|h|^2
      //  xmmtmpD2 contains 4 LLRs
     xmm1 = simde_mm_abs_pi16(xmm0);
     xmm1 = simde_mm_subs_pi16(ch_magb[nb_re64-1],xmm1); // contains 4 LLRs
     xmm2 = simde_mm_abs_pi16(xmm1);
     xmm2 = simde_mm_subs_pi16(ch_magc[nb_re64-1],xmm2); // contains 4 LLRs
     // rxF[i] A0 A1
     // xmm0   B0 B1
     // xmm1   C0 C1
     // xmm2   D0 D1
     llr64[0] = simde_m_punpckldq(rxF[nb_re64-1],xmm0); // A0 B0
     llr64[2] = simde_m_punpckhdq(rxF[nb_re64-1],xmm0);  // A1 B1
     llr64[1] = simde_m_punpckldq(xmm1,xmm2);         // C0 D0
     llr64[3] = simde_m_punpckhdq(xmm1,xmm2);         // C1 D1
  }

}

void nr_slsch_compute_llr(int32_t *rxdataF_comp,
                          int32_t *sl_ch_mag,
                          int32_t *sl_ch_magb,
                          int16_t *slsch_llr,
                          uint32_t nb_rb,
                          uint32_t nb_re,
                          uint8_t  symbol,
                          uint8_t  mod_order)
{
  switch(mod_order){
    case 2:
      nr_slsch_qpsk_llr(rxdataF_comp,
                        slsch_llr,
                        nb_re,
                        symbol);
      break;
    case 4:
      nr_slsch_16qam_llr(rxdataF_comp,
                         sl_ch_mag,
                         slsch_llr,
                         nb_rb,
                         nb_re,
                         symbol);
      break;
    case 6:
    nr_slsch_64qam_llr(rxdataF_comp,
                       sl_ch_mag,
                       sl_ch_magb,
                       slsch_llr,
                       nb_rb,
                       nb_re,
                       symbol);
      break;
    default:
      AssertFatal(1==0,"nr_slsch_compute_llr: invalid Qm value, symbol = %d, Qm = %d\n",symbol, mod_order);
      break;
  }
}

uint32_t nr_ue_slsch_rx_procedures(PHY_VARS_NR_UE *rxUE,
                            unsigned char harq_pid,
                            uint32_t frame,
                            uint8_t slot,
                            c16_t **rxdata,
                            uint32_t multiplex_input_len,
                            uint32_t Nidx,
                            UE_nr_rxtx_proc_t *proc) {
  int UE_id = 0;
  NR_UE_DLSCH_t *slsch_ue_rx = rxUE->slsch_rx[0][0];
  NR_DL_UE_HARQ_t *slsch_ue_rx_harq = slsch_ue_rx->harq_processes[harq_pid];
  uint16_t nb_rb          = slsch_ue_rx_harq->nb_rb;
  uint16_t bwp_start      = slsch_ue_rx_harq->BWPStart;
  uint16_t pssch_start_rb = slsch_ue_rx_harq->start_rb;
  uint16_t start_sym      = slsch_ue_rx_harq->start_symbol;
  uint8_t nb_symb_sch     = slsch_ue_rx_harq->nb_symbols;
  uint8_t mod_order       = 2; //nr_get_Qm_ul(slsch_ue_rx_harq->mcs, 0);
  uint16_t dmrs_pos       = slsch_ue_rx_harq->dlDmrsSymbPos;
  uint8_t dmrs_config     = slsch_ue_rx_harq->dmrsConfigType;
  uint8_t SCI2_mod_order  = 2;
  uint8_t Nl              = slsch_ue_rx_harq->Nl;
  // TODO: has to be checked if rx has access to these info.
  int nb_re_SCI2 = slsch_ue_rx->harq_processes[0]->B_sci2 / SCI2_mod_order;

  uint8_t nb_re_dmrs;
  if (slsch_ue_rx_harq->dmrsConfigType == NFAPI_NR_DMRS_TYPE1) {
    nb_re_dmrs = 6 * slsch_ue_rx_harq->n_dmrs_cdm_groups;
  } else {
    nb_re_dmrs = 4 * slsch_ue_rx_harq->n_dmrs_cdm_groups;
  }
  uint32_t dmrs_data_re = 12 - nb_re_dmrs;
  uint16_t dmrs_len = get_num_dmrs(slsch_ue_rx_harq->dlDmrsSymbPos);
  uint32_t rx_size_symbol = rxUE->slsch_rx[0][0]->dlsch_config.number_rbs * NR_NB_SC_PER_RB;
  unsigned int G = nr_get_G(nb_rb, nb_symb_sch,
                            nb_re_dmrs, dmrs_len, mod_order,
                            Nl);
  const uint32_t rx_llr_buf_sz = ((G + 15) / 16) * 16;
  const uint32_t nb_codewords = NR_MAX_NB_LAYERS > 4 ? 2 : 1;
  int16_t* llr[2];
  for (int i = 0; i < nb_codewords; i++) {
    llr[i] = (int16_t *)malloc16_clear(rx_llr_buf_sz * sizeof(int16_t));
  }
  int16_t* layer_llr[NR_MAX_NB_LAYERS];
  const uint32_t rx_llr_layer_size = (G + Nl - 1) / Nl;
  for (int i=0; i < NR_MAX_NB_LAYERS; i++)
    layer_llr[i] = (int16_t *)malloc16_clear(rx_llr_layer_size*sizeof(int16_t));
  int16_t **slsch_llr_layers = rxUE->pssch_vars[UE_id]->llr_layers;
  uint16_t num_data_symbs = (G << 1) / mod_order;
  uint32_t M_SCI2_bits = slsch_ue_rx->harq_processes[0]->B_sci2 * Nl;
  uint16_t num_sci2_symbs = (M_SCI2_bits << 1) / SCI2_mod_order;
  uint16_t num_sci2_samples = num_sci2_symbs >> 1;

  int avgs = 0;
  int avg[16];
  int32_t median[16];
  uint32_t sci2_offset = 0;
  uint32_t data_offset = num_sci2_samples;
  uint32_t diff_re_comp;
  const int nl = slsch_ue_rx[0].Nl;
  const uint32_t rxdataF_sz = rxUE->frame_parms.samples_per_slot_wCP;
  const uint32_t pssch_est_size = ((rxUE->frame_parms.symbols_per_slot * rxUE->frame_parms.ofdm_symbol_size + 15) / 16) * 16;
  __attribute__((aligned(32))) int32_t dl_ch_estimates_ext[rxUE->frame_parms.nb_antennas_rx * nl][pssch_est_size];
  memset(dl_ch_estimates_ext, 0, sizeof(int32_t) * rxUE->frame_parms.nb_antennas_rx * nl * pssch_est_size);

  __attribute__((aligned(32))) int32_t dl_ch_mag[nl][rxUE->frame_parms.nb_antennas_rx][rx_size_symbol];
  memset(dl_ch_mag, 0, sizeof(dl_ch_mag));

  __attribute__((aligned(32))) int32_t dl_ch_magb[nl][rxUE->frame_parms.nb_antennas_rx][rx_size_symbol];
  memset(dl_ch_magb, 0, sizeof(dl_ch_magb));

  __attribute__((aligned(32))) int32_t dl_ch_magr[nl][rxUE->frame_parms.nb_antennas_rx][rx_size_symbol];
  memset(dl_ch_magr, 0, sizeof(dl_ch_magr));

  __attribute__ ((aligned(32))) c16_t rxdataF[rxUE->frame_parms.nb_antennas_rx][rxdataF_sz];
  memset(rxdataF, 0, sizeof(rxdataF));

  __attribute__ ((aligned(32))) int32_t rxdataF_comp[nl][rxUE->frame_parms.nb_antennas_rx][rx_size_symbol];
  memset(rxdataF_comp, 0, sizeof(rxdataF_comp));

  __attribute__ ((aligned(32))) c16_t rxdataF_ext[rxUE->frame_parms.nb_antennas_rx][rxdataF_sz];
  memset(rxdataF_ext, 0, sizeof(rxdataF_ext));

  /////////////// Channel Estimation ///////////////////////
  unsigned short port = 0;
  unsigned char nscid = 0; // it is not used for SL, so should be zero
  unsigned short Nid = Nidx%(1<<16);

  for (int sym = start_sym ; sym < (start_sym+nb_symb_sch) ; sym++){
    if (dmrs_pos & (1 << sym)){
      for (uint8_t aatx=0; aatx<Nl; aatx++) {
        port = get_dmrs_port(aatx,slsch_ue_rx_harq->dmrs_ports);//get_dmrs_port(1,slsch_ue_rx_harq->dmrs_ports);
        if (nr_pdsch_channel_estimation(rxUE, proc, 0, port, sym, nscid, Nid, bwp_start, dmrs_config,
                                        rxUE->frame_parms.first_carrier_offset + (bwp_start + pssch_start_rb) * 12,
                                        nb_rb, pssch_est_size, dl_ch_estimates_ext, rxdataF_sz, rxdataF) == -1)
          return 1; // Non-zero return represents a failure in nr_ue_slsch_rx_procedures
      }
    }
  }

  if (rxUE->chest_time == 1) { // averaging time domain channel estimates
    nr_chest_time_domain_avg(&rxUE->frame_parms,
                              rxUE->pssch_vars[UE_id]->sl_ch_estimates,
                              nb_symb_sch,
                              start_sym,
                              dmrs_pos,
                              nb_rb);
  }

  nr_ue_sl_pssch_rsrp_measurements(rxUE, harq_pid, 0, proc);

  //----------------------------------------------------------
  //--------------------- RBs extraction ---------------------
  //----------------------------------------------------------

  int first_symbol_flag = 0;
  uint16_t first_symbol_with_data = start_sym;
  while ((dmrs_data_re == 0) && (dmrs_pos & (1 << first_symbol_with_data))) {
    first_symbol_with_data++;
  }

  for (int sym = start_sym ; sym < start_sym + nb_symb_sch ; sym++){
    uint8_t pilots = (dmrs_pos >> sym) & 1;
    uint16_t nb_re_sci1 = 0;
    if (1 <= sym && sym <= 3) {
      nb_re_sci1 = NR_NB_SC_PER_RB * NB_RB_SCI1;
    }
    uint32_t allocatable_sci2_re = min(nb_re_SCI2, NR_NB_SC_PER_RB * nb_rb / 2 - nb_re_sci1);

    if (sym==first_symbol_with_data)
      first_symbol_flag = 1;
    else
      first_symbol_flag = 0;

    start_meas(&rxUE->generic_stat_bis[UE_id]);
    nr_slsch_extract_rbs((int32_t **) rxdata,
                        rxUE->pssch_vars[UE_id],
                        slot,
                        sym,
                        pilots,
                        &slsch_ue_rx_harq->pssch_pdu,
                        &rxUE->frame_parms,
                        slsch_ue_rx_harq,
                        rxUE->chest_time);

    stop_meas(&rxUE->generic_stat_bis[UE_id]);
  //----------------------------------------------------------
  //--------------------- Channel Scaling --------------------
  //----------------------------------------------------------
    // Todo: this line should be double check
    #if 1
    int32_t nb_re_pssch = (pilots==1)? (nb_rb*dmrs_data_re) : (nb_rb*12);
    start_meas(&rxUE->generic_stat_bis[UE_id]);
    nr_dlsch_scale_channel(rx_size_symbol,
                          dl_ch_estimates_ext,
                          &rxUE->frame_parms,
                          Nl,
                          rxUE->frame_parms.nb_antennas_rx,
                          sym,
                          pilots,
                          nb_re_pssch,
                          nb_rb);
    stop_meas(&rxUE->generic_stat_bis[UE_id]);

    //----------------------------------------------------------
    //--------------------- Channel Level Calc. ----------------
    //----------------------------------------------------------
    start_meas(&rxUE->generic_stat_bis[UE_id]);
    if (first_symbol_flag==1) {
      nr_dlsch_channel_level(rx_size_symbol,
                            dl_ch_estimates_ext,
                            &rxUE->frame_parms,
                            Nl,
                            avg,
                            sym,
                            nb_re_pssch,
                            nb_rb);
      avgs = 0;
      for (int aatx=0;aatx<Nl;aatx++)
        for (int aarx=0;aarx<rxUE->frame_parms.nb_antennas_rx;aarx++) {
          //LOG_I(PHY, "nb_rb %d len %d avg_%d_%d Power per SC is %d\n",nb_rb, len,aarx, aatx,avg[aatx*rxUE->frame_parms.nb_antennas_rx+aarx]);
          avgs = cmax(avgs,avg[(aatx*rxUE->frame_parms.nb_antennas_rx)+aarx]);
          //LOG_I(PHY, "avgs Power per SC is %d\n", avgs);
          median[(aatx*rxUE->frame_parms.nb_antennas_rx)+aarx] = avg[(aatx*rxUE->frame_parms.nb_antennas_rx)+aarx];
        }
      if (slsch_ue_rx_harq->Nl > 1) {
        nr_dlsch_channel_level_median(rx_size_symbol,
                                      dl_ch_estimates_ext,
                                      median,
                                      Nl,
                                      rxUE->frame_parms.nb_antennas_rx,
                                      nb_re_pssch);
        for (int aatx = 0; aatx < Nl; aatx++) {
          for (int aarx = 0; aarx < rxUE->frame_parms.nb_antennas_rx; aarx++) {
            avgs = cmax(avgs, median[aatx*rxUE->frame_parms.nb_antennas_rx + aarx]);
          }
        }
      }
      rxUE->pssch_vars[UE_id]->log2_maxh = (log2_approx(avgs)/2) + 1;
      //LOG_I(PHY, "avgs Power per SC is %d lg2_maxh %d\n", avgs,  rxUE->pssch_vars[UE_id]->log2_maxh);
      LOG_D(PHY,"[SLSCH] AbsSubframe %d.%d log2_maxh = %d [log2_maxh0 %d log2_maxh1 %d] (%d,%d)\n",
            frame%1024,
            slot,
            rxUE->pssch_vars[UE_id]->log2_maxh,
            rxUE->pssch_vars[UE_id]->log2_maxh0,
            rxUE->pssch_vars[UE_id]->log2_maxh1,
            avg[0],
            avgs);
    }
    stop_meas(&rxUE->generic_stat_bis[UE_id]);

  /////////////////////////////////////////////////////////
  ////////////// Channel Compensation /////////////////////
    start_meas(&rxUE->generic_stat_bis[UE_id]);

    if (pilots==0){
      nr_dlsch_channel_compensation(rx_size_symbol,
                                    rxUE->frame_parms.nb_antennas_rx,
                                    rxdataF_ext,
                                    dl_ch_estimates_ext,
                                    dl_ch_mag,
                                    dl_ch_magb,
                                    dl_ch_magr,
                                    rxdataF_comp,
                                    NULL,//NULL:disable meas. rxUE->pssch_vars[UE_id]->rho:enable meas.
                                    &rxUE->frame_parms,
                                    Nl,
                                    sym,
                                    nb_re_pssch,
                                    first_symbol_flag,
                                    slsch_ue_rx_harq->Qm,
                                    nb_rb,
                                    rxUE->pssch_vars[UE_id]->log2_maxh,
                                    &rxUE->measurements); // log2_maxh+I0_shift

    } else { // DMRS symbol
        if (allocatable_sci2_re > 0) {
          // for SCI2
          nr_dlsch_channel_compensation(rx_size_symbol,
                                        rxUE->frame_parms.nb_antennas_rx,
                                        rxdataF_ext,
                                        dl_ch_estimates_ext,
                                        dl_ch_mag,
                                        dl_ch_magb,
                                        dl_ch_magr,
                                        rxdataF_comp,
                                        NULL,//NULL:disable meas. rxUE->pssch_vars[UE_id]->rho:enable meas.
                                        &rxUE->frame_parms,
                                        Nl,
                                        sym,
                                        allocatable_sci2_re,
                                        first_symbol_flag,
                                        SCI2_mod_order,
                                        nb_rb,
                                        rxUE->pssch_vars[UE_id]->log2_maxh,
                                        &rxUE->measurements);
          diff_re_comp = NR_NB_SC_PER_RB * slsch_ue_rx_harq->nb_rb / 2 - nb_re_sci1 - allocatable_sci2_re;
        } else {
          diff_re_comp = nb_re_pssch;
        }
        nr_dlsch_channel_compensation(rx_size_symbol,
                                      rxUE->frame_parms.nb_antennas_rx,
                                      rxdataF_ext,
                                      dl_ch_estimates_ext,
                                      dl_ch_mag,
                                      dl_ch_magb,
                                      dl_ch_magr,
                                      rxdataF_comp,
                                      NULL,//NULL:disable meas. rxUE->pssch_vars[UE_id]->rho:enable meas.
                                      &rxUE->frame_parms,
                                      Nl,
                                      sym,
                                      diff_re_comp,
                                      first_symbol_flag,
                                      slsch_ue_rx_harq->Qm,
                                      nb_rb,
                                      rxUE->pssch_vars[UE_id]->log2_maxh,
                                      &rxUE->measurements);
    }

    stop_meas(&rxUE->generic_stat_bis[UE_id]);

    start_meas(&rxUE->generic_stat_bis[UE_id]);

    if (rxUE->frame_parms.nb_antennas_rx > 1) {
      nr_dlsch_detection_mrc(rx_size_symbol,
                             Nl,
                             rxUE->frame_parms.nb_antennas_rx,
                             rxdataF_comp,
                             (Nl > 1)? rxUE->pssch_vars[UE_id]->rho : NULL,
                             dl_ch_mag,
                             dl_ch_magb,
                             dl_ch_magr,
                             sym,
                             nb_rb,
                             nb_re_pssch);
      if (Nl >= 2)//Apply zero forcing for 2, 3, and 4 Tx layers
        nr_zero_forcing_rx(rx_size_symbol,
                           rxUE->frame_parms.nb_antennas_rx,
                           Nl,
                           rxdataF_comp,
                           dl_ch_mag,
                           dl_ch_magb,
                           dl_ch_magr,
                           dl_ch_estimates_ext,
                           nb_rb,
                           slsch_ue_rx_harq->Qm,
                           rxUE->pssch_vars[UE_id]->log2_maxh,
                           sym,
                           nb_re_pssch);
    }
    stop_meas(&rxUE->generic_stat_bis[UE_id]);
#endif
  ////////////////////////////////////////////////////////
  /////////////// LLR calculation ////////////////////////
    memcpy(rxUE->pssch_vars[0]->sl_ch_mag, dl_ch_mag[0], rx_size_symbol * sizeof(dl_ch_mag));
    memcpy(rxUE->pssch_vars[0]->sl_ch_magb, dl_ch_mag[0], rx_size_symbol * sizeof(dl_ch_mag));
    memcpy(rxUE->pssch_vars[0]->rxdataF_comp, rxdataF_comp[0], rx_size_symbol * sizeof(rxdataF_comp));
  
    for (int aatx = 0; aatx < Nl; aatx++) {
      if (pilots == 0) {
        nr_slsch_compute_llr(&rxUE->pssch_vars[UE_id]->rxdataF_comp[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                             &rxUE->pssch_vars[UE_id]->sl_ch_mag0[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                             &rxUE->pssch_vars[UE_id]->sl_ch_magb0[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                             &slsch_llr_layers[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                             (nb_rb * NR_NB_SC_PER_RB - nb_re_sci1) / NR_NB_SC_PER_RB,
                             nb_rb * NR_NB_SC_PER_RB - nb_re_sci1,
                             sym,
                             mod_order);

        memcpy(&layer_llr[aatx*rxUE->frame_parms.nb_antennas_rx][data_offset * 2],
                &slsch_llr_layers[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                sizeof(uint32_t) * (nb_rb * NR_NB_SC_PER_RB - nb_re_sci1));
        data_offset += nb_rb * NR_NB_SC_PER_RB - nb_re_sci1;
      } else {
        if (allocatable_sci2_re > 0) {

          nr_slsch_compute_llr(&rxUE->pssch_vars[UE_id]->rxdataF_comp[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                               &rxUE->pssch_vars[UE_id]->sl_ch_mag0[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                               &rxUE->pssch_vars[UE_id]->sl_ch_magb0[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                               &slsch_llr_layers[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB],
                               allocatable_sci2_re / 6,
                               allocatable_sci2_re,
                               sym,
                               SCI2_mod_order);

          memcpy(&layer_llr[aatx*rxUE->frame_parms.nb_antennas_rx][sci2_offset * 2],
                 &slsch_llr_layers[aatx*rxUE->frame_parms.nb_antennas_rx][sym * slsch_ue_rx_harq->nb_rb * NR_NB_SC_PER_RB],
                 sizeof(uint32_t) * allocatable_sci2_re);

          sci2_offset += allocatable_sci2_re;
        }
        uint32_t diff_re = NR_NB_SC_PER_RB * nb_rb / 2 - nb_re_sci1 - allocatable_sci2_re;
        if (diff_re > 0) {
          uint32_t offset = allocatable_sci2_re;

          nr_slsch_compute_llr(&rxUE->pssch_vars[UE_id]->rxdataF_comp[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB + offset],
                               &rxUE->pssch_vars[UE_id]->sl_ch_mag0[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB + offset],
                               &rxUE->pssch_vars[UE_id]->sl_ch_magb0[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB + offset],
                               &slsch_llr_layers[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB + offset],
                               diff_re / NR_NB_SC_PER_RB,
                               diff_re,
                               sym,
                               mod_order);

          memcpy(&layer_llr[aatx*rxUE->frame_parms.nb_antennas_rx][data_offset * 2],
                 &slsch_llr_layers[aatx*rxUE->frame_parms.nb_antennas_rx][sym * nb_rb * NR_NB_SC_PER_RB + offset],
                 sizeof(uint32_t) * diff_re);

          data_offset += diff_re;
        }
        if (allocatable_sci2_re > 0) {
          nb_re_SCI2 -= allocatable_sci2_re;
        }
      }
    }
  }//symbol

  #ifdef DEBUG_PSSCH_MAPPING
  sprintf(filename,"ch_est_ext_output_%d.m", slot);
  LOG_M(filename,"ch_est_ext_output",dl_ch_estimates_ext[0],5*(rxUE->frame_parms.ofdm_symbol_size), 1, 13);
  sprintf(filename,"rxdata_ext_%d.m", slot);
  LOG_M(filename,"rxdata_ext",rxdataF_ext[0],5*(rxUE->frame_parms.ofdm_symbol_size), 1, 13);
  sprintf(filename,"rxdata_comp_%d.m", slot);
  LOG_M(filename,"rxdata_comp",rxUE->pssch_vars[UE_id]->rxdataF_comp[0],5*(rxUE->frame_parms.ofdm_symbol_size), 1, 13);
  sprintf(filename,"layer_llr.m");
  LOG_M(filename,"layer_llr",layer_llr[0],5*(rxUE->frame_parms.ofdm_symbol_size), 1, 13);
  #endif
  /////////////// Layer demapping ////////////////////////
  // For SCI2
  nr_dlsch_layer_demapping(llr,
                          Nl,
                          SCI2_mod_order,
                          num_sci2_symbs,
                          slsch_ue_rx_harq->codeword,
                          -1,
                          layer_llr);
  int16_t *dst_data = llr[0] + num_sci2_symbs * slsch_ue_rx_harq->Nl;
  int16_t *src_data = layer_llr[0] + num_sci2_symbs;
  for (int i = 0; i < NR_MAX_NB_LAYERS; i++)
    free(layer_llr[i]);
  // For Data
  nr_dlsch_layer_demapping(&dst_data,
                          Nl,
                          mod_order,
                          num_data_symbs,
                          slsch_ue_rx_harq->codeword,
                          -1,
                          &src_data);
  ////////////////////////////////////////////////////////
  /////////////// Unscrambling ///////////////////////////
  nr_codeword_unscrambling_sl(llr[0], multiplex_input_len,
                              slsch_ue_rx->harq_processes[0]->B_sci2,
                              Nidx, Nl);
  ///////////////////////////////////////////////////////
  #ifdef DEBUG_PSSCH_MAPPING
  sprintf(filename,"llr_decoding.m");
  LOG_M(filename,"llr_decoding",llr[0],5*(rxUE->frame_parms.ofdm_symbol_size), 1, 13);
  #endif
  /////////////// Decoding SLSCH and SCIA2 //////////////
  uint32_t ret = nr_slsch_decoding(rxUE, proc, llr[0],
                            &rxUE->frame_parms, slsch_ue_rx,
                            slsch_ue_rx->harq_processes[0], frame,
                            nb_symb_sch, slot, harq_pid);
  ///////////////////////////////////////////////////////
  for (int i = 0; i < nb_codewords; i++)
    free(llr[i]);
  return ret;

}
