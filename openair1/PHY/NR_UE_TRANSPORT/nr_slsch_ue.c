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

/*! \file PHY/NR_UE_TRANSPORT/nr_slsch_ue.c
* \brief Top-level routines for transmission of the PSSCH TS 38.211 v 15.4.0
* \author Melissa Elkadi, David Kim
* \date 2023
* \version 0.1
* \company EpiSci, Episys Sciene Inc., LLC
* \email: melissa@episci.com, david.kim@episci.com
* \note
* \warning
*/

#include <time.h>
#include <stdint.h>

#include "common/utils/assertions.h"
#include "PHY/TOOLS/tools_defs.h"
#include "PHY/defs_nr_common.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_common.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_sch_dmrs_sl.h"

#define SCI2_LEN_SIZE 35
#define NB_RB_SCI1 20
static void nr_pssch_codeword_scrambling_sl(uint8_t *in,
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
    LOG_D(NR_PHY, "i %d b_idx %d in %d s 0x%08x out 0x%08x\n", i, b_idx, in[i], s, *out);
  }
}


static void nr_pssch_data_control_multiplexing(uint8_t *in_slssh,
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

static int16_t** virtual_resource_mapping(NR_DL_FRAME_PARMS *frame_parms,
                              sl_nr_tx_config_pscch_pssch_pdu_t  *pssch_pdu,
                              unsigned int G_SCI2_bits,
                              uint8_t  sci2_mod_order,
                              int16_t **tx_layers,
                              uint32_t **pssch_dmrs)
{

  uint16_t start_sc = frame_parms->first_carrier_offset + (pssch_pdu->pssch_startsym + pssch_pdu->bwp_start) * NR_NB_SC_PER_RB;
  uint16_t n_dmrs = (pssch_pdu->bwp_start + pssch_pdu->pssch_startsym + pssch_pdu->pssch_numrbs) * 6;
  int16_t mod_dmrs[n_dmrs << 1] __attribute((aligned(16)));
  if (start_sc >= frame_parms->ofdm_symbol_size)
    start_sc -= frame_parms->ofdm_symbol_size;

  pssch_pdu->dmrs_ports = (pssch_pdu->num_layers == 2) ? 0b11 : 0b01;
  uint16_t nb_re_sci1 = NB_RB_SCI1 * NR_NB_SC_PER_RB;

  int encoded_length = frame_parms->N_RB_SL * 14 * NR_NB_SC_PER_RB * pssch_pdu->mod_order * pssch_pdu->num_layers;
  int16_t **tx_precoding = (int16_t **)malloc16_clear(pssch_pdu->num_layers * sizeof(int16_t *));
  for (int nl = 0; nl < pssch_pdu->num_layers; nl++)
    tx_precoding[nl] = (int16_t *)malloc16_clear((encoded_length << 1) * sizeof(int16_t));

  for (int nl = 0; nl < pssch_pdu->num_layers; nl++) {
    uint8_t k_prime = 0;
    uint16_t m = G_SCI2_bits / pssch_pdu->mod_order;
    uint16_t m0 = 0;
    int dmrs_port = get_dmrs_port(nl, pssch_pdu->dmrs_ports);
    int16_t Wf[2], Wt[2];
    get_Wt_sl(Wt, dmrs_port);
    get_Wf_sl(Wf, dmrs_port);

    for (int l = pssch_pdu->pssch_startsym; l < pssch_pdu->pssch_startsym + pssch_pdu->pssch_numsym; l++) {

      uint8_t is_dmrs_sym = 0;
      uint16_t dmrs_idx = 0;
      if ((pssch_pdu->dmrs_symbol_position >> l) & 0x01) {
        is_dmrs_sym = 1;
        dmrs_idx = (pssch_pdu->bwp_start + pssch_pdu->pssch_startsym) * 6;
        nr_modulation(pssch_dmrs[l], n_dmrs * 2, DMRS_MOD_ORDER, mod_dmrs);
      }

      uint16_t n = 0;
      uint16_t k = (1 <= l && l <= 3) ? start_sc + nb_re_sci1 : start_sc;
      for (int i = k; i < pssch_pdu->pssch_numrbs * NR_NB_SC_PER_RB; i++) {
        uint8_t is_dmrs = 0;
        int sample_offsetF = l * frame_parms->ofdm_symbol_size + k;
        int l_prime[2] = {0};
        if (is_dmrs_sym) {
          if (k == (start_sc + get_dmrs_freq_idx_sl(n, k_prime, get_delta_sl(dmrs_port)) % frame_parms->ofdm_symbol_size))
            is_dmrs = 1;
        }
        if (is_dmrs == 1) {
          ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1] = (Wt[l_prime[0]] * Wf[k_prime] * AMP * mod_dmrs[dmrs_idx << 1]) >> 15;
          ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = (Wt[l_prime[0]] * Wf[k_prime] * AMP * mod_dmrs[(dmrs_idx << 1) + 1]) >> 15;
          dmrs_idx++;
          k_prime++;
          k_prime &= 1;
          n += (k_prime) ? 0 : 1;
        } else if (!is_dmrs_sym) {
          ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1] = ((int16_t *)tx_layers[nl])[m << 1];
          ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = ((int16_t *)tx_layers[nl])[(m << 1) + 1];
          m++;
        } else if ((is_dmrs_sym) && (is_dmrs != 1)) {
          if (m0 < G_SCI2_bits / pssch_pdu->mod_order) {
            ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1] = ((int16_t *)tx_layers[nl])[m0 << 1];
            ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = ((int16_t *)tx_layers[nl])[(m0 << 1) + 1];
            m0++;
          } else {
            ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1] = ((int16_t *)tx_layers[nl])[m << 1];
            ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = ((int16_t *)tx_layers[nl])[(m << 1) + 1];
            m++;
          }
        } else {
          ((int16_t *)tx_precoding[nl])[(sample_offsetF) << 1] = 0;
          ((int16_t *)tx_precoding[nl])[((sample_offsetF) << 1) + 1] = 0;
        }
        if (++k >= frame_parms->ofdm_symbol_size)
          k -= frame_parms->ofdm_symbol_size;
      } //for (i = 0; i < pssch_pdu->pssch_numrbs * NR_NB_SC_PER_RB; i++)
    } //for (l = start_symbol; l < start_symbol + number_of_symbols; l++)
  } //for (nl = 0; nl < Nl; nl++)
  return tx_precoding;
}

static void physical_resource_mapping(NR_DL_FRAME_PARMS *frame_parms,
                               sl_nr_tx_config_pscch_pssch_pdu_t  *pssch_pdu,
                               int16_t** tx_precoding,
                               c16_t **txdataF)
{
  uint16_t nb_re_sci1 = NB_RB_SCI1 * NR_NB_SC_PER_RB;
  uint16_t start_sc = frame_parms->first_carrier_offset + (pssch_pdu->startrb + pssch_pdu->bwp_start) * NR_NB_SC_PER_RB;
  AssertFatal(frame_parms->nb_antennas_tx == pssch_pdu->num_layers,
              "Invalid num TX antennas %d of num layers %d\n",
              frame_parms->nb_antennas_tx, pssch_pdu->num_layers);
  for (int ap = 0; ap < frame_parms->nb_antennas_tx; ap++) {
    // Copying symbol 1 to symbol 0 (AGC)
    memcpy((int16_t *)&txdataF[ap][start_sc],
          &tx_precoding[ap][2 * (frame_parms->ofdm_symbol_size + start_sc)],
          NR_NB_SC_PER_RB * sizeof(int32_t));

    for (int l = pssch_pdu->pssch_startsym; l < pssch_pdu->pssch_startsym + pssch_pdu->pscch_numsym; l++) {
      uint16_t k = start_sc;
      if (1 <= l && l <= 3) {
        k = start_sc + nb_re_sci1;
      }
      for (int rb = 0; rb < pssch_pdu->pssch_numrbs; rb++) {
        if (k + NR_NB_SC_PER_RB <= frame_parms->ofdm_symbol_size) {
          memcpy((int16_t *)&txdataF[ap][l * frame_parms->ofdm_symbol_size + k],
                &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size + k)],
                NR_NB_SC_PER_RB * sizeof(int32_t));
        } else {
          int neg_length = frame_parms->ofdm_symbol_size - k;
          int pos_length = NR_NB_SC_PER_RB - neg_length;
          memcpy((int16_t *)&txdataF[ap][l * frame_parms->ofdm_symbol_size + k],
                  &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size + k)],
                  neg_length * sizeof(int32_t));
          memcpy((int16_t *)&txdataF[ap][l * frame_parms->ofdm_symbol_size],
                &tx_precoding[ap][2 * (l * frame_parms->ofdm_symbol_size)],
                pos_length * sizeof(int32_t));
        }
        k += NR_NB_SC_PER_RB;
        if (k >= frame_parms->ofdm_symbol_size) {
          k -= frame_parms->ofdm_symbol_size;
        }
      } // RB loop
    } // symbol loop
  } // port loop
}

void nr_tx_pssch(PHY_VARS_NR_UE *UE, uint32_t frame_tx,
                uint32_t slot_tx,
                sl_nr_tx_config_pscch_pssch_pdu_t *pssch_pssch_vars,
                c16_t **txdataF)
{
  unsigned int G_slsch_bits = nr_get_G(pssch_pssch_vars->pssch_numrbs,
                                       pssch_pssch_vars->pssch_numsym,
                                       get_num_dmrs(pssch_pssch_vars->dmrs_symbol_position) * 6,
                                       get_num_dmrs(pssch_pssch_vars->dmrs_symbol_position),
                                       pssch_pssch_vars->mod_order,
                                       pssch_pssch_vars->num_layers);
  LOG_I(PHY, "This is G_slsch_bits %d \n", G_slsch_bits);

  if (nr_slsch_encoding(UE, pssch_pssch_vars, &UE->SL_UE_PHY_PARAMS.sl_frame_params, pssch_pssch_vars->harq_pid, G_slsch_bits) == -1)
    return;
  NR_UL_UE_HARQ_t *harq_process_ul_ue = pssch_pssch_vars->harq_processes_ul;
  nr_pssch_data_control_multiplexing(harq_process_ul_ue->f,
                                     harq_process_ul_ue->f_sci2,
                                     G_slsch_bits,
                                     harq_process_ul_ue->B_sci2,
                                     pssch_pssch_vars->num_layers,
                                     pssch_pssch_vars->mod_order,
                                     harq_process_ul_ue->f_multiplexed);
  uint32_t scrambled_output[(harq_process_ul_ue->B >> 5) + 1];
  memset(scrambled_output, 0, ((harq_process_ul_ue->B >> 5) + 1) * sizeof(uint32_t));
  nr_pssch_codeword_scrambling_sl(harq_process_ul_ue->f_multiplexed,
                                  harq_process_ul_ue->B_multiplexed,
                                  (harq_process_ul_ue->B_sci2 * pssch_pssch_vars->num_layers) / 8,
                                  pssch_pssch_vars->nid_x,
                                  scrambled_output);

  int max_num_re = pssch_pssch_vars->num_layers * pssch_pssch_vars->pssch_numsym * pssch_pssch_vars->pssch_numrbs * NR_NB_SC_PER_RB;
  int32_t d_mod[max_num_re] __attribute__ ((aligned(16)));
  uint32_t num_sci2_bits = harq_process_ul_ue->B_sci2 * pssch_pssch_vars->num_layers;
  uint8_t  sci2_mod_order = pssch_pssch_vars->mod_order;
  nr_modulation(scrambled_output, num_sci2_bits, sci2_mod_order, (int16_t *)d_mod);
  nr_modulation(scrambled_output + (num_sci2_bits >> 5), G_slsch_bits,
                pssch_pssch_vars->mod_order, (int16_t *)(d_mod + num_sci2_bits / sci2_mod_order));

  int16_t **tx_layers = (int16_t **)malloc16_clear(pssch_pssch_vars->num_layers * sizeof(int16_t *));
  uint16_t num_sci2_symbs = (num_sci2_bits << 1) / sci2_mod_order;
  uint16_t num_data_symbs = (G_slsch_bits << 1) / pssch_pssch_vars->mod_order;
  uint32_t num_total_symbs = (num_sci2_symbs + num_data_symbs) >> 1;
  for (int nl = 0; nl < pssch_pssch_vars->num_layers; nl++)
    tx_layers[nl] = (int16_t *)malloc16_clear((num_sci2_symbs + num_data_symbs) * sizeof(int16_t));

  nr_ue_layer_mapping((int16_t *)d_mod, pssch_pssch_vars->num_layers, num_total_symbs, tx_layers);
  nr_init_pssch_dmrs(UE, UE->target_Nid_cell);
  uint32_t **pssch_dmrs = UE->nr_gold_pssch_dmrs[slot_tx];
  int16_t** tx_precoding = virtual_resource_mapping(&UE->SL_UE_PHY_PARAMS.sl_frame_params,
                                                    pssch_pssch_vars,
                                                    harq_process_ul_ue->B_sci2,
                                                    sci2_mod_order,
                                                    tx_layers,
                                                    pssch_dmrs);
  physical_resource_mapping(&UE->SL_UE_PHY_PARAMS.sl_frame_params, pssch_pssch_vars, tx_precoding, txdataF);

  NR_UL_UE_HARQ_t *harq_process_slsch = NULL;
  harq_process_slsch = &pssch_pssch_vars->harq_processes_ul[pssch_pssch_vars->harq_pid];
  harq_process_slsch->status = SCH_IDLE;
  nr_ue_pssch_common_procedures(UE, slot_tx, &UE->SL_UE_PHY_PARAMS.sl_frame_params, pssch_pssch_vars->num_layers, txdataF, link_type_sl);

  for (int nl = 0; nl < pssch_pssch_vars->num_layers; nl++) {
    free_and_zero(tx_layers[nl]);
    free_and_zero(tx_precoding[nl]);
  }
  free_and_zero(tx_layers);
  free_and_zero(tx_precoding);

}


uint8_t nr_ue_pssch_common_procedures(PHY_VARS_NR_UE *UE,
                                      uint8_t slot,
                                      NR_DL_FRAME_PARMS *frame_parms,
                                      uint8_t n_antenna_ports,
                                      c16_t **txdataF,
                                      int link_type) {

  c16_t **txdata = UE->common_vars.txData;
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

