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

/*! \file PHY/NR_UE_TRANSPORT/nr_ulsch.c
* \brief Top-level routines for transmission of the PUSCH TS 38.211 v 15.4.0
* \author Khalid Ahmed
* \date 2019
* \version 0.1
* \company Fraunhofer IIS
* \email: khalid.ahmed@iis.fraunhofer.de
* \note
* \warning
*/
#include <stdint.h>
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_REFSIG/refsig_defs_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/MODULATION/modulation_common.h"
#include "common/utils/assertions.h"
#include "common/utils/nr/nr_common.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/defs_nr_common.h"
#include "PHY/TOOLS/tools_defs.h"
#include "executables/nr-softmodem.h"
#include "executables/softmodem-common.h"
#include "PHY/NR_REFSIG/ul_ref_seq_nr.h"
#include <openair2/UTIL/OPT/opt.h>

//#define DEBUG_PUSCH_MAPPING
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
      j = j + 1;
    } else {
      m_ij =  (i < SCI2_bits) ? j : SCI2_bits;
      c = (uint8_t)((s >> ((i - m_ij) % 32)) & 1);
      *out ^= ((in[i] + c) & 1) << b_idx;
    }
    //LOG_D(NR_PHY, "i %d b_idx %d in %d s 0x%08x out 0x%08x\n", i, b_idx, in[i], s, *out);
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

void nr_ue_slsch_tx_procedures(PHY_VARS_NR_UE *UE,
                            unsigned char harq_pid,
                            uint32_t frame,
                            uint8_t slot,
                            uint8_t thread_id,
                            int gNB_id) {

  LOG_D(PHY,"nr_ue_slsch_tx_procedures hard_id %d %d.%d\n",harq_pid,frame,slot);

  int8_t Wf[2], Wt[2];
  int l_prime[2], delta;
  uint8_t nb_dmrs_re_per_rb;
  int i;
  int sample_offsetF, N_RE_prime;

  NR_DL_FRAME_PARMS *frame_parms = &UE->frame_parms;
  int32_t **txdataF = UE->common_vars.txdataF;

  int      N_PRB_oh = 0; // higher layer (RRC) parameter xOverhead in PUSCH-ServingCellConfig
  uint16_t number_dmrs_symbols = 0;

  NR_UE_ULSCH_t *ulsch_ue = UE->ulsch[thread_id][gNB_id];//TODO delete it
  NR_UE_ULSCH_t *slsch_ue = UE->slsch[thread_id][gNB_id];
  NR_UL_UE_HARQ_t *harq_process_ul_ue = slsch_ue->harq_processes[harq_pid];
  nfapi_nr_ue_pssch_pdu_t *pssch_pdu = &harq_process_ul_ue->pssch_pdu;

  int start_symbol          = pssch_pdu->start_symbol_index;
  uint16_t sl_dmrs_symb_pos = pssch_pdu->sl_dmrs_symb_pos;
  uint8_t number_of_symbols = pssch_pdu->nr_of_symbols;
  uint8_t dmrs_type         = pssch_pdu->dmrs_config_type;
  uint16_t start_rb         = pssch_pdu->rb_start;
  uint16_t nb_rb            = pssch_pdu->rb_size;
  uint8_t Nl                = pssch_pdu->nrOfLayers;
  uint8_t mod_order         = pssch_pdu->qam_mod_order;
  uint16_t rnti             = pssch_pdu->rnti;
  uint8_t cdm_grps_no_data  = pssch_pdu->num_dmrs_cdm_grps_no_data;
  uint16_t start_sc         = frame_parms->first_carrier_offset + (start_rb+pssch_pdu->bwp_start)*NR_NB_SC_PER_RB;

  if (start_sc >= frame_parms->ofdm_symbol_size)
    start_sc -= frame_parms->ofdm_symbol_size;

  slsch_ue->Nid_cell = frame_parms->Nid_cell;

  for (int i = start_symbol; i < start_symbol + number_of_symbols; i++) {
    if((sl_dmrs_symb_pos >> i) & 0x01)
      number_dmrs_symbols += 1;
  }

  nb_dmrs_re_per_rb = ((dmrs_type == pusch_dmrs_type1) ? 6:4)*cdm_grps_no_data;

  LOG_D(PHY,"ulsch TX %x : start_rb %d nb_rb %d mod_order %d Nl %d Tpmi %d bwp_start %d start_sc %d start_symbol %d num_symbols %d cdmgrpsnodata %d num_dmrs %d dmrs_re_per_rb %d\n",
        rnti,start_rb,nb_rb,mod_order,Nl,pssch_pdu->Tpmi,pssch_pdu->bwp_start,start_sc,start_symbol,number_of_symbols,cdm_grps_no_data,number_dmrs_symbols,nb_dmrs_re_per_rb);
  // TbD num_of_mod_symbols is set but never used
  N_RE_prime = NR_NB_SC_PER_RB*number_of_symbols - nb_dmrs_re_per_rb*number_dmrs_symbols - N_PRB_oh;
  harq_process_ul_ue->num_of_mod_symbols = N_RE_prime*nb_rb;


  /////////////////////////SLSCH SCI2 coding/////////////////////////
  // TODO: update the following
    /* payload is 56 bits */
  PSSCH_SCI2_payload pssch_payload;             // NR Side Link Payload for Rel 16
  pssch_payload.coverageIndicator = 0;     // 1 bit
  pssch_payload.tddConfig = 0xFFF;         // 12 bits for TDD configuration
  pssch_payload.DFN = 0x3FF;               // 10 bits for DFN
  pssch_payload.slotIndex = 0x2A;          // 7 bits for Slot Index //frame_parms->p_TDD_UL_DL_ConfigDedicated->slotIndex;
  pssch_payload.reserved = 0;              // 2 bits reserved

  NR_UE_PSSCH m_pssch;
  UE->pssch_vars[0] = &m_pssch;
  NR_UE_PSSCH *pssch = UE->pssch_vars[0];
  memset((void *)pssch, 0, sizeof(NR_UE_PSSCH));

  pssch->pssch_a = *((uint32_t *)&pssch_payload);
  pssch->pssch_a_interleaved = pssch->pssch_a; // skip interlevaing for Sidelink

  // Encoder reversal
  uint64_t a_reversed = 0;
  for (int i = 0; i < NR_POLAR_PSSCH_PAYLOAD_BITS; i++)
    a_reversed |= (((uint64_t)pssch->pssch_a_interleaved >> i) & 1) << (31 - i);
  uint16_t Nidx;
  Nidx = get_Nidx_from_CRC(&a_reversed, 0, 0,
                           NR_POLAR_PSSCH_MESSAGE_TYPE,
                           NR_POLAR_PSSCH_PAYLOAD_BITS,
                           NR_POLAR_PSSCH_AGGREGATION_LEVEL);

  /// CRC, coding and rate matching
  polar_encoder_fast(&a_reversed, (void*)pssch->pssch_e, 0, 0,
                     NR_POLAR_PSSCH_MESSAGE_TYPE,
                     NR_POLAR_PSSCH_PAYLOAD_BITS,
                     NR_POLAR_PSSCH_AGGREGATION_LEVEL);

  /////////////////////////SLSCH data coding/////////////////////////
  unsigned int G_slsch = nr_get_G(nb_rb, number_of_symbols,
                            nb_dmrs_re_per_rb, number_dmrs_symbols, mod_order, Nl);


  trace_NRpdu(DIRECTION_UPLINK,
              harq_process_ul_ue->a,
              harq_process_ul_ue->pssch_pdu.pssch_data.tb_size,
              WS_C_RNTI, rnti, frame, slot, 0, 0);

  if (nr_slsch_encoding(UE, slsch_ue, frame_parms, harq_pid, G_slsch) == -1)
    return;

  //////////////////SLSCH data and control multiplexing//////////////
  uint32_t available_bits = G_slsch + NR_POLAR_PSSCH_E;
  uint32_t SCI2_bits = NR_POLAR_PSSCH_E;
  uint8_t  SCI2_mod_order = 2;
  uint8_t multiplexed_output[available_bits];
  memset(multiplexed_output, 0, available_bits * sizeof(uint8_t));

  nr_pssch_data_control_multiplexing(harq_process_ul_ue->f,
                                     (uint8_t*)pssch->pssch_e,
                                     G_slsch,
                                     SCI2_bits,
                                     Nl,
                                     SCI2_mod_order,
                                     multiplexed_output);
  /////////////////////////SLSCH scrambling/////////////////////////
  uint32_t scrambled_output[(available_bits >> 5) + 1];
  memset(scrambled_output, 0, ((available_bits >> 5) + 1)*sizeof(uint32_t));

  nr_pusch_codeword_scrambling_sl(multiplexed_output,
                                  available_bits,
                                  SCI2_bits,
                                  Nidx,
                                  scrambled_output);

  //////////////////////////////////////////////////////////////////////////

  /////////////////////////SLSCH modulation/////////////////////////

  int max_num_re = Nl * number_of_symbols * nb_rb * NR_NB_SC_PER_RB;
  int32_t d_mod[max_num_re] __attribute__ ((aligned(16)));

  // modulating for the 2nd-stage SCI bits
  nr_modulation(scrambled_output, // assume one codeword for the moment
                SCI2_bits,
                SCI2_mod_order,
                (int16_t *)d_mod);

  // modulating SL-SCH bits
  nr_modulation(scrambled_output + (SCI2_bits >> 5), // assume one codeword for the moment
                G_slsch,
                mod_order,
                (int16_t *)(d_mod + SCI2_bits / SCI2_mod_order));

  ////////////////////////////////////////////////////////////////////////

  /////////////////////////DMRS Modulation/////////////////////////

  nr_init_pssch_dmrs(UE, Nidx);

  uint32_t **pssch_dmrs = UE->nr_gold_pssch_dmrs[slot];
  uint16_t n_dmrs = (pssch_pdu->bwp_start + start_rb + nb_rb) * ((dmrs_type == pusch_dmrs_type1) ? 6 : 4);
  int16_t mod_dmrs[n_dmrs << 1] __attribute((aligned(16)));

  ////////////////////////////////////////////////////////////////////////


  /////////////////////////PTRS parameters' initialization/////////////////////////

  int16_t mod_ptrs[nb_rb] __attribute((aligned(16))); // assume maximum number of PTRS per pusch allocation
  uint8_t L_ptrs, K_ptrs = 0;
  uint16_t beta_ptrs = 1; // temp value until power control is implemented

  if (pssch_pdu->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {

    K_ptrs = harq_process_ul_ue->pssch_pdu.pssch_ptrs.ptrs_freq_density;
    L_ptrs = 1<<harq_process_ul_ue->pssch_pdu.pssch_ptrs.ptrs_time_density;

    beta_ptrs = 1; // temp value until power control is implemented

    slsch_ue->ptrs_symbols = 0;

    set_ptrs_symb_idx(&slsch_ue->ptrs_symbols,
                      number_of_symbols,
                      start_symbol,
                      L_ptrs,
                      sl_dmrs_symb_pos);
  }

  ////////////////////////////////////////////////////////////////////////////////

  /////////////////////////SLSCH layer mapping/////////////////////////

  int16_t **tx_layers = (int16_t **)malloc16_clear(Nl * sizeof(int16_t *));
  uint16_t n_symbs = (SCI2_bits << 1) / SCI2_mod_order + (G_slsch << 1) / mod_order;
  for (int nl = 0; nl < Nl; nl++)
    tx_layers[nl] = (int16_t *)malloc16_clear(n_symbs * sizeof(int16_t));

  nr_ue_layer_mapping((int16_t *)d_mod, Nl, n_symbs, tx_layers);

  ////////////////////////////////////////////////////////////////////////


  //////////////////////// SLSCH transform precoding ////////////////////////

  l_prime[0] = 0; // single symbol ap 0

  uint16_t index;
  uint8_t u = 0, v = 0;
  int16_t *dmrs_seq = NULL;

  /// Transform-coded "y"-sequences (for definition see 38-211 V15.3.0 2018-09, subsection 6.3.1.4)
  int32_t y[max_num_re] __attribute__ ((aligned(16)));

  if (pssch_pdu->transform_precoding == transformPrecoder_enabled) {

    uint32_t nb_re_pssch=nb_rb * NR_NB_SC_PER_RB;
    uint32_t y_offset = 0;
    uint16_t num_dmrs_res_per_symbol = nb_rb*(NR_NB_SC_PER_RB/2);

    // Calculate index to dmrs seq array based on number of DMRS Subcarriers on this symbol
    index = get_index_for_dmrs_lowpapr_seq(num_dmrs_res_per_symbol);
    u = pssch_pdu->dfts_ofdm.low_papr_group_number;
    v = pssch_pdu->dfts_ofdm.low_papr_sequence_number;
    dmrs_seq = dmrs_lowpaprtype1_ul_ref_sig[u][v][index];

    AssertFatal(index >= 0, "Num RBs not configured according to 3GPP 38.211 section 6.3.1.4. For PUSCH with transform precoding, num RBs cannot be multiple of any other primenumber other than 2,3,5\n");
    AssertFatal(dmrs_seq != NULL, "DMRS low PAPR seq not found, check if DMRS sequences are generated");

    LOG_D(PHY,"Transform Precoding params. u: %d, v: %d, index for dmrsseq: %d\n", u, v, index);

    for (int l = start_symbol; l < start_symbol + number_of_symbols; l++) {

      if((sl_dmrs_symb_pos >> l) & 0x01)
        /* In the symbol with DMRS no data would be transmitted CDM groups is 2*/
        continue;

      nr_dft(&y[y_offset], &((int32_t*)tx_layers[0])[y_offset], nb_re_pssch);

      y_offset = y_offset + nb_re_pssch;

      LOG_D(NR_PHY, "Transform precoding being done on data- symbol: %d, nb_re_pusch: %d, y_offset: %d\n",
                l, nb_re_pssch, y_offset);

#ifdef DEBUG_PUSCH_MAPPING
      printf("NR_ULSCH_UE: y_offset %d\t nb_re_pusch %d \t Symbol %d \t nb_rb %d \n",
             y_offset, nb_re_pssch, l, nb_rb);
#endif
    }

#ifdef DEBUG_DFT_IDFT
    int32_t debug_symbols[MAX_NUM_NR_RE] __attribute__ ((aligned(16)));
    int offset = 0;

    for (int ll = 0; ll < n_symbs; ll++) {
      debug_symbols[ll] = slsch_ue->y[ll];
    }
    printf("NR_ULSCH_UE: numSym: %d, num_dmrs_sym: %d", number_of_symbols, number_dmrs_symbols);
    for (int ll = 0; ll < (number_of_symbols - number_dmrs_symbols); ll++) {
      nr_idft(&debug_symbols[offset], nb_re_pusch);
      offset = offset + nb_re_pusch;
    }
    LOG_M("preDFT_all_symbols.m", "UE_preDFT", tx_layers[0], number_of_symbols * nb_re_pssch, 1, 1);
    LOG_M("postDFT_all_symbols.m", "UE_postDFT", y, number_of_symbols * nb_re_pssch, 1, 1);
    LOG_M("DEBUG_IDFT_SYMBOLS.m", "UE_Debug_IDFT", debug_symbols, number_of_symbols * nb_re_pssch, 1, 1);
    LOG_M("UE_DMRS_SEQ.m", "UE_DMRS_SEQ", dmrs_seq, nb_re_pusch, 1, 1);
#endif
  }
  ////////////////////////////////////////////////////////////////////////



  /////////////////////////ULSCH RE mapping/////////////////////////
  ///////////

  int encoded_length = frame_parms->N_RB_UL*14*NR_NB_SC_PER_RB*mod_order*Nl;
  int16_t **tx_precoding = (int16_t **)malloc16_clear(Nl*sizeof(int16_t *));
  for (int nl=0; nl<Nl; nl++)
    tx_precoding[nl] = (int16_t *)malloc16_clear((encoded_length<<1)*sizeof(int16_t));

  for (int nl=0; nl < Nl; nl++) {
    uint8_t k_prime = 0;
    uint16_t m = 0;
    
#ifdef DEBUG_PUSCH_MAPPING
    printf("NR_ULSCH_UE: Value of CELL ID %d /t, u %d \n", frame_parms->Nid_cell, u);
#endif

    int dmrs_port = get_dmrs_port(nl,pssch_pdu->dmrs_ports);
    // DMRS params for this dmrs port
    get_Wt(Wt, dmrs_port, dmrs_type);
    get_Wf(Wf, dmrs_port, dmrs_type);
    delta = get_delta(dmrs_port, dmrs_type);

    for (int l=start_symbol; l<start_symbol+number_of_symbols; l++) {

      uint16_t k = start_sc;
      uint16_t n = 0;
      uint8_t is_dmrs_sym = 0;
      uint8_t is_ptrs_sym = 0;
      uint16_t dmrs_idx = 0, ptrs_idx = 0;

      if ((sl_dmrs_symb_pos >> l) & 0x01) {
        is_dmrs_sym = 1;

        if (pssch_pdu->transform_precoding == transformPrecoder_disabled){
        
          if (dmrs_type == pusch_dmrs_type1)
            dmrs_idx = (pssch_pdu->bwp_start + start_rb)*6;
          else
            dmrs_idx = (pssch_pdu->bwp_start + start_rb)*4;

          // TODO: performance improvement, we can skip the modulation of DMRS symbols outside the bandwidth part
          // Perform this on gold sequence, not required when SC FDMA operation is done,
          LOG_D(PHY,"DMRS in symbol %d\n",l);
          nr_modulation(pssch_dmrs[l][pssch_pdu->scid], n_dmrs*2, DMRS_MOD_ORDER, mod_dmrs); // currently only codeword 0 is modulated. Qm = 2 as DMRS is QPSK modulated
        } else {
          dmrs_idx = 0;
        }
      } else if (pssch_pdu->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {

        AssertFatal(pssch_pdu->transform_precoding == transformPrecoder_disabled, "PTRS NOT SUPPORTED IF TRANSFORM PRECODING IS ENABLED\n");

        if(is_ptrs_symbol(l, slsch_ue->ptrs_symbols)) {
          is_ptrs_sym = 1;
          nr_modulation(pssch_dmrs[l][pssch_pdu->scid], nb_rb, DMRS_MOD_ORDER, mod_ptrs);
        }
      }

      for (i=0; i< nb_rb*NR_NB_SC_PER_RB; i++) {
        uint8_t is_dmrs = 0;
        uint8_t is_ptrs = 0;

        sample_offsetF = l*frame_parms->ofdm_symbol_size + k;

        if (is_dmrs_sym) {
          if (k == ((start_sc+get_dmrs_freq_idx_ul(n, k_prime, delta, dmrs_type))%frame_parms->ofdm_symbol_size))
            is_dmrs = 1;
        } else if (is_ptrs_sym) {
          is_ptrs = is_ptrs_subcarrier(k,
                                       rnti,
                                       nl,
                                       dmrs_type,
                                       K_ptrs,
                                       nb_rb,
                                       pssch_pdu->pssch_ptrs.ptrs_ports_list[0].ptrs_re_offset,
                                       start_sc,
                                       frame_parms->ofdm_symbol_size);
        }

        if (is_dmrs == 1) {
          // if transform precoding is enabled
          if (pssch_pdu->transform_precoding == transformPrecoder_enabled) {
            ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1] = (Wt[l_prime[0]]*Wf[k_prime]*AMP*dmrs_seq[2*dmrs_idx]) >> 15;
            ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1] = (Wt[l_prime[0]]*Wf[k_prime]*AMP*dmrs_seq[(2*dmrs_idx) + 1]) >> 15;
          } else {
            ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1] = (Wt[l_prime[0]]*Wf[k_prime]*AMP*mod_dmrs[dmrs_idx<<1]) >> 15;
            ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1] = (Wt[l_prime[0]]*Wf[k_prime]*AMP*mod_dmrs[(dmrs_idx<<1) + 1]) >> 15;
          }

#ifdef DEBUG_PUSCH_MAPPING
          printf("DMRS: Layer: %d\t, dmrs_idx %d\t l %d \t k %d \t k_prime %d \t n %d \t dmrs: %d %d\n",
                 nl, dmrs_idx, l, k, k_prime, n, ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1],
                 ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1]);
#endif

          dmrs_idx++;
          k_prime++;
          k_prime&=1;
          n+=(k_prime)?0:1;
      
        }  else if (is_ptrs == 1) {
          ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1] = (beta_ptrs*AMP*mod_ptrs[ptrs_idx<<1]) >> 15;
          ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1] = (beta_ptrs*AMP*mod_ptrs[(ptrs_idx<<1) + 1]) >> 15;
          ptrs_idx++;
        } else if (!is_dmrs_sym || allowed_xlsch_re_in_dmrs_symbol(k, start_sc, frame_parms->ofdm_symbol_size, cdm_grps_no_data, dmrs_type)) {
          if (pssch_pdu->transform_precoding == transformPrecoder_disabled) {
            ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1]       = ((int16_t *)tx_layers[nl])[m<<1];
            ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1] = ((int16_t *)tx_layers[nl])[(m<<1) + 1];
          }
          else {
            ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1]       = ((int16_t *) y)[m<<1];
            ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1] = ((int16_t *) y)[(m<<1) + 1];
          }

#ifdef DEBUG_PUSCH_MAPPING
          printf("DATA: layer %d\t m %d\t l %d \t k %d \t tx_precoding: %d %d\n",
                 nl, m, l, k, ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1],
                 ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1]);
#endif

          m++;

        } else {
          ((int16_t*)tx_precoding[nl])[(sample_offsetF)<<1]       = 0;
          ((int16_t*)tx_precoding[nl])[((sample_offsetF)<<1) + 1] = 0;
        }

        if (++k >= frame_parms->ofdm_symbol_size)
          k -= frame_parms->ofdm_symbol_size;
      } //for (i=0; i< nb_rb*NR_NB_SC_PER_RB; i++) 
    }//for (l=start_symbol; l<start_symbol+number_of_symbols; l++)
  }//for (nl=0; nl < Nl; nl++)



  /////////////////////////ULSCH precoding/////////////////////////
  ///////////
  ///Layer Precoding and Antenna port mapping
  // tx_layers 0-3 are mapped on antenna ports
  // The precoding info is supported by nfapi such as num_prgs, prg_size, prgs_list and pm_idx
  // The same precoding matrix is applied on prg_size RBs, Thus
  //        pmi = prgs_list[rbidx/prg_size].pm_idx, rbidx =0,...,rbSize-1
  // The Precoding matrix:
  for (int ap=0; ap<frame_parms->nb_antennas_tx; ap++) {
    for (int l=start_symbol; l<start_symbol+number_of_symbols; l++) {
      uint16_t k = start_sc;

      for (int rb=0; rb<nb_rb; rb++) {
        //get pmi info
        uint8_t pmi=pssch_pdu->Tpmi;
          
        if (pmi == 0) {//unitary Precoding
          if (k + NR_NB_SC_PER_RB <= frame_parms->ofdm_symbol_size) { // RB does not cross DC
            if (ap<pssch_pdu->nrOfLayers)
              memcpy(&txdataF[ap][l*frame_parms->ofdm_symbol_size  + k],
                     &tx_precoding[ap][2*(l*frame_parms->ofdm_symbol_size + k)],
                     NR_NB_SC_PER_RB*sizeof(int32_t));
            else
              memset(&txdataF[ap][l*frame_parms->ofdm_symbol_size + k],
                     0,
                     NR_NB_SC_PER_RB*sizeof(int32_t));
          } else { // RB does cross DC
            int neg_length = frame_parms->ofdm_symbol_size - k;
            int pos_length = NR_NB_SC_PER_RB - neg_length;
            if (ap<pssch_pdu->nrOfLayers) {
              memcpy(&txdataF[ap][l*frame_parms->ofdm_symbol_size + k],
                     &tx_precoding[ap][2*(l*frame_parms->ofdm_symbol_size + k)],
                     neg_length*sizeof(int32_t));
              memcpy(&txdataF[ap][l*frame_parms->ofdm_symbol_size],
                     &tx_precoding[ap][2*(l*frame_parms->ofdm_symbol_size)],
                     pos_length*sizeof(int32_t));
            } else {
              memset(&txdataF[ap][l*frame_parms->ofdm_symbol_size + k],
                     0,
                     neg_length*sizeof(int32_t));
              memset(&txdataF[ap][l*frame_parms->ofdm_symbol_size],
                     0,
                     pos_length*sizeof(int32_t));
            }
          }
          k += NR_NB_SC_PER_RB;
          if (k >= frame_parms->ofdm_symbol_size) {
            k -= frame_parms->ofdm_symbol_size;
          }
        }
        else {
          //get the precoding matrix weights:
          char *W_prec;
          switch (frame_parms->nb_antennas_tx) {
            case 1://1 antenna port
              W_prec = nr_W_1l_2p[pmi][ap];
              break;
            case 2://2 antenna ports
              if (pssch_pdu->nrOfLayers == 1)//1 layer
                W_prec = nr_W_1l_2p[pmi][ap];
              else//2 layers
                W_prec = nr_W_2l_2p[pmi][ap];
              break;
            case 4://4 antenna ports
              if (pssch_pdu->nrOfLayers == 1)//1 layer
                W_prec = nr_W_1l_4p[pmi][ap];
              else if (pssch_pdu->nrOfLayers == 2)//2 layers
                W_prec = nr_W_2l_4p[pmi][ap];
              else if (pssch_pdu->nrOfLayers == 3)//3 layers
                W_prec = nr_W_3l_4p[pmi][ap];
              else//4 layers
                W_prec = nr_W_4l_4p[pmi][ap];
              break;
            default:
              LOG_D(PHY,"Precoding 1,2, or 4 antenna ports are currently supported\n");
              W_prec = nr_W_1l_2p[pmi][ap];
              break;
          }

          for (int i=0; i<NR_NB_SC_PER_RB; i++) {
            int32_t re_offset = l*frame_parms->ofdm_symbol_size + k;
            int32_t precodatatx_F = nr_layer_precoder(tx_precoding, W_prec, pssch_pdu->nrOfLayers, re_offset);
            ((int16_t*)txdataF[ap])[(re_offset<<1)] = ((int16_t *) &precodatatx_F)[0];
            ((int16_t*)txdataF[ap])[(re_offset<<1) + 1] = ((int16_t *) &precodatatx_F)[1];
                            
            if (++k >= frame_parms->ofdm_symbol_size) {
              k -= frame_parms->ofdm_symbol_size;
            }
          }
        }
      } //RB loop
    } // symbol loop
  }// port loop

  NR_UL_UE_HARQ_t *harq_process_slsch=NULL;
  harq_process_slsch = UE->slsch[thread_id][gNB_id]->harq_processes[harq_pid];
  harq_process_slsch->status = SCH_IDLE;

  for (int nl = 0; nl < Nl; nl++) {
    free_and_zero(tx_layers[nl]);
    free_and_zero(tx_precoding[nl]);
  }
  free_and_zero(tx_layers);
  free_and_zero(tx_precoding);

  ///////////
  ////////////////////////////////////////////////////////////////////////

}


uint8_t nr_ue_pssch_common_procedures(PHY_VARS_NR_UE *UE,
                                      uint8_t slot,
                                      NR_DL_FRAME_PARMS *frame_parms,
                                      uint8_t n_antenna_ports) {

  int tx_offset, ap;
  int32_t **txdata;
  int32_t **txdataF;

  /////////////////////////IFFT///////////////////////
  ///////////

  tx_offset = frame_parms->get_samples_slot_timestamp(slot, frame_parms, 0);

  // clear the transmit data array for the current subframe
  /*for (int aa=0; aa<UE->frame_parms.nb_antennas_tx; aa++) {
	  memset(&UE->common_vars.txdata[aa][tx_offset],0,UE->frame_parms.samples_per_slot*sizeof(int32_t));
	  //memset(&UE->common_vars.txdataF[aa][tx_offset],0,UE->frame_parms.samples_per_slot*sizeof(int32_t));
  }*/


  txdata = UE->common_vars.txdata;
  txdataF = UE->common_vars.txdataF;

  int symb_offset = (slot%frame_parms->slots_per_subframe)*frame_parms->symbols_per_slot;
  for(ap = 0; ap < n_antenna_ports; ap++) {
    for (int s=0;s<NR_NUMBER_OF_SYMBOLS_PER_SLOT;s++){
      c16_t *this_symbol = (c16_t *)&txdataF[ap][frame_parms->ofdm_symbol_size * s];
      c16_t rot=frame_parms->symbol_rotation[1][s + symb_offset];
      LOG_D(PHY,"rotating txdataF symbol %d (%d) => (%d.%d)\n",
	    s,
	    s + symb_offset,
	    rot.r, rot.i);

      if (frame_parms->N_RB_UL & 1) {
        rotate_cpx_vector(this_symbol, &rot, this_symbol,
                          (frame_parms->N_RB_UL + 1) * 6, 15);
        rotate_cpx_vector(this_symbol + frame_parms->first_carrier_offset - 6,
                          &rot,
                          this_symbol + frame_parms->first_carrier_offset - 6,
                          (frame_parms->N_RB_UL + 1) * 6, 15);
      } else {
        rotate_cpx_vector(this_symbol, &rot, this_symbol,
                          frame_parms->N_RB_UL * 6, 15);
        rotate_cpx_vector(this_symbol + frame_parms->first_carrier_offset,
                          &rot,
                          this_symbol + frame_parms->first_carrier_offset,
                          frame_parms->N_RB_UL * 6, 15);
      }
    }
  }

  for (ap = 0; ap < n_antenna_ports; ap++) {
    if (frame_parms->Ncp == 1) { // extended cyclic prefix
      PHY_ofdm_mod(txdataF[ap],
                   &txdata[ap][tx_offset],
                   frame_parms->ofdm_symbol_size,
                   12,
                   frame_parms->nb_prefix_samples,
                   CYCLIC_PREFIX);
    } else { // normal cyclic prefix
      nr_normal_prefix_mod(txdataF[ap],
                           &txdata[ap][tx_offset],
                           14,
                           frame_parms,
                           slot);
    }
  }

  ///////////
  ////////////////////////////////////////////////////
  return 0;
}
