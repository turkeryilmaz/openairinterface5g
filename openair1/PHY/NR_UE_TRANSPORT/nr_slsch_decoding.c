/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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

// [from nrUE coding]
#include "PHY/defs_UE.h"
#include "PHY/phy_extern_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_slsch.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include <openair2/UTIL/OPT/opt.h>
//#define DEBUG_SLSCH_DECODING
//#define nrUE_DEBUG_TRACE

// [from gNB coding]
/*
#include "PHY/defs_gNB.h"
#include "PHY/phy_extern.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "SCHED_NR/fapi_nr_l1.h"
#include "defs.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"
#include <syscall.h>
*/

#define OAI_UL_LDPC_MAX_NUM_LLR 27000//26112 // NR_LDPC_NCOL_BG1*NR_LDPC_ZMAX = 68*384
//#define PRINT_CRC_CHECK

//extern double cpuf;

void free_nr_ue_slsch_rx(NR_UE_ULSCH_t **slschptr, uint16_t N_RB_UL)
{

  uint16_t a_segments = MAX_NUM_NR_SLSCH_SEGMENTS_PER_LAYER * NR_MAX_NB_LAYERS_SL;  //number of segments to be allocated
  NR_UE_ULSCH_t *slsch = *slschptr;

  if (N_RB_UL != 273) {
    a_segments = a_segments * N_RB_UL;
    a_segments = a_segments / 273 + 1;
  }

  for (int i = 0; i < NR_MAX_SLSCH_HARQ_PROCESSES; i++) {
    if (slsch->harq_processes[i]) {
      if (slsch->harq_processes[i]->b) {
        free_and_zero(slsch->harq_processes[i]->b);
        slsch->harq_processes[i]->b = NULL;
      }
      for (int r = 0; r < a_segments; r++) {
        free_and_zero(slsch->harq_processes[i]->c[r]);
        free_and_zero(slsch->harq_processes[i]->d[r]);
      }
      free_and_zero(slsch->harq_processes[i]->c);
      free_and_zero(slsch->harq_processes[i]->d);
      free_and_zero(slsch->harq_processes[i]);
      slsch->harq_processes[i] = NULL;
    }
  }
  free_and_zero(*slschptr);
}


NR_UE_ULSCH_t *new_nr_ue_slsch_rx(uint16_t N_RB_UL, int number_of_harq_pids, NR_DL_FRAME_PARMS* frame_parms)
{

  int max_layers = (frame_parms->nb_antennas_tx < NR_MAX_NB_LAYERS_SL) ? frame_parms->nb_antennas_tx : NR_MAX_NB_LAYERS_SL;
  uint16_t a_segments = MAX_NUM_NR_SLSCH_SEGMENTS_PER_LAYER * max_layers;  //number of segments to be allocated

  if (N_RB_UL != 273) {
    a_segments = a_segments * N_RB_UL;
    a_segments = a_segments / 273 + 1;
  }

  uint32_t slsch_bytes = a_segments * 1056;  // allocated bytes per segment

  NR_UE_ULSCH_t *slsch = malloc16(sizeof(NR_UE_ULSCH_t));
  DevAssert(slsch);
  memset(slsch, 0, sizeof(*slsch));

  slsch->max_ldpc_iterations = MAX_LDPC_ITERATIONS;

  for (int i = 0; i < number_of_harq_pids; i++) {

    slsch->harq_processes[i] = (NR_UL_UE_HARQ_t *)malloc16_clear(sizeof(NR_UL_UE_HARQ_t));
    slsch->harq_processes[i]->b = (uint8_t*)malloc16_clear(slsch_bytes);
    slsch->harq_processes[i]->c = (uint8_t**)malloc16_clear(a_segments * sizeof(uint8_t *));
    slsch->harq_processes[i]->d = (int16_t**)malloc16_clear(a_segments * sizeof(int16_t *));
    for (int r = 0; r < a_segments; r++) {
      slsch->harq_processes[i]->c[r] = (uint8_t*)malloc16_clear(8448 * sizeof(uint8_t));
      slsch->harq_processes[i]->d[r] = (int16_t*)malloc16_clear((68 * 384) * sizeof(int16_t));
    }
  }

  return(slsch);
}

#ifdef PRINT_CRC_CHECK
  static uint32_t prnt_crc_cnt = 0;
#endif

void nr_processSLSegment(void* arg) {
  ldpcDecode_ue_t *rdata = (ldpcDecode_ue_t*) arg;
  PHY_VARS_NR_UE *ue = rdata->phy_vars_ue;
  NR_UL_UE_HARQ_t *slsch_harq = rdata->slsch_harq;
  t_nrLDPC_dec_params *p_decoderParms = &rdata->decoderParms;
  int length_dec;
  int no_iteration_ldpc;
  int Kr;
  int Kr_bytes;
  int K_bits_F;
  uint8_t crc_type;
  int i;
  int j;
  int r = rdata->segment_r;
  int A = rdata->A;
  int E = rdata->E;
  int Qm = rdata->Qm;
  int rv_index = rdata->rv_index;
  int r_offset = rdata->r_offset;
  uint8_t kc = rdata->Kc;
  short* slsch_llr = rdata->slsch_llr;
  int max_ldpc_iterations = p_decoderParms->numMaxIter;
  int8_t llrProcBuf[OAI_UL_LDPC_MAX_NUM_LLR] __attribute__ ((aligned(32)));

  int16_t  z [68 * 384 + 16] __attribute__ ((aligned(16)));
  int8_t   l [68 * 384 + 16] __attribute__ ((aligned(16)));

  __m128i *pv = (__m128i*)&z;
  __m128i *pl = (__m128i*)&l;

  Kr = slsch_harq->K;
  Kr_bytes = Kr >> 3;
  K_bits_F = Kr - slsch_harq->F;

  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats* p_procTime = &procTime ;

  //start_meas(&ue->slsch_deinterleaving_stats);

  ////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// nr_deinterleaving_ldpc ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////// slsch_llr =====> slsch_harq->e //////////////////////////////

  /// code blocks after bit selection in rate matching for LDPC code (38.212 V15.4.0 section 5.4.2.1)
  int16_t harq_e[3 * 8448];

  nr_deinterleaving_ldpc(E,
                         Qm,
                         harq_e,
                         slsch_llr + r_offset);

  //for (int i = 0; i < 16; i++)
  //          printf("rx output deinterleaving w[%d] = %d r_offset %d\n", i, slsch_harq->w[r][i], r_offset);

  //stop_meas(&ue->slsch_deinterleaving_stats);


 //////////////////////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// nr_rate_matching_ldpc_rx ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////// slsch_harq->e =====> slsch_harq->d /////////////////////////

  //start_meas(&ue->slsch_rate_unmatching_stats);

  if (nr_rate_matching_ldpc_rx(rdata->Tbslbrm,
                               p_decoderParms->BG,
                               p_decoderParms->Z,
                               slsch_harq->d[r],
                               harq_e,
                               slsch_harq->C,
                               rv_index,
                               slsch_harq->new_rx,
                               E,
                               slsch_harq->F,
                               Kr - slsch_harq->F - 2 * (p_decoderParms->Z),
                               true) == -1) {

    stop_meas(&ue->slsch_rate_unmatching_stats);

    LOG_E(NR_PHY, "ulsch_decoding.c: Problem in rate_matching\n");
    rdata->decodeIterations = max_ldpc_iterations + 1;
    return;
  } else {
    stop_meas(&ue->slsch_rate_unmatching_stats);
  }

  memset(slsch_harq->c[r], 0, Kr_bytes);

  if (slsch_harq->C == 1) {
    if (A > 3824)
      crc_type = CRC24_A;
    else
      crc_type = CRC16;

    length_dec = slsch_harq->B;
  } else {
    crc_type = CRC24_B;
    length_dec = (slsch_harq->B + 24 * slsch_harq->C) / slsch_harq->C;
  }

  //start_meas(&ue->slsch_ldpc_decoding_stats);

  //set first 2 * Z_c bits to zeros
  memset(&z[0], 0, 2 * slsch_harq->Z * sizeof(int16_t));
  //set Filler bits
  memset((&z[0] + K_bits_F), 127, slsch_harq->F * sizeof(int16_t));
  //Move coded bits before filler bits
  memcpy((&z[0] + 2 * slsch_harq->Z), slsch_harq->d[r], (K_bits_F - 2 * slsch_harq->Z) * sizeof(int16_t));
  //skip filler bits
  memcpy((&z[0] + Kr), slsch_harq->d[r] + (Kr - 2 * slsch_harq->Z), (kc * slsch_harq->Z - Kr) * sizeof(int16_t));
  //Saturate coded bits before decoding into 8 bits values
  for (i = 0, j = 0; j < ((kc * slsch_harq->Z) >> 4) + 1;  i += 2, j++) {
    pl[j] = _mm_packs_epi16(pv[i], pv[i + 1]);
  }
  //////////////////////////////////////////////////////////////////////////////////////////


  //////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// nrLDPC_decoder /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////// pl =====> llrProcBuf //////////////////////////////////

  no_iteration_ldpc = nrLDPC_decoder(p_decoderParms,
                                     (int8_t*)&pl[0],
                                     llrProcBuf,
                                     p_procTime);

  if (check_crc((uint8_t*)llrProcBuf, length_dec, slsch_harq->F, crc_type)) {
#ifdef PRINT_CRC_CHECK
      LOG_I(NR_PHY, "Segment %d CRC OK, iterations %d/%d\n", r, no_iteration_ldpc, max_ldpc_iterations);
#endif
    rdata->decodeIterations = no_iteration_ldpc;
    if (rdata->decodeIterations > p_decoderParms->numMaxIter) rdata->decodeIterations--;
  } else {
#ifdef PRINT_CRC_CHECK
      LOG_I(NR_PHY, "CRC NOK\n");
#endif
    rdata->decodeIterations = max_ldpc_iterations + 1;
  }

  for (int m = 0; m < Kr >> 3; m++) {
    slsch_harq->c[r][m] = (uint8_t) llrProcBuf[m];
  }

  //stop_meas(&ue->slsch_ldpc_decoding_stats);
}

uint32_t nr_slsch_decoding(PHY_VARS_NR_UE *ue,
                           uint8_t SLSCH_id,
                           short *slsch_llr,
                           NR_DL_FRAME_PARMS *frame_parms,
                           nfapi_nr_pssch_pdu_t *pssch_pdu,
                           uint32_t frame,
                           uint8_t slot,
                           uint8_t harq_pid,
                           uint32_t G) {

  uint32_t A;
  uint32_t r;
  uint32_t r_offset;
  uint32_t offset;
  int E;
  int8_t llrProcBuf[22 * 384];
  int ret = 0;
  int i, j;
  int8_t enable_ldpc_offload = ue->ldpc_offload_flag;
  int16_t  z_ol [68 * 384];
  int8_t   l_ol [68 * 384];
  __m128i *pv_ol128 = (__m128i*)&z_ol;
  __m128i *pl_ol128 = (__m128i*)&l_ol;
  int no_iteration_ldpc = 2;
  int length_dec;
  uint8_t crc_type;
  int K_bits_F;
  int16_t  z [68 * 384 + 16] __attribute__ ((aligned(16)));
  int8_t   l [68 * 384 + 16] __attribute__ ((aligned(16)));

  __m128i *pv = (__m128i*)&z;
  __m128i *pl = (__m128i*)&l;

#ifdef PRINT_CRC_CHECK
  prnt_crc_cnt++;
#endif


  NR_UE_ULSCH_t     *slsch          = ue->slsch_rx[SLSCH_id];
  NR_UE_PSSCH       *pssch          = ue->pssch_vars[SLSCH_id];
  NR_UL_UE_HARQ_t   *harq_process   = slsch->harq_processes[harq_pid];

  if (!harq_process) {
    LOG_E(NR_PHY, "slsch_decoding.c: NULL harq_process pointer\n");
    return 1;
  }
  uint8_t dtx_det = 0;
  t_nrLDPC_dec_params decParams;
  t_nrLDPC_dec_params *p_decParams  = &decParams;

  int Kr;
  int Kr_bytes;

  ue->nbDecode = 0;
  harq_process->processedSegments = 0;

  // ------------------------------------------------------------------
  uint16_t nb_rb          = pssch_pdu->rb_size;
  uint8_t Qm              = pssch_pdu->qam_mod_order;
  uint8_t mcs             = pssch_pdu->mcs_index;
  uint8_t n_layers        = pssch_pdu->nrOfLayers;
  // ------------------------------------------------------------------

  if (!slsch_llr) {
    LOG_E(NR_PHY, "slsch_decoding.c: NULL slsch_llr pointer\n");
    return 1;
  }

  harq_process->TBS = pssch_pdu->pssch_data.tb_size;
  harq_process->round = nr_rv_to_round(pssch_pdu->pssch_data.rv_index);

  harq_process->new_rx = false; // flag to indicate if this is a new reception for this harq (initialized to false)
  dtx_det = 0;
  if (harq_process->round == 0) {
    harq_process->new_rx = true;
    harq_process->ndi = pssch_pdu->pssch_data.new_data_indicator;
  }

  // this happens if there was a DTX in round 0
  if (harq_process->ndi != pssch_pdu->pssch_data.new_data_indicator) {
    harq_process->new_rx = true;
    harq_process->ndi = pssch_pdu->pssch_data.new_data_indicator;

    LOG_D(NR_PHY, "Missed ULSCH detection. NDI toggled but rv %d does not correspond to first reception\n", pssch_pdu->pssch_data.rv_index);
  }

  A = (harq_process->TBS) << 3;

  // target_code_rate is in 0.1 units
  float Coderate = (float) pssch_pdu->target_code_rate / 10240.0f;

  LOG_D(NR_PHY, "SLSCH Decoding, harq_pid %d TBS %d G %d mcs %d Nl %d nb_rb %d, Qm %d, Coderate %f RV %d round %d\n",
        harq_pid, A, G, mcs, n_layers, nb_rb, Qm, Coderate, pssch_pdu->pssch_data.rv_index, harq_process->round);

  p_decParams->BG = pssch_pdu->maintenance_parms_v3.ldpcBaseGraph;
  int kc;
  if (p_decParams->BG == 2){
    kc = 52;
    if (Coderate < 0.3333) {
      p_decParams->R = 15;
    }
    else if (Coderate < 0.6667) {
      p_decParams->R = 13;
    }
    else {
      p_decParams->R = 23;
    }
  } else {
    kc = 68;
    if (Coderate < 0.6667) {
      p_decParams->R = 13;
    }
    else if (Coderate < 0.8889) {
      p_decParams->R = 23;
    }
    else {
      p_decParams->R = 89;
    }
  }

  NR_UE_SLSCH_STATS_t *stats = NULL;
  int first_free = -1;
  for (int i = 0; i < NUMBER_OF_NR_SLSCH_STATS_MAX; i++) {
    if (ue->slsch_stats[i].rnti == 0 && first_free == -1) {
      first_free = i;
      stats = &ue->slsch_stats[i];
    }
    if (ue->slsch_stats[i].rnti == slsch->rnti) {
      stats = &ue->slsch_stats[i];
      break;
    }
  }
  if (stats) {
    stats->frame = frame;
    stats->rnti = slsch->rnti;
    stats->round_trials[harq_process->round]++;
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
       stats->power[aarx] = dB_fixed_x10(pssch->slsch_power[aarx]);
       stats->noise_power[aarx] = dB_fixed_x10(pssch->slsch_noise_power[aarx]);
    }
    if (harq_process->new_rx == 0) {
      stats->current_Qm = Qm;
      stats->current_RI = n_layers;
      stats->total_bytes_tx += harq_process->TBS;
    }
  }
  if (A > 3824)
    harq_process->B = A + 24;
  else
    harq_process->B = A + 16;

// [hna] Perform nr_segmenation with input and output set to NULL to calculate only (B, C, K, Z, F)
  nr_segmentation(NULL,
                  NULL,
                  harq_process->B,
                  &harq_process->C,
                  &harq_process->K,
                  &harq_process->Z, // [hna] Z is Zc
                  &harq_process->F,
                  p_decParams->BG);

  if (harq_process->C > MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER*n_layers) {
    LOG_E(NR_PHY, "nr_segmentation.c: too many segments %d, B %d\n",harq_process->C,harq_process->B);
    return(-1);
  }


#ifdef DEBUG_SLSCH_DECODING
  printf("slsch decoding nr segmentation Z %d\n", harq_process->Z);
  if (!frame % 100)
    printf("K %d C %d Z %d \n", harq_process->K, harq_process->C, harq_process->Z);
#endif

  p_decParams->Z = harq_process->Z;


  p_decParams->numMaxIter = slsch->max_ldpc_iterations;
  p_decParams->outMode = 0;

  r_offset = 0;

  uint16_t a_segments = MAX_NUM_NR_ULSCH_SEGMENTS_PER_LAYER * n_layers;  //number of segments to be allocated

  if (nb_rb != 273) {
    a_segments = a_segments * nb_rb;
    a_segments = a_segments / 273 + 1;
  }

  if (harq_process->C > a_segments) {
    LOG_E(NR_PHY, "Illegal harq_process->C %d > %d\n", harq_process->C, a_segments);
    return 1;
  }

#ifdef DEBUG_ULSCH_DECODING
  printf("Segmentation: C %d, K %d\n", harq_process->C, harq_process->K);
#endif

  Kr = harq_process->K;
  Kr_bytes = Kr >> 3;
  offset = 0;

  if (enable_ldpc_offload) {
    if (harq_process->C == 1) {
      if (A > 3824)
        crc_type = CRC24_A;
      else
        crc_type = CRC16;

      length_dec = harq_process->B;
    } else {
      crc_type = CRC24_B;
      length_dec = (harq_process->B + 24 * harq_process->C) / harq_process->C;
    }

    for (r = 0; r < harq_process->C; r++) {
      E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
      memset(harq_process->c[r], 0, Kr_bytes);

      if ((dtx_det == 0) && (pssch_pdu->pssch_data.rv_index == 0)) {
        if (mcs > 9) {
          memcpy((&z_ol[0]), slsch_llr + r_offset, E * sizeof(short));

          for (i = 0, j = 0; j < ((kc * harq_process->Z) >> 4) + 1;  i += 2, j++)
          {
            pl_ol128[j] = _mm_packs_epi16(pv_ol128[i], pv_ol128[i + 1]);
          }

          ret = nrLDPC_decoder_offload(p_decParams, harq_pid,
              SLSCH_id, r,
              pssch_pdu->pssch_data.rv_index,
              harq_process->F,
              E,
              Qm,
              (int8_t*)&pl_ol128[0],
              llrProcBuf, 1);
          if (ret < 0) {
            LOG_E(NR_PHY, "slsch_decoding.c: Problem in LDPC decoder offload\n");
            no_iteration_ldpc = slsch->max_ldpc_iterations + 1;
            return 1;
          }
        } else {
          K_bits_F = Kr - harq_process->F;

          t_nrLDPC_time_stats procTime = {0};
          t_nrLDPC_time_stats* p_procTime     = &procTime ;
          /// code blocks after bit selection in rate matching for LDPC code (38.212 V15.4.0 section 5.4.2.1)
          int16_t harq_e[3 * 8448];

          nr_deinterleaving_ldpc(E, Qm, harq_e, slsch_llr + r_offset);

          if (nr_rate_matching_ldpc_rx(pssch_pdu->maintenance_parms_v3.tbSizeLbrmBytes,
              p_decParams->BG,
              p_decParams->Z,
              harq_process->d[r],
              harq_e,
              harq_process->C,
              pssch_pdu->pssch_data.rv_index,
              harq_process->new_rx,
              E,
              harq_process->F,
              Kr-harq_process->F - 2 * (p_decParams->Z),
              true) == -1) {

            LOG_E(NR_PHY, "slsch_decoding.c: Problem in rate_matching\n");
            no_iteration_ldpc = slsch->max_ldpc_iterations + 1;
            return 1;
          }

          //set first 2*Z_c bits to zeros
          memset(&z[0], 0, 2 * harq_process->Z * sizeof(int16_t));
          //set Filler bits
          memset((&z[0] + K_bits_F), 127, harq_process->F * sizeof(int16_t));
          //Move coded bits before filler bits
          memcpy((&z[0] + 2 * harq_process->Z), harq_process->d[r], (K_bits_F - 2 * harq_process->Z) * sizeof(int16_t));
          //skip filler bits
          memcpy((&z[0] + Kr), harq_process->d[r] + (Kr - 2 * harq_process->Z), (kc * harq_process->Z - Kr) * sizeof(int16_t));
          //Saturate coded bits before decoding into 8 bits values
          for (i = 0, j = 0; j < ((kc * harq_process->Z) >> 4) + 1; i += 2, j++) {
            pl[j] = _mm_packs_epi16(pv[i], pv[i+1]);
          }

          no_iteration_ldpc = nrLDPC_decoder(p_decParams,
                    (int8_t*)&pl[0],
                    llrProcBuf,
                    p_procTime);

        }

        for (int m = 0; m < Kr >> 3; m++) {
          harq_process->c[r][m] = (uint8_t) llrProcBuf[m];
        }

        if (check_crc((uint8_t*)llrProcBuf, length_dec, harq_process->F, crc_type)) {
    #ifdef PRINT_CRC_CHECK
          LOG_I(NR_PHY, "Segment %d CRC OK\n", r);
    #endif
          no_iteration_ldpc = 2;
        } else {
    #ifdef PRINT_CRC_CHECK
          LOG_I(NR_PHY, "segment %d CRC NOK\n", r);
    #endif
          no_iteration_ldpc = slsch->max_ldpc_iterations + 1;
        }
        r_offset += E;
        /*
          for (int k = 0; k < 8; k++) {
            printf("output decoder [%d] =  0x%02x \n", k, harq_process->c[r][k]);
            printf("llrprocbuf [%d] =  %x adr %p\n", k, llrProcBuf[k], llrProcBuf+k);
          }
        */
      } else {
        dtx_det = 0;
        no_iteration_ldpc = slsch->max_ldpc_iterations + 1;
      }
      bool decodeSuccess = (no_iteration_ldpc <= slsch->max_ldpc_iterations);
      if (decodeSuccess) {
        memcpy(harq_process->b + offset,
              harq_process->c[r],
              Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
              offset += (Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
        harq_process->processedSegments++;
      } else {
        LOG_D(NR_PHY, "sidelink segment error %d/%d\n", r, harq_process->C);
        LOG_D(NR_PHY, "SLSCH %d in error\n", SLSCH_id);
        break; //don't even attempt to decode other segments
      }
    }

    if ((harq_process->processedSegments == (harq_process->C))) {
      LOG_D(NR_PHY, "[UE %d] SLSCH: Setting ACK for slot %d TBS %d\n",
            ue->Mod_id, harq_process->slot, harq_process->TBS);
      harq_process->status = SCH_IDLE;
      harq_process->round = 0;
      slsch->harq_mask &= ~(1 << harq_pid);

      LOG_D(NR_PHY, "SLSCH received ok \n");
      //nr_fill_indication(ue, harq_process->frame, harq_process->slot, SLSCH_id, harq_pid, 0, 0);

    } else {
      LOG_D(NR_PHY, "[UE %d] SLSCH: Setting NAK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d) r %d\n",
            ue->Mod_id, harq_process->frame, harq_process->slot,
            harq_pid,harq_process->status, harq_process->round, harq_process->TBS, r);
      harq_process->handled = 1;
      no_iteration_ldpc = slsch->max_ldpc_iterations + 1;
      LOG_D(NR_PHY, "SLSCH %d in error\n", SLSCH_id);
      //nr_fill_indication(ue, harq_process->frame, harq_process->slot, SLSCH_id, harq_pid, 1, 0);
    }
    slsch->last_iteration_cnt = no_iteration_ldpc;
  } else {
    dtx_det = 0;
    void (*nr_processSLSegment_ptr)(void*) = &nr_processSLSegment;

    for (r = 0; r < harq_process->C; r++) {

      E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
      union ldpcReqUnion_ue id = {.s = {slsch->rnti, frame, slot, 0, 0}};
      notifiedFIFO_elt_t *req = newNotifiedFIFO_elt(sizeof(ldpcDecode_ue_t), id.p, &ue->respDecode, nr_processSLSegment_ptr);
      ldpcDecode_ue_t *rdata = (ldpcDecode_ue_t *) NotifiedFifoData(req);

      rdata->phy_vars_ue = ue;
      rdata->slsch_harq = harq_process;
      rdata->decoderParms = decParams;
      rdata->slsch_llr = slsch_llr;
      rdata->Kc = kc;
      rdata->harq_pid = harq_pid;
      rdata->segment_r = r;
      rdata->nbSegments = harq_process->C;
      rdata->E = E;
      rdata->A = A;
      rdata->Qm = Qm;
      rdata->r_offset = r_offset;
      rdata->Kr_bytes = Kr_bytes;
      rdata->rv_index = pssch_pdu->pssch_data.rv_index;
      rdata->offset = offset;
      rdata->slsch = slsch;
      rdata->slsch_id = SLSCH_id;
      rdata->Tbslbrm = pssch_pdu->maintenance_parms_v3.tbSizeLbrmBytes;
      pushTpool(&ue->threadPool, req);
      ue->nbDecode++;
      LOG_D(NR_PHY, "Added a block to decode, in pipe: %d\n", ue->nbDecode);
      r_offset += E;
      offset += (Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
      //////////////////////////////////////////////////////////////////////////////////////////
    }
  }
  return 1;
}
