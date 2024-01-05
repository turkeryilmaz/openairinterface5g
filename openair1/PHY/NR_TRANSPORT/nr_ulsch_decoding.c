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

/*! \file PHY/NR_TRANSPORT/nr_ulsch_decoding.c
* \brief Top-level routines for decoding  LDPC (ULSCH) transport channels from 38.212, V15.4.0 2018-12
* \author Ahmed Hussein
* \date 2019
* \version 0.1
* \company Fraunhofer IIS
* \email: ahmed.hussein@iis.fraunhofer.de
* \note
* \warning
*/


// [from gNB coding]
#include "PHY/defs_gNB.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/CODING/nrLDPC_decoder/nrLDPC_decoder_offload_xdma.h"
#include "PHY/NR_TRANSPORT/nr_transport_common_proto.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/NR_TRANSPORT/nr_ulsch.h"
#include "PHY/NR_TRANSPORT/nr_dlsch.h"
#include "SCHED_NR/sched_nr.h"
#include "SCHED_NR/fapi_nr_l1.h"
#include "defs.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include "common/utils/LOG/log.h"
#include <syscall.h>
//#define DEBUG_ULSCH_DECODING
//#define gNB_DEBUG_TRACE

#define OAI_UL_LDPC_MAX_NUM_LLR 27000//26112 // NR_LDPC_NCOL_BG1*NR_LDPC_ZMAX = 68*384
//#define DEBUG_CRC
#ifdef DEBUG_CRC
#define PRINT_CRC_CHECK(a) a
#else
#define PRINT_CRC_CHECK(a)
#endif

//extern double cpuf;

void free_gNB_ulsch(NR_gNB_ULSCH_t *ulsch, uint16_t N_RB_UL)
{

  uint16_t a_segments = MAX_NUM_NR_ULSCH_SEGMENTS_PER_LAYER*NR_MAX_NB_LAYERS;  //number of segments to be allocated

  if (N_RB_UL != 273) {
    a_segments = a_segments*N_RB_UL;
    a_segments = a_segments/273 +1;
  }

  if (ulsch->harq_process) {
    if (ulsch->harq_process->b) {
      free_and_zero(ulsch->harq_process->b);
      ulsch->harq_process->b = NULL;
    }
    for (int r = 0; r < a_segments; r++) {
      free_and_zero(ulsch->harq_process->c[r]);
      free_and_zero(ulsch->harq_process->d[r]);
    }
    free_and_zero(ulsch->harq_process->c);
    free_and_zero(ulsch->harq_process->d);
    free_and_zero(ulsch->harq_process->d_to_be_cleared);
    free_and_zero(ulsch->harq_process);
    ulsch->harq_process = NULL;
  }
}

NR_gNB_ULSCH_t new_gNB_ulsch(uint8_t max_ldpc_iterations, uint16_t N_RB_UL)
{

  uint16_t a_segments = MAX_NUM_NR_ULSCH_SEGMENTS_PER_LAYER*NR_MAX_NB_LAYERS;  //number of segments to be allocated

  if (N_RB_UL != 273) {
    a_segments = a_segments*N_RB_UL;
    a_segments = a_segments/273 +1;
  }

  uint32_t ulsch_bytes = a_segments * 1056; // allocated bytes per segment
  NR_gNB_ULSCH_t ulsch = {0};

  ulsch.max_ldpc_iterations = max_ldpc_iterations;
  ulsch.harq_pid = -1;
  ulsch.active = false;

  NR_UL_gNB_HARQ_t *harq = malloc16_clear(sizeof(*harq));
  init_abort(&harq->abort_decode);
  ulsch.harq_process = harq;
  harq->b = malloc16_clear(ulsch_bytes * sizeof(*harq->b));
  harq->c = malloc16_clear(a_segments * sizeof(*harq->c));
  harq->d = malloc16_clear(a_segments * sizeof(*harq->d));
  for (int r = 0; r < a_segments; r++) {
    harq->c[r] = malloc16_clear(8448 * sizeof(*harq->c[r]));
    harq->d[r] = malloc16_clear(68 * 384 * sizeof(*harq->d[r]));
  }
  harq->d_to_be_cleared = calloc(a_segments, sizeof(bool));
  AssertFatal(harq->d_to_be_cleared != NULL, "out of memory\n");
  return(ulsch);
}

static void nr_processULSegment(void *arg)
{
  ldpcDecode_t *rdata = (ldpcDecode_t *)arg;
  NR_UL_gNB_HARQ_t *ulsch_harq = rdata->ulsch_harq;
  t_nrLDPC_dec_params *p_decoderParms = &rdata->decoderParms;
  const int Kr = ulsch_harq->K;
  const int Kr_bytes = Kr >> 3;
  const int K_bits_F = Kr - ulsch_harq->F;
  const int r = rdata->segment_r;
  const int A = rdata->A;
  const int E = rdata->E;
  const int Qm = rdata->Qm;
  const int rv_index = rdata->rv_index;
  const int r_offset = rdata->r_offset;
  const uint8_t kc = rdata->Kc;
  short *ulsch_llr = rdata->ulsch_llr;
  const int max_ldpc_iterations = p_decoderParms->numMaxIter;
  int8_t llrProcBuf[OAI_UL_LDPC_MAX_NUM_LLR] __attribute__((aligned(32)));

  t_nrLDPC_time_stats procTime = {0};
  t_nrLDPC_time_stats *p_procTime = &procTime;

  ////////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////// nr_deinterleaving_ldpc ///////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////// ulsch_llr =====> ulsch_harq->e //////////////////////////////

  /// code blocks after bit selection in rate matching for LDPC code (38.212 V15.4.0 section 5.4.2.1)
  int16_t harq_e[E];

  nr_deinterleaving_ldpc(E, Qm, harq_e, ulsch_llr + r_offset);

  // for (int i =0; i<16; i++)
  //          printf("rx output deinterleaving w[%d]= %d r_offset %d\n", i,ulsch_harq->w[r][i], r_offset);


  //////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////// nr_rate_matching_ldpc_rx ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  ///////////////////////// ulsch_harq->e =====> ulsch_harq->d /////////////////////////


  if (nr_rate_matching_ldpc_rx(rdata->tbslbrm,
                               p_decoderParms->BG,
                               p_decoderParms->Z,
                               ulsch_harq->d[r],
                               harq_e,
                               ulsch_harq->C,
                               rv_index,
                               ulsch_harq->d_to_be_cleared[r],
                               E,
                               ulsch_harq->F,
                               Kr - ulsch_harq->F - 2 * (p_decoderParms->Z))
      == -1) {

    LOG_E(PHY, "ulsch_decoding.c: Problem in rate_matching\n");
    rdata->decodeIterations = max_ldpc_iterations + 1;
    return;
  }

  ulsch_harq->d_to_be_cleared[r] = false;

  memset(ulsch_harq->c[r], 0, Kr_bytes);
  p_decoderParms->crc_type = crcType(ulsch_harq->C, A);
  p_decoderParms->E = lenWithCrc(ulsch_harq->C, A);
  // start_meas(&phy_vars_gNB->ulsch_ldpc_decoding_stats);

  // set first 2*Z_c bits to zeros

  int16_t z[68 * 384 + 16] __attribute__((aligned(16)));

  memset(z, 0, 2 * ulsch_harq->Z * sizeof(*z));
  // set Filler bits
  memset(z + K_bits_F, 127, ulsch_harq->F * sizeof(*z));
  // Move coded bits before filler bits
  memcpy(z + 2 * ulsch_harq->Z, ulsch_harq->d[r], (K_bits_F - 2 * ulsch_harq->Z) * sizeof(*z));
  // skip filler bits
  memcpy(z + Kr, ulsch_harq->d[r] + (Kr - 2 * ulsch_harq->Z), (kc * ulsch_harq->Z - Kr) * sizeof(*z));
  // Saturate coded bits before decoding into 8 bits values
  simde__m128i *pv = (simde__m128i *)&z;
  int8_t l[68 * 384 + 16] __attribute__((aligned(16)));
  simde__m128i *pl = (simde__m128i *)&l;
  for (int i = 0, j = 0; j < ((kc * ulsch_harq->Z) >> 4) + 1; i += 2, j++) {
    pl[j] = simde_mm_packs_epi16(pv[i], pv[i + 1]);
  }
  //////////////////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////// nrLDPC_decoder /////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////

  ////////////////////////////////// pl =====> llrProcBuf //////////////////////////////////
  rdata->decodeIterations =
      ldpc_interface.LDPCdecoder(p_decoderParms, 0, 0, 0, l, llrProcBuf, p_procTime, &ulsch_harq->abort_decode);

  if (rdata->decodeIterations <= p_decoderParms->numMaxIter)
    memcpy(ulsch_harq->c[r],llrProcBuf,  Kr>>3);
}

int decode_offload(PHY_VARS_gNB *phy_vars_gNB,
                   uint8_t ULSCH_id,
                   short *ulsch_llr,
                   nfapi_nr_pusch_pdu_t *pusch_pdu,
                   t_nrLDPC_dec_params *decParams,
                   uint8_t harq_pid,
                   uint32_t G)
{
  NR_gNB_ULSCH_t *ulsch = &phy_vars_gNB->ulsch[ULSCH_id];
  NR_UL_gNB_HARQ_t *harq_process = ulsch->harq_process;
  int16_t z_ol[LDPC_MAX_CB_SIZE] __attribute__((aligned(16)));
  int8_t l_ol[LDPC_MAX_CB_SIZE] __attribute__((aligned(16)));
  uint8_t Qm = pusch_pdu->qam_mod_order;
  uint8_t n_layers = pusch_pdu->nrOfLayers;
  const int Kr = harq_process->K;
  const int Kr_bytes = Kr >> 3;
  uint32_t A = (harq_process->TBS) << 3;
  const int kc = decParams->BG == 2 ? 52 : 68;
  ulsch->max_ldpc_iterations = 20;
  int decodeIterations = 2;
  int r_offset = 0, offset = 0;
  for (int r = 0; r < harq_process->C; r++) {
    int E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
    memset(harq_process->c[r], 0, Kr_bytes);
    decParams->R = nr_get_R_ldpc_decoder(pusch_pdu->pusch_data.rv_index,
                                         E,
                                         decParams->BG,
                                         decParams->Z,
                                         &harq_process->llrLen,
                                         harq_process->round);

    memcpy(z_ol, ulsch_llr + r_offset, E * sizeof(short));
    simde__m128i *pv_ol128 = (simde__m128i *)&z_ol;
    simde__m128i *pl_ol128 = (simde__m128i *)&l_ol;
    for (int i = 0, j = 0; j < ((kc * harq_process->Z) >> 4) + 1; i += 2, j++) {
      pl_ol128[j] = simde_mm_packs_epi16(pv_ol128[i], pv_ol128[i + 1]);
    }
    decParams->E = E;
    decParams->rv = pusch_pdu->pusch_data.rv_index;
    decParams->F = harq_process->F;
    decParams->Qm = Qm;
    decodeIterations =
        ldpc_interface_offload
            .LDPCdecoder(decParams, harq_pid, ULSCH_id, r, (int8_t *)&pl_ol128[0], (int8_t *)harq_process->c[r], NULL, NULL);
    if (decodeIterations < 0) {
      LOG_E(PHY, "ulsch_decoding.c: Problem in LDPC decoder offload\n");
      return -1;
    }
    bool decodeSuccess = check_crc((uint8_t *)harq_process->c[r], lenWithCrc(harq_process->C, A), crcType(harq_process->C, A));
    if (decodeSuccess) {
      memcpy(harq_process->b + offset, harq_process->c[r], Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
      offset += (Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
      harq_process->processedSegments++;
    } else {
      LOG_D(PHY, "uplink segment error %d/%d\n", r, harq_process->C);
      LOG_D(PHY, "ULSCH %d in error\n", ULSCH_id);
    }
    r_offset += E;
  }
  bool crc_valid = false;
  if (harq_process->processedSegments == harq_process->C) {
    // When the number of code blocks is 1 (C = 1) and ulsch_harq->processedSegments = 1, we can assume a good TB because of the
    // CRC check made by the LDPC for early termination, so, no need to perform CRC check twice for a single code block
    crc_valid = true;
    if (harq_process->C > 1) {
      crc_valid = check_crc(harq_process->b, lenWithCrc(1, A), crcType(1, A));
    }
  }
  if (crc_valid) {
    LOG_D(PHY, "ULSCH: Setting ACK for slot %d TBS %d\n", ulsch->slot, harq_process->TBS);
    ulsch->active = false;
    harq_process->round = 0;
    LOG_D(PHY, "ULSCH received ok \n");
    nr_fill_indication(phy_vars_gNB, ulsch->frame, ulsch->slot, ULSCH_id, harq_pid, 0, 0);
  } else {
    LOG_D(PHY,
        "[gNB %d] ULSCH: Setting NAK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d)\n",
        phy_vars_gNB->Mod_id,
        ulsch->frame,
        ulsch->slot,
        harq_pid,
        ulsch->active,
        harq_process->round,
        harq_process->TBS);
    ulsch->handled = 1;
    decodeIterations = ulsch->max_ldpc_iterations + 1;
    LOG_D(PHY, "ULSCH %d in error\n", ULSCH_id);
    nr_fill_indication(phy_vars_gNB, ulsch->frame, ulsch->slot, ULSCH_id, harq_pid, 1, 0);
  }

  ulsch->last_iteration_cnt = decodeIterations;
  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_ULSCH_DECODING,0);
  return 0;
}

/*!
 * \typedef args_fpga_decode_prepare_t
 * \struct args_fpga_decode_prepare_s
 * \brief arguments structure for passing arguments to the nr_ulsch_FPGA_decoding_prepare_blocks function
 *
 * \var multi_indata
 * pointer to the head of the block destination array that is then passed to the FPGA decoding
 * \var no_iteration_ldpc
 * pointer to the number of iteration set by this function
 * \var r_first
 * index of the first block to be prepared within this function
 * \var r_span
 * number of blocks to be prepared within this function
 * \var n_layers
 * number of MIMO layers
 * \var G
 * number of soft channel bits
 * \var ulsch_harq
 * harq process information
 * \var decoderParams
 * decoder parameters
 * \var ulsch
 * uplink shared channel information
 * \var ulsch_llr
 * pointer to the head of the block source array
 * \var rv_index
 * an argument of rate dematching
 * \var E
 * size of the block between deinterleaving and rate matching
 * \var Qm
 * modulation order
 * \var r_offset
 * r index expressed in bits
 * \var tbslbrm
 * an argument of rate dematching
 *
 */
typedef struct args_fpga_decode_prepare_s
{
  int8_t *multi_indata;
  int no_iteration_ldpc;
  uint32_t r_first;
  uint32_t r_span;
  uint8_t n_layers;
  uint32_t G;
  NR_UL_gNB_HARQ_t *ulsch_harq;
  t_nrLDPC_dec_params *decoderParms;
  NR_gNB_ULSCH_t *ulsch;
  short* ulsch_llr; 
  int rv_index;
  int E;
  int Qm;
  int r_offset;
  uint32_t tbslbrm;
  
} args_fpga_decode_prepare_t;

/*!
 * \fn nr_ulsch_FPGA_decoding_prepare_blocks(void *args)
 * \brief prepare blocks for LDPC decoding on FPGA
 *
 * \param args pointer to the arguments of the function in a structure of type args_fpga_decode_prepare_t
 *
 */
void nr_ulsch_FPGA_decoding_prepare_blocks(void *args)
{
  //extract the arguments
  args_fpga_decode_prepare_t *arguments = (args_fpga_decode_prepare_t *)args;
  int8_t *multi_indata = arguments->multi_indata;
  int no_iteration_ldpc = arguments->no_iteration_ldpc;
  uint32_t r_first = arguments->r_first;
  uint32_t r_span = arguments->r_span;
  uint8_t n_layers = arguments->n_layers;
  uint32_t G = arguments->G;
  short* ulsch_llr = arguments->ulsch_llr;
  NR_UL_gNB_HARQ_t *harq_process = arguments->ulsch_harq;
  t_nrLDPC_dec_params *decParams = arguments->decoderParms;
  NR_gNB_ULSCH_t *ulsch = arguments->ulsch;
  int E = arguments->E;
  int Qm = arguments->Qm;
  uint32_t r_offset = arguments->r_offset;

  /* 
   * extract additional required information
   *
   * Kr number of bits per block
   *
   * initialise other required variables
   *
   * dtx_det
   * input_CBoffset
   * kc
   * K_bits_F
   *
   */
  int Kr = harq_process->K;

  uint8_t dtx_det = 0;

  int mbmb = 0;
  if (decParams->BG == 1)
    mbmb = 68;
  else
    mbmb = 52;

  // Calc input CB offset
  int input_CBoffset = decParams->Z * mbmb * 8;
  if ((input_CBoffset & 0x7F) == 0)
    input_CBoffset = input_CBoffset / 8;
  else
    input_CBoffset = 16 * ((input_CBoffset / 128) + 1);

  int kc;
  if (decParams->BG == 2) {
    kc = 52;
  } else {
    kc = 68;
  }

  int K_bits_F = Kr - harq_process->F;

  int16_t z[68 * 384 + 16] __attribute__((aligned(16)));
  simde__m128i *pv = (simde__m128i *)&z;

  /*
   * the function processes r_span blocks starting from block at index r_first in ulsch_llr 
   */
  for(uint32_t r = r_first; r < ( r_first + r_span ); r++)
  {

    E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
    // ----------------------- FPGA pre process ------------------------
    simde__m128i ones = simde_mm_set1_epi8(255); // Generate a vector with all elements set to 255
    simde__m128i *temp_multi_indata = (simde__m128i *)&multi_indata[r * input_CBoffset];
    // -----------------------------------------------------------------

    // code blocks after bit selection in rate matching for LDPC code (38.212 V15.4.0 section 5.4.2.1)
    int16_t harq_e[E];
    // -------------------------------------------------------------------------------------------
    // deinterleaving
    // -------------------------------------------------------------------------------------------
    //start_meas(&phy_vars_gNB->ulsch_deinterleaving_stats);
    nr_deinterleaving_ldpc(E, Qm, harq_e, ulsch_llr + r_offset);
    //stop_meas(&phy_vars_gNB->ulsch_deinterleaving_stats);

    // -------------------------------------------------------------------------------------------
    // dematching
    // -------------------------------------------------------------------------------------------
    //start_meas(&phy_vars_gNB->ulsch_rate_unmatching_stats);
    if (nr_rate_matching_ldpc_rx(arguments->tbslbrm,
                                 decParams->BG,
                                 decParams->Z,
                                 harq_process->d[r],
                                 harq_e,
                                 harq_process->C,
                                 arguments->rv_index,
                                 harq_process->d_to_be_cleared[r],
                                 E,
                                 harq_process->F,
                                 Kr - harq_process->F - 2 * (decParams->Z)
                                ) == -1) 
    {
        //stop_meas(&phy_vars_gNB->ulsch_rate_unmatching_stats);
        LOG_E(PHY, "ulsch_decoding.c: Problem in rate_matching\n");
        no_iteration_ldpc = ulsch->max_ldpc_iterations + 1;
        return;
    } else {
      //stop_meas(&phy_vars_gNB->ulsch_rate_unmatching_stats);
    }

    harq_process->d_to_be_cleared[r] = false;

    memset(harq_process->c[r], 0, Kr >> 3);

    // set first 2*Z_c bits to zeros
    memset(&z[0], 0, 2 * harq_process->Z * sizeof(int16_t));
    // set Filler bits
    memset((&z[0] + K_bits_F), 127, harq_process->F * sizeof(int16_t));
    // Move coded bits before filler bits
    memcpy((&z[0] + 2 * harq_process->Z), harq_process->d[r], (K_bits_F - 2 * harq_process->Z) * sizeof(int16_t));
    // skip filler bits
    memcpy((&z[0] + Kr), harq_process->d[r] + (Kr - 2 * harq_process->Z), (kc * harq_process->Z - Kr) * sizeof(int16_t));

    // Saturate coded bits before decoding into 8 bits values

    for (int i = 0, j = 0; j < ((kc * harq_process->Z) >> 4); i += 2, j++) {
      temp_multi_indata[j] = simde_mm_xor_si128(simde_mm_packs_epi16(pv[i], pv[i + 1]), simde_mm_cmpeq_epi32(ones, ones)); // Perform NOT operation and write the result to temp_multi_indata[j]
    }

    // the last bytes before reaching "kc * harq_process->Z" should not be written 128 bits at a time to avoid overwritting the following block in multi_indata
    simde__m128i tmp = simde_mm_xor_si128(simde_mm_packs_epi16(pv[2*((kc * harq_process->Z) >> 4)], pv[2*((kc * harq_process->Z) >> 4) + 1]), simde_mm_cmpeq_epi32(ones, ones)); // Perform NOT operation and write the result to temp_multi_indata[j]
    int8_t *tmp_p = (int8_t *)&tmp;
    for (int i = 0, j = ((kc * harq_process->Z)&0xfffffff0); j < kc * harq_process->Z; i++, j++) {
      multi_indata[r * input_CBoffset + j] = tmp_p[i];
    }

    r_offset += E;

  }

  arguments->no_iteration_ldpc=no_iteration_ldpc;

}

int decode_xdma(PHY_VARS_gNB *phy_vars_gNB,
                   uint8_t ULSCH_id,
                   short *ulsch_llr,
                   nfapi_nr_pusch_pdu_t *pusch_pdu,
                   t_nrLDPC_dec_params *decParams,
                   uint32_t frame,
                   uint8_t nr_tti_rx,
                   uint8_t harq_pid,
                   uint32_t G)
{
  NR_gNB_ULSCH_t *ulsch = &phy_vars_gNB->ulsch[ULSCH_id];
  NR_UL_gNB_HARQ_t *harq_process = ulsch->harq_process;
  uint8_t Qm = pusch_pdu->qam_mod_order;
  uint8_t n_layers = pusch_pdu->nrOfLayers;
  const int Kr = harq_process->K;
  const int Kr_bytes = Kr >> 3;
  uint32_t A = (harq_process->TBS) << 3;
  const int kc = decParams->BG == 2 ? 52 : 68;
  ulsch->max_ldpc_iterations = 20;
  int r_offset = 0, offset = 0;

  //LDPC decode is offloaded to FPGA using the xdma driver

  int K_bits_F = Kr - harq_process->F;
  //-------------------- FPGA parameter preprocessing ---------------------
  static int8_t multi_indata[27000 * 25]; // FPGA input data
  static int8_t multi_outdata[1100 * 25]; // FPGA output data

  int mbmb = 0;
  if (decParams->BG == 1)
    mbmb = 68;
  else
    mbmb = 52;

  int bg_len = 0;
  if (decParams->BG == 1)
    bg_len = 22;
  else
    bg_len = 10;

  // Calc input CB offset
  int input_CBoffset = decParams->Z * mbmb * 8;
  if ((input_CBoffset & 0x7F) == 0)
    input_CBoffset = input_CBoffset / 8;
  else
    input_CBoffset = 16 * ((input_CBoffset / 128) + 1);

  DecIFConf dec_conf;
  dec_conf.Zc = decParams->Z;
  dec_conf.BG = decParams->BG;
  dec_conf.max_iter = decParams->numMaxIter;
  dec_conf.numCB = harq_process->C;
  dec_conf.numChannelLls = (K_bits_F - 2 * harq_process->Z) + (kc * harq_process->Z - Kr); // input soft bits length, Zc x 66 - length of filler bits
  dec_conf.numFillerBits = harq_process->F; // filler bits length

  dec_conf.max_iter = 8;
  dec_conf.max_schedule = 0;
  dec_conf.SetIdx = 12;
  // dec_conf.max_iter = 8;
  if (dec_conf.BG == 1)
    dec_conf.nRows = 46;
  else
    dec_conf.nRows = 42;

  int out_CBoffset = dec_conf.Zc * bg_len;
  if ((out_CBoffset & 0x7F) == 0)
    out_CBoffset = out_CBoffset / 8;
  else
    out_CBoffset = 16 * ((out_CBoffset / 128) + 1);

#ifdef LDPC_DATA
  printf("\n------------------------\n");
  printf("BG:\t\t%d\n", dec_conf.BG);
  printf("harq_process->B: %d\n", harq_process->B);
  printf("harq_process->C: %d\n", harq_process->C);
  printf("harq_process->K: %d\n", harq_process->K);
  printf("harq_process->Z: %d\n", harq_process->Z);
  printf("harq_process->F: %d\n", harq_process->F);
  printf("numChannelLls:\t %d = (%d - 2 * %d) + (%d * %d - %d)\n", dec_conf.numChannelLls, K_bits_F, harq_process->Z, kc, harq_process->Z, Kr);
  printf("numFillerBits:\t %d\n", harq_process->F);
  printf("------------------------\n");
  // ===================================
  // debug mode
  // ===================================
  FILE *fptr_llr, *fptr_ldpc;
  fptr_llr = fopen("../../../cmake_targets/log/ulsim_ldpc_llr.txt", "w");
  fptr_ldpc = fopen("../../../cmake_targets/log/ulsim_ldpc_output.txt", "w");
  // ===================================
#endif
  //----------------------------------------------------------------------

  int length_dec = lenWithCrc(harq_process->C, A);
  uint8_t crc_type = crcType(harq_process->C, A);
  int no_iteration_ldpc = 2;

  uint8_t dtx_det = 0;
  uint32_t num_threads_prepare_max = &phy_vars_gNB->ldpc_xdma_number_threads_predecoding;
  uint32_t num_threads_prepare = 0;
  uint32_t r_remaining = 0;
  //start the prepare jobs


  for (uint32_t r = 0; r < harq_process->C; r++) {
    int E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
    if (r_remaining == 0 ) {
      void (*nr_ulsch_FPGA_decoding_prepare_blocks_ptr)(void *) = &nr_ulsch_FPGA_decoding_prepare_blocks;
      union ldpcReqUnion id = {.s={ulsch->rnti,frame,nr_tti_rx,0,0}};
      notifiedFIFO_elt_t *req = newNotifiedFIFO_elt(sizeof(args_fpga_decode_prepare_t), id.p, &phy_vars_gNB->respDecode, nr_ulsch_FPGA_decoding_prepare_blocks_ptr);
      args_fpga_decode_prepare_t * args = (args_fpga_decode_prepare_t *) NotifiedFifoData(req);

      args->multi_indata = multi_indata;
      args->no_iteration_ldpc = 2;
      args->r_first = r;
      uint32_t r_span_max = ((harq_process->C-r)%(num_threads_prepare_max-num_threads_prepare))==0 ? (harq_process->C-r)/(num_threads_prepare_max-num_threads_prepare) : ((harq_process->C-r)/(num_threads_prepare_max-num_threads_prepare))+1 ;
      uint32_t r_span = harq_process->C-r<r_span_max ? harq_process->C-r : r_span_max;
      args->r_span = r_span;
      r_remaining = r_span;
      args->n_layers = n_layers;
      args->G = G;
      args->ulsch_harq = harq_process;
      args->decoderParms = decParams;
      args->ulsch = ulsch;
      args->ulsch_llr = ulsch_llr;
      args->rv_index = pusch_pdu->pusch_data.rv_index;
      args->E = E;
      args->Qm = Qm;
      args->r_offset = r_offset;
      args->tbslbrm = pusch_pdu->maintenance_parms_v3.tbSizeLbrmBytes;
      pushTpool(&phy_vars_gNB->threadPool, req);
      LOG_D(PHY, "Added %d block(s) to prepare for decoding, in pipe: %d to %d\n", r_span, r, r+r_span-1);
      num_threads_prepare++;
    }
    r_offset += E;
    offset += (Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
    r_remaining -= 1;
    //////////////////////////////////////////////////////////////////////////////////////////
  }
  
  //reset offset in order to properly fill the output array later
  offset = 0;


  //wait for the prepare jobs to complete
  while(num_threads_prepare>0){
    notifiedFIFO_elt_t *req = (notifiedFIFO_elt_t *)pullTpool(&phy_vars_gNB->respDecode, &phy_vars_gNB->threadPool);
    if (req == NULL)
      LOG_E(PHY, "FPGA decoding preparation: pullTpool returned NULL\n");
    args_fpga_decode_prepare_t *args = (args_fpga_decode_prepare_t *)NotifiedFifoData(req);
    if (args->no_iteration_ldpc > ulsch->max_ldpc_iterations)
      no_iteration_ldpc = ulsch->max_ldpc_iterations + 1;
    num_threads_prepare -= 1;
  }

  //launch decode with FPGA
  // printf("Run the LDPC ------[FPGA version]------\n");
  //==================================================================
  // Xilinx FPGA LDPC decoding function -> nrLDPC_decoder_FPGA_PYM()
  //==================================================================
  start_meas(&phy_vars_gNB->ulsch_ldpc_decoding_stats);
  nrLDPC_decoder_FPGA_PYM((int8_t *)&multi_indata[0], (int8_t *)&multi_outdata[0], dec_conf);
  // printf("Xilinx FPGA -> CB = %d\n", harq_process->C);
  // nrLDPC_decoder_FPGA_PYM((int8_t *)&temp_multi_indata[0], (int8_t *)&multi_outdata[0], dec_conf);
  stop_meas(&phy_vars_gNB->ulsch_ldpc_decoding_stats);

  for (uint32_t r = 0; r < harq_process->C; r++) {
    // -----------------------------------------------------------------------------------------------
    // --------------------- copy FPGA output to harq_process->c[r][i] -------------------------------
    // -----------------------------------------------------------------------------------------------
    if (check_crc((uint8_t *)multi_outdata, length_dec, crc_type)) {
#ifdef PRINT_CRC_CHECK
      LOG_I(PHY, "Segment %d CRC OK\n", r);
#endif
      no_iteration_ldpc = 2;
    } else {
#ifdef PRINT_CRC_CHECK
      LOG_I(PHY, "segment %d CRC NOK\n", r);
#endif
      no_iteration_ldpc = ulsch->max_ldpc_iterations + 1;
    }
    for (int i = 0; i < out_CBoffset; i++) {
      harq_process->c[r][i] = (uint8_t)multi_outdata[i + r * out_CBoffset];
    }
    bool decodeSuccess = (no_iteration_ldpc <= ulsch->max_ldpc_iterations);
    if (decodeSuccess) {
      memcpy(harq_process->b + offset, harq_process->c[r], Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
      offset += (Kr_bytes - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
      harq_process->processedSegments++;
    } else {
      LOG_D(PHY, "uplink segment error %d/%d\n", r, harq_process->C);
      LOG_D(PHY, "ULSCH %d in error\n", ULSCH_id);
      break; // don't even attempt to decode other segments
    }
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_ULSCH_DECODING, 0);

  if (harq_process->processedSegments == harq_process->C) {
    LOG_D(PHY, "[gNB %d] ULSCH: Setting ACK for slot %d TBS %d\n", phy_vars_gNB->Mod_id, ulsch->slot, harq_process->TBS);
    ulsch->active = false;
    harq_process->round = 0;

    LOG_D(PHY, "ULSCH received ok \n");
    nr_fill_indication(phy_vars_gNB, ulsch->frame, ulsch->slot, ULSCH_id, harq_pid, 0, 0);

  } else {
    LOG_D(PHY,
          "[gNB %d] ULSCH: Setting NAK for SFN/SF %d/%d (pid %d, status %d, round %d, TBS %d)\n",
          phy_vars_gNB->Mod_id,
          ulsch->frame,
          ulsch->slot,
          harq_pid,
          ulsch->active,
          harq_process->round,
          harq_process->TBS);
    ulsch->handled = 1;
    no_iteration_ldpc = ulsch->max_ldpc_iterations + 1;
    LOG_D(PHY, "ULSCH %d in error\n", ULSCH_id);
    nr_fill_indication(phy_vars_gNB, ulsch->frame, ulsch->slot, ULSCH_id, harq_pid, 1, 0);
  }
  ulsch->last_iteration_cnt = no_iteration_ldpc;

}

int nr_ulsch_decoding(PHY_VARS_gNB *phy_vars_gNB,
                      uint8_t ULSCH_id,
                      short *ulsch_llr,
                      NR_DL_FRAME_PARMS *frame_parms,
                      nfapi_nr_pusch_pdu_t *pusch_pdu,
                      uint32_t frame,
                      uint8_t nr_tti_rx,
                      uint8_t harq_pid,
                      uint32_t G)
{
  if (!ulsch_llr) {
    LOG_E(PHY, "ulsch_decoding.c: NULL ulsch_llr pointer\n");
    return -1;
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_ULSCH_DECODING, 1);

  NR_gNB_ULSCH_t *ulsch = &phy_vars_gNB->ulsch[ULSCH_id];
  NR_gNB_PUSCH *pusch = &phy_vars_gNB->pusch_vars[ULSCH_id];
  NR_UL_gNB_HARQ_t *harq_process = ulsch->harq_process;

  if (!harq_process) {
    LOG_E(PHY, "ulsch_decoding.c: NULL harq_process pointer\n");
    return -1;
  }

  // ------------------------------------------------------------------
  const uint16_t nb_rb = pusch_pdu->rb_size;
  const uint8_t Qm = pusch_pdu->qam_mod_order;
  const uint8_t mcs = pusch_pdu->mcs_index;
  const uint8_t n_layers = pusch_pdu->nrOfLayers;
  // ------------------------------------------------------------------

  harq_process->processedSegments = 0;
  harq_process->TBS = pusch_pdu->pusch_data.tb_size;

  t_nrLDPC_dec_params decParams = {.check_crc = check_crc};
  decParams.BG = pusch_pdu->maintenance_parms_v3.ldpcBaseGraph;
  const uint32_t A = (harq_process->TBS) << 3;
  NR_gNB_PHY_STATS_t *stats = get_phy_stats(phy_vars_gNB, ulsch->rnti);
  if (stats) {
    stats->frame = frame;
    stats->ulsch_stats.round_trials[harq_process->round]++;
    for (int aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
      stats->ulsch_stats.power[aarx] = dB_fixed_x10(pusch->ulsch_power[aarx]);
      stats->ulsch_stats.noise_power[aarx] = dB_fixed_x10(pusch->ulsch_noise_power[aarx]);
    }
    if (!harq_process->harq_to_be_cleared) {
      stats->ulsch_stats.current_Qm = Qm;
      stats->ulsch_stats.current_RI = n_layers;
      stats->ulsch_stats.total_bytes_tx += harq_process->TBS;
    }
  }

  LOG_D(PHY,
        "ULSCH Decoding, harq_pid %d rnti %x TBS %d G %d mcs %d Nl %d nb_rb %d, Qm %d, Coderate %f RV %d round %d new RX %d\n",
        harq_pid,
        ulsch->rnti,
        A,
        G,
        mcs,
        n_layers,
        nb_rb,
        Qm,
        pusch_pdu->target_code_rate / 10240.0f,
        pusch_pdu->pusch_data.rv_index,
        harq_process->round,
        harq_process->harq_to_be_cleared);

  // [hna] Perform nr_segmenation with input and output set to NULL to calculate only (C, K, Z, F)
  nr_segmentation(NULL,
                  NULL,
                  lenWithCrc(1, A), // size in case of 1 segment
                  &harq_process->C,
                  &harq_process->K,
                  &harq_process->Z, // [hna] Z is Zc
                  &harq_process->F,
                  decParams.BG);

  uint16_t a_segments = MAX_NUM_NR_ULSCH_SEGMENTS_PER_LAYER * n_layers; // number of segments to be allocated
  if (harq_process->C > a_segments) {
    LOG_E(PHY, "nr_segmentation.c: too many segments %d, A %d\n", harq_process->C, A);
    return(-1);
  }
  if (nb_rb != 273) {
    a_segments = a_segments*nb_rb;
    a_segments = a_segments/273 +1;
  }
  if (harq_process->C > a_segments) {
    LOG_E(PHY,"Illegal harq_process->C %d > %d\n",harq_process->C,a_segments);
    return -1;
  }

#ifdef DEBUG_ULSCH_DECODING
  printf("ulsch decoding nr segmentation Z %d\n", harq_process->Z);
  if (!frame % 100)
    printf("K %d C %d Z %d \n", harq_process->K, harq_process->C, harq_process->Z);
  printf("Segmentation: C %d, K %d\n",harq_process->C,harq_process->K);
#endif

  decParams.Z = harq_process->Z;
  decParams.numMaxIter = ulsch->max_ldpc_iterations;
  decParams.outMode = 0;
  decParams.setCombIn = !harq_process->harq_to_be_cleared;
  if (harq_process->harq_to_be_cleared) {
    for (int r = 0; r < harq_process->C; r++)
      harq_process->d_to_be_cleared[r] = true;
    harq_process->harq_to_be_cleared = false;
  }

  if (phy_vars_gNB->ldpc_offload_flag)
    return decode_offload(phy_vars_gNB, ULSCH_id, ulsch_llr, pusch_pdu, &decParams, harq_pid, G);

  if (phy_vars_gNB->ldpc_xdma_flag)
    return decode_xdma(phy_vars_gNB, ULSCH_id, ulsch_llr, pusch_pdu, &decParams, frame, nr_tti_rx, harq_pid, G);

  uint32_t offset = 0, r_offset = 0;
  set_abort(&harq_process->abort_decode, false);
  for (int r = 0; r < harq_process->C; r++) {
    int E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
    union ldpcReqUnion id = {.s = {ulsch->rnti, frame, nr_tti_rx, 0, 0}};
    notifiedFIFO_elt_t *req = newNotifiedFIFO_elt(sizeof(ldpcDecode_t), id.p, &phy_vars_gNB->respDecode, &nr_processULSegment);
    ldpcDecode_t *rdata = (ldpcDecode_t *)NotifiedFifoData(req);
    decParams.R = nr_get_R_ldpc_decoder(pusch_pdu->pusch_data.rv_index,
                                        E,
                                        decParams.BG,
                                        decParams.Z,
                                        &harq_process->llrLen,
                                        harq_process->round);
    rdata->gNB = phy_vars_gNB;
    rdata->ulsch_harq = harq_process;
    rdata->decoderParms = decParams;
    rdata->ulsch_llr = ulsch_llr;
    rdata->Kc = decParams.BG == 2 ? 52 : 68;
    rdata->harq_pid = harq_pid;
    rdata->segment_r = r;
    rdata->nbSegments = harq_process->C;
    rdata->E = E;
    rdata->A = A;
    rdata->Qm = Qm;
    rdata->r_offset = r_offset;
    rdata->Kr_bytes = harq_process->K >> 3;
    rdata->rv_index = pusch_pdu->pusch_data.rv_index;
    rdata->offset = offset;
    rdata->ulsch = ulsch;
    rdata->ulsch_id = ULSCH_id;
    rdata->tbslbrm = pusch_pdu->maintenance_parms_v3.tbSizeLbrmBytes;
    pushTpool(&phy_vars_gNB->threadPool, req);
    LOG_D(PHY, "Added a block to decode, in pipe: %d\n", r);
    r_offset += E;
    offset += ((harq_process->K >> 3) - (harq_process->F >> 3) - ((harq_process->C > 1) ? 3 : 0));
  }
  return harq_process->C;
}
