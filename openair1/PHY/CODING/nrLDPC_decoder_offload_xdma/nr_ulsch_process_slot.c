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
//#define PRINT_CRC_CHECK

//extern double cpuf;

#include "SCHED_NR/phy_procedures_nr_gNB.h"
#include "common/utils/thread_pool/task_manager.h"
#include <stdint.h>
#include <time.h>

#include <stdalign.h>

#include <omp.h>
#include "PHY/NR_TRANSPORT/nr_dci.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_interface.h"
#include "nfapi/open-nFAPI/nfapi/public_inc/nfapi_nr_interface.h"
#include "PHY/INIT/nr_phy_init.h"
#include "PHY/MODULATION/nr_modulation.h"
#include "PHY/NR_UE_TRANSPORT/srs_modulation_nr.h"
#include "T.h"
#include "executables/nr-softmodem.h"
#include "executables/softmodem-common.h"
#include "nfapi/oai_integration/vendor_ext.h"
//#include "NR_SRS-ResourceSet.h"

#include "assertions.h"

#include <time.h>

#include "PHY/CODING/nrLDPC_decoder_offload_xdma/nrLDPC_decoder_offload_xdma.h" // XDMA header file
#define NUM_THREADS_PREPARE 8

static inline
int64_t time_now_us(void)
{
  struct timespec tms;

  /* The C11 way */
  /* if (! timespec_get(&tms, TIME_UTC))  */

  /* POSIX.1-2008 way */
  if (clock_gettime(CLOCK_REALTIME,&tms)) {
    return -1;
  }
  /* seconds, multiplied with 1 million */
  int64_t micros = tms.tv_sec * 1000000;
  /* Add full microseconds */
  micros += tms.tv_nsec/1000;
  /* round up if necessary */
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  return micros;
}

#ifdef PRINT_CRC_CHECK
  static uint32_t prnt_crc_cnt = 0;
#endif

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
 * \var decode
 * ldpcDecode_t structure containing required information for decoding
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
  ldpcDecode_t decode;
  
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
  ldpcDecode_t *decode = &arguments->decode;

  /* 
   * extract all required information from decode
   *
   * ulsch_llr pointer to the head of the block source array
   * harq_process harq process information
   * decParams decoder parameters
   * phy_vars_gNB informations on the gNB
   * ulsch uplink shared channel information
   * E size of the block between deinterleaving and rate matching
   * Qm modulation order
   * G total number of coded bits available for transmission of the transport block
   * Kr number of bits per block
   * r_offset r index expressed in bits
   *
   * initialise other required variables
   *
   * dtx_det
   * input_CBoffset
   * kc
   * K_bits_F
   *
   */
  short* ulsch_llr = decode->ulsch_llr;
  NR_UL_gNB_HARQ_t *harq_process = decode->ulsch_harq;
  t_nrLDPC_dec_params decParams = decode->decoderParms;
  //PHY_VARS_gNB *phy_vars_gNB = decode->gNB;
  NR_gNB_ULSCH_t *ulsch = decode->ulsch;
  int E = decode->E;
  int Qm = decode->Qm;
  int Kr = harq_process->K;
  uint32_t r_offset = decode->r_offset;

  uint8_t dtx_det = 0;

  int mbmb = 0;
  if (decParams.BG == 1)
    mbmb = 68;
  else
    mbmb = 52;

  // Calc input CB offset
  int input_CBoffset = decParams.Z * mbmb * 8;
  if ((input_CBoffset & 0x7F) == 0)
    input_CBoffset = input_CBoffset / 8;
  else
    input_CBoffset = 16 * ((input_CBoffset / 128) + 1);

  int kc;
  if (decParams.BG == 2) {
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
    memset(harq_process->c[r], 0, Kr >> 3);
    // ----------------------- FPGA pre process ------------------------
    simde__m128i ones = simde_mm_set1_epi8(255); // Generate a vector with all elements set to 255
    simde__m128i *temp_multi_indata = (simde__m128i *)&multi_indata[r * input_CBoffset];
    // -----------------------------------------------------------------

    decParams.R = nr_get_R_ldpc_decoder(decode->rv_index, E, decParams.BG, decParams.Z, &harq_process->llrLen, harq_process->round);

    if ((dtx_det == 0) && (decode->rv_index == 0)) {

      /// code blocks after bit selection in rate matching for LDPC code (38.212 V15.4.0 section 5.4.2.1)
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
      if (nr_rate_matching_ldpc_rx(decode->tbslbrm,
                                   decParams.BG,
                                   decParams.Z,
                                   harq_process->d[r],
                                   harq_e,
                                   harq_process->C,
                                   decode->rv_index,
                                   harq_process->d_to_be_cleared[r],
                                   E,
                                   harq_process->F,
                                   Kr - harq_process->F - 2 * (decParams.Z)
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

    } else {
      dtx_det = 0;
      no_iteration_ldpc = ulsch->max_ldpc_iterations + 1;
    }

  }

  arguments->no_iteration_ldpc=no_iteration_ldpc;

}

uint32_t nr_ulsch_decoding_tb(PHY_VARS_gNB *phy_vars_gNB,
                           uint8_t ULSCH_id,
                           short *ulsch_llr,
                           NR_DL_FRAME_PARMS *frame_parms,
                           nfapi_nr_pusch_pdu_t *pusch_pdu,
                           uint32_t frame,
                           uint8_t nr_tti_rx,
                           uint8_t harq_pid,
                           uint32_t G) {

  //const int64_t now = time_now_us(); 
  //printf("id %lu nr_ulsch_decoding %lu \n", now, pthread_self());


  uint32_t r;
  int E;
  int8_t llrProcBuf[22*384];
  int ret = 0;
  int i,j;
  int8_t enable_ldpc_offload = phy_vars_gNB->ldpc_offload_flag;
  int16_t  z_ol [68*384];
  int8_t   l_ol [68*384];
  simde__m128i *pv_ol128 = (simde__m128i*)&z_ol;
  simde__m128i *pl_ol128 = (simde__m128i*)&l_ol;
  int no_iteration_ldpc = 2;
  int length_dec;
  uint8_t crc_type;
  int K_bits_F;
  int16_t  z [68*384 + 16] __attribute__ ((aligned(16)));
  int8_t   l [68*384 + 16] __attribute__ ((aligned(16)));

  simde__m128i *pv = (simde__m128i*)&z;
  simde__m128i *pl = (simde__m128i*)&l;

#ifdef PRINT_CRC_CHECK
  prnt_crc_cnt++;
#endif

  NR_gNB_ULSCH_t *ulsch = &phy_vars_gNB->ulsch[ULSCH_id];
  NR_gNB_PUSCH *pusch = &phy_vars_gNB->pusch_vars[ULSCH_id];
  NR_UL_gNB_HARQ_t *harq_process = ulsch->harq_process;

  if (!harq_process) {
    LOG_E(PHY,"ulsch_decoding.c: NULL harq_process pointer\n");
    return 1;
  }
  uint8_t dtx_det = 0;

  int Kr;
  int Kr_bytes;
    
  phy_vars_gNB->nbDecode = 0;
  harq_process->processedSegments = 0;
  
  // ------------------------------------------------------------------
  uint16_t nb_rb          = pusch_pdu->rb_size;
  uint8_t Qm              = pusch_pdu->qam_mod_order;
  uint8_t mcs             = pusch_pdu->mcs_index;
  uint8_t n_layers        = pusch_pdu->nrOfLayers;
  // ------------------------------------------------------------------

  if (!ulsch_llr) {
    LOG_E(PHY,"ulsch_decoding.c: NULL ulsch_llr pointer\n");
    return 1;
  }

  VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_PHY_gNB_ULSCH_DECODING,1);
  harq_process->TBS = pusch_pdu->pusch_data.tb_size;

  dtx_det = 0;

  uint32_t A = (harq_process->TBS) << 3;

  // target_code_rate is in 0.1 units
  float Coderate = (float) pusch_pdu->target_code_rate / 10240.0f;

  LOG_D(PHY,"ULSCH Decoding, harq_pid %d rnti %x TBS %d G %d mcs %d Nl %d nb_rb %d, Qm %d, Coderate %f RV %d round %d new RX %d\n",
        harq_pid, ulsch->rnti, A, G, mcs, n_layers, nb_rb, Qm, Coderate, pusch_pdu->pusch_data.rv_index, harq_process->round, harq_process->harq_to_be_cleared);
  t_nrLDPC_dec_params decParams = {0};
  decParams.BG = pusch_pdu->maintenance_parms_v3.ldpcBaseGraph;
  int kc;
  if (decParams.BG == 2) {
    kc = 52;
  } else {
    kc = 68;
  }

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
  if (A > 3824)
    harq_process->B = A+24;
  else
    harq_process->B = A+16;

  // [hna] Perform nr_segmenation with input and output set to NULL to calculate only (B, C, K, Z, F)
  nr_segmentation(NULL,
                  NULL,
                  harq_process->B,
                  &harq_process->C,
                  &harq_process->K,
                  &harq_process->Z, // [hna] Z is Zc
                  &harq_process->F,
                  decParams.BG);

  if (harq_process->C>MAX_NUM_NR_DLSCH_SEGMENTS_PER_LAYER*n_layers) {
    LOG_E(PHY,"nr_segmentation.c: too many segments %d, B %d\n",harq_process->C,harq_process->B);
    return(-1);
  }


#ifdef DEBUG_ULSCH_DECODING
  printf("ulsch decoding nr segmentation Z %d\n", harq_process->Z);
  if (!frame%100)
    printf("K %d C %d Z %d \n", harq_process->K, harq_process->C, harq_process->Z);
#endif
  decParams.Z = harq_process->Z;

  decParams.numMaxIter = ulsch->max_ldpc_iterations;
  decParams.outMode = 0;

  uint32_t r_offset = 0;

  uint16_t a_segments = MAX_NUM_NR_ULSCH_SEGMENTS_PER_LAYER*n_layers;  //number of segments to be allocated

  if (nb_rb != 273) {
    a_segments = a_segments*nb_rb;
    a_segments = a_segments/273 +1;
  }

  if (harq_process->C > a_segments) {
    LOG_E(PHY,"Illegal harq_process->C %d > %d\n",harq_process->C,a_segments);
    return 1;
  }
#ifdef DEBUG_ULSCH_DECODING
  printf("Segmentation: C %d, K %d\n",harq_process->C,harq_process->K);
#endif

  if (harq_process->harq_to_be_cleared) {
    for (int r = 0; r < harq_process->C; r++)
      harq_process->d_to_be_cleared[r] = true;
    harq_process->harq_to_be_cleared = false;
  }

  Kr = harq_process->K;
  Kr_bytes = Kr >> 3;

  uint32_t offset = 0;

  //LDPC decode is offloaded to FPGA using the xdma driver

  K_bits_F = Kr - harq_process->F;
  //-------------------- FPGA parameter preprocessing ---------------------
  static int8_t multi_indata[27000 * 25]; // FPGA input data
  static int8_t multi_outdata[1100 * 25]; // FPGA output data

  int mbmb = 0;
  if (decParams.BG == 1)
    mbmb = 68;
  else
    mbmb = 52;

  int bg_len = 0;
  if (decParams.BG == 1)
    bg_len = 22;
  else
    bg_len = 10;

  // Calc input CB offset
  int input_CBoffset = decParams.Z * mbmb * 8;
  if ((input_CBoffset & 0x7F) == 0)
    input_CBoffset = input_CBoffset / 8;
  else
    input_CBoffset = 16 * ((input_CBoffset / 128) + 1);

  DecIFConf dec_conf;
  dec_conf.Zc = decParams.Z;
  dec_conf.BG = decParams.BG;
  dec_conf.max_iter = decParams.numMaxIter;
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
  no_iteration_ldpc = 2;

  dtx_det = 0;
  uint32_t num_threads_prepare_max = NUM_THREADS_PREPARE;
  uint32_t num_threads_prepare = 0;
  uint32_t r_remaining = 0;
  //start the prepare jobs

#ifdef TASK_MANAGER
  args_fpga_decode_prepare_t* arr = calloc(harq_process->C, sizeof(args_fpga_decode_prepare_t)); 
  int idx_arr = 0;
  _Atomic int cancel_decoding = 0;
#elif OMP_TP 
  args_fpga_decode_prepare_t* arr = calloc(harq_process->C, sizeof(args_fpga_decode_prepare_t)); 
  int idx_arr = 0;
  omp_set_num_threads(4);
#pragma omp parallel
{
#pragma omp single
{

#endif

  for (r = 0; r < harq_process->C; r++) {
    E = nr_get_E(G, harq_process->C, Qm, n_layers, r);
    if (r_remaining == 0 ) {
#ifdef TASK_MANAGER
      args_fpga_decode_prepare_t* args = &arr[idx_arr]; 
      ++idx_arr; 
#elif OMP_TP 
      args_fpga_decode_prepare_t* args = &arr[idx_arr]; 
      ++idx_arr; 
#else
      void (*nr_ulsch_FPGA_decoding_prepare_blocks_ptr)(void *) = &nr_ulsch_FPGA_decoding_prepare_blocks;
      union ldpcReqUnion id = {.s={ulsch->rnti,frame,nr_tti_rx,0,0}};
      notifiedFIFO_elt_t *req = newNotifiedFIFO_elt(sizeof(args_fpga_decode_prepare_t), id.p, &phy_vars_gNB->respDecode, nr_ulsch_FPGA_decoding_prepare_blocks_ptr);
      args_fpga_decode_prepare_t * args = (args_fpga_decode_prepare_t *) NotifiedFifoData(req);
#endif

      args->multi_indata = multi_indata;
      args->no_iteration_ldpc = 2;
      args->r_first = r;
      uint32_t r_span_max = ((harq_process->C-r)%(num_threads_prepare_max-num_threads_prepare))==0 ? (harq_process->C-r)/(num_threads_prepare_max-num_threads_prepare) : ((harq_process->C-r)/(num_threads_prepare_max-num_threads_prepare))+1 ;
      uint32_t r_span = harq_process->C-r<r_span_max ? harq_process->C-r : r_span_max;
      args->r_span = r_span;
      r_remaining = r_span;
      args->n_layers = n_layers;
      args->G = G;
      ldpcDecode_t *rdata = &args->decode;
#ifdef TASK_MANAGER
      rdata->cancel_decoding = &cancel_decoding;
#endif

      rdata->gNB = phy_vars_gNB;
      rdata->ulsch_harq = harq_process;
      rdata->decoderParms = decParams;
      rdata->ulsch_llr = ulsch_llr;
      rdata->Kc = kc;
      rdata->harq_pid = harq_pid;
      rdata->segment_r = r;
      rdata->nbSegments = harq_process->C;
      rdata->E = E;
      rdata->A = A;
      rdata->Qm = Qm;
      rdata->r_offset = r_offset;
      rdata->Kr_bytes = Kr_bytes;
      rdata->rv_index = pusch_pdu->pusch_data.rv_index;
      rdata->offset = offset;
      rdata->ulsch = ulsch;
      rdata->ulsch_id = ULSCH_id;
      rdata->tbslbrm = pusch_pdu->maintenance_parms_v3.tbSizeLbrmBytes;
#ifdef TASK_MANAGER
      task_t t = { .args = args, .func =  &nr_ulsch_FPGA_decoding_prepare_blocks };
      async_task_manager(&phy_vars_gNB->man, t);
#elif OMP_TP 
#pragma omp task
      nr_ulsch_FPGA_decoding_prepare_blocks(args); 
#else
      pushTpool(&phy_vars_gNB->threadPool, req);
#endif
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

#ifdef OMP_TP
}
}
#endif


#ifdef TASK_MANAGER
  stop_spin_task_manager(&phy_vars_gNB->man);
  wait_all_spin_task_manager(&phy_vars_gNB->man);
  free(arr); 
#elif OMP_TP
#pragma omp taskwait
  free(arr); 
#else

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
#endif

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

  for (r = 0; r < harq_process->C; r++) {
    // -----------------------------------------------------------------------------------------------
    // --------------------- copy FPGA output to harq_process->c[r][i] -------------------------------
    // -----------------------------------------------------------------------------------------------
    if (check_crc((uint8_t *)multi_outdata, length_dec, harq_process->F, crc_type)) {
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

  //const int64_t end = time_now_us();
  //printf("After waiting all the tasks %ld tstamp %ld \n", end-now, end);


  return 1;
}

void nr_ulsch_procedures_tb(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx, int ULSCH_id, uint8_t harq_pid)
{

#ifdef TASK_MANAGER
  wake_and_spin_task_manager(&gNB->man);
#endif

  NR_DL_FRAME_PARMS *frame_parms = &gNB->frame_parms;
  nfapi_nr_pusch_pdu_t *pusch_pdu = &gNB->ulsch[ULSCH_id].harq_process->ulsch_pdu;

  uint16_t nb_re_dmrs;

#ifndef TASK_MANAGER
#ifndef OMP_TP
  uint8_t enable_ldpc_offload = gNB->ldpc_offload_flag;
#endif
#endif
  uint16_t start_symbol = pusch_pdu->start_symbol_index;
  uint16_t number_symbols = pusch_pdu->nr_of_symbols;

  uint8_t number_dmrs_symbols = 0;
  for (int l = start_symbol; l < start_symbol + number_symbols; l++)
    number_dmrs_symbols += ((pusch_pdu->ul_dmrs_symb_pos)>>l)&0x01;

  if (pusch_pdu->dmrs_config_type==pusch_dmrs_type1)
    nb_re_dmrs = 6*pusch_pdu->num_dmrs_cdm_grps_no_data;
  else
    nb_re_dmrs = 4*pusch_pdu->num_dmrs_cdm_grps_no_data;

  uint32_t G = nr_get_G(pusch_pdu->rb_size,
                        number_symbols,
                        nb_re_dmrs,
                        number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
                        pusch_pdu->qam_mod_order,
                        pusch_pdu->nrOfLayers);

  AssertFatal(G>0,"G is 0 : rb_size %u, number_symbols %d, nb_re_dmrs %d, number_dmrs_symbols %d, qam_mod_order %u, nrOfLayer %u\n",
	      pusch_pdu->rb_size,
	      number_symbols,
	      nb_re_dmrs,
	      number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
	      pusch_pdu->qam_mod_order,
	      pusch_pdu->nrOfLayers);
  LOG_I(PHY,"rb_size %d, number_symbols %d, nb_re_dmrs %d, dmrs symbol positions %d, number_dmrs_symbols %d, qam_mod_order %d, nrOfLayer %d, G %d\n",
	pusch_pdu->rb_size,
	number_symbols,
	nb_re_dmrs,
        pusch_pdu->ul_dmrs_symb_pos,
	number_dmrs_symbols, // number of dmrs symbols irrespective of single or double symbol dmrs
	pusch_pdu->qam_mod_order,
	pusch_pdu->nrOfLayers,
	G);

  if (gNB->use_pusch_tp == 0 ) { 
    nr_ulsch_layer_demapping(gNB->pusch_vars[ULSCH_id].llr,
                             pusch_pdu->nrOfLayers,
                             pusch_pdu->qam_mod_order,
                             G,
                             gNB->pusch_vars[ULSCH_id].llr_layers);
  //----------------------------------------------------------
  //------------------- ULSCH unscrambling -------------------
  //----------------------------------------------------------
    start_meas(&gNB->ulsch_unscrambling_stats);
    nr_ulsch_unscrambling(gNB->pusch_vars[ULSCH_id].llr, G, pusch_pdu->data_scrambling_id, pusch_pdu->rnti);
    stop_meas(&gNB->ulsch_unscrambling_stats);
  }
  //----------------------------------------------------------
  //--------------------- ULSCH decoding ---------------------
  //----------------------------------------------------------

  start_meas(&gNB->ulsch_decoding_stats);
  //int64_t start = time_now_us();
  //printf(" nr_ulsch_decoding %lu \n", start );
  nr_ulsch_decoding_tb(gNB, ULSCH_id, gNB->pusch_vars[ULSCH_id].llr, frame_parms, pusch_pdu, frame_rx, slot_rx, harq_pid, G);
#ifndef TASK_MANAGER
#ifndef OMP_TP
  if (enable_ldpc_offload == 0) {
    while (gNB->nbDecode > 0) {
      notifiedFIFO_elt_t *req = pullTpool(&gNB->respDecode, &gNB->threadPool);
      if (req == NULL)
	        break; // Tpool has been stopped
      nr_postDecode(gNB, req);
      delNotifiedFIFO_elt(req);
    }
  } 
#endif
#endif
  //int64_t end = time_now_us();
  //printf(" nr_ulsch_decoding %lu \n", end-start );

  stop_meas(&gNB->ulsch_decoding_stats);
}

void nr_ulsch_process_slot(PHY_VARS_gNB *gNB, int frame_rx, int slot_rx)
{
  for (int ULSCH_id = 0; ULSCH_id < gNB->max_nb_pusch; ULSCH_id++) {
    NR_gNB_ULSCH_t *ulsch = &gNB->ulsch[ULSCH_id];

    if ((ulsch->active == true) && (ulsch->frame == frame_rx) && (ulsch->slot == slot_rx) && (ulsch->handled == 0)) {
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_NR_ULSCH_PROCEDURES_RX, 1);
      nr_ulsch_procedures_tb(gNB, frame_rx, slot_rx, ULSCH_id, ulsch->harq_pid);
      VCD_SIGNAL_DUMPER_DUMP_FUNCTION_BY_NAME(VCD_SIGNAL_DUMPER_FUNCTIONS_NR_ULSCH_PROCEDURES_RX, 0);
    }
  }
}
