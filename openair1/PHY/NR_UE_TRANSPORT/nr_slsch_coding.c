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
/*! \file PHY/NR_UE_TRANSPORT/nr_slsch_coding.c
* \brief 5G Sidelink TX encoding procedures for the SLSCH
* \author Melissa Elkadi, David Kim
* \date 01/01/2023
* \version
* \company EpiSci, Episys Science Inc.
* \email: melissa@episci.com, david.kim@episci.com
* \note
* \warning
*/

#include "PHY/defs_UE.h"
#include "PHY/phy_extern_ue.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "PHY/CODING/coding_defs.h"
#include "PHY/CODING/coding_extern.h"
#include "PHY/CODING/lte_interleaver_inline.h"
#include "PHY/CODING/nrLDPC_extern.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_ue.h"
#include "common/utils/LOG/vcd_signal_dumper.h"
#include <openair2/UTIL/OPT/opt.h>
#include <inttypes.h>
#include "openair1/PHY/CODING/nrLDPC_defs.h"
#include "PHY/defs_nr_common.h"

//#define DEBUG_SLSCH_CODING
#define NR_POLAR_SCI2_MESSAGE_TYPE 4
#define NR_POLAR_SCI2_PAYLOAD_BITS 35
#define NR_POLAR_SCI2_AGGREGATION_LEVEL 0


static uint16_t polar_encoder_output_length(sl_nr_tx_config_pscch_pssch_pdu_t *pdu) {
  // calculating length of sequence comming out of rate matching for SCI2 based on 8.4.4 TS38212
  uint8_t Osci2 = NR_POLAR_SCI2_PAYLOAD_BITS;
  uint8_t Lsci2 = 24;
  uint8_t Qmsci2 = 2; //modulation order of SCI2
  double beta = 1.125; // pssch_pdu->sci1.beta_offset;
  double alpha = 1; // hardcoded for now
  float R = (float)pdu->target_coderate / (1024*10);

  double tmp1 = (Osci2 + Lsci2) * beta / (Qmsci2 * R);
  double tmp2 = alpha * pdu->mod_order; // it is assumed that number of RB for PSCCH is 0.
  int gamma = 461;
  uint16_t Qprime = min(ceil(tmp1), ceil(tmp2)) + gamma;
  uint16_t Gsci2 = Qprime * Qmsci2;
  return Gsci2;
}

static uint16_t get_Nidx_from_CRC(uint64_t *A,
                           int32_t crcmask,
                           uint8_t ones_flag,
                           int8_t messageType,
                           uint16_t messageLength,
                           uint8_t aggregation_level) {

  t_nrPolar_params *polarParams = nr_polar_params(messageType, messageLength, aggregation_level, false);
  //  AssertFatal(polarParams->K > 32, "K = %d < 33, is not supported yet\n",polarParams->K);
  AssertFatal(polarParams->K < 129, "K = %d > 128, is not supported yet\n",polarParams->K);
  AssertFatal(polarParams->payloadBits < 65, "payload bits = %d > 64, is not supported yet\n",polarParams->payloadBits);
  int bitlen = polarParams->payloadBits;
  AssertFatal(bitlen < 129, "support for payloads <= 128 bits\n");
  uint64_t tcrc = 0;
  uint8_t offset = 0;

  // appending 24 ones before a0 for DCI as stated in 38.212 7.3.2
  if (ones_flag) offset = 3;
  if (bitlen <= 32) {
    uint8_t A32_flip[4 + offset];
    if (ones_flag) {
      A32_flip[0] = 0xff;
      A32_flip[1] = 0xff;
      A32_flip[2] = 0xff;
    }
    uint32_t Aprime = (uint32_t)(((uint32_t)*A) << (32 - bitlen));
    A32_flip[0 + offset] = ((uint8_t *)&Aprime)[3];
    A32_flip[1 + offset] = ((uint8_t *)&Aprime)[2];
    A32_flip[2 + offset] = ((uint8_t *)&Aprime)[1];
    A32_flip[3 + offset] = ((uint8_t *)&Aprime)[0];
    if      (polarParams->crcParityBits == 24)
      tcrc = (uint64_t)(((crcmask ^ (crc24c(A32_flip, 8 * offset + bitlen) >> 8))) & 0xffffff);
    else if (polarParams->crcParityBits == 11)
      tcrc = (uint64_t)(((crcmask ^ (crc11(A32_flip, bitlen) >> 21))) & 0x7ff);
    else if (polarParams->crcParityBits == 6)
      tcrc = (uint64_t)(((crcmask ^ (crc6(A32_flip, bitlen) >> 26))) & 0x3f);
  } else if (bitlen <= 64) {
    uint8_t A64_flip[8 + offset];
    if (ones_flag) {
      A64_flip[0] = 0xff;
      A64_flip[1] = 0xff;
      A64_flip[2] = 0xff;
    }
    uint64_t Aprime = (uint64_t)(((uint64_t)*A) << (64 - bitlen));
    A64_flip[0 + offset] = ((uint8_t *) & Aprime)[7];
    A64_flip[1 + offset] = ((uint8_t *) & Aprime)[6];
    A64_flip[2 + offset] = ((uint8_t *) & Aprime)[5];
    A64_flip[3 + offset] = ((uint8_t *) & Aprime)[4];
    A64_flip[4 + offset] = ((uint8_t *) & Aprime)[3];
    A64_flip[5 + offset] = ((uint8_t *) & Aprime)[2];
    A64_flip[6 + offset] = ((uint8_t *) & Aprime)[1];
    A64_flip[7 + offset] = ((uint8_t *) & Aprime)[0];
    if (polarParams->crcParityBits == 24)
      tcrc = (uint64_t)((crcmask ^ (crc24c(A64_flip, 8 * offset + bitlen) >> 8))) & 0xffffff;
    else if (polarParams->crcParityBits == 11)
      tcrc = (uint64_t)((crcmask ^ (crc11(A64_flip, bitlen) >> 21))) & 0x7ff;
  } else if (bitlen <= 128) {
    uint8_t A128_flip[16 + offset];
    if (ones_flag) {
      A128_flip[0] = 0xff;
      A128_flip[1] = 0xff;
      A128_flip[2] = 0xff;
    }
    uint128_t Aprime = (uint128_t)(((uint128_t)*A) << (128 - bitlen));
    for (int i = 0; i < 16 ; i++)
      A128_flip[i + offset] = ((uint8_t*)&Aprime)[15 - i];
    if (polarParams->crcParityBits == 24)
      tcrc = (uint64_t)((crcmask ^ (crc24c(A128_flip, 8 * offset + bitlen) >> 8))) & 0xffffff;
    else if (polarParams->crcParityBits == 11)
      tcrc = (uint64_t)((crcmask ^ (crc11(A128_flip, bitlen) >> 21))) & 0x7ff;
  }
  return tcrc & 0xFFFF;
}

static void nr_attach_crc_to_payload(unsigned char *in, uint8_t *out, int max_payload_bytes, uint32_t in_size, uint32_t *out_size) {

    unsigned int crc = 1;
    if (in_size > NR_MAX_PSSCH_TBS) {
      // Add 24-bit crc (polynomial A) to payload
      crc = crc24a(in, in_size) >> 8;
      in[in_size >> 3] = ((uint8_t*)&crc)[2];
      in[1 + (in_size >> 3)] = ((uint8_t*)&crc)[1];
      in[2 + (in_size >> 3)] = ((uint8_t*)&crc)[0];
      *out_size = in_size + 24;

      AssertFatal((in_size / 8) + 4 <= max_payload_bytes,
                  "A %d is too big (A / 8 + 4 = %d > %d)\n", in_size, (in_size / 8) + 4, max_payload_bytes);

      memcpy(out, in, (in_size / 8) + 4);
    } else {
      // Add 16-bit crc (polynomial A) to payload
      crc = crc16(in, in_size) >> 16;
      in[in_size >> 3] = ((uint8_t*)&crc)[1];
      in[1 + (in_size >> 3)] = ((uint8_t*)&crc)[0];
      *out_size = in_size + 16;

      AssertFatal((in_size / 8) + 3 <= max_payload_bytes,
                  "A %d is too big (A / 8 + 3 = %d > %d)\n", in_size, (in_size / 8) + 3, max_payload_bytes);

      memcpy(out, in, (in_size / 8) + 3);  // using 3 bytes to mimic the case of 24 bit crc
    }
}

static void byte2bit(uint8_t *in_bytes, uint8_t *out_bits, uint16_t num_bytes) {

  for (int i=0 ; i<num_bytes ; i++) {
    for (int b=0 ; b<8 ; b++){
      out_bits[i*8 + b] = (in_bytes[i]>>b) & 1;
    }
  }
  return;
}

int nr_slsch_encoding(PHY_VARS_NR_UE *ue,
                      sl_nr_tx_config_pscch_pssch_pdu_t *pssch_pdu,
                      NR_DL_FRAME_PARMS* frame_parms,
                      uint8_t harq_pid,
                      unsigned int G) {

  NR_UL_UE_HARQ_t *harq_process = &pssch_pdu->harq_processes_ul[harq_pid];
  if (harq_process->first_tx == 1 || harq_process->ndi != pssch_pdu->ndi) {
    harq_process->first_tx = 0;
    int max_payload_bytes = MAX_NUM_NR_SLSCH_SEGMENTS_PER_LAYER * pssch_pdu->num_layers * 1056;
    uint16_t polar_encoder_output_len = polar_encoder_output_length(pssch_pdu);
    polar_encoder_fast(harq_process->a_sci2, (void*)harq_process->b_sci2, 0, 0,
                       NR_POLAR_SCI2_MESSAGE_TYPE,
                       polar_encoder_output_len,
                       NR_POLAR_SCI2_AGGREGATION_LEVEL);
    pssch_pdu->nid_x = get_Nidx_from_CRC(harq_process->a_sci2, 0, 0, NR_POLAR_SCI2_MESSAGE_TYPE,
                                         polar_encoder_output_len, NR_POLAR_SCI2_AGGREGATION_LEVEL);
    harq_process->B_sci2 = polar_encoder_output_len;

    byte2bit(harq_process->b, harq_process->f_sci2, polar_encoder_output_len>>3);
    uint32_t A = pssch_pdu->tb_size << 3;
    nr_attach_crc_to_payload(harq_process->a, harq_process->b, max_payload_bytes, A, &harq_process->B);
    float code_rate = (float) pssch_pdu->target_coderate / 10240.0f;
    if ((A <= 292) || ((A <= 3824) && (code_rate <= 0.6667)) || code_rate <= 0.25) {
      harq_process->BG = 2;
    } else {
      harq_process->BG = 1;
    }

    uint32_t  Kb = nr_segmentation(harq_process->b,
                                   harq_process->c,
                                   harq_process->B,
                                   &harq_process->C,
                                   &harq_process->K,
                                   &harq_process->Z,
                                   &harq_process->F,
                                   harq_process->BG);

    if (harq_process->C > MAX_NUM_NR_SLSCH_SEGMENTS_PER_LAYER * pssch_pdu->num_layers) {
      LOG_E(NR_PHY, "nr_segmentation.c: too many segments %d, B %d\n", harq_process->C, harq_process->B);
      return(-1);
    }

    encoder_implemparams_t impp = {
      .n_segments = harq_process->C,
      .macro_num = 0,
      .tinput  = NULL,
      .tprep   = NULL,
      .tparity = NULL,
      .toutput = NULL};

    for (int j = 0; j < (harq_process->C / 8 + 1); j++) {
      impp.macro_num = j;
      nrLDPC_encoder(harq_process->c, harq_process->d, harq_process->Z, Kb, harq_process->K, harq_process->BG, &impp);
    }

    harq_process->ndi = pssch_pdu->ndi;
  }

  int r_offset = 0;
  for (int r = 0; r < harq_process->C; r++) {
    if (harq_process->F > 0) {
      for (int k = (harq_process->K - harq_process->F - 2 * (harq_process->Z)); k < harq_process->K - 2 * (harq_process->Z); k++) {
        harq_process->d[r][k] = NR_NULL;
      }
    }
    uint32_t E = nr_get_E(G, harq_process->C, pssch_pdu->mod_order, pssch_pdu->num_layers, r);
    if (nr_rate_matching_ldpc(pssch_pdu->tbslbrm,
                              harq_process->BG,
                              harq_process->Z,
                              harq_process->d[r],
                              harq_process->e + r_offset,
                              harq_process->C,
                              harq_process->F,
                              harq_process->K - harq_process->F - 2*(harq_process->Z),
                              pssch_pdu->rv_index,
                              E) == -1)
      return -1;
    nr_interleaving_ldpc(E, pssch_pdu->mod_order, harq_process->e + r_offset, harq_process->f + r_offset);
    r_offset += E;
  }

  harq_process->B_multiplexed = G + harq_process->B_sci2 * pssch_pdu->num_layers;

  return(0);
}
