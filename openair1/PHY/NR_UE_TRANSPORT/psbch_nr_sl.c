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

/*! \file PHY/NR_TRANSPORT/nr_psbch.c
* \brief Top-level routines for generating the PSBCH/BCH physical/transport channel V15.1 03/2018
* \author Guy De Souza
* \thanks Special Thanks to Son Dang for helpful contributions and testing
* \date 2018
* \version 0.1
* \company Eurecom
* \email: desouza@eurecom.fr
* \note
* \warning
*/

#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/NR_TRANSPORT/nr_transport_proto.h"
#include "PHY/LTE_REFSIG/lte_refsig.h"
#include "PHY/sse_intrin.h"
#include "executables/softmodem-common.h"

//#define DEBUG_PSBCH
//#define DEBUG_PSBCH_ENCODING
//#define DEBUG_PSBCH_DMRS

extern short nr_qpsk_mod_table[8];

int nr_generate_psbch_dmrs(uint32_t *gold_psbch_dmrs,
                           int32_t *txdataF,
                           int16_t amp,
                           uint8_t ssb_start_symbol,
                           NR_DL_FRAME_PARMS *frame_parms) {
  int dmrs_modulations_per_symbol = 33;
  int16_t mod_dmrs[NR_PSBCH_DMRS_LENGTH << 1];
  LOG_D(NR_PHY, "PSBCH DMRS mapping started at symbol %d\n", ssb_start_symbol);

  /// QPSK modulation
  for (int m = 0; m < NR_PSBCH_DMRS_LENGTH; m++) {
    AssertFatal(((m << 1) >> 5) < NR_PSBCH_DMRS_LENGTH_DWORD, "Invalid index size %d\n", (m << 1) >> 5);
    int idx = (((gold_psbch_dmrs[(m << 1) >> 5]) >> ((m << 1) & 0x1f)) & 3);
    AssertFatal(((idx << 1) + 1) < (sizeof(nr_qpsk_mod_table) / sizeof(nr_qpsk_mod_table[0])), "Invalid index size %d\n", (idx << 1) + 1);
    AssertFatal((m << 1) + 1 < (sizeof(mod_dmrs) / sizeof(mod_dmrs[0])), "Invalid index size %d\n", (idx << 1) + 1);
    mod_dmrs[m << 1] = nr_qpsk_mod_table[idx << 1];
    mod_dmrs[(m << 1) + 1] = nr_qpsk_mod_table[(idx << 1) + 1];
#ifdef DEBUG_PSBCH_DMRS
    printf("m %d idx %d gold seq %d b0-b1 %d-%d mod_dmrs %d %d\n", m, idx, gold_psbch_dmrs[(m << 1) >> 5], (((gold_psbch_dmrs[(m << 1) >> 5]) >> ((m << 1) & 0x1f)) & 1),
           (((gold_psbch_dmrs[((m << 1) + 1) >> 5]) >> (((m << 1)+1) & 0x1f)) & 1), mod_dmrs[(m << 1)], mod_dmrs[(m << 1)+1]);
#endif
  }

  /// Resource mapping
  // PSBCH DMRS are mapped  within the SSB block on every fourth subcarrier starting from nushift of symbols 1, 2, 3
  ///symbol 0  [0+nushift:4:236+nushift] -- 33 mod symbols
  int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier;
  int l = ssb_start_symbol;
  int m = 0;
  for (; m < dmrs_modulations_per_symbol; m++) {
#ifdef DEBUG_PSBCH_DMRS
    printf("m %d at k %d of l %d\n", m, k, l);
#endif
    AssertFatal(((m << 1) + 1) < (sizeof(mod_dmrs) / sizeof(mod_dmrs[0])), "Invalid index into mod_dmrs. Index %d > %lu\n",
              (m << 1) + 1, (sizeof(mod_dmrs) / sizeof(mod_dmrs[0])));
    int idx = (l * frame_parms->ofdm_symbol_size + k) << 1;
    AssertFatal((idx + 1) < frame_parms->samples_per_frame_wCP, "txdataF index %d invalid!\n", idx + 1);
    ((int16_t *)txdataF)[idx] = (amp * mod_dmrs[m << 1]) >> 15;
    ((int16_t *)txdataF)[idx + 1] = (amp * mod_dmrs[(m << 1) + 1]) >> 15;
#ifdef DEBUG_PSBCH_DMRS
    printf("(%d,%d)\n",
           ((int16_t *)txdataF)[(idx) << 1],
           ((int16_t *)txdataF)[((idx) << 1)+1]);
#endif
    k+=4;
    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }

  int N_SSSB_Symb = 13;
  l = ssb_start_symbol + 5;
  while (l < N_SSSB_Symb)
  {
    int mod_count = 0;
    while (m < NR_PSBCH_DMRS_LENGTH) {
#ifdef DEBUG_PSBCH_DMRS
      printf("m %d at k %d of l %d\n", m, k, l);
#endif
      AssertFatal(((m << 1) + 1) < (sizeof(mod_dmrs) / sizeof(mod_dmrs[0])), "Invalid index into mod_dmrs. Index %d > %lu\n",
                (m << 1) + 1, (sizeof(mod_dmrs) / sizeof(mod_dmrs[0])));
      int idx = (l * frame_parms->ofdm_symbol_size + k) << 1;
      AssertFatal((idx + 1) < frame_parms->samples_per_frame_wCP, "txdataF index %d invalid!\n", idx + 1);
      ((int16_t *)txdataF)[idx] = (amp * mod_dmrs[m << 1]) >> 15;
      ((int16_t *)txdataF)[idx + 1] = (amp * mod_dmrs[(m << 1) + 1]) >> 15;
#ifdef DEBUG_PSBCH_DMRS
      printf("(%d,%d)\n",
             ((int16_t *)txdataF)[idx],
             ((int16_t *)txdataF)[(idx)+1]);
#endif
      k+=4;
      if (k >= frame_parms->ofdm_symbol_size)
        k-=frame_parms->ofdm_symbol_size;
      mod_count++;
      m++;
      if (mod_count == dmrs_modulations_per_symbol)
        break;
    }
    l++;
  }

#ifdef DEBUG_PSBCH_DMRS
  write_output("txdataF_psbch_dmrs.m", "txdataF_psbch_dmrs", txdataF[0], frame_parms->samples_per_frame_wCP>>1, 1, 1);
#endif
  return 0;
}

static void nr_psbch_scrambling(NR_UE_PSBCH *psbch,
                                uint32_t Nid,
                                uint8_t nushift,
                                uint16_t M,
                                uint16_t length,
                                uint8_t encoded,
                                uint32_t unscrambling_mask) {
  uint32_t *psbch_e = psbch->psbch_e;
  uint8_t reset = 1;
  uint32_t x1, s = 0;
  uint32_t x2 = Nid;
  // The Gold sequence is shifted by nushift* M, so we skip (nushift*M /32) double words
  for (int i = 0; i < (uint16_t)ceil(((float)M) / 32); i++) {
    s = lte_gold_generic(&x1, &x2, reset);
    reset = 0;
  }

  // Scrambling is now done with offset (nushift*M)%32
  uint8_t offset = 0; //(nushift*M)&0x1f;
#ifdef DEBUG_PSBCH_ENCODING
  printf("Scrambling params: nushift %d M %d length %d encoded %d offset %d\n", nushift, M, length, encoded, offset);
#endif
#ifdef DEBUG_PSBCH_ENCODING
  printf("s: %04x\t", s);
#endif

  int k = 0;
  if (!encoded) {
    /// 1st Scrambling
    for (int i = 0; i < length; ++i) {
      if ((unscrambling_mask>>i)&1)
        psbch->psbch_a_prime ^= ((psbch->psbch_a_interleaved >> i) & 1) << i;
      else {
        if (((k + offset) & 0x1f) == 0) {
          s = lte_gold_generic(&x1, &x2, reset);
          reset = 0;
        }

        psbch->psbch_a_prime ^= (((psbch->psbch_a_interleaved >> i) & 1) ^ ((s >> ((k + offset) & 0x1f)) & 1)) << i;
        k++;                  /// k increase only when payload bit is not special bit
      }
    }
  } else {
    /// 2nd Scrambling
    for (int i = 0; i < length; ++i) {
      if (((i + offset) & 0x1f) == 0) {
        s = lte_gold_generic(&x1, &x2, reset);
        reset = 0;
      }
      AssertFatal((i >> 5) < NR_POLAR_PSBCH_E_DWORD, "Invalid index into psbch->psbch_e. Index %d > %d\n",
                 (i >> 5), NR_POLAR_PSBCH_E_DWORD);
      psbch_e[i >> 5] ^= (((s >> ((i + offset) & 0x1f)) & 1) << (i & 0x1f));
    }
  }
}

int nr_generate_sl_psbch(PHY_VARS_NR_UE *ue,
                         int32_t *txdataF,
                         int16_t amp,
                         uint8_t ssb_start_symbol,
                         uint8_t n_hf,
                         int sfn,
                         NR_DL_FRAME_PARMS *frame_parms) {
  LOG_D(NR_PHY, "PSBCH SL generation started\n");

  /* payload is 56 bits */
  PSBCH_payload psbch_payload;             // NR Side Link Payload for Rel 16
  psbch_payload.coverageIndicator = 0;     // 1 bit
  psbch_payload.tddConfig = 0xFFF;         // 12 bits for TDD configuration
  psbch_payload.DFN = 0x3FF;               // 10 bits for DFN
  psbch_payload.slotIndex = 0x2A;          // 7 bits for Slot Index //frame_parms->p_TDD_UL_DL_ConfigDedicated->slotIndex;
  psbch_payload.reserved = 0;              // 2 bits reserved

  NR_UE_PSBCH m_psbch;
  ue->psbch_vars[0] = &m_psbch;
  NR_UE_PSBCH *psbch = ue->psbch_vars[0];
  memset((void *)psbch, 0, sizeof(NR_UE_PSBCH));
  psbch->psbch_a = *((uint32_t *)&psbch_payload);
  psbch->psbch_a_interleaved = psbch->psbch_a; // skip interlevaing for Sidelink

  psbch->psbch_a_prime = 0;

  printf("PSBCH payload generated 0x%x\t ------> ", psbch->psbch_a);

  #ifdef DEBUG_PSBCH_ENCODING
    printf("PSBCH payload = 0x%08x\n",psbch->psbch_a);
  #endif

  // Encoder reversal
  uint64_t a_reversed = 0;
  for (int i = 0; i < NR_POLAR_PSBCH_PAYLOAD_BITS; i++)
    a_reversed |= (((uint64_t)psbch->psbch_a_interleaved >> i) & 1) << (31 - i);

  /// CRC, coding and rate matching
  polar_encoder_fast(&a_reversed, (void*)psbch->psbch_e, 0, 0,
                     NR_POLAR_PSBCH_MESSAGE_TYPE, NR_POLAR_PSBCH_PAYLOAD_BITS, NR_POLAR_PSBCH_AGGREGATION_LEVEL);

#ifdef DEBUG_PSBCH_ENCODING
  printf("PSBCH SL generation started\n");
  printf("Channel coding:\n");
  for (int i=0; i<NR_POLAR_PSBCH_E_DWORD; i++)
    printf("sl_psbch_e[%d]: 0x%08x\n", i, psbch->psbch_e[i]);
  printf("\n");
#endif

  /// Scrambling
  uint16_t M = NR_POLAR_PSBCH_E;
  uint8_t nushift = 0;
  nr_psbch_scrambling(psbch, (uint32_t)frame_parms->Nid_SL, nushift, M, NR_POLAR_PSBCH_E, 1, 0);
#ifdef DEBUG_PSBCH_ENCODING
  printf("Scrambling:\n");

  for (int i=0; i<NR_POLAR_PSBCH_E_DWORD; i++) {
    printf("sl_psbch_e[%d]: 0x%08x\n", i, psbch->psbch_e[i]);
}
  printf("\n");
#endif

  /// QPSK modulation
  int16_t mod_psbch_e[NR_POLAR_PSBCH_E];
  for (int i = 0; i < NR_POLAR_PSBCH_E >> 1; i++) {
    AssertFatal(((i << 1) >> 5) < NR_POLAR_PSBCH_E_DWORD, "Invalid index into psbch->psbch_e. Index %d > %d\n",
                ((i << 1) >> 5), NR_POLAR_PSBCH_E_DWORD);
    uint8_t idx = ((psbch->psbch_e[(i << 1) >> 5] >> ((i << 1) & 0x1f)) & 3);
    AssertFatal(((idx << 1) + 1) < 8, "Invalid index into nr_qpsk_mod_table. Index %d > 8\n",
                (idx << 1) + 1);
    AssertFatal(((i << 1) + 1) < (sizeof(mod_psbch_e) / sizeof(mod_psbch_e[0])), "Invalid index into mod_psbch_e. Index %d > %lu\n",
                (i << 1) + 1, sizeof(mod_psbch_e) / sizeof(mod_psbch_e[0]));
    mod_psbch_e[i << 1] = nr_qpsk_mod_table[idx << 1];
    mod_psbch_e[(i << 1) + 1] = nr_qpsk_mod_table[(idx << 1) + 1];
#ifdef DEBUG_PSBCH
    printf("i %d idx %d  mod_psbch %d %d\n", i, idx, mod_psbch_e[2*i], mod_psbch_e[2*i+1]);
#endif
  }

  /// Resource mapping
  nushift = 0; //config->cell_config.phy_cell_id.value &3;
  // PSBCH modulated symbols are mapped  within the SSB block on symbols 1, 2, 3 excluding the subcarriers used for the PSBCH DMRS
  ///symbol 1  [0:132] -- 99 mod symbols
  int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier;
  int l = ssb_start_symbol;
  int m = 0;

  for (int ssb_sc_idx = 0; ssb_sc_idx < NR_PSBCH_MAX_NB_CARRIERS; ssb_sc_idx++) {
    if ((ssb_sc_idx & 3) == nushift) {  //skip DMRS
      k++;
      continue;
    } else {
#ifdef DEBUG_PSBCH
      printf("m %d ssb_sc_idx %d at k %d of l %d\n", m, ssb_sc_idx, k, l);
#endif
      AssertFatal(((m << 1) + 1) < (sizeof(mod_psbch_e) / sizeof(mod_psbch_e[0])), "Invalid index into mod_psbch_e. Index %d > %lu\n",
                (m << 1) + 1, sizeof(mod_psbch_e) / sizeof(mod_psbch_e[0]));
      int idx = (l * frame_parms->ofdm_symbol_size + k) << 1;
      AssertFatal((idx + 1) < frame_parms->samples_per_frame_wCP, "txdataF index %d invalid!\n", idx + 1);
      ((int16_t *)txdataF)[idx] = (amp * mod_psbch_e[m << 1]) >> 15;
      ((int16_t *)txdataF)[(idx) + 1] = (amp * mod_psbch_e[(m << 1) + 1]) >> 15;
      k++;
      m++;
    }

    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }

 int N_SSSB_Symb = 14;
  ///symbol 5  to N_SSSB_Symb [0:132] -- 99 mod symbols
  l = ssb_start_symbol + 5;
  AssertFatal(m == 99, "m does not equal 99");
  m = 99;
  while (l < N_SSSB_Symb - 1)
  {
    k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier;

    for (int ssb_sc_idx = 0; ssb_sc_idx < NR_PSBCH_MAX_NB_CARRIERS; ssb_sc_idx++) {
      if ((ssb_sc_idx & 3) == nushift) {  //skip DMRS
        k++;
        continue;
      } else {
  #ifdef DEBUG_PSBCH
        printf("m %d ssb_sc_idx %d at k %d of l %d\n", m, ssb_sc_idx, k, l);
  #endif

        AssertFatal((m << 1) + 1 < (sizeof(mod_psbch_e) / sizeof(mod_psbch_e[0])),
                    "Indexing outside of mod_psbch_e bounds. %d > %lu",
                    (m << 1) + 1 , (sizeof(mod_psbch_e) / sizeof(mod_psbch_e[0])));

        ((int16_t *)txdataF)[(l * frame_parms->ofdm_symbol_size + k) << 1] = (amp * mod_psbch_e[m << 1]) >> 15;
        ((int16_t *)txdataF)[((l * frame_parms->ofdm_symbol_size + k) << 1) + 1] = (amp * mod_psbch_e[(m << 1) + 1]) >> 15;
        k++;
        m++;
      }

      if (k >= frame_parms->ofdm_symbol_size)
        k-=frame_parms->ofdm_symbol_size;
    }
    l++;
  }

  return 0;
}
