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

#include <stdio.h>
#include <assert.h>
#include <errno.h>
#include <math.h>
#include <nr-uesoftmodem.h>

#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"
#include "executables/softmodem-common.h"

//#define DEBUG_PSBCH
//#define DEBUG_PSBCH_ENCODING
//#define DEBUG_PSBCH_DMRS


int nr_sl_generate_psbch_dmrs(uint32_t *gold_psbch_dmrs,
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
           ((int16_t *)txdataF)[(idx)],
           ((int16_t *)txdataF)[(idx)+1]);
#endif
    k+=4;
    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }

  int N_SSSB_Symb = 13;
  l = ssb_start_symbol + 5;
  while (l < N_SSSB_Symb)
  {
    k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier;
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
      printf("%d (%d,%d)\n", m,
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