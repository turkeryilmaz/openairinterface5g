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
#include "PHY/NR_REFSIG/ss_pbch_nr.h"

int nr_sl_generate_pss(int32_t *txdataF,
                       int16_t amp,
                       uint8_t ssb_start_symbol,
                       NR_DL_FRAME_PARMS *frame_parms)
{
  int16_t x[NR_PSS_LENGTH];
  const int x_initial[7] = {0, 1, 1 , 0, 1, 1, 1};

  /// Sequence generation
  for (int i = 0; i < 7; i++)
    x[i] = x_initial[i];

  for (int i = 0; i < (NR_PSS_LENGTH - 7); i++) {
    x[i + 7] = (x[i + 4] + x[i]) % 2;
  }

#ifdef NR_PSS_DEBUG
  write_output("d_pss.m", "d_pss", (void*)d_pss, NR_PSS_LENGTH, 1, 0);
  printf("PSS: ofdm_symbol_size %d, first_carrier_offset %d\n",frame_parms->ofdm_symbol_size,frame_parms->first_carrier_offset);
#endif

  /// Resource mapping

  // PSS occupies a predefined position (subcarriers 2-128, symbol 0) within the SSB block starting from
  int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + SPSS_SSSS_SUB_CARRIER_START; //and
  if (k >= frame_parms->ofdm_symbol_size) k -= frame_parms->ofdm_symbol_size;

  int l = ssb_start_symbol + 1;

  uint8_t Nid2 = frame_parms->Nid_SL / 336;
  uint8_t idx = 2 * (l * frame_parms->ofdm_symbol_size + frame_parms->ofdm_symbol_size);
  AssertFatal(idx < frame_parms->samples_per_frame_wCP, "Invalid index into txdataF. Index %d >= %d\n",
                idx, frame_parms->samples_per_frame_wCP);
  for (int i = 0; i < NR_PSS_LENGTH; i++) {
    int m = (i + 22 + 43 * Nid2) % (NR_PSS_LENGTH);
    int16_t d_pss = (1 - 2 * x[m]) * 23170;
    ((int16_t*)txdataF)[2 * (l * frame_parms->ofdm_symbol_size + k)] = (((int16_t)amp) * d_pss) >> 15;
    k++;

    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }

  // PSS occupies a predefined position (subcarriers 2-128, symbol 0) within the SSB block starting from
  k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + 2; //and
  if (k>= frame_parms->ofdm_symbol_size) k-=frame_parms->ofdm_symbol_size;

  l = ssb_start_symbol + 2;
  idx = 2 * (l * frame_parms->ofdm_symbol_size + frame_parms->ofdm_symbol_size);
  AssertFatal(idx < frame_parms->samples_per_frame_wCP, "Invalid index into txdataF. Index %d >= %d\n",
                idx, frame_parms->samples_per_frame_wCP);
  for (int i = 0; i < NR_PSS_LENGTH; i++) {
    int m = (i + 22 + 43 * Nid2)%(NR_PSS_LENGTH);
    int16_t d_pss = (1 - 2 * x[m]) * 23170;
    //      printf("pss: writing position k %d / %d\n",k,frame_parms->ofdm_symbol_size);
    ((int16_t*)txdataF)[2 * (l * frame_parms->ofdm_symbol_size + k)] = (((int16_t)amp) * d_pss) >> 15;
    k++;

    if (k >= frame_parms->ofdm_symbol_size)
      k-=frame_parms->ofdm_symbol_size;
  }

#ifdef NR_PSS_DEBUG
  LOG_M("pss_0.m", "pss_0",
    (void*)&txdataF[0][ssb_start_symbol*frame_parms->ofdm_symbol_size],
    frame_parms->ofdm_symbol_size, 1, 1);
#endif

  return 0;
}
