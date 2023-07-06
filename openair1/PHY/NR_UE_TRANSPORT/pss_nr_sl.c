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
#include "openair1/PHY/NR_REFSIG/pss_nr.h"

static int16_t *primary_synchro_nr2_sl[NUMBER_PSS_SEQUENCE] = {0};
static c16_t *primary_synchro_time_nr_sl[NUMBER_PSS_SEQUENCE] = {0};
int nr_sl_generate_pss(c16_t *txdataF, int16_t amp, uint8_t ssb_start_symbol, NR_DL_FRAME_PARMS *frame_parms)
{
  int16_t d_pss[LENGTH_PSS_NR];
  int16_t x[LENGTH_PSS_NR];

  c16_t primary_synchro[LENGTH_PSS_NR] = {0};
  uint8_t Nid2 = frame_parms->Nid_SL / 336;
  assert(Nid2 < NUMBER_PSS_SEQUENCE);
  LOG_I(NR_PHY, "Nid_SL %d, Nid2 %d\n", frame_parms->Nid_SL, Nid2);
  int16_t *primary_synchro2 = primary_synchro_nr2_sl[Nid2]; /* pss in complex with alternatively i then q */

  const int16_t x_initial[INITIAL_PSS_NR] = {0, 1, 1, 0, 1, 1, 1};
  memcpy(x, x_initial, sizeof(x_initial));

  for (int i = 0; i < (LENGTH_PSS_NR - INITIAL_PSS_NR); i++)
    x[i + INITIAL_PSS_NR] = (x[i + 4] + x[i]) % (2);

  // PSS occupies a predefined position (subcarriers 2-128, symbol 0) within the SSB block starting from
  int k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + PSS_SSS_SUB_CARRIER_START_SL;
  if (k >= frame_parms->ofdm_symbol_size)
    k -= frame_parms->ofdm_symbol_size;

  int l = ssb_start_symbol + 1;
  for (int i = 0; i < NR_PSS_LENGTH; i++) {
    int m = (i + 22 + 43 * Nid2) % (NR_PSS_LENGTH);
    d_pss[i] = (1 - 2 * x[m]) * 23170;
    txdataF[(l * frame_parms->ofdm_symbol_size + k)].r = (((int16_t)amp) * d_pss[i]) >> 15;
    txdataF[(l * frame_parms->ofdm_symbol_size + k)].i = 0;
    primary_synchro[i].r = (d_pss[i] * SHRT_MAX) >> SCALING_PSS_NR;
    primary_synchro2[i] = d_pss[i];
    k++;

    if (k >= frame_parms->ofdm_symbol_size)
      k -= frame_parms->ofdm_symbol_size;
  }

  // PSS occupies a predefined position (subcarriers 2-128, symbol 0) within the SSB block starting from
  k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + PSS_SSS_SUB_CARRIER_START_SL;
  if (k >= frame_parms->ofdm_symbol_size)
    k -= frame_parms->ofdm_symbol_size;

  l = ssb_start_symbol + 2;

  for (int i = 0; i < NR_PSS_LENGTH; i++) {
    int m = (i + 22 + 43 * Nid2) % (NR_PSS_LENGTH);
    d_pss[i] = (1 - 2 * x[m]) * 23170;
    //      printf("pss: writing position k %d / %d\n",k,frame_parms->ofdm_symbol_size);
    txdataF[(l * frame_parms->ofdm_symbol_size + k)].r = (((int16_t)amp) * d_pss[i]) >> 15;
    txdataF[(l * frame_parms->ofdm_symbol_size + k)].i = 0;
    k++;

    if (k >= frame_parms->ofdm_symbol_size)
      k -= frame_parms->ofdm_symbol_size;
  }

#ifdef NR_PSS_DEBUG
  LOG_M("pss_0.m",
        "pss_0",
        (void *)&txdataF[0][ssb_start_symbol * frame_parms->ofdm_symbol_size],
        frame_parms->ofdm_symbol_size,
        1,
        1);
  char buffer[frame_parms->ofdm_symbol_size];
  for (int i = 1; i < 3; i++) {
    bzero(buffer, sizeof(buffer));
    LOG_I(NR_PHY,
          "PSS %d = %s\n",
          i,
          hexdump(&txdataF[frame_parms->ofdm_symbol_size * i], frame_parms->ofdm_symbol_size, buffer, sizeof(buffer)));
  }
#endif
  k = frame_parms->first_carrier_offset + frame_parms->ssb_start_subcarrier + PSS_SSS_SUB_CARRIER_START_SL;
  if (k >= frame_parms->ofdm_symbol_size)
    k -= frame_parms->ofdm_symbol_size;
  c16_t in[sizeof(int16_t) * frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
  memset(in, 0, sizeof(in));
  for (int i = 0; i < LENGTH_PSS_NR; i++) {
    in[k] = primary_synchro[i];
    k++;
    if (k == frame_parms->ofdm_symbol_size)
      k = 0;
  }

  c16_t out[sizeof(int16_t) * frame_parms->ofdm_symbol_size] __attribute__((aligned(32)));
  memset(out, 0, sizeof(out));
  memset(primary_synchro_time_nr_sl[Nid2], 0, sizeof(int16_t) * frame_parms->ofdm_symbol_size);
  idft((int16_t)get_idft(frame_parms->ofdm_symbol_size), (int16_t *)in, (int16_t *)out, 1);
  for (unsigned int i = 0; i < frame_parms->ofdm_symbol_size; i++) {
    primary_synchro_time_nr_sl[Nid2][i] = out[i];
  }
  return 0;
}
