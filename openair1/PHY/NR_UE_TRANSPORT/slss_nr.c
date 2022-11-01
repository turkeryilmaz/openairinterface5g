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

/**********************************************************************
*
* FILENAME    :  slss_nr.c
*
* MODULE      :  Functions for check and generate synchronisation signal
*
* DESCRIPTION :  generation of slss
*                3GPP TS 38.211 7.4.2.3 Sidelink Synchronisation signal
*
************************************************************************/

#ifndef __NR_TRANSPORT_SLSS__C__
#define __NR_TRANSPORT_SLSS__C__
#include <stdio.h>
#include <assert.h>
#include <errno.h>

#include "PHY/defs_nr_UE.h"
//#include "PHY/MODULATION/modulation_UE.h"
#include "PHY/NR_UE_TRANSPORT/nr_transport_proto_ue.h"


void check_and_generate_slss_nr(PHY_VARS_NR_UE *ue, int frame_tx, int slot_tx)
{
  NR_SLSS_t *slss = ue->slss;
  int tx_amp;

  AssertFatal(slss!=NULL,"slss is null\n");

  LOG_D(NR_PHY, "check_and_generate_slss: frame_tx %d, slot_tx %d slss_id %u, \
                sl_numssb_withinperiod_r16 %ld, sl_timeoffsetssb_r16 %ld, \
                sl_timeinterval_r16 %ld, sl_mib_length %u\n",
                frame_tx, slot_tx, slss->slss_id,
                slss->sl_timeoffsetssb_r16, slss->sl_numssb_withinperiod_r16,
                slss-> sl_timeinterval_r16, slss->sl_mib_length);

  if (ue->sync_ref == 0) return;

  if (slss->sl_mib_length == 0) return;

  // here we have a transmission opportunity for SLSS
  ue->frame_parms.Nid_SL = slss->slss_id;

#if 0
  if (ue->SLghinitialized == 0) {
    generate_sl_grouphop(ue);
    ue->SLghinitialized = 1;
  }
#endif

  ue->tx_power_dBm[slot_tx] = -6;
  ue->tx_total_RE[slot_tx] = 72;

#if defined(EXMIMO) || defined(OAI_USRP) || defined(OAI_BLADERF) || defined(OAI_LMSSDR)
  tx_amp = get_tx_amp(ue->tx_power_dBm[slot_tx],
                      ue->tx_power_max_dBm,
                      ue->frame_parms.N_RB_UL,
                      6);
#else
  tx_amp = AMP;
#endif

  if (frame_tx == 0) LOG_I(PHY, "slss: ue->tx_power_dBm: %d, tx_amp: %d\n", ue->tx_power_dBm[slot_tx], tx_amp);

  int num_samples_per_slot = ue->frame_parms.ofdm_symbol_size * ue->frame_parms.symbols_per_slot;
  if (ue->generate_ul_signal[0] == 0) {
    for(int i = 0; i < ue->frame_parms.nb_antennas_tx; ++i) {
      LOG_D(NR_PHY,"%d.%d: clearing ul signal\n", frame_tx, slot_tx);
      AssertFatal(i < sizeof(ue->common_vars.txdataF), "Array index %d is over the Array size %lu\n", i, sizeof(ue->common_vars.txdataF));
      memset(&ue->common_vars.txdataF[i], 0, sizeof(int)* num_samples_per_slot);
    }
  }

#if 0  // PSS
  generate_slpss(ue->common_vars.txdataF,
                 tx_amp << 1,
                 &ue->frame_parms,
                 1,
                 slot_tx
                 );

  generate_slpss(ue->common_vars.txdataF,
                 tx_amp << 1,
                 &ue->frame_parms,
                 2,
                 slot_tx
                 );

  generate_slsss(ue->common_vars.txdataF,
                 slot_tx,
                 tx_amp << 2,
                 &ue->frame_parms,
                 3);
  generate_slsss(ue->common_vars.txdataF,
                 slot_tx,
                 tx_amp << 2,
                 &ue->frame_parms,
                 4);

  generate_slbch(ue->common_vars.txdataF,
                 tx_amp,
                 &ue->frame_parms,
                 slot_tx,
                 ue->slss->sl_mib);

  ue->sl_chan = PSBCH;

  generate_drs_pusch(ue,
                     NULL,
                     0,
                     tx_amp << 2,
                     slot_tx,
                     (ue->frame_parms.N_RB_UL/2)-3,
                     6,
                     0,
                     NULL,
                     0);

  LOG_D(NR_PHY,"%d.%d : NEED: SLSS nbrb %d, first rb %d\n", frame_tx, slot_tx, 6, (ue->frame_parms.N_RB_UL / 2) - 3);

  ue->generate_ul_signal[0] = 1;
  ue->slss_generated = 1;

  LOG_D(NR_PHY,"ULSCH (after slss) : signal F energy %d dB (txdataF %p) at SFN/SLT: %d/%d \n",
        dB_fixed(signal_energy(&ue->common_vars.txdataF[0], num_samples_per_slot)), &ue->common_vars.txdataF[0], frame_tx, slot_tx);
#endif
}
#endif
