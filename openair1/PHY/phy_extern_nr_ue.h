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

#ifndef __PHY_EXTERN_NR_UE__H__
#define __PHY_EXTERN_NR_UE__H__

#include "PHY/defs_nr_UE.h"
//#include "common/ran_context.h"

#ifdef XFORMS
  #include "PHY/TOOLS/nr_phy_scope.h"
  extern uint32_t do_forms;
#endif

extern char* namepointer_chMag ;
extern char* namepointer_log2;
extern char  fmageren_name2[512];

extern unsigned int RX_DMA_BUFFER[4][NB_ANTENNAS_RX];
extern unsigned int TX_DMA_BUFFER[4][NB_ANTENNAS_TX];

extern uint64_t downlink_frequency[MAX_NUM_CCs][4];
extern int32_t uplink_frequency_offset[MAX_NUM_CCs][4];
extern uint64_t sidelink_frequency[MAX_NUM_CCs][4];

extern const short conjugate[8],conjugate2[8];
extern int number_of_cards;


extern PHY_VARS_NR_UE ***PHY_vars_UE_g;

#endif /*__PHY_EXTERN_H__ */

