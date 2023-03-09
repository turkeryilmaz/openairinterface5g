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

#include "PHY/defs_nr_UE.h"

void free_nr_ue_slsch_rx(NR_UE_ULSCH_t **slschptr, uint16_t N_RB_UL);

uint32_t nr_slsch_decoding(PHY_VARS_NR_UE *ue,
                           uint8_t SLSCH_id,
                           short *slsch_llr,
                           NR_DL_FRAME_PARMS *frame_parms,
                           nfapi_nr_pssch_pdu_t *pssch_pdu,
                           uint32_t frame,
                           uint8_t slot,
                           uint8_t harq_pid,
                           uint32_t G);
