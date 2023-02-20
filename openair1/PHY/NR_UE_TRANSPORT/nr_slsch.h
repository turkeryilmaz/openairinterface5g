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

/*! \file PHY/NR_TRANSPORT/nr_ulsch.h
* \brief functions used for PUSCH/ULSCH physical and transport channels for gNB
* \author Ahmed Hussein
* \date 2019
* \version 0.1
* \company Fraunhofer IIS
* \email: ahmed.hussein@iis.fraunhofer.de
* \note
* \warning
*/

#include "PHY/defs_gNB.h"
#include "common/utils/threadPool/thread-pool.h"

void free_gNB_ulsch(NR_gNB_ULSCH_t **ulsch, uint16_t N_RB_UL);

NR_gNB_ULSCH_t *new_gNB_ulsch(uint8_t max_ldpc_iterations, uint16_t N_RB_UL);


uint32_t nr_slsch_decoding(PHY_VARS_NR_UE *ue,
                           uint8_t SLSCH_id,
                           short *slsch_llr,
                           NR_DL_FRAME_PARMS *frame_parms,
                           nfapi_nr_pssch_pdu_t *pssch_pdu,
                           uint32_t frame,
                           uint8_t nr_tti_rx,
                           uint8_t harq_pid,
                           uint32_t G);

/*! \brief Perform PUSCH unscrambling. TS 38.211 V15.4.0 subclause 6.3.1.1
  @param llr, Pointer to llr bits
  @param size, length of llr bits
  @param q, codeword index (0,1)
  @param Nid, cell id
  @param n_RNTI, CRNTI
*/

void nr_ulsch_unscrambling(int16_t* llr, uint32_t size, uint32_t Nid, uint32_t n_RNTI);

void nr_ulsch_layer_demapping(int16_t *llr_cw,
				     uint8_t Nl,
				     uint8_t mod_order,
				     uint32_t length,
				     int16_t **llr_layers); 

void nr_ulsch_procedures(PHY_VARS_gNB *gNB,
                         int frame_rx,
                         int slot_rx,
                         int UE_id,
                         uint8_t harq_pid);
int16_t find_nr_ulsch(uint16_t rnti, PHY_VARS_gNB *gNB,find_type_t type);

void dump_pusch_stats(FILE *fd,PHY_VARS_gNB *gNB);

void dump_nr_I0_stats(FILE *fd,PHY_VARS_gNB *gNB);
void clear_pusch_stats(PHY_VARS_gNB *gNB);

NR_gNB_SCH_STATS_t *get_ulsch_stats(PHY_VARS_gNB *gNB,NR_gNB_ULSCH_t *ulsch);
