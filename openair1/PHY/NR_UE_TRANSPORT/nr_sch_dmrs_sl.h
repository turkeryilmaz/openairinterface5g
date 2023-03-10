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
 */

/*! \file PHY/NR_TRANSPORT/nr_sch_dmrs_sl.h
* \brief
* \author
* \date
* \version
* \company
* \email:
* \note
* \warning
*/

#ifndef NR_SCH_DMRS_SL_H
#define NR_SCH_DMRS_SL_H

#include "PHY/defs_nr_common.h"

#define NR_PSSCH_DMRS_ANTENNA_PORT0 1000
#define NR_PSSCH_DMRS_NB_ANTENNA_PORTS 2

void get_antenna_ports_sl(uint8_t *ap, uint8_t n_symbs);

void get_Wt_sl(int8_t *Wt, uint8_t ap);

void get_Wf_sl(int8_t *Wf, uint8_t ap);

uint8_t get_delta_sl(uint8_t ap);

uint16_t get_dmrs_freq_idx_sl(uint16_t n, uint8_t k_prime, uint8_t delta);

uint8_t get_l0(uint16_t dlDmrsSymbPos);

#endif
