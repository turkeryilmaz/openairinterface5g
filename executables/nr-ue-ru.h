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

#ifndef NR_UE_RU_H
#define NR_UE_RU_H

#include "PHY/defs_nr_UE.h"
#include "radio/COMMON/common_lib.h"

void nr_ue_ru_start(void);
void nr_ue_ru_end(void);
void nr_ue_ru_set_freq(PHY_VARS_NR_UE *UE, uint64_t ul_carrier, uint64_t dl_carrier, int freq_offset);
int nr_ue_ru_adjust_rx_gain(PHY_VARS_NR_UE *UE, int gain_change);
int nr_ue_ru_read(PHY_VARS_NR_UE *UE, openair0_timestamp_t *ptimestamp, void **buff, int nsamps, int num_antennas);
int nr_ue_ru_write(PHY_VARS_NR_UE *UE, openair0_timestamp_t timestamp, void **buff, int nsamps, int num_antennas, int flags);
int nr_ue_ru_write_reorder(PHY_VARS_NR_UE *UE, openair0_timestamp_t timestamp, void **txp, int nsamps, int nbAnt, int flags);
void nr_ue_ru_write_reorder_clear_context(PHY_VARS_NR_UE *UE);

#endif
