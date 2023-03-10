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

/*! \file PHY/NR_TRANSPORT/nr_sch_dmrs.c
* \brief
* \author
* \date
* \version
* \company Eurecom
* \email:
* \note
* \warning
*/

#include "nr_sch_dmrs_sl.h"

/*Table 8.4.1.1.2-2 38211 Columns: ap - CDM group - Delta - Wf(0) - Wf(1) - Wt(0)*/
int8_t pssch_dmrs[2][6] = {{1000, 0, 0, 1, 1, 1},
                            {1001, 0, 0, 1, -1, 1}};

void get_antenna_ports_sl(uint8_t *ap, uint8_t n_symbs) {
  for (int i = 0; i < 2; i++)
    *(ap + i) = 1000 + i;
}

void get_Wt_sl(int8_t *Wt, uint8_t ap) {
    *(Wt) = pssch_dmrs[ap][5];
}

void get_Wf_sl(int8_t *Wf, uint8_t ap) {
  for (int i = 0; i < 2; i++)
    *(Wf + i) = pssch_dmrs[ap][3 + i];
}

uint8_t get_delta_sl(uint8_t ap) {
  return pssch_dmrs[ap][2];
}

uint16_t get_dmrs_freq_idx_sl(uint16_t n, uint8_t k_prime, uint8_t delta) {
  uint16_t dmrs_idx = 6 * n + k_prime + delta;
  return dmrs_idx;
}

uint8_t get_l0(uint16_t dlDmrsSymbPos) {

  uint16_t mask=dlDmrsSymbPos;
  int l0;
  for (l0=0;l0<14;l0++) {
    if ((mask&1) == 1) break;
    mask>>=1;
  }
  return (l0);
}
