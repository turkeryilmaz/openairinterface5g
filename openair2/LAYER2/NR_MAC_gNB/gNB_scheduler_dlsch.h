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

#ifndef DLSCH_SCHEDULER_H
#define DLSCH_SCHEDULER_H

#include "NR_MAC_gNB/nr_mac_gNB.h"

typedef struct UEsched_s {
  float coef;
  NR_UE_info_t *UE;
  uint16_t num_rbs_sched; // number of resource blocks scheduled
  uint32_t num_bytes_sched;
  uint32_t tbs;
  bool status;
  uint64_t order_id;
} UEsched_t;

typedef struct LCIDsched_s {
  float coef;
  NR_UE_info_t *UE;
  uint8_t curr_lcid; // to specify the lcid of the current UE after sorting (not an optimum approach)
  uint8_t curr_ue; // to keep track of to which UE this lcid belongs to
  uint8_t lcid_priority; // calculated lcid priority based on qos flow priorities
  uint16_t num_rbs_lcid_allocated; // number of resource blocks allocated for this LCID initially
  uint32_t allocated_bytes;
  uint32_t lc_remaining_bytes;
  uint16_t overhead;
  uint32_t tbs;
} LCIDsched_t;

#endif