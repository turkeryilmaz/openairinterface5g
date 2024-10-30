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

#ifndef QOS_AWARE_DLSCH_SCHEDULER
#define QOS_AWARE_DLSCH_SCHEDULER

#define DEBUG_SCHEDULER

#include "NR_MAC_gNB/gNB_scheduler_dlsch.h"

#define LC_iterator(BaSe, VaR)         \
  LCIDsched_t *VaR##pptr = BaSe, *VaR; \
  while ((VaR = (VaR##pptr++)))
int lc_comparator(const void *p, const void *q);

int nr_find_nb_bytes(int bytes,
                     int oh_bytes,
                     uint32_t *tbs,
                     uint16_t *rbsize,
                     uint16_t nb_rb_max,
                     uint16_t nb_rb_min,
                     NR_sched_pdsch_t *sched_pdsch,
                     NR_tda_info_t *tda_info);

int nr_get_num_prioritized_bytes(int prbs_available, UEsched_t *ue_iterator, LCIDsched_t *lc_sched, int oh);

void qos_aware_post_processor(int prbs_available, UEsched_t *ue_iterator, LCIDsched_t *lc_sched, int oh);

void fill_lc_list(NR_UE_info_t *UE, frame_t frame, slot_t slot, int *lc_currid, int ue_currid, LCIDsched_t *lc_sched);

void qos_aware_scheduler_dl(module_id_t module_id,
                            frame_t frame,
                            sub_frame_t slot,
                            NR_UE_info_t **UE_list,
                            int max_num_ue,
                            int n_rb_sched
                            );

uint8_t qos_aware_pre_processor(module_id_t module_id,
                                 frame_t frame,
                                 sub_frame_t slot,
                                 LCIDsched_t *lc_sched,
                                 UEsched_t *ue_sched,
                                 int *n_rb_sched,
                                 uint8_t total_lcs,
                                 uint8_t total_UEs);

int allocate_numrbs_per_lc(module_id_t module_id,
                           frame_t frame,
                           sub_frame_t slot,
                           LCIDsched_t *lc, 
                           UEsched_t *ue_sched,
                           uint16_t *remain_prbs,
                           uint16_t *mask,
                           int32_t round_id, 
                           bool *lcids_data_status);


#endif
