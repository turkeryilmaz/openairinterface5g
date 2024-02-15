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

#include "NR_MAC_gNB/gNB_scheduler_dlsch.h"

#define LC_iterator(BaSe, VaR)         \
  LCIDsched_t *VaR##pptr = BaSe, *VaR; \
  while ((VaR = (VaR##pptr++)))

int lc_comparator(const void *p, const void *q);

int comparator(const void *p, const void *q);

int preallocate_numrbs_per_lc(LCIDsched_t *lc_sched, int num_rbs_data, float *remainPRBs);

int nr_find_nb_bytes(int bytes,
                     int oh_bytes,
                     uint32_t *tbs,
                     uint16_t *rbsize,
                     uint16_t nb_rb_max,
                     NR_sched_pdsch_t *sched_pdsch,
                     NR_tda_info_t *tda_info);

int nr_get_num_prioritized_bytes(int max_rbsize, UEsched_t *ue_iterator, LCIDsched_t *lc_sched);

uint8_t rb_allocation_lcid(module_id_t module_id,
                           frame_t frame,
                           sub_frame_t slot,
                           LCIDsched_t *lc_sched,
                           UEsched_t *ue_sched,
                           int n_rb_sched,
                           uint16_t *rballoc_mask);

void fill_lc_sched_list(NR_UE_info_t *UE, frame_t frame, int *lc_currid, int ue_currid, LCIDsched_t *lc_sched);

void qos_aware_scheduler_dl(module_id_t module_id,
                            frame_t frame,
                            sub_frame_t slot,
                            NR_UE_info_t **UE_list,
                            int max_num_ue,
                            int n_rb_sched,
                            uint16_t *rballoc_mask);

#endif
