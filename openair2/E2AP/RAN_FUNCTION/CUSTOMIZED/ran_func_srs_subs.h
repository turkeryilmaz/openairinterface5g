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

#ifndef RAN_FUNC_SM_SRS_SUBSCRIPTION_AGENT_H
#define RAN_FUNC_SM_SRS_SUBSCRIPTION_AGENT_H

#include "openair2/E2AP/flexric/src/sm/srs_sm/ie/srs_data_ie.h"
#include "common/utils/ds/seq_arr.h"

typedef struct ran_param_data {
  uint32_t ric_req_id;
  //srs_event_trigger_t ev_tr;
} ran_param_data_t;


// seq_arr_t srs_subs_data;

void init_srs_subs_data(seq_arr_t *srs_subs_data);
void insert_srs_subs_data(seq_arr_t *seq_arr, ran_param_data_t *data);
void remove_srs_subs_data(seq_arr_t *srs_subs_data, uint32_t ric_req_id);

#endif
