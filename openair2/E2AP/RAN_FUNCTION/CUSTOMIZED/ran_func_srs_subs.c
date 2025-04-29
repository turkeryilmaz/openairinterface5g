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

#include "ran_func_srs_subs.h"
#include "common/utils/assertions.h"
#include "common/utils/alg/find.h"

#include <assert.h>
#include <pthread.h>


// static pthread_mutex_t srs_mutex = PTHREAD_MUTEX_INITIALIZER;

 static bool eq_int(const void* value, const void* it)
 {
   const uint32_t ric_req_id = *(uint32_t *)value;
   const ran_param_data_t *dit = (const ran_param_data_t *)it;
   return ric_req_id == dit->ric_req_id;
 }

void init_srs_subs_data(seq_arr_t *srs_subs_data)
{
  seq_arr_init(srs_subs_data, sizeof(ran_param_data_t));

}
void insert_srs_subs_data(seq_arr_t *seq_arr, ran_param_data_t *data)
{
    // Insert RIC Req ID
    seq_arr_push_back(seq_arr, data, sizeof(*data));
}
void remove_srs_subs_data(seq_arr_t *srs_subs_data, uint32_t ric_req_id){
    elm_arr_t elm = find_if(srs_subs_data, (void *)&ric_req_id, eq_int);
    ran_param_data_t *data = elm.it;
    if (data != NULL) {
      // free_srs_event_trigger...
      seq_arr_erase(srs_subs_data, elm.it);
    }
}