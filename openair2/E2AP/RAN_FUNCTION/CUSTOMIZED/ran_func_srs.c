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

#include "ran_func_srs.h"
#include <stdio.h>
#include <assert.h>
#include "openair2/E2AP/flexric/src/util/time_now_us.h"
#include <stdlib.h>
#include <time.h>


// define static gunctions for trigger
/*
static void log_srs_measurement_report
*/
bool read_srs_sm(void* data)
{
  assert(data != NULL);

  srand(time(0)); // tmp for now
  srs_ind_data_t* srs = (srs_ind_data_t*)data;
   // fill data tsstamp, len and srs fapi
  srs->msg.tstamp = time_now_us();
  srs->msg.len = 1; // tmp for now

  if(srs->msg.len > 0 ){  
    srs->msg.indication_stats = calloc(srs->msg.len, sizeof(srs_indication_stats_impl_t));
    assert(srs->msg.indication_stats != NULL && "Memory exhausted");
  }
  srs_indication_stats_impl_t* indication_stats = &srs->msg.indication_stats[0]; // len =1
  indication_stats->rnti=rand()%100; // tmp for now

  return true;
}

void read_srs_setup_sm(void* data)
{
  assert(data != NULL);
  assert(0 !=0 && "Not supported");
}

sm_ag_if_ans_t write_ctrl_srs_sm(void const* src)
{
  assert(src != NULL);
  assert(0 !=0 && "Not supported");
}

