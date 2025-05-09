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
#include "ran_func_srs_extern.h"
#include "ran_func_srs_subs.h"
#include <stdio.h>
#include <assert.h>
#include "openair2/E2AP/flexric/src/util/time_now_us.h"
#include "openair2/E2AP/flexric/test/rnd/fill_rnd_data_srs.h"

#include "../../flexric/src/agent/e2_agent_api.h"
#include <unistd.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

// for the packing functions
#include "nfapi/open-nFAPI/common/public_inc/nfapi.h"
static seq_arr_t srs_subs_data = {0};

bool read_srs_sm(void* data)
{
  assert(data != NULL);
  assert(0!=0 && "Not implemented");
  return true;
}


void read_srs_setup_sm(void* data)
{
  assert(data != NULL);
  assert(0!=0 && "Not implemented");
}

sm_ag_if_ans_t write_ctrl_srs_sm(void const* src)
{
  assert(src != NULL);
  assert(0 !=0 && "Not supported");
}


static srs_ind_hdr_t fill_srs_ind_hdr(void)
{
  srs_ind_hdr_t hdr = {0};
  hdr.ev_trigger_cond_id = 2;
  return hdr;
}

// from nfapi/open-nFAPI/fapi/srs/nr_fapi_p7.c
// report tlv is the what contains the srs ul channel estimates
// ppWritePackedMsg is the destination buffer (byte_array our case) 
uint8_t pack_nr_srs_report_tlv(const nfapi_srs_report_tlv_t *report_tlv, uint8_t **ppWritePackedMsg, uint8_t *end) {

  if(!(push16(report_tlv->tag, ppWritePackedMsg, end) &&
        push32(report_tlv->length, ppWritePackedMsg, end))) {
    return 0;
  }

  for (int i = 0; i < (report_tlv->length + 3) / 4; i++) {
    if (!push32(report_tlv->value[i], ppWritePackedMsg, end)) {
      return 0;
    }
  }

  return 1;
}

uint8_t pack_nr_srs_indication_body(const nfapi_nr_srs_indication_pdu_t *value, uint8_t **ppWritePackedMsg, uint8_t *end) {

  if(!(push32(value->handle, ppWritePackedMsg, end) &&
        push16(value->rnti, ppWritePackedMsg, end) &&
        push16(value->timing_advance_offset, ppWritePackedMsg, end) &&
        pushs16(value->timing_advance_offset_nsec, ppWritePackedMsg, end) &&
        push8(value->srs_usage, ppWritePackedMsg, end) &&
        push8(value->report_type, ppWritePackedMsg, end))) {
    return 0;
  }

  if (!pack_nr_srs_report_tlv(&value->report_tlv, ppWritePackedMsg, end)) {
    return 0;
  }

  return 1;
}
/*
static uint8_t unpack_nr_srs_report_tlv(nfapi_srs_report_tlv_t *report_tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {

  if(!(pull16(ppReadPackedMsg, &report_tlv->tag, end) &&
        pull32(ppReadPackedMsg, &report_tlv->length, end))) {
    return 0;
  }
  return 1;
}

static uint8_t unpack_nr_srs_indication_body(nfapi_nr_srs_indication_pdu_t *value, uint8_t **ppReadPackedMsg, uint8_t *end) {

  if(!(pull32(ppReadPackedMsg, &value->handle, end) &&
        pull16(ppReadPackedMsg, &value->rnti, end) &&
        pull16(ppReadPackedMsg, &value->timing_advance_offset, end) &&
        pulls16(ppReadPackedMsg, &value->timing_advance_offset_nsec, end) &&
        pull8(ppReadPackedMsg, &value->srs_usage, end) &&
        pull8(ppReadPackedMsg, &value->report_type, end))) {
    return 0;
  }

  if (!unpack_nr_srs_report_tlv(&value->report_tlv, ppReadPackedMsg, end)) {
    return 0;
  }

  return 1;
}
*/
static srs_ind_msg_t fill_srs_ind_msg(nfapi_nr_srs_indication_pdu_t *nfapi_srs_ind)
{
  srs_ind_msg_t msg = {0};
  msg.len = 1; // only for now
  msg.tstamp = time_now_us();
  
  if(msg.len > 0 ){  
    msg.indication_stats = calloc(msg.len, sizeof(srs_indication_stats_impl_t));
    assert(msg.indication_stats != NULL && "Memory exhausted");
  }

  for(uint32_t i = 0; i < msg.len; ++i){
    srs_indication_stats_impl_t* indication_stats = &msg.indication_stats[i];
  
    indication_stats->rnti=nfapi_srs_ind->rnti;
    printf("SRS.indication RNTI: %u\n", indication_stats->rnti);
    byte_array_t ba = {.len = sizeof(nfapi_nr_srs_indication_pdu_t)}; //copy_byte_array(srs_ba);
    ba.buf = malloc(ba.len);
    uint8_t *pPackedBuf = ba.buf;
    uint8_t *pWritePackedMessage    = pPackedBuf;
    uint8_t *pPackMessageEnd =  pPackedBuf + ba.len;
    uint8_t result1 = pack_nr_srs_indication_body(nfapi_srs_ind, &pWritePackedMessage, pPackMessageEnd);

    size_t packedBufLen = pWritePackedMessage - pPackedBuf;// this should be eq to the ba.len
    ba.len = packedBufLen;
    indication_stats->srs_unpacked_pdu = copy_byte_array(ba); //.len = packedBufLen;
    //indication_stats->srs_unpacked_pdu.buf = pWritePackedMessage;

    // uint8_t *pReadPackedMessage = pPackedBuf;
    // uint8_t *pUnpackMessageEnd = pPackedBuf + packedBufLen;
    // nfapi_nr_srs_indication_pdu_t srs_ind_pdu = {0};

    // uint8_t result2 = unpack_nr_srs_indication_body(&srs_ind_pdu, &pReadPackedMessage, pUnpackMessageEnd);
    // printf("Unpacked RNTI:%u\n", srs_ind_pdu.rnti);
    // printf("unpacking result: %u\n",result2);

    // Clean up
    free_byte_array(ba);
  }

  return msg;
}

static void send_ric_indication(const uint32_t ric_req_id, srs_ind_data_t* srs_ind_data)
{
  async_event_agent_api(ric_req_id, srs_ind_data);
  printf("Event for RIC Req ID %u generated\n", ric_req_id);
}


static void free_aperiodic_subscription(uint32_t ric_req_id)
{
  remove_srs_subs_data(&srs_subs_data, ric_req_id);
}


static srs_ind_data_t* fill_fapi_srs_indication(nfapi_nr_srs_indication_pdu_t *nfapi_srs_ind)
{
  srs_ind_data_t* srs_ind = calloc(1,sizeof(srs_ind_data_t));
  assert(srs_ind != NULL && "Memory exhausted");

  srs_ind->hdr = fill_srs_ind_hdr();
  srs_ind->msg = fill_srs_ind_msg(nfapi_srs_ind);

  return srs_ind;
}

void signal_nfapi_srs_indication(nfapi_nr_srs_indication_pdu_t *nfapi_srs_ind)
{
  // Check number of subscriptions:
  const size_t num_subs = seq_arr_size(&srs_subs_data);
  for (size_t sub_idx = 0; sub_idx < num_subs; sub_idx++) {
    const ran_param_data_t data = *(const ran_param_data_t *)seq_arr_at(&srs_subs_data, sub_idx);
    srs_ind_data_t* srs_ind_data = fill_fapi_srs_indication(nfapi_srs_ind);
    //Send RIC indication
    // printf("Sending buf: %p \t len: %lu\n", (void*) &srs_ind_data->msg.indication_stats->srs_unpacked_pdu.buf, &srs_ind_data->msg.indication_stats->srs_unpacked_pdu.len);

    send_ric_indication(data.ric_req_id, srs_ind_data);
  }
}





sm_ag_if_ans_t write_subs_srs_sm(void const* src)
{
  assert(src != NULL);
  wr_srs_sub_data_t* wr_srs = (wr_srs_sub_data_t*)src;
  assert(wr_srs->srs.ad != NULL && "Cannot be NULL");

  sm_ag_if_ans_t ans = {0};

  const uint32_t ric_req_id = wr_srs->ric_req_id;

  struct ran_param_data data = { .ric_req_id = ric_req_id};
  init_srs_subs_data(&srs_subs_data);
  insert_srs_subs_data(&srs_subs_data,&data);

  ans.type = SUBS_OUTCOME_SM_AG_IF_ANS_V0;
  ans.subs_out.type = APERIODIC_SUBSCRIPTION_FLRC;
  ans.subs_out.aper.free_aper_subs = free_aperiodic_subscription;


  return ans;
}