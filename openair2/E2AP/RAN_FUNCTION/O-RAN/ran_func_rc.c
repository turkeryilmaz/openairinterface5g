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

#include "ran_func_rc.h"
#include "ran_func_rc_subs.h"
#include "ran_func_rc_extern.h"
#include "ran_e2sm_ue_id.h"
#include "../../flexric/src/sm/rc_sm/ie/ir/lst_ran_param.h"
#include "../../flexric/src/sm/rc_sm/ie/ir/ran_param_list.h"
#include "../../flexric/src/agent/e2_agent_api.h"
#include "openair2/E2AP/flexric/src/lib/sm/enc/enc_ue_id.h"
#include "openair2/E2AP/flexric/src/sm/rc_sm/rc_sm_id.h"

#include <stdio.h>
#include <unistd.h>
#include "common/ran_context.h"

static pthread_once_t once_rc_mutex = PTHREAD_ONCE_INIT;
static rc_subs_data_t rc_subs_data = {0};
static pthread_mutex_t rc_mutex = PTHREAD_MUTEX_INITIALIZER;

static ngran_node_t get_e2_node_type(void)
{
  ngran_node_t node_type = 0;

#if defined(NGRAN_GNB_DU) && defined(NGRAN_GNB_CUUP) && defined(NGRAN_GNB_CUCP)
  node_type = RC.nrrrc[0]->node_type;
#elif defined (NGRAN_GNB_CUUP)
  node_type =  ngran_gNB_CUUP;
#endif

  return node_type;
}

static void init_once_rc(void)
{
  init_rc_subs_data(&rc_subs_data);
}

static seq_ev_trg_style_t fill_ev_tr_format_4(const char *ev_style_name)
{
  seq_ev_trg_style_t ev_trig_style = {0};

  // RIC Event Trigger Style Type
  // Mandatory
  // 9.3.3
  ev_trig_style.style = 4;

  // RIC Event Trigger Style Name
  // Mandatory
  // 9.3.4
  ev_trig_style.name = cp_str_to_ba(ev_style_name);

  // RIC Event Trigger Format Type
  // Mandatory
  // 9.3.5
  ev_trig_style.format = FORMAT_4_E2SM_RC_EV_TRIGGER_FORMAT;

  return ev_trig_style;
}

static seq_ev_trg_style_t fill_ev_tr_format_1(const char *ev_style_name)
{
  seq_ev_trg_style_t ev_trig_style = {0};

  // RIC Event Trigger Style Type
  // Mandatory
  // 9.3.3
  ev_trig_style.style = 1;

  // RIC Event Trigger Style Name
  // Mandatory
  // 9.3.4
  ev_trig_style.name = cp_str_to_ba(ev_style_name);

  // RIC Event Trigger Format Type
  // Mandatory
  // 9.3.5
  ev_trig_style.format = FORMAT_1_E2SM_RC_EV_TRIGGER_FORMAT;

  return ev_trig_style;
}

static void fill_rc_ev_trig(ran_func_def_ev_trig_t* ev_trig)
{
  // Sequence of EVENT TRIGGER styles
  // [1 - 63]
  ev_trig->sz_seq_ev_trg_style = 2;
  ev_trig->seq_ev_trg_style = calloc(ev_trig->sz_seq_ev_trg_style, sizeof(seq_ev_trg_style_t));
  assert(ev_trig->seq_ev_trg_style != NULL && "Memory exhausted");

  ev_trig->seq_ev_trg_style[0] = fill_ev_tr_format_1("Message Event");
  ev_trig->seq_ev_trg_style[1] = fill_ev_tr_format_4("UE Information Change");

  // Sequence of RAN Parameters for L2 Variables
  // [0 - 65535]
  ev_trig->sz_seq_ran_param_l2_var = 0;
  ev_trig->seq_ran_param_l2_var = NULL;

  //Sequence of Call Process Types
  // [0-65535]
  ev_trig->sz_seq_call_proc_type = 0;
  ev_trig->seq_call_proc_type = NULL;

  // Sequence of RAN Parameters for Identifying UEs
  // 0-65535
  ev_trig->sz_seq_ran_param_id_ue = 0;
  ev_trig->seq_ran_param_id_ue = NULL;

  // Sequence of RAN Parameters for Identifying Cells
  // 0-65535
  ev_trig->sz_seq_ran_param_id_cell = 0;
  ev_trig->seq_ran_param_id_cell = NULL;
}

static seq_report_sty_t fill_report_style_4(const char *report_name)
{
  seq_report_sty_t report_style = {0};

  // RIC Report Style Type
  // Mandatory
  // 9.3.3
  report_style.report_type = 4;

  // RIC Report Style Name
  // Mandatory
  // 9.3.4
  report_style.name = cp_str_to_ba(report_name);

  // Supported RIC Event Trigger Style Type 
  // Mandatory
  // 9.3.3
  report_style.ev_trig_type = FORMAT_4_E2SM_RC_EV_TRIGGER_FORMAT;

  // RIC Report Action Format Type
  // Mandatory
  // 9.3.5
  report_style.act_frmt_type = FORMAT_1_E2SM_RC_ACT_DEF;

  // RIC Indication Header Format Type
  // Mandatory
  // 9.3.5
  report_style.ind_hdr_type = FORMAT_1_E2SM_RC_IND_HDR;

  // RIC Indication Message Format Type
  // Mandatory
  // 9.3.5
  report_style.ind_msg_type = FORMAT_2_E2SM_RC_IND_MSG;

  // Sequence of RAN Parameters Supported
  // [0 - 65535]
  report_style.sz_seq_ran_param = 1;
  report_style.ran_param = calloc(report_style.sz_seq_ran_param, sizeof(seq_ran_param_3_t));
  assert(report_style.ran_param != NULL && "Memory exhausted");

  // RAN Parameter ID
  // Mandatory
  // 9.3.8
  // [1- 4294967295]
  report_style.ran_param[0].id = E2SM_RC_RS4_RRC_STATE_CHANGED_TO;

  // RAN Parameter Name
  // Mandatory
  // 9.3.9
  // [1-150] 
  const char ran_param_name[] = "RRC State Changed To";
  report_style.ran_param[0].name = cp_str_to_ba(ran_param_name);

  // RAN Parameter Definition
  // Optional
  // 9.3.51
  report_style.ran_param[0].def = NULL;

  return report_style;
}

static seq_report_sty_t fill_report_style_1(const char *report_name)
{
  seq_report_sty_t report_style = {0};

  // RIC Report Style Type
  // Mandatory
  // 9.3.3
  report_style.report_type = 1;

  // RIC Report Style Name
  // Mandatory
  // 9.3.4
  report_style.name = cp_str_to_ba(report_name);

  // Supported RIC Event Trigger Style Type 
  // Mandatory
  // 9.3.3
  report_style.ev_trig_type = FORMAT_1_E2SM_RC_EV_TRIGGER_FORMAT;

  // RIC Report Action Format Type
  // Mandatory
  // 9.3.5
  report_style.act_frmt_type = FORMAT_1_E2SM_RC_ACT_DEF;

  // RIC Indication Header Format Type
  // Mandatory
  // 9.3.5
  report_style.ind_hdr_type = FORMAT_1_E2SM_RC_IND_HDR;

  // RIC Indication Message Format Type
  // Mandatory
  // 9.3.5
  report_style.ind_msg_type = FORMAT_1_E2SM_RC_IND_MSG;

  // Sequence of RAN Parameters Supported
  // [0 - 65535]
  report_style.sz_seq_ran_param = 2;
  report_style.ran_param = calloc_or_fail(report_style.sz_seq_ran_param, sizeof(seq_ran_param_3_t));

  // RAN Parameter ID
  // Mandatory
  // 9.3.8
  // [1- 4294967295]
  report_style.ran_param[0].id = E2SM_RC_RS1_RRC_MESSAGE;
  report_style.ran_param[1].id = E2SM_RC_RS1_UE_ID;

  // RAN Parameter Name
  // Mandatory
  // 9.3.9
  // [1-150] 
  const char ran_param_name_0[] = "RRC Message";
  report_style.ran_param[0].name = cp_str_to_ba(ran_param_name_0);
  const char ran_param_name_1[] = "UE ID";
  report_style.ran_param[1].name = cp_str_to_ba(ran_param_name_1);

  // RAN Parameter Definition
  // Optional
  // 9.3.51
  report_style.ran_param[0].def = NULL;
  report_style.ran_param[1].def = NULL;

  return report_style;
}

static void fill_rc_report(ran_func_def_report_t* report)
{
  // Sequence of REPORT styles
  // [1 - 63]
  report->sz_seq_report_sty = 2;
  report->seq_report_sty = calloc(report->sz_seq_report_sty, sizeof(seq_report_sty_t));
  assert(report->seq_report_sty != NULL && "Memory exhausted");

  report->seq_report_sty[0] = fill_report_style_1("Message Copy");
  report->seq_report_sty[1] = fill_report_style_4("UE Information");
}

static void fill_rc_control(ran_func_def_ctrl_t* ctrl)
{
  // Sequence of CONTROL styles
  // [1 - 63]
  ctrl->sz_seq_ctrl_style = 1;
  ctrl->seq_ctrl_style = calloc(ctrl->sz_seq_ctrl_style, sizeof(seq_ctrl_style_t));
  assert(ctrl->seq_ctrl_style != NULL && "Memory exhausted");

  seq_ctrl_style_t* ctrl_style = &ctrl->seq_ctrl_style[0];

  // RIC Control Style Type
  // Mandatory
  // 9.3.3
  // 6.2.2.2
  ctrl_style->style_type = 1;

  //RIC Control Style Name
  //Mandatory
  //9.3.4
  // [1 -150]
  const char control_name[] = "Radio Bearer Control";
  ctrl_style->name = cp_str_to_ba(control_name);

  // RIC Control Header Format Type
  // Mandatory
  // 9.3.5
  ctrl_style->hdr = FORMAT_1_E2SM_RC_CTRL_HDR;

  // RIC Control Message Format Type
  // Mandatory
  // 9.3.5
  ctrl_style->msg = FORMAT_1_E2SM_RC_CTRL_MSG;

  // RIC Call Process ID Format Type
  // Optional
  ctrl_style->call_proc_id_type = NULL;

  // RIC Control Outcome Format Type
  // Mandatory
  // 9.3.5
  ctrl_style->out_frmt = FORMAT_1_E2SM_RC_CTRL_OUT;

  // Sequence of Control Actions
  // [0-65535]
  ctrl_style->sz_seq_ctrl_act = 1;
  ctrl_style->seq_ctrl_act = calloc(ctrl_style->sz_seq_ctrl_act, sizeof(seq_ctrl_act_2_t));
  assert(ctrl_style->seq_ctrl_act != NULL && "Memory exhausted");

  seq_ctrl_act_2_t* ctrl_act = &ctrl_style->seq_ctrl_act[0];

  // Control Action ID
  // Mandatory
  // 9.3.6
  // [1-65535]
  ctrl_act->id = 2;

  // Control Action Name
  // Mandatory
  // 9.3.7
  // [1-150]
  const char control_act_name[] = "QoS flow mapping configuration";
  ctrl_act->name = cp_str_to_ba(control_act_name);

  // Sequence of Associated RAN Parameters
  // [0-65535]
  ctrl_act->sz_seq_assoc_ran_param = 2;
  ctrl_act->assoc_ran_param = calloc(ctrl_act->sz_seq_assoc_ran_param, sizeof(seq_ran_param_3_t));
  assert(ctrl_act->assoc_ran_param != NULL && "Memory exhausted");

  seq_ran_param_3_t* assoc_ran_param = ctrl_act->assoc_ran_param;

  // DRB ID
  assoc_ran_param[0].id = 1;
  const char ran_param_drb_id[] = "DRB ID";
  assoc_ran_param[0].name = cp_str_to_ba(ran_param_drb_id);
  assoc_ran_param[0].def = NULL;

  // List of QoS Flows to be modified in DRB
  assoc_ran_param[1].id = 2;
  const char ran_param_list_qos_flow[] = "List of QoS Flows to be modified in DRB";
  assoc_ran_param[1].name = cp_str_to_ba(ran_param_list_qos_flow);
  assoc_ran_param[1].def = calloc(1, sizeof(ran_param_def_t));
  assert(assoc_ran_param[1].def != NULL && "Memory exhausted");

  assoc_ran_param[1].def->type = LIST_RAN_PARAMETER_DEF_TYPE;
  assoc_ran_param[1].def->lst = calloc(1, sizeof(ran_param_type_t));
  assert(assoc_ran_param[1].def->lst != NULL && "Memory exhausted");

  ran_param_type_t* lst = assoc_ran_param[1].def->lst;
  lst->sz_ran_param = 2;
  lst->ran_param = calloc(lst->sz_ran_param, sizeof(ran_param_lst_struct_t));
  assert(lst->ran_param != NULL && "Memory exhausted");

  // QoS Flow Identifier
  lst->ran_param[0].ran_param_id = 4;
  const char ran_param_qos_flow_id[] = "QoS Flow Identifier";
  lst->ran_param[0].ran_param_name = cp_str_to_ba(ran_param_qos_flow_id);
  lst->ran_param[0].ran_param_def = NULL;

  // QoS Flow Mapping Indication
  lst->ran_param[1].ran_param_id = 5;
  const char ran_param_qos_flow_mapping_ind[] = "QoS Flow Mapping Indication";
  lst->ran_param[1].ran_param_name = cp_str_to_ba(ran_param_qos_flow_mapping_ind);
  lst->ran_param[1].ran_param_def = NULL;

  // Sequence of Associated RAN 
  // Parameters for Control Outcome
  // [0- 255]
  ctrl_style->sz_ran_param_ctrl_out = 0;
  ctrl_style->ran_param_ctrl_out = NULL;
}

static ran_function_name_t fill_rc_ran_func_name(void)
{
  ran_function_name_t dst = {
    .name = cp_str_to_ba(SM_RAN_CTRL_SHORT_NAME),
    .oid = cp_str_to_ba(SM_RAN_CTRL_OID),
    .description = cp_str_to_ba(SM_RAN_CTRL_DESCRIPTION),
    .instance = NULL
  };
  return dst;
}

e2sm_rc_func_def_t fill_rc_ran_def_gnb(void)
{
  e2sm_rc_func_def_t def = {0};

  // RAN Function Name
  // Mandatory
  // 9.3.2
  def.name = fill_rc_ran_func_name();

  // RAN Function Definition for EVENT TRIGGER
  // Optional
  // 9.2.2.2
  def.ev_trig = calloc(1, sizeof(ran_func_def_ev_trig_t));
  assert(def.ev_trig != NULL && "Memory exhausted");
  fill_rc_ev_trig(def.ev_trig);

  // RAN Function Definition for REPORT
  // Optional
  // 9.2.2.3
  def.report = calloc(1, sizeof(ran_func_def_report_t));
  assert(def.report != NULL && "Memory exhausted");
  fill_rc_report(def.report);

  // RAN Function Definition for INSERT
  // Optional
  // 9.2.2.4
  def.insert = NULL;

  // RAN Function Definition for CONTROL
  // Optional
  // 9.2.2.5
  def.ctrl = calloc(1, sizeof(ran_func_def_ctrl_t));
  assert(def.ctrl != NULL && "Memory exhausted");
  fill_rc_control(def.ctrl);

  // RAN Function Definition for POLICY
  // Optional
  // 9.2.2.6
  def.policy = NULL;

  return def;
}

static e2sm_rc_func_def_t fill_rc_ran_def_cu(void)
{
  e2sm_rc_func_def_t def = {0};

  // RAN Function Name
  // Mandatory
  // 9.3.2
  def.name = fill_rc_ran_func_name();

  // RAN Function Definition for EVENT TRIGGER
  // Optional
  // 9.2.2.2
  def.ev_trig = calloc(1, sizeof(ran_func_def_ev_trig_t));
  assert(def.ev_trig != NULL && "Memory exhausted");
  fill_rc_ev_trig(def.ev_trig);

  // RAN Function Definition for REPORT
  // Optional
  // 9.2.2.3
  def.report = calloc(1, sizeof(ran_func_def_report_t));
  assert(def.report != NULL && "Memory exhausted");
  fill_rc_report(def.report);

  // RAN Function Definition for INSERT
  // Optional
  // 9.2.2.4
  def.insert = NULL;

  // RAN Function Definition for CONTROL
  // Optional
  // 9.2.2.5
  def.ctrl = calloc(1, sizeof(ran_func_def_ctrl_t));
  assert(def.ctrl != NULL && "Memory exhausted");
  fill_rc_control(def.ctrl);

  // RAN Function Definition for POLICY
  // Optional
  // 9.2.2.6
  def.policy = NULL;

  return def;
}

static e2sm_rc_func_def_t fill_rc_ran_def_null(void)
{
  e2sm_rc_func_def_t def = {0};

  // RAN Function Name
  // Mandatory
  // 9.3.2
  def.name = fill_rc_ran_func_name();

  // RAN Function Definition for EVENT TRIGGER
  // Optional
  // 9.2.2.2
  def.ev_trig = NULL;

  // RAN Function Definition for REPORT
  // Optional
  // 9.2.2.3
  def.report = NULL;

  // RAN Function Definition for INSERT
  // Optional
  // 9.2.2.4
  def.insert = NULL;

  // RAN Function Definition for CONTROL
  // Optional
  // 9.2.2.5
  def.ctrl = NULL;

  // RAN Function Definition for POLICY
  // Optional
  // 9.2.2.6
  def.policy = NULL;

  return def;
}

static e2sm_rc_func_def_t fill_rc_ran_def_cucp(void)
{
  e2sm_rc_func_def_t def = {0};

  // RAN Function Name
  // Mandatory
  // 9.3.2
  def.name = fill_rc_ran_func_name();

  // RAN Function Definition for EVENT TRIGGER
  // Optional
  // 9.2.2.2
  def.ev_trig = calloc(1, sizeof(ran_func_def_ev_trig_t));
  assert(def.ev_trig != NULL && "Memory exhausted");
  fill_rc_ev_trig(def.ev_trig);

  // RAN Function Definition for REPORT
  // Optional
  // 9.2.2.3
  def.report = calloc(1, sizeof(ran_func_def_report_t));
  assert(def.report != NULL && "Memory exhausted");
  fill_rc_report(def.report);

  // RAN Function Definition for INSERT
  // Optional
  // 9.2.2.4
  def.insert = NULL;

  // RAN Function Definition for CONTROL
  // Optional
  // 9.2.2.5
  def.ctrl = NULL;

  // RAN Function Definition for POLICY
  // Optional
  // 9.2.2.6
  def.policy = NULL;

  return def;
}

typedef e2sm_rc_func_def_t (*fp_rc_func_def)(void);

static const fp_rc_func_def ran_def_rc[END_NGRAN_NODE_TYPE] =
{
  NULL,
  NULL,
  fill_rc_ran_def_gnb,
  NULL,
  NULL,
  fill_rc_ran_def_cu,
  NULL,
  fill_rc_ran_def_null, // DU - at the moment, no Service is supported
  NULL,
  fill_rc_ran_def_cucp,
  fill_rc_ran_def_null, // CU-UP - at the moment, no Service is supported
};

void read_rc_setup_sm(void* data)
{
  assert(data != NULL);
//  assert(data->type == RAN_CTRL_V1_3_AGENT_IF_E2_SETUP_ANS_V0);
  rc_e2_setup_t* rc = (rc_e2_setup_t*)data;

  /* Fill the RAN Function Definition with currently supported measurements */
  const ngran_node_t node_type = get_e2_node_type();
  rc->ran_func_def = ran_def_rc[node_type]();

  // E2 Setup Request is sent periodically until the connection is established
  // RC subscritpion data should be initialized only once
  const int ret = pthread_once(&once_rc_mutex, init_once_rc);
  DevAssert(ret == 0);
}

static seq_ran_param_t fill_rrc_state_change_seq_ran(const rrc_state_e2sm_rc_e rrc_state)
{
  seq_ran_param_t seq_ran_param = {0};

  seq_ran_param.ran_param_id = E2SM_RC_RS4_RRC_STATE_CHANGED_TO;
  seq_ran_param.ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
  seq_ran_param.ran_param_val.flag_false = calloc(1, sizeof(ran_parameter_value_t));
  assert(seq_ran_param.ran_param_val.flag_false != NULL && "Memory exhausted");
  seq_ran_param.ran_param_val.flag_false->type = INTEGER_RAN_PARAMETER_VALUE;
  seq_ran_param.ran_param_val.flag_false->int_ran = rrc_state;  

  return seq_ran_param;
}

static rc_ind_data_t* fill_ue_rrc_state_change(const gNB_RRC_UE_t *rrc_ue_context, const rrc_state_e2sm_rc_e rrc_state, const uint16_t cond_id)
{
  rc_ind_data_t* rc_ind = calloc(1, sizeof(rc_ind_data_t));
  assert(rc_ind != NULL && "Memory exhausted");

  // Generate Indication Header
  rc_ind->hdr.format = FORMAT_1_E2SM_RC_IND_HDR;
  rc_ind->hdr.frmt_1.ev_trigger_id = malloc_or_fail(sizeof(uint32_t));
  *rc_ind->hdr.frmt_1.ev_trigger_id = cond_id;

  // Generate Indication Message
  rc_ind->msg.format = FORMAT_2_E2SM_RC_IND_MSG;

  //Sequence of UE Identifier
  //[1-65535]
  rc_ind->msg.frmt_2.sz_seq_ue_id = 1;
  rc_ind->msg.frmt_2.seq_ue_id = calloc(rc_ind->msg.frmt_2.sz_seq_ue_id, sizeof(seq_ue_id_t));
  assert(rc_ind->msg.frmt_2.seq_ue_id != NULL && "Memory exhausted");

  // UE ID
  // Mandatory
  // 9.3.10
  const ngran_node_t node_type = get_e2_node_type();
  rc_ind->msg.frmt_2.seq_ue_id[0].ue_id = fill_ue_id_data[node_type](rrc_ue_context, 0, 0);

  // Sequence of
  // RAN Parameter
  // [1- 65535]
  rc_ind->msg.frmt_2.seq_ue_id[0].sz_seq_ran_param = 1;
  rc_ind->msg.frmt_2.seq_ue_id[0].seq_ran_param = calloc(rc_ind->msg.frmt_2.seq_ue_id[0].sz_seq_ran_param, sizeof(seq_ran_param_t));
  assert(rc_ind->msg.frmt_2.seq_ue_id[0].seq_ran_param != NULL && "Memory exhausted");
  rc_ind->msg.frmt_2.seq_ue_id[0].seq_ran_param[0] = fill_rrc_state_change_seq_ran(rrc_state);

  return rc_ind;
}

static void send_aper_ric_ind(const uint32_t ric_req_id, rc_ind_data_t* rc_ind_data)
{
  async_event_agent_api(ric_req_id, rc_ind_data);
  printf("[E2 AGENT] Event for RIC request ID %d generated\n", ric_req_id);
}

static rc_ind_data_t* fill_ue_id(const gNB_RRC_UE_t *rrc_ue_context, const uint16_t cond_id)
{
  rc_ind_data_t* rc_ind = malloc_or_fail(sizeof(rc_ind_data_t));

  // Generate Indication Header
  rc_ind->hdr.format = FORMAT_1_E2SM_RC_IND_HDR;
  rc_ind->hdr.frmt_1.ev_trigger_id = malloc_or_fail(sizeof(uint32_t));
  *rc_ind->hdr.frmt_1.ev_trigger_id = cond_id;

  // Generate Indication Message
  rc_ind->msg.format = FORMAT_1_E2SM_RC_IND_MSG;

  // Initialize RAN Parameter
  rc_ind->msg.frmt_1.sz_seq_ran_param = 1;
  rc_ind->msg.frmt_1.seq_ran_param = calloc_or_fail(rc_ind->msg.frmt_1.sz_seq_ran_param, sizeof(seq_ran_param_t));

  // Fill the RAN Parameter details for UE ID
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_id = E2SM_RC_RS1_UE_ID;
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.flag_false = malloc_or_fail(sizeof(ran_parameter_value_t));
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.flag_false->type = OCTET_STRING_RAN_PARAMETER_VALUE;

  const ngran_node_t node_type = get_e2_node_type();
  ue_id_e2sm_t ue_id_data = fill_ue_id_data[node_type](rrc_ue_context, 0, 0);

  UEID_t enc_ue_id_data = enc_ue_id_asn(&ue_id_data);

  const enum asn_transfer_syntax syntax = ATS_ALIGNED_BASIC_PER;
  asn_encode_to_new_buffer_result_t res = asn_encode_to_new_buffer(NULL, syntax, &asn_DEF_UEID, &enc_ue_id_data);
  assert(res.buffer != NULL && res.result.encoded > 0 && "[E2 agent] Failed to encode UE ID.");
  byte_array_t ba = {.buf = res.buffer, .len = res.result.encoded};

  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.flag_false->octet_str_ran = copy_byte_array(ba);

  // Call Process ID
  rc_ind->proc_id = NULL;

  free_ue_id_e2sm(&ue_id_data);
  ASN_STRUCT_RESET(asn_DEF_UEID, &enc_ue_id_data);
  free(res.buffer);

  return rc_ind;
}

static void check_ue_id_cond(const gNB_RRC_UE_t *rrc_ue_context, const uint16_t class, const uint32_t msg_id, const uint32_t ric_req_id, const e2sm_rc_ev_trg_frmt_1_t *frmt_1)
{
  for (size_t i = 0; i < frmt_1->sz_msg_ev_trg; i++) {
    msg_ev_trg_t *ev_item = &frmt_1->msg_ev_trg[i];
    if ((ev_item->msg_type == RRC_MSG_MSG_TYPE_EV_TRG && ev_item->rrc_msg.type == NR_RRC_MESSAGE_ID && ev_item->rrc_msg.nr == class && ev_item->rrc_msg.rrc_msg_id == msg_id)  // rrcSetupComplete
         || (ev_item->msg_type == NETWORK_INTERFACE_MSG_TYPE_EV_TRG && ev_item->net.ni_type == class)) {  // "F1 UE Context Setup Request", but in the subscription the class (F1) can only be specified which translates to any F1 msg
      rc_ind_data_t* rc_ind_data = fill_ue_id(rrc_ue_context, ev_item->ev_trigger_cond_id);
      send_aper_ric_ind(ric_req_id, rc_ind_data);
    }
  }
}

void signal_ue_id(const gNB_RRC_UE_t *rrc_ue_context, const uint16_t class, const uint32_t msg_id)
{
  pthread_mutex_lock(&rc_mutex);
  if (rc_subs_data.rs1_param4.data == NULL) {
    pthread_mutex_unlock(&rc_mutex);
    return;
  }

  const size_t num_subs = seq_arr_size(&rc_subs_data.rs1_param4);
  for (size_t sub_idx = 0; sub_idx < num_subs; sub_idx++) {
    const ran_param_data_t data = *(const ran_param_data_t *)seq_arr_at(&rc_subs_data.rs1_param4, sub_idx);
    check_ue_id_cond(rrc_ue_context, class, msg_id, data.ric_req_id, &data.ev_tr.frmt_1);
  }

  pthread_mutex_unlock(&rc_mutex);
}

static void check_rrc_state(const gNB_RRC_UE_t *rrc_ue_context, const rrc_state_e2sm_rc_e rrc_state, const uint32_t ric_req_id, const e2sm_rc_ev_trg_frmt_4_t *frmt_4)
{
  for (size_t i = 0; i < frmt_4->sz_ue_info_chng; i++) {
    const uint16_t cond_id = frmt_4->ue_info_chng[i].ev_trig_cond_id;
    const rrc_state_lst_t *rrc_elem = &frmt_4->ue_info_chng[i].rrc_state;
    for (size_t j = 0; j < rrc_elem->sz_rrc_state; j++) {
      const rrc_state_e2sm_rc_e ev_tr_rrc_state = rrc_elem->state_chng_to[j].state_chngd_to;
      if (ev_tr_rrc_state == rrc_state || ev_tr_rrc_state == ANY_RRC_STATE_E2SM_RC) {
        rc_ind_data_t* rc_ind_data = fill_ue_rrc_state_change(rrc_ue_context, rrc_state, cond_id);
        send_aper_ric_ind(ric_req_id, rc_ind_data);
      }
    }
  }
}

void signal_rrc_state_changed_to(const gNB_RRC_UE_t *rrc_ue_context, const rrc_state_e2sm_rc_e rrc_state)
{
  pthread_mutex_lock(&rc_mutex);
  if (rc_subs_data.rs4_param202.data == NULL) {
    pthread_mutex_unlock(&rc_mutex);
    return;
  }

  const size_t num_subs = seq_arr_size(&rc_subs_data.rs4_param202);
  for (size_t sub_idx = 0; sub_idx < num_subs; sub_idx++) {
    const ran_param_data_t data = *(const ran_param_data_t *)seq_arr_at(&rc_subs_data.rs4_param202, sub_idx);
    check_rrc_state(rrc_ue_context, rrc_state, data.ric_req_id, &data.ev_tr.frmt_4);
  }

  pthread_mutex_unlock(&rc_mutex);
}

static rc_ind_data_t* fill_rrc_msg_copy(const byte_array_t rrc_ba, const uint16_t cond_id)
{
  rc_ind_data_t* rc_ind = malloc_or_fail(sizeof(rc_ind_data_t));

  // Generate Indication Header
  rc_ind->hdr.format = FORMAT_1_E2SM_RC_IND_HDR;
  rc_ind->hdr.frmt_1.ev_trigger_id = malloc_or_fail(sizeof(uint32_t));
  *rc_ind->hdr.frmt_1.ev_trigger_id = cond_id;

  // Generate Indication Message
  rc_ind->msg.format = FORMAT_1_E2SM_RC_IND_MSG;

  // Sequence of
  // RAN Parameter
  rc_ind->msg.frmt_1.sz_seq_ran_param = 1;
  rc_ind->msg.frmt_1.seq_ran_param = calloc_or_fail(rc_ind->msg.frmt_1.sz_seq_ran_param, sizeof(seq_ran_param_t));

  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_id = E2SM_RC_RS1_RRC_MESSAGE;
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.type = ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE;
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.flag_false = malloc_or_fail(sizeof(ran_parameter_value_t));
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.flag_false->type = OCTET_STRING_RAN_PARAMETER_VALUE;
  
  rc_ind->msg.frmt_1.seq_ran_param[0].ran_param_val.flag_false->octet_str_ran = copy_byte_array(rrc_ba);

  // Call Process ID
  rc_ind->proc_id = NULL;

  return rc_ind;
}

static void check_rrc_msg_copy(const nr_rrc_class_e nr_channel, const uint32_t rrc_msg_id, const byte_array_t rrc_ba, const uint32_t ric_req_id, const e2sm_rc_ev_trg_frmt_1_t *frmt_1)
{
  for (size_t i = 0; i < frmt_1->sz_msg_ev_trg; i++) {
    if (frmt_1->msg_ev_trg[i].msg_type != RRC_MSG_MSG_TYPE_EV_TRG)
      continue;
    if (frmt_1->msg_ev_trg[i].rrc_msg.type != NR_RRC_MESSAGE_ID)
      continue;
    if (frmt_1->msg_ev_trg[i].rrc_msg.nr == nr_channel && frmt_1->msg_ev_trg[i].rrc_msg.rrc_msg_id == rrc_msg_id) {
      rc_ind_data_t* rc_ind_data = fill_rrc_msg_copy(rrc_ba, frmt_1->msg_ev_trg[i].ev_trigger_cond_id);
      send_aper_ric_ind(ric_req_id, rc_ind_data);
    }
  }
}

void signal_rrc_msg(const nr_rrc_class_e nr_channel, const uint32_t rrc_msg_id, const byte_array_t rrc_ba)
{
  pthread_mutex_lock(&rc_mutex);
  if (rc_subs_data.rs1_param3.data == NULL) {
    pthread_mutex_unlock(&rc_mutex);
    return;
  }

  const size_t num_subs = seq_arr_size(&rc_subs_data.rs1_param3);
  for (size_t sub_idx = 0; sub_idx < num_subs; sub_idx++) {
    const ran_param_data_t data = *(const ran_param_data_t *)seq_arr_at(&rc_subs_data.rs1_param3, sub_idx);
    check_rrc_msg_copy(nr_channel, rrc_msg_id, rrc_ba, data.ric_req_id, &data.ev_tr.frmt_1);
  }

  pthread_mutex_unlock(&rc_mutex);
}

static void free_aperiodic_subscription(uint32_t ric_req_id)
{
  remove_rc_subs_data(&rc_subs_data, ric_req_id);
}

static seq_arr_t *get_sa(const e2sm_rc_event_trigger_t *ev_tr, const uint32_t ran_param_id)
{
  seq_arr_t *sa = NULL;

  switch (ev_tr->format) {
    case FORMAT_1_E2SM_RC_EV_TRIGGER_FORMAT:
      if (ran_param_id == E2SM_RC_RS1_RRC_MESSAGE) {
        sa = &rc_subs_data.rs1_param3;
      } else if (ran_param_id == E2SM_RC_RS1_UE_ID) {
        sa = &rc_subs_data.rs1_param4;
      }
      break;

    case FORMAT_4_E2SM_RC_EV_TRIGGER_FORMAT:
      if (ran_param_id == E2SM_RC_RS4_RRC_STATE_CHANGED_TO) {
        sa = &rc_subs_data.rs4_param202;
      }
      break;

    default:
      printf("[E2 AGENT] RC REPORT Style %d not yet implemented.\n", ev_tr->format + 1);
      break;
  }

  return sa;
}

static void get_list_for_report_style(const uint32_t ric_req_id, const e2sm_rc_event_trigger_t *ev_tr, const size_t sz, const param_report_def_t *param_def)
{
  for (size_t i = 0; i < sz; i++) {
    seq_arr_t *sa = get_sa(ev_tr, param_def[i].ran_param_id);
    if (!sa) {
      printf("[E2 AGENT] Requested RAN Parameter ID %d not yet implemented", param_def[i].ran_param_id);
    } else {
      struct ran_param_data data = { .ric_req_id = ric_req_id, .ev_tr = cp_e2sm_rc_event_trigger(ev_tr) };
      insert_rc_subs_data(sa, &data);
    }
  }
}

sm_ag_if_ans_t write_subs_rc_sm(void const* src)
{
  assert(src != NULL); // && src->type == RAN_CTRL_SUBS_V1_03);
  wr_rc_sub_data_t* wr_rc = (wr_rc_sub_data_t*)src;
  assert(wr_rc->rc.ad != NULL && "Cannot be NULL");

  sm_ag_if_ans_t ans = {0};

  const uint32_t ric_req_id = wr_rc->ric_req_id;
  const uint32_t report_style = wr_rc->rc.ad->ric_style_type;
  // 9.2.1.2  RIC ACTION DEFINITION IE
  switch (wr_rc->rc.ad->format) {
    case FORMAT_1_E2SM_RC_ACT_DEF: { // for all REPORT styles
      // Parameters to be Reported List
      // [1-65535]
      if (wr_rc->rc.et.format + 1 != report_style) { // wr_rc->rc.et.format is an enum -> initialization starts from 0
        AssertError(false, return ans, "[E2 AGENT] Event Trigger Definition Format %d doesn't correspond to REPORT style %d.\n", wr_rc->rc.et.format + 1, report_style);
      }
      get_list_for_report_style(ric_req_id, &wr_rc->rc.et, wr_rc->rc.ad->frmt_1.sz_param_report_def, wr_rc->rc.ad->frmt_1.param_report_def);
      break;
    }

    default:
      AssertError(wr_rc->rc.ad->format == FORMAT_1_E2SM_RC_ACT_DEF, return ans, "[E2 AGENT] Action Definition Format %d not yet implemented", wr_rc->rc.ad->format + 1);
  }

  ans.type = SUBS_OUTCOME_SM_AG_IF_ANS_V0;
  ans.subs_out.type = APERIODIC_SUBSCRIPTION_FLRC;
  ans.subs_out.aper.free_aper_subs = free_aperiodic_subscription;

  return ans;
}


sm_ag_if_ans_t write_ctrl_rc_sm(void const* data)
{
  assert(data != NULL);
//  assert(data->type == RAN_CONTROL_CTRL_V1_03 );

  rc_ctrl_req_data_t const* ctrl = (rc_ctrl_req_data_t const*)data;

  assert(ctrl->hdr.format == FORMAT_1_E2SM_RC_CTRL_HDR && "Indication Header Format received not valid");
  assert(ctrl->msg.format == FORMAT_1_E2SM_RC_CTRL_MSG && "Indication Message Format received not valid");
  assert(ctrl->hdr.frmt_1.ctrl_act_id == 2 && "Currently only QoS flow mapping configuration supported");

  printf("QoS flow mapping configuration\n");

  const seq_ran_param_t* ran_param = ctrl->msg.frmt_1.ran_param;

  // DRB ID
  assert(ran_param[0].ran_param_id == 1 && "First RAN Parameter ID has to be DRB ID");
  assert(ran_param[0].ran_param_val.type == ELEMENT_KEY_FLAG_TRUE_RAN_PARAMETER_VAL_TYPE);
  printf("DRB ID %ld \n", ran_param[0].ran_param_val.flag_true->int_ran);


  // List of QoS Flows to be modified in DRB
  assert(ran_param[1].ran_param_id == 2 && "Second RAN Parameter ID has to be List of QoS Flows");
  assert(ran_param[1].ran_param_val.type == LIST_RAN_PARAMETER_VAL_TYPE);
  printf("List of QoS Flows to be modified in DRB\n");
  const lst_ran_param_t* lrp = ran_param[1].ran_param_val.lst->lst_ran_param;

  // The following assertion should be true, but there is a bug in the std
  // check src/sm/rc_sm/enc/rc_enc_asn.c:1085 and src/sm/rc_sm/enc/rc_enc_asn.c:984 
  // assert(lrp->ran_param_struct.ran_param_struct[0].ran_param_id == 3);

  // QoS Flow Identifier
  assert(lrp->ran_param_struct.ran_param_struct[0].ran_param_id == 4);
  assert(lrp->ran_param_struct.ran_param_struct[0].ran_param_val.type == ELEMENT_KEY_FLAG_TRUE_RAN_PARAMETER_VAL_TYPE);
  int64_t qfi = lrp->ran_param_struct.ran_param_struct[0].ran_param_val.flag_true->int_ran;
  assert(qfi > -1 && qfi < 65);

  // QoS Flow Mapping Indication
  assert(lrp->ran_param_struct.ran_param_struct[1].ran_param_id == 5);
  assert(lrp->ran_param_struct.ran_param_struct[1].ran_param_val.type == ELEMENT_KEY_FLAG_FALSE_RAN_PARAMETER_VAL_TYPE);
  int64_t dir = lrp->ran_param_struct.ran_param_struct[1].ran_param_val.flag_false->int_ran;
  assert(dir == 0 || dir == 1);

  printf("qfi = %ld, dir %ld \n", qfi, dir);


  sm_ag_if_ans_t ans = {.type = CTRL_OUTCOME_SM_AG_IF_ANS_V0};
  ans.ctrl_out.type = RAN_CTRL_V1_3_AGENT_IF_CTRL_ANS_V0;
  return ans;
}


bool read_rc_sm(void* data)
{
  assert(data != NULL);
//  assert(data->type == RAN_CTRL_STATS_V1_03);
  assert(0!=0 && "Not implemented");

  return true;
}
