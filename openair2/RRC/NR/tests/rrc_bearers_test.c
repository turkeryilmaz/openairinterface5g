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

#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include "common/utils/utils.h"
#include "common/utils/assertions.h"
#include "openair2/RRC/NR/rrc_gNB_radio_bearers.h"
#include "ds/seq_arr.h"

int nr_rlc_get_available_tx_space(int module_id, int rnti, int drb_id) { return 0; }
softmodem_params_t *get_softmodem_params(void) { return NULL; }
configmodule_interface_t *uniqCfg = NULL;

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  printf("detected error at %s:%d:%s: %s\n", file, line, function, s);
  abort();
}

static void test_add_and_find_pduSession(void)
{
  gNB_RRC_UE_t ue = {0};
  seq_arr_init(&ue.pduSessions, sizeof(rrc_pdu_session_param_t));
  seq_arr_init(&ue.drbs, sizeof(drb_t));

  pdusession_t input = {0};
  input.pdusession_id = 5;
  input.n3_incoming.teid = 0x1001;

  rrc_pdu_session_param_t *session = add_pduSession(&ue.pduSessions, &input);
  AssertFatal(session != NULL, "Could not add PDU Session\n");
  AssertFatal(session->param.pdusession_id == input.pdusession_id, "PDU Session ID mismatch in added PDU Session\n");
  AssertFatal(session->param.n3_incoming.teid == input.n3_incoming.teid, "teid mismatch in added PDU Session\n");

  rrc_pdu_session_param_t *found = find_pduSession(&ue.pduSessions, input.pdusession_id);
  AssertFatal(found != NULL, "Could not find PDU Session\n");
  AssertFatal(found == session, "Found PDU Session mismatch\n");

  seq_arr_free(&ue.pduSessions, free_pdusession);
}

static void test_duplicate_add_pduSession(void)
{
  gNB_RRC_UE_t ue = {0};
  seq_arr_init(&ue.pduSessions, sizeof(rrc_pdu_session_param_t));
  seq_arr_init(&ue.drbs, sizeof(drb_t));

  pdusession_t input1 = {0};
  input1.pdusession_id = 3;
  input1.n3_incoming.teid = 2002;

  pdusession_t input2 = {0};
  input2.pdusession_id = 3;
  input2.n3_incoming.teid = 9999;

  rrc_pdu_session_param_t *s1 = add_pduSession(&ue.pduSessions, &input1);
  AssertFatal(s1 != NULL, "First add_pduSession failed\\n");

  rrc_pdu_session_param_t *s2 = add_pduSession(&ue.pduSessions, &input2);
  AssertFatal(s2 == s1, "Duplicate add_pduSession returned different pointer\\n");
  AssertFatal(s2->param.n3_incoming.teid == input1.n3_incoming.teid, "Original TEID should be retained\\n");

  seq_arr_free(&ue.pduSessions, free_pdusession);
}

static void test_find_pduSession_from_drbId(void)
{
  gNB_RRC_UE_t ue = {0};
  seq_arr_init(&ue.pduSessions, sizeof(rrc_pdu_session_param_t));
  seq_arr_init(&ue.drbs, sizeof(drb_t));

  const int session_id = 66;
  pdusession_t in = {0};
  in.pdusession_id = session_id;
  add_pduSession(&ue.pduSessions, &in);

  nr_pdcp_configuration_t pdcp = {.drb.discard_timer = 100, .drb.sn_size = 18, .drb.t_reordering = 50};
  drb_t *added = nr_rrc_add_drb(&ue.drbs, session_id, &pdcp);
  AssertFatal(added != NULL, "Failed to add DRB");

  rrc_pdu_session_param_t *found = find_pduSession_from_drbId(&ue, added->drb_id);
  AssertFatal(found && found->param.pdusession_id == session_id, "find_pduSession_from_drbId failed");

  seq_arr_free(&ue.pduSessions, free_pdusession);
  seq_arr_free(&ue.drbs, free_drb);
}

// ---------------- DRB TESTS ----------------

static void test_add_rrc_drb(void)
{
  seq_arr_t pduSessions = {0};
  seq_arr_t drbs = {0};
  seq_arr_init(&pduSessions, sizeof(rrc_pdu_session_param_t));
  seq_arr_init(&drbs, sizeof(drb_t));

  const int session_id = 70;
  pdusession_t in = {0};
  in.pdusession_id = session_id;
  add_pduSession(&pduSessions, &in);

  nr_pdcp_configuration_t pdcp = {.drb.discard_timer = 100, .drb.sn_size = 18, .drb.t_reordering = 50};
  drb_t *added = nr_rrc_add_drb(&drbs, session_id, &pdcp);
  AssertFatal(added, "add_rrc_drb failed");

  drb_t *found = get_drb(&drbs, 1);
  AssertFatal(found && found == added, "get_drb failed");

  seq_arr_free(&drbs, free_drb);
}

static void test_get_drb(void)
{
  seq_arr_t pduSessions = {0};
  seq_arr_t drbs = {0};
  seq_arr_init(&pduSessions, sizeof(rrc_pdu_session_param_t));
  seq_arr_init(&drbs, sizeof(drb_t));

  const int session_id = 3;
  pdusession_t in = {0};
  in.pdusession_id = session_id;
  add_pduSession(&pduSessions, &in);

  nr_pdcp_configuration_t pdcp = {.drb.discard_timer = 100, .drb.sn_size = 18, .drb.t_reordering = 50};
  drb_t *added = nr_rrc_add_drb(&drbs, session_id, &pdcp);
  AssertFatal(added, "add_rrc_drb failed");

  drb_t *got = get_drb(&drbs, 1);
  AssertFatal(got && got->drb_id == 1, "get_drb failed");

  seq_arr_free(&drbs, free_drb);
}

static void test_find_drb(void)
{
  seq_arr_t pduSessions = {0};
  seq_arr_t drbs = {0};
  seq_arr_init(&pduSessions, sizeof(rrc_pdu_session_param_t));
  seq_arr_init(&drbs, sizeof(drb_t));

  const int session_id = 99;
  pdusession_t in = {0};
  in.pdusession_id = session_id;
  add_pduSession(&pduSessions, &in);

  nr_pdcp_configuration_t pdcp = {.drb.discard_timer = 100, .drb.sn_size = 18, .drb.t_reordering = 50};
  drb_t *added = nr_rrc_add_drb(&drbs, session_id, &pdcp);
  AssertFatal(added && added->pdusession_id == session_id, "add_rrc_drb failed");

  drb_t *found = get_drb(&drbs, 1);
  AssertFatal(found && found->drb_id == 1, "get_drb failed");

  seq_arr_free(&drbs, free_drb);
}

int main()
{
  // PDU Session
  test_add_and_find_pduSession();
  test_duplicate_add_pduSession();
  test_find_pduSession_from_drbId();
  // DRB
  test_add_rrc_drb();
  test_get_drb();
  test_find_drb();
  return 0;
}