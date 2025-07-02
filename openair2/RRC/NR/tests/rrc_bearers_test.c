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

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  printf("detected error at %s:%d:%s: %s\n", file, line, function, s);
  abort();
}

int nr_rlc_get_available_tx_space(int module_id, int rnti, int drb_id) { return 0; }
softmodem_params_t *get_softmodem_params(void) { return NULL; }
configmodule_interface_t *uniqCfg = NULL;

static void test_add_and_find_pduSession(void)
{
  gNB_RRC_UE_t *ue = calloc_or_fail(1, sizeof(*ue));
  ue->rrc_ue_id = 0x4321;

  pdusession_t input = {0};
  input.pdusession_id = 5;
  input.n3_incoming.teid = 0x1001;

  pdusession_t *session = add_pduSession(&ue->pduSessions_to_addmod, ue->rrc_ue_id, &input);
  AssertFatal(session != NULL, "Could not add PDU Session\n");
  AssertFatal(session->pdusession_id == input.pdusession_id, "PDU Session ID mismatch in added PDU Session\n");
  AssertFatal(session->n3_incoming.teid == input.n3_incoming.teid, "teid mismatch in added PDU Session\n");

  pdusession_t *found = find_pduSession(ue->pduSessions_to_addmod, input.pdusession_id);
  AssertFatal(found != NULL, "Could not find PDU Session\n");
  AssertFatal(found == session, "Found PDU Session mismatch\n");

  SEQ_ARR_CLEANUP_AND_FREE(ue->pduSessions_to_addmod, NULL);
  free(ue);
}

static void test_duplicate_add_pduSession(void)
{
  gNB_RRC_UE_t ue = {0};
  ue.rrc_ue_id = 0x4444;

  pdusession_t input1 = {0};
  input1.pdusession_id = 3;
  input1.n3_incoming.teid = 2002;

  pdusession_t input2 = {0};
  input2.pdusession_id = 3;
  input2.n3_incoming.teid = 9999;

  pdusession_t *s1 = add_pduSession(&ue.pduSessions_to_addmod, ue.rrc_ue_id, &input1);
  AssertFatal(s1 != NULL, "First add_pduSession failed\\n");

  pdusession_t *s2 = add_pduSession(&ue.pduSessions_to_addmod, ue.rrc_ue_id, &input2);
  AssertFatal(s2 == s1, "Duplicate add_pduSession returned different pointer\\n");
  AssertFatal(s2->n3_incoming.teid == input1.n3_incoming.teid, "Original TEID should be retained\\n");

  SEQ_ARR_CLEANUP_AND_FREE(ue.pduSessions_to_addmod, NULL);

}

static void test_rm_pduSession(void)
{
  gNB_RRC_UE_t ue = {0};
  ue.rrc_ue_id = 0x1234;

  pdusession_t input = {0};
  input.pdusession_id = 4;
  input.n3_incoming.teid = 777;

  pdusession_t *s = add_pduSession(&ue.pduSessions_to_addmod, ue.rrc_ue_id, &input);
  AssertFatal(s != NULL, "add_pduSession failed\\n");

  rm_pduSession(ue.pduSessions_to_addmod, input.pdusession_id);

  AssertFatal(find_pduSession(ue.pduSessions_to_addmod, input.pdusession_id) == NULL, "PDU Session was not removed\\n");

  SEQ_ARR_CLEANUP_AND_FREE(ue.pduSessions_to_addmod, NULL);

}

static void test_update_pduSession(void)
{
  gNB_RRC_UE_t ue = {0};

  pdusession_t in = {.pdusession_id = 1, .n3_incoming.teid = 111};
  add_pduSession(&ue.pduSessions_to_addmod, ue.rrc_ue_id, &in);

  pdusession_t *orig = find_pduSession(ue.pduSessions_to_addmod, 1);
  AssertFatal(orig != NULL, "Original PDU session not found before update");
  AssertFatal(orig->n3_incoming.teid == 111, "Original TEID mismatch");

  pdusession_t updated = {.pdusession_id = 1, .n3_incoming.teid = 999};
  bool ok = update_pduSession(&ue.pduSessions_to_addmod, &updated);
  AssertFatal(ok, "update_pduSession failed");

  pdusession_t *found = find_pduSession(ue.pduSessions_to_addmod, 1);
  AssertFatal(found && found->n3_incoming.teid == 999, "TEID not updated");

  SEQ_ARR_CLEANUP_AND_FREE(ue.pduSessions_to_addmod, NULL);
}

static void test_add_pduSession_to_release(void)
{
  gNB_RRC_UE_t ue = {0};

  rrc_pdusession_release_t rel = {.pdusession_id = 20};
  rrc_pdusession_release_t *r = add_pduSession_to_release(&ue.pduSessions_to_release, ue.rrc_ue_id, rel);
  AssertFatal(r != NULL && r->pdusession_id == 20, "add_pduSession_to_release failed");

  SEQ_ARR_CLEANUP_AND_FREE(ue.pduSessions_to_release, NULL);

}

static void test_find_pduSession_from_drbId(void)
{
  gNB_RRC_UE_t ue = {0};

  drb_t drb = {0};
  drb.drb_id = 3;
  drb.pdusession_id = 66;

  add_rrc_drb(&ue.drbs, drb);
  AssertFatal(ue.drbs != NULL, "Failed to add DRB");

  pdusession_t in = {0};
  in.pdusession_id = 66;

  add_pduSession(&ue.pduSessions_to_addmod, ue.rrc_ue_id, &in);

  pdusession_t *found = find_pduSession_from_drbId(&ue, ue.pduSessions_to_addmod, 3);
  AssertFatal(found && found->pdusession_id == 66, "find_pduSession_from_drbId failed");

  SEQ_ARR_CLEANUP_AND_FREE(ue.pduSessions_to_addmod, NULL);
  SEQ_ARR_CLEANUP_AND_FREE(ue.drbs, NULL);

}

// ---------------- DRB TESTS ----------------

static void test_add_rrc_drb(void)
{
  seq_arr_t *drbs = NULL;

  drb_t in = {0};
  in.drb_id = 7;
  in.pdusession_id = 70;

  drb_t *added = add_rrc_drb(&drbs, in);
  AssertFatal(added && added->drb_id == 7, "add_rrc_drb failed");

  drb_t *found = find_drb(drbs, 70);
  AssertFatal(found && found == added, "find_drb failed");

  SEQ_ARR_CLEANUP_AND_FREE(drbs, NULL);
}

static void test_get_drb(void)
{
  seq_arr_t *drbs = NULL;
  drb_t in = {.drb_id = 5};
  drb_t *added = add_rrc_drb(&drbs, in);
  AssertFatal(added && added->drb_id == 5, "add_rrc_drb failed");

  drb_t *got = get_drb(drbs, 5);
  AssertFatal(got && got->drb_id == 5, "get_drb failed");

  SEQ_ARR_CLEANUP_AND_FREE(drbs, NULL);
}

static void test_find_drb(void)
{
  seq_arr_t *drbs = NULL;
  drb_t in = {.drb_id = 2, .pdusession_id = 88};
  drb_t *added = add_rrc_drb(&drbs, in);
  AssertFatal(added && added->pdusession_id == 88, "add_rrc_drb failed");

  drb_t *found = find_drb(drbs, 88);
  AssertFatal(found && found->drb_id == 2, "find_drb failed");

  SEQ_ARR_CLEANUP_AND_FREE(drbs, NULL);
}

static void test_remove_drbs_by_pdu_session(void)
{
  seq_arr_t *drbs = NULL;
  drb_t in1 = {.drb_id = 1, .pdusession_id = 10};
  drb_t in2 = {.drb_id = 2, .pdusession_id = 10};
  drb_t in3 = {.drb_id = 3, .pdusession_id = 20};
  drb_t *add1 = add_rrc_drb(&drbs, in1);
  AssertFatal(add1 && add1->pdusession_id == 10, "add_rrc_drb failed");
  drb_t *add2 = add_rrc_drb(&drbs, in2);
  AssertFatal(add2 && add2->drb_id == 2, "add_rrc_drb failed");
  drb_t *add3 = add_rrc_drb(&drbs, in3);
  AssertFatal(add3 && add3->drb_id == 3, "add_rrc_drb failed");

  remove_drbs_by_pdu_session(&drbs, 20); // Remove DRB 3
  AssertFatal(seq_arr_size(drbs) == 2, "unexpected size of drbs seq_arr");

  remove_drbs_by_pdu_session(&drbs, 10); // Remove all remaining DRBs
  AssertFatal(drbs == NULL, "remove_drbs_by_pdu_session failed");

  SEQ_ARR_CLEANUP_AND_FREE(drbs, NULL);
}

int main()
{
  // PDU Session
  test_add_and_find_pduSession();
  test_duplicate_add_pduSession();
  test_rm_pduSession();
  test_update_pduSession();
  test_add_pduSession_to_release();
  test_find_pduSession_from_drbId();
  // DRB
  test_add_rrc_drb();
  test_get_drb();
  test_find_drb();
  test_remove_drbs_by_pdu_session();
  return 0;
}