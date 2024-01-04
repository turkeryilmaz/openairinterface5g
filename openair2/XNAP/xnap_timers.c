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

#include "xnap_timers.h"
#include "assertions.h"
#include "PHY/defs_common.h"         /* TODO: try to not include this */
#include "xnap_messages_types.h"
#include "xnap_gNB_defs.h"
#include "xnap_ids.h"
#include "xnap_gNB_management_procedures.h"
#include "xnap_gNB_generate_messages.h"

void xnap_timers_init(xnap_timers_t *t,
    int t_reloc_prep,
    int tx2_reloc_overall,
    int t_dc_prep,
    int t_dc_overall)
{
  t->tti               = 0;
  t->t_reloc_prep      = t_reloc_prep;
  t->tx2_reloc_overall = tx2_reloc_overall;
  t->t_dc_prep         = t_dc_prep;
  t->t_dc_overall      = t_dc_overall;
}
/*
void xnap_check_timers(instance_t instance)
{
  xnap_gNB_instance_t          *instance_p;
  xnap_timers_t                *t;
  xnap_id_manager              *m;
  int                          i;
  xnap_handover_cancel_cause_t cause;
  void                         *target;
  MessageDef                   *msg;
  int                          xnap_ongoing;

  instance_p = xnap_gNB_get_instance(instance);
  DevAssert(instance_p != NULL);

  t = &instance_p->timers;
  m = &instance_p->id_manager;

  // increment subframe count 
  t->tti++;

  xnap_ongoing = 0;

  for (i = 0; i < XNAP_MAX_IDS; i++) {
    if (m->ids[i].rnti == -1) continue;

    if (m->ids[i].state == XNAPID_STATE_SOURCE_PREPARE ||
        m->ids[i].state == XNAPID_STATE_SOURCE_OVERALL)
      xnap_ongoing++;

    if (m->ids[i].state == XNAPID_STATE_SOURCE_PREPARE &&
        t->tti > m->ids[i].t_reloc_prep_start + t->t_reloc_prep) {
      LOG_I(XNAP, "XNAP timeout reloc prep\n");
      // t_reloc_prep timed out 
      cause = XNAP_T_RELOC_PREP_TIMEOUT;
      goto xnap_handover_timeout;
    }

    if (m->ids[i].state == XNAPID_STATE_SOURCE_OVERALL &&
        t->tti > m->ids[i].tx2_reloc_overall_start + t->tx2_reloc_overall) {
      LOG_I(XNAP, "XNAP timeout reloc overall\n");
      // tx2_reloc_overall timed out 
      cause = XNAP_TX2_RELOC_OVERALL_TIMEOUT;
      goto xnap_handover_timeout;
    }

    if (m->ids[i].state == XNAPID_STATE_NSA_GNB_PREPARE &&
        t->tti > m->ids[i].t_dc_prep_start + t->t_dc_prep) {
      int id_source;
      int id_target;

      LOG_I(XNAP, "XNAP timeout DC prep\n");
      // t_dc_prep timed out 
      target = xnap_id_get_target(m, i);
      id_source = xnap_id_get_id_source(m, i);
      id_target = xnap_id_get_id_target(m, i);
      xnap_gNB_generate_ENDC_xnap_SgNB_release_request(instance_p, target,
                                                     id_source, id_target,
                                                     XNAP_CAUSE_T_DC_PREP_TIMEOUT);

      // inform RRC of timeout 
      msg = itti_alloc_new_message(TASK_XNAP, 0, XNAP_ENDC_DC_PREP_TIMEOUT);
      XNAP_ENDC_DC_PREP_TIMEOUT(msg).rnti  = xnap_id_get_rnti(m, i);
      itti_send_msg_to_task(TASK_RRC_GNB, instance_p->instance, msg);

      // remove UE from XNAP 
      xnap_release_id(m, i);

      continue;
    }

    if (m->ids[i].state == XNAPID_STATE_NSA_GNB_OVERALL &&
        t->tti > m->ids[i].t_dc_overall_start + t->t_dc_overall) {
      int id_source;
      int id_target;

      LOG_I(XNAP, "XNAP timeout DC overall\n");
      // t_dc_overall timed out 
      target = xnap_id_get_target(m, i);
      id_source = xnap_id_get_id_source(m, i);
      id_target = xnap_id_get_id_target(m, i);
      xnap_gNB_generate_ENDC_xnap_SgNB_release_required(instance_p, target,
              id_source, id_target, XNAP_CAUSE_T_DC_OVERALL_TIMEOUT);

      // inform RRC of timeout 
      msg = itti_alloc_new_message(TASK_XNAP, 0, XNAP_ENDC_DC_OVERALL_TIMEOUT);
      XNAP_ENDC_DC_OVERALL_TIMEOUT(msg).rnti  = xnap_id_get_rnti(m, i);
      itti_send_msg_to_task(TASK_RRC_GNB, instance_p->instance, msg);

      // remove UE from X2AP 
      xnap_release_id(m, i);

      continue;
    }

    // no timeout -> check next UE 
    continue;

    xnap_handover_timeout:
    // inform target about timeout 
    target = xnap_id_get_target(m, i);
    xnap_gNB_generate_xnap_handover_cancel(instance_p, target, i, cause);

    // inform RRC of cancellation 
    msg = itti_alloc_new_message(TASK_XNAP, 0, XNAP_HANDOVER_CANCEL);
    XNAP_HANDOVER_CANCEL(msg).rnti  = xnap_id_get_rnti(m, i);
    XNAP_HANDOVER_CANCEL(msg).cause = cause;
    itti_send_msg_to_task(TASK_RRC_GNB, instance_p->instance, msg);

    // remove UE from XNAP 
    xnap_release_id(m, i);
  }

  if (xnap_ongoing && t->tti % 1000 == 0)
    LOG_I(XNAP, "XNAP has %d process ongoing\n", xnap_ongoing);
}

uint64_t xnap_timer_get_tti(xnap_timers_t *t)
{
  return t->tti;
}  **/
