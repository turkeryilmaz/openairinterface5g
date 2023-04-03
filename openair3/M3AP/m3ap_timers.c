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

#include "m3ap_timers.h"
#include "assertions.h"
#include "PHY/defs_common.h"         /* TODO: try to not include this */
#include "m3ap_messages_types.h"
#include "m3ap_MCE_defs.h"
#include "m3ap_ids.h"
#include "m3ap_MCE_management_procedures.h"
//#include "m3ap_eNB_generate_messages.h"

void m3ap_timers_init(m3ap_timers_t *t, int t_reloc_prep, int tm3_reloc_overall)
{
  t->tti               = 0;
  t->t_reloc_prep      = t_reloc_prep;
  t->tm3_reloc_overall = tm3_reloc_overall;
}

void m3ap_check_timers(instance_t instance)
{
}

uint64_t m3ap_timer_get_tti(m3ap_timers_t *t)
{
  return t->tti;
}
