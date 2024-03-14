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
 *
 * AUTHOR  : Vijay Chadachan
 * COMPANY : Firecell
 * EMAIL   : Vijay.chadachan@firecell.io
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>

#include "hashtable.h"
#include "intertask_interface.h"
#include "SidlCommon.h"

#ifndef SS_ENB_VT_TIMER_TASK_H_
#define SS_ENB_VT_TIMER_TASK_H_

void *ss_eNB_vt_timer_process_itti_msg(void *);
void *ss_eNB_vt_timer_task(void *arg);

int vt_timer_push_msg(struct TimingInfo_Type* at, task_id_t task_id,instance_t instance, MessageDef *msg_p);
void vt_add_sf(struct TimingInfo_Type* at, int offset);

typedef struct vt_timer_elm_s {
  //uint8_t msg_type;     ///MSG type
  task_id_t task_id;
  instance_t instance;
  void *msg; ///< Optional argument that will be passed when timer expires
} vt_timer_elm_t ;


#endif
