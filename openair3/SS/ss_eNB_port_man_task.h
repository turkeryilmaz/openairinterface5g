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

#ifndef SS_ENB_TASK_PORT_MAN_H_
#define SS_ENB_TASK_PORT_MAN_H_
#include "acpSys.h"

void  ss_eNB_port_man_init(void);
void *ss_eNB_port_man_eNB_task(void *arg);
void *ss_eNB_port_man_acp_task(void *arg);
bool ss_eNB_port_man_handle_enquiryTiming(struct SYSTEM_CTRL_REQ *sys_req);

#endif /* SS_ENB_TASK_PORT_MAN_H_ */
