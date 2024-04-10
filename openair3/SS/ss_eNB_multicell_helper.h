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

#ifndef MULTICELL_HELPER_H

#define MULTICELL_HELPER_H
#include "common/ran_context.h"
#include "ss_eNB_context.h"

int get_cell_index(uint16_t, SS_Cell_Context_t[]);

int get_cell_index_pci(uint16_t physCellId, SS_Cell_Context_t SSCell_list[]);
void init_ss_context(SS_Cell_Context_t SSCell_list[]);
void init_cell_context(int cell_index, int enb_id,MessageDef *msg_p) ;


#endif
