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
#ifndef _SS_GNB_CONTEXT_
#define _SS_GNB_CONTEXT_

#include "SidlCommon.h"

#include "ss_messages_types.h"
#include "hashtable.h"

typedef struct SS_Cell_Context_s {
  int State;	
  uint16_t dl_earfcn;
  uint16_t ul_earfcn;
  uint16_t cellId;
  int16_t maxRefPower;
  //TO DO: Need to remove one of the following cellId
  uint16_t nr_cellId;
  uint16_t PhysicalCellId;
  uint16_t ss_rnti_g;
  bool cell_configured_flag;
  long absoluteFrequencyPointA;
  /** TODO: To add more */
} SS_Cell_Context_t;

typedef struct SSConfigContext_s {
  /** List of Cells */
  SS_Cell_Context_t SSCell_list[8];
 
 
 
  uint8_t vtp_enabled;
  ss_set_timinfo_t vtinfo;
  hash_table_t *vt_timer_table   ; // key is SFN_SLOT
  /** Timing info */
  uint8_t mu;	 	/*subcarrierSpace*/
  uint16_t hsfn;
  uint16_t sfn;
  uint32_t  slot;
  /** TODO: To add more */
} SSConfigContext_t;

typedef enum {
  SS_STATE_NOT_CONFIGURED = 0,
  SS_STATE_CELL_CONFIGURED,
  SS_STATE_CELL_ACTIVE,
  SS_STATE_AS_SECURITY_ACTIVE,
  SS_STATE_AS_RBS_ACTIVE,
  SS_STATE_CELL_BROADCASTING,
  SS_STATE_MAX_STATE
} SS_STATE_t;
#endif /* _SS_GNB_CONTEXT_ */
