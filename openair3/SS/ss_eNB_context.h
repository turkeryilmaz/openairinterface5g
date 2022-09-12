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
#ifndef _SS_ENB_CONTEXT_
#define _SS_ENB_CONTEXT_

#include "SidlCommon.h"

#if 0
#define SYS_L LOG_SYS(evId, SSConfig.sfn, SSConfig.sf, __VA_ARGS__) 
#define SRB_L LOG_SRB(evId, SSConfig.sfn, SSConfig.sf, __VA_ARGS__) 
#define DRB_L LOG_DRB(evId, SSConfig.sfn, SSConfig.sf, __VA_ARGS__)
#define VNG_L LOG_VNG(evId, SSConfig.sfn, SSConfig.sf, __VA_ARGS__)
#endif

typedef enum SSEvents_E {
	SS_PORTMAN_INVLALID_MSG = 1,
	SS_PORTMAN_DUMMY_EVT,

	SS_SYS_DUMMY_EVT,
	SS_SYS_INVALID_MSG,

	SS_SRB_DUMMY_EVT,
	SS_SRB_INVALID_MSG,

	SS_DRB_DUMMY_EVT,
	SS_DRB_INVALID_MSG,

	SS_VNG_DUMMY_EVT,
	SS_VNG_INVALID_MSG
} SSEventsE;

/** Structure Containing Cell-specific values only */
typedef struct SSCellConfigContext_s{
  int State;	
  uint16_t dl_earfcn;
  uint16_t ul_earfcn;
  uint32_t dl_freq;
  uint32_t ul_freq;
  uint16_t curr_attn;
  uint16_t cellId;
  int16_t maxRefPower;
  //TO DO: Need to remove one of the following cellId
  EUTRA_CellId_Type ttcn_cell_id;
  uint16_t eutra_cellId;
  uint16_t ss_rnti_g;
} SSCellConfigContext_t;

typedef struct SSConfigContext_s {
  /** List of Cells */
  SSCellConfigContext_t SSCell_list[8];	
  /** Timing info */
  uint16_t sfn;
  uint8_t  sf;

  bool vtp_enabled;
  ss_set_timinfo_t vtinfo;
  hash_table_t   *vt_timer_table   ; // key is SFN_SF

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

#endif /* _SS_ENB_CONTEXT_ */
