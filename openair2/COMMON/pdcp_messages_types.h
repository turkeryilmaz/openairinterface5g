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

/*
 * pdcp_messages_types.h
 *
 *  Created on: Oct 24, 2013
 *      Author: winckel and Navid Nikaein
 */

#ifndef PDCP_MESSAGES_TYPES_H_
#define PDCP_MESSAGES_TYPES_H_

//-------------------------------------------------------------------------------------------//
// Messages between RRC and PDCP layers
typedef struct RrcDcchDataReq_s {
  uint32_t frame;
  uint8_t enb_flag;
  rb_id_t rb_id;
  uint32_t muip;
  uint32_t confirmp;
  uint32_t sdu_size;
  uint8_t *sdu_p;
  uint8_t mode;
  uint16_t     rnti;
  uint8_t      module_id;
  uint8_t eNB_index;
} RrcDcchDataReq;

typedef struct RrcDcchDataInd_s {
  uint32_t frame;
  uint8_t dcch_index;
  uint32_t sdu_size;
  uint8_t *sdu_p;
  uint16_t     rnti;
  uint8_t      module_id;
  uint8_t      eNB_index; // LG: needed in UE
} RrcDcchDataInd;

typedef struct RrcDcchDataCopyInd_s {
  uint8_t dcch_index;
  uint32_t sdu_size;
  uint8_t *sdu_p;
  uint8_t      eNB_index;
} RrcDcchDataCopyInd;

typedef struct NRRrcDcchDataReq_s {
  uint32_t frame;
  uint8_t  gnb_flag;
  rb_id_t  rb_id;
  uint32_t muip;
  uint32_t confirmp;
  uint32_t sdu_size;
  uint8_t *sdu_p;
  uint8_t  mode;
  uint16_t rnti;
  uint8_t  module_id;
  uint8_t  gNB_index;
} NRRrcDcchDataReq;

typedef struct NRRrcDcchDataInd_s {
  uint32_t frame;
  uint8_t dcch_index;
  uint32_t sdu_size;
  uint8_t *sdu_p;
  uint16_t     rnti;
  uint8_t      module_id;
  uint8_t      gNB_index; // LG: needed in UE
} NRRrcDcchDataInd;

typedef struct RrcPcchDataReq_s {
  uint32_t     sdu_size;
  uint8_t      *sdu_p;
  uint8_t      mode;
  uint16_t     rnti;
  uint8_t      ue_index;
  uint8_t      CC_id;
} RrcPcchDataReq;
#define MESSAGE_DEF(iD, pRIO, sTRUCT)              \
  static inline sTRUCT *iD##_data(MessageDef *msg) \
  {                                                \
    return (sTRUCT *)msg->ittiMsg;                 \
  }
#include "openair2/COMMON/pdcp_messages_def.h"
#undef MESSAGE_DEF
#define MESSAGE_DEF(iD, pRIO, sTRUCT)                                                 \
  static inline MessageDef *iD##_alloc(task_id_t origintaskID, instance_t originINST) \
  {                                                                                   \
    return itti_alloc_sized(origintaskID, originINST, iD, sizeof(sTRUCT));            \
  }
#include "openair2/COMMON/pdcp_messages_def.h"
 #undef MESSAGE_DEF

#endif /* PDCP_MESSAGES_TYPES_H_ */
