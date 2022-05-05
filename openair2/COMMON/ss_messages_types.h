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

#ifndef SS_MESSAGES_TYPES_H_
#define SS_MESSAGES_TYPES_H_

#define SS_RRC_PDU_REQ(mSGpTR)                (mSGpTR)->ittiMsg.ss_rrc_pdu_req
#define SS_RRC_PDU_IND(mSGpTR)                (mSGpTR)->ittiMsg.ss_rrc_pdu_ind

#define SDU_SIZE                           (512)

/** SRB */
typedef struct ss_rrc_pdu_req_s {
  uint8_t   srb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
} ss_rrc_pdu_req_t;

typedef struct ss_rrc_pdu_ind_s {
  uint8_t   srb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
  frame_t     frame;         /*!< \brief  LTE frame number.*/
  sub_frame_t subframe;      /*!< \brief  LTE sub frame number.*/
} ss_rrc_pdu_ind_t;
#endif /* SS_MESSAGES_TYPES_H_ */
