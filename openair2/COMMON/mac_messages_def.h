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
 * mac_messages_def.h
 *
 *  Created on: Oct 24, 2013
 *      Author: L. winckel and Navid Nikaein
 */

//-------------------------------------------------------------------------------------------//
// Messages between RRC and MAC layers
MESSAGE_DEF(RRC_MAC_IN_SYNC_IND, MESSAGE_PRIORITY_MED_PLUS, RrcMacInSyncInd)
MESSAGE_DEF(RRC_MAC_OUT_OF_SYNC_IND, MESSAGE_PRIORITY_MED_PLUS, RrcMacOutOfSyncInd)

MESSAGE_DEF(RRC_MAC_BCCH_DATA_REQ, MESSAGE_PRIORITY_MED_PLUS, RrcMacBcchDataReq)
MESSAGE_DEF(RRC_MAC_BCCH_DATA_IND, MESSAGE_PRIORITY_MED_PLUS, RrcMacBcchDataInd)

MESSAGE_DEF(RRC_MAC_BCCH_MBMS_DATA_REQ, MESSAGE_PRIORITY_MED_PLUS, RrcMacBcchMbmsDataReq)
MESSAGE_DEF(RRC_MAC_BCCH_MBMS_DATA_IND, MESSAGE_PRIORITY_MED_PLUS, RrcMacBcchMbmsDataInd)

MESSAGE_DEF(RRC_MAC_CCCH_DATA_REQ, MESSAGE_PRIORITY_MED_PLUS, RrcMacCcchDataReq)
MESSAGE_DEF(RRC_MAC_CCCH_DATA_CNF, MESSAGE_PRIORITY_MED_PLUS, RrcMacCcchDataCnf)
MESSAGE_DEF(RRC_MAC_CCCH_DATA_IND, MESSAGE_PRIORITY_MED_PLUS, RrcMacCcchDataInd)

MESSAGE_DEF(RRC_MAC_MCCH_DATA_REQ, MESSAGE_PRIORITY_MED_PLUS, RrcMacMcchDataReq)
MESSAGE_DEF(RRC_MAC_MCCH_DATA_IND, MESSAGE_PRIORITY_MED_PLUS, RrcMacMcchDataInd)

MESSAGE_DEF(RRC_MAC_PCCH_DATA_REQ, MESSAGE_PRIORITY_MED_PLUS, RrcMacPcchDataReq)

/* RRC configures DRX context (MAC timers) of a UE */
MESSAGE_DEF(RRC_MAC_DRX_CONFIG_REQ, MESSAGE_PRIORITY_MED, rrc_mac_drx_config_req_t)

// gNB
MESSAGE_DEF(NR_RRC_MAC_CCCH_DATA_IND, MESSAGE_PRIORITY_MED_PLUS, NRRrcMacCcchDataInd)
MESSAGE_DEF(NR_RRC_MAC_BCCH_DATA_IND, MESSAGE_PRIORITY_MED_PLUS, NRRrcMacBcchDataInd)
