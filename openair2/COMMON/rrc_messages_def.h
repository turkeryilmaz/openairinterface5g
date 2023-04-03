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
 * rrc_messages_def.h
 *
 *  Created on: Oct 24, 2013
 *      Author: winckel
 */

//-------------------------------------------------------------------------------------------//
// Messages for RRC logging
MESSAGE_DEF(RRC_DL_BCCH, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)
MESSAGE_DEF(RRC_DL_CCCH, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)
MESSAGE_DEF(RRC_DL_DCCH, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)
MESSAGE_DEF(RRC_DL_MCCH, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)

MESSAGE_DEF(RRC_UE_EUTRA_CAPABILITY, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)

MESSAGE_DEF(RRC_UL_CCCH, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)
MESSAGE_DEF(RRC_UL_DCCH, MESSAGE_PRIORITY_MED_PLUS, IttiMsgText)
MESSAGE_DEF(RRC_STATE_IND, MESSAGE_PRIORITY_MED, RrcStateInd)

//-------------------------------------------------------------------------------------------//
// eNB: ENB_APP -> RRC messages
MESSAGE_DEF(RRC_CONFIGURATION_REQ, MESSAGE_PRIORITY_MED, RrcConfigurationReq)
MESSAGE_DEF(NBIOTRRC_CONFIGURATION_REQ, MESSAGE_PRIORITY_MED, NbIoTRrcConfigurationReq)
MESSAGE_DEF(NRRRC_CONFIGURATION_REQ, MESSAGE_PRIORITY_MED, gNB_RrcConfigurationReq)

// UE: NAS -> RRC messages
MESSAGE_DEF(NAS_KENB_REFRESH_REQ, MESSAGE_PRIORITY_MED, NasKenbRefreshReq)
MESSAGE_DEF(NAS_CELL_SELECTION_REQ, MESSAGE_PRIORITY_MED, NasCellSelectionReq)
MESSAGE_DEF(NAS_CONN_ESTABLI_REQ, MESSAGE_PRIORITY_MED, NasConnEstabliReq)
MESSAGE_DEF(NAS_UPLINK_DATA_REQ, MESSAGE_PRIORITY_MED, NasUlDataReq)
MESSAGE_DEF(NAS_DEREGISTRATION_REQ,     MESSAGE_PRIORITY_MED,       NasDeregistrationReq)

MESSAGE_DEF(NAS_RAB_ESTABLI_RSP, MESSAGE_PRIORITY_MED, NasRabEstRsp)

MESSAGE_DEF(NAS_OAI_TUN_NSA, MESSAGE_PRIORITY_MED, NasOaiTunNsa)

// UE: RRC -> NAS messages
MESSAGE_DEF(NAS_CELL_SELECTION_CNF, MESSAGE_PRIORITY_MED, NasCellSelectionCnf)
MESSAGE_DEF(NAS_CELL_SELECTION_IND, MESSAGE_PRIORITY_MED, NasCellSelectionInd)
MESSAGE_DEF(NAS_PAGING_IND, MESSAGE_PRIORITY_MED, NasPagingInd)
MESSAGE_DEF(NAS_CONN_ESTABLI_CNF, MESSAGE_PRIORITY_MED, NasConnEstabCnf)
MESSAGE_DEF(NAS_CONN_RELEASE_IND, MESSAGE_PRIORITY_MED, NasConnReleaseInd)
MESSAGE_DEF(NAS_UPLINK_DATA_CNF, MESSAGE_PRIORITY_MED, NasUlDataCnf)
MESSAGE_DEF(NAS_DOWNLINK_DATA_IND, MESSAGE_PRIORITY_MED, NasDlDataInd)

// eNB: realtime -> RRC messages
MESSAGE_DEF(RRC_SUBFRAME_PROCESS, MESSAGE_PRIORITY_MED, RrcSubframeProcess)

// eNB: RLC -> RRC messages
MESSAGE_DEF(RLC_SDU_INDICATION, MESSAGE_PRIORITY_MED, RlcSduIndication)
MESSAGE_DEF(NR_DU_RRC_DL_INDICATION, MESSAGE_PRIORITY_MED, NRDuDlReq_t)
