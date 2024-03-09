/*
 * Copyright 2022 Sequans Communications.
 *
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#pragma once

#include "SIDL_NR_SYSTEM_PORT.h"

SIDL_BEGIN_C_INTERFACE

const char* adbgUtilsNrSysNR_PdcpCountGetReq_TypeToStr(int select);

const char* adbgUtilsNrSysNR_PDCP_CountReq_TypeToStr(int select);

const char* adbgUtilsNrSysSdapConfigInfo_TypeToStr(int select);

const char* adbgUtilsNrSysSDAP_Configuration_TypeToStr(int select);

const char* adbgUtilsNrSysNR_PDCP_DRB_HeaderCompression_TypeToStr(int select);

const char* adbgUtilsNrSysNR_PDCP_RB_Config_Parameters_TypeToStr(int select);

const char* adbgUtilsNrSysNR_PDCP_RbConfig_TypeToStr(int select);

const char* adbgUtilsNrSysNR_PDCP_Configuration_TypeToStr(int select);

const char* adbgUtilsNrSysNR_ASN1_UL_AM_RLC_TypeToStr(int select);

const char* adbgUtilsNrSysNR_ASN1_DL_AM_RLC_TypeToStr(int select);

const char* adbgUtilsNrSysNR_ASN1_UL_UM_RLC_TypeToStr(int select);

const char* adbgUtilsNrSysNR_ASN1_DL_UM_RLC_TypeToStr(int select);

const char* adbgUtilsNrSysNR_RLC_RbConfig_TypeToStr(int select);

const char* adbgUtilsNrSysNR_RLC_TransparentModeToStr(int select);

const char* adbgUtilsNrSysNR_RLC_TestModeInfo_TypeToStr(int select);

const char* adbgUtilsNrSysNR_RLC_TestModeConfig_TypeToStr(int select);

const char* adbgUtilsNrSysNR_MAC_TestModeConfig_TypeToStr(int select);

const char* adbgUtilsNrSysNR_RlcBearerConfig_TypeToStr(int select);

const char* adbgUtilsNrSysNR_RadioBearerConfig_TypeToStr(int select);

const char* adbgUtilsNrSysNR_PDCP_ActTime_TypeToStr(int select);

const char* adbgUtilsNrSysNR_AS_Security_TypeToStr(int select);

const char* adbgUtilsNrSysNR_DciFormat_2_2_TpcBlock_TypeToStr(int select);

const char* adbgUtilsNrSysNR_DciFormat_2_3_TypeA_B_TypeToStr(int select);

const char* adbgUtilsNrSysNR_DCI_TriggerFormat_TypeToStr(int select);

const char* adbgUtilsNrSysNR_SystemRequest_TypeToStr(int select);

const char* adbgUtilsNrSysNR_SystemConfirm_TypeToStr(int select);

SIDL_END_C_INTERFACE
