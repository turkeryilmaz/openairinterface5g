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

#include "SIDL_EUTRA_SYSTEM_PORT.h"

SIDL_BEGIN_C_INTERFACE

const char* adbgUtilsSysPdcpCountGetReq_TypeToStr(int select);

const char* adbgUtilsSysPDCP_CountReq_TypeToStr(int select);

const char* adbgUtilsSysPDCP_TestModeInfo_TypeToStr(int select);

const char* adbgUtilsSysPDCP_TestModeConfig_TypeToStr(int select);

const char* adbgUtilsSysPDCP_Config_TypeToStr(int select);

const char* adbgUtilsSysPDCP_RBConfig_TypeToStr(int select);

const char* adbgUtilsSysPDCP_Configuration_TypeToStr(int select);

const char* adbgUtilsSysUL_AM_RLC_TypeToStr(int select);

const char* adbgUtilsSysDL_AM_RLC_TypeToStr(int select);

const char* adbgUtilsSysUL_UM_RLC_TypeToStr(int select);

const char* adbgUtilsSysDL_UM_RLC_TypeToStr(int select);

const char* adbgUtilsSysRLC_RbConfig_TypeToStr(int select);

const char* adbgUtilsSysRLC_TestModeInfo_TypeToStr(int select);

const char* adbgUtilsSysRLC_TestModeConfig_TypeToStr(int select);

const char* adbgUtilsSysMAC_Test_DLLogChID_TypeToStr(int select);

const char* adbgUtilsSysMAC_TestModeConfig_TypeToStr(int select);

const char* adbgUtilsSysRadioBearerConfig_TypeToStr(int select);

const char* adbgUtilsSysPDCP_ActTime_TypeToStr(int select);

const char* adbgUtilsSysAS_Security_TypeToStr(int select);

const char* adbgUtilsSysPDCP_HandoverControlReq_TypeToStr(int select);

const char* adbgUtilsSysSystemRequest_TypeToStr(int select);

const char* adbgUtilsSysSystemConfirm_TypeToStr(int select);

SIDL_END_C_INTERFACE
