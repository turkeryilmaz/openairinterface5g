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

#include "SidlCompiler.h"
#include "SidlASN1.h"
#include "SidlASN1Base.h"
#include "SidlBase.h"
#include "SidlCommon.h"
#include "SidlCommonBase.h"
#include "SidlNrASN1.h"
#include "SidlNrCommon.h"
#include "SidlParts.h"
#include "SidlVals.h"
#include "TtcnCommon.h"
#include "SidlCommon_BcchConfig.h"
#include "SidlCommon_Bcch_BRConfig.h"
#include "SidlCommon_CcchDcchDtchConfig.h"
#include "SidlCommon_InitialCellPower.h"
#include "SidlCommon_PhysicalLayerConfigDL.h"
#include "SidlCommon_PhysicalLayerConfigUL.h"
#include "SidlCommon_RachProcedureConfig.h"
#include "SidlCommon_SciSchedulingConfig.h"
#include "SidlCommon_ServingCellConfig.h"
#include "SidlCommon_StaticCellInfo.h"
#include "SidlCommon_NR_BcchConfig_Type.h"
#include "SidlCommon_NR_CellConfigCommon_Type.h"
#include "SidlCommon_NR_CellConfigPhysicalLayer_Type.h"
#include "SidlCommon_NR_DcchDtchConfig_Type.h"
#include "SidlCommon_NR_PcchConfig_Type.h"
#include "SidlCommon_NR_RachProcedureConfig_Type.h"
#include "SidlCommon_NR_SS_StaticCellResourceConfig_Type.h"
#include "SidlCommon_NR_ServingCellConfig_Type.h"
#include "SidlCommon_CellConfigRequest.h"
#include "SidlCommon_NR_CellConfigRequest.h"

SIDL_BEGIN_C_INTERFACE

enum NR_RRC_MSG_Request_Type_Sel {
	NR_RRC_MSG_Request_Type_UNBOUND_VALUE = 0,
	NR_RRC_MSG_Request_Type_Ccch = 1,
	NR_RRC_MSG_Request_Type_Dcch = 2,
};

struct uint8_t_NR_RRC_MSG_Request_Type_Ccch_Dynamic {
	size_t d;
	uint8_t* v;
};

struct uint8_t_NR_RRC_MSG_Request_Type_Dcch_Dynamic {
	size_t d;
	uint8_t* v;
};

union NR_RRC_MSG_Request_Type_Value {
	struct uint8_t_NR_RRC_MSG_Request_Type_Ccch_Dynamic Ccch;
	struct uint8_t_NR_RRC_MSG_Request_Type_Dcch_Dynamic Dcch;
};

struct NR_RRC_MSG_Request_Type {
	enum NR_RRC_MSG_Request_Type_Sel d;
	union NR_RRC_MSG_Request_Type_Value v;
};

enum NR_RRC_MSG_Indication_Type_Sel {
	NR_RRC_MSG_Indication_Type_UNBOUND_VALUE = 0,
	NR_RRC_MSG_Indication_Type_Ccch = 1,
	NR_RRC_MSG_Indication_Type_Ccch1 = 2,
	NR_RRC_MSG_Indication_Type_Dcch = 3,
};

struct uint8_t_NR_RRC_MSG_Indication_Type_Ccch_Dynamic {
	size_t d;
	uint8_t* v;
};

struct uint8_t_NR_RRC_MSG_Indication_Type_Ccch1_Dynamic {
	size_t d;
	uint8_t* v;
};

struct uint8_t_NR_RRC_MSG_Indication_Type_Dcch_Dynamic {
	size_t d;
	uint8_t* v;
};

union NR_RRC_MSG_Indication_Type_Value {
	struct uint8_t_NR_RRC_MSG_Indication_Type_Ccch_Dynamic Ccch;
	struct uint8_t_NR_RRC_MSG_Indication_Type_Ccch1_Dynamic Ccch1;
	struct uint8_t_NR_RRC_MSG_Indication_Type_Dcch_Dynamic Dcch;
};

struct NR_RRC_MSG_Indication_Type {
	enum NR_RRC_MSG_Indication_Type_Sel d;
	union NR_RRC_MSG_Indication_Type_Value v;
};

struct NR_RRC_PDU_REQ {
	struct NR_ReqAspCommonPart_Type Common;
	struct NR_RRC_MSG_Request_Type RrcPdu;
};

struct NR_RRC_PDU_IND {
	struct NR_IndAspCommonPart_Type Common;
	struct NR_RRC_MSG_Indication_Type RrcPdu;
};

SIDL_END_C_INTERFACE
