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

SIDL_BEGIN_C_INTERFACE

struct AcpHandshakeVersion {
	char release[32];
	char checksum[64];
};

struct AcpHandshakeReqControlInfo {
	bool productMode;
	char* desc;
};

enum AcpHandshakeCnfResult {
	ACP_HANDSHAKE_OK = 0,
	ACP_HANDSHAKE_GENERAL_ERROR = 1,
	ACP_HANDSHAKE_VERSION_ERROR = 2,
	ACP_HANDSHAKE_SERVICE_ERROR = 3,
};

typedef enum AcpHandshakeCnfResult AcpHandshakeCnfResult;

struct AcpHandshakeReqCommon {
	struct AcpHandshakeVersion version;
	struct AcpHandshakeReqControlInfo control;
};

struct AcpHandshakeCnfCommon {
	struct AcpHandshakeVersion version;
	AcpHandshakeCnfResult result;
};

struct AcpHandshakeService {
	uint32_t id;
	char* name;
};

struct AcpHandshakeService_AcpHandshakeReqService_services_Dynamic {
	size_t d;
	struct AcpHandshakeService* v;
};

struct AcpHandshakeReqService {
	struct AcpHandshakeService_AcpHandshakeReqService_services_Dynamic services;
};

struct AcpHandshakeService_AcpHandshakeCnfService_services_Dynamic {
	size_t d;
	struct AcpHandshakeService* v;
};

struct AcpHandshakeCnfService {
	struct AcpHandshakeService_AcpHandshakeCnfService_services_Dynamic services;
};

struct AcpHandshakeReq {
	struct AcpHandshakeReqCommon common;
	struct AcpHandshakeReqService request;
};

struct AcpHandshakeCnf {
	struct AcpHandshakeCnfCommon common;
	struct AcpHandshakeCnfService confirm;
};

SIDL_END_C_INTERFACE
