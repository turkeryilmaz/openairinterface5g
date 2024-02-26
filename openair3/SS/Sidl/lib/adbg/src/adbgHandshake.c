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

#include "adbgHandshake.h"

static void _adbgHandshake__AcpHandshakeVersion(acpCtx_t _ctx, const struct AcpHandshakeVersion* p)
{
	adbgPrintLog(_ctx, "release := '");
	for (size_t i1 = 0; i1 < 32; i1++) {
		adbgPrintLog(_ctx, "%02X", p->release[i1]);
	}
	adbgPrintLog(_ctx, "'O");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "checksum := '");
	for (size_t i1 = 0; i1 < 64; i1++) {
		adbgPrintLog(_ctx, "%02X", p->checksum[i1]);
	}
	adbgPrintLog(_ctx, "'O");
}

static void _adbgHandshake__AcpHandshakeReqControlInfo(acpCtx_t _ctx, const struct AcpHandshakeReqControlInfo* p)
{
	adbgPrintLog(_ctx, "productMode := %s", (p->productMode ? "true" : "false"));
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "desc := \"%s\"", p->desc);
}

static void _adbgHandshake__AcpHandshakeReqCommon(acpCtx_t _ctx, const struct AcpHandshakeReqCommon* p)
{
	adbgPrintLog(_ctx, "version := { ");
	_adbgHandshake__AcpHandshakeVersion(_ctx, &p->version);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "control := { ");
	_adbgHandshake__AcpHandshakeReqControlInfo(_ctx, &p->control);
	adbgPrintLog(_ctx, " }");
}

static void _adbgHandshake__AcpHandshakeService(acpCtx_t _ctx, const struct AcpHandshakeService* p)
{
	adbgPrintLog(_ctx, "id := %u", (unsigned int)p->id);
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "name := \"%s\"", p->name);
}

static void _adbgHandshake__AcpHandshakeReqService(acpCtx_t _ctx, const struct AcpHandshakeReqService* p)
{
	adbgPrintLog(_ctx, "services := { ");
	for (size_t i1 = 0; i1 < p->services.d; i1++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgHandshake__AcpHandshakeService(_ctx, &p->services.v[i1]);
		adbgPrintLog(_ctx, " }");
		if (i1 != p->services.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgHandshake__AcpHandshakeReq(acpCtx_t _ctx, const struct AcpHandshakeReq* p)
{
	adbgPrintLog(_ctx, "common := { ");
	_adbgHandshake__AcpHandshakeReqCommon(_ctx, &p->common);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "request := { ");
	_adbgHandshake__AcpHandshakeReqService(_ctx, &p->request);
	adbgPrintLog(_ctx, " }");
}

void adbgHandshakeProcessLogIn(acpCtx_t _ctx, const struct AcpHandshakeReq* FromSS)
{
	adbgPrintLog(_ctx, "@HandshakeProcess In Args : { ");

	adbgPrintLog(_ctx, "FromSS := { ");
	_adbgHandshake__AcpHandshakeReq(_ctx, FromSS);
	adbgPrintLog(_ctx, " }");

	adbgPrintLog(_ctx, " }");
	adbgPrintFormatLog(_ctx);
}

static const char* adbgHandshake__AcpHandshakeCnfResult__ToString(AcpHandshakeCnfResult v)
{
	switch(v) {
		case ACP_HANDSHAKE_OK: return "ACP_HANDSHAKE_OK";
		case ACP_HANDSHAKE_GENERAL_ERROR: return "ACP_HANDSHAKE_GENERAL_ERROR";
		case ACP_HANDSHAKE_VERSION_ERROR: return "ACP_HANDSHAKE_VERSION_ERROR";
		case ACP_HANDSHAKE_SERVICE_ERROR: return "ACP_HANDSHAKE_SERVICE_ERROR";
		default: return "Unknown";
	}
}

static void _adbgHandshake__AcpHandshakeCnfCommon(acpCtx_t _ctx, const struct AcpHandshakeCnfCommon* p)
{
	adbgPrintLog(_ctx, "version := { ");
	_adbgHandshake__AcpHandshakeVersion(_ctx, &p->version);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "result := %s (%d)", adbgHandshake__AcpHandshakeCnfResult__ToString(p->result), (int)p->result);
}

static void _adbgHandshake__AcpHandshakeCnfService(acpCtx_t _ctx, const struct AcpHandshakeCnfService* p)
{
	adbgPrintLog(_ctx, "services := { ");
	for (size_t i1 = 0; i1 < p->services.d; i1++) {
		adbgPrintLog(_ctx, "{ ");
		_adbgHandshake__AcpHandshakeService(_ctx, &p->services.v[i1]);
		adbgPrintLog(_ctx, " }");
		if (i1 != p->services.d - 1) { adbgPrintLog(_ctx, ", "); }
	}
	adbgPrintLog(_ctx, " }");
}

static void _adbgHandshake__AcpHandshakeCnf(acpCtx_t _ctx, const struct AcpHandshakeCnf* p)
{
	adbgPrintLog(_ctx, "common := { ");
	_adbgHandshake__AcpHandshakeCnfCommon(_ctx, &p->common);
	adbgPrintLog(_ctx, " }");
	adbgPrintLog(_ctx, ", ");
	adbgPrintLog(_ctx, "confirm := { ");
	_adbgHandshake__AcpHandshakeCnfService(_ctx, &p->confirm);
	adbgPrintLog(_ctx, " }");
}

void adbgHandshakeProcessLogOut(acpCtx_t _ctx, const struct AcpHandshakeCnf* ToSS)
{
	adbgPrintLog(_ctx, "@HandshakeProcess Out Args : { ");

	adbgPrintLog(_ctx, "ToSS := { ");
	_adbgHandshake__AcpHandshakeCnf(_ctx, ToSS);
	adbgPrintLog(_ctx, " }");

	adbgPrintLog(_ctx, " }");
	adbgPrintFormatLog(_ctx);
}
