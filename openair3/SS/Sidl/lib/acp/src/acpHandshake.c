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

#include "acpHandshake.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serHandshake.h"

void acpHandshakeProcessInitClt(acpCtx_t _ctx, struct AcpHandshakeReq** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serHandshakeProcessInitClt(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

int acpHandshakeProcessEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct AcpHandshakeReq* FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serHandshakeProcessEncClt(_buffer, *_size, &_lidx, FromSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_HandshakeProcess, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpHandshakeProcessDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct AcpHandshakeReq** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serHandshakeProcessDecSrv(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

void acpHandshakeProcessFree0Srv(struct AcpHandshakeReq* FromSS)
{
	serHandshakeProcessFree0Srv(FromSS);
}

void acpHandshakeProcessFreeSrv(struct AcpHandshakeReq* FromSS)
{
	serHandshakeProcessFreeSrv(FromSS);
}

void acpHandshakeProcessFree0CltSrv(struct AcpHandshakeReq* FromSS)
{
	serHandshakeProcessFree0Srv(FromSS);
}

void acpHandshakeProcessFreeCltSrv(struct AcpHandshakeReq* FromSS)
{
	serHandshakeProcessFreeSrv(FromSS);
}

void acpHandshakeProcessInitSrv(acpCtx_t _ctx, struct AcpHandshakeCnf** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serHandshakeProcessInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpHandshakeProcessEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct AcpHandshakeCnf* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serHandshakeProcessEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_HandshakeProcess, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpHandshakeProcessDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct AcpHandshakeCnf** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serHandshakeProcessDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpHandshakeProcessFree0Clt(struct AcpHandshakeCnf* ToSS)
{
	serHandshakeProcessFree0Clt(ToSS);
}

void acpHandshakeProcessFreeClt(struct AcpHandshakeCnf* ToSS)
{
	serHandshakeProcessFreeClt(ToSS);
}

void acpHandshakeProcessFree0SrvClt(struct AcpHandshakeCnf* ToSS)
{
	serHandshakeProcessFree0Clt(ToSS);
}

void acpHandshakeProcessFreeSrvClt(struct AcpHandshakeCnf* ToSS)
{
	serHandshakeProcessFreeClt(ToSS);
}
