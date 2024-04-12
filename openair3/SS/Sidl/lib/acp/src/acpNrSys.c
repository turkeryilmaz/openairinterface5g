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

#include "acpNrSys.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serNrSys.h"

void acpNrSysProcessInitClt(acpCtx_t _ctx, struct NR_SYSTEM_CTRL_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serNrSysProcessInitClt(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

int acpNrSysProcessEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct NR_SYSTEM_CTRL_REQ* FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serNrSysProcessEncClt(_buffer, *_size, &_lidx, FromSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_NrSysProcess, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpNrSysProcessDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct NR_SYSTEM_CTRL_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serNrSysProcessDecSrv(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

void acpNrSysProcessFree0Srv(struct NR_SYSTEM_CTRL_REQ* FromSS)
{
	serNrSysProcessFree0Srv(FromSS);
}

void acpNrSysProcessFreeSrv(struct NR_SYSTEM_CTRL_REQ* FromSS)
{
	serNrSysProcessFreeSrv(FromSS);
}

void acpNrSysProcessFree0CltSrv(struct NR_SYSTEM_CTRL_REQ* FromSS)
{
	serNrSysProcessFree0Srv(FromSS);
}

void acpNrSysProcessFreeCltSrv(struct NR_SYSTEM_CTRL_REQ* FromSS)
{
	serNrSysProcessFreeSrv(FromSS);
}

void acpNrSysProcessInitSrv(acpCtx_t _ctx, struct NR_SYSTEM_CTRL_CNF** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serNrSysProcessInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpNrSysProcessEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct NR_SYSTEM_CTRL_CNF* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serNrSysProcessEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_NrSysProcess, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpNrSysProcessDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct NR_SYSTEM_CTRL_CNF** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serNrSysProcessDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpNrSysProcessFree0Clt(struct NR_SYSTEM_CTRL_CNF* ToSS)
{
	serNrSysProcessFree0Clt(ToSS);
}

void acpNrSysProcessFreeClt(struct NR_SYSTEM_CTRL_CNF* ToSS)
{
	serNrSysProcessFreeClt(ToSS);
}

void acpNrSysProcessFree0SrvClt(struct NR_SYSTEM_CTRL_CNF* ToSS)
{
	serNrSysProcessFree0Clt(ToSS);
}

void acpNrSysProcessFreeSrvClt(struct NR_SYSTEM_CTRL_CNF* ToSS)
{
	serNrSysProcessFreeClt(ToSS);
}
