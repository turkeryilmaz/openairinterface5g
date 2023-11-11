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

#include "acpSys.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serSys.h"

void acpSysProcessInitClt(acpCtx_t _ctx, struct SYSTEM_CTRL_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysProcessInitClt(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

int acpSysProcessEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct SYSTEM_CTRL_REQ* FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysProcessEncClt(_buffer, *_size, &_lidx, FromSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysProcess, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysProcessDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct SYSTEM_CTRL_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysProcessDecSrv(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

void acpSysProcessFree0Srv(struct SYSTEM_CTRL_REQ* FromSS)
{
	serSysProcessFree0Srv(FromSS);
}

void acpSysProcessFreeSrv(struct SYSTEM_CTRL_REQ* FromSS)
{
	serSysProcessFreeSrv(FromSS);
}

void acpSysProcessFree0CltSrv(struct SYSTEM_CTRL_REQ* FromSS)
{
	serSysProcessFree0Srv(FromSS);
}

void acpSysProcessFreeCltSrv(struct SYSTEM_CTRL_REQ* FromSS)
{
	serSysProcessFreeSrv(FromSS);
}

void acpSysProcessInitSrv(acpCtx_t _ctx, struct SYSTEM_CTRL_CNF** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysProcessInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpSysProcessEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct SYSTEM_CTRL_CNF* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysProcessEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysProcess, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysProcessDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct SYSTEM_CTRL_CNF** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysProcessDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpSysProcessFree0Clt(struct SYSTEM_CTRL_CNF* ToSS)
{
	serSysProcessFree0Clt(ToSS);
}

void acpSysProcessFreeClt(struct SYSTEM_CTRL_CNF* ToSS)
{
	serSysProcessFreeClt(ToSS);
}

void acpSysProcessFree0SrvClt(struct SYSTEM_CTRL_CNF* ToSS)
{
	serSysProcessFree0Clt(ToSS);
}

void acpSysProcessFreeSrvClt(struct SYSTEM_CTRL_CNF* ToSS)
{
	serSysProcessFreeClt(ToSS);
}
