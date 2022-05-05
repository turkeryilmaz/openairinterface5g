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

#include "acpSysInd.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serSysInd.h"

void acpSysIndProcessToSSInitSrv(acpCtx_t _ctx, struct SYSTEM_IND** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serSysIndProcessToSSInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpSysIndProcessToSSEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct SYSTEM_IND* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serSysIndProcessToSSEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_SysIndProcessToSS, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpSysIndProcessToSSDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct SYSTEM_IND** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serSysIndProcessToSSDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpSysIndProcessToSSFree0Clt(struct SYSTEM_IND* ToSS)
{
	serSysIndProcessToSSFree0Clt(ToSS);
}

void acpSysIndProcessToSSFreeClt(struct SYSTEM_IND* ToSS)
{
	serSysIndProcessToSSFreeClt(ToSS);
}

void acpSysIndProcessToSSFree0SrvClt(struct SYSTEM_IND* ToSS)
{
	serSysIndProcessToSSFree0Clt(ToSS);
}

void acpSysIndProcessToSSFreeSrvClt(struct SYSTEM_IND* ToSS)
{
	serSysIndProcessToSSFreeClt(ToSS);
}
