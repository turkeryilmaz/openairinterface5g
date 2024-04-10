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

#include "acpDrb.h"
#include "acpCtx.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "serDrb.h"

void acpDrbProcessFromSSInitClt(acpCtx_t _ctx, struct DRB_COMMON_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serDrbProcessFromSSInitClt(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

int acpDrbProcessFromSSEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct DRB_COMMON_REQ* FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serDrbProcessFromSSEncClt(_buffer, *_size, &_lidx, FromSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_DrbProcessFromSS, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpDrbProcessFromSSDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct DRB_COMMON_REQ** FromSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serDrbProcessFromSSDecSrv(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, FromSS);
}

void acpDrbProcessFromSSFree0Srv(struct DRB_COMMON_REQ* FromSS)
{
	serDrbProcessFromSSFree0Srv(FromSS);
}

void acpDrbProcessFromSSFreeSrv(struct DRB_COMMON_REQ* FromSS)
{
	serDrbProcessFromSSFreeSrv(FromSS);
}

void acpDrbProcessFromSSFree0CltSrv(struct DRB_COMMON_REQ* FromSS)
{
	serDrbProcessFromSSFree0Srv(FromSS);
}

void acpDrbProcessFromSSFreeCltSrv(struct DRB_COMMON_REQ* FromSS)
{
	serDrbProcessFromSSFreeSrv(FromSS);
}

void acpDrbProcessToSSInitSrv(acpCtx_t _ctx, struct DRB_COMMON_IND** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		SIDL_ASSERT(_ctx != _ctx);
	}
	serDrbProcessToSSInitSrv(ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

int acpDrbProcessToSSEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct DRB_COMMON_IND* ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	size_t _lidx = ACP_HEADER_SIZE;
	int _ret = serDrbProcessToSSEncSrv(_buffer, *_size, &_lidx, ToSS);
	if (_ret == SIDL_STATUS_OK) {
		acpBuildHeader(_ctx, ACP_LID_DrbProcessToSS, _lidx, _buffer);
	}
	*_size = _lidx;
	return _ret;
}

int acpDrbProcessToSSDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct DRB_COMMON_IND** ToSS)
{
	if (!acpCtxIsValid(_ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return serDrbProcessToSSDecClt(_buffer + ACP_HEADER_SIZE, _size - ACP_HEADER_SIZE, ACP_CTX_CAST(_ctx)->arena, ACP_CTX_CAST(_ctx)->aSize, ToSS);
}

void acpDrbProcessToSSFree0Clt(struct DRB_COMMON_IND* ToSS)
{
	serDrbProcessToSSFree0Clt(ToSS);
}

void acpDrbProcessToSSFreeClt(struct DRB_COMMON_IND* ToSS)
{
	serDrbProcessToSSFreeClt(ToSS);
}

void acpDrbProcessToSSFree0SrvClt(struct DRB_COMMON_IND* ToSS)
{
	serDrbProcessToSSFree0Clt(ToSS);
}

void acpDrbProcessToSSFreeSrvClt(struct DRB_COMMON_IND* ToSS)
{
	serDrbProcessToSSFreeClt(ToSS);
}
