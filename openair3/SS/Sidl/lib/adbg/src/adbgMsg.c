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

#include "adbgMsg.h"
#include "acpCtx.h"
#include "adbg.h"
#include "adbgMsgMap.h"

void adbgMsgLog(acpCtx_t ctx, enum adbgMsgLogDir dir, size_t size, const unsigned char* buffer)
{
	if (!acpCtxIsValid(ctx)) {
		SIDL_ASSERT(ctx != ctx);
		ACP_DEBUG_LOG("Invalid context");
		return;
	}

	bool isServer = ACP_CTX_CAST(ctx)->isServer;

	enum acpMsgLocalId localId;

	if (acpGetMsgLocalId(size, buffer, &localId) < -1) {
		adbgPrintLog(ctx, "Invalid buffer");
		adbgPrintLog(ctx, NULL);
		return;
	}

	// Service kind (0 - NTF, 1 - ONEWAY, 2 - CMD)
	int kind = acpCtxGetMsgKindFromId(localId);
	SIDL_ASSERT(kind != -1);
	if (kind < -1) {
		adbgPrintLog(ctx, "Cannot find service");
		adbgPrintLog(ctx, NULL);
		return;
	}

	if ((kind == 1 || kind == 2) && dir == ADBG_MSG_LOG_RECV_DIR && isServer) {
		adbgMsgLogInArgs(ctx, localId, size - ACP_HEADER_SIZE, buffer + ACP_HEADER_SIZE);
	} else if ((kind == 0 || kind == 2) && dir == ADBG_MSG_LOG_SEND_DIR && isServer) {
		adbgMsgLogOutArgs(ctx, localId, size - ACP_HEADER_SIZE, buffer + ACP_HEADER_SIZE);
	} else if ((kind == 1 || kind == 2) && dir == ADBG_MSG_LOG_SEND_DIR && !isServer) {
		adbgMsgLogInArgs(ctx, localId, size - ACP_HEADER_SIZE, buffer + ACP_HEADER_SIZE);
	} else if ((kind == 0 || kind == 2) && dir == ADBG_MSG_LOG_RECV_DIR && !isServer) {
		adbgMsgLogOutArgs(ctx, localId, size - ACP_HEADER_SIZE, buffer + ACP_HEADER_SIZE);
	} else {
		SIDL_ASSERT(0);
	}
}
