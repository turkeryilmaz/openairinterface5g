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

#include <string.h>
#include <strings.h>

// Internal includes
#include "acpCtx.h"
#include "acpMem.h"
#include "acpIdMap.h"
#include "adbg.h"

// Definitions
#define ACP_USER_ID_DEFAULT_MAP_SIZE 8

// Use assert if the interface in ACP host package doesn't match the interface on server
// #define ACP_ASSERT_INTERFACE_MISMATCH

// Static variables
static struct acpCtx _contexts[ACP_MAX_CTX_QTY];

void acpCtxInit(void)
{
	ACP_DEBUG_ENTER_LOG();
	ACP_DEBUG_LOG("Clearing the context pool");
	memset(_contexts, 0, sizeof(_contexts));
	ACP_DEBUG_EXIT_LOG(NULL);
}

acpCtx_t acpTakeCtx(void)
{
	ACP_DEBUG_ENTER_LOG();

	for (int index = 0; index < ACP_MAX_CTX_QTY; index++) {
		if (!_contexts[index].ptr) {
			_contexts[index].ptr = &_contexts[index];

			_contexts[index].desc = NULL;
			_contexts[index].arena = NULL;
			_contexts[index].aSize = 0;
			_contexts[index].isServer = false;
			_contexts[index].handle = NULL;
			_contexts[index].sHandle = NULL;
			_contexts[index].sock = -1;
			for (int peer = 0; peer < ACP_MAX_PEER_QTY; peer++) {
				_contexts[index].peers[peer] = -1;
				_contexts[index].peersHandshaked[peer] = 0;
				_contexts[index].peersServices[peer] = NULL;
				_contexts[index].peersServicesSize[peer] = 0;
			}
			_contexts[index].peersSize = 0;
			_contexts[index].lastPeer = -1;
			_contexts[index].lastType = 0;
			_contexts[index].lastKind = -1;
			_contexts[index].userIdMap = (struct acpUserService*)acpMalloc(ACP_USER_ID_DEFAULT_MAP_SIZE * sizeof(struct acpUserService));
			SIDL_ASSERT(_contexts[index].userIdMap);
			_contexts[index].userIdMapMaxSize = ACP_USER_ID_DEFAULT_MAP_SIZE;
			_contexts[index].userIdMapSize = 0;
			_contexts[index].tmpBuf = NULL;
			_contexts[index].tmpBufSize = 0;
			_contexts[index].opaqueBuf = NULL;
			_contexts[index].opaqueBufSize = 0;

			_contexts[index].logger = NULL;
			_contexts[index].logBuf = NULL;
			_contexts[index].logBufMaxSize = 0;
			_contexts[index].logBufSize = 0;
			_contexts[index].logFormat = false;
			_contexts[index].logFormatBuf = NULL;
			_contexts[index].logFormatBufMaxSize = 0;

			ACP_DEBUG_LOG("Adding a new context: index=%d, _contexts[%d].ptr=%p", index, index, _contexts[index].ptr);

			ACP_DEBUG_EXIT_LOG(NULL);
			return &_contexts[index];
		}
	}

	ACP_DEBUG_EXIT_LOG(NULL);
	return NULL;
}

void acpGiveCtx(acpCtx_t ctx)
{
	ACP_DEBUG_ENTER_LOG();

	for (int index = 0; index < ACP_MAX_CTX_QTY; index++) {
		if (_contexts[index].ptr == ctx) {
			ACP_DEBUG_LOG("Clearing the context: index=%d, _contexts[%d].ptr=%p", index, index, _contexts[index].ptr);
			_contexts[index].ptr = NULL;

			if (_contexts[index].desc) {
				acpFree(_contexts[index].desc);
			}
			if (_contexts[index].arena) {
				acpFree(_contexts[index].arena);
			}
			for (int peer = 0; peer < ACP_MAX_PEER_QTY; peer++) {
				if (_contexts[index].peersServices[peer]) {
					acpFree(_contexts[index].peersServices[peer]);
				}
			}
			if (_contexts[index].userIdMap) {
				acpFree(_contexts[index].userIdMap);
			}
			if (_contexts[index].tmpBuf) {
				acpFree(_contexts[index].tmpBuf);
			}

			if (_contexts[index].logBuf) {
				acpFree(_contexts[index].logBuf);
			}
			if (_contexts[index].logFormatBuf) {
				acpFree(_contexts[index].logFormatBuf);
			}

			break;
		}
	}

	ACP_DEBUG_EXIT_LOG(NULL);
}

const char* acpCtxGetItfNameFrom_localId(int id)
{
	int itf_id = (id >> 8) & 0xFF;

	for (unsigned int i = 0; i < acpItfMapSize; i++) {
		if (acpItfMap[i].id == itf_id) {
			return acpItfMap[i].name;
		}
	}

	return NULL;
}

int acpGetIndexFrom_localId_name(int id, const char* name)
{
	static int last_itf_id_index = 0;

	// FIXME: how to handle SIDL_PUBLIC_SERVICES
	if ((id & 0xF0000000) != 0x90000000) {
		return -ACP_ERR_UNKNOWN_SERVICE_NAME;
	}

	int itf_id = (id >> 8) & 0xFF;

	if (acpItfMap[last_itf_id_index].id != itf_id) {
		for (unsigned int i = 0; i < acpItfMapSize; i++) {
			if (acpItfMap[i].id == itf_id) {
				last_itf_id_index = i;
				break;
			}
		}
	}

	if (last_itf_id_index == (int)acpItfMapSize) {
		last_itf_id_index = 0;
		return -ACP_ERR_UNKNOWN_SERVICE_NAME;
	}

	int service_id_index = acpItfMap[last_itf_id_index].startIndex + (id & 0xFF);

	// This is just to check that the current server service matches the generated one
	if (name) {
		size_t remoteNameOffset = strlen(acpItfMap[last_itf_id_index].name);
		remoteNameOffset = 0; // Services names already have interface prefix in their names in this implementation

		do {
			if (service_id_index >= (int)acpIdMapSize) {
#ifndef ACP_ASSERT_INTERFACE_MISMATCH
				service_id_index = -ACP_ERR_UNKNOWN_SERVICE_NAME;
#else
				SIDL_ASSERT(service_id_index < (int)acpIdMapSize);
#endif
				break;
			}

#ifndef ACP_ASSERT_INTERFACE_MISMATCH
			if (remoteNameOffset >= strlen(acpIdMap[service_id_index].name)) {
				service_id_index = -ACP_ERR_UNKNOWN_SERVICE_NAME;
				break;
			}
#else
			SIDL_ASSERT(remoteNameOffset < strlen(acpIdMap[service_id_index].name));
#endif

			const char* remoteName = acpIdMap[service_id_index].name + remoteNameOffset;
			int ret = strcasecmp(remoteName, name);
#ifndef ACP_ASSERT_INTERFACE_MISMATCH
			if (ret != 0) {
				service_id_index = -ACP_ERR_UNKNOWN_SERVICE_NAME;
				break;
			}
#else
			SIDL_ASSERT(ret == 0);
#endif
		} while (false);

		// iterate interface to find the name
		if (service_id_index < 0) {
			unsigned int start = acpItfMap[last_itf_id_index].startIndex;
			unsigned int end = start + acpItfMap[last_itf_id_index].servicesQty;
			for (unsigned int i = start; i < end; i++) {
				if (remoteNameOffset >= strlen(acpIdMap[i].name)) {
					continue;
				}
				const char* remoteName = acpIdMap[i].name + remoteNameOffset;
				int ret = strcasecmp(remoteName, name);
				if (ret == 0) {
					service_id_index = i;
					break;
				}
			}
		}
	} else {
		if (service_id_index >= (int)acpIdMapSize) {
#ifndef ACP_ASSERT_INTERFACE_MISMATCH
			service_id_index = -ACP_ERR_UNKNOWN_SERVICE_NAME;
#else
			SIDL_ASSERT(service_id_index < (int)acpIdMapSize);
#endif
		}
	}

	return service_id_index;
}

static int acpGetIndexFrom_name(const char* name)
{
	// TODO: better search
	for (unsigned int i = 0; i < acpIdMapSize; i++) {
		if (strcasecmp(acpIdMap[i].name, name) == 0) {
			return (int)i;
		}
	}
	return -ACP_ERR_UNKNOWN_SERVICE_NAME;
}

int acpCtxGetMsgUserId(struct acpCtx* ctx, unsigned int type)
{
	for (unsigned int i = 0; i < ctx->userIdMapSize; i++) {
		if (acpIdMap[ctx->userIdMap[i].id_index].remote_id == type) {
			return ctx->userIdMap[i].user_id;
		}
	}
	return -ACP_ERR_SERVICE_NOT_MAPPED;
}

static int acpCtxSetMsgUserIdFromIndex(struct acpCtx* ctx, int index, int userId)
{
	if (ctx->userIdMapSize == ctx->userIdMapMaxSize) {
		struct acpUserService* userIdMap = (struct acpUserService*)acpMalloc((ctx->userIdMapMaxSize + ACP_USER_ID_DEFAULT_MAP_SIZE) * sizeof(struct acpUserService));
		SIDL_ASSERT(userIdMap);

		memcpy(userIdMap, ctx->userIdMap, ctx->userIdMapSize * sizeof(struct acpUserService));

		acpFree(ctx->userIdMap);

		ctx->userIdMap = userIdMap;
		ctx->userIdMapMaxSize += ACP_USER_ID_DEFAULT_MAP_SIZE;
	}

	int userIdIndex = ctx->userIdMapSize++;

	ctx->userIdMap[userIdIndex].user_id = userId;
	ctx->userIdMap[userIdIndex].id_index = index;
	acpIdMap[index].remote_id = acpIdMap[index].local_id;

	return 0;
}

int acpCtxSetMsgUserId(struct acpCtx* ctx, const char* name, int userId)
{
	int index = acpGetIndexFrom_name(name);
	if (index < 0) return index;
	SIDL_ASSERT(index < (int)acpIdMapSize);

	return acpCtxSetMsgUserIdFromIndex(ctx, index, userId);
}

int acpCtxSetMsgUserIdById(struct acpCtx* ctx, enum acpMsgLocalId lid, int userId)
{
	int index = acpGetIndexFrom_localId_name(lid, NULL);
	if (index < 0) return index;
	SIDL_ASSERT(index < (int)acpIdMapSize);

	return acpCtxSetMsgUserIdFromIndex(ctx, index, userId);
}

bool acpCtxIsValid(acpCtx_t ctx)
{
	struct acpCtx* c = ACP_CTX_CAST(ctx);
	if (c && (c->ptr == ctx)) {
		return true;
	}
	return false;
}

int acpCtxResolveId(int id, const char* name)
{
	int index = acpGetIndexFrom_localId_name(id, name);
	if (index < 0) return index;
	SIDL_ASSERT(index < (int)acpIdMapSize);

	acpIdMap[index].remote_id = id;

	return index;
}

const char* acpCtxGetMsgNameFromId(int id)
{
	int index = acpGetIndexFrom_localId_name(id, NULL);
	if (index < 0 || index >= (int)acpIdMapSize) return "UNKNOWN";
	return acpIdMap[index].name;
}

const char* acpCtxGetMsgNameFromIdStrict(int id)
{
	int index = acpGetIndexFrom_localId_name(id, NULL);
	if (index < 0 || index >= (int)acpIdMapSize) {
		SIDL_ASSERT(0);
		return NULL;
	}
	return acpIdMap[index].name;
}

int acpCtxGetMsgKindFromId(int id)
{
	int index = acpGetIndexFrom_localId_name(id, NULL);
	if (index < 0 || index >= (int)acpIdMapSize) return -ACP_ERR_UNKNOWN_SERVICE_NAME;
	return acpIdMap[index].kind;
}

int acpCtxGetMsgKindFromName(const char* name)
{
	int id = acpGetIndexFrom_name(name);
	if (id < 0) return id;
	return acpIdMap[id].kind;
}

int acpCtxSetDescription(struct acpCtx* ctx, const char* desc)
{
	size_t descLen = strlen(desc) + 1;
	if (ctx->desc) {
		acpFree(ctx->desc);
	}
	ctx->desc = (char*)acpMalloc(descLen);
	SIDL_ASSERT(ctx->desc);
	memcpy(ctx->desc, desc, descLen);
	return 0;
}

void acpCtxAllocateTmpBuf(struct acpCtx* ctx, size_t size)
{
	if (ctx->tmpBuf && (ctx->tmpBufSize < size)) {
		acpFree(ctx->tmpBuf);
		ctx->tmpBuf = NULL;
	}
	if (!ctx->tmpBuf) {
		ctx->tmpBuf = (unsigned char*)acpMalloc(size);
	}
	SIDL_ASSERT(ctx->tmpBuf);
	ctx->tmpBufSize = size;
}

int acpCtxGetPeerNum(struct acpCtx* ctx, int peer)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == peer) {
			return i;
		}
	}
	return -1;
}

bool acpCtxPeerHandshaked(struct acpCtx* ctx, int peer)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == peer) {
			return (ctx->peersHandshaked[i] != 0) ? true : false;
		}
	}
	return false;
}

bool acpCtxAnyPeerHandshaked(struct acpCtx* ctx)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peersHandshaked[i] > 0) {
			return true;
		}
	}
	return false;
}

void acpCtxPeerSetHandshaked(struct acpCtx* ctx, int peer, int flag)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == peer) {
			ctx->peersHandshaked[i] = flag;
			return;
		}
	}
}

void acpCtxAddPeer(struct acpCtx* ctx, int peer)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == -1) {
			ctx->peers[i] = peer;
			ctx->peersHandshaked[i] = 0;
			SIDL_ASSERT(!ctx->peersServices[i]);
			ctx->peersSize++;
			return;
		}
	}
	SIDL_ASSERT(0);
}

void acpCtxRemovePeer(struct acpCtx* ctx, int peer)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == peer) {
			ctx->peers[i] = -1;
			ctx->peersHandshaked[i] = 0;
			if (ctx->peersServices[i]) {
				acpFree(ctx->peersServices[i]);
				ctx->peersServices[i] = NULL;
				ctx->peersServicesSize[i] = 0;
			}
			ctx->peersSize--;
			return;
		}
	}
}

bool acpCtxPeerRespondsToService(struct acpCtx* ctx, int peer, int type)
{
	int peerNum = acpCtxGetPeerNum(ctx, peer);
	if (peerNum == -1) {
		return false;
	}
	return acpCtxPeerNumRespondsToService(ctx, peerNum, type);
}

bool acpCtxPeerNumRespondsToService(struct acpCtx* ctx, int peerNum, int type)
{
	for (size_t i = 0; i < ctx->peersServicesSize[peerNum]; i++) {
		if (ctx->peersServices[peerNum][i] == type) {
			return true;
		}
	}
	return false;
}
