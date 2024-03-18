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

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// Internal includes
#include "acp.h"
#include "acpMem.h"
#include "acpCtx.h"
#include "acpSocket.h"
#include "acpProto.h"
#include "acpMsgIds.h"
#include "acpInternal.h"
#include "adbg.h"
#include "adbgMsg.h"

// Static variables
static acpMalloc_t _alloc = NULL;
static acpFree_t _release = NULL;
static MSec_t _socketTimeout = 1000;
static bool _initialized = false;

// Behaviour specific variables to use outside this module
bool acp_printPrettyMessages = false;
bool acp_ProductMode = true;

// ___________________________ / Interface / ____________________________

void* acpMalloc(size_t size)
{
	if (_alloc) {
		return _alloc(size);
	}
	return NULL;
}

void acpFree(void* ptr)
{
	if (_release) {
		_release(ptr);
		return;
	}
}

void acpInit(acpMalloc_t alloc, acpFree_t release, MSec_t socketTimeout)
{
	ACP_DEBUG_ENTER_TRACE_LOG();

	if (_initialized) {
		ACP_DEBUG_EXIT_TRACE_LOG("FALSE");
		return;
	}

	SIDL_ASSERT((alloc && release) || (!alloc && !release));

	_alloc = alloc ?: malloc;

	_release = release ?: free;

	_socketTimeout = socketTimeout ?: ACP_DEFAULT_TIMEOUT;

	acpCtxInit();

	_initialized = true;

	ACP_DEBUG_EXIT_TRACE_LOG("TRUE");
}

int acpCreateCtx(acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);

	*ctx = acpTakeCtx();
	if (!*ctx) {
		return -ACP_ERR_INVALID_CTX;
	}

	if (acp_printPrettyMessages) {
		adbgSetPrintLogFormat(*ctx, true);
	}

	return 0;
}

int acpDeleteCtx(acpCtx_t ctx)
{
	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	acpGiveCtx(ctx);

	return 0;
}

bool acpIsConnected(acpCtx_t ctx)
{
	if (!acpCtxIsValid(ctx)) {
		return false;
	}

	return (ACP_CTX_CAST(ctx)->sHandle != NULL);
}

int acpGetMsgSidlStatus(size_t size, const unsigned char* buffer, SidlStatus* sidlStatus)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	if (sidlStatus) {
		*sidlStatus = (SidlStatus)acpGetServiceStatus(size, buffer);
	}

	return 0;
}

int acpSetMsgSidlStatus(size_t size, unsigned char* buffer, SidlStatus sidlStatus)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	acpSetServiceStatus(size, buffer, (int)sidlStatus);

	return 0;
}

int acpGetMsgLength(size_t size, const unsigned char* buffer)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	return acpGetServicePayloadLength(size, buffer);
}

int acpGetMsgId(acpCtx_t ctx, size_t size, const unsigned char* buffer)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	int type = acpGetServiceType(size, buffer);

	return acpCtxGetMsgUserId(ACP_CTX_CAST(ctx), type);
}

int acpGetMsgLocalId(size_t size, const unsigned char* buffer, enum acpMsgLocalId* localMsgId)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	int type = acpGetServiceType(size, buffer);

	if (localMsgId) {
		*localMsgId = (enum acpMsgLocalId)type;
	}

	return 0;
}

/** Performs full connection to server. */
static int acpConnectToSrv(struct acpCtx* ctx, int sock)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	/* Perform handshake to server */
	int ret = acpHandleHandshakeToServer(ctx, sock);

	if (ret < 0) {
		ACP_DEBUG_EXIT_CLOG(ctx, "ERROR: acpHandleHandshakeToServer failed");
		return ret;
	}

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return ret;
}

int acpRecvMsg(acpCtx_t ctx, size_t* size, unsigned char* buffer)
{
	SIDL_ASSERT(size && buffer);
#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_ENTER_TRACE_CLOG(ctx);
#endif

	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	/*
	if (ACP_CTX_CAST(ctx)->isServer && (ACP_CTX_CAST(ctx)->lastKind == ACP_CMD)) {
		ACP_DEBUG_EXIT_TRACE_LOG("ERROR: server should respond to CMD");
		return -ACP_ERR_INTERNAL;
	}
	*/

	int ret;
	int userId;

	int sock = ACP_CTX_CAST(ctx)->sock;
	size_t peersSize = 0;
	int* peers = NULL;

	if (ACP_CTX_CAST(ctx)->isServer) {
		peersSize = ACP_CTX_CAST(ctx)->peersSize;
		peers = ACP_CTX_CAST(ctx)->peers;
	}

	ret = acpSocketSelectMulti(&sock, _socketTimeout, peersSize, peers);
	if (ret == 0) {
#ifdef ACP_DEBUG_TRACE_FLOOD
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
#endif
		return 0;
	}
	if (ret < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR: acpSocketSelectMulti failed");
		return -ACP_ERR_SOCK_ERROR;
	}

	ACP_DEBUG_TRACE_CLOG(ctx, "Data on socket(s)");

	if (ACP_CTX_CAST(ctx)->isServer && ACP_CTX_CAST(ctx)->sock == sock) {
		int peerSock = acpSocketAccept(sock);
		if (peerSock < 0) {
			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR: acpSocketAccept failed");
			return -ACP_ERR_SOCK_ERROR;
		}
		acpCtxAddPeer(ACP_CTX_CAST(ctx), peerSock);
		ACP_DEBUG_CLOG(ctx, "Peer connected");

		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
		return 0;
	}

	/* Read SIDL message */
	ret = acpRecvMsgInternal(sock, size, buffer);
	if (ret < 0) {
		if (ACP_CTX_CAST(ctx)->isServer && (ret == -ACP_PEER_DISCONNECTED)) {
			acpCtxRemovePeer(ACP_CTX_CAST(ctx), sock);
			acpSocketClose(sock);
			ACP_DEBUG_CLOG(ctx, "Peer disconnected");

			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
			return -ACP_PEER_DISCONNECTED;
		} else if (!ACP_CTX_CAST(ctx)->isServer && (ret == -ACP_PEER_DISCONNECTED)) {
			ret = -ACP_ERR_SOCKCONN_ABORTED;
		}

		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR: acpRecvMsgInternal failed");
		return ret;
	}

	/* Get message ID */
	userId = acpGetMsgId(ctx, ACP_HEADER_SIZE, buffer);

	int handle = acpGetServiceHandle(ACP_HEADER_SIZE, buffer);
	if (handle == (int)ACP_KEEPALIVE_HANDLE) {
		userId = 0;
	}

	int type = acpGetServiceType(ACP_HEADER_SIZE, buffer);
	ACP_DEBUG_CLOG(ctx, "Receiving message '%s' (userId: %d, type: 0x%x)", acpGetMsgName(ACP_HEADER_SIZE, buffer), userId, type);

	int kind = acpCtxGetMsgKindFromId(type);
	if (kind < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR: acpCtxGetMsgKindFromId failed");
		return -ACP_ERR_INTERNAL;
	}

	if (ACP_CTX_CAST(ctx)->isServer) {
		SIDL_ASSERT((kind == ACP_CMD) || (kind == ACP_ONEWAY));
	} else {
		SIDL_ASSERT((kind == ACP_CMD) || (kind == ACP_NTF));
	}

	if (ACP_CTX_CAST(ctx)->isServer) {
		bool handshaked = acpCtxPeerHandshaked(ctx, sock);
		if (!handshaked && type != (int)ACP_MSG_ID(HandshakeProcess)) {
			/* The first message should be HandshakeProcess from client */
			ACP_DEBUG_CLOG(ctx, "ERROR: wrong message, should be HandshakeProcess");
			ret = -ACP_ERR_INTERNAL;
		} else if (type == (int)ACP_MSG_ID(HandshakeProcess)) {
			if (handshaked) {
				ACP_DEBUG_CLOG(ctx, "ERROR: client has been already handshaked, ignore");
				userId = 0;
			} else {
				ACP_CTX_CAST(ctx)->opaqueBufSize = *size;
				ACP_CTX_CAST(ctx)->opaqueBuf = buffer;

				/* Handle handshake from client */
				ret = acpHandleHandshakeFromClient(ACP_CTX_CAST(ctx), sock);

				ACP_CTX_CAST(ctx)->opaqueBuf = NULL;

				if (ret == 0) {
					ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
					return -ACP_PEER_CONNECTED;
				} else {
					ACP_DEBUG_CLOG(ctx, "ERROR: acpHandleHandshakeFromClient failed");
				}
			}
		}
	}

	do {
		if (ret < 0) {
			break;
		}

		acpCtxAllocateTmpBuf(ctx, ACP_HEADER_SIZE);
		acpBuildHeader(ctx, type, ACP_HEADER_SIZE, ACP_CTX_CAST(ctx)->tmpBuf);

		if (userId == -ACP_ERR_SERVICE_NOT_MAPPED) {
			acpSetServiceStatus(ACP_HEADER_SIZE, ACP_CTX_CAST(ctx)->tmpBuf, SIDL_STATUS_NOTIMP);
		} else {
			break;
		}

		ret = acpSendMsgInternal(sock, ACP_HEADER_SIZE, ACP_CTX_CAST(ctx)->tmpBuf);
		if (ret < 0) {
			ACP_DEBUG_CLOG(ctx, "ERROR: acpSendMsgInternal failed");
			break;
		}
	} while (false);

	if (ret < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR");
		return ret;
	}

	if (userId >= 0) {
		if ((userId > 0) && ACP_CTX_CAST(ctx)->isServer) {
			if (kind == ACP_CMD) {
				ACP_CTX_CAST(ctx)->lastPeer = sock;
			}
			ACP_CTX_CAST(ctx)->lastType = type;
			ACP_CTX_CAST(ctx)->lastKind = kind;
		}

#ifdef ACP_DEBUG
		ACP_DEBUG_PREFIX_CLOG(ctx, "Received message '%s'", acpGetMsgName(ACP_HEADER_SIZE, buffer));
		adbgMsgLog(ctx, ADBG_MSG_LOG_RECV_DIR, *size, buffer);
#else
		ACP_DEBUG_CLOG(ctx, "Received message '%s'", acpGetMsgName(ACP_HEADER_SIZE, buffer));
#endif
	}

#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
#endif
	return userId;
}

int acpSendMsg(acpCtx_t ctx, size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);
#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_ENTER_TRACE_CLOG(ctx);
#endif
	int ret = -ACP_ERR_INTERNAL;

	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	int type = acpGetServiceType(ACP_HEADER_SIZE, buffer);

	int kind = acpCtxGetMsgKindFromId(type);
	if (kind < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR: acpCtxGetMsgKindFromId failed");
		return -ACP_ERR_INTERNAL;
	}

	if (ACP_CTX_CAST(ctx)->isServer) {
		SIDL_ASSERT((kind == ACP_CMD) || (kind == ACP_NTF));
	} else {
		SIDL_ASSERT((kind == ACP_CMD) || (kind == ACP_ONEWAY));
	}

	if (ACP_CTX_CAST(ctx)->isServer && (ACP_CTX_CAST(ctx)->lastKind != ACP_CMD) && (kind == ACP_CMD)) {
		ACP_DEBUG_EXIT_TRACE_LOG("ERROR: server tries to respond CMD without request");
		return -ACP_ERR_INTERNAL;
	}

#ifdef ACP_DEBUG
	ACP_DEBUG_PREFIX_CLOG(ctx, "Sending message '%s'", acpGetMsgName(ACP_HEADER_SIZE, buffer));
	adbgMsgLog(ctx, ADBG_MSG_LOG_SEND_DIR, size, buffer);
#else
	ACP_DEBUG_CLOG(ctx, "Sending message '%s'", acpGetMsgName(ACP_HEADER_SIZE, buffer));
#endif

	int sock = ACP_CTX_CAST(ctx)->sock;
	if (ACP_CTX_CAST(ctx)->isServer) {
		if (kind == ACP_CMD) {
			SIDL_ASSERT(ACP_CTX_CAST(ctx)->lastPeer != -1);
			sock = ACP_CTX_CAST(ctx)->lastPeer;
		} else if (kind == ACP_NTF) {
			SIDL_ASSERT(ACP_CTX_CAST(ctx)->lastPeer == -1);
		}
	}

	ret = 0;

	/* Write SIDL message */
	if (kind != ACP_NTF) {
		if (ACP_CTX_CAST(ctx)->isServer && !acpCtxPeerRespondsToService(ACP_CTX_CAST(ctx), sock, type)) {
			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR: no peer for the message");
			return -ACP_ERR_INTERNAL;
		}

		ACP_DEBUG_CLOG(ctx, "Sending message '%s' to peer=%d", acpGetMsgName(ACP_HEADER_SIZE, buffer), sock);
		ret = acpSendMsgInternal(sock, size, buffer);
		if (ret < 0) {
			ACP_DEBUG_CLOG(ctx, "ERROR: acpSendMsgInternal failed");
		}
	} else {
		SIDL_ASSERT(ACP_CTX_CAST(ctx)->isServer);
		for (size_t peerNum = 0; peerNum < ACP_MAX_PEER_QTY; peerNum++) {
			sock = ACP_CTX_CAST(ctx)->peers[peerNum];
			if (sock == -1) {
				continue;
			}
			if (!acpCtxPeerNumRespondsToService(ACP_CTX_CAST(ctx), peerNum, type)) {
				continue;
			}

			ACP_DEBUG_CLOG(ctx, "Sending message '%s' to peer=%d", acpGetMsgName(ACP_HEADER_SIZE, buffer), sock);
			ret = acpSendMsgInternal(sock, size, buffer);
			if (ret < 0) {
				ACP_DEBUG_CLOG(ctx, "ERROR: acpSendMsgInternal failed");
				break;
			}
		}
	}

	if (ACP_CTX_CAST(ctx)->isServer) {
		ACP_CTX_CAST(ctx)->lastPeer = -1;
		ACP_CTX_CAST(ctx)->lastType = 0;
		ACP_CTX_CAST(ctx)->lastKind = -1;
	}

	if (ret < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ERROR");
		return ret;
	}

#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
#endif
	return 0;
}

int acpInitCtx(const struct acpMsgTable* msgTable, acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);
	ACP_DEBUG_ENTER_LOG();

	if (acpCreateCtx(ctx) < 0) {
		ACP_DEBUG_EXIT_LOG("ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	if (msgTable) {
		while (msgTable->name) {
			int ret = acpSetMsgId(*ctx, msgTable->name, msgTable->userId);
			if (ret < 0) {
				acpDeleteCtx(*ctx);
				*ctx = NULL;
				ACP_DEBUG_EXIT_LOG("ERROR: acpSetMsgId failed");
				return ret;
			}
			msgTable++;
		}
	}

	ACP_DEBUG_EXIT_LOG(NULL);
	return 0;
}

int acpClientInit(acpCtx_t ctx, const char* host, int port, size_t aSize)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	int sock = acpSocketConnect(host, port);
	if (sock < 0) {
		ACP_DEBUG_EXIT_CLOG(ctx, "ACP_ERR_SOCKCONN_ABORTED");
		return -ACP_ERR_SOCKCONN_ABORTED;
	}

	int ret = acpConnectToSrv(ACP_CTX_CAST(ctx), sock);
	if (ret < 0) {
		acpSocketClose(sock);
		ACP_DEBUG_CLOG(ctx, "ERROR: failed to connect to server, error=%d", ret);

		ACP_DEBUG_EXIT_CLOG(ctx, NULL);
		return ret;
	}

	/* Should not happen */
	/*if (!acpIsConnected(ctx)) {
		acpSocketClose(sock);
		return -ACP_ERR_NOT_CONNECTED;
	}*/

	ACP_CTX_CAST(ctx)->sock = sock;
	ACP_CTX_CAST(ctx)->isServer = false;
	ACP_CTX_CAST(ctx)->aSize = aSize;
	ACP_CTX_CAST(ctx)->arena = (unsigned char*)acpMalloc(aSize);
	SIDL_ASSERT(ACP_CTX_CAST(ctx)->arena);

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return 0;
}

int acpClientInitWithCtx(const char* host, int port, const struct acpMsgTable* msgTable, size_t aSize, acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);
	ACP_DEBUG_ENTER_LOG();

	acpCtx_t _ctx = NULL;
	int ret = acpInitCtx(msgTable, &_ctx);
	if (ret < 0) {
		ACP_DEBUG_EXIT_LOG("ERROR");
		return ret;
	}

	ret = acpClientInit(_ctx, host, port, aSize);
	if (ret < 0) {
		acpDeleteCtx(_ctx);

		ACP_DEBUG_EXIT_LOG("ERROR: acpClientInit failed");
		return ret;
	}

	*ctx = _ctx;

	ACP_DEBUG_EXIT_LOG(NULL);
	return 0;
}

int acpServerInit(acpCtx_t ctx, const char* host, int port, size_t aSize)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	int sock = acpSocketListen(host, port);
	if (sock < 0) {
		ACP_DEBUG_EXIT_CLOG(ctx, "ACP_ERR_SOCKCONN_ABORTED");
		return -ACP_ERR_SOCKCONN_ABORTED;
	}

	ACP_CTX_CAST(ctx)->sock = sock;
	ACP_CTX_CAST(ctx)->isServer = true;
	ACP_CTX_CAST(ctx)->aSize = aSize;
	ACP_CTX_CAST(ctx)->arena = (unsigned char*)acpMalloc(aSize);
	SIDL_ASSERT(ACP_CTX_CAST(ctx)->arena);

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return 0;
}

int acpServerInitWithCtx(const char* host, int port, const struct acpMsgTable* msgTable, size_t aSize, acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);
	ACP_DEBUG_ENTER_LOG();

	acpCtx_t _ctx = NULL;
	int ret = acpInitCtx(msgTable, &_ctx);
	if (ret < 0) {
		ACP_DEBUG_EXIT_LOG("ERROR");
		return ret;
	}

	ret = acpServerInit(_ctx, host, port, aSize);
	if (ret < 0) {
		acpDeleteCtx(_ctx);

		ACP_DEBUG_EXIT_LOG("ERROR: acpServerInit failed");
		return ret;
	}

	*ctx = _ctx;

	ACP_DEBUG_EXIT_LOG(NULL);
	return 0;
}

int acpClose0(acpCtx_t ctx)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_EXIT_LOG("ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	acpSocketClose(ACP_CTX_CAST(ctx)->sock);

	if (ACP_CTX_CAST(ctx)->isServer) {
		for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
			if (ACP_CTX_CAST(ctx)->peers[i] != -1) {
				acpSocketClose(ACP_CTX_CAST(ctx)->peers[i]);
			}
		}
	}

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return 0;
}

int acpClose(acpCtx_t ctx)
{
	ACP_DEBUG_ENTER_LOG();

	int ret = acpClose0(ctx);
	if (ret < 0) {
		ACP_DEBUG_EXIT_LOG("ERROR");
		return ret;
	}

	acpDeleteCtx(ctx);

	ACP_DEBUG_EXIT_LOG(NULL);
	return 0;
}

int acpGetSocketFd(acpCtx_t ctx)
{
	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	return ACP_CTX_CAST(ctx)->sock;
}

void acpSetSocketFd(acpCtx_t ctx, int socketfd)
{
	if (acpCtxIsValid(ctx)) {
		ACP_CTX_CAST(ctx)->sock = socketfd;
	}
}

int acpGetLastPeerSocketFd(acpCtx_t ctx)
{
	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	return ACP_CTX_CAST(ctx)->lastPeer;
}

void acpSetLastPeerSocketFd(acpCtx_t ctx, int socketfd)
{
	if (acpCtxIsValid(ctx)) {
		ACP_CTX_CAST(ctx)->lastPeer = socketfd;
	}
}

MSec_t acpGetSocketTimeout(void)
{
	return _socketTimeout;
}

int acpSetMsgId(acpCtx_t ctx, const char* name, int userMsgId)
{
	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}
	return acpCtxSetMsgUserId(ACP_CTX_CAST(ctx), name, userMsgId);
}

int acpSetMsgIdFromLocalId(acpCtx_t ctx, enum acpMsgLocalId localMsgId, int userMsgId)
{
	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	return acpCtxSetMsgUserIdById(ACP_CTX_CAST(ctx), localMsgId, userMsgId);
}

const char* acpGetMsgName(size_t size, const unsigned char* buffer)
{
	if (size < ACP_HEADER_SIZE) {
		return "INVALID_HEADER";
	}
	int type = acpGetServiceType(size, buffer);
	return acpCtxGetMsgNameFromId(type);
}

int acpSetDescription(acpCtx_t ctx, const char* desc)
{
	SIDL_ASSERT(desc);

	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	return acpCtxSetDescription(ACP_CTX_CAST(ctx), desc);
}

bool acpGetProductMode(void)
{
	return acp_ProductMode;
}
