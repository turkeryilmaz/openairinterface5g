/*
 *****************************************************************
 *
 * Module : Asynchronous Communication Protocol
 * Purpose: Interface
 *
 *****************************************************************
 *
 *  Copyright (c) 2019-2021 SEQUANS Communications.
 *  All rights reserved.
 *
 *  This is confidential and proprietary source code of SEQUANS
 *  Communications. The use of the present source code and all
 *  its derived forms is exclusively governed by the restricted
 *  terms and conditions set forth in the SEQUANS
 *  Communications' EARLY ADOPTER AGREEMENT and/or LICENCE
 *  AGREEMENT. The present source code and all its derived
 *  forms can ONLY and EXCLUSIVELY be used with SEQUANS
 *  Communications' products. The distribution/sale of the
 *  present source code and all its derived forms is EXCLUSIVELY
 *  RESERVED to regular LICENCE holder and otherwise STRICTLY
 *  PROHIBITED.
 *
 *****************************************************************
 */

// System includes
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

// #define ACP_LOG_DEBUG(...) printf(__VA_ARGS__);
#define ACP_LOG_DEBUG(...)

// Static variables
static acpMalloc_t _alloc = NULL;
static acpFree_t _release = NULL;
static MSec_t _socketTimeout = 1000;
static bool _initialized = false;
bool acp_printPrettyMessages = false;

#if 0
// Performs full connection to server.
static int acpConnectToMs(struct acpCtx* ctx, int sock)
{
	// Allocate buffer to read received messages,
	// there should be at least 16K for now, since ACP_SERVICE_PUSH_TYPE is a large message,
	// the size maybe increased in the future
	const size_t size = 16*1024;
	unsigned char* buffer = (unsigned char*)acpMalloc(size);
	SIDL_ASSERT(buffer);

	// Allocate arena to decode received message,
	// the size of 1K should be enough for messages used here
	const size_t aSize = 1024;
	unsigned char* arena = (unsigned char*)acpMalloc(aSize);
	SIDL_ASSERT(arena);

	int sz, length;
	enum acpMsgLocalId localId;
	size_t msgSize;
	SidlStatus sidlStatus;

	bool isGetAvailableMsSent = false;
	bool isConnectToMsSent = false;

	int ret = 0;

#if 0 // TODO
	// Loop
	for (;;) {
		// Read ACP message header
		sz = acpSocketReceive(sock, ACP_HEADER_SIZE, buffer, _socketTimeout);
		if (sz < 0) {
			ret = -ACP_ERR_SOCK_ERROR;
			break;
		}
		if (sz != ACP_HEADER_SIZE) {
			ret = -ACP_ERR_SOCK_TIMEOUT;
			break;
		}

		// Get payload length
		length = acpGetMsgLength(ACP_HEADER_SIZE, buffer);
		if (length < 0) {
			ret = -ACP_ERR_INVALID_HEADER;
			break;
		}

		// Check decode buffer is enough to read received message
		if (length > (size - ACP_HEADER_SIZE)) {
			ret = -ACP_ERR_INTERNAL;
			break;
		}

		if (length > 0) {
			// Read ACP message payload
			sz = acpSocketReceive(sock, length, &buffer[ACP_HEADER_SIZE], _socketTimeout);
			if (sz < 0) {
				ret = -ACP_ERR_SOCK_ERROR;
				break;
			}
			if (sz != length) {
				ret = -ACP_ERR_SOCK_TIMEOUT;
				break;
			}
		}

		// Get local message ID
		if (acpGetMsgLocalId(ACP_HEADER_SIZE, buffer, &localId) < -1) {
			ret = -ACP_ERR_INVALID_HEADER;
			break;
		}

		ACP_LOG_DEBUG("<<< Receive '%s' service message [localId=%08X]\n", acpGetMsgName(ACP_HEADER_SIZE, buffer), localId);

		if (acpGetMsgSidlStatus(ACP_HEADER_SIZE, buffer, &sidlStatus) < 0) {
			ret = -ACP_ERR_INVALID_HEADER;
			break;
		}
		if (sidlStatus != SIDL_STATUS_OK) {
			ret = -ACP_ERR_SIDL_FAILURE;
			break;
		}

		if (localId == ACP_SERVICE_PUSH_TYPE) {
			acpUpdateCtx(ctx, length+ACP_HEADER_SIZE, buffer);

			if (!acpIsConnected(ctx) && !isGetAvailableMsSent) {
				ACP_LOG_DEBUG(">>> Send '%s' request\n", "GetAvailableMs");
				if (acpInternalGetAvailableMsEnc(ctx, buffer, size, &msgSize) == 0) {
					if (acpSocketSend(sock, msgSize, buffer) < 0) {
						ret = -ACP_ERR_SOCK_ERROR;
						break;
					}
					isGetAvailableMsSent = true;
				} else {
					ret = -ACP_ERR_INTERNAL;
					break;
				}
			}
		}
		else if (localId == ACP_LID_InternalGetAvailableMs) {
			struct thpMsDescriptor* msArray = NULL;
			size_t msQty = 0;
			MacAddress_t mac = 0LL;
			if (acpInternalGetAvailableMsDec(ctx, buffer, length+ACP_HEADER_SIZE, arena, aSize, &msQty, &msArray) == 0) {
				if (msQty > 0) {
					mac = msArray[0].mac;
				}
				acpInternalGetAvailableMsFree(msArray, msQty);
			} else {
				ret = -ACP_ERR_INTERNAL;
				break;
			}

			ACP_LOG_DEBUG("<<< mac=%08X msQty=%d\n", (int)mac, (int)msQty);

			if (mac) {
				if (!acpIsConnected(ctx) && !isConnectToMsSent) {
					// Send RequestNtf ONEWAY before ConnectToMs
					// in order to register to UE notifications
					{
						size_t ntfQty = 0;
						struct thpNtfName *ntfArray = NULL;
						size_t ntfAllocated = 0;

						for (unsigned int i = 0; i < ctx->userIdMapSize; i++) {
							const struct acpIdMapService *service = &(acpIdMap[ctx->userIdMap[i].id_index]);
							if (service->kind == 0) {
								ntfAllocated++;
							}
						}

						ntfArray = (struct thpNtfName*) acpMalloc(sizeof(struct thpNtfName) * ntfAllocated);

						for (unsigned int i = 0; i < ctx->userIdMapSize; i++) {
							const struct acpIdMapService *service = &(acpIdMap[ctx->userIdMap[i].id_index]);
							if (service->kind == 0) {
								const char *itfname = acpCtxGetItfNameFrom_localId(service->local_id);
								if (itfname) {
									SIDL_ASSERT(strlen(itfname) < strlen(service->name));
									ntfArray[ntfQty++].serviceName = (char*)service->name + strlen(itfname);
								}
								else {
									ntfArray[ntfQty++].serviceName = (char*)service->name;
								}
							}
						}

						SIDL_ASSERT(ntfQty == ntfAllocated);

						ACP_LOG_DEBUG(">>> Send '%s' request\n", "RequestNtf");
						if (acpInternalRequestNtfEnc(ctx, buffer, size, &msgSize, ntfQty, ntfArray) == 0) {
							if (acpSocketSend(sock, msgSize, buffer) < 0) {
								ret = -ACP_ERR_SOCK_ERROR;
							}
						} else {
							ret = -ACP_ERR_INTERNAL;
						}

						acpFree(ntfArray);
						if (ret < 0) {
							break;
						}
					}

					ACP_LOG_DEBUG(">>> Send '%s' request\n", "ConnectToMs");
					if (acpInternalConnectToMsEnc(ctx, buffer, size, &msgSize, mac) == 0) {
						if (acpSocketSend(sock, msgSize, buffer) < 0) {
							ret = -ACP_ERR_SOCK_ERROR;
							break;
						}
						isConnectToMsSent = true;
					} else {
						ret = -ACP_ERR_INTERNAL;
						break;
					}
				}
			} else {
				ret = -ACP_ERR_MS_UNAVAILABLE;
				break;
			}
		}
		else if (localId == ACP_LID_InternalConnectToMs) {
			uint32_t sHandle = 0;
			if (acpInternalConnectToMsDec(ctx, buffer, length + ACP_HEADER_SIZE, &sHandle) == 0) {
				ACP_LOG_DEBUG("<<< sHandle=%08X\n", sHandle);
				if (sHandle) {
					ctx->sHandle = (void*)(uintptr_t)sHandle;

					// ========================================
					// Connection to the device is completed,
					// any servie usage now is possible
					// ========================================
					break;
				}
			} else {
				ret = -ACP_ERR_INTERNAL;
				break;
			}
		}
		else {
			// Unexpected message
		}
	}
#endif

	acpFree(buffer);
	acpFree(arena);

	return ret;
}
#endif

// Adds peer to the server context.
static void acpAddPeer(struct acpCtx* ctx, int peer)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == -1) {
			ctx->peers[i] = peer;
			ctx->peersSize++;
			return;
		}
	}
	SIDL_ASSERT(0);
}

// Removes peer from the server context.
static void acpRemovePeer(struct acpCtx* ctx, int peer)
{
	for (int i = 0; i < ACP_MAX_PEER_QTY; i++) {
		if (ctx->peers[i] == peer) {
			ctx->peers[i] = -1;
			ctx->peersSize--;
			return;
		}
	}
}

// ___________________________ / Interface // ___________________________

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

	SIDL_ASSERT(alloc && release);

	_alloc = alloc;

	_release = release;

	_socketTimeout = socketTimeout;

	acpCtxInit();

	_initialized = true;
	ACP_DEBUG_EXIT_TRACE_LOG("TRUE");
}

int acpUpdateCtx(acpCtx_t ctx, size_t size, const unsigned char* buffer)
{
	ACP_DEBUG_ENTER_TRACE_LOG();

	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	if (size < ACP_HEADER_SIZE) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_INVALID_HEADER");
		return -ACP_ERR_INVALID_HEADER;
	}

	acpProcessPushMsg(ctx, size, buffer);

	ACP_DEBUG_EXIT_TRACE_LOG(NULL);
	return 0;
}

int acpCreateCtx(acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);

	*ctx = acpTakeCtx();
	if (!*ctx) {
		return -ACP_ERR_INVALID_CTX;
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

	struct acpCtx* c = ACP_CTX_CAST(ctx);
	return (c->sHandle != NULL);
}

int acpGetMsgSidlStatus(size_t size, const unsigned char* buffer, SidlStatus* sidlStatus)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	if (sidlStatus) {
		*sidlStatus = (SidlStatus)(uint32_t)((buffer[12] << 24) | (buffer[13] << 16) | (buffer[14] << 8) | (buffer[15]));
	}

	return 0;
}

int acpGetMsgLength(size_t size, unsigned char* buffer)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	int length = (int)((buffer[2] << 8) | buffer[3]);

	return length;
}

int acpGetMsgId(acpCtx_t ctx, size_t size, const unsigned char* buffer)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	int type = (int)((buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | (buffer[7]));

	if (type == (int)ACP_SERVICE_PUSH_TYPE) {
		acpUpdateCtx(ctx, size, buffer);
		return type;
	}

#if 0 // TODO
	if (type == ACP_LID_InternalConnectToMs) {
		SidlStatus sidlStatus;
		if (acpGetMsgSidlStatus(ACP_HEADER_SIZE, buffer, &sidlStatus) < 0) {
			return -ACP_ERR_INVALID_HEADER;
		}
		if (sidlStatus == SIDL_STATUS_OK) {
			uint32_t sHandle = 0;
			if (acpInternalConnectToMsDec(ctx, buffer, size, &sHandle) == 0) {
				if (sHandle) {
					ACP_CTX_CAST(ctx)->sHandle = (void*)(uintptr_t)sHandle;
				}
			}
		}
	}
#endif

	return acpCtxGetMsgUserId(ACP_CTX_CAST(ctx), type);
}

int acpGetMsgLocalId(size_t size, const unsigned char* buffer, enum acpMsgLocalId* localMsgId)
{
	if (size < ACP_HEADER_SIZE) {
		return -ACP_ERR_INVALID_HEADER;
	}

	int type = (int)((buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | (buffer[7]));

	if (localMsgId) {
		*localMsgId = (enum acpMsgLocalId)type;
	}

	if (type == (int)ACP_SERVICE_PUSH_TYPE) {
		return type;
	}

	return 0;
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

	int err, sz, length, userId;
	unsigned int handle;
	SidlStatus sidlStatus;

	int sock = ACP_CTX_CAST(ctx)->sock;
	size_t peersSize = 0;
	int* peers = NULL;

	bool disconnected = false;

	if (ACP_CTX_CAST(ctx)->isServer) {
		peersSize = ACP_CTX_CAST(ctx)->peersSize;
		peers = ACP_CTX_CAST(ctx)->peers;
	}

	err = acpSocketSelectMulti(&sock, _socketTimeout, peersSize, peers);
	if (err == 0) {
#ifdef ACP_DEBUG_TRACE_FLOOD
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
#endif
		return 0;
	}
	if (err < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "acpSocketSelectMulti failed");
		return -ACP_ERR_SOCK_ERROR;
	}

	if (ACP_CTX_CAST(ctx)->isServer) {
		if (ACP_CTX_CAST(ctx)->sock == sock) {
			int peerSock = acpSocketAccept(sock);
			if (peerSock < 0) {
				ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "acpSocketAccept failed");
				return -ACP_ERR_SOCK_ERROR;
			}
			acpAddPeer(ACP_CTX_CAST(ctx), peerSock);
			ACP_DEBUG_CLOG(ctx, "Peer connected");
			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
			return 0;
		}

		ACP_CTX_CAST(ctx)->lastPeer = sock;
	}

	// Check decode buffer is enough to read message header
	if (ACP_HEADER_SIZE > *size) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SMALL_BUFFER");
		return -ACP_ERR_SMALL_BUFFER;
	}

	// Read SIDL message header
	sz = acpSocketReceive(sock, ACP_HEADER_SIZE, buffer, _socketTimeout, (ACP_CTX_CAST(ctx)->isServer ? &disconnected : NULL));
	if (sz == 0) {
		if (ACP_CTX_CAST(ctx)->isServer && disconnected) {
			acpRemovePeer(ACP_CTX_CAST(ctx), sock);
			acpSocketClose(sock);
			ACP_DEBUG_CLOG(ctx, "Peer disconnected");
		}
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
		return 0;
	}
	if (sz < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SOCK_ERROR");
		return -ACP_ERR_SOCK_ERROR;
	}
	if (sz != ACP_HEADER_SIZE) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SOCK_TIMEOUT");
		return -ACP_ERR_SOCK_TIMEOUT;
	}

	// Get payload length
	length = acpGetMsgLength(ACP_HEADER_SIZE, buffer);
	if (length < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_HEADER");
		return -ACP_ERR_INVALID_HEADER;
	}

	// Check decode buffer is enough to read received message
	if (length > (int)(*size - ACP_HEADER_SIZE)) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SMALL_BUFFER");
		return -ACP_ERR_SMALL_BUFFER;
	}

	if (length > 0) {
		// Read ACP message payload
		sz = acpSocketReceive(sock, length, &buffer[ACP_HEADER_SIZE], _socketTimeout, NULL);
		if (sz < 0) {
			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SOCK_ERROR");
			return -ACP_ERR_SOCK_ERROR;
		}
		if (sz != length) {
			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SOCK_TIMEOUT");
			return -ACP_ERR_SOCK_TIMEOUT;
		}
	}

	if (acpGetMsgSidlStatus(ACP_HEADER_SIZE, buffer, &sidlStatus) < 0) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_HEADER");
		return -ACP_ERR_INVALID_HEADER;
	}
	if (sidlStatus != SIDL_STATUS_OK) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_SIDL_FAILURE");
		return -ACP_ERR_SIDL_FAILURE;
	}

	// Get message ID
	userId = acpGetMsgId(ctx, length + ACP_HEADER_SIZE, buffer);

	if (userId == (int)ACP_SERVICE_PUSH_TYPE) {
		userId = 0;
	}

	handle = (unsigned int)((buffer[8] << 24) | (buffer[9] << 16) | (buffer[10] << 8) | (buffer[11]));

	if (handle == ACP_KEEPALIVE_HANDLE) {
		userId = 0;
	}

	*size = length + ACP_HEADER_SIZE;
	int type = (int)((buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | (buffer[7]));
	ACP_DEBUG_PREFIX_CLOG(ctx, "Receiving message '%s' (userId: %d, type: 0x%x)\r\n", acpGetMsgName(*size, buffer), userId, type);
	if (userId >= 0) {
#ifdef ACP_DEBUG
		adbgMsgLog(ctx, ADBG_MSG_LOG_RECV_DIR, *size, buffer);
#endif
	}
	ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
	return userId;
}

int acpSendMsg(acpCtx_t ctx, size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);

	ACP_DEBUG_ENTER_TRACE_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	int sock = ACP_CTX_CAST(ctx)->sock;
	if (ACP_CTX_CAST(ctx)->isServer) {
		sock = ACP_CTX_CAST(ctx)->lastPeer;
		if (sock == -1) {
			ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INVALID_CTX");
			return -ACP_ERR_INVALID_CTX;
		}
	}

	ACP_DEBUG_PREFIX_CLOG(ctx, "Sending message '%s'", acpGetMsgName(size, buffer));
#ifdef ACP_DEBUG
	adbgMsgLog(ctx, ADBG_MSG_LOG_SEND_DIR, size, buffer);
#endif

	if (acpSocketSend(sock, size, buffer) != (int)size) {
		ACP_DEBUG_EXIT_TRACE_CLOG(ctx, "ACP_ERR_INTERNAL");
		return -ACP_ERR_INTERNAL;
	}

	/*
	if (ACP_CTX_CAST(ctx)->isServer) {
		ACP_CTX_CAST(ctx)->lastPeer = -1;
	}
	*/

	ACP_DEBUG_EXIT_TRACE_CLOG(ctx, NULL);
	return 0;
}

int acpClientInit(acpCtx_t ctx, IpAddress_t ipaddr, int port, size_t aSize)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	int sock = acpSocketConnect(ipaddr, port);
	if (sock < 0) {
		ACP_DEBUG_EXIT_CLOG(ctx, "ACP_ERR_SOCKCONN_ABORTED");
		return -ACP_ERR_SOCKCONN_ABORTED;
	}

#if 0
	int ret = acpConnectToMs(ACP_CTX_CAST(ctx), sock);
	if (ret < 0) {
		acpSocketClose(sock);
		return ret;
	}

	// Should not happen
	if (!acpIsConnected(ctx)) {
		acpSocketClose(sock);
		return -ACP_ERR_NOT_CONNECTED;
	}
#endif

	ACP_CTX_CAST(ctx)->sock = sock;
	ACP_CTX_CAST(ctx)->isServer = false;
	ACP_CTX_CAST(ctx)->aSize = aSize;
	ACP_CTX_CAST(ctx)->arena = (unsigned char*)acpMalloc(aSize);
	SIDL_ASSERT(ACP_CTX_CAST(ctx)->arena);

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return 0;
}

int acpClientInitWithCtx(IpAddress_t ipaddr, int port, const struct acpMsgTable* msgTable, size_t aSize, acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);

	ACP_DEBUG_ENTER_LOG();

	acpCtx_t _ctx = NULL;
	if (acpCreateCtx(&_ctx) < 0) {
		ACP_DEBUG_EXIT_LOG("ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	if (msgTable) {
		while (msgTable->name) {
			int err = acpSetMsgId(_ctx, msgTable->name, msgTable->userId);
			if (err < 0) {
				acpDeleteCtx(_ctx);
				ACP_DEBUG_EXIT_LOG("acpSetMsgId failed");
				return err;
			}
			msgTable++;
		}
	}

	int ret = acpClientInit(_ctx, ipaddr, port, aSize);
	if (ret < 0) {
		acpDeleteCtx(_ctx);
		ACP_DEBUG_EXIT_LOG("acpClientInit failed");
		return ret;
	}

	if (acp_printPrettyMessages) {
		adbgSetPrintLogFormat(_ctx, true);
	}
	*ctx = _ctx;

	ACP_DEBUG_EXIT_LOG(NULL);
	return 0;
}

int acpServerInit(acpCtx_t ctx, IpAddress_t ipaddr, int port, size_t aSize)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
		return -ACP_ERR_INVALID_CTX;
	}

	int sock = acpSocketListen(ipaddr, port);
	if (sock < 0) {
		ACP_DEBUG_EXIT_CLOG(ctx, "ACP_ERR_SOCKCONN_ABORTED");
		return -ACP_ERR_SOCKCONN_ABORTED;
	}

#if 0
	int ret = acpConnectToMs(ACP_CTX_CAST(ctx), sock);
	if (ret < 0) {
		acpSocketClose(sock);
		return ret;
	}

	// Should not happen
	if (!acpIsConnected(ctx)) {
		acpSocketClose(sock);
		return -ACP_ERR_NOT_CONNECTED;
	}
#endif

	ACP_CTX_CAST(ctx)->sock = sock;
	ACP_CTX_CAST(ctx)->isServer = true;
	ACP_CTX_CAST(ctx)->aSize = aSize;
	ACP_CTX_CAST(ctx)->arena = (unsigned char*)acpMalloc(aSize);
	SIDL_ASSERT(ACP_CTX_CAST(ctx)->arena);

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return 0;
}

int acpServerInitWithCtx(IpAddress_t ipaddr, int port, const struct acpMsgTable* msgTable, size_t aSize, acpCtx_t* ctx)
{
	SIDL_ASSERT(ctx);

	ACP_DEBUG_ENTER_LOG();

	acpCtx_t _ctx = NULL;
	if (acpCreateCtx(&_ctx) < 0) {
		ACP_DEBUG_EXIT_LOG("ACP_ERR_INVALID_CTX");
		return -ACP_ERR_INVALID_CTX;
	}

	if (msgTable) {
		while (msgTable->name) {
			int err = acpSetMsgId(_ctx, msgTable->name, msgTable->userId);
			if (err < 0) {
				acpDeleteCtx(_ctx);
				ACP_DEBUG_EXIT_LOG("acpSetMsgId failed");
				return err;
			}
			msgTable++;
		}
	}

	int ret = acpServerInit(_ctx, ipaddr, port, aSize);
	if (ret < 0) {
		acpDeleteCtx(_ctx);
		ACP_DEBUG_EXIT_LOG("acpServerInit failed");
		return ret;
	}

	*ctx = _ctx;

	ACP_DEBUG_EXIT_LOG(NULL);
	return 0;
}

int acpClose2(acpCtx_t ctx)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	if (!acpCtxIsValid(ctx)) {
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
	int type = (int)((buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | (buffer[7]));
	return acpCtxGetMsgNameFromId(type);
}
