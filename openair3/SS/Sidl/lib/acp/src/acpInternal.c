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
#include "acpInternal.h"
#include "acpProto.h"
#include "acpCtx.h"
#include "acpSocket.h"
#include "adbg.h"
#include "adbgMsg.h"
#include "acpVer.h"

/** Sets ACP version to the handshake request/confirm. */
static inline void acpHandshakeSetVersion(struct AcpHandshakeVersion* version)
{
	memset(version->release, 0, sizeof(version->release));
	snprintf(version->release, sizeof(version->release) - 1, "%s", ACP_VERSION);
	memset(version->checksum, 0, sizeof(version->checksum));
	snprintf(version->checksum, sizeof(version->checksum) - 1, "%s", ACP_VERSION_CKSM);
}

/** Sets ACP server confirmed services to the handshake confirm. */
static inline void acpHandshakeSetServicesFromClient(struct acpCtx* ctx, int peerNum, acpHandshakeCnfServices* services, unsigned char* tmpBuf, size_t tmpBufSize)
{
	SIDL_ASSERT((ctx->peersServicesSize[peerNum] * sizeof(struct AcpHandshakeService)) <= tmpBufSize);
	services->d = ctx->peersServicesSize[peerNum];
	services->v = (struct AcpHandshakeService*)tmpBuf;
	for (size_t i = 0; i < ctx->peersServicesSize[peerNum]; i++) {
		struct AcpHandshakeService* service = &services->v[i];
		unsigned int id = ctx->peersServices[peerNum][i];
		const char* name = acpCtxGetMsgNameFromIdStrict(id);
		service->id = (uint32_t)id;
		service->name = (char*)name;
	}
}

/** Sets ACP client services to the handshake request. */
static inline void acpHandshakeSetServicesToServer(struct acpCtx* ctx, acpHandshakeReqServices* services, unsigned char* tmpBuf, size_t tmpBufSize)
{
	SIDL_ASSERT((ctx->userIdMapSize * sizeof(struct AcpHandshakeService)) <= tmpBufSize);
	services->d = ctx->userIdMapSize;
	services->v = (struct AcpHandshakeService*)tmpBuf;
	for (size_t i = 0; i < ctx->userIdMapSize; i++) {
		struct acpUserService* user = &ctx->userIdMap[i];
		struct AcpHandshakeService* service = &services->v[i];
		service->id = (uint32_t)acpIdMap[user->id_index].remote_id;
		service->name = (char*)acpIdMap[user->id_index].name;
	}
}

/** Registers ACP client services on the server. */
static inline void acpHandshakeRegisterServicesFromClient(struct acpCtx* ctx, int peerNum, const acpHandshakeReqServices* services)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	SIDL_ASSERT(!ctx->peersServices[peerNum]);
	ctx->peersServicesSize[peerNum] = 0;
	ctx->peersServices[peerNum] = (int*)acpMalloc(services->d * sizeof(int));
	SIDL_ASSERT(ctx->peersServices[peerNum]);

	for (size_t i = 0; i < services->d; i++) {
		unsigned int clientServiceId = services->v[i].id;
		const char* serviceName = services->v[i].name;
		unsigned int serverServiceId = 0;

		for (size_t j = 0; j < ctx->userIdMapSize; j++) {
			if (!strcmp(serviceName, acpIdMap[ctx->userIdMap[j].id_index].name)) {
				serverServiceId = acpIdMap[ctx->userIdMap[j].id_index].remote_id;
				break;
			}
		}

		if (serverServiceId) {
			ACP_DEBUG_CLOG(ctx, "Client requested service '%s' (0x%08X -> 0x%08X)", serviceName, clientServiceId, serverServiceId);
			if (clientServiceId != serverServiceId) {
				ACP_DEBUG_CLOG(ctx, "ERROR: client requested service '%s' has different ID", serviceName);
			}
			ctx->peersServices[peerNum][ctx->peersServicesSize[peerNum]++] = serverServiceId;
		} else {
			ACP_DEBUG_CLOG(ctx, "ERROR: client requested service '%s' which is not implemented on the server", serviceName);
		}
	}

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
}

/** Registers ACP server confirmed services on the client. */
static inline void acpHandshakeRegisterServicesToServer(struct acpCtx* ctx, const acpHandshakeCnfServices* services)
{
	ACP_DEBUG_ENTER_CLOG(ctx);

	for (size_t i = 0; i < services->d; i++) {
		unsigned int clientServiceId = 0;
		const char* serviceName = services->v[i].name;
		unsigned int serverServiceId = services->v[i].id;

		for (size_t j = 0; j < ctx->userIdMapSize; j++) {
			if (!strcmp(serviceName, acpIdMap[ctx->userIdMap[j].id_index].name)) {
				clientServiceId = acpIdMap[ctx->userIdMap[j].id_index].remote_id;
				break;
			}
		}

		if (clientServiceId) {
			ACP_DEBUG_CLOG(ctx, "Server confirmed service '%s' (0x%08X -> 0x%08X)", serviceName, clientServiceId, serverServiceId);
			if (clientServiceId != serverServiceId) {
				ACP_DEBUG_CLOG(ctx, "ERROR: server confirmed service '%s' has different ID", serviceName);
			}
			if (acpCtxResolveId(serverServiceId, serviceName) < 0) {
				ACP_DEBUG_CLOG(ctx, "ERROR: failed to resolve service '%s' (0x%08X))", serviceName, serverServiceId);
				SIDL_ASSERT(0);
			}
		} else {
			ACP_DEBUG_CLOG(ctx, "ERROR: server confirmed service '%s' which is not implemented on the client", serviceName);
			SIDL_ASSERT(0);
		}
	}

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
}

int acpHandleHandshakeFromClient(struct acpCtx* ctx, int sock)
{
	SIDL_ASSERT(ctx);
	ACP_DEBUG_ENTER_CLOG(ctx);

	int ret = 0;

	struct AcpHandshakeReq* hs_req;
	struct AcpHandshakeCnf hs_cnf;

	int peerNum = acpCtxGetPeerNum(ctx, sock);
	SIDL_ASSERT(peerNum != -1);

	/* Process received handshake req */

	ret = acpHandshakeProcessDecSrv(ctx, ctx->opaqueBuf, ctx->opaqueBufSize, &hs_req);
	if (ret < 0) {
		ACP_DEBUG_CLOG(ctx, "ERROR: acpHandshakeProcessDecSrv failed");
		ACP_DEBUG_EXIT_CLOG(ctx, NULL);
		return ret;
	}

	ACP_DEBUG_CLOG(ctx, "ACP version: client version [%s, cksm: %s], server version [%s, chksm: %s], description '%s'", hs_req->common.version.release, hs_req->common.version.checksum, ACP_VERSION, ACP_VERSION_CKSM, (hs_req->common.control.desc ?: "NULL"));

	acpHandshakeRegisterServicesFromClient(ctx, peerNum, &hs_req->request.services);

	bool acpVerMatched = true;
	if (strncmp(ACP_VERSION, hs_req->common.version.release, strlen(ACP_VERSION)) || strncmp(ACP_VERSION_CKSM, hs_req->common.version.checksum, strlen(ACP_VERSION_CKSM))) {
		acpVerMatched = false;
	}
	bool acpVerIgnore = !hs_req->common.control.productMode;

	bool acpServicesMatched = (hs_req->request.services.d == ctx->peersServicesSize[peerNum]);

	acpHandshakeProcessFreeSrv(hs_req);

	const size_t size = 16 * 1024;
	acpCtxAllocateTmpBuf(ctx, size + ctx->peersServicesSize[peerNum] * sizeof(struct AcpHandshakeService));

	/* Prepare handshake cnf */

	acpHandshakeSetVersion(&hs_cnf.common.version);
	hs_cnf.common.result = ACP_HANDSHAKE_OK;

	acpHandshakeSetServicesFromClient(ctx, peerNum, &hs_cnf.confirm.services, ctx->tmpBuf + size, ctx->peersServicesSize[peerNum] * sizeof(struct AcpHandshakeService));

	if (!acpVerMatched && !acpVerIgnore) {
		hs_cnf.common.result = ACP_HANDSHAKE_VERSION_ERROR;
	} else if (!acpServicesMatched) {
		hs_cnf.common.result = ACP_HANDSHAKE_SERVICE_ERROR;
	}

	ACP_DEBUG_CLOG(ctx, "ACP version: %s%s, services: %s", acpVerMatched ? "matched" : "not matched", (!acpVerMatched && acpVerIgnore) ? " (ignored due to disabled Product Mode)" : "", acpServicesMatched ? "matched" : "not matched");

	ctx->tmpBufSize = size;

	ret = acpHandshakeProcessEncSrv(ctx, ctx->tmpBuf, &ctx->tmpBufSize, &hs_cnf);
	if (ret < 0) {
		ACP_DEBUG_CLOG(ctx, "ERROR: acpHandshakeProcessEncSrv failed");
		acpCtxPeerSetHandshaked(ctx, sock, 0);
	} else {
		/* Send handshake cnf */
		ret = acpSendMsgInternal(sock, ctx->tmpBufSize, ctx->tmpBuf);
		if (ret < 0) {
			ACP_DEBUG_CLOG(ctx, "ERROR: acpSendMsgInternal failed");
			acpCtxPeerSetHandshaked(ctx, sock, 0);
		} else {
			ACP_DEBUG_CLOG(ctx, "Peer handshaked");
			acpCtxPeerSetHandshaked(ctx, sock, (acpVerMatched || acpVerIgnore) ? 1 : 0);
		}
	}

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return ret;
}

int acpHandleHandshakeToServer(struct acpCtx* ctx, int sock)
{
	SIDL_ASSERT(ctx);
	ACP_DEBUG_ENTER_CLOG(ctx);

	int ret = 0;

	struct AcpHandshakeReq hs_req;
	struct AcpHandshakeCnf* hs_cnf;

	const size_t size = 16 * 1024;
	acpCtxAllocateTmpBuf(ctx, size + ctx->userIdMapSize * sizeof(struct AcpHandshakeService));

	/* Prepare handshake req */

	acpHandshakeSetVersion(&hs_req.common.version);
	hs_req.common.control.productMode = acpGetProductMode();
	hs_req.common.control.desc = ctx->desc;

	acpHandshakeSetServicesToServer(ctx, &hs_req.request.services, ctx->tmpBuf + size, ctx->userIdMapSize * sizeof(struct AcpHandshakeService));

	ctx->tmpBufSize = size;

	ret = acpHandshakeProcessEncClt(ctx, ctx->tmpBuf, &ctx->tmpBufSize, &hs_req);
	if (ret < 0) {
		ACP_DEBUG_CLOG(ctx, "ERROR: acpHandshakeProcessEncClt failed");
	} else {
		int retry = 0;
		enum acpMsgLocalId localId;

		/* Send handshake req to the server */
		ret = acpSendMsgInternal(sock, ctx->tmpBufSize, ctx->tmpBuf);
		if (ret < 0) {
			ACP_DEBUG_EXIT_CLOG(ctx, "ERROR: acpSendMsgInternal failed");
			return ret;
		}

		ctx->tmpBufSize = size;

		/* Now wait for handshake cnf from the server */
		/* Loop */
		for (;;) {
			ret = acpRecvMsgInternal(sock, &ctx->tmpBufSize, ctx->tmpBuf);
			if (ret < 0) {
				ACP_DEBUG_CLOG(ctx, "ERROR: failed to receive ACP message, error=%d", ret);
				ret = -ACP_ERR_SOCK_ERROR;
				break;
			}

			/* Get local message ID */
			ret = acpGetMsgLocalId(ACP_HEADER_SIZE, ctx->tmpBuf, &localId);
			if (ret < 0) {
				ret = -ACP_ERR_INVALID_HEADER;
				break;
			}

			ACP_DEBUG_CLOG(ctx, "<<< Receive '%s' service message [localId=%08X]", acpGetMsgName(ACP_HEADER_SIZE, ctx->tmpBuf), localId);

			if (localId != ACP_MSG_ID(HandshakeProcess)) {
				/* Discard message other than handshake cnf */
				if (++retry == 10) {
					ret = -ACP_ERR_SOCK_TIMEOUT;
					break;
				}
				continue;
			}

			ret = acpHandshakeProcessDecClt(ctx, ctx->tmpBuf, ctx->tmpBufSize, &hs_cnf);
			if (ret < 0) {
				ACP_DEBUG_CLOG(ctx, "ERROR: acpHandshakeProcessDecClt failed");
				break;
			}

			ACP_DEBUG_CLOG(ctx, "ACP version: client version [%s, cksm: %s], server version [%s, chksm: %s]", ACP_VERSION, ACP_VERSION_CKSM, hs_cnf->common.version.release, hs_cnf->common.version.checksum);

			acpHandshakeRegisterServicesToServer(ctx, &hs_cnf->confirm.services);

			switch (hs_cnf->common.result) {
				case ACP_HANDSHAKE_OK:
					ACP_DEBUG_CLOG(ctx, "ACP handshake OK");
					break;
				case ACP_HANDSHAKE_VERSION_ERROR:
					ACP_DEBUG_CLOG(ctx, "ERROR: ACP version not matched");
					ret = -ACP_ERR_INVALID_VERSION;
					break;
				case ACP_HANDSHAKE_SERVICE_ERROR:
					ACP_DEBUG_CLOG(ctx, "ERROR: ACP service not confirmed");
					ret = -ACP_ERR_SERVICE_MISSING;
					break;
				default:
					ACP_DEBUG_CLOG(ctx, "ERROR: ACP handshake unknown error");
					ret = -ACP_ERR_INTERNAL;
					break;
			}

			acpHandshakeProcessFreeClt(hs_cnf);

			break;
		}
	}

	ACP_DEBUG_EXIT_CLOG(ctx, NULL);
	return ret;
}

int acpRecvMsgInternal(int sock, size_t* size, unsigned char* buffer)
{
	SIDL_ASSERT(size && buffer);
#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_ENTER_TRACE_LOG();
#endif

	int sz = 0, length = 0;
	bool disconnected = false;
	SidlStatus sidlStatus;

	/* Check decode buffer is enough to read message header */
	if (ACP_HEADER_SIZE > *size) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SMALL_BUFFER");
		return -ACP_ERR_SMALL_BUFFER;
	}

	/* Read SIDL message header */
	sz = acpSocketReceive(sock, ACP_HEADER_SIZE, buffer, acpGetSocketTimeout(), &disconnected);
	if (sz == 0) {
		if (disconnected) {
			ACP_DEBUG_EXIT_TRACE_LOG(NULL);
			return -ACP_PEER_DISCONNECTED;
		} else {
			ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SOCKCONN_ABORTED");
			return -ACP_ERR_SOCKCONN_ABORTED;
		}
	}
	if (sz < 0) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SOCK_ERROR");
		return -ACP_ERR_SOCK_ERROR;
	}
	if (sz != ACP_HEADER_SIZE) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SOCK_TIMEOUT");
		return -ACP_ERR_SOCK_TIMEOUT;
	}

	/* Get payload length */
	length = acpGetMsgLength(ACP_HEADER_SIZE, buffer);
	if (length < 0) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_INVALID_HEADER");
		return -ACP_ERR_INVALID_HEADER;
	}

	/* Check decode buffer is enough to read received message */
	if (length > (int)(*size - ACP_HEADER_SIZE)) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SMALL_BUFFER");
		return -ACP_ERR_SMALL_BUFFER;
	}

	if (length > 0) {
		/* Read ACP message payload */
		sz = acpSocketReceive(sock, length, &buffer[ACP_HEADER_SIZE], acpGetSocketTimeout(), NULL);
		if (sz < 0) {
			ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SOCK_ERROR");
			return -ACP_ERR_SOCK_ERROR;
		}
		if (sz != length) {
			ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SOCK_TIMEOUT");
			return -ACP_ERR_SOCK_TIMEOUT;
		}
	}

	*size = length + ACP_HEADER_SIZE;

	if (acpGetMsgSidlStatus(ACP_HEADER_SIZE, buffer, &sidlStatus) < 0) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_INVALID_HEADER");
		return -ACP_ERR_INVALID_HEADER;
	}
	if (sidlStatus != SIDL_STATUS_OK) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_SIDL_FAILURE");
		return -ACP_ERR_SIDL_FAILURE;
	}

#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_EXIT_TRACE_LOG(NULL);
#endif
	return 0;
}

int acpSendMsgInternal(int sock, size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);
#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_ENTER_TRACE_LOG();
#endif

	if (acpSocketSend(sock, size, buffer) != (int)size) {
		ACP_DEBUG_EXIT_TRACE_LOG("ACP_ERR_INTERNAL");
		return -ACP_ERR_INTERNAL;
	}

#ifdef ACP_DEBUG_TRACE_FLOOD
	ACP_DEBUG_EXIT_TRACE_LOG(NULL);
#endif
	return 0;
}
