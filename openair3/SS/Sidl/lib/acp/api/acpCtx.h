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

#pragma once

// System includes
#include <stdbool.h>

// Internal includes
#include "acp.h"
#include "acpIdMap.h"

SIDL_BEGIN_C_INTERFACE

#define ACP_CTX_CAST(pTR) ((struct acpCtx*)(pTR))

/** Defines user ID. */
struct acpUserService {
	/** User ID */
	int user_id;
	/** Offset in acpIdMap table */
	int id_index;
};

/** Defines ACP Context. */
struct acpCtx {
	/** Context pointer. */
	acpCtx_t ptr;
	/** ACP client description. */
	char* desc;
	/** Arena to decode received message. */
	unsigned char* arena;
	/** Arena size to decode received message. */
	size_t aSize;
	/** If context is server side. */
	bool isServer;
	/** ACP handle. */
	void* handle;
	/** ACP server handle. */
	void* sHandle;
	/** Socket descriptor (on client side, or master socket on server side). */
	int sock;
	/** Peers socket descriptors on server side. */
	int peers[ACP_MAX_PEER_QTY];
	/** Peers socket handshake state on server side. */
	int peersHandshaked[ACP_MAX_PEER_QTY];
	/** Peers registered services. */
	int* peersServices[ACP_MAX_PEER_QTY];
	/** Peers registered services size. */
	size_t peersServicesSize[ACP_MAX_PEER_QTY];
	/** Peers size. */
	size_t peersSize;
	/** Last peer socket descriptor who sent the message. */
	int lastPeer;
	/** Last message type. */
	int lastType;
	/** Last message kind. */
	int lastKind;
	/** User ID map. */
	struct acpUserService* userIdMap;
	/** User ID map max size. */
	size_t userIdMapMaxSize;
	/** ID map size. */
	size_t userIdMapSize;
	/** Internal temporary buffer. */
	unsigned char* tmpBuf;
	/** Internal temporary buffer size. */
	size_t tmpBufSize;
	/** Internal opaque buffer. */
	unsigned char* opaqueBuf;
	/** Internal opaque buffer size. */
	size_t opaqueBufSize;

	/** Debug logger callback. */
	void (*logger)(const char*);
	/** Debug logger buffer. */
	unsigned char* logBuf;
	/** Debug logger buffer max size. */
	size_t logBufMaxSize;
	/** Debug logger buffer size. */
	size_t logBufSize;
	/** Debug logger use formatting. */
	bool logFormat;
	/** Debug logger formatting buffer. */
	unsigned char* logFormatBuf;
	/** Debug logger formatting buffer max size. */
	size_t logFormatBufMaxSize;
};

/** Init. */
void acpCtxInit(void);

/** Returns new available context. */
acpCtx_t acpTakeCtx(void);

/** Makes the context available. */
void acpGiveCtx(acpCtx_t ctx);

/** Gets message user id. */
int acpCtxGetMsgUserId(struct acpCtx* ctx, unsigned int type);

/** Sets message user id. */
int acpCtxSetMsgUserId(struct acpCtx* ctx, const char* name, int userId);

/** Sets message user id. */
int acpCtxSetMsgUserIdById(struct acpCtx* ctx, enum acpMsgLocalId lid, int userId);

/** Checks if the context is valid. */
bool acpCtxIsValid(acpCtx_t ctx);

/** Resolves message id. */
int acpCtxResolveId(int id, const char* name);

/** Gets message name from id (for debug). */
const char* acpCtxGetMsgNameFromId(int id);

/** Gets message name from id (strict case). */
const char* acpCtxGetMsgNameFromIdStrict(int id);

/** Gets message kind from id (for debug). */
int acpCtxGetMsgKindFromId(int id);

/** Gets sidl name from local id. */
const char* acpCtxGetItfNameFrom_localId(int id);

/** Gets message kind from name */
int acpCtxGetMsgKindFromName(const char* name);

/** Sets ACP description. */
int acpCtxSetDescription(struct acpCtx* ctx, const char* desc);

/** Allocates temporary buffer. */
void acpCtxAllocateTmpBuf(struct acpCtx* ctx, size_t size);

/** Returns peer number from the socket. */
int acpCtxGetPeerNum(struct acpCtx* ctx, int peer);

/** Checks whether peer handshake is performed. */
bool acpCtxPeerHandshaked(struct acpCtx* ctx, int peer);

/** Checks whether any peer is handshaked. */
bool acpCtxAnyPeerHandshaked(struct acpCtx* ctx);

/** Sets peer to have been handshaked. */
void acpCtxPeerSetHandshaked(struct acpCtx* ctx, int peer, int flag);

/** Adds peer to the server context. */
void acpCtxAddPeer(struct acpCtx* ctx, int peer);

/** Removes peer from the server context. */
void acpCtxRemovePeer(struct acpCtx* ctx, int peer);

/** Checks whether a peer registered service. */
bool acpCtxPeerRespondsToService(struct acpCtx* ctx, int peer, int type);

/** Checks whether a peer registered service (check by peer number). */
bool acpCtxPeerNumRespondsToService(struct acpCtx* ctx, int peerNum, int type);

SIDL_END_C_INTERFACE
