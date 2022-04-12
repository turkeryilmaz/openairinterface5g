/*
 *****************************************************************
 *
 * Module : Asynchronous Communication Protocol
 * Purpose: Context
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
	/** Arena to devode received message. */
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
	/** Peers socket descriptors size. */
	size_t peersSize;
	/** Last peer socket descriptor who sent the message. */
	int lastPeer;
	/** User ID map. */
	struct acpUserService* userIdMap;
	/** User ID map max size. */
	size_t userIdMapMaxSize;
	/** ID map size. */
	size_t userIdMapSize;

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

/** Gets message kind from id (for debug). */
int acpCtxGetMsgKindFromId(int id);

/** Gets sidl name from local id. */
const char* acpCtxGetItfNameFrom_localId(int id);

SIDL_END_C_INTERFACE
