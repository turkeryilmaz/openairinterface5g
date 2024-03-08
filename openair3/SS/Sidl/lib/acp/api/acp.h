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
#include "SidlCompiler.h"
#include "acpMsgIds.h"
#include "acpMem.h"
#include "acpSocket.h"

SIDL_BEGIN_C_INTERFACE

/** Defines MAX ACP context quantity. */
#define ACP_MAX_CTX_QTY 10

/** Defines MAX peers quantity. */
#define ACP_MAX_PEER_QTY 10

/** Defines ACP Context. */
typedef void* acpCtx_t;

/** Defines the API table for registration */
struct acpMsgTable {
	/** Service name */
	const char* name;
	/** User ID */
	int userId;
};

/** Defines ACP Header size. */
#define ACP_HEADER_SIZE 16

/** This message may be received when server sends KEEPALIVE. */
#define ACP_KEEPALIVE_HANDLE 0xAAAAAAAA

/** Default socket timeout (for acpInit), in milli-seconds. */
#define ACP_DEFAULT_TIMEOUT 10

/** Creates an ACP context.
 *
 * @param[out] ctx 	        ACP Context
 * @return   0 on success, or an error code on failure.
 */
int acpCreateCtx(acpCtx_t* ctx);

/** Deletes a given ACP context.
 *
 * @param[in]  ctx      ACP context
 * @return   0 on success, or an error code on failure.
 */
int acpDeleteCtx(acpCtx_t ctx);

/** Checks if the connection to the device is completed.
 *
 * @param[in]  ctx      ACP context
 * @return   true if connected, false otherwise
 */
bool acpIsConnected(acpCtx_t ctx);

/** Returns SIDL status from the response message header.
 *
 * @param[in]  size	It should be equal to ACP_HEADER_SIZE
 * @param[in]  buffer 	Input header message buffer
 * @param[out] sidlStatus 	SIDL status
 * @return   0 on success, or an error code on failure.
 */
int acpGetMsgSidlStatus(size_t size, const unsigned char* buffer, SidlStatus* sidlStatus);

/** Sets SIDL status to the response message header.
 *
 * @param[in]  size	It should be equal to ACP_HEADER_SIZE
 * @param[in/out]  buffer 	Input header message buffer
 * @param[in] sidlStatus 	SIDL status
 * @return   0 on success, or an error code on failure.
 */
int acpSetMsgSidlStatus(size_t size, unsigned char* buffer, SidlStatus sidlStatus);

/** Returns message length from header buffer.
 * This is used after the read of ACP header to get the remaining bytes quantity to read for the
 * current message.
 *
 * @param[in]  size	It should be equal to ACP_HEADER_SIZE
 * @param[in]  buffer 	Input header message buffer
 * @return   Message length or a negative error code
 */
int acpGetMsgLength(size_t size, const unsigned char* buffer);

/** Returns message ID.
 *
 * @param[in]  ctx      ACP context
 * @param[in]  size	Total message size (header + payload)
 * @param[in]  buffer   Total message buffer
 * @return   User id for the message, or a negative error code
 */
int acpGetMsgId(acpCtx_t ctx, size_t size, const unsigned char* buffer);

/** Returns message ID.
 *
 * @param[in]  size	It should be equal to ACP_HEADER_SIZE
 * @param[in]  buffer 	Input header message buffer
 * @param[out] localMsgId 	Local message id
 * @return   0 on success, or an error code on failure.
 */
int acpGetMsgLocalId(size_t size, const unsigned char* buffer, enum acpMsgLocalId* localMsgId);

/** Reads the next message on the context socket and returns the message ID.
 *
 * @param[in]  ctx      ACP context
 * @param[in,out]  size	Buffer size, and received size on output
 * @param[out]  buffer   Buffer used for received data
 * @return   User id for the message, 0 on no message (timeout), or a negative error code on failure
 */
int acpRecvMsg(acpCtx_t ctx, size_t* size, unsigned char* buffer);

/** Sends the message on the context socket.
 *
 * @param[in]  ctx      ACP context
 * @param[in]  size	Buffer size
 * @param[in]  buffer   Buffer used for sent data
 * @return   0 on success, or a negative error code on failure
 */
int acpSendMsg(acpCtx_t ctx, size_t size, const unsigned char* buffer);

/** Memory management callbacks
 * Note: NOT called for marshalling / demarshalling usage.
 */
typedef void* (*acpMalloc_t)(size_t size);
typedef void (*acpFree_t)(void* ptr);

/** Module initialization.
 * Shall be called once.
 * Alloc/release callbacks shall be both supplied to use in acpMalloc/acpFree, otherwise they default to stdlib malloc/free.
 * Internally acpMalloc/acpFree are used only in context creation and during connection to server.
 * The timeout argument specifies the block interval on socket operations, or default timeout if zero supplied.
 *
 * @param[in]  alloc		         Memory allocation callback
 * @param[in]  release 	             Memory release callback
 * @param[in]  socketTimeout		Timeout on socket, with value in milli-seconds
 */
void acpInit(acpMalloc_t alloc, acpFree_t release, MSec_t socketTimeout);

/** Allocates the context with registering to requested notifications.
 *
 * @param[in]  msgTable		Requested service responses/notifications to register, last element should be with name==NULL
 * @param[out]  ctx 	        ACP Context
 * @return   0 on success, or an error code on failure.
 */
int acpInitCtx(const struct acpMsgTable* msgTable, acpCtx_t* ctx);

/** Connects to server.
 * Context should be allocated with acpCreateCtx.
 *
 * @param[in]  ctx 	        ACP Context
 * @param[in]  ipaddr		Server ip address
 * @param[in]  port		Server TCP port
 * @param[in] aSize Arena size
 * @return   0 on success, or an error code on failure.
 */
int acpClientInit(acpCtx_t ctx, const char* host, int port, size_t aSize);

/** Connects to server.
 * Allocates the context with registering to requested notifications.
 *
 * @param[in]  host		Server ip address
 * @param[in]  port		Server TCP port
 * @param[in]  msgTable		Requested service responses/notifications to register, last element should be with name==NULL
 * @param[in] aSize Arena size
 * @param[out]  ctx 	        ACP Context
 * @return   0 on success, or an error code on failure.
 */
int acpClientInitWithCtx(const char* host, int port, const struct acpMsgTable* msgTable, size_t aSize, acpCtx_t* ctx);

/** Runs server.
 * Context should be allocated with acpCreateCtx.
 *
 * @param[in]  ctx 	        ACP Context
 * @param[in]  host		Server ip address / unixsocket pipe file
 * @param[in]  port		Server TCP port
 * @param[in] aSize Arena size
 * @return   0 on success, or an error code on failure.
 */
int acpServerInit(acpCtx_t ctx, const char* host, int port, size_t aSize);

/** Runs server.
 * Allocates the context with registering to requested notifications.
 *
 * @param[in]  host		Server ip address / unixsocket pipe path
 * @param[in]  port		Server TCP port
 * @param[in]  msgTable		Requested service responses/notifications to register, last element should be with name==NULL
 * @param[in] aSize Arena size
 * @param[out]  ctx 	        ACP Context
 * @return   0 on success, or an error code on failure.
 */
int acpServerInitWithCtx(const char* host, int port, const struct acpMsgTable* msgTable, size_t aSize, acpCtx_t* ctx);

/** Closes the socket from the context,
 * an ACP context is still valid after call.
 *
 * @param[in] ctx 	        ACP Context
 * @return   0 on success, or an error code on failure.
 */
int acpClose0(acpCtx_t ctx);

/** Closes the socket from the context,
 * and deletes a given ACP context.
 *
 * @param[in] ctx 	        ACP Context
 * @return   0 on success, or an error code on failure.
 */
int acpClose(acpCtx_t ctx);

/** Gets socket fd from the context.
 *
 * @param[in] ctx 	        ACP Context
 * @return   socket fd
 */
int acpGetSocketFd(acpCtx_t ctx);

/** Sets socket fd to the context.
 * Actually should not be used, since the socket fd is managed by ACP library.
 *
 * @param[in] ctx 	        ACP Context
 * @param[in] socketfd 	    Socket fd
 */
void acpSetSocketFd(acpCtx_t ctx, int socketfd);

/** Gets last peer socket fd from the context, who sent the message.
 *
 * @param[in] ctx 	        ACP Context
 * @return   peer socket fd
 */
int acpGetLastPeerSocketFd(acpCtx_t ctx);

/** Sets last peer socket fd to the context.
 * Actually should not be used, since the peer socket fd is managed by ACP library.
 *
 * @param[in] ctx 	        ACP Context
 * @param[in] socketfd 	    Socket fd
 */
void acpSetLastPeerSocketFd(acpCtx_t ctx, int socketfd);

/** Gets socket reception timeout, passed as parameter to acpInit.
 *
 * @return    socketTimeout with value in milli-seconds
 */
MSec_t acpGetSocketTimeout(void);

/** Registers the user id associated to a named service/notification.
 *
 * @param[in] ctx 	        ACP Context
 * @param[in] name 	        Service/Notification name
 * @param[in] userMsgId	        User message ID
 * @return   negative value for an unsupported name, and 0 on success
 */
int acpSetMsgId(acpCtx_t ctx, const char* name, int userMsgId);

/** Registers the user id associated to service/notification local ID.
 *
 * @param[in] ctx 	        ACP Context
 * @param[in] localMsgId 	Service/Notification local ID
 * @param[in] userMsgId	        User message ID
 * @return   negative value for an unsupported local ID, and 0 on success
 */
int acpSetMsgIdFromLocalId(acpCtx_t ctx, enum acpMsgLocalId localMsgId, int userMsgId);

/** Returns message name (for debug).
 *
 * @param[in]  size	It should be equal to ACP_HEADER_SIZE
 * @param[in]  buffer 	Input header message buffer
 * @return   Service name
 */
const char* acpGetMsgName(size_t size, const unsigned char* buffer);

/** Sets ACP description to identify client.
 * Works only if using after acpInitCtx.
 *
 * @param[in] ctx 	        ACP Context
 * @param[in] desc 	        Description
 * @return   0 on success, or an error code on failure.
 */
int acpSetDescription(acpCtx_t ctx, const char* desc);

/** Gets Product Mode value.
 *
 * @return    true if enabled, false if disabled
 */
bool acpGetProductMode(void);

/** Defines getter for ACP message ID. */
#define ACP_MSG_ID(mSG) (ACP_LID_##mSG)

/** Defines helper to bind message to user id. */
#define ACP_SET_MSG_ID(cTX, mSG, uID) acpSetMsgIdFromLocalId((cTX), ACP_MSG_ID(mSG), (uID))

// #define ACP_SET_MSG_ID(cTX, mSG, uID) acpSetMsgId((cTX), #mSG, (uID))

/** These IDs (negated, -ACP_PEER_CONNECTED, -ACP_PEER_DISCONNECTED) are returned by acpRecvMsg on peer connection establishment/ordered shutdown.
 * They should not appear on client side.
 */
#define ACP_PEER_CONNECTED (1)
#define ACP_PEER_DISCONNECTED (2)

/** Defines error codes. */
enum acpErrorCode {
	/** All errors below this are reserved for other codes. */
	ACP_ERR_RESERVED = 5,

	/** API called with invalid context. */
	ACP_ERR_INVALID_CTX,
	/** Invalid ACP header. */
	ACP_ERR_INVALID_HEADER,
	/** Socket connection failure. */
	ACP_ERR_SOCKCONN_ABORTED,
	/** Socket read/write failure. */
	ACP_ERR_SOCK_ERROR,
	/** Socket read/write timeout. */
	ACP_ERR_SOCK_TIMEOUT,
	/** Unknown service name. */
	ACP_ERR_UNKNOWN_SERVICE_NAME,
	/** Service not mapped to user id. */
	ACP_ERR_SERVICE_NOT_MAPPED,
	/** Not connected to server. */
	ACP_ERR_NOT_CONNECTED,
	/** Supplied buffer has not enough space. */
	ACP_ERR_SMALL_BUFFER,
	/** Other internal failures. */
	ACP_ERR_INTERNAL,
	/** SIDL service failure. */
	ACP_ERR_SIDL_FAILURE,
	/** ACP version not match */
	ACP_ERR_INVALID_VERSION,
	/** ACP service not found on server */
	ACP_ERR_SERVICE_MISSING,

	ACP_ERR_QTY
};

SIDL_END_C_INTERFACE
