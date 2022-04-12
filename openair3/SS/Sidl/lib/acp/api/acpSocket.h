/*
 *****************************************************************
 *
 * Module : Asynchronous Communication Protocol
 * Purpose : Socket functions
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
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

/** Connects to server.
 *
 * @param[in]  ipaddr Server ip address
 * @param[in]  port Server ip port
 * @return   socket fd, or negative number on failure
 */
int acpSocketConnect(IpAddress_t ipaddr, int port);

/** Opens listening socket.
 *
 * @param[in]  ipaddr Server ip address
 * @param[in]  port Server ip port
 * @return   socket fd, or negative number on failure
 */
int acpSocketListen(IpAddress_t ipaddr, int port);

int acpSocketSelect(int sock, MSec_t socketTimeout);

int acpSocketSelectMulti(int* sock, MSec_t socketTimeout, size_t peersSize, int* peers);

int acpSocketAccept(int sock);

/** Sends a message on socket.
 *
 * @param[in]  sock Socket fd
 * @param[in]  size Buffer size
 * @param[in]  buffer Buffer to send
 * @return   number of bytes sent, or negative number on failure
 */
int acpSocketSend(int sock, size_t size, const unsigned char* buffer);

/** Receives a message on socket.
 *
 * @param[in]  sock Socket fd
 * @param[in]  size Buffer size
 * @param[in]  buffer Buffer where to put received data
 * @param[in]  socketTimeout Timeout on socket reception, with value in milli-seconds
 * @param[out]  disconnected Check if other side performed an orderly shutdown
 * @return   number of bytes received, or negative number on failure
 */
int acpSocketReceive(int sock, size_t size, unsigned char* buffer, MSec_t socketTimeout, bool* disconnected);

/** Closes the socket.
 *
 * @param[in]  sock Socket fd
 * @return   0 on success, or negative number on failure
 */
int acpSocketClose(int sock);

/** Converts ip address from string to IpAddress_t.
 *
 * @param[in]  ipstr IP address in string
 * @param[out]  ipaddr Converted IP address
 * @return   true on success, false otherwise
 */
bool acpConvertIp(const char* ipstr, IpAddress_t* ipaddr);

SIDL_END_C_INTERFACE
