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

SIDL_BEGIN_C_INTERFACE

/** Connects to server.
 *
 * @param[in]  host Server ip address / unixsocket file
 * @param[in]  port Server ip port
 * @return   socket fd, or negative number on failure
 */
int acpSocketConnect(const char* host, int port);

/** Opens listening socket.
 *
 * @param[in]  host Server ip address / unixsocket file
 * @param[in]  port Server ip port
 * @return   socket fd, or negative number on failure
 */
int acpSocketListen(const char* host, int port);

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
