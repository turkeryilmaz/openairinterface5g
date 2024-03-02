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

// Internal includes
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

/** Add ACP header in the begining of the buffer. */
void acpBuildHeader(acpCtx_t ctx, int type, size_t size, unsigned char* buffer);

/** Get ACP header payload length of the service. */
int acpGetServicePayloadLength(size_t size, const unsigned char* buffer);

/** Get ACP header type of the service. */
int acpGetServiceType(size_t size, const unsigned char* buffer);

/** Get ACP header handle of the service. */
int acpGetServiceHandle(size_t size, const unsigned char* buffer);

/** Get ACP header SIDL status. */
int acpGetServiceStatus(size_t size, const unsigned char* buffer);

/** Update ACP header SIDL status .*/
void acpSetServiceStatus(size_t size, unsigned char* buffer, int sidlStatus);

SIDL_END_C_INTERFACE
