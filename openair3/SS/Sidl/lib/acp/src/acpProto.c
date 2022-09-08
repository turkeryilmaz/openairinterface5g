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

#include <stdint.h>
#include <arpa/inet.h>

// Internal includes
#include "acpProto.h"
#include "acpCtx.h"

void acpBuildHeader(acpCtx_t ctx, int type, size_t size, unsigned char* buffer)
{
	SIDL_ASSERT(ctx);
	SIDL_ASSERT(buffer);
	SIDL_ASSERT(size >= ACP_HEADER_SIZE);

	struct acpCtx* c = ACP_CTX_CAST(ctx);

	int handle = (int)(uintptr_t)(c->sHandle ? c->sHandle : c->handle);
	int sidlStatus = 0;

	int payloadSize = (int)size - ACP_HEADER_SIZE; // payload size

	buffer[0] = 0x01;
	buffer[1] = 0x06;
	buffer[2] = (payloadSize >> 8) & 0xFF;
	buffer[3] = payloadSize & 0xFF;
	buffer[4] = (type >> 24) & 0xFF;
	buffer[5] = (type >> 16) & 0xFF;
	buffer[6] = (type >> 8) & 0xFF;
	buffer[7] = type & 0xFF;
	buffer[8] = (handle >> 24) & 0xFF;
	buffer[9] = (handle >> 16) & 0xFF;
	buffer[10] = (handle >> 8) & 0xFF;
	buffer[11] = handle & 0xFF;
	buffer[12] = (sidlStatus >> 24) & 0xFF;
	buffer[13] = (sidlStatus >> 16) & 0xFF;
	buffer[14] = (sidlStatus >> 8) & 0xFF;
	buffer[15] = sidlStatus & 0xFF;
}

int acpGetServicePayloadLength(size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);
	SIDL_ASSERT(size >= ACP_HEADER_SIZE);

	return (int)((buffer[2] << 8) | buffer[3]);
}

int acpGetServiceType(size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);
	SIDL_ASSERT(size >= ACP_HEADER_SIZE);

	return (int)((buffer[4] << 24) | (buffer[5] << 16) | (buffer[6] << 8) | (buffer[7]));
}

int acpGetServiceHandle(size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);
	SIDL_ASSERT(size >= ACP_HEADER_SIZE);

	return (int)((buffer[8] << 24) | (buffer[9] << 16) | (buffer[10] << 8) | (buffer[11]));
}

int acpGetServiceStatus(size_t size, const unsigned char* buffer)
{
	SIDL_ASSERT(buffer);
	SIDL_ASSERT(size >= ACP_HEADER_SIZE);

	return (int)((buffer[12] << 24) | (buffer[13] << 16) | (buffer[14] << 8) | (buffer[15]));
}

void acpSetServiceStatus(size_t size, unsigned char* buffer, int sidlStatus)
{
	SIDL_ASSERT(buffer);
	SIDL_ASSERT(size >= ACP_HEADER_SIZE);

	buffer[12] = (sidlStatus >> 24) & 0xFF;
	buffer[13] = (sidlStatus >> 16) & 0xFF;
	buffer[14] = (sidlStatus >> 8) & 0xFF;
	buffer[15] = sidlStatus & 0xFF;
}
