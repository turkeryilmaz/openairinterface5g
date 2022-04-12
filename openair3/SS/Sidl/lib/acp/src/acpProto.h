/*
 *****************************************************************
 *
 * Module : Asynchronous Communication Protocol
 * Purpose : Protocol
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

SIDL_BEGIN_C_INTERFACE

#pragma pack(push, 1)
struct acpWklmService {
	unsigned int id;	   // Service ID
	/*const char name[];*/ // NULL terminated service name
};
#pragma pack(pop)

#pragma pack(push, 1)
struct acpWklmServicePushMessage {
	unsigned int serviceQty;			  // Number of pushed services
	/*struct dcxWklmService services[];*/ // Array of services
};
#pragma pack(pop)

/** Process push message and resolve ids. */
void acpProcessPushMsg(acpCtx_t ctx, size_t size, const unsigned char* buffer);

/** Add ACP header in the begining of the buffer. */
void acpBuildHeader(acpCtx_t ctx, unsigned int type, size_t size, unsigned char* buffer);

SIDL_END_C_INTERFACE
