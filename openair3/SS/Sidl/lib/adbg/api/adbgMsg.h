/*
 *****************************************************************
 *
 * Module : ACP Debug
 * Purpose: Debug logging
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

// Internal includes
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

enum adbgMsgLogDir {
	ADBG_MSG_LOG_RECV_DIR,
	ADBG_MSG_LOG_SEND_DIR
};

void adbgMsgLog(acpCtx_t ctx, enum adbgMsgLogDir dir, size_t size, const unsigned char* buffer);

SIDL_END_C_INTERFACE
