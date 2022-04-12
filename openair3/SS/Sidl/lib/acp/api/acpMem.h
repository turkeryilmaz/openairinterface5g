/*
 *****************************************************************
 *
 * Module : Asynchronous Communication Protocol
 * Purpose: Memory management
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
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

/** Allocates size bytes and returns a pointer to the allocated memory.
 *
 * @param[in]  size	Buffer size to allocate
 * @return   pointer to the allocation buffer or NULL on failure
 */
void* acpMalloc(size_t size);

/** Frees the memory space pointed to by ptr, which must have been returned
 * by a previous call acpMalloc.
 *
 * @param[in]  ptr Pointer to the allocated buffer
 */
void acpFree(void* ptr);

SIDL_END_C_INTERFACE
