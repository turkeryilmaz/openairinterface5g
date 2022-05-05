/*
 *****************************************************************
 *
 * Module : Serialization
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

// System includes
#include <stdbool.h>
#include <stdint.h>

// Internal includes
#include "SidlCompiler.h"

SIDL_BEGIN_C_INTERFACE

/** Defines magic value for memory context.*/
#define SER_MEM_MAGIC 0x0AC90AC9

/** Defines magic value for dynamic memory context.*/
#define SER_MEM_DYNAMIC_MAGIC 0x1AC91AC9

/** Defines memory context. */
struct serMemCtx {
	uint32_t magic;
	size_t size;
	size_t index;
};

/** Memory context handle. */
typedef struct serMemCtx* serMem_t;

/** Initialize memory context. */
serMem_t serMemInit(unsigned char* arena, unsigned int aSize);

/** Allocates size bytes and returns a pointer to the allocated memory. */
void* serMalloc(serMem_t mem, size_t size);

/** frees the memory space pointed to by ptr, which must have been returned
 * by a previous call serMalloc.*/
void serFree(void* ptr);

/** Defines preinitialized memory context for dynamic memory allocation. */
extern serMem_t serMemDyn;

/** Defines convenient way to allocate memory using dynamic allocation. */
#define SER_DYN_ALLOC(sIZE) serMalloc(serMemDyn, (sIZE))

SIDL_END_C_INTERFACE
