/*
 *****************************************************************
 *
 * Module : Serialization
 * Purpose: Utils
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

// Macro definitions

#define HTON_8(dst, src, idx) \
	{                         \
		*dst = src;           \
		*idx += 1;            \
	}

#define NTOH_8(dst, src, idx)           \
	{                                   \
		dst = ((unsigned char*)src)[0]; \
		*idx += 1;                      \
	}

#define HTON_16(dst, src, idx)      \
	{                               \
		*(dst) = (src >> 8) & 0xFF; \
		*(dst + 1) = (src)&0xFF;    \
		*idx += 2;                  \
	}

#define NTOH_16(dst, src, idx)                   \
	{                                            \
		dst = ((((unsigned char*)src)[0] << 8) | \
			   ((unsigned char*)src)[1]);        \
		*idx += 2;                               \
	}

#define HTON_32(dst, src, idx)           \
	{                                    \
		*(dst) = (src >> 24) & 0xFF;     \
		*(dst + 1) = (src >> 16) & 0xFF; \
		*(dst + 2) = (src >> 8) & 0xFF;  \
		*(dst + 3) = (src)&0xFF;         \
		*idx += 4;                       \
	}

#define NTOH_32(dst, src, idx)                      \
	{                                               \
		(dst) = ((((unsigned char*)src)[0] << 24) | \
				 (((unsigned char*)src)[1] << 16) | \
				 (((unsigned char*)src)[2] << 8) |  \
				 ((unsigned char*)src)[3]);         \
		*(idx) += 4;                                \
	}

#define HTON_64(dst, src, idx)                       \
	{                                                \
		*(dst) = ((uint64_t)(src) >> 56) & 0xFF;     \
		*(dst + 1) = ((uint64_t)(src) >> 48) & 0xFF; \
		*(dst + 2) = ((uint64_t)(src) >> 40) & 0xFF; \
		*(dst + 3) = ((uint64_t)(src) >> 32) & 0xFF; \
		*(dst + 4) = ((uint64_t)(src) >> 24) & 0xFF; \
		*(dst + 5) = ((uint64_t)(src) >> 16) & 0xFF; \
		*(dst + 6) = ((uint64_t)(src) >> 8) & 0xFF;  \
		*(dst + 7) = ((uint64_t)(src) >> 0) & 0xFF;  \
		*idx += 8;                                   \
	}

#define NTOH_64(dst, src, idx)                                  \
	{                                                           \
		(dst) = (((uint64_t)(((unsigned char*)src)[0]) << 56) | \
				 ((uint64_t)(((unsigned char*)src)[1]) << 48) | \
				 ((uint64_t)(((unsigned char*)src)[2]) << 40) | \
				 ((uint64_t)(((unsigned char*)src)[3]) << 32) | \
				 ((uint64_t)(((unsigned char*)src)[4]) << 24) | \
				 ((uint64_t)(((unsigned char*)src)[5]) << 16) | \
				 ((uint64_t)(((unsigned char*)src)[6]) << 8) |  \
				 ((uint64_t)(((unsigned char*)src)[7]) << 0));  \
		*(idx) += 8;                                            \
	}

SIDL_END_C_INTERFACE
