/*
 *****************************************************************
 *
 * Module : SIDL base - compiler support
 *
 * Purpose: Provide abstraction layers and feature defines to help
 *          using compiler specific features in a portable way.
 *
 *****************************************************************
 *
 *  Copyright (c) 2009-2021 SEQUANS Communications.
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

// Prefer standard definitions whenever possible.
#include <unistd.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>

// C in C++ usage.
#ifdef __cplusplus
#define SIDL_BEGIN_C_INTERFACE extern "C" {
#define SIDL_END_C_INTERFACE }
#else // !__cplusplus
#define SIDL_BEGIN_C_INTERFACE
#define SIDL_END_C_INTERFACE
#endif // __cplusplus

#define SIDL_ASSERT(eXPR) assert(eXPR)

/// SIDL API result code.
typedef enum {
	// Mandatory return codes for all SIDL interfaces.
	SIDL_STATUS_OK = 0,
	SIDL_STATUS_ERROR = 1,
	SIDL_STATUS_NOTIMP = 2,

	// Start of the service specific error status space. Each service can
	// define its error codes, possibly with an enum, starting from this
	// value included.
	SIDL_STATUS_LOCAL = 0x10
} SidlStatus;

/// Milliseconds type.
typedef uint32_t MSec_t;

/// The IP protocol type (v4 or v6).
typedef enum IpType {
	IP_V4,
	IP_V6
} IpType_t;

/// An IPv4 address.
typedef uint32_t Ip4Address_t;

/// The IPv6 address.
typedef struct Ip6Address {
	uint64_t firstPart;
	uint64_t secondPart;
} Ip6Address_t;

/// An IP address, whether v4 or v6, as an union.
typedef struct IpAddress {
	IpType_t d;
	union {
		Ip4Address_t ipv4;
		Ip6Address_t ipv6;
	} v;
} IpAddress_t;

/// Stubs for not implemented fields.
struct SidlNotImplemented {
};
typedef struct SidlNotImplemented SidlNotImplemented_t;
#define SIDL_NOT_IMPLEMENTED(tYPE) SidlNotImplemented_t
#define SIDL_NOT_IMPLEMENTED_S(tYPE) SidlNotImplemented

// Internal helper macros.
#define _PP_CAT(_1, _2) _1##_2
#define _PP_JOIN2(_c, _1, _2) _1##_c##_2
#define _PP_JOIN3(_c, _1, _2, _3) _1##_c##_2##_c##_3
#define _PP_JOIN4(_c, _1, _2, _3, _4) _1##_c##_2##_c##_3##_c##_4

#define _PP_CAT_N_1(_1) _1
#define _PP_CAT_N_2(_1, _2) _1##_##_2
#define _PP_CAT_N_3(_1, _2, _3) _1##_##_2##_##_3
#define _PP_CAT_N_4(_1, _2, _3, _4) _1##_##_2##_##_3##_##_4
#define _PP_CAT_N_5(_1, _2, _3, _4, _5) _1##_##_2##_##_3##_##_4##_##_5
#define _PP_CAT_N_6(_1, _2, _3, _4, _5, _6) _1##_##_2##_##_3##_##_4##_##_5##_##_6

#define _PP_CAT_DETAIL_ARG_N(_1, _2, _3, _4, _5, _6, N, ...) N
#define _PP_CAT_DETAIL_NARG(args) _PP_CAT_DETAIL_ARG_N args
#define _PP_CAT_DETAIL_RSEQ_N() 6, 5, 4, 3, 2, 1, 0
#define _PP_CAT_NARGS(...) _PP_CAT_DETAIL_NARG((__VA_ARGS__, _PP_CAT_DETAIL_RSEQ_N()))
#define _PP_CAT_CASE(cASE) _PP_CAT(_PP_CAT_N_, cASE)
#define _PP_AUTO_CAT(...)                    \
	_PP_CAT_CASE(_PP_CAT_NARGS(__VA_ARGS__)) \
	(__VA_ARGS__)
