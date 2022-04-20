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

// System includes
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>

// Internal includes
#include "adbg.h"
#include "acpCtx.h"

adbgGlobalLogger_t adbgGlobalLogger = NULL;

adbgLogger_t adbgLogger = NULL;

void adbgDefaultGlobalLogger(const char* fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	int rc = vprintf(fmt, ap);
	rc = printf("\n");
	(void)rc;
	va_end(ap);
}

void adbgDefaultLogger(const char* str)
{
	int rc = printf("%s\n", str);
	(void)rc; // ignore
}

void adbgSetGlobalLoggers(adbgGlobalLogger_t globalLogger, adbgLogger_t logger)
{
	adbgGlobalLogger = globalLogger;
	adbgLogger = logger;
}

void adbgCheckAndSetGlobalLogger()
{
	if (!adbgGlobalLogger) {
		adbgGlobalLogger = adbgDefaultGlobalLogger;
	}
}

void adbgSetLogger(acpCtx_t ctx, adbgLogger_t logger, size_t size)
{
	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_LOG("Invalid context");
		SIDL_ASSERT(ctx != ctx);
		return;
	}

	if (logger) {
		if (ACP_CTX_CAST(ctx)->logger) {
			SIDL_ASSERT(!ACP_CTX_CAST(ctx)->logger);
		}
		ACP_CTX_CAST(ctx)->logger = logger;
	}

	if (size) {
		if (ACP_CTX_CAST(ctx)->logBuf) {
			SIDL_ASSERT(!ACP_CTX_CAST(ctx)->logBuf);
		}
		ACP_CTX_CAST(ctx)->logBuf = (unsigned char*)acpMalloc(size);
		SIDL_ASSERT(ACP_CTX_CAST(ctx)->logBuf);
		ACP_CTX_CAST(ctx)->logBufMaxSize = size;
		ACP_CTX_CAST(ctx)->logBufSize = 0;
	}
}

static void adbgCheckAndSetLogger(acpCtx_t ctx)
{
	if (!ACP_CTX_CAST(ctx)->logger) {
		ACP_CTX_CAST(ctx)->logger = adbgLogger ? adbgLogger : adbgDefaultLogger;
	}

	if (!ACP_CTX_CAST(ctx)->logBuf) {
		ACP_CTX_CAST(ctx)->logBuf = (unsigned char*)acpMalloc(ADBG_DEFAULT_LOG_BUFFER_SIZE);
		SIDL_ASSERT(ACP_CTX_CAST(ctx)->logBuf);
		ACP_CTX_CAST(ctx)->logBufMaxSize = ADBG_DEFAULT_LOG_BUFFER_SIZE;
		ACP_CTX_CAST(ctx)->logBufSize = 0;
	}

	if (ACP_CTX_CAST(ctx)->logFormat) {
		if (ACP_CTX_CAST(ctx)->logFormatBufMaxSize != (ACP_CTX_CAST(ctx)->logBufMaxSize + 1024)) {
			if (ACP_CTX_CAST(ctx)->logFormatBuf) {
				acpFree(ACP_CTX_CAST(ctx)->logFormatBuf);
			}
			ACP_CTX_CAST(ctx)->logFormatBufMaxSize = ACP_CTX_CAST(ctx)->logBufMaxSize + 1024;
			ACP_CTX_CAST(ctx)->logFormatBuf = (unsigned char*)acpMalloc(ACP_CTX_CAST(ctx)->logFormatBufMaxSize);
			SIDL_ASSERT(ACP_CTX_CAST(ctx)->logFormatBuf);
		}
	}
}

void adbgPrintLog(acpCtx_t ctx, const char* fmt, ...)
{
	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_LOG("Invalid context");
		SIDL_ASSERT(ctx != ctx);
		return;
	}

	adbgCheckAndSetLogger(ctx);

	if (fmt) {
		va_list ap;
		va_start(ap, fmt);
		int rc = vsnprintf((char*)(&ACP_CTX_CAST(ctx)->logBuf[ACP_CTX_CAST(ctx)->logBufSize]), ACP_CTX_CAST(ctx)->logBufMaxSize - ACP_CTX_CAST(ctx)->logBufSize, fmt, ap);
		va_end(ap);
		SIDL_ASSERT(rc >= 0);
		ACP_CTX_CAST(ctx)->logBufSize += rc;
		SIDL_ASSERT(ACP_CTX_CAST(ctx)->logBufMaxSize > ACP_CTX_CAST(ctx)->logBufSize);
	} else {
		if (ACP_CTX_CAST(ctx)->logBufSize) {
			ACP_CTX_CAST(ctx)->logger((const char*)ACP_CTX_CAST(ctx)->logBuf);
			ACP_CTX_CAST(ctx)->logBufSize = 0;
		}
	}
}

void adbgPrintFormatLog(acpCtx_t ctx)
{
	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_LOG("Invalid context");
		SIDL_ASSERT(ctx != ctx);
		return;
	}

	if (!ACP_CTX_CAST(ctx)->logBufSize) {
		return;
	}

	if (!ACP_CTX_CAST(ctx)->logFormat) {
		ACP_CTX_CAST(ctx)->logger((const char*)ACP_CTX_CAST(ctx)->logBuf);
		ACP_CTX_CAST(ctx)->logBufSize = 0;
		return;
	}

	adbgCheckAndSetLogger(ctx);

	size_t sz = 0;
	size_t lead = 0;
	bool trim = false;
	char ch;
	size_t i;

	const char* src = (const char*)ACP_CTX_CAST(ctx)->logBuf;
	char* res = (char*)ACP_CTX_CAST(ctx)->logFormatBuf;

	while ((ch = *src++)) {
		if (ch == '{') {
			trim = true;
			lead += 4;
			*res++ = '{';
			*res++ = '\n';
			for (i = 0; i < lead; i++) *res++ = ' ';
			sz += lead + 2;
		} else if (ch == '}') {
			lead -= 4;
			*res++ = '\n';
			for (i = 0; i < lead; i++) *res++ = ' ';
			*res++ = '}';
			sz += lead + 2;
		} else if (ch == ',') {
			trim = true;
			*res++ = ',';
			*res++ = '\n';
			for (i = 0; i < lead; i++) *res++ = ' ';
			sz += lead + 2;
		} else {
			if (trim) {
				if (isspace(ch)) {
					continue;
				} else {
					trim = false;
				}
			}

			*res++ = ch;
			sz++;
		}
	}

	*res = '\0';
	sz++;

	SIDL_ASSERT(ACP_CTX_CAST(ctx)->logFormatBufMaxSize > sz);

	ACP_CTX_CAST(ctx)->logger((const char*)ACP_CTX_CAST(ctx)->logFormatBuf);
	ACP_CTX_CAST(ctx)->logBufSize = 0;
}

void adbgSetPrintLogFormat(acpCtx_t ctx, bool format)
{
	if (!acpCtxIsValid(ctx)) {
		ACP_DEBUG_LOG("Invalid context");
		SIDL_ASSERT(ctx != ctx);
		return;
	}

	ACP_CTX_CAST(ctx)->logFormat = format;

	if (!format) {
		if (ACP_CTX_CAST(ctx)->logFormatBuf) {
			acpFree(ACP_CTX_CAST(ctx)->logFormatBuf);
			ACP_CTX_CAST(ctx)->logFormatBuf = NULL;
		}
	}
}
