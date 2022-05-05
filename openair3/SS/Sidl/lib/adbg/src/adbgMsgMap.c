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

#include "adbgMsgMap.h"

#include "serTest.h"
#include "serHandshake.h"
#if defined(PROJECT_HAS_RAT_EUTRA)
#include "serSys.h"
#include "serSysInd.h"
#include "serSysSrb.h"
#include "serVng.h"
#include "serDrb.h"
#endif
#if defined(PROJECT_HAS_RAT_NR)
#include "serNrSysSrb.h"
#include "serNrSys.h"
#include "serNrDrb.h"
#endif

#include "adbgTest.h"
#include "adbgHandshake.h"
#if defined(PROJECT_HAS_RAT_EUTRA)
#include "adbgSys.h"
#include "adbgSysInd.h"
#include "adbgSysSrb.h"
#include "adbgVng.h"
#include "adbgDrb.h"
#endif
#if defined(PROJECT_HAS_RAT_NR)
#include "adbgNrSysSrb.h"
#include "adbgNrSys.h"
#include "adbgNrDrb.h"
#endif

void adbgMsgLogInArgs(acpCtx_t ctx, enum acpMsgLocalId id, size_t size, const unsigned char* buffer)
{
	if (id == ACP_LID_TestHelloFromSS) {
		return;
	}
	if (id == ACP_LID_TestPing) {
		return;
	}
	if (id == ACP_LID_TestEcho) {
		return;
	}
	if (id == ACP_LID_TestTest1) {
		return;
	}
	if (id == ACP_LID_TestTest2) {
		return;
	}
	if (id == ACP_LID_TestOther) {
		return;
	}
#if defined(PROJECT_HAS_RAT_EUTRA)
	if (id == ACP_LID_SysProcess) {
		struct SYSTEM_CTRL_REQ* FromSS;
		if (serSysProcessDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgSysProcessLogIn(ctx, FromSS);
			serSysProcessFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_SysSrbProcessFromSS) {
		struct EUTRA_RRC_PDU_REQ* FromSS;
		if (serSysSrbProcessFromSSDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgSysSrbProcessFromSSLogIn(ctx, FromSS);

			serSysSrbProcessFromSSFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_VngProcess) {
		struct EUTRA_VNG_CTRL_REQ* FromSS;
		if (serVngProcessDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgVngProcessLogIn(ctx, FromSS);
			serVngProcessFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_DrbProcessFromSS) {
		struct DRB_COMMON_REQ* FromSS;
		if (serDrbProcessFromSSDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgDrbProcessFromSSLogIn(ctx, FromSS);

			serDrbProcessFromSSFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
#endif
	if (id == ACP_LID_HandshakeProcess) {
		return;
	}
#if defined(PROJECT_HAS_RAT_NR)
	if (id == ACP_LID_NrSysSrbProcessFromSS) {
		struct NR_RRC_PDU_REQ* FromSS;
		if (serNrSysSrbProcessFromSSDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgNrSysSrbProcessFromSSLogIn(ctx, FromSS);

			serNrSysSrbProcessFromSSFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_NrSysProcess) {
		struct NR_SYSTEM_CTRL_REQ* FromSS;
		if (serNrSysProcessDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgNrSysProcessLogIn(ctx, FromSS);

			serNrSysProcessFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_NrDrbProcessFromSS) {
		struct NR_DRB_COMMON_REQ* FromSS;
		if (serNrDrbProcessFromSSDecSrv(buffer, size, NULL, 0, &FromSS) == 0) {
			adbgNrDrbProcessFromSSLogIn(ctx, FromSS);

			serNrDrbProcessFromSSFreeSrv(FromSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
#endif
	if (id == ACP_LID_SysVTEnquireTimingAck) {
		return;
	}
	SIDL_ASSERT(0);
}

void adbgMsgLogOutArgs(acpCtx_t ctx, enum acpMsgLocalId id, size_t size, const unsigned char* buffer)
{
	if (id == ACP_LID_TestHelloToSS) {
		return;
	}
	if (id == ACP_LID_TestPing) {
		return;
	}
	if (id == ACP_LID_TestEcho) {
		return;
	}
	if (id == ACP_LID_TestTest1) {
		return;
	}
	if (id == ACP_LID_TestTest2) {
		return;
	}
	if (id == ACP_LID_TestOther) {
		return;
	}
#if defined(PROJECT_HAS_RAT_EUTRA)
	if (id == ACP_LID_SysProcess) {
		struct SYSTEM_CTRL_CNF* ToSS;
		if (serSysProcessDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgSysProcessLogOut(ctx, ToSS);
			serSysProcessFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_SysSrbProcessToSS) {
		struct EUTRA_RRC_PDU_IND* ToSS;
		if (serSysSrbProcessToSSDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgSysSrbProcessToSSLogOut(ctx, ToSS);
			serSysSrbProcessToSSFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}

	if (id == ACP_LID_VngProcess) {
		struct EUTRA_VNG_CTRL_CNF* ToSS;
		if (serVngProcessDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgVngProcessLogOut(ctx, ToSS);
			serVngProcessFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_DrbProcessToSS) {
		struct DRB_COMMON_IND* ToSS;
		if (serDrbProcessToSSDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgDrbProcessToSSLogOut(ctx, ToSS);
			serDrbProcessToSSFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_SysIndProcessToSS) {
		struct SYSTEM_IND* ToSS = NULL;
		if (serSysIndProcessToSSDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgSysIndProcessToSSLogOut(ctx, ToSS);
			serSysIndProcessToSSFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
#endif
	if (id == ACP_LID_HandshakeProcess) {
		return;
	}
#if defined(PROJECT_HAS_RAT_NR)
	if (id == ACP_LID_NrSysSrbProcessToSS) {
		struct NR_RRC_PDU_IND* ToSS;
		if (serNrSysSrbProcessToSSDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgNrSysSrbProcessToSSLogOut(ctx, ToSS);
			serNrSysSrbProcessToSSFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_NrSysProcess) {
		struct NR_SYSTEM_CTRL_CNF* ToSS;
		if (serNrSysProcessDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgNrSysProcessLogOut(ctx, ToSS);
			serNrSysProcessFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
	if (id == ACP_LID_NrDrbProcessToSS) {
		struct NR_DRB_COMMON_IND* ToSS;
		if (serNrDrbProcessToSSDecClt(buffer, size, NULL, 0, &ToSS) == 0) {
			adbgNrDrbProcessToSSLogOut(ctx, ToSS);
			serNrDrbProcessToSSFreeClt(ToSS);
		} else {
			adbgPrintLog(ctx, "cannot decode buffer");
			adbgPrintLog(ctx, NULL);
		}
		return;
	}
#endif
	if (id == ACP_LID_SysVTEnquireTimingUpd) {
		return;
	}

	SIDL_ASSERT(0);
}
