#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER SSeNB

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./common/utils/LOG/ss-tp.h"

#if !defined(_SS_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _SS_TP_H

#include <lttng/tracepoint.h>

TRACEPOINT_EVENT(
    SSeNB,
    SS_SYS,
    TP_ARGS(
        const char*, log_modName,
	int, event_id,
	int, sfn,
	int, slot,
	const char*, funcName,
	int, lineNo,
	const char*, msg
    ),
    TP_FIELDS(
        ctf_string(MODNAME, log_modName)
	ctf_integer(int32_t, EVENTID, event_id)
	ctf_integer(int32_t, SFN, sfn)
	ctf_integer(int32_t, SLOT, slot)
	ctf_string(FUNCTION, funcName)
	ctf_integer(int32_t, LINE, lineNo)
	ctf_string(MSG, msg)
    )
)

TRACEPOINT_EVENT(
    SSeNB,
    SS_SRB,
    TP_ARGS(
        const char*, log_modName,
	int, event_id,
	int, sfn,
	int, slot,
	const char*, funcName,
	int, lineNo,
	const char*, msg
    ),
    TP_FIELDS(
        ctf_string(MODNAME, log_modName)
	ctf_integer(int32_t, EVENTID, event_id)
	ctf_integer(int32_t, SFN, sfn)
	ctf_integer(int32_t, SLOT, slot)
	ctf_string(FUNCTION, funcName)
	ctf_integer(int32_t, LINE, lineNo)
	ctf_string(MSG, msg)
    )
)

TRACEPOINT_EVENT(
    SSeNB,
    SS_VNG,
    TP_ARGS(
        const char*, log_modName,
	int, event_id,
	int, sfn,
	int, slot,
	const char*, funcName,
	int, lineNo,
	const char*, msg
    ),
    TP_FIELDS(
        ctf_string(MODNAME, log_modName)
	ctf_integer(int32_t, EVENTID, event_id)
	ctf_integer(int32_t, SFN, sfn)
	ctf_integer(int32_t, SLOT, slot)
	ctf_string(FUNCTION, funcName)
	ctf_integer(int32_t, LINE, lineNo)
	ctf_string(MSG, msg)
    )
)

TRACEPOINT_EVENT(
    SSeNB,
    SS_DRB,
    TP_ARGS(
        const char*, log_modName,
	int, event_id,
	int, sfn,
	int, slot,
	const char*, funcName,
	int, lineNo,
	const char*, msg
    ),
    TP_FIELDS(
        ctf_string(MODNAME, log_modName)
	ctf_integer(int32_t, EVENTID, event_id)
	ctf_integer(int32_t, SFN, sfn)
	ctf_integer(int32_t, SLOT, slot)
	ctf_string(FUNCTION, funcName)
	ctf_integer(int32_t, LINE, lineNo)
	ctf_string(MSG, msg)
    )
)

TRACEPOINT_LOGLEVEL(SSeNB, SS_SYS, TRACE_DEBUG_FUNCTION)
TRACEPOINT_LOGLEVEL(SSeNB, SS_SRB, TRACE_DEBUG_FUNCTION)
TRACEPOINT_LOGLEVEL(SSeNB, SS_VNG, TRACE_DEBUG_FUNCTION)
TRACEPOINT_LOGLEVEL(SSeNB, SS_DRB, TRACE_DEBUG_FUNCTION)

/** ------------------------------------- Sublayers ------------------------------------------- */
TRACEPOINT_EVENT(
    SSeNB,
    SS_LOG,
    TP_ARGS(
	const char*, component,
	int, event_id,
	int, sfn,
	int, slot,
	const char*, funcName,
	int, lineNo,
	const char*, msg
    ),
    TP_FIELDS(
	ctf_string(MODNAME, component)
	ctf_integer(int32_t, EVENTID, event_id)
	ctf_integer(int32_t, SFN, sfn)
	ctf_integer(int32_t, SLOT, slot)
	ctf_string(FUNCTION, funcName)
	ctf_integer(int32_t, LINE, lineNo)
	ctf_string(MSG, msg)
    )
)

/** ------------------------------------- Packet Dump------------------------------------------- */
TRACEPOINT_EVENT(
    SSeNB,
    SS_PKT,
    TP_ARGS(
	const char*, component,
	int, event_id,
	int, sfn,
	int, slot,
        const char*, log_string,
        uint8_t *, array_arg,
        unsigned int, length
    ),
    TP_FIELDS(
	ctf_string(MODNAME, component)
	ctf_integer(int32_t, EVENTID, event_id)
	ctf_integer(int32_t, SFN, sfn)
	ctf_integer(int32_t, SLOT, slot)
        ctf_string(Event, log_string)
        ctf_sequence_hex(uint8_t, Buffer, array_arg, unsigned int, length)
    )
)



#endif /* _SS_TP_H */

#include <lttng/tracepoint-event.h>
