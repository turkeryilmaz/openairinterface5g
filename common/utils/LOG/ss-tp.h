#undef TRACEPOINT_PROVIDER
#define TRACEPOINT_PROVIDER SSeNB

#undef TRACEPOINT_INCLUDE
#define TRACEPOINT_INCLUDE "./common/utils/LOG/ss-tp.h"

#if !defined(_SS_TP_H) || defined(TRACEPOINT_HEADER_MULTI_READ)
#define _SS_TP_H

#include <stdbool.h>
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

#if !defined(_SS_MAC_PKT_T)
typedef struct mac_pkt_info_s {
     int direction;
     int rnti_type;
     int rnti;
     int harq_pid;
     int preamble;
} mac_pkt_info_t;
#define _SS_MAC_PKT_T
#endif

TRACEPOINT_EVENT(
    SSeNB,
    SS_MAC_PKT,
    TP_ARGS(
		 const char*, component,
		 int, event_id,
		 int, sfn,
		 int, slot,
		 const char*, funcName,
		 int, lineNo,
		 mac_pkt_info_t, mac_pkt,
		 const char*, log_string,
		 uint8_t *, array_arg,
		 unsigned int, length
    ),
    TP_FIELDS(
		 ctf_string(MODNAME, component)
		 ctf_integer(int32_t, EVENTID, event_id)
		 ctf_integer(int32_t, SFN, sfn)
		 ctf_integer(int32_t, SLOT, slot)
		 ctf_string(FUNCTION, funcName)
		 ctf_integer(int32_t, LINE, lineNo)
		 ctf_integer(int32_t, DIRECTION, mac_pkt.direction)
		 ctf_integer(int32_t, RNTI_TYPE, mac_pkt.rnti_type)
		 ctf_integer(int32_t, RNTI, mac_pkt.rnti)
		 ctf_integer(int32_t, HARQ_PID, mac_pkt.harq_pid)
		 ctf_integer(int32_t, PREAMBLE, mac_pkt.preamble)
		 ctf_string(Event, log_string)
		 ctf_sequence_hex(uint8_t, Buffer, array_arg, unsigned int, length)
    )
)

#if !defined(_SS_NR_RLC_PKT_T)
typedef struct nr_rlc_pkt_info_s {
     uint8_t          rlcMode;
     uint8_t          direction;
     uint8_t          sequenceNumberLength;
     uint8_t          bearerType;
     uint8_t          bearerId;
     uint16_t         ueid;
     uint16_t         pduLength;
} nr_rlc_pkt_info_t;
#define _SS_NR_RLC_PKT_T
#endif

TRACEPOINT_EVENT(
    SSeNB,
    SS_NR_RLC_PKT,
    TP_ARGS(
		 const char*, component,
		 int, event_id,
		 int, sfn,
		 int, slot,
		 const char*, funcName,
		 int, lineNo,
		 nr_rlc_pkt_info_t, rlc_pkt,
		 const char*, log_string,
		 uint8_t *, array_arg,
		 unsigned int, length
    ),
    TP_FIELDS(
		 ctf_string(MODNAME, component)
		 ctf_integer(int32_t, EVENTID, event_id)
		 ctf_integer(int32_t, SFN, sfn)
		 ctf_integer(int32_t, SLOT, slot)
		 ctf_string(FUNCTION, funcName)
		 ctf_integer(int32_t, LINE, lineNo)
		 ctf_integer(int8_t,  RLCMODE, rlc_pkt.rlcMode)
		 ctf_integer(int8_t,  DIRECTION, rlc_pkt.direction)
		 ctf_integer(int8_t, SN_LENGTH, rlc_pkt.sequenceNumberLength)
		 ctf_integer(int8_t, BEARER_TYPE, rlc_pkt.bearerType)
		 ctf_integer(int8_t, BEARER_ID, rlc_pkt.bearerId)
		 ctf_integer(int16_t, UEID, rlc_pkt.ueid)
		 ctf_integer(int16_t, PDU_LENGTH, rlc_pkt.pduLength)
		 ctf_string(Event, log_string)
		 ctf_sequence_hex(uint8_t, Buffer, array_arg, unsigned int, length)
    )
)

#if !defined(_SS_NR_PDCP_PKT_T)
typedef enum NRBearerTypeE
{
    Bearer_UNDEFINED_e = 0,
    Bearer_DCCH_e=1,
    Bearer_BCCH_BCH_e=2,
    Bearer_BCCH_DL_SCH_e=3,
    Bearer_CCCH_e=4,
    Bearer_PCCH_e=5,
} NRBearerTypeE;
#define PDCP_NR_SN_LENGTH_12_BITS 12
#define PDCP_NR_SN_LENGTH_18_BITS 18

#define PDCP_NR_UL_SDAP_HEADER_PRESENT 0x01
#define PDCP_NR_DL_SDAP_HEADER_PRESENT 0x02

enum pdcp_nr_plane_e
{
    NR_PLANE_UNDEFINED_E = 0,
    NR_SIGNALING_PLANE_E = 1,
    NR_USER_PLANE_E = 2
};
#define MAX_CID      15

#define ROHC_PROFILE_UNCOMPRESSED   0
#define ROHC_PROFILE_RTP            1
#define ROHC_PROFILE_UDP            2
#define ROHC_PROFILE_IP             4
#define ROHC_PROFILE_UNKNOWN        0xFFFF


typedef struct rohc_info
{
    bool           rohc_compression;
    uint8_t             rohc_ip_version;
} rohc_info;

typedef struct nr_pdcp_pkt_info_s {
    uint8_t             direction;
    uint16_t            ueid;
    NRBearerTypeE       bearerType;
    uint8_t             bearerId;

    enum pdcp_nr_plane_e plane;
    uint8_t             seqnum_length;
    bool                maci_present;
    bool                ciphering_disabled;
    uint8_t             sdap_header;

    //rohc_info          rohc;

    uint8_t             is_retx;

    uint16_t            pdu_length;
} nr_pdcp_pkt_info_t;

typedef enum
{
	UNDEFINED_TRANSPORT = 0,
    bch_TRANSPORT=1,
    dlsch_TRANSPORT=2
} BCCHTransportType_e;

enum pdcp_plane_lte
{
    CONTROL_PLANE_E= 1,
    DATA_PLANE_E = 2
};

typedef struct pdcp_info
{
    uint8_t                 direction;
    uint16_t                ueid;
    NRBearerTypeE         channelType;
    uint16_t                channelId;
    enum pdcp_plane_lte          plane; //1 control plane 2 user plane
    uint8_t                 seqnum_length;
    uint8_t                 is_retx;
    BCCHTransportType_e   BCCHTransport;
    bool               no_header_pdu;
    uint16_t                pdu_length;
} pdcp_info_t;



//TODO: PDCP_NR_ROHC_COMPRESSION_TAG to be set to 0 always for now.
#define _SS_NR_PDCP_PKT_T
#endif

TRACEPOINT_EVENT(
    SSeNB,
    SS_NR_PDCP_PKT,
    TP_ARGS(
		 const char*, component,
		 int, event_id,
		 int, sfn,
		 int, slot,
		 const char*, funcName,
		 int, lineNo,
		 nr_pdcp_pkt_info_t, pdcp_pkt,
		 const char*, log_string,
		 uint8_t *, array_arg,
		 unsigned int, length
    ),
    TP_FIELDS(
		 ctf_string(MODNAME, component)
		 ctf_integer(int32_t, EVENTID, event_id)
		 ctf_integer(int32_t, SFN, sfn)
		 ctf_integer(int32_t, SLOT, slot)
		 ctf_string(FUNCTION, funcName)
		 ctf_integer(int32_t, LINE, lineNo)
		 ctf_integer(int8_t,  DIRECTION, pdcp_pkt.direction)
		 ctf_integer(int16_t, UEID, pdcp_pkt.ueid)
		 ctf_integer(int8_t, BEARER_TYPE, pdcp_pkt.bearerType)
		 ctf_integer(int8_t, BEARER_ID, pdcp_pkt.bearerId)
		 ctf_integer(int8_t, PLANE, pdcp_pkt.plane)
		 ctf_integer(int8_t, SN_LENGTH, pdcp_pkt.seqnum_length)
		 ctf_integer(bool, CIPHER_DISABLED, pdcp_pkt.ciphering_disabled)
		 ctf_integer(bool, MACI_DISABLED, pdcp_pkt.maci_present)
		 ctf_integer(int8_t, SDAP_PR, pdcp_pkt.sdap_header)
		 ctf_integer(int8_t, RETX, pdcp_pkt.is_retx)
		 ctf_integer(int16_t, PDU_LEN, pdcp_pkt.pdu_length)
		 ctf_string(Event, log_string)
		 ctf_sequence_hex(uint8_t, Buffer, array_arg, unsigned int, length)
    )
)


TRACEPOINT_EVENT(
    SSeNB,
    SS_LTE_PDCP_PKT,
    TP_ARGS(
		 const char*, component,
		 int, event_id,
		 int, sfn,
		 int, slot,
		 const char*, funcName,
		 int, lineNo,
		 pdcp_info_t, pdcp_pkt,
		 const char*, log_string,
		 uint8_t *, array_arg,
		 unsigned int, length
    ),
    TP_FIELDS(
		 ctf_string(MODNAME, component)
		 ctf_integer(int32_t, EVENTID, event_id)
		 ctf_integer(int32_t, SFN, sfn)
		 ctf_integer(int32_t, SLOT, slot)
		 ctf_string(FUNCTION, funcName)
		 ctf_integer(int32_t, LINE, lineNo)
		 ctf_integer(uint8_t,  DIRECTION, pdcp_pkt.direction)
		 ctf_integer(uint16_t, UEID, pdcp_pkt.ueid)
		 ctf_integer(uint8_t, BEARER_TYPE, pdcp_pkt.channelType)
		 ctf_integer(uint16_t, BEARER_ID, pdcp_pkt.channelId)
		 ctf_integer(uint8_t, PLANE, pdcp_pkt.plane)
		 ctf_integer(uint8_t, SN_LENGTH, pdcp_pkt.seqnum_length)
		 ctf_integer(uint8_t, RETX, pdcp_pkt.is_retx)
		 ctf_integer(uint8_t, BCCH_TRANSPORT, pdcp_pkt.BCCHTransport)
		 ctf_integer(uint8_t, PDU_HEADER_PRESENT, pdcp_pkt.no_header_pdu)
		 ctf_integer(uint16_t, PDU_LEN, pdcp_pkt.pdu_length)
		 ctf_string(Event, log_string)
		 ctf_sequence_hex(uint8_t, Buffer, array_arg, unsigned int, length)
    )
)

#if !defined(_SS_LTE_RLC_PKT_T)
typedef struct lte_rlc_pkt_info
{
    uint8_t          rlcMode;
    uint8_t          direction;
    uint8_t          priority;
    uint8_t          sequenceNumberLength;
    uint16_t         ueid;
    uint16_t         channelType;
    uint16_t         channelId; /* for SRB: 1=SRB1, 2=SRB2, 3=SRB1bis; for DRB: DRB ID */
    uint16_t         pduLength;
    bool             extendedLiField;
} lte_rlc_pkt_info_t;
#define _SS_LTE_RLC_PKT_T
#endif
TRACEPOINT_EVENT(
    SSeNB,
    SS_LTE_RLC_PKT,
    TP_ARGS(
		 const char*, component,
		 int, event_id,
		 int, sfn,
		 int, sf,
		 const char*, funcName,
		 int, lineNo,
		 lte_rlc_pkt_info_t, rlc_pkt,
		 const char*, log_string,
		 uint8_t *, array_arg,
		 unsigned int, length
    ),
    TP_FIELDS(
		 ctf_string(MODNAME, component)
		 ctf_integer(int32_t, EVENTID, event_id)
		 ctf_integer(int32_t, SFN, sfn)
		 ctf_integer(int32_t, SF, sf)
		 ctf_string(FUNCTION, funcName)
		 ctf_integer(uint32_t, LINE, lineNo)
		 ctf_integer(uint8_t,  RLCMODE, rlc_pkt.rlcMode)
		 ctf_integer(uint8_t,  DIRECTION, rlc_pkt.direction)
		 ctf_integer(uint8_t,  PRIORITY, rlc_pkt.priority)
		 ctf_integer(uint8_t,  SN_LENGTH, rlc_pkt.sequenceNumberLength)
		 ctf_integer(uint16_t, UEID, rlc_pkt.ueid)
		 ctf_integer(uint8_t,  CHANNEL_TYPE, rlc_pkt.channelType)
		 ctf_integer(uint8_t,  CHANNEL_ID, rlc_pkt.channelId)
		 ctf_integer(uint16_t, PDU_LENGTH, rlc_pkt.pduLength)
		 ctf_integer(uint8_t, EXTENDEDLIFIELD, rlc_pkt.extendedLiField)
		 ctf_string(Event, log_string)
		 ctf_sequence_hex(uint8_t, Buffer, array_arg, unsigned int, length)
    )
)

#endif /* _SS_TP_H */

#include <lttng/tracepoint-event.h>
