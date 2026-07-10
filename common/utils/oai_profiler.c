/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "oai_profiler.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <pwd.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#include "common/oai_version.h"
#include "common/utils/LOG/log.h"

#define OAI_PROFILE_DEFAULT_BUFFER_RECORDS 131072U
#define OAI_PROFILE_DEFAULT_FLUSH_US 100000U
#define OAI_PROFILE_DEFAULT_HOST_METRICS_US 1000000U
#define OAI_PROFILE_MIN_BUFFER_RECORDS 1024U
#define OAI_PROFILE_THREAD_NAME_LEN 16
#define OAI_PROFILE_COMPONENT_LEN 128
#define OAI_PROFILE_RPI_GET_THROTTLED 0x00030046U
#define OAI_PROFILE_RPI_MAILBOX_PROPERTY _IOWR(100, 0, char *)

int oai_profiler_enabled = 0;

typedef struct {
  uint64_t seq;
  uint64_t start_tick;
  uint64_t duration_tick;
  uint64_t span_id;
  uint64_t parent_id;
  uint64_t correlation_id;
  int64_t absolute_slot;
  uint32_t event_id;
  int32_t frame;
  int32_t slot;
  int32_t cpu_start;
  int32_t cpu_end;
  int64_t aux0;
  int64_t aux1;
  int64_t aux2;
  int64_t aux3;
  uint32_t flags;
  uint16_t nesting_depth;
  uint8_t event_kind;
  uint8_t reserved;
} oai_profile_record_t;

typedef struct {
  bool active;
  pid_t tid;
  char name[OAI_PROFILE_THREAD_NAME_LEN];
  oai_profile_record_t *records;
  uint32_t capacity;
  uint64_t write_count;
  uint64_t read_count;
  uint64_t dropped_records;
  uint64_t next_span_sequence;
  uint64_t span_stack_overflows;
  uint64_t span_stack_mismatches;
} oai_profile_thread_buffer_t;

#define AUX_FIELDS(name0, unit0, name1, unit1, name2, unit2, name3, unit3) \
  .aux_name = {name0, name1, name2, name3}, .aux_unit = {unit0, unit1, unit2, unit3}

static const oai_profile_event_descriptor_t event_descriptors[OAI_PROFILE_EVENT_MAX] = {
    [OAI_PROFILE_EVENT_UNSPEC] =
        {
            .name = "UNSPEC",
            .role = "common",
            .subsystem = "unknown",
            .event_class = "unknown",
            .default_kind = OAI_PROFILE_EVENT_KIND_UNKNOWN,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("", "", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_SLOT_LOOP] =
        {
            .name = "UE_SLOT_LOOP",
            .role = "nrUE",
            .subsystem = "orchestration",
            .event_class = "loop",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("duration_rx_to_tx",
                       "slot",
                       "read_samples",
                       "sample",
                       "write_samples",
                       "sample",
                       "timing_advance",
                       "sample"),
            .flags_name = "variant",
        },
    [OAI_PROFILE_EVENT_UE_RF_READ] =
        {
            .name = "UE_RF_READ",
            .role = "nrUE",
            .subsystem = "radio",
            .event_class = "io",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("requested_samples",
                       "sample",
                       "antenna_count",
                       "count",
                       "returned_samples",
                       "sample",
                       "device_timestamp",
                       "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_RF_READ_DRIFT] =
        {
            .name = "UE_RF_READ_DRIFT",
            .role = "nrUE",
            .subsystem = "radio",
            .event_class = "io",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("requested_samples",
                       "sample",
                       "antenna_count",
                       "count",
                       "returned_samples",
                       "sample",
                       "device_timestamp",
                       "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_SCOPE_COPY] =
        {
            .name = "UE_SCOPE_COPY",
            .role = "nrUE",
            .subsystem = "observability",
            .event_class = "copy",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("sample_count", "sample", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TIMING_COMPUTE] =
        {
            .name = "UE_TIMING_COMPUTE",
            .role = "nrUE",
            .subsystem = "timing",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("write_samples", "sample", "timing_advance", "sample", "n_ta_offset", "sample", "absolute_deadline", "us"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_PREPROCESS] =
        {
            .name = "UE_DL_PREPROCESS",
            .role = "nrUE",
            .subsystem = "phy_dl",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("preprocess_result", "sample", "rx_slot_type", "enum", "tx_slot_type", "enum", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_PROCESSING] =
        {
            .name = "UE_DL_PROCESSING",
            .role = "nrUE",
            .subsystem = "phy_dl",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("tx_frame", "frame", "tx_slot", "slot", "rx_slot_type", "enum", "sidelink_mode", "boolean"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_ACTOR_DISPATCH] =
        {
            .name = "UE_DL_ACTOR_DISPATCH",
            .role = "nrUE",
            .subsystem = "scheduling",
            .event_class = "dispatch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("actor_count", "count", "preprocess_result", "sample", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_NTN_CONFIG_APPLY] =
        {
            .name = "UE_NTN_CONFIG_APPLY",
            .role = "nrUE",
            .subsystem = "ntn",
            .event_class = "configuration",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("duration_rx_to_tx", "slot", "timing_advance", "sample", "koffset", "slot", "target_cell", "boolean"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_SCHEDULE] =
        {
            .name = "UE_TX_SCHEDULE",
            .role = "nrUE",
            .subsystem = "scheduling",
            .event_class = "dispatch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("write_samples", "sample", "wait_previous", "count", "dlsch_waiters", "count", "duration_rx_to_tx", "slot"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_SLOT] =
        {
            .name = "UE_TX_SLOT",
            .role = "nrUE",
            .subsystem = "phy_ul",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("rx_frame", "frame", "rx_slot", "slot", "write_samples", "sample", "tx_slot_type", "enum"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_UL_INDICATION] =
        {
            .name = "UE_TX_UL_INDICATION",
            .role = "nrUE",
            .subsystem = "mac",
            .event_class = "callback",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("tx_action", "enum", "", "", "", "", "", ""),
            .flags_name = "path",
        },
    [OAI_PROFILE_EVENT_UE_TX_BARRIER_WAIT] =
        {
            .name = "UE_TX_BARRIER_WAIT",
            .role = "nrUE",
            .subsystem = "scheduling",
            .event_class = "wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("", "", "", "", "", "", "", ""),
            .flags_name = "path",
        },
    [OAI_PROFILE_EVENT_UE_TX_PHY_PROCEDURES] =
        {
            .name = "UE_TX_PHY_PROCEDURES",
            .role = "nrUE",
            .subsystem = "phy_ul",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("tx_action", "enum", "", "", "", "", "", ""),
            .flags_name = "path",
        },
    [OAI_PROFILE_EVENT_UE_TX_RU_WRITE] =
        {
            .name = "UE_TX_RU_WRITE",
            .role = "nrUE",
            .subsystem = "radio",
            .event_class = "dispatch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("write_samples", "sample", "sidelink_action", "boolean", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_RF_WRITE] =
        {
            .name = "UE_RF_WRITE",
            .role = "nrUE",
            .subsystem = "radio",
            .event_class = "io",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("requested_samples",
                       "sample",
                       "antenna_count",
                       "count",
                       "tx_flags",
                       "bitmask",
                       "returned_samples",
                       "sample"),
            .flags_name = "dummy_block",
        },
    [OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS] =
        {
            .name = "UE_TX_DEADLINE_MISS",
            .role = "nrUE",
            .subsystem = "timing",
            .event_class = "deadline",
            .default_kind = OAI_PROFILE_EVENT_KIND_INSTANT,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("current_time", "us", "deadline", "us", "lateness", "us", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_SLOT_INDICATION] =
        {
            .name = "GNB_SLOT_INDICATION",
            .role = "gNB",
            .subsystem = "mac",
            .event_class = "callback",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("frame_rx", "frame", "slot_rx", "slot", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RX_TRIGGER] =
        {
            .name = "GNB_RX_TRIGGER",
            .role = "gNB",
            .subsystem = "scheduling",
            .event_class = "dispatch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("frame_tx", "frame", "slot_tx", "slot", "tx_timestamp", "sample", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PHY_TX] =
        {
            .name = "GNB_PHY_TX",
            .role = "gNB",
            .subsystem = "phy_dl",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("tx_slot_type", "enum", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RU_TX] =
        {
            .name = "GNB_RU_TX",
            .role = "gNB",
            .subsystem = "radio",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("tx_timestamp", "sample", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_L1_TX_JOB] =
        {
            .name = "GNB_L1_TX_JOB",
            .role = "gNB",
            .subsystem = "scheduling",
            .event_class = "job",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("rx_frame", "frame", "rx_slot", "slot", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_L1_RX_JOB] =
        {
            .name = "GNB_L1_RX_JOB",
            .role = "gNB",
            .subsystem = "scheduling",
            .event_class = "job",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("", "", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PRACH_QUEUE_DRAIN] =
        {
            .name = "GNB_PRACH_QUEUE_DRAIN",
            .role = "gNB",
            .subsystem = "prach",
            .event_class = "queue",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("dequeued_items", "count", "rach_indications", "count", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PHASE_COMP] =
        {
            .name = "GNB_PHASE_COMP",
            .role = "gNB",
            .subsystem = "phy_ul",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("antenna_count", "count", "prb_count", "count", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PHY_UESPEC_RX] =
        {
            .name = "GNB_PHY_UESPEC_RX",
            .role = "gNB",
            .subsystem = "phy_ul",
            .event_class = "compute",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("rx_pdu_count", "count", "crc_count", "count", "uci_count", "count", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_UL_INDICATION] =
        {
            .name = "GNB_UL_INDICATION",
            .role = "gNB",
            .subsystem = "mac",
            .event_class = "callback",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("rx_pdu_count", "count", "crc_count", "count", "rach_count", "count", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RF_READ] =
        {
            .name = "GNB_RF_READ",
            .role = "gNB",
            .subsystem = "radio",
            .event_class = "io",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("requested_samples",
                       "sample",
                       "antenna_count",
                       "count",
                       "returned_samples",
                       "sample",
                       "device_timestamp",
                       "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RF_READ_ALIGN] =
        {
            .name = "GNB_RF_READ_ALIGN",
            .role = "gNB",
            .subsystem = "radio",
            .event_class = "io",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("requested_samples",
                       "sample",
                       "antenna_count",
                       "count",
                       "returned_samples",
                       "sample",
                       "device_timestamp",
                       "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RF_WRITE] =
        {
            .name = "GNB_RF_WRITE",
            .role = "gNB",
            .subsystem = "radio",
            .event_class = "io",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("requested_samples",
                       "sample",
                       "antenna_count",
                       "count",
                       "tx_flags",
                       "bitmask",
                       "returned_samples",
                       "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_DISPATCH_TO_START] =
        {
            .name = "UE_DL_DISPATCH_TO_START",
            .role = "nrUE",
            .subsystem = "scheduling",
            .event_class = "dispatch_wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("actor_count", "count", "rx_slot_type", "enum", "tx_frame", "frame", "tx_slot", "slot"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_DISPATCH_TO_START] =
        {
            .name = "UE_TX_DISPATCH_TO_START",
            .role = "nrUE",
            .subsystem = "scheduling",
            .event_class = "dispatch_wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("actor_count", "count", "tx_slot_type", "enum", "write_samples", "sample", "deadline", "us"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RU_SLOT_LOOP] =
        {
            .name = "GNB_RU_SLOT_LOOP",
            .role = "gNB",
            .subsystem = "orchestration",
            .event_class = "loop",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_BOUNDARY,
            AUX_FIELDS("tx_frame", "frame", "tx_slot", "slot", "slot_type", "enum", "tx_timestamp", "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_L1_TX_DISPATCH_TO_START] =
        {
            .name = "GNB_L1_TX_DISPATCH_TO_START",
            .role = "gNB",
            .subsystem = "scheduling",
            .event_class = "dispatch_wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("rx_frame", "frame", "rx_slot", "slot", "tx_timestamp", "sample", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_L1_RX_DISPATCH_TO_START] =
        {
            .name = "GNB_L1_RX_DISPATCH_TO_START",
            .role = "gNB",
            .subsystem = "scheduling",
            .event_class = "dispatch_wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tx_timestamp", "sample", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RU_RX_TTI_WAIT] =
        {
            .name = "GNB_RU_RX_TTI_WAIT",
            .role = "gNB",
            .subsystem = "scheduling",
            .event_class = "wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("slot_depth_index", "index", "waited", "boolean", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_MAC_SLOT_INDICATION] =
        {
            .name = "UE_MAC_SLOT_INDICATION",
            .role = "nrUE",
            .subsystem = "mac",
            .event_class = "callback",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("module_id", "index", "rx_slot_type", "enum", "tx_slot_type", "enum", "", ""),
            .flags_name = "downlink",
        },
    [OAI_PROFILE_EVENT_UE_MAC_DL_INDICATION] =
        {
            .name = "UE_MAC_DL_INDICATION",
            .role = "nrUE",
            .subsystem = "mac",
            .event_class = "callback",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("module_id", "index", "search_spaces", "count", "dlsch_codewords", "count", "", ""),
            .flags_name = "interface_present",
        },
    [OAI_PROFILE_EVENT_UE_DL_PBCH] =
        {
            .name = "UE_DL_PBCH",
            .role = "nrUE",
            .subsystem = "phy_dl_pbch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("sample_shift", "sample", "ssb_index", "index", "symbols", "count", "rx_antennas", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PBCH_FFT] =
        {
            .name = "UE_PBCH_FFT",
            .role = "nrUE",
            .subsystem = "phy_dl_pbch",
            .event_class = "fft",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "fft_size", "sample", "rx_antennas", "count", "ssb_index", "index"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PBCH_CHANNEL_ESTIMATION] =
        {
            .name = "UE_PBCH_CHANNEL_ESTIMATION",
            .role = "nrUE",
            .subsystem = "phy_dl_pbch",
            .event_class = "channel_estimation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("relative_symbol", "index", "ssb_index", "index", "rx_antennas", "count", "fft_size", "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PBCH_LLR] =
        {
            .name = "UE_PBCH_LLR",
            .role = "nrUE",
            .subsystem = "phy_dl_pbch",
            .event_class = "demodulation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ssb_symbol", "index", "ssb_index", "index", "coded_bits", "bit", "rx_antennas", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PBCH_DECODING] =
        {
            .name = "UE_PBCH_DECODING",
            .role = "nrUE",
            .subsystem = "phy_dl_pbch",
            .event_class = "decoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ssb_index", "index", "half_frame_bit", "bit", "decoded_ssb_index", "index", "symbol_offset", "symbol"),
            .flags_name = "decode_failed",
        },
    [OAI_PROFILE_EVENT_UE_PBCH_MEASUREMENTS] =
        {
            .name = "UE_PBCH_MEASUREMENTS",
            .role = "nrUE",
            .subsystem = "measurements",
            .event_class = "measurement",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ssb_index", "index", "rx_antennas", "count", "fft_size", "sample", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_PDCCH] =
        {
            .name = "UE_DL_PDCCH",
            .role = "nrUE",
            .subsystem = "phy_dl_pdcch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("search_spaces", "count", "rx_antennas", "count", "fft_size", "sample", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PDCCH_FFT] =
        {
            .name = "UE_PDCCH_FFT",
            .role = "nrUE",
            .subsystem = "phy_dl_pdcch",
            .event_class = "fft",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "fft_size", "sample", "rx_antennas", "count", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PDCCH_LLR] =
        {
            .name = "UE_PDCCH_LLR",
            .role = "nrUE",
            .subsystem = "phy_dl_pdcch",
            .event_class = "demodulation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "search_spaces", "count", "llr_per_symbol", "count", "monitoring_occasions", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PDCCH_DCI] =
        {
            .name = "UE_PDCCH_DCI",
            .role = "nrUE",
            .subsystem = "phy_dl_pdcch",
            .event_class = "decoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("search_spaces", "count", "llr_count", "count", "monitoring_occasions", "count", "max_symbols", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_CSI_IM] =
        {
            .name = "UE_DL_CSI_IM",
            .role = "nrUE",
            .subsystem = "phy_dl_csi",
            .event_class = "measurement",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("fft_symbols", "count", "rx_antennas", "count", "fft_size", "sample", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DL_CSI_RS] =
        {
            .name = "UE_DL_CSI_RS",
            .role = "nrUE",
            .subsystem = "phy_dl_csi",
            .event_class = "measurement",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("fft_symbols", "count", "rx_antennas", "count", "fft_size", "sample", "row", "index"),
            .flags_name = "resource_index",
        },
    [OAI_PROFILE_EVENT_UE_PDSCH_FFT] =
        {
            .name = "UE_PDSCH_FFT",
            .role = "nrUE",
            .subsystem = "phy_dl_pdsch",
            .event_class = "fft",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "fft_size", "sample", "rx_antennas", "count", "codeword", "index"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PDSCH_PROCEDURES] =
        {
            .name = "UE_PDSCH_PROCEDURES",
            .role = "nrUE",
            .subsystem = "phy_dl_pdsch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("codeword", "index", "coded_bits", "bit", "resource_blocks", "count", "harq_pid", "index"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_UE_PDSCH_CHANNEL_ESTIMATION] =
        {
            .name = "UE_PDSCH_CHANNEL_ESTIMATION",
            .role = "nrUE",
            .subsystem = "phy_dl_pdsch",
            .event_class = "channel_estimation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "layers", "count", "rx_antennas", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_PDSCH_DEMODULATION] =
        {
            .name = "UE_PDSCH_DEMODULATION",
            .role = "nrUE",
            .subsystem = "phy_dl_pdsch",
            .event_class = "demodulation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_blocks", "count", "modulation_order", "bit_per_symbol", "layers", "count"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_UE_DLSCH_PROCEDURES] =
        {
            .name = "UE_DLSCH_PROCEDURES",
            .role = "nrUE",
            .subsystem = "phy_dl_dlsch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("codeword", "index", "coded_bits", "bit", "resource_blocks", "count", "harq_pid", "index"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DLSCH_UNSCRAMBLING] =
        {
            .name = "UE_DLSCH_UNSCRAMBLING",
            .role = "nrUE",
            .subsystem = "phy_dl_dlsch",
            .event_class = "unscrambling",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("codeword", "index", "coded_bits", "bit", "rnti", "value", "scrambling_id", "value"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_DLSCH_DECODING] =
        {
            .name = "UE_DLSCH_DECODING",
            .role = "nrUE",
            .subsystem = "phy_dl_dlsch",
            .event_class = "decoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("codeword", "index", "coded_bits", "bit", "transport_block", "bit", "harq_pid", "index"),
            .flags_name = "decode_success",
        },
    [OAI_PROFILE_EVENT_UE_DLSCH_MAC_INDICATION] =
        {
            .name = "UE_DLSCH_MAC_INDICATION",
            .role = "nrUE",
            .subsystem = "mac",
            .event_class = "callback",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pdu_type", "enum", "transport_block", "bit", "harq_pid", "index", "codeword", "index"),
            .flags_name = "decode_success",
        },
    [OAI_PROFILE_EVENT_UE_TX_BUFFER_CLEAR] =
        {
            .name = "UE_TX_BUFFER_CLEAR",
            .role = "nrUE",
            .subsystem = "phy_ul",
            .event_class = "memory",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("frequency_samples", "sample", "tx_antennas", "count", "bytes", "byte", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_ULSCH] =
        {
            .name = "UE_TX_ULSCH",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks",
                       "count",
                       "symbols",
                       "count",
                       "modulation_order",
                       "bit_per_symbol",
                       "transport_block",
                       "byte"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_PRE_ENCODING] =
        {
            .name = "UE_ULSCH_PRE_ENCODING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "coding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("harq_pid", "index", "coded_bits", "bit", "transport_block", "byte", "resource_blocks", "count"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_ENCODING] =
        {
            .name = "UE_ULSCH_ENCODING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "coding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("harq_pid", "index", "coded_bits", "bit", "transport_block", "byte", "resource_blocks", "count"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_UCI] =
        {
            .name = "UE_ULSCH_UCI",
            .role = "nrUE",
            .subsystem = "phy_ul_uci",
            .event_class = "coding_mapping",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ack_bits", "bit", "csi1_bits", "bit", "csi2_bits", "bit", "coded_bits", "bit"),
            .flags_name = "present",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_SCRAMBLING] =
        {
            .name = "UE_ULSCH_SCRAMBLING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "scrambling",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("coded_bits", "bit", "rnti", "value", "scrambling_id", "value", "harq_pid", "index"),
            .flags_name = "uci_present",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_MODULATION] =
        {
            .name = "UE_ULSCH_MODULATION",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "modulation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("coded_bits", "bit", "modulation_order", "bit_per_symbol", "layers", "count", "resource_elements", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_TRANSFORM_PRECODING] =
        {
            .name = "UE_ULSCH_TRANSFORM_PRECODING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "transform_precoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "data_symbols", "count", "layers", "count", "dft_size", "sample"),
            .flags_name = "enabled",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_RE_MAPPING] =
        {
            .name = "UE_ULSCH_RE_MAPPING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "resource_mapping",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "layers", "count", "dmrs_symbols", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_PRECODING] =
        {
            .name = "UE_ULSCH_PRECODING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "precoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "layers", "count", "tx_antennas", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_SRS] =
        {
            .name = "UE_TX_SRS",
            .role = "nrUE",
            .subsystem = "phy_ul_srs",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "tx_antennas", "count", "config_index", "index"),
            .flags_name = "generated",
        },
    [OAI_PROFILE_EVENT_UE_TX_PUCCH] =
        {
            .name = "UE_TX_PUCCH",
            .role = "nrUE",
            .subsystem = "phy_ul_pucch",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("format", "enum", "resource_blocks", "count", "symbols", "count", "payload_bits", "bit"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_PHASE_ROTATION] =
        {
            .name = "UE_TX_PHASE_ROTATION",
            .role = "nrUE",
            .subsystem = "phy_ul",
            .event_class = "phase_rotation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("used_symbols", "count", "tx_antennas", "count", "resource_blocks", "count", "fft_size", "sample"),
            .flags_name = "disabled",
        },
    [OAI_PROFILE_EVENT_UE_TX_OFDM] =
        {
            .name = "UE_TX_OFDM",
            .role = "nrUE",
            .subsystem = "phy_ul",
            .event_class = "ofdm",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("used_symbols", "count", "tx_antennas", "count", "fft_size", "sample", "cyclic_prefix", "enum"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_TX_PRACH] =
        {
            .name = "UE_TX_PRACH",
            .role = "nrUE",
            .subsystem = "phy_ul_prach",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("preamble_index", "index", "format", "enum", "tx_power", "dBm", "digital_power", "dBW"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_UE_ULSCH_LAYER_MAPPING] =
        {
            .name = "UE_ULSCH_LAYER_MAPPING",
            .role = "nrUE",
            .subsystem = "phy_ul_ulsch",
            .event_class = "layer_mapping",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("modulated_symbols", "count", "layers", "count", "modulation_order", "bit_per_symbol", "coded_bits", "bit"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_LOCK_WAIT] =
        {
            .name = "GNB_MAC_SCHED_LOCK_WAIT",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "lock_wait",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("module_id", "index", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_PREPARE] =
        {
            .name = "GNB_MAC_SCHED_PREPARE",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "preparation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("component_carriers", "count", "slots_per_frame", "count", "beam_mode", "enum", "beams_per_period", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_STATS] =
        {
            .name = "GNB_MAC_SCHED_STATS",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "observability",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("stats_max_ue", "count", "frame_period", "frame", "", "", "", ""),
            .flags_name = "enabled_after",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_TIMERS] =
        {
            .name = "GNB_MAC_SCHED_TIMERS",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "timers",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("module_id", "index", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_COMMON] =
        {
            .name = "GNB_MAC_SCHED_COMMON",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "common_channels",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("dl_pdus", "count", "tx_pdus", "count", "sa_mode", "boolean", "", ""),
            .flags_name = "prach_ready_or_phy_test",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_PRACH] =
        {
            .name = "GNB_MAC_SCHED_PRACH",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "random_access",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("target_frame", "frame", "target_slot", "slot", "prach_length", "slot", "ntn_koffset", "slot"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_CSI_SRS] =
        {
            .name = "GNB_MAC_SCHED_CSI_SRS",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "csi_srs",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("dl_pdus", "count", "module_id", "index", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_RA] =
        {
            .name = "GNB_MAC_SCHED_RA",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "random_access",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ul_dci_pdus", "count", "dl_pdus", "count", "tx_pdus", "count", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_ULSCH] =
        {
            .name = "GNB_MAC_SCHED_ULSCH",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "ulsch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ul_dci_pdus", "count", "module_id", "index", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_DLSCH] =
        {
            .name = "GNB_MAC_SCHED_DLSCH",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "dlsch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("dl_pdus", "count", "tx_pdus", "count", "module_id", "index", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_PUCCH] =
        {
            .name = "GNB_MAC_SCHED_PUCCH",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "pucch",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("module_id", "index", "", "", "", "", "", ""),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_MAC_SCHED_FINALIZE] =
        {
            .name = "GNB_MAC_SCHED_FINALIZE",
            .role = "gNB",
            .subsystem = "mac_scheduler",
            .event_class = "publication",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ul_pdus", "count", "ul_groups", "count", "dl_pdus", "count", "tx_pdus", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_BUFFER_CLEAR] =
        {
            .name = "GNB_TX_BUFFER_CLEAR",
            .role = "gNB",
            .subsystem = "phy_dl",
            .event_class = "memory",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("frequency_samples", "sample", "tx_antennas", "count", "bytes", "byte", "dl_pdus", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_PRS] =
        {
            .name = "GNB_TX_PRS",
            .role = "gNB",
            .subsystem = "phy_dl_prs",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_id", "index", "repetition_index", "index", "prs_slot", "slot", "repetitions", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_PDCCH] =
        {
            .name = "GNB_TX_PDCCH",
            .role = "gNB",
            .subsystem = "phy_dl_pdcch",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pdu_index", "index", "source", "enum", "source_pdus", "count", "dl_pdus", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_SSB] =
        {
            .name = "GNB_TX_SSB",
            .role = "gNB",
            .subsystem = "phy_dl_ssb",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pdu_index", "index", "dl_pdus", "count", "tx_antennas", "count", "fft_size", "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_CSI_RS] =
        {
            .name = "GNB_TX_CSI_RS",
            .role = "gNB",
            .subsystem = "phy_dl_csi",
            .event_class = "generation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pdu_index", "index", "dl_pdus", "count", "tx_antennas", "count", "fft_size", "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_PDSCH] =
        {
            .name = "GNB_TX_PDSCH",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pdsch_pdus", "count", "tx_pdus", "count", "tx_antennas", "count", "fft_size", "sample"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_TX_PHASE_ROTATION] =
        {
            .name = "GNB_TX_PHASE_ROTATION",
            .role = "gNB",
            .subsystem = "phy_dl",
            .event_class = "phase_rotation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tx_antennas", "count", "frequency_samples", "sample", "resource_blocks", "count", "symbols", "count"),
            .flags_name = "enabled",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_ENCODING] =
        {
            .name = "GNB_PDSCH_ENCODING",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "coding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pdsch_pdus", "count", "transport_blocks", "count", "output_bits", "bit", "output_bytes", "byte"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_SCRAMBLING] =
        {
            .name = "GNB_PDSCH_SCRAMBLING",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "scrambling",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("codeword", "index", "coded_bits", "bit", "rnti", "value", "scrambling_id", "value"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_MODULATION] =
        {
            .name = "GNB_PDSCH_MODULATION",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "modulation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("codeword",
                       "index",
                       "coded_bits",
                       "bit",
                       "modulation_order",
                       "bit_per_symbol",
                       "modulated_symbols",
                       "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_LAYER_MAPPING] =
        {
            .name = "GNB_PDSCH_LAYER_MAPPING",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "layer_mapping",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("coded_bits", "bit", "resource_elements", "count", "layers", "count", "resource_blocks", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_SYMBOL_PROCESSING] =
        {
            .name = "GNB_PDSCH_SYMBOL_PROCESSING",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "dispatch_join",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tasks", "count", "symbols", "count", "symbols_per_task", "count", "resource_blocks", "count"),
            .flags_name = "thread_pool_enabled",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_SYMBOL_TASK] =
        {
            .name = "GNB_PDSCH_SYMBOL_TASK",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "worker",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("start_symbol", "index", "symbols", "count", "resource_blocks", "count", "layers", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_RESOURCE_MAPPING] =
        {
            .name = "GNB_PDSCH_RESOURCE_MAPPING",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "resource_mapping",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_blocks", "count", "layers", "count", "dmrs", "boolean"),
            .flags_name = "ptrs",
        },
    [OAI_PROFILE_EVENT_GNB_PDSCH_PRECODING] =
        {
            .name = "GNB_PDSCH_PRECODING",
            .role = "gNB",
            .subsystem = "phy_dl_pdsch",
            .event_class = "precoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_blocks", "count", "layers", "count", "logical_ports", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RU_RX_FRONTEND] =
        {
            .name = "GNB_RU_RX_FRONTEND",
            .role = "gNB",
            .subsystem = "ru_rx",
            .event_class = "frontend",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("rx_antennas", "count", "fft_size", "sample", "symbols", "count", "slot_depth", "count"),
            .flags_name = "callback_present",
        },
    [OAI_PROFILE_EVENT_GNB_RU_PRACH_FRONTEND] =
        {
            .name = "GNB_RU_PRACH_FRONTEND",
            .role = "gNB",
            .subsystem = "ru_rx_prach",
            .event_class = "frontend",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("occasion_index", "index", "frequency_index", "index", "start_symbol", "index", "sequence_length", "enum"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RU_TX_PRECODING] =
        {
            .name = "GNB_RU_TX_PRECODING",
            .role = "gNB",
            .subsystem = "ru_tx",
            .event_class = "precoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tx_antennas", "count", "fft_size", "sample", "symbols", "count", "ru_index", "index"),
            .flags_name = "callback_present",
        },
    [OAI_PROFILE_EVENT_GNB_RU_TX_OFDM] =
        {
            .name = "GNB_RU_TX_OFDM",
            .role = "gNB",
            .subsystem = "ru_tx",
            .event_class = "ofdm",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tx_antennas", "count", "fft_size", "sample", "symbols", "count", "ru_index", "index"),
            .flags_name = "callback_present",
        },
    [OAI_PROFILE_EVENT_GNB_RU_TX_SOUTH] =
        {
            .name = "GNB_RU_TX_SOUTH",
            .role = "gNB",
            .subsystem = "ru_tx",
            .event_class = "fronthaul_or_rf",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tx_timestamp", "sample", "tx_antennas", "count", "ru_index", "index", "", ""),
            .flags_name = "callback_present",
        },
    [OAI_PROFILE_EVENT_GNB_RU_TX_NORTH] =
        {
            .name = "GNB_RU_TX_NORTH",
            .role = "gNB",
            .subsystem = "ru_tx",
            .event_class = "fronthaul",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ru_index", "index", "", "", "", "", "", ""),
            .flags_name = "callback_present",
        },
    [OAI_PROFILE_EVENT_GNB_RX_NOISE_MEASUREMENT] =
        {
            .name = "GNB_RX_NOISE_MEASUREMENT",
            .role = "gNB",
            .subsystem = "phy_ul",
            .event_class = "measurement",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("first_symbol", "index", "symbols", "count", "pucch_pdus", "count", "pusch_pdus", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RX_PUCCH] =
        {
            .name = "GNB_RX_PUCCH",
            .role = "gNB",
            .subsystem = "phy_ul_pucch",
            .event_class = "decoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("format", "enum", "resource_blocks", "count", "symbols", "count", "rnti", "value"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RX_PUSCH_FRONTEND] =
        {
            .name = "GNB_RX_PUSCH_FRONTEND",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "modulation_order", "bit_per_symbol", "layers", "count"),
            .flags_name = "dtx",
        },
    [OAI_PROFILE_EVENT_GNB_RX_ULSCH_DECODING] =
        {
            .name = "GNB_RX_ULSCH_DECODING",
            .role = "gNB",
            .subsystem = "phy_ul_ulsch",
            .event_class = "decoding",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("pusch_pdus", "count", "max_iterations", "count", "crc_before", "count", "rx_pdus_before", "count"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_GNB_RX_ULSCH_CRC] =
        {
            .name = "GNB_RX_ULSCH_CRC",
            .role = "gNB",
            .subsystem = "phy_ul_ulsch",
            .event_class = "crc_indication",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("ulsch_id", "index", "transport_block", "byte", "segments", "count", "harq_round", "count"),
            .flags_name = "crc_valid",
        },
    [OAI_PROFILE_EVENT_GNB_RX_SRS] =
        {
            .name = "GNB_RX_SRS",
            .role = "gNB",
            .subsystem = "phy_ul_srs",
            .event_class = "procedure",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "antenna_ports", "count", "rx_streams", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_RX_PRACH_DETECTION] =
        {
            .name = "GNB_RX_PRACH_DETECTION",
            .role = "gNB",
            .subsystem = "phy_ul_prach",
            .event_class = "detection",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("queue_index", "index", "rapid_pdus_before", "count", "rapid_pdus_after", "count", "antenna_start", "index"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_SRS_CHANNEL_ESTIMATION] =
        {
            .name = "GNB_SRS_CHANNEL_ESTIMATION",
            .role = "gNB",
            .subsystem = "phy_ul_srs",
            .event_class = "channel_estimation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "antenna_ports", "count", "rx_streams", "count"),
            .flags_name = "detected",
        },
    [OAI_PROFILE_EVENT_GNB_SRS_REPORT] =
        {
            .name = "GNB_SRS_REPORT",
            .role = "gNB",
            .subsystem = "phy_ul_srs",
            .event_class = "report",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("usage", "enum", "report_type", "enum", "report_bytes", "byte", "snr", "dB"),
            .flags_name = "detected",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_CHANNEL_ESTIMATION] =
        {
            .name = "GNB_PUSCH_CHANNEL_ESTIMATION",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "channel_estimation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("resource_blocks", "count", "symbols", "count", "dmrs_symbols", "count", "layers", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_INITIALIZATION] =
        {
            .name = "GNB_PUSCH_INITIALIZATION",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "initialization",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("coded_bits", "bit", "resource_elements", "count", "spatial_streams", "count", "layers", "count"),
            .flags_name = "ptrs",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_SYMBOL_PROCESSING] =
        {
            .name = "GNB_PUSCH_SYMBOL_PROCESSING",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "dispatch_join",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("tasks", "count", "symbols", "count", "symbols_per_task", "count", "resource_blocks", "count"),
            .flags_name = "ptrs_inline",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_SYMBOL_TASK] =
        {
            .name = "GNB_PUSCH_SYMBOL_TASK",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "worker",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("start_symbol", "index", "symbols", "count", "resource_blocks", "count", "layers", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_EXTRACTION] =
        {
            .name = "GNB_PUSCH_EXTRACTION",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "extraction",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_elements", "count", "spatial_streams", "count", "layers", "count"),
            .flags_name = "dmrs",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_CHANNEL_COMPENSATION] =
        {
            .name = "GNB_PUSCH_CHANNEL_COMPENSATION",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "equalization",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_elements", "count", "modulation_order", "bit_per_symbol", "layers", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_LLR] =
        {
            .name = "GNB_PUSCH_LLR",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "demodulation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_elements", "count", "modulation_order", "bit_per_symbol", "layers", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_LAYER_DEMAPPING] =
        {
            .name = "GNB_PUSCH_LAYER_DEMAPPING",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "layer_demapping",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "resource_elements", "count", "layers", "count", "modulation_order", "bit_per_symbol"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_GNB_PUSCH_UNSCRAMBLING] =
        {
            .name = "GNB_PUSCH_UNSCRAMBLING",
            .role = "gNB",
            .subsystem = "phy_ul_pusch",
            .event_class = "unscrambling",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("symbol", "index", "soft_bits", "count", "rnti", "value", "scrambling_id", "value"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_DECODER_SEGMENT] =
        {
            .name = "LDPC_DECODER_SEGMENT",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_decoder",
            .event_class = "segment",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index", "index", "segment", "index", "segments", "count", "transport_block", "bit"),
            .flags_name = "decode_success",
        },
    [OAI_PROFILE_EVENT_LDPC_DECODER_DEINTERLEAVE] =
        {
            .name = "LDPC_DECODER_DEINTERLEAVE",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_decoder",
            .event_class = "deinterleave",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index",
                       "index",
                       "segment",
                       "index",
                       "input_llrs",
                       "count",
                       "modulation_order",
                       "bit_per_symbol"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_DECODER_RATE_RECOVERY] =
        {
            .name = "LDPC_DECODER_RATE_RECOVERY",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_decoder",
            .event_class = "rate_recovery",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index", "index", "segment", "index", "input_llrs", "count", "redundancy_version", "index"),
            .flags_name = "failed",
        },
    [OAI_PROFILE_EVENT_LDPC_DECODER_SEGMENT_PREPARATION] =
        {
            .name = "LDPC_DECODER_SEGMENT_PREPARATION",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_decoder",
            .event_class = "preparation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index", "index", "segment", "index", "code_block", "bit", "lifting_size", "bit"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_DECODER_KERNEL] =
        {
            .name = "LDPC_DECODER_KERNEL",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_decoder",
            .event_class = "iterative_decode",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index", "index", "segment", "index", "base_graph", "index", "iterations", "count"),
            .flags_name = "decode_success",
        },
    [OAI_PROFILE_EVENT_LDPC_ENCODER_DISPATCH_JOIN] =
        {
            .name = "LDPC_ENCODER_DISPATCH_JOIN",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_encoder",
            .event_class = "dispatch_join",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_blocks", "count", "tasks", "count", "segments", "count", "max_segment_output", "bit"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_ENCODER_TASK] =
        {
            .name = "LDPC_ENCODER_TASK",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_encoder",
            .event_class = "segment_group",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index", "index", "first_segment", "index", "task_segments", "count", "segments", "count"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_ENCODER_KERNEL] =
        {
            .name = "LDPC_ENCODER_KERNEL",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_encoder",
            .event_class = "encode",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index", "index", "first_segment", "index", "task_segments", "count", "base_graph", "index"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_ENCODER_RATE_MATCHING] =
        {
            .name = "LDPC_ENCODER_RATE_MATCHING",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_encoder",
            .event_class = "rate_matching",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index",
                       "index",
                       "first_segment",
                       "index",
                       "max_segment_output",
                       "bit",
                       "redundancy_version",
                       "index"),
            .flags_name = "",
        },
    [OAI_PROFILE_EVENT_LDPC_ENCODER_INTERLEAVING] =
        {
            .name = "LDPC_ENCODER_INTERLEAVING",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_encoder",
            .event_class = "interleave",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_block_index",
                       "index",
                       "first_segment",
                       "index",
                       "segment_output",
                       "bit",
                       "second_segment_output",
                       "bit"),
            .flags_name = "output_size_shift",
        },
    [OAI_PROFILE_EVENT_LDPC_ENCODER_CONCATENATION] =
        {
            .name = "LDPC_ENCODER_CONCATENATION",
            .role = "nrUE/gNB",
            .subsystem = "ldpc_encoder",
            .event_class = "concatenation",
            .default_kind = OAI_PROFILE_EVENT_KIND_DURATION,
            .detail = OAI_PROFILE_DETAIL_STAGE,
            AUX_FIELDS("transport_blocks", "count", "tasks", "count", "segments", "count", "max_segment_output", "bit"),
            .flags_name = "",
        },
};

#undef AUX_FIELDS

static oai_profile_thread_buffer_t thread_buffers[OAI_PROFILE_MAX_THREADS];
static pthread_mutex_t registry_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lifecycle_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t writer_thread;
static bool profiler_initialized;
static bool writer_started;
static bool profiler_shutdown_requested;
static __thread int thread_buffer_index = -1;
static __thread oai_profile_context_t thread_context = {
    .absolute_slot = OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN,
    .correlation_id = 0,
    .parent_id = 0,
};
static __thread uint64_t thread_span_stack[OAI_PROFILE_MAX_NESTING_DEPTH];
static __thread uint16_t thread_span_depth;
static uint32_t global_buffer_records = OAI_PROFILE_DEFAULT_BUFFER_RECORDS;
static uint32_t global_flush_us = OAI_PROFILE_DEFAULT_FLUSH_US;
static uint32_t global_host_metrics_us = OAI_PROFILE_DEFAULT_HOST_METRICS_US;
static uint64_t global_seq;
static uint64_t global_correlation_seq;
static uint64_t counter_hz;
static uint64_t profile_start_realtime_ns;
static uint64_t previous_cpu_total;
static uint64_t previous_cpu_idle;
static bool previous_cpu_times_valid;
static uid_t output_uid = (uid_t)-1;
static gid_t output_gid = (gid_t)-1;
static char output_dir[PATH_MAX];
static char profile_root[PATH_MAX];
static char profile_repository_root[PATH_MAX];
static char config_archive_dir[PATH_MAX];
static char profile_role[OAI_PROFILE_COMPONENT_LEN];
static char profile_hostname[OAI_PROFILE_COMPONENT_LEN];
static char profile_run_id[OAI_PROFILE_COMPONENT_LEN];
static char profile_experiment_id[OAI_PROFILE_COMPONENT_LEN];
static char profile_config_source[PATH_MAX];
static FILE *events_file;
static FILE *sync_file;
static FILE *drops_file;
static FILE *settings_file;
static FILE *host_metrics_file;
static pthread_mutex_t settings_mutex = PTHREAD_MUTEX_INITIALIZER;
static int rpi_mailbox_fd = -1;

const char *oai_profiler_event_name(oai_profile_event_id_t event_id)
{
  const oai_profile_event_descriptor_t *descriptor = oai_profiler_event_descriptor(event_id);
  if (descriptor == NULL)
    return "UNKNOWN";
  return descriptor->name;
}

const oai_profile_event_descriptor_t *oai_profiler_event_descriptor(oai_profile_event_id_t event_id)
{
  if (event_id <= OAI_PROFILE_EVENT_UNSPEC || event_id >= OAI_PROFILE_EVENT_MAX || event_descriptors[event_id].name == NULL)
    return NULL;
  return &event_descriptors[event_id];
}

const char *oai_profiler_event_kind_name(oai_profile_event_kind_t kind)
{
  switch (kind) {
    case OAI_PROFILE_EVENT_KIND_DURATION:
      return "duration";
    case OAI_PROFILE_EVENT_KIND_INSTANT:
      return "instant";
    default:
      return "unknown";
  }
}

static const char *event_detail_name(oai_profile_detail_t detail)
{
  switch (detail) {
    case OAI_PROFILE_DETAIL_BOUNDARY:
      return "boundary";
    case OAI_PROFILE_DETAIL_STAGE:
      return "stage";
    case OAI_PROFILE_DETAIL_KERNEL:
      return "kernel";
    default:
      return "unknown";
  }
}

static uint64_t read_counter_hz(void)
{
#if defined(__aarch64__)
  uint64_t hz = 0;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(hz));
  return hz;
#else
  return (uint64_t)(get_cpu_freq_GHz() * 1000000000.0);
#endif
}

static uint32_t parse_u32_or_default(const char *value, uint32_t default_value)
{
  if (value == NULL || value[0] == '\0')
    return default_value;
  char *end = NULL;
  errno = 0;
  unsigned long parsed = strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX)
    return default_value;
  return (uint32_t)parsed;
}

static uint64_t timespec_to_ns(const struct timespec *ts)
{
  return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

static bool parse_env_enable(bool cli_enabled)
{
  const char *env = getenv("OAI_PROFILE");
  if (env == NULL || env[0] == '\0')
    return cli_enabled;
  if (!strcmp(env, "0") || !strcasecmp(env, "false") || !strcasecmp(env, "off"))
    return false;
  return true;
}

static void detect_output_owner(void)
{
  const char *uid_text = getenv("SUDO_UID");
  const char *gid_text = getenv("SUDO_GID");
  if (uid_text == NULL || gid_text == NULL)
    return;

  char *uid_end = NULL;
  char *gid_end = NULL;
  errno = 0;
  unsigned long uid_value = strtoul(uid_text, &uid_end, 10);
  unsigned long gid_value = strtoul(gid_text, &gid_end, 10);
  if (errno == 0 && uid_end != uid_text && *uid_end == '\0' && gid_end != gid_text && *gid_end == '\0') {
    output_uid = (uid_t)uid_value;
    output_gid = (gid_t)gid_value;
  }
}

static void set_output_owner(const char *path)
{
  if (output_uid != (uid_t)-1 && chown(path, output_uid, output_gid) != 0)
    LOG_D(UTIL, "OAI profiler could not set ownership of %s: %s\n", path, strerror(errno));
}

static int mkdir_owned(const char *path)
{
  if (mkdir(path, 0775) == 0) {
    set_output_owner(path);
    return 0;
  }
  return errno == EEXIST ? 0 : -1;
}

static int mkdir_p(const char *path)
{
  char tmp[PATH_MAX];
  if (path == NULL || path[0] == '\0')
    return -1;
  int ret = snprintf(tmp, sizeof(tmp), "%s", path);
  if (ret < 0 || (size_t)ret >= sizeof(tmp))
    return -1;
  size_t len = strlen(tmp);
  if (len == 0)
    return -1;
  if (tmp[len - 1] == '/')
    tmp[len - 1] = '\0';
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir_owned(tmp) != 0)
        return -1;
      *p = '/';
    }
  }
  if (mkdir_owned(tmp) != 0)
    return -1;
  return 0;
}

static void sanitize_component(const char *input, char *output, size_t output_size, const char *fallback)
{
  size_t out = 0;
  if (input != NULL) {
    for (size_t i = 0; input[i] != '\0' && out + 1 < output_size; i++) {
      const unsigned char c = (unsigned char)input[i];
      output[out++] = isalnum(c) || c == '-' || c == '_' || c == '.' ? (char)c : '_';
    }
  }
  if (out == 0 && fallback != NULL) {
    int ret = snprintf(output, output_size, "%s", fallback);
    out = ret > 0 && (size_t)ret < output_size ? (size_t)ret : 0;
  }
  output[out] = '\0';
}

static void set_profile_identity(const char *process_name)
{
  const char *role = process_name;
  if (process_name != NULL && strstr(process_name, "uesoftmodem") != NULL)
    role = "nrUE";
  else if (process_name != NULL && strstr(process_name, "softmodem") != NULL)
    role = "gNB";
  sanitize_component(role, profile_role, sizeof(profile_role), "softmodem");

  char hostname[256] = {0};
  if (gethostname(hostname, sizeof(hostname) - 1) != 0)
    snprintf(hostname, sizeof(hostname), "unknown-host");
  sanitize_component(hostname, profile_hostname, sizeof(profile_hostname), "unknown-host");

  const char *experiment = getenv("OAI_PROFILE_EXPERIMENT_ID");
  sanitize_component(experiment, profile_experiment_id, sizeof(profile_experiment_id), "");
}

static int get_invoking_home(char *home, size_t home_size)
{
  const char *sudo_user = getenv("SUDO_USER");
  struct passwd *pw = sudo_user != NULL && sudo_user[0] != '\0' ? getpwnam(sudo_user) : getpwuid(getuid());
  const char *path = pw != NULL ? pw->pw_dir : getenv("HOME");
  if (path == NULL || path[0] == '\0')
    return -1;
  int ret = snprintf(home, home_size, "%s", path);
  return ret < 0 || (size_t)ret >= home_size ? -1 : 0;
}

static int resolve_default_profile_root(char *root, size_t root_size)
{
  char executable[PATH_MAX] = {0};
  ssize_t len = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
  if (len > 0) {
    executable[len] = '\0';
    char *build_marker = strstr(executable, "/cmake_targets/ran_build/build/");
    if (build_marker != NULL) {
      *build_marker = '\0';
      if (snprintf(profile_repository_root, sizeof(profile_repository_root), "%s", executable)
          >= (int)sizeof(profile_repository_root))
        return -1;
      char *repository_parent = strrchr(executable, '/');
      if (repository_parent != NULL) {
        *repository_parent = '\0';
        int ret = snprintf(root, root_size, "%s/PerformanceProfiles", executable);
        if (ret >= 0 && (size_t)ret < root_size)
          return 0;
      }
    }
  }

  char home[PATH_MAX];
  if (get_invoking_home(home, sizeof(home)) != 0)
    return -1;
  int ret = snprintf(root, root_size, "%s/Documents/OpenAirInterface/PerformanceProfiles", home);
  return ret < 0 || (size_t)ret >= root_size ? -1 : 0;
}

static int format_run_timestamp(char *timestamp, size_t timestamp_size)
{
  time_t now = time(NULL);
  struct tm local = {0};
  if (now == (time_t)-1 || localtime_r(&now, &local) == NULL)
    return -1;
  return strftime(timestamp, timestamp_size, "%Y-%m-%d_%H-%M-%S", &local) == 0 ? -1 : 0;
}

static bool path_has_profile_output(const char *path)
{
  static const char *const names[] = {"events.csv", "metadata.txt", "sync.csv", "drops.csv"};
  char candidate[PATH_MAX];
  for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
    int ret = snprintf(candidate, sizeof(candidate), "%s/%s", path, names[i]);
    if (ret >= 0 && (size_t)ret < sizeof(candidate) && access(candidate, F_OK) == 0)
      return true;
  }
  return false;
}

static int prepare_profile_paths(const char *process_name, const char *requested_dir)
{
  detect_output_owner();
  set_profile_identity(process_name);
  profile_repository_root[0] = '\0';

  const char *env_root = getenv("OAI_PROFILE_ROOT");
  if (env_root != NULL && env_root[0] != '\0') {
    int ret = snprintf(profile_root, sizeof(profile_root), "%s", env_root);
    if (ret < 0 || (size_t)ret >= sizeof(profile_root))
      return -1;
  } else if (resolve_default_profile_root(profile_root, sizeof(profile_root)) != 0) {
    return -1;
  }

  if (mkdir_p(profile_root) != 0)
    return -1;
  char configs_dir[PATH_MAX];
  int ret = snprintf(configs_dir, sizeof(configs_dir), "%s/configs", profile_root);
  if (ret < 0 || (size_t)ret >= sizeof(configs_dir) || mkdir_p(configs_dir) != 0)
    return -1;

  char gnb_configs[PATH_MAX];
  char ue_configs[PATH_MAX];
  ret = snprintf(gnb_configs, sizeof(gnb_configs), "%s/gNB", configs_dir);
  if (ret < 0 || (size_t)ret >= sizeof(gnb_configs) || mkdir_p(gnb_configs) != 0)
    return -1;
  ret = snprintf(ue_configs, sizeof(ue_configs), "%s/nrUE", configs_dir);
  if (ret < 0 || (size_t)ret >= sizeof(ue_configs) || mkdir_p(ue_configs) != 0)
    return -1;
  const char *role_configs = strcmp(profile_role, "nrUE") == 0 ? ue_configs : gnb_configs;
  if (snprintf(config_archive_dir, sizeof(config_archive_dir), "%s", role_configs) >= (int)sizeof(config_archive_dir))
    return -1;

  char timestamp[32];
  if (format_run_timestamp(timestamp, sizeof(timestamp)) != 0)
    return -1;
  ret = snprintf(profile_run_id, sizeof(profile_run_id), "%s_%s_%s", timestamp, profile_role, profile_hostname);
  if (ret < 0 || (size_t)ret >= sizeof(profile_run_id))
    return -1;

  const char *env_dir = getenv("OAI_PROFILE_DIR");
  const char *explicit_dir = env_dir != NULL && env_dir[0] != '\0' ? env_dir : requested_dir;
  if (explicit_dir != NULL && explicit_dir[0] != '\0') {
    ret = snprintf(output_dir, sizeof(output_dir), "%s", explicit_dir);
    if (ret < 0 || (size_t)ret >= sizeof(output_dir) || path_has_profile_output(output_dir)) {
      errno = EEXIST;
      return -1;
    }
    return mkdir_p(output_dir);
  }

  for (unsigned int collision = 0; collision < 1000; collision++) {
    ret = collision == 0 ? snprintf(output_dir, sizeof(output_dir), "%s/%s", profile_root, profile_run_id)
                         : snprintf(output_dir, sizeof(output_dir), "%s/%s_%02u", profile_root, profile_run_id, collision);
    if (ret < 0 || (size_t)ret >= sizeof(output_dir))
      return -1;
    if (mkdir(output_dir, 0775) == 0) {
      set_output_owner(output_dir);
      const char *directory_name = strrchr(output_dir, '/');
      int id_ret = snprintf(profile_run_id, sizeof(profile_run_id), "%s", directory_name ? directory_name + 1 : output_dir);
      if (id_ret < 0 || (size_t)id_ret >= sizeof(profile_run_id))
        return -1;
      return 0;
    }
    if (errno != EEXIST)
      return -1;
  }
  errno = EEXIST;
  return -1;
}

static int open_profile_file(FILE **file, const char *name, const char *mode)
{
  char path[PATH_MAX];
  int ret = snprintf(path, sizeof(path), "%s/%s", output_dir, name);
  if (ret < 0 || (size_t)ret >= sizeof(path))
    return -1;
  *file = fopen(path, mode);
  if (*file == NULL)
    return -1;
  if (output_uid != (uid_t)-1 && fchown(fileno(*file), output_uid, output_gid) != 0)
    LOG_D(UTIL, "OAI profiler could not set ownership of %s: %s\n", path, strerror(errno));
  setvbuf(*file, NULL, _IOFBF, 1 << 20);
  return 0;
}

static void write_event_catalog(void)
{
  FILE *file = NULL;
  if (open_profile_file(&file, "event_catalog.csv", "w") != 0)
    return;
  fprintf(file,
          "schema_version,event_id,event_name,role,subsystem,event_class,default_kind,detail_level,"
          "aux0_name,aux0_unit,aux1_name,aux1_unit,aux2_name,aux2_unit,aux3_name,aux3_unit,flags_name\n");
  for (int i = 1; i < OAI_PROFILE_EVENT_MAX; i++) {
    const oai_profile_event_descriptor_t *descriptor = oai_profiler_event_descriptor(i);
    if (descriptor == NULL)
      continue;
    fprintf(file,
            "%u,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
            OAI_PROFILE_SCHEMA_VERSION,
            i,
            descriptor->name,
            descriptor->role,
            descriptor->subsystem,
            descriptor->event_class,
            oai_profiler_event_kind_name(descriptor->default_kind),
            event_detail_name(descriptor->detail),
            descriptor->aux_name[0],
            descriptor->aux_unit[0],
            descriptor->aux_name[1],
            descriptor->aux_unit[1],
            descriptor->aux_name[2],
            descriptor->aux_unit[2],
            descriptor->aux_name[3],
            descriptor->aux_unit[3],
            descriptor->flags_name);
  }
  fclose(file);
}

static void find_config_source(int argc, char **argv)
{
  profile_config_source[0] = '\0';
  for (int i = 1; i < argc; i++) {
    const char *value = NULL;
    if (strcmp(argv[i], "-O") == 0 && i + 1 < argc)
      value = argv[i + 1];
    else if (strncmp(argv[i], "-O", 2) == 0 && argv[i][2] != '\0')
      value = argv[i] + 2;
    if (value != NULL) {
      snprintf(profile_config_source, sizeof(profile_config_source), "%s", value);
      return;
    }
  }
}

static bool read_first_line(const char *path, char *line, size_t line_size)
{
  FILE *file = fopen(path, "r");
  if (file == NULL)
    return false;
  const bool valid = fgets(line, line_size, file) != NULL;
  fclose(file);
  if (!valid)
    return false;
  line[strcspn(line, "\r\n")] = '\0';
  return true;
}

static bool resolve_git_directory(char *git_directory, size_t git_directory_size)
{
  if (profile_repository_root[0] == '\0')
    return false;
  char dot_git[PATH_MAX];
  int ret = snprintf(dot_git, sizeof(dot_git), "%s/.git", profile_repository_root);
  if (ret < 0 || (size_t)ret >= sizeof(dot_git))
    return false;
  struct stat info = {0};
  if (stat(dot_git, &info) != 0)
    return false;
  if (S_ISDIR(info.st_mode))
    return snprintf(git_directory, git_directory_size, "%s", dot_git) < (int)git_directory_size;

  char line[PATH_MAX];
  if (!read_first_line(dot_git, line, sizeof(line)) || strncmp(line, "gitdir: ", 8) != 0)
    return false;
  const char *path = line + 8;
  if (path[0] == '/')
    ret = snprintf(git_directory, git_directory_size, "%s", path);
  else
    ret = snprintf(git_directory, git_directory_size, "%s/%s", profile_repository_root, path);
  return ret >= 0 && (size_t)ret < git_directory_size;
}

static bool read_packed_ref(const char *git_directory, const char *reference, char *commit, size_t commit_size)
{
  char packed_refs[PATH_MAX];
  int ret = snprintf(packed_refs, sizeof(packed_refs), "%s/packed-refs", git_directory);
  if (ret < 0 || (size_t)ret >= sizeof(packed_refs))
    return false;
  FILE *file = fopen(packed_refs, "r");
  if (file == NULL)
    return false;
  char line[PATH_MAX + 128];
  bool found = false;
  while (fgets(line, sizeof(line), file) != NULL) {
    char hash[128];
    char name[PATH_MAX];
    if (line[0] != '#' && line[0] != '^' && sscanf(line, "%127s %4095s", hash, name) == 2 && strcmp(name, reference) == 0) {
      found = snprintf(commit, commit_size, "%s", hash) < (int)commit_size;
      break;
    }
  }
  fclose(file);
  return found;
}

static void read_runtime_git_identity(char *branch, size_t branch_size, char *commit, size_t commit_size)
{
  snprintf(branch, branch_size, "unavailable");
  snprintf(commit, commit_size, "unavailable");
  char git_directory[PATH_MAX];
  char head_path[PATH_MAX];
  char head[PATH_MAX];
  if (!resolve_git_directory(git_directory, sizeof(git_directory))
      || snprintf(head_path, sizeof(head_path), "%s/HEAD", git_directory) >= (int)sizeof(head_path)
      || !read_first_line(head_path, head, sizeof(head)))
    return;
  if (strncmp(head, "ref: ", 5) != 0) {
    snprintf(branch, branch_size, "detached");
    if (commit_size > 0) {
      const size_t copy_length = strnlen(head, commit_size - 1);
      memcpy(commit, head, copy_length);
      commit[copy_length] = '\0';
    }
    return;
  }
  const char *reference = head + 5;
  const char *short_branch = strncmp(reference, "refs/heads/", 11) == 0 ? reference + 11 : reference;
  snprintf(branch, branch_size, "%s", short_branch);
  char ref_path[PATH_MAX];
  if (snprintf(ref_path, sizeof(ref_path), "%s/%s", git_directory, reference) < (int)sizeof(ref_path)
      && read_first_line(ref_path, commit, commit_size))
    return;
  read_packed_ref(git_directory, reference, commit, commit_size);
}

static bool is_sensitive_option(const char *argument)
{
  static const char *const sensitive_names[] = {"password", "passwd", "secret", "token", ".key", ".opc", "credential"};
  if (argument == NULL)
    return false;
  for (size_t i = 0; i < sizeof(sensitive_names) / sizeof(sensitive_names[0]); i++) {
    if (strcasestr(argument, sensitive_names[i]) != NULL)
      return true;
  }
  return false;
}

static void write_redacted_cmdline(FILE *file, int argc, char **argv)
{
  bool redact_next = false;
  for (int i = 0; i < argc; i++) {
    fprintf(file, "%s", i == 0 ? "" : " ");
    if (redact_next) {
      fprintf(file, "<redacted>");
      redact_next = false;
      continue;
    }
    if (!is_sensitive_option(argv[i])) {
      fprintf(file, "%s", argv[i]);
      continue;
    }
    const char *equals = strchr(argv[i], '=');
    if (equals != NULL)
      fprintf(file, "%.*s=<redacted>", (int)(equals - argv[i]), argv[i]);
    else {
      fprintf(file, "%s", argv[i]);
      redact_next = true;
    }
  }
}

static void write_metadata(const char *process_name,
                           int argc,
                           char **argv,
                           uint32_t buffer_records,
                           uint32_t flush_us,
                           uint32_t host_metrics_us)
{
  FILE *file = NULL;
  if (open_profile_file(&file, "metadata.txt", "w") != 0)
    return;
  struct timespec rt = {0};
  struct timespec mt = {0};
  clock_gettime(CLOCK_REALTIME, &rt);
  clock_gettime(CLOCK_MONOTONIC_RAW, &mt);
  profile_start_realtime_ns = timespec_to_ns(&rt);
  struct utsname system = {0};
  uname(&system);
  char working_dir[PATH_MAX] = {0};
  if (getcwd(working_dir, sizeof(working_dir)) == NULL)
    snprintf(working_dir, sizeof(working_dir), "unknown");

  char runtime_git_branch[PATH_MAX];
  char runtime_git_head[128];
  read_runtime_git_identity(runtime_git_branch, sizeof(runtime_git_branch), runtime_git_head, sizeof(runtime_git_head));
  fprintf(file, "schema_version=%u\n", OAI_PROFILE_SCHEMA_VERSION);
  fprintf(file, "event_record_size_bytes=%zu\n", sizeof(oai_profile_record_t));
  fprintf(file, "max_nesting_depth=%u\n", OAI_PROFILE_MAX_NESTING_DEPTH);
  fprintf(file, "counter_semantics=elapsed_time_counter\n");
  fprintf(file, "process_name=%s\n", process_name ? process_name : "unknown");
  fprintf(file, "role=%s\n", profile_role);
  fprintf(file, "run_id=%s\n", profile_run_id);
  fprintf(file, "experiment_id=%s\n", profile_experiment_id);
  fprintf(file, "pid=%ld\n", (long)getpid());
  fprintf(file, "hostname=%s\n", profile_hostname);
  fprintf(file, "profile_root=%s\n", profile_root);
  fprintf(file, "output_dir=%s\n", output_dir);
  fprintf(file, "config_archive_dir=%s\n", config_archive_dir);
  fprintf(file, "config_source=%s\n", profile_config_source);
  fprintf(file, "config_archive_policy=path-only-no-secret-copy\n");
  fprintf(file, "working_directory=%s\n", working_dir);
  fprintf(file, "repository_root=%s\n", profile_repository_root);
  fprintf(file, "runtime_git_branch=%s\n", runtime_git_branch);
  fprintf(file, "runtime_git_head=%s\n", runtime_git_head);
  fprintf(file, "runtime_git_dirty=not-checked\n");
  fprintf(file, "build_oai_version=%s\n", OAI_PACKAGE_VERSION);
  fprintf(file, "kernel_sysname=%s\n", system.sysname);
  fprintf(file, "kernel_release=%s\n", system.release);
  fprintf(file, "machine=%s\n", system.machine);
  fprintf(file, "online_cpus=%ld\n", sysconf(_SC_NPROCESSORS_ONLN));
  fprintf(file, "page_size_bytes=%ld\n", sysconf(_SC_PAGESIZE));
  fprintf(file, "counter_hz=%" PRIu64 "\n", counter_hz);
  fprintf(file, "start_realtime_ns=%" PRIu64 "\n", profile_start_realtime_ns);
  fprintf(file, "start_monotonic_raw_ns=%" PRIu64 "\n", timespec_to_ns(&mt));
  fprintf(file, "buffer_records_per_thread=%u\n", buffer_records);
  fprintf(file, "flush_us=%u\n", flush_us);
  fprintf(file, "host_metrics_us=%u\n", host_metrics_us);
  fprintf(file, "max_threads=%u\n", OAI_PROFILE_MAX_THREADS);
  fprintf(file, "cmdline=");
  write_redacted_cmdline(file, argc, argv);
  fprintf(file, "\n");
  fclose(file);
}

static void write_completion_metadata(void)
{
  FILE *file = NULL;
  if (open_profile_file(&file, "metadata.txt", "a") != 0)
    return;
  struct timespec rt = {0};
  struct timespec mt = {0};
  clock_gettime(CLOCK_REALTIME, &rt);
  clock_gettime(CLOCK_MONOTONIC_RAW, &mt);
  const uint64_t end_realtime_ns = timespec_to_ns(&rt);
  fprintf(file, "end_realtime_ns=%" PRIu64 "\n", end_realtime_ns);
  fprintf(file, "end_monotonic_raw_ns=%" PRIu64 "\n", timespec_to_ns(&mt));
  fprintf(file, "duration_realtime_ns=%" PRIu64 "\n", end_realtime_ns - profile_start_realtime_ns);
  fprintf(file, "clean_shutdown=1\n");
  fclose(file);
}

static void write_csv_field(FILE *file, const char *value)
{
  fputc('"', file);
  if (value != NULL) {
    for (const char *p = value; *p != '\0'; p++) {
      if (*p == '"')
        fputc('"', file);
      fputc(*p, file);
    }
  }
  fputc('"', file);
}

static void write_setting(const char *key, const char *value, const char *source)
{
  if (settings_file == NULL || key == NULL)
    return;
  struct timespec rt = {0};
  clock_gettime(CLOCK_REALTIME, &rt);
  fprintf(settings_file, "%" PRIu64 ",", timespec_to_ns(&rt));
  write_csv_field(settings_file, key);
  fputc(',', settings_file);
  write_csv_field(settings_file, value ? value : "");
  fputc(',', settings_file);
  write_csv_field(settings_file, source ? source : "");
  fputc('\n', settings_file);
  fflush(settings_file);
}

void oai_profiler_record_setting(const char *key, const char *value, const char *source)
{
  if (!oai_profiler_is_enabled())
    return;
  pthread_mutex_lock(&settings_mutex);
  write_setting(key, value, source);
  pthread_mutex_unlock(&settings_mutex);
}

void oai_profiler_record_setting_int(const char *key, int64_t value, const char *source)
{
  char text[32];
  snprintf(text, sizeof(text), "%" PRId64, value);
  oai_profiler_record_setting(key, text, source);
}

static bool read_i64_file(const char *path, int64_t *value)
{
  FILE *file = fopen(path, "r");
  if (file == NULL)
    return false;
  int64_t parsed = 0;
  const bool valid = fscanf(file, "%" SCNd64, &parsed) == 1;
  fclose(file);
  if (valid)
    *value = parsed;
  return valid;
}

static void read_thermal_metrics(int64_t *zone0_millicelsius, int64_t *max_millicelsius, int *sample_count)
{
  *zone0_millicelsius = -1;
  *max_millicelsius = -1;
  *sample_count = 0;
  for (int zone = 0; zone < 64; zone++) {
    char path[PATH_MAX];
    int ret = snprintf(path, sizeof(path), "/sys/class/thermal/thermal_zone%d/temp", zone);
    int64_t temperature = -1;
    if (ret < 0 || (size_t)ret >= sizeof(path) || !read_i64_file(path, &temperature))
      continue;
    if (zone == 0)
      *zone0_millicelsius = temperature;
    if (*max_millicelsius < temperature)
      *max_millicelsius = temperature;
    (*sample_count)++;
  }
}

static void read_cpu_frequency_metrics(int64_t *min_khz, int64_t *avg_khz, int64_t *max_khz, int *sample_count)
{
  *min_khz = -1;
  *avg_khz = -1;
  *max_khz = -1;
  *sample_count = 0;
  uint64_t total_khz = 0;
  const long cpu_count = sysconf(_SC_NPROCESSORS_CONF);
  for (long cpu = 0; cpu < cpu_count; cpu++) {
    char path[PATH_MAX];
    int ret = snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%ld/cpufreq/scaling_cur_freq", cpu);
    int64_t frequency = -1;
    if (ret < 0 || (size_t)ret >= sizeof(path) || !read_i64_file(path, &frequency))
      continue;
    if (*min_khz < 0 || frequency < *min_khz)
      *min_khz = frequency;
    if (frequency > *max_khz)
      *max_khz = frequency;
    total_khz += (uint64_t)frequency;
    (*sample_count)++;
  }
  if (*sample_count > 0)
    *avg_khz = (int64_t)(total_khz / (uint64_t)*sample_count);
}

static void read_memory_metrics(int64_t *mem_available_kb, int64_t *swap_free_kb)
{
  *mem_available_kb = -1;
  *swap_free_kb = -1;
  FILE *file = fopen("/proc/meminfo", "r");
  if (file == NULL)
    return;
  char key[64];
  uint64_t value = 0;
  char unit[16];
  while (fscanf(file, "%63s %" SCNu64 " %15s", key, &value, unit) == 3) {
    if (strcmp(key, "MemAvailable:") == 0)
      *mem_available_kb = (int64_t)value;
    else if (strcmp(key, "SwapFree:") == 0)
      *swap_free_kb = (int64_t)value;
  }
  fclose(file);
}

static int64_t read_process_rss_kb(void)
{
  FILE *file = fopen("/proc/self/status", "r");
  if (file == NULL)
    return -1;
  char line[256];
  int64_t rss_kb = -1;
  while (fgets(line, sizeof(line), file) != NULL) {
    if (sscanf(line, "VmRSS: %" SCNd64 " kB", &rss_kb) == 1)
      break;
  }
  fclose(file);
  return rss_kb;
}

static double read_cpu_busy_percent(void)
{
  FILE *file = fopen("/proc/stat", "r");
  if (file == NULL)
    return -1.0;
  uint64_t user = 0;
  uint64_t nice = 0;
  uint64_t system = 0;
  uint64_t idle = 0;
  uint64_t iowait = 0;
  uint64_t irq = 0;
  uint64_t softirq = 0;
  uint64_t steal = 0;
  int fields = fscanf(file,
                      "cpu  %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
                      &user,
                      &nice,
                      &system,
                      &idle,
                      &iowait,
                      &irq,
                      &softirq,
                      &steal);
  fclose(file);
  if (fields < 4)
    return -1.0;

  const uint64_t current_idle = idle + iowait;
  const uint64_t current_total = user + nice + system + idle + iowait + irq + softirq + steal;
  double busy_percent = -1.0;
  if (previous_cpu_times_valid && current_total > previous_cpu_total) {
    const uint64_t total_delta = current_total - previous_cpu_total;
    const uint64_t idle_delta = current_idle - previous_cpu_idle;
    busy_percent = 100.0 * (double)(total_delta - idle_delta) / (double)total_delta;
  }
  previous_cpu_total = current_total;
  previous_cpu_idle = current_idle;
  previous_cpu_times_valid = true;
  return busy_percent;
}

static bool read_rpi_throttled(uint32_t *throttled)
{
  if (rpi_mailbox_fd < 0)
    return false;
  uint32_t request[7] = {
      sizeof(request),
      0,
      OAI_PROFILE_RPI_GET_THROTTLED,
      sizeof(uint32_t),
      0,
      0,
      0,
  };
  if (ioctl(rpi_mailbox_fd, OAI_PROFILE_RPI_MAILBOX_PROPERTY, request) < 0 || request[1] != 0x80000000U)
    return false;
  *throttled = request[5];
  return true;
}

static uint64_t timeval_to_us(const struct timeval *value)
{
  return (uint64_t)value->tv_sec * 1000000ULL + (uint64_t)value->tv_usec;
}

static void write_host_metrics_header(void)
{
  fprintf(host_metrics_file,
          "realtime_ns,monotonic_raw_ns,tick,writer_cpu,"
          "thermal_zone0_millicelsius,thermal_max_millicelsius,thermal_samples,"
          "rpi_throttled_valid,rpi_throttled_raw,"
          "cpu_frequency_samples,cpu_frequency_min_khz,cpu_frequency_avg_khz,cpu_frequency_max_khz,"
          "cpu_busy_percent,load1,load5,load15,mem_available_kb,swap_free_kb,"
          "process_rss_kb,process_maxrss_kb,process_user_us,process_system_us,"
          "voluntary_context_switches,involuntary_context_switches,minor_faults,major_faults,"
          "block_input_ops,block_output_ops\n");
}

static void write_host_metrics_sample(void)
{
  if (host_metrics_file == NULL)
    return;

  struct timespec realtime = {0};
  struct timespec monotonic = {0};
  clock_gettime(CLOCK_REALTIME, &realtime);
  clock_gettime(CLOCK_MONOTONIC_RAW, &monotonic);
  const uint64_t tick = (uint64_t)rdtsc_oai();

  int64_t zone0_millicelsius = -1;
  int64_t max_millicelsius = -1;
  int thermal_samples = 0;
  read_thermal_metrics(&zone0_millicelsius, &max_millicelsius, &thermal_samples);

  int64_t min_frequency_khz = -1;
  int64_t avg_frequency_khz = -1;
  int64_t max_frequency_khz = -1;
  int frequency_samples = 0;
  read_cpu_frequency_metrics(&min_frequency_khz, &avg_frequency_khz, &max_frequency_khz, &frequency_samples);

  int64_t mem_available_kb = -1;
  int64_t swap_free_kb = -1;
  read_memory_metrics(&mem_available_kb, &swap_free_kb);

  double load[3] = {-1.0, -1.0, -1.0};
  getloadavg(load, 3);
  const double cpu_busy_percent = read_cpu_busy_percent();

  uint32_t throttled = 0;
  const bool throttled_valid = read_rpi_throttled(&throttled);

  struct rusage usage = {0};
  getrusage(RUSAGE_SELF, &usage);
  const int64_t process_rss_kb = read_process_rss_kb();

  fprintf(host_metrics_file,
          "%" PRIu64 ",%" PRIu64 ",%" PRIu64
          ",%d,"
          "%" PRId64 ",%" PRId64 ",%d,%d,%" PRIu32
          ","
          "%d,%" PRId64 ",%" PRId64 ",%" PRId64
          ","
          "%.3f,%.3f,%.3f,%.3f,%" PRId64 ",%" PRId64
          ","
          "%" PRId64 ",%ld,%" PRIu64 ",%" PRIu64
          ","
          "%ld,%ld,%ld,%ld,%ld,%ld\n",
          timespec_to_ns(&realtime),
          timespec_to_ns(&monotonic),
          tick,
          sched_getcpu(),
          zone0_millicelsius,
          max_millicelsius,
          thermal_samples,
          throttled_valid,
          throttled,
          frequency_samples,
          min_frequency_khz,
          avg_frequency_khz,
          max_frequency_khz,
          cpu_busy_percent,
          load[0],
          load[1],
          load[2],
          mem_available_kb,
          swap_free_kb,
          process_rss_kb,
          usage.ru_maxrss,
          timeval_to_us(&usage.ru_utime),
          timeval_to_us(&usage.ru_stime),
          usage.ru_nvcsw,
          usage.ru_nivcsw,
          usage.ru_minflt,
          usage.ru_majflt,
          usage.ru_inblock,
          usage.ru_oublock);
  fflush(host_metrics_file);
}

static void write_sync_sample(void)
{
  if (sync_file == NULL)
    return;
  struct timespec rt = {0};
  struct timespec mt = {0};
  clock_gettime(CLOCK_REALTIME, &rt);
  clock_gettime(CLOCK_MONOTONIC_RAW, &mt);
  uint64_t tick = rdtsc_oai();
  fprintf(sync_file, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", timespec_to_ns(&rt), timespec_to_ns(&mt), tick);
}

static void drain_thread_buffer(oai_profile_thread_buffer_t *tb)
{
  uint64_t read_count = __atomic_load_n(&tb->read_count, __ATOMIC_RELAXED);
  const uint64_t write_count = __atomic_load_n(&tb->write_count, __ATOMIC_ACQUIRE);
  while (read_count < write_count) {
    const oai_profile_record_t *r = &tb->records[read_count % tb->capacity];
    const double duration_us = counter_hz == 0 ? 0.0 : ((double)r->duration_tick * 1000000.0) / (double)counter_hz;
    const bool cpu_migrated = r->cpu_start >= 0 && r->cpu_end >= 0 && r->cpu_start != r->cpu_end;
    fprintf(events_file,
            "%u,%" PRIu64 ",%ld,%s,%u,%s,%s,%u,%d,%d,%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%d,%d,%d,%u,%" PRId64
            ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%.3f\n",
            OAI_PROFILE_SCHEMA_VERSION,
            r->seq,
            (long)tb->tid,
            tb->name,
            r->event_id,
            oai_profiler_event_name((oai_profile_event_id_t)r->event_id),
            oai_profiler_event_kind_name((oai_profile_event_kind_t)r->event_kind),
            r->nesting_depth,
            r->frame,
            r->slot,
            r->absolute_slot,
            r->correlation_id,
            r->span_id,
            r->parent_id,
            r->cpu_start,
            r->cpu_end,
            cpu_migrated,
            r->flags,
            r->aux0,
            r->aux1,
            r->aux2,
            r->aux3,
            r->start_tick,
            r->duration_tick,
            duration_us);
    read_count++;
  }
  __atomic_store_n(&tb->read_count, read_count, __ATOMIC_RELEASE);
}

static void drain_all_buffers(void)
{
  pthread_mutex_lock(&registry_mutex);
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    if (thread_buffers[i].active)
      drain_thread_buffer(&thread_buffers[i]);
  }
  pthread_mutex_unlock(&registry_mutex);
  if (events_file != NULL)
    fflush(events_file);
  if (sync_file != NULL)
    fflush(sync_file);
}

static void write_drop_summary(void)
{
  if (drops_file == NULL)
    return;
  fprintf(drops_file, "thread_index,tid,thread_name,dropped_records,span_stack_overflows,span_stack_mismatches\n");
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    if (thread_buffers[i].active)
      fprintf(drops_file,
              "%d,%ld,%s,%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n",
              i,
              (long)thread_buffers[i].tid,
              thread_buffers[i].name,
              __atomic_load_n(&thread_buffers[i].dropped_records, __ATOMIC_RELAXED),
              __atomic_load_n(&thread_buffers[i].span_stack_overflows, __ATOMIC_RELAXED),
              __atomic_load_n(&thread_buffers[i].span_stack_mismatches, __ATOMIC_RELAXED));
  }
  fflush(drops_file);
}

static void *profiler_writer_thread(void *arg)
{
  (void)arg;
  pthread_setname_np(pthread_self(), "oai_profile");
  uint64_t next_host_metrics_ns = 0;
  while (!__atomic_load_n(&profiler_shutdown_requested, __ATOMIC_ACQUIRE)) {
    drain_all_buffers();
    write_sync_sample();
    struct timespec now = {0};
    clock_gettime(CLOCK_MONOTONIC_RAW, &now);
    const uint64_t now_ns = timespec_to_ns(&now);
    if (now_ns >= next_host_metrics_ns) {
      write_host_metrics_sample();
      next_host_metrics_ns = now_ns + (uint64_t)global_host_metrics_us * 1000ULL;
    }
    usleep(global_flush_us);
  }
  drain_all_buffers();
  write_sync_sample();
  write_host_metrics_sample();
  write_drop_summary();
  return NULL;
}

static int register_thread_buffer(void)
{
  if (thread_buffer_index >= 0)
    return thread_buffer_index;

  pthread_mutex_lock(&registry_mutex);
  int idx = -1;
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    if (!thread_buffers[i].active) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    pthread_mutex_unlock(&registry_mutex);
    return -1;
  }

  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  memset(tb, 0, sizeof(*tb));
  tb->records = calloc(global_buffer_records, sizeof(*tb->records));
  if (tb->records == NULL) {
    pthread_mutex_unlock(&registry_mutex);
    return -1;
  }
  tb->capacity = global_buffer_records;
  tb->tid = (pid_t)syscall(SYS_gettid);
  if (pthread_getname_np(pthread_self(), tb->name, sizeof(tb->name)) != 0 || tb->name[0] == '\0')
    snprintf(tb->name, sizeof(tb->name), "tid-%ld", (long)tb->tid);
  tb->active = true;
  thread_buffer_index = idx;
  pthread_mutex_unlock(&registry_mutex);
  return idx;
}

void oai_profiler_register_thread(void)
{
  if (!oai_profiler_is_enabled())
    return;
  (void)register_thread_buffer();
}

void oai_profiler_set_context(oai_profile_context_t context)
{
  thread_context = context;
}

oai_profile_context_t oai_profiler_get_context(void)
{
  oai_profile_context_t context = thread_context;
  if (thread_span_depth > 0)
    context.parent_id = thread_span_stack[thread_span_depth - 1];
  return context;
}

void oai_profiler_clear_context(void)
{
  thread_context = (oai_profile_context_t){
      .absolute_slot = OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN,
      .correlation_id = 0,
      .parent_id = 0,
  };
}

uint64_t oai_profiler_next_correlation_id(void)
{
  if (!oai_profiler_is_enabled())
    return 0;
  uint64_t correlation_id = __atomic_add_fetch(&global_correlation_seq, 1, __ATOMIC_RELAXED);
  if (correlation_id == 0)
    correlation_id = __atomic_add_fetch(&global_correlation_seq, 1, __ATOMIC_RELAXED);
  return correlation_id;
}

oai_profile_work_t oai_profiler_capture_work(int64_t absolute_slot)
{
  oai_profile_work_t work = {
      .context.absolute_slot = OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN,
  };
  if (!oai_profiler_is_enabled())
    return work;

  work.context = oai_profiler_get_context();
  if (absolute_slot != OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN)
    work.context.absolute_slot = absolute_slot;
  work.dispatch_tick = rdtsc_oai();
  return work;
}

oai_profile_context_t oai_profiler_enter_work(oai_profile_work_t work)
{
  const oai_profile_context_t previous_context = oai_profiler_get_context();
  oai_profiler_set_context(work.context);
  return previous_context;
}

void oai_profiler_leave_work(oai_profile_context_t previous_context)
{
  oai_profiler_set_context(previous_context);
}

static uint64_t next_span_id(oai_profile_thread_buffer_t *tb, int thread_index)
{
  const uint64_t sequence_mask = UINT64_C(0x0000ffffffffffff);
  uint64_t sequence = (++tb->next_span_sequence) & sequence_mask;
  if (sequence == 0)
    sequence = (++tb->next_span_sequence) & sequence_mask;
  return ((uint64_t)(thread_index + 1) << 48) | sequence;
}

oai_profile_span_t oai_profiler_span_start_enabled(void)
{
  oai_profile_span_t span = {
      .absolute_slot = OAI_PROFILE_ABSOLUTE_SLOT_UNKNOWN,
      .cpu_start = -1,
  };
  if (!oai_profiler_is_enabled())
    return span;

  const int idx = register_thread_buffer();
  if (idx < 0)
    return span;

  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  span.span_id = next_span_id(tb, idx);
  span.parent_id = thread_span_depth > 0 ? thread_span_stack[thread_span_depth - 1] : thread_context.parent_id;
  span.correlation_id = thread_context.correlation_id;
  span.absolute_slot = thread_context.absolute_slot;
  span.depth = thread_span_depth;
  span.thread_index = (uint16_t)idx;
  if (thread_span_depth < OAI_PROFILE_MAX_NESTING_DEPTH) {
    thread_span_stack[thread_span_depth++] = span.span_id;
    span.stack_registered = 1;
  } else {
    __atomic_fetch_add(&tb->span_stack_overflows, 1, __ATOMIC_RELAXED);
  }
  span.cpu_start = sched_getcpu();
  span.start_tick = rdtsc_oai();
  return span;
}

static void retire_span(oai_profile_thread_buffer_t *tb, int thread_index, oai_profile_span_t span)
{
  if (!span.stack_registered)
    return;
  if (span.thread_index != thread_index) {
    __atomic_fetch_add(&tb->span_stack_mismatches, 1, __ATOMIC_RELAXED);
    return;
  }
  if (thread_span_depth > 0 && thread_span_stack[thread_span_depth - 1] == span.span_id) {
    thread_span_depth--;
    return;
  }

  __atomic_fetch_add(&tb->span_stack_mismatches, 1, __ATOMIC_RELAXED);
  int found = -1;
  for (int i = (int)thread_span_depth - 1; i >= 0; i--) {
    if (thread_span_stack[i] == span.span_id) {
      found = i;
      break;
    }
  }
  if (found < 0)
    return;
  for (uint16_t i = (uint16_t)found; i + 1 < thread_span_depth; i++)
    thread_span_stack[i] = thread_span_stack[i + 1];
  thread_span_depth--;
}

static void publish_record(oai_profile_thread_buffer_t *tb,
                           oai_profile_event_id_t event_id,
                           oai_profile_event_kind_t event_kind,
                           uint64_t start_tick,
                           uint64_t duration_tick,
                           uint64_t span_id,
                           uint64_t parent_id,
                           uint64_t correlation_id,
                           int64_t absolute_slot,
                           uint16_t nesting_depth,
                           int cpu_start,
                           int cpu_end,
                           int frame,
                           int slot,
                           int64_t aux0,
                           int64_t aux1,
                           int64_t aux2,
                           int64_t aux3,
                           uint32_t flags)
{
  const uint64_t write_count = __atomic_load_n(&tb->write_count, __ATOMIC_RELAXED);
  const uint64_t read_count = __atomic_load_n(&tb->read_count, __ATOMIC_ACQUIRE);
  if (write_count - read_count >= tb->capacity) {
    __atomic_fetch_add(&tb->dropped_records, 1, __ATOMIC_RELAXED);
    return;
  }

  oai_profile_record_t *r = &tb->records[write_count % tb->capacity];
  r->seq = __atomic_fetch_add(&global_seq, 1, __ATOMIC_RELAXED);
  r->start_tick = start_tick;
  r->duration_tick = duration_tick;
  r->span_id = span_id;
  r->parent_id = parent_id;
  r->correlation_id = correlation_id;
  r->absolute_slot = absolute_slot;
  r->event_id = event_id;
  r->frame = frame;
  r->slot = slot;
  r->cpu_start = cpu_start;
  r->cpu_end = cpu_end;
  r->aux0 = aux0;
  r->aux1 = aux1;
  r->aux2 = aux2;
  r->aux3 = aux3;
  r->flags = flags;
  r->nesting_depth = nesting_depth;
  r->event_kind = event_kind;
  r->reserved = 0;
  __atomic_store_n(&tb->write_count, write_count + 1, __ATOMIC_RELEASE);
}

void oai_profiler_record_span(oai_profile_event_id_t event_id,
                              oai_profile_span_t span,
                              int frame,
                              int slot,
                              int64_t aux0,
                              int64_t aux1,
                              int64_t aux2,
                              int64_t aux3,
                              uint32_t flags)
{
  if (!oai_profiler_is_enabled() || span.start_tick == 0)
    return;

  const uint64_t end_tick = rdtsc_oai();
  const int cpu_end = sched_getcpu();
  int idx = thread_buffer_index;
  if (idx < 0)
    idx = register_thread_buffer();
  if (idx < 0)
    return;
  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  retire_span(tb, idx, span);
  if (event_id <= OAI_PROFILE_EVENT_UNSPEC || event_id >= OAI_PROFILE_EVENT_MAX)
    return;

  publish_record(tb,
                 event_id,
                 OAI_PROFILE_EVENT_KIND_DURATION,
                 span.start_tick,
                 end_tick - span.start_tick,
                 span.span_id,
                 span.parent_id,
                 span.correlation_id,
                 span.absolute_slot,
                 span.depth,
                 span.cpu_start,
                 cpu_end,
                 frame,
                 slot,
                 aux0,
                 aux1,
                 aux2,
                 aux3,
                 flags);
}

void oai_profiler_record_duration(oai_profile_event_id_t event_id,
                                  uint64_t start_tick,
                                  int frame,
                                  int slot,
                                  int64_t aux0,
                                  int64_t aux1,
                                  int64_t aux2,
                                  int64_t aux3,
                                  uint32_t flags)
{
  if (!oai_profiler_is_enabled() || start_tick == 0 || event_id <= OAI_PROFILE_EVENT_UNSPEC || event_id >= OAI_PROFILE_EVENT_MAX)
    return;
  const uint64_t end_tick = rdtsc_oai();
  const int cpu_end = sched_getcpu();
  const int idx = register_thread_buffer();
  if (idx < 0)
    return;
  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  const uint64_t parent_id = thread_span_depth > 0 ? thread_span_stack[thread_span_depth - 1] : thread_context.parent_id;
  publish_record(tb,
                 event_id,
                 OAI_PROFILE_EVENT_KIND_DURATION,
                 start_tick,
                 end_tick - start_tick,
                 next_span_id(tb, idx),
                 parent_id,
                 thread_context.correlation_id,
                 thread_context.absolute_slot,
                 thread_span_depth,
                 -1,
                 cpu_end,
                 frame,
                 slot,
                 aux0,
                 aux1,
                 aux2,
                 aux3,
                 flags);
}

void oai_profiler_record_instant(oai_profile_event_id_t event_id,
                                 int frame,
                                 int slot,
                                 int64_t aux0,
                                 int64_t aux1,
                                 int64_t aux2,
                                 int64_t aux3,
                                 uint32_t flags)
{
  if (!oai_profiler_is_enabled() || event_id <= OAI_PROFILE_EVENT_UNSPEC || event_id >= OAI_PROFILE_EVENT_MAX)
    return;
  const int idx = register_thread_buffer();
  if (idx < 0)
    return;
  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  const uint64_t parent_id = thread_span_depth > 0 ? thread_span_stack[thread_span_depth - 1] : thread_context.parent_id;
  const int cpu = sched_getcpu();
  const uint64_t tick = rdtsc_oai();
  publish_record(tb,
                 event_id,
                 OAI_PROFILE_EVENT_KIND_INSTANT,
                 tick,
                 0,
                 next_span_id(tb, idx),
                 parent_id,
                 thread_context.correlation_id,
                 thread_context.absolute_slot,
                 thread_span_depth,
                 cpu,
                 cpu,
                 frame,
                 slot,
                 aux0,
                 aux1,
                 aux2,
                 aux3,
                 flags);
}

void oai_profiler_init(const char *process_name,
                       int argc,
                       char **argv,
                       bool enable_from_cli,
                       const char *profile_dir,
                       uint32_t buffer_records,
                       uint32_t flush_us)
{
  pthread_mutex_lock(&lifecycle_mutex);
  if (profiler_initialized) {
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }

  const bool enabled = parse_env_enable(enable_from_cli);
  if (!enabled) {
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }

  if (prepare_profile_paths(process_name, profile_dir) != 0) {
    LOG_W(UTIL, "OAI profiler disabled: cannot prepare archive path: %s\n", strerror(errno));
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }
  find_config_source(argc, argv);

  global_buffer_records = buffer_records ? buffer_records : OAI_PROFILE_DEFAULT_BUFFER_RECORDS;
  global_buffer_records = parse_u32_or_default(getenv("OAI_PROFILE_BUFFER_RECORDS"), global_buffer_records);
  if (global_buffer_records < OAI_PROFILE_MIN_BUFFER_RECORDS)
    global_buffer_records = OAI_PROFILE_MIN_BUFFER_RECORDS;
  global_flush_us = flush_us ? flush_us : OAI_PROFILE_DEFAULT_FLUSH_US;
  global_flush_us = parse_u32_or_default(getenv("OAI_PROFILE_FLUSH_US"), global_flush_us);
  if (global_flush_us == 0)
    global_flush_us = OAI_PROFILE_DEFAULT_FLUSH_US;
  global_host_metrics_us = parse_u32_or_default(getenv("OAI_PROFILE_HOST_METRICS_US"), OAI_PROFILE_DEFAULT_HOST_METRICS_US);
  if (global_host_metrics_us < 100000U)
    global_host_metrics_us = 100000U;

  counter_hz = read_counter_hz();
  if (open_profile_file(&events_file, "events.csv", "w") != 0 || open_profile_file(&sync_file, "sync.csv", "w") != 0
      || open_profile_file(&drops_file, "drops.csv", "w") != 0 || open_profile_file(&settings_file, "settings.csv", "w") != 0
      || open_profile_file(&host_metrics_file, "host_metrics.csv", "w") != 0) {
    LOG_W(UTIL, "OAI profiler disabled: cannot open output files under %s\n", output_dir);
    FILE **files[] = {&events_file, &sync_file, &drops_file, &settings_file, &host_metrics_file};
    for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); i++) {
      if (*files[i] != NULL)
        fclose(*files[i]);
      *files[i] = NULL;
    }
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }

  fprintf(events_file,
          "schema_version,seq,tid,thread_name,event_id,event_name,event_kind,nesting_depth,frame,slot,"
          "absolute_slot,correlation_id,span_id,parent_id,cpu_start,cpu_end,cpu_migrated,flags,"
          "aux0,aux1,aux2,aux3,start_tick,duration_tick,duration_us\n");
  fprintf(sync_file, "realtime_ns,monotonic_raw_ns,tick\n");
  fprintf(settings_file, "realtime_ns,key,value,source\n");
  write_host_metrics_header();
  rpi_mailbox_fd = open("/dev/vcio", O_RDWR | O_CLOEXEC);

  write_event_catalog();
  write_metadata(process_name, argc, argv, global_buffer_records, global_flush_us, global_host_metrics_us);
  char setting_value[32];
  snprintf(setting_value, sizeof(setting_value), "%u", global_buffer_records);
  write_setting("profile.buffer_records_per_thread", setting_value, "resolved");
  snprintf(setting_value, sizeof(setting_value), "%u", global_flush_us);
  write_setting("profile.flush_us", setting_value, "resolved");
  snprintf(setting_value, sizeof(setting_value), "%u", global_host_metrics_us);
  write_setting("profile.host_metrics_us", setting_value, "resolved");
  write_sync_sample();
  write_host_metrics_sample();

  __atomic_store_n(&profiler_shutdown_requested, false, __ATOMIC_RELEASE);
  profiler_initialized = true;
  __atomic_store_n(&oai_profiler_enabled, 1, __ATOMIC_RELEASE);
  if (pthread_create(&writer_thread, NULL, profiler_writer_thread, NULL) == 0) {
    writer_started = true;
    LOG_I(UTIL, "OAI profiler enabled, writing to %s\n", output_dir);
  } else {
    __atomic_store_n(&oai_profiler_enabled, 0, __ATOMIC_RELEASE);
    profiler_initialized = false;
    LOG_W(UTIL, "OAI profiler disabled: cannot create writer thread\n");
    fclose(events_file);
    fclose(sync_file);
    fclose(drops_file);
    fclose(settings_file);
    fclose(host_metrics_file);
    events_file = NULL;
    sync_file = NULL;
    drops_file = NULL;
    settings_file = NULL;
    host_metrics_file = NULL;
    if (rpi_mailbox_fd >= 0)
      close(rpi_mailbox_fd);
    rpi_mailbox_fd = -1;
  }
  pthread_mutex_unlock(&lifecycle_mutex);
}

void oai_profiler_shutdown(void)
{
  pthread_mutex_lock(&lifecycle_mutex);
  if (!profiler_initialized) {
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }
  __atomic_store_n(&oai_profiler_enabled, 0, __ATOMIC_RELEASE);
  __atomic_store_n(&profiler_shutdown_requested, true, __ATOMIC_RELEASE);
  const bool join_writer = writer_started;
  pthread_t writer = writer_thread;
  pthread_mutex_unlock(&lifecycle_mutex);

  if (join_writer)
    pthread_join(writer, NULL);

  write_completion_metadata();
  pthread_mutex_lock(&lifecycle_mutex);
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    free(thread_buffers[i].records);
    memset(&thread_buffers[i], 0, sizeof(thread_buffers[i]));
  }
  if (events_file != NULL)
    fclose(events_file);
  if (sync_file != NULL)
    fclose(sync_file);
  if (drops_file != NULL)
    fclose(drops_file);
  if (settings_file != NULL)
    fclose(settings_file);
  if (host_metrics_file != NULL)
    fclose(host_metrics_file);
  if (rpi_mailbox_fd >= 0)
    close(rpi_mailbox_fd);
  events_file = NULL;
  sync_file = NULL;
  drops_file = NULL;
  settings_file = NULL;
  host_metrics_file = NULL;
  rpi_mailbox_fd = -1;
  writer_started = false;
  profiler_initialized = false;
  __atomic_store_n(&profiler_shutdown_requested, false, __ATOMIC_RELEASE);
  previous_cpu_times_valid = false;
  __atomic_store_n(&global_seq, 0, __ATOMIC_RELAXED);
  __atomic_store_n(&global_correlation_seq, 0, __ATOMIC_RELAXED);
  thread_buffer_index = -1;
  thread_span_depth = 0;
  oai_profiler_clear_context();
  pthread_mutex_unlock(&lifecycle_mutex);
}
