/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/**
 * @file flight_monitor.h
 * @brief Bounded native progress telemetry for flight recovery supervision.
 *
 * The monitor remains disabled unless the supervisor supplies a connected
 * AF_UNIX SOCK_DGRAM descriptor through _OAI_FLIGHT_MONITOR_FD and
 * _OAI_FLIGHT_CAPTURE_PARENT numerically equals getppid() at initialization.
 * Producers only perform relaxed lock-free atomic updates. They never allocate,
 * lock, perform I/O, signal the monitor, or retry work.
 *
 * Datagrams are best effort. A missing, delayed, malformed, or dropped snapshot
 * cannot establish the absence of a protocol restriction or other state.
 */

#ifndef FLIGHT_MONITOR_H_
#define FLIGHT_MONITOR_H_

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FLIGHT_MONITOR_SCHEMA_VERSION 1U

/** Keep this catalog within the fixed 32-field monitor capacity. */
typedef enum {
  FLIGHT_MONITOR_RX_SAMPLES = 0,
  FLIGHT_MONITOR_TX_SAMPLES,
  FLIGHT_MONITOR_SEARCH_ATTEMPTS,
  FLIGHT_MONITOR_SYNC_SUCCESSES,
  FLIGHT_MONITOR_RRC_MESSAGES,
  FLIGHT_MONITOR_NAS_MESSAGES,
  FLIGHT_MONITOR_RRC_STATE,
  FLIGHT_MONITOR_PDU_ACCEPTS,
  FLIGHT_MONITOR_PDU_ACTIVE,
  FLIGHT_MONITOR_NAS_REJECT,
  FLIGHT_MONITOR_RRC_HOLD_UNTIL_NS,
  FLIGHT_MONITOR_UE_SLOT_INPUTS,
  FLIGHT_MONITOR_UE_DL_COMPLETED,
  FLIGHT_MONITOR_UE_TX_COMPLETED,
  /* Binary RRC configured/resumed-DRB control-plane observation. This does not verify SDAP mapping, retained NAS session, PDU
     acceptance, payload delivery, or user-plane health. */
  FLIGHT_MONITOR_DRB_CONTEXT_ACTIVE,
  FLIGHT_MONITOR_FIELD_COUNT,
} flight_monitor_field_t;

/**
 * Pack one coherent NAS rejection decision for FLIGHT_MONITOR_NAS_REJECT.
 *
 * Bits 63..56 are generation, 55..48 cause, 47 is explicit T3502 presence, 46..40 policy,
 * 39..32 are the raw T3502 GPRS timer-2 value,
 * and 31..0 are wait_seconds. A publisher must issue one set() with this
 * packed value; consumers must not combine independent decision fields.
 */
static inline uint64_t flight_monitor_pack_nas_reject(uint8_t generation,
                                                      uint8_t cause,
                                                      uint8_t policy,
                                                      uint8_t t3502_raw,
                                                      uint32_t wait_seconds)
{
  return ((uint64_t)generation << 56) | ((uint64_t)cause << 48) | ((uint64_t)policy << 40) | ((uint64_t)t3502_raw << 32)
         | (uint64_t)wait_seconds;
}

/** Initialize before real-time producers. Disabled startup creates no thread. */
void flight_monitor_init(void);

/** Disable producers, send a final best-effort snapshot, join, and close the monitor descriptor. */
void flight_monitor_shutdown(void);

/** True only while producer updates can be sampled by the monitor process. */
bool flight_monitor_enabled(void);

/** Add amount to a field with bounded relaxed lock-free atomics only. */
void flight_monitor_add(flight_monitor_field_t field, uint64_t amount);

/** Set a field with bounded relaxed lock-free atomics only. */
void flight_monitor_set(flight_monitor_field_t field, uint64_t value);

#ifdef FLIGHT_MONITOR_TESTING
uint64_t flight_monitor_test_send_drops(void);
int flight_monitor_test_worker_policy(void);
int flight_monitor_test_wakeup_clock(void);
#endif

#ifdef __cplusplus
}
#endif

#endif /* FLIGHT_MONITOR_H_ */
