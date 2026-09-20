/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

/** \file radio_health.h
 * \brief Bounded, backend-independent radio health observations.
 *
 * Registration is enabled by the process before radio devices are created. A
 * successful registration reserves one process-lifetime slot; slots are never
 * reused. Producers hold only an opaque handle and make no allocations, I/O,
 * or locking calls. Snapshot consumers run outside real-time producer paths.
 */

#ifndef RADIO_HEALTH_H
#define RADIO_HEALTH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RADIO_HEALTH_MAX_DEVICES 4U

typedef struct radio_health_device_s radio_health_device_t;

typedef enum {
  RADIO_HEALTH_BACKEND_UNKNOWN = 0,
  RADIO_HEALTH_BACKEND_UHD,
} radio_health_backend_t;

typedef enum {
  RADIO_HEALTH_LIFECYCLE_UNUSED = 0,
  RADIO_HEALTH_LIFECYCLE_ACTIVE,
  RADIO_HEALTH_LIFECYCLE_CLOSED,
} radio_health_lifecycle_t;

typedef enum {
  RADIO_HEALTH_CAP_TX_SEND = UINT64_C(1) << 0,
  RADIO_HEALTH_CAP_TX_ASYNC = UINT64_C(1) << 1,
  RADIO_HEALTH_CAP_RX_STREAM = UINT64_C(1) << 2,
  RADIO_HEALTH_CAP_TX_QUEUE = UINT64_C(1) << 3,
} radio_health_capability_t;

typedef enum {
  RADIO_HEALTH_METRIC_TX_SEND_CALLS = 0,
  RADIO_HEALTH_METRIC_TX_SEND_REQUESTED_SAMPLES,
  RADIO_HEALTH_METRIC_TX_SEND_ACCEPTED_SAMPLES,
  RADIO_HEALTH_METRIC_TX_SEND_SHORT_CALLS,
  RADIO_HEALTH_METRIC_TX_SEND_EXCEPTIONS,
  RADIO_HEALTH_METRIC_TX_SEND_INFLIGHT,
  RADIO_HEALTH_METRIC_TX_SAMPLE_RATE_HZ,

  RADIO_HEALTH_METRIC_TX_ASYNC_POLLS,
  RADIO_HEALTH_METRIC_TX_ASYNC_MESSAGES,
  RADIO_HEALTH_METRIC_TX_ASYNC_EXCEPTIONS,
  RADIO_HEALTH_METRIC_TX_ASYNC_TIME_ERROR,
  RADIO_HEALTH_METRIC_TX_ASYNC_UNDERFLOW,
  RADIO_HEALTH_METRIC_TX_ASYNC_UNDERFLOW_IN_PACKET,
  RADIO_HEALTH_METRIC_TX_ASYNC_SEQ_ERROR,
  RADIO_HEALTH_METRIC_TX_ASYNC_SEQ_ERROR_IN_BURST,
  RADIO_HEALTH_METRIC_TX_ASYNC_BURST_ACK,
  RADIO_HEALTH_METRIC_TX_ASYNC_UNKNOWN,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_POLL_MONO_NS,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS,
  RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID,

  RADIO_HEALTH_METRIC_RX_RECV_CALLS,
  RADIO_HEALTH_METRIC_RX_REQUESTED_SAMPLES,
  RADIO_HEALTH_METRIC_RX_RETURNED_SAMPLES,
  RADIO_HEALTH_METRIC_RX_SHORT_CALLS,
  RADIO_HEALTH_METRIC_RX_ZERO_RETURN_CALLS,
  RADIO_HEALTH_METRIC_RX_RECV_INFLIGHT,
  RADIO_HEALTH_METRIC_RX_ERROR_NONE,
  RADIO_HEALTH_METRIC_RX_ERROR_TIMEOUT,
  RADIO_HEALTH_METRIC_RX_ERROR_LATE_COMMAND,
  RADIO_HEALTH_METRIC_RX_ERROR_BROKEN_CHAIN,
  RADIO_HEALTH_METRIC_RX_ERROR_OVERFLOW,
  RADIO_HEALTH_METRIC_RX_ERROR_ALIGNMENT,
  RADIO_HEALTH_METRIC_RX_ERROR_BAD_PACKET,
  RADIO_HEALTH_METRIC_RX_ERROR_OTHER,
  RADIO_HEALTH_METRIC_RX_OUT_OF_SEQUENCE,
  RADIO_HEALTH_METRIC_RX_TIMESTAMP_GAPS,
  RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE,
  RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS,
  RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID,
  RADIO_HEALTH_METRIC_RX_SAMPLE_RATE_HZ,

  RADIO_HEALTH_METRIC_TX_QUEUE_ENQUEUES,
  RADIO_HEALTH_METRIC_TX_QUEUE_DEQUEUES,
  RADIO_HEALTH_METRIC_TX_QUEUE_DEPTH,
  RADIO_HEALTH_METRIC_TX_QUEUE_HIGH_WATER,
  RADIO_HEALTH_METRIC_TX_QUEUE_OVERFLOW_DISCARDS,

  RADIO_HEALTH_METRIC_TX_SAMPLE_RATE_MICROHZ,
  RADIO_HEALTH_METRIC_RX_SAMPLE_RATE_MICROHZ,

  RADIO_HEALTH_METRIC_COUNT,
} radio_health_metric_t;

#ifdef __cplusplus
static_assert(RADIO_HEALTH_METRIC_COUNT <= 64, "radio-health observed metric mask is uint64_t");
#else
_Static_assert(RADIO_HEALTH_METRIC_COUNT <= 64, "radio-health observed metric mask is uint64_t");
#endif

#define RADIO_HEALTH_METRIC_BIT(metric) (UINT64_C(1) << (unsigned int)(metric))

typedef enum {
  RADIO_HEALTH_METRIC_COUNTER = 0,
  RADIO_HEALTH_METRIC_GAUGE,
} radio_health_metric_kind_t;

/** Best-effort monitor copy: counters can change during copying; last metadata groups are coherent. */
typedef struct {
  /** Registry slot and monitor payload device_id. */
  uint32_t device_id;
  radio_health_backend_t backend;
  uint32_t device_type;
  radio_health_lifecycle_t lifecycle;
  uint64_t capabilities;
  /** A set bit means values[metric] is available; an unset gauge is missing. */
  uint64_t observed_metrics;
  uint64_t values[RADIO_HEALTH_METRIC_COUNT];
} radio_health_snapshot_t;

/** Enable or disable future registrations. Existing handles remain valid until closed. */
void radio_health_set_enabled(bool enabled);

/** Reserve an opaque process-lifetime slot, or return NULL when disabled/full/unsupported. */
radio_health_device_t *radio_health_register(radio_health_backend_t backend, uint32_t device_type, uint64_t capabilities);

/** Mark a device closed. Its final counters and gauges remain available to snapshots. */
void radio_health_close(radio_health_device_t *device);

/** Number of ever-reserved registry slots, bounded by RADIO_HEALTH_MAX_DEVICES. */
uint32_t radio_health_device_count(void);

/** Copy one registered active or closed slot. Returns false for an unused slot or invalid arguments. */
bool radio_health_snapshot(uint32_t device_id, radio_health_snapshot_t *snapshot);

/** Fixed schema helpers. Unknown metric indices return NULL / zero. */
const char *radio_health_backend_name(radio_health_backend_t backend);
const char *radio_health_metric_name(radio_health_metric_t metric);
radio_health_metric_kind_t radio_health_metric_kind(radio_health_metric_t metric);
uint64_t radio_health_metric_capabilities(radio_health_metric_t metric);
bool radio_health_metric_supported(uint64_t capabilities, radio_health_metric_t metric);

/**
 * Encode an actual backend API rate as rounded microhertz. This is API-value
 * resolution, not a calibrated oscillator-frequency measurement. Returns false
 * for invalid, unrepresentable, or rounded-zero rates and leaves encoded unchanged.
 */
bool radio_health_rate_microhz(double rate_hz, uint64_t *encoded);

/**
 * Lock-free producer operations. They return false for NULL or closed handles,
 * unsupported metrics, or the wrong metric kind.
 */
bool radio_health_counter_add(radio_health_device_t *device, radio_health_metric_t metric, uint64_t value);
bool radio_health_gauge_set(radio_health_device_t *device, radio_health_metric_t metric, uint64_t value);

/**
 * Atomically coherent last-observation updates for raw backend metadata.
 * A false validity flag is itself observed and prevents consumers from using
 * a prior device timestamp or channel as the latest observation. Each group
 * has one producer thread.
 */
void radio_health_observe_tx_async(radio_health_device_t *device,
                                   uint64_t monotonic_ns,
                                   uint64_t raw_code,
                                   bool channel_valid,
                                   uint64_t channel,
                                   bool device_time_valid,
                                   uint64_t device_ticks);
void radio_health_observe_rx_metadata(radio_health_device_t *device,
                                      uint64_t raw_error_code,
                                      bool device_time_valid,
                                      uint64_t device_ticks);

#ifdef __cplusplus
}
#endif

#endif /* RADIO_HEALTH_H */
