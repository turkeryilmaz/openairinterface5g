/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "radio_health.h"

#include <math.h>
#include <stdatomic.h>

typedef _Atomic(uint64_t) radio_health_atomic_u64_t;

typedef struct {
  const char *name;
  radio_health_metric_kind_t kind;
  uint64_t capabilities;
} radio_health_metric_info_t;

struct radio_health_device_s {
  uint32_t device_id;
};

typedef struct {
  struct radio_health_device_s handle;
  radio_health_backend_t backend;
  uint32_t device_type;
  uint64_t capabilities;
  _Atomic(uint32_t) lifecycle;
  radio_health_atomic_u64_t observed_metrics;
  radio_health_atomic_u64_t values[RADIO_HEALTH_METRIC_COUNT];
  radio_health_atomic_u64_t tx_async_sequence;
  radio_health_atomic_u64_t rx_metadata_sequence;
} radio_health_slot_t;

static _Atomic(bool) radio_health_enabled;
static _Atomic(uint32_t) radio_health_next_slot;
static radio_health_slot_t radio_health_slots[RADIO_HEALTH_MAX_DEVICES];

static const radio_health_metric_info_t radio_health_metrics[RADIO_HEALTH_METRIC_COUNT] = {
    [RADIO_HEALTH_METRIC_TX_SEND_CALLS] = {"tx_send_calls", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_TX_SEND_REQUESTED_SAMPLES] = {"tx_send_requested_samples",
                                                       RADIO_HEALTH_METRIC_COUNTER,
                                                       RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_TX_SEND_ACCEPTED_SAMPLES] = {"tx_send_accepted_samples",
                                                      RADIO_HEALTH_METRIC_COUNTER,
                                                      RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_TX_SEND_SHORT_CALLS] = {"tx_send_short_calls", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_TX_SEND_EXCEPTIONS] = {"tx_send_exceptions", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_TX_SEND_INFLIGHT] = {"tx_send_inflight", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_TX_SAMPLE_RATE_HZ] = {"tx_sample_rate_hz", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_TX_SEND},

    [RADIO_HEALTH_METRIC_TX_ASYNC_POLLS] = {"tx_async_polls", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_MESSAGES] = {"tx_async_messages", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_EXCEPTIONS] = {"tx_async_exceptions", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_TIME_ERROR] = {"tx_async_time_error", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_UNDERFLOW] = {"tx_async_underflow", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_UNDERFLOW_IN_PACKET] = {"tx_async_underflow_in_packet",
                                                          RADIO_HEALTH_METRIC_COUNTER,
                                                          RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_SEQ_ERROR] = {"tx_async_seq_error", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_SEQ_ERROR_IN_BURST] = {"tx_async_seq_error_in_burst",
                                                         RADIO_HEALTH_METRIC_COUNTER,
                                                         RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_BURST_ACK] = {"tx_async_burst_ack", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_UNKNOWN] = {"tx_async_unknown", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_POLL_MONO_NS] = {"tx_async_last_poll_mono_ns",
                                                        RADIO_HEALTH_METRIC_GAUGE,
                                                        RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS] = {"tx_async_last_event_mono_ns",
                                                         RADIO_HEALTH_METRIC_GAUGE,
                                                         RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE] = {"tx_async_last_event_raw_code",
                                                          RADIO_HEALTH_METRIC_GAUGE,
                                                          RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL] = {"tx_async_last_event_channel",
                                                         RADIO_HEALTH_METRIC_GAUGE,
                                                         RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID] = {"tx_async_last_event_channel_valid",
                                                               RADIO_HEALTH_METRIC_GAUGE,
                                                               RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS] = {"tx_async_last_event_device_ticks",
                                                              RADIO_HEALTH_METRIC_GAUGE,
                                                              RADIO_HEALTH_CAP_TX_ASYNC},
    [RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID] = {"tx_async_last_event_device_time_valid",
                                                                   RADIO_HEALTH_METRIC_GAUGE,
                                                                   RADIO_HEALTH_CAP_TX_ASYNC},

    [RADIO_HEALTH_METRIC_RX_RECV_CALLS] = {"rx_recv_calls", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_REQUESTED_SAMPLES] = {"rx_requested_samples", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_RETURNED_SAMPLES] = {"rx_returned_samples", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_SHORT_CALLS] = {"rx_short_calls", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ZERO_RETURN_CALLS] = {"rx_zero_return_calls", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_RECV_INFLIGHT] = {"rx_recv_inflight", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_NONE] = {"rx_error_none", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_TIMEOUT] = {"rx_error_timeout", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_LATE_COMMAND] = {"rx_error_late_command",
                                                   RADIO_HEALTH_METRIC_COUNTER,
                                                   RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_BROKEN_CHAIN] = {"rx_error_broken_chain",
                                                   RADIO_HEALTH_METRIC_COUNTER,
                                                   RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_OVERFLOW] = {"rx_error_overflow", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_ALIGNMENT] = {"rx_error_alignment", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_BAD_PACKET] = {"rx_error_bad_packet", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_ERROR_OTHER] = {"rx_error_other", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_OUT_OF_SEQUENCE] = {"rx_out_of_sequence", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_TIMESTAMP_GAPS] = {"rx_timestamp_gaps", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE] = {"rx_last_error_raw_code",
                                                    RADIO_HEALTH_METRIC_GAUGE,
                                                    RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS] = {"rx_last_device_ticks", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID] = {"rx_last_device_time_valid",
                                                       RADIO_HEALTH_METRIC_GAUGE,
                                                       RADIO_HEALTH_CAP_RX_STREAM},
    [RADIO_HEALTH_METRIC_RX_SAMPLE_RATE_HZ] = {"rx_sample_rate_hz", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_RX_STREAM},

    [RADIO_HEALTH_METRIC_TX_QUEUE_ENQUEUES] = {"tx_queue_enqueues", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_QUEUE},
    [RADIO_HEALTH_METRIC_TX_QUEUE_DEQUEUES] = {"tx_queue_dequeues", RADIO_HEALTH_METRIC_COUNTER, RADIO_HEALTH_CAP_TX_QUEUE},
    [RADIO_HEALTH_METRIC_TX_QUEUE_DEPTH] = {"tx_queue_depth", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_TX_QUEUE},
    [RADIO_HEALTH_METRIC_TX_QUEUE_HIGH_WATER] = {"tx_queue_high_water", RADIO_HEALTH_METRIC_GAUGE, RADIO_HEALTH_CAP_TX_QUEUE},
    [RADIO_HEALTH_METRIC_TX_QUEUE_OVERFLOW_DISCARDS] = {"tx_queue_overflow_discards",
                                                        RADIO_HEALTH_METRIC_COUNTER,
                                                        RADIO_HEALTH_CAP_TX_QUEUE},

    [RADIO_HEALTH_METRIC_TX_SAMPLE_RATE_MICROHZ] = {"tx_sample_rate_microhz",
                                                     RADIO_HEALTH_METRIC_GAUGE,
                                                     RADIO_HEALTH_CAP_TX_SEND},
    [RADIO_HEALTH_METRIC_RX_SAMPLE_RATE_MICROHZ] = {"rx_sample_rate_microhz",
                                                     RADIO_HEALTH_METRIC_GAUGE,
                                                     RADIO_HEALTH_CAP_RX_STREAM},
};

static bool radio_health_metric_valid(radio_health_metric_t metric)
{
  return metric >= 0 && metric < RADIO_HEALTH_METRIC_COUNT;
}

static radio_health_slot_t *radio_health_slot(radio_health_device_t *device)
{
  return (radio_health_slot_t *)(void *)device;
}

static bool radio_health_slot_active(const radio_health_slot_t *slot)
{
  return slot != NULL && atomic_load_explicit(&slot->lifecycle, memory_order_relaxed) == RADIO_HEALTH_LIFECYCLE_ACTIVE;
}

static void radio_health_set_gauge(radio_health_slot_t *slot, radio_health_metric_t metric, uint64_t value)
{
  atomic_store_explicit(&slot->values[metric], value, memory_order_relaxed);
  /* An acquiring snapshot that sees this bit also sees the gauge value. */
  atomic_fetch_or_explicit(&slot->observed_metrics, RADIO_HEALTH_METRIC_BIT(metric), memory_order_release);
}

static void radio_health_clear_last_tx_async(radio_health_snapshot_t *snapshot)
{
  const uint64_t bits = RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID);
  snapshot->observed_metrics &= ~bits;
}

static void radio_health_clear_last_rx_metadata(radio_health_snapshot_t *snapshot)
{
  const uint64_t bits = RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS)
                        | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID);
  snapshot->observed_metrics &= ~bits;
}

void radio_health_set_enabled(bool enabled)
{
  atomic_store_explicit(&radio_health_enabled, enabled, memory_order_release);
}

radio_health_device_t *radio_health_register(radio_health_backend_t backend, uint32_t device_type, uint64_t capabilities)
{
  if (!atomic_load_explicit(&radio_health_enabled, memory_order_acquire))
    return NULL;

  radio_health_atomic_u64_t lock_free_probe;
  atomic_init(&lock_free_probe, 0);
  if (!atomic_is_lock_free(&lock_free_probe))
    return NULL;

  uint32_t device_id = atomic_load_explicit(&radio_health_next_slot, memory_order_relaxed);
  while (device_id < RADIO_HEALTH_MAX_DEVICES
         && !atomic_compare_exchange_weak_explicit(&radio_health_next_slot,
                                                   &device_id,
                                                   device_id + 1,
                                                   memory_order_acq_rel,
                                                   memory_order_relaxed)) {
  }
  if (device_id >= RADIO_HEALTH_MAX_DEVICES)
    return NULL;

  radio_health_slot_t *slot = &radio_health_slots[device_id];
  slot->handle.device_id = device_id;
  slot->backend = backend;
  slot->device_type = device_type;
  slot->capabilities = capabilities;
  for (radio_health_metric_t metric = 0; metric < RADIO_HEALTH_METRIC_COUNT; metric++) {
    if (radio_health_metrics[metric].kind == RADIO_HEALTH_METRIC_COUNTER && radio_health_metric_supported(capabilities, metric))
      atomic_fetch_or_explicit(&slot->observed_metrics, RADIO_HEALTH_METRIC_BIT(metric), memory_order_relaxed);
  }
  atomic_store_explicit(&slot->lifecycle, RADIO_HEALTH_LIFECYCLE_ACTIVE, memory_order_release);
  return &slot->handle;
}

void radio_health_close(radio_health_device_t *device)
{
  radio_health_slot_t *slot = radio_health_slot(device);
  if (slot == NULL)
    return;
  atomic_store_explicit(&slot->lifecycle, RADIO_HEALTH_LIFECYCLE_CLOSED, memory_order_release);
}

uint32_t radio_health_device_count(void)
{
  const uint32_t count = atomic_load_explicit(&radio_health_next_slot, memory_order_acquire);
  return count < RADIO_HEALTH_MAX_DEVICES ? count : RADIO_HEALTH_MAX_DEVICES;
}

bool radio_health_snapshot(uint32_t device_id, radio_health_snapshot_t *snapshot)
{
  if (snapshot == NULL || device_id >= radio_health_device_count())
    return false;

  const radio_health_slot_t *slot = &radio_health_slots[device_id];
  const uint32_t lifecycle = atomic_load_explicit(&slot->lifecycle, memory_order_acquire);
  if (lifecycle == RADIO_HEALTH_LIFECYCLE_UNUSED)
    return false;

  snapshot->device_id = device_id;
  snapshot->backend = slot->backend;
  snapshot->device_type = slot->device_type;
  snapshot->lifecycle = (radio_health_lifecycle_t)lifecycle;
  snapshot->capabilities = slot->capabilities;
  snapshot->observed_metrics = atomic_load_explicit(&slot->observed_metrics, memory_order_acquire);
  for (radio_health_metric_t metric = 0; metric < RADIO_HEALTH_METRIC_COUNT; metric++)
    snapshot->values[metric] = atomic_load_explicit(&slot->values[metric], memory_order_relaxed);

  bool tx_async_coherent = false;
  for (unsigned int attempt = 0; attempt < 4; attempt++) {
    const uint64_t start = atomic_load_explicit(&slot->tx_async_sequence, memory_order_acquire);
    if (start & 1)
      continue;
    snapshot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID], memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    if (start == atomic_load_explicit(&slot->tx_async_sequence, memory_order_acquire) && !(start & 1)) {
      tx_async_coherent = true;
      break;
    }
  }
  if (!tx_async_coherent)
    radio_health_clear_last_tx_async(snapshot);

  bool rx_metadata_coherent = false;
  for (unsigned int attempt = 0; attempt < 4; attempt++) {
    const uint64_t start = atomic_load_explicit(&slot->rx_metadata_sequence, memory_order_acquire);
    if (start & 1)
      continue;
    snapshot->values[RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS], memory_order_relaxed);
    snapshot->values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID] =
        atomic_load_explicit(&slot->values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID], memory_order_relaxed);
    atomic_thread_fence(memory_order_seq_cst);
    if (start == atomic_load_explicit(&slot->rx_metadata_sequence, memory_order_acquire) && !(start & 1)) {
      rx_metadata_coherent = true;
      break;
    }
  }
  if (!rx_metadata_coherent)
    radio_health_clear_last_rx_metadata(snapshot);

  return true;
}

const char *radio_health_backend_name(radio_health_backend_t backend)
{
  switch (backend) {
    case RADIO_HEALTH_BACKEND_UHD:
      return "uhd";
    case RADIO_HEALTH_BACKEND_UNKNOWN:
    default:
      return "unknown";
  }
}

const char *radio_health_metric_name(radio_health_metric_t metric)
{
  return radio_health_metric_valid(metric) ? radio_health_metrics[metric].name : NULL;
}

radio_health_metric_kind_t radio_health_metric_kind(radio_health_metric_t metric)
{
  return radio_health_metric_valid(metric) ? radio_health_metrics[metric].kind : RADIO_HEALTH_METRIC_COUNTER;
}

uint64_t radio_health_metric_capabilities(radio_health_metric_t metric)
{
  return radio_health_metric_valid(metric) ? radio_health_metrics[metric].capabilities : 0;
}

bool radio_health_metric_supported(uint64_t capabilities, radio_health_metric_t metric)
{
  if (!radio_health_metric_valid(metric))
    return false;
  const uint64_t required = radio_health_metrics[metric].capabilities;
  return required != 0 && (capabilities & required) == required;
}

bool radio_health_rate_microhz(double rate_hz, uint64_t *encoded)
{
  if (encoded == NULL || !isfinite(rate_hz) || rate_hz <= 0)
    return false;

  const long double scaled = (long double)rate_hz * 1000000.0L;
  const long double limit = 18446744073709551616.0L; /* 2^64 */
  if (!isfinite(scaled) || scaled >= limit)
    return false;

  const uint64_t integer_part = (uint64_t)scaled;
  const long double fractional_part = scaled - (long double)integer_part;
  uint64_t rounded = integer_part;
  if (fractional_part >= 0.5L) {
    if (integer_part == UINT64_MAX)
      return false;
    rounded = integer_part + 1;
  }
  if (rounded == 0)
    return false;

  *encoded = rounded;
  return true;
}

bool radio_health_counter_add(radio_health_device_t *device, radio_health_metric_t metric, uint64_t value)
{
  radio_health_slot_t *slot = radio_health_slot(device);
  if (!radio_health_slot_active(slot) || !radio_health_metric_valid(metric)
      || radio_health_metrics[metric].kind != RADIO_HEALTH_METRIC_COUNTER
      || !radio_health_metric_supported(slot->capabilities, metric))
    return false;
  atomic_fetch_add_explicit(&slot->values[metric], value, memory_order_relaxed);
  return true;
}

bool radio_health_gauge_set(radio_health_device_t *device, radio_health_metric_t metric, uint64_t value)
{
  radio_health_slot_t *slot = radio_health_slot(device);
  if (!radio_health_slot_active(slot) || !radio_health_metric_valid(metric)
      || radio_health_metrics[metric].kind != RADIO_HEALTH_METRIC_GAUGE
      || !radio_health_metric_supported(slot->capabilities, metric))
    return false;
  radio_health_set_gauge(slot, metric, value);
  return true;
}

void radio_health_observe_tx_async(radio_health_device_t *device,
                                   uint64_t monotonic_ns,
                                   uint64_t raw_code,
                                   bool channel_valid,
                                   uint64_t channel,
                                   bool device_time_valid,
                                   uint64_t device_ticks)
{
  radio_health_slot_t *slot = radio_health_slot(device);
  if (!radio_health_slot_active(slot) || !(slot->capabilities & RADIO_HEALTH_CAP_TX_ASYNC))
    return;

  atomic_fetch_add_explicit(&slot->tx_async_sequence, 1, memory_order_acq_rel);
  /* Keep payload stores after the odd sequence publication for snapshot retry. */
  atomic_thread_fence(memory_order_release);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS], monotonic_ns, memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE], raw_code, memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL], channel, memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID],
                        channel_valid ? 1 : 0,
                        memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS],
                        device_ticks,
                        memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID],
                        device_time_valid ? 1 : 0,
                        memory_order_relaxed);
  atomic_fetch_or_explicit(&slot->observed_metrics,
                           RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID),
                           memory_order_release);
  atomic_fetch_add_explicit(&slot->tx_async_sequence, 1, memory_order_release);
}

void radio_health_observe_rx_metadata(radio_health_device_t *device,
                                      uint64_t raw_error_code,
                                      bool device_time_valid,
                                      uint64_t device_ticks)
{
  radio_health_slot_t *slot = radio_health_slot(device);
  if (!radio_health_slot_active(slot) || !(slot->capabilities & RADIO_HEALTH_CAP_RX_STREAM))
    return;

  atomic_fetch_add_explicit(&slot->rx_metadata_sequence, 1, memory_order_acq_rel);
  /* Keep payload stores after the odd sequence publication for snapshot retry. */
  atomic_thread_fence(memory_order_release);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE], raw_error_code, memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS], device_ticks, memory_order_relaxed);
  atomic_store_explicit(&slot->values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID],
                        device_time_valid ? 1 : 0,
                        memory_order_relaxed);
  atomic_fetch_or_explicit(&slot->observed_metrics,
                           RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS)
                               | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID),
                           memory_order_release);
  atomic_fetch_add_explicit(&slot->rx_metadata_sequence, 1, memory_order_release);
}
