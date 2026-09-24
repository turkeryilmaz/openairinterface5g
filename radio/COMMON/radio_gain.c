/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "radio_gain.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum radio_gain_mailbox_state {
  RADIO_GAIN_MAILBOX_IDLE = 0,
  RADIO_GAIN_MAILBOX_WRITING,
  RADIO_GAIN_MAILBOX_PENDING,
  RADIO_GAIN_MAILBOX_RUNNING,
  RADIO_GAIN_MAILBOX_DONE,
  RADIO_GAIN_MAILBOX_READING,
  RADIO_GAIN_MAILBOX_CLOSED,
};

typedef struct {
  _Atomic(uint64_t) sequence;
  _Atomic(unsigned int) operation;
  _Atomic(unsigned int) status;
  _Atomic(uint64_t) request_generation;
  _Atomic(uint64_t) request_id;
  _Atomic(uint64_t) result_generation;
  _Atomic(uint64_t) gain_db;
  _Atomic(uint64_t) rx_frequency_hz;
  _Atomic(uint64_t) tx_frequency_hz;
  _Atomic(uint64_t) tune_offset_hz;
  _Atomic(uint64_t) reported_rx_db;
  _Atomic(uint64_t) reported_tx_db;
  _Atomic(uint64_t) begin_device_ticks;
  _Atomic(uint64_t) end_device_ticks;
  _Atomic(unsigned int) flags;
} radio_gain_atomic_result_t;

enum {
  RADIO_GAIN_FLAG_DEVICE_TIME_VALID = 1U << 0,
  RADIO_GAIN_FLAG_RX_GAIN_VALID = 1U << 1,
  RADIO_GAIN_FLAG_TX_GAIN_VALID = 1U << 2,
  RADIO_GAIN_SNAPSHOT_ATTEMPTS = 4,
  RADIO_GAIN_TX_END_CAS_ATTEMPTS = 2,
};

struct radio_gain_owner {
  radio_gain_api_t api;
  void *device;
  unsigned int rx_channel;
  unsigned int tx_channel;
  bool host_rx_control;

  pthread_t worker;
  pthread_mutex_t startup_mutex;
  pthread_cond_t startup_cond;
  bool startup_complete;
  bool startup_ok;

  pthread_mutex_t channels_mutex;
  radio_gain_channel_t rx_channel_info;
  radio_gain_channel_t tx_channel_info;
  bool channel_profiles_valid;

  _Atomic(bool) accepting;
  _Atomic(int) terminal_failure;
  void (*failure_handler)(int);
  _Atomic(int) mailbox_state;
  radio_gain_request_t mailbox_request;
  radio_gain_result_t mailbox_result;

  _Atomic(uint64_t) generation;
  _Atomic(bool) tx_admission_open;
  _Atomic(bool) tx_settings_inflight;
  _Atomic(bool) tx_end_known;
  _Atomic(bool) tx_end_uncertain;
  _Atomic(int64_t) last_tx_end;
  radio_gain_atomic_result_t snapshot;
};

_Static_assert(sizeof(double) == sizeof(uint64_t), "radio gain snapshots require 64-bit double");

static void radio_gain_atomic_result_init(radio_gain_atomic_result_t *snapshot)
{
  atomic_init(&snapshot->sequence, 0);
  atomic_init(&snapshot->operation, 0);
  atomic_init(&snapshot->status, 0);
  atomic_init(&snapshot->request_generation, 0);
  atomic_init(&snapshot->request_id, 0);
  atomic_init(&snapshot->result_generation, 0);
  atomic_init(&snapshot->gain_db, 0);
  atomic_init(&snapshot->rx_frequency_hz, 0);
  atomic_init(&snapshot->tx_frequency_hz, 0);
  atomic_init(&snapshot->tune_offset_hz, 0);
  atomic_init(&snapshot->reported_rx_db, 0);
  atomic_init(&snapshot->reported_tx_db, 0);
  atomic_init(&snapshot->begin_device_ticks, 0);
  atomic_init(&snapshot->end_device_ticks, 0);
  atomic_init(&snapshot->flags, 0);
}

/* RT producers only touch atomic fields. Reject a target where any of these
 * atomics could route through a hidden libatomic lock. */
static bool radio_gain_atomics_lock_free(struct radio_gain_owner *owner)
{
  const radio_gain_atomic_result_t *snapshot = &owner->snapshot;
  return atomic_is_lock_free(&owner->accepting) && atomic_is_lock_free(&owner->terminal_failure)
         && atomic_is_lock_free(&owner->mailbox_state) && atomic_is_lock_free(&owner->generation)
         && atomic_is_lock_free(&owner->tx_admission_open) && atomic_is_lock_free(&owner->tx_settings_inflight)
         && atomic_is_lock_free(&owner->tx_end_known) && atomic_is_lock_free(&owner->tx_end_uncertain)
         && atomic_is_lock_free(&owner->last_tx_end) && atomic_is_lock_free(&snapshot->sequence)
         && atomic_is_lock_free(&snapshot->operation) && atomic_is_lock_free(&snapshot->status)
         && atomic_is_lock_free(&snapshot->request_generation) && atomic_is_lock_free(&snapshot->request_id)
         && atomic_is_lock_free(&snapshot->result_generation) && atomic_is_lock_free(&snapshot->gain_db)
         && atomic_is_lock_free(&snapshot->rx_frequency_hz) && atomic_is_lock_free(&snapshot->tx_frequency_hz)
         && atomic_is_lock_free(&snapshot->tune_offset_hz) && atomic_is_lock_free(&snapshot->reported_rx_db)
         && atomic_is_lock_free(&snapshot->reported_tx_db) && atomic_is_lock_free(&snapshot->begin_device_ticks)
         && atomic_is_lock_free(&snapshot->end_device_ticks) && atomic_is_lock_free(&snapshot->flags);
}

static uint64_t radio_gain_double_bits(double value)
{
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static double radio_gain_bits_double(uint64_t bits)
{
  double value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint64_t radio_gain_i64_bits(int64_t value)
{
  uint64_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static int64_t radio_gain_bits_i64(uint64_t bits)
{
  int64_t value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static radio_gain_result_t radio_gain_empty_result(const radio_gain_request_t *request, uint64_t generation)
{
  radio_gain_result_t result = {
      .status = RADIO_GAIN_INVALID,
      .generation = generation,
      .reported_rx_db = NAN,
      .reported_tx_db = NAN,
  };
  if (request != NULL)
    result.request = *request;
  return result;
}

static void radio_gain_snapshot_store(struct radio_gain_owner *owner, const radio_gain_result_t *result)
{
  radio_gain_atomic_result_t *snapshot = &owner->snapshot;
  /* Settings transitions are infrequent. Use one SC order for the complete
   * atomic payload and bounded reader retry, rather than relying on subtle
   * mixed-order seqlock reasoning on weak-memory targets. */
  atomic_fetch_add_explicit(&snapshot->sequence, 1, memory_order_seq_cst);
  atomic_store_explicit(&snapshot->operation, result->request.operation, memory_order_seq_cst);
  atomic_store_explicit(&snapshot->status, result->status, memory_order_seq_cst);
  atomic_store_explicit(&snapshot->request_generation, result->request.generation, memory_order_seq_cst);
  atomic_store_explicit(&snapshot->request_id, result->request.request_id, memory_order_seq_cst);
  atomic_store_explicit(&snapshot->result_generation, result->generation, memory_order_seq_cst);
  atomic_store_explicit(&snapshot->gain_db, radio_gain_double_bits(result->request.gain_db), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->rx_frequency_hz, radio_gain_double_bits(result->request.rx_frequency_hz), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->tx_frequency_hz, radio_gain_double_bits(result->request.tx_frequency_hz), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->tune_offset_hz, radio_gain_double_bits(result->request.tune_offset_hz), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->reported_rx_db, radio_gain_double_bits(result->reported_rx_db), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->reported_tx_db, radio_gain_double_bits(result->reported_tx_db), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->begin_device_ticks, radio_gain_i64_bits(result->begin_device_ticks), memory_order_seq_cst);
  atomic_store_explicit(&snapshot->end_device_ticks, radio_gain_i64_bits(result->end_device_ticks), memory_order_seq_cst);
  const unsigned int flags = (result->device_time_valid ? RADIO_GAIN_FLAG_DEVICE_TIME_VALID : 0)
                             | (result->rx_gain_valid ? RADIO_GAIN_FLAG_RX_GAIN_VALID : 0)
                             | (result->tx_gain_valid ? RADIO_GAIN_FLAG_TX_GAIN_VALID : 0);
  atomic_store_explicit(&snapshot->flags, flags, memory_order_seq_cst);
  atomic_fetch_add_explicit(&snapshot->sequence, 1, memory_order_seq_cst);
}

static bool radio_gain_snapshot_load(const struct radio_gain_owner *owner, radio_gain_result_t *result)
{
  const radio_gain_atomic_result_t *snapshot = &owner->snapshot;
  for (unsigned int attempt = 0; attempt < RADIO_GAIN_SNAPSHOT_ATTEMPTS; ++attempt) {
    const uint64_t start = atomic_load_explicit(&snapshot->sequence, memory_order_seq_cst);
    if (start & 1U)
      continue;

    radio_gain_result_t copy = {
        .request =
            {
                .operation = (radio_gain_operation_t)atomic_load_explicit(&snapshot->operation, memory_order_seq_cst),
                .generation = atomic_load_explicit(&snapshot->request_generation, memory_order_seq_cst),
                .request_id = atomic_load_explicit(&snapshot->request_id, memory_order_seq_cst),
                .gain_db = radio_gain_bits_double(atomic_load_explicit(&snapshot->gain_db, memory_order_seq_cst)),
                .rx_frequency_hz = radio_gain_bits_double(atomic_load_explicit(&snapshot->rx_frequency_hz, memory_order_seq_cst)),
                .tx_frequency_hz = radio_gain_bits_double(atomic_load_explicit(&snapshot->tx_frequency_hz, memory_order_seq_cst)),
                .tune_offset_hz = radio_gain_bits_double(atomic_load_explicit(&snapshot->tune_offset_hz, memory_order_seq_cst)),
            },
        .status = (radio_gain_status_t)atomic_load_explicit(&snapshot->status, memory_order_seq_cst),
        .generation = atomic_load_explicit(&snapshot->result_generation, memory_order_seq_cst),
        .reported_rx_db = radio_gain_bits_double(atomic_load_explicit(&snapshot->reported_rx_db, memory_order_seq_cst)),
        .reported_tx_db = radio_gain_bits_double(atomic_load_explicit(&snapshot->reported_tx_db, memory_order_seq_cst)),
        .begin_device_ticks = radio_gain_bits_i64(atomic_load_explicit(&snapshot->begin_device_ticks, memory_order_seq_cst)),
        .end_device_ticks = radio_gain_bits_i64(atomic_load_explicit(&snapshot->end_device_ticks, memory_order_seq_cst)),
    };
    const unsigned int flags = atomic_load_explicit(&snapshot->flags, memory_order_seq_cst);
    copy.device_time_valid = (flags & RADIO_GAIN_FLAG_DEVICE_TIME_VALID) != 0;
    copy.rx_gain_valid = (flags & RADIO_GAIN_FLAG_RX_GAIN_VALID) != 0;
    copy.tx_gain_valid = (flags & RADIO_GAIN_FLAG_TX_GAIN_VALID) != 0;

    if (start == atomic_load_explicit(&snapshot->sequence, memory_order_seq_cst) && !(start & 1U)) {
      *result = copy;
      return true;
    }
  }
  return false;
}

static bool radio_gain_channel_valid(const radio_gain_channel_t *channel)
{
  return isfinite(channel->minimum_db) && isfinite(channel->maximum_db) && channel->minimum_db <= channel->maximum_db
         && isfinite(channel->step_db) && channel->step_db >= 0.0 && isfinite(channel->reported_db)
         && channel->reported_db >= channel->minimum_db && channel->reported_db <= channel->maximum_db
         && isfinite(channel->frequency_hz) && channel->frequency_hz >= 0.0 && isfinite(channel->sample_rate_hz)
         && channel->sample_rate_hz >= 0.0 && (!channel->power_reference_valid || isfinite(channel->power_reference_dbm));
}

static bool radio_gain_query_channels(struct radio_gain_owner *owner)
{
  radio_gain_channel_t rx = {0};
  radio_gain_channel_t tx = {0};
  if (owner->api.query(owner->device, RADIO_GAIN_RX, owner->rx_channel, &rx) != 0
      || owner->api.query(owner->device, RADIO_GAIN_TX, owner->tx_channel, &tx) != 0)
    return false;

  rx.identity[sizeof(rx.identity) - 1] = '\0';
  rx.antenna[sizeof(rx.antenna) - 1] = '\0';
  tx.identity[sizeof(tx.identity) - 1] = '\0';
  tx.antenna[sizeof(tx.antenna) - 1] = '\0';
  if (!radio_gain_channel_valid(&rx) || !radio_gain_channel_valid(&tx))
    return false;

  pthread_mutex_lock(&owner->channels_mutex);
  owner->rx_channel_info = rx;
  owner->tx_channel_info = tx;
  owner->channel_profiles_valid = true;
  pthread_mutex_unlock(&owner->channels_mutex);
  return true;
}

static void radio_gain_invalidate_channel_profiles(struct radio_gain_owner *owner)
{
  pthread_mutex_lock(&owner->channels_mutex);
  owner->channel_profiles_valid = false;
  pthread_mutex_unlock(&owner->channels_mutex);
}

static bool radio_gain_current_channel(struct radio_gain_owner *owner,
                                       radio_gain_direction_t direction,
                                       radio_gain_channel_t *channel)
{
  pthread_mutex_lock(&owner->channels_mutex);
  const bool valid = owner->channel_profiles_valid;
  if (valid)
    *channel = direction == RADIO_GAIN_RX ? owner->rx_channel_info : owner->tx_channel_info;
  pthread_mutex_unlock(&owner->channels_mutex);
  return valid;
}

static void radio_gain_update_reported_gain(struct radio_gain_owner *owner, radio_gain_direction_t direction, double reported_db)
{
  pthread_mutex_lock(&owner->channels_mutex);
  if (owner->channel_profiles_valid) {
    if (direction == RADIO_GAIN_RX)
      owner->rx_channel_info.reported_db = reported_db;
    else
      owner->tx_channel_info.reported_db = reported_db;
  }
  pthread_mutex_unlock(&owner->channels_mutex);
}

static bool radio_gain_read_ticks(struct radio_gain_owner *owner, double sample_rate_hz, int64_t *ticks)
{
  return owner->api.device_ticks != NULL && isfinite(sample_rate_hz) && sample_rate_hz > 0.0
         && owner->api.device_ticks(owner->device, sample_rate_hz, ticks) == 0;
}

/* A gain setter can alter the analogue state even when it reports an error.
 * Publish the pre-call tick before the transition and only qualify the final
 * result when a later tick produces a monotonic bracket. */
static void radio_gain_begin_device_time_bracket(struct radio_gain_owner *owner, double sample_rate_hz, radio_gain_result_t *result)
{
  int64_t begin_ticks;
  if (!radio_gain_read_ticks(owner, sample_rate_hz, &begin_ticks))
    return;

  result->begin_device_ticks = begin_ticks;
  result->device_time_valid = true;
}

static void radio_gain_finish_device_time_bracket(struct radio_gain_owner *owner,
                                                  double sample_rate_hz,
                                                  radio_gain_result_t *result)
{
  if (!result->device_time_valid)
    return;

  int64_t end_ticks;
  if (!radio_gain_read_ticks(owner, sample_rate_hz, &end_ticks) || end_ticks < result->begin_device_ticks) {
    result->device_time_valid = false;
    return;
  }

  result->end_device_ticks = end_ticks;
}

static void radio_gain_release_tx_gate(struct radio_gain_owner *owner)
{
  atomic_store_explicit(&owner->tx_settings_inflight, false, memory_order_release);
}

/* The returned true owns tx_settings_inflight through the backend transaction.
 * An opener first acquires the same gate, so it cannot admit a new TX interval
 * between this final recheck and the setter/retune call. */
static bool radio_gain_acquire_tx_quiescence(struct radio_gain_owner *owner, double sample_rate_hz, radio_gain_result_t *result)
{
  if (atomic_load_explicit(&owner->tx_admission_open, memory_order_acquire)
      || atomic_load_explicit(&owner->tx_end_uncertain, memory_order_acquire))
    return false;

  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&owner->tx_settings_inflight,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return false;
  if (atomic_load_explicit(&owner->tx_admission_open, memory_order_acquire)
      || atomic_load_explicit(&owner->tx_end_uncertain, memory_order_acquire)) {
    radio_gain_release_tx_gate(owner);
    return false;
  }

  if (!atomic_load_explicit(&owner->tx_end_known, memory_order_acquire))
    return true; /* Pre-stream startup: no submitted TX interval needs retirement. */

  int64_t ticks;
  if (!radio_gain_read_ticks(owner, sample_rate_hz, &ticks)) {
    radio_gain_release_tx_gate(owner);
    return false;
  }

  const int64_t scheduled_end = atomic_load_explicit(&owner->last_tx_end, memory_order_acquire);
  if (atomic_load_explicit(&owner->tx_admission_open, memory_order_acquire)
      || atomic_load_explicit(&owner->tx_end_uncertain, memory_order_acquire)
      || !atomic_load_explicit(&owner->tx_end_known, memory_order_acquire) || ticks < scheduled_end) {
    radio_gain_release_tx_gate(owner);
    return false;
  }

  result->begin_device_ticks = ticks;
  result->device_time_valid = true;
  return true;
}

static bool radio_gain_advance_generation(struct radio_gain_owner *owner, radio_gain_result_t *result)
{
  const uint64_t current = atomic_load_explicit(&owner->generation, memory_order_acquire);
  if (current == UINT64_MAX)
    return false;
  result->generation = current + 1;
  atomic_store_explicit(&owner->generation, result->generation, memory_order_release);
  return true;
}

static void radio_gain_publish_attempt(struct radio_gain_owner *owner,
                                       radio_gain_result_t *result,
                                       radio_gain_direction_t direction,
                                       bool retune)
{
  radio_gain_result_t previous;
  if (!radio_gain_snapshot_load(owner, &previous))
    previous = radio_gain_empty_result(&result->request, result->generation);
  previous.request = result->request;
  previous.generation = result->generation;
  previous.status = RADIO_GAIN_BUSY;
  /* Preserve the pre-call device-time bracket while the transition snapshot
   * invalidates the setting itself. */
  previous.begin_device_ticks = result->begin_device_ticks;
  previous.end_device_ticks = result->end_device_ticks;
  previous.device_time_valid = result->device_time_valid;
  if (retune || direction == RADIO_GAIN_RX) {
    previous.rx_gain_valid = false;
    previous.reported_rx_db = NAN;
  }
  if (retune || direction == RADIO_GAIN_TX) {
    previous.tx_gain_valid = false;
    previous.reported_tx_db = NAN;
  }
  *result = previous;
  radio_gain_snapshot_store(owner, result);
}

static bool radio_gain_set_gain(struct radio_gain_owner *owner,
                                const radio_gain_request_t *request,
                                radio_gain_direction_t direction,
                                radio_gain_result_t *result)
{
  if (!isfinite(request->gain_db)) {
    result->status = RADIO_GAIN_INVALID;
    return false;
  }
  if (owner->api.set_gain == NULL) {
    result->status = RADIO_GAIN_UNSUPPORTED;
    return false;
  }

  radio_gain_channel_t channel;
  if (!radio_gain_current_channel(owner, direction, &channel)) {
    result->status = RADIO_GAIN_BACKEND_ERROR;
    return false;
  }
  const bool tx_transaction = direction == RADIO_GAIN_TX;
  if (tx_transaction && !radio_gain_acquire_tx_quiescence(owner, channel.sample_rate_hz, result)) {
    result->status = RADIO_GAIN_TX_PENDING;
    return false;
  }
  if (!tx_transaction)
    radio_gain_begin_device_time_bracket(owner, channel.sample_rate_hz, result);
  if (!radio_gain_advance_generation(owner, result)) {
    if (tx_transaction)
      radio_gain_release_tx_gate(owner);
    result->status = RADIO_GAIN_INVALID;
    return false;
  }

  radio_gain_publish_attempt(owner, result, direction, false);
  const double requested = fmin(fmax(request->gain_db, channel.minimum_db), channel.maximum_db);
  double reported = NAN;
  const unsigned int index = direction == RADIO_GAIN_RX ? owner->rx_channel : owner->tx_channel;
  const int set_status = owner->api.set_gain(owner->device, direction, index, requested, &reported);
  if (!tx_transaction)
    radio_gain_finish_device_time_bracket(owner, channel.sample_rate_hz, result);
  if (set_status != 0 || !isfinite(reported) || reported < channel.minimum_db || reported > channel.maximum_db) {
    result->status = RADIO_GAIN_BACKEND_ERROR;
    if (direction == RADIO_GAIN_RX) {
      result->reported_rx_db = NAN;
      result->rx_gain_valid = false;
    } else {
      result->reported_tx_db = NAN;
      result->tx_gain_valid = false;
    }
    radio_gain_snapshot_store(owner, result);
    if (tx_transaction)
      radio_gain_release_tx_gate(owner);
    return true;
  }

  result->status = RADIO_GAIN_OK;
  if (direction == RADIO_GAIN_RX) {
    result->reported_rx_db = reported;
    result->rx_gain_valid = true;
  } else {
    result->reported_tx_db = reported;
    result->tx_gain_valid = true;
  }
  radio_gain_update_reported_gain(owner, direction, reported);
  if (tx_transaction && result->device_time_valid) {
    int64_t end_ticks;
    if (radio_gain_read_ticks(owner, channel.sample_rate_hz, &end_ticks))
      result->end_device_ticks = end_ticks;
    else
      result->device_time_valid = false;
  }
  radio_gain_snapshot_store(owner, result);
  if (tx_transaction)
    radio_gain_release_tx_gate(owner);
  return true;
}

static bool radio_gain_retune(struct radio_gain_owner *owner, const radio_gain_request_t *request, radio_gain_result_t *result)
{
  if (!isfinite(request->rx_frequency_hz) || request->rx_frequency_hz <= 0.0 || !isfinite(request->tx_frequency_hz)
      || request->tx_frequency_hz <= 0.0 || !isfinite(request->tune_offset_hz)) {
    result->status = RADIO_GAIN_INVALID;
    return false;
  }
  if (owner->api.retune == NULL) {
    result->status = RADIO_GAIN_UNSUPPORTED;
    return false;
  }

  radio_gain_channel_t tx_channel;
  if (!radio_gain_current_channel(owner, RADIO_GAIN_TX, &tx_channel)) {
    result->status = RADIO_GAIN_BACKEND_ERROR;
    return false;
  }
  if (!radio_gain_acquire_tx_quiescence(owner, tx_channel.sample_rate_hz, result)) {
    result->status = RADIO_GAIN_TX_PENDING;
    return false;
  }
  /* Startup has no TX interval to retire, but this RX-affecting transaction
   * still needs device-time bounds before its new sample epoch can settle. */
  if (!result->device_time_valid)
    radio_gain_begin_device_time_bracket(owner, tx_channel.sample_rate_hz, result);
  if (!radio_gain_advance_generation(owner, result)) {
    radio_gain_release_tx_gate(owner);
    result->status = RADIO_GAIN_INVALID;
    return false;
  }

  radio_gain_publish_attempt(owner, result, RADIO_GAIN_RX, true);
  radio_gain_invalidate_channel_profiles(owner);
  if (owner->api.retune(owner->device,
                        owner->rx_channel,
                        owner->tx_channel,
                        request->rx_frequency_hz,
                        request->tx_frequency_hz,
                        request->tune_offset_hz)
          != 0
      || !radio_gain_query_channels(owner)) {
    radio_gain_finish_device_time_bracket(owner, tx_channel.sample_rate_hz, result);
    result->status = RADIO_GAIN_BACKEND_ERROR;
    result->reported_rx_db = NAN;
    result->reported_tx_db = NAN;
    result->rx_gain_valid = false;
    result->tx_gain_valid = false;
    radio_gain_snapshot_store(owner, result);
    radio_gain_release_tx_gate(owner);
    return true;
  }

  radio_gain_channel_t rx_channel;
  if (!radio_gain_current_channel(owner, RADIO_GAIN_RX, &rx_channel)
      || !radio_gain_current_channel(owner, RADIO_GAIN_TX, &tx_channel)) {
    result->status = RADIO_GAIN_BACKEND_ERROR;
    radio_gain_snapshot_store(owner, result);
    radio_gain_release_tx_gate(owner);
    return true;
  }
  result->status = RADIO_GAIN_OK;
  result->reported_rx_db = rx_channel.reported_db;
  result->reported_tx_db = tx_channel.reported_db;
  result->rx_gain_valid = true;
  result->tx_gain_valid = true;
  radio_gain_finish_device_time_bracket(owner, tx_channel.sample_rate_hz, result);
  radio_gain_snapshot_store(owner, result);
  radio_gain_release_tx_gate(owner);
  return true;
}

static bool radio_gain_process_request(struct radio_gain_owner *owner,
                                       const radio_gain_request_t *request,
                                       radio_gain_result_t *result)
{
  const uint64_t generation = atomic_load_explicit(&owner->generation, memory_order_acquire);
  *result = radio_gain_empty_result(request, generation);
  if (request->generation != generation) {
    result->status = RADIO_GAIN_STALE;
    return false;
  }

  switch (request->operation) {
    case RADIO_GAIN_SET_RX:
      return radio_gain_set_gain(owner, request, RADIO_GAIN_RX, result);
    case RADIO_GAIN_SET_TX:
      return radio_gain_set_gain(owner, request, RADIO_GAIN_TX, result);
    case RADIO_GAIN_RETUNE:
      return radio_gain_retune(owner, request, result);
    default:
      result->status = RADIO_GAIN_INVALID;
      return false;
  }
}

static void radio_gain_sleep(void)
{
  const struct timespec interval = {.tv_sec = 0, .tv_nsec = 1000000L};
  (void)nanosleep(&interval, NULL);
}

static bool radio_gain_worker_initialize(struct radio_gain_owner *owner)
{
  if (!radio_gain_query_channels(owner))
    return false;
  if (owner->host_rx_control
      && (owner->api.set_rx_agc == NULL || owner->api.set_rx_agc(owner->device, owner->rx_channel, false) != 0))
    return false;

  radio_gain_result_t initial = radio_gain_empty_result(NULL, 1);
  initial.status = RADIO_GAIN_OK;
  initial.generation = 1;
  initial.reported_rx_db = owner->rx_channel_info.reported_db;
  initial.reported_tx_db = owner->tx_channel_info.reported_db;
  initial.rx_gain_valid = true;
  initial.tx_gain_valid = true;
  radio_gain_snapshot_store(owner, &initial);
  return true;
}

static void *radio_gain_worker(void *opaque)
{
  struct radio_gain_owner *owner = opaque;
  const bool initialized = radio_gain_worker_initialize(owner);
  pthread_mutex_lock(&owner->startup_mutex);
  owner->startup_ok = initialized;
  owner->startup_complete = true;
  pthread_cond_signal(&owner->startup_cond);
  pthread_mutex_unlock(&owner->startup_mutex);
  if (!initialized) {
    atomic_store_explicit(&owner->accepting, false, memory_order_release);
    atomic_store_explicit(&owner->mailbox_state, RADIO_GAIN_MAILBOX_CLOSED, memory_order_release);
    return NULL;
  }

  bool failure_delivered = false;
  for (;;) {
    const int failure = atomic_load_explicit(&owner->terminal_failure, memory_order_acquire);
    if (failure > 0 && !failure_delivered) {
      failure_delivered = true;
      if (owner->failure_handler)
        owner->failure_handler(failure);
    }
    int state = atomic_load_explicit(&owner->mailbox_state, memory_order_acquire);
    if (state == RADIO_GAIN_MAILBOX_PENDING) {
      int expected = RADIO_GAIN_MAILBOX_PENDING;
      if (atomic_compare_exchange_strong_explicit(&owner->mailbox_state,
                                                  &expected,
                                                  RADIO_GAIN_MAILBOX_RUNNING,
                                                  memory_order_acq_rel,
                                                  memory_order_acquire)) {
        radio_gain_result_t result;
        if (atomic_load_explicit(&owner->accepting, memory_order_acquire)) {
          (void)radio_gain_process_request(owner, &owner->mailbox_request, &result);
        } else {
          result = radio_gain_empty_result(&owner->mailbox_request, atomic_load_explicit(&owner->generation, memory_order_acquire));
          result.status = RADIO_GAIN_CLOSED;
        }
        owner->mailbox_result = result;
        atomic_store_explicit(&owner->mailbox_state, RADIO_GAIN_MAILBOX_DONE, memory_order_release);
        continue;
      }
      continue;
    }

    if (!atomic_load_explicit(&owner->accepting, memory_order_acquire)) {
      if (state == RADIO_GAIN_MAILBOX_IDLE || state == RADIO_GAIN_MAILBOX_DONE || state == RADIO_GAIN_MAILBOX_CLOSED) {
        atomic_store_explicit(&owner->mailbox_state, RADIO_GAIN_MAILBOX_CLOSED, memory_order_release);
        break;
      }
    }
    radio_gain_sleep();
  }
  return NULL;
}

static bool radio_gain_api_valid(const radio_gain_api_t *api)
{
  const size_t needed = offsetof(radio_gain_api_t, device_ticks) + sizeof(api->device_ticks);
  return api != NULL && api->abi_version == OAI_RADIO_GAIN_ABI && api->struct_size >= needed;
}

static void radio_gain_prefault(void *memory, size_t size)
{
  volatile unsigned char *bytes = memory;
  for (size_t offset = 0; offset < size; offset += 4096U)
    bytes[offset] = 0;
  if (size != 0)
    bytes[size - 1] = 0;
}

radio_gain_owner_t *radio_gain_owner_create(const radio_gain_api_t *api,
                                            void *device,
                                            unsigned int rx_channel,
                                            unsigned int tx_channel,
                                            bool host_rx_control)
{
  if (!radio_gain_api_valid(api))
    return NULL;

  struct radio_gain_owner *owner = malloc(sizeof(*owner));
  if (owner == NULL)
    return NULL;
  memset(owner, 0, sizeof(*owner));
  radio_gain_prefault(owner, sizeof(*owner));
  memcpy(&owner->api, api, sizeof(owner->api));
  if (owner->api.query == NULL) {
    free(owner);
    return NULL;
  }
  owner->device = device;
  owner->rx_channel = rx_channel;
  owner->tx_channel = tx_channel;
  owner->host_rx_control = host_rx_control;
  atomic_init(&owner->accepting, true);
  atomic_init(&owner->terminal_failure, 0);
  atomic_init(&owner->mailbox_state, RADIO_GAIN_MAILBOX_IDLE);
  atomic_init(&owner->generation, 1);
  atomic_init(&owner->tx_admission_open, false);
  atomic_init(&owner->tx_settings_inflight, false);
  atomic_init(&owner->tx_end_known, false);
  atomic_init(&owner->tx_end_uncertain, false);
  atomic_init(&owner->last_tx_end, INT64_MIN);
  radio_gain_atomic_result_init(&owner->snapshot);
  if (!radio_gain_atomics_lock_free(owner)) {
    free(owner);
    return NULL;
  }

  if (pthread_mutex_init(&owner->startup_mutex, NULL) != 0) {
    free(owner);
    return NULL;
  }
  if (pthread_cond_init(&owner->startup_cond, NULL) != 0) {
    pthread_mutex_destroy(&owner->startup_mutex);
    free(owner);
    return NULL;
  }
  if (pthread_mutex_init(&owner->channels_mutex, NULL) != 0) {
    pthread_cond_destroy(&owner->startup_cond);
    pthread_mutex_destroy(&owner->startup_mutex);
    free(owner);
    return NULL;
  }
  pthread_attr_t worker_attributes;
  if (pthread_attr_init(&worker_attributes) != 0) {
    pthread_mutex_destroy(&owner->channels_mutex);
    pthread_cond_destroy(&owner->startup_cond);
    pthread_mutex_destroy(&owner->startup_mutex);
    free(owner);
    return NULL;
  }
  const struct sched_param normal_priority = {.sched_priority = 0};
  const int attributes_ok = pthread_attr_setinheritsched(&worker_attributes, PTHREAD_EXPLICIT_SCHED) == 0
                            && pthread_attr_setschedpolicy(&worker_attributes, SCHED_OTHER) == 0
                            && pthread_attr_setschedparam(&worker_attributes, &normal_priority) == 0;
  const int worker_created = attributes_ok ? pthread_create(&owner->worker, &worker_attributes, radio_gain_worker, owner) : -1;
  const int attributes_destroyed = pthread_attr_destroy(&worker_attributes);
  if (!attributes_ok || worker_created != 0) {
    /* No affinity attribute is set: the worker retains the creator's affinity
     * while its scheduling class and priority are explicitly normal. */
    pthread_mutex_destroy(&owner->channels_mutex);
    pthread_cond_destroy(&owner->startup_cond);
    pthread_mutex_destroy(&owner->startup_mutex);
    free(owner);
    return NULL;
  }
  if (attributes_destroyed != 0) {
    /* The worker exists already. Do not release owner storage until it exits,
     * even though pthread_attr_destroy() failure is not expected for a local
     * initialized attribute object. */
    atomic_store_explicit(&owner->accepting, false, memory_order_release);
    (void)pthread_join(owner->worker, NULL);
    pthread_mutex_destroy(&owner->channels_mutex);
    pthread_cond_destroy(&owner->startup_cond);
    pthread_mutex_destroy(&owner->startup_mutex);
    free(owner);
    return NULL;
  }

  pthread_mutex_lock(&owner->startup_mutex);
  while (!owner->startup_complete)
    pthread_cond_wait(&owner->startup_cond, &owner->startup_mutex);
  const bool startup_ok = owner->startup_ok;
  pthread_mutex_unlock(&owner->startup_mutex);
  if (!startup_ok) {
    (void)pthread_join(owner->worker, NULL);
    pthread_mutex_destroy(&owner->channels_mutex);
    pthread_cond_destroy(&owner->startup_cond);
    pthread_mutex_destroy(&owner->startup_mutex);
    free(owner);
    return NULL;
  }
  return owner;
}

void radio_gain_owner_close(radio_gain_owner_t *owner)
{
  if (owner != NULL)
    atomic_store_explicit(&owner->accepting, false, memory_order_release);
}

void radio_gain_owner_destroy(radio_gain_owner_t *owner)
{
  if (owner == NULL)
    return;
  radio_gain_owner_close(owner);
  if (pthread_equal(pthread_self(), owner->worker))
    return; /* A callback cannot safely reclaim its own worker's storage. */
  (void)pthread_join(owner->worker, NULL);
  pthread_mutex_destroy(&owner->channels_mutex);
  pthread_cond_destroy(&owner->startup_cond);
  pthread_mutex_destroy(&owner->startup_mutex);
  free(owner);
}

radio_gain_status_t radio_gain_submit(radio_gain_owner_t *owner, const radio_gain_request_t *request)
{
  if (owner == NULL || request == NULL)
    return RADIO_GAIN_INVALID;
  if (!atomic_load_explicit(&owner->accepting, memory_order_acquire))
    return RADIO_GAIN_CLOSED;

  int expected = RADIO_GAIN_MAILBOX_IDLE;
  if (!atomic_compare_exchange_strong_explicit(&owner->mailbox_state,
                                               &expected,
                                               RADIO_GAIN_MAILBOX_WRITING,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return expected == RADIO_GAIN_MAILBOX_CLOSED ? RADIO_GAIN_CLOSED : RADIO_GAIN_BUSY;
  if (!atomic_load_explicit(&owner->accepting, memory_order_acquire)) {
    atomic_store_explicit(&owner->mailbox_state, RADIO_GAIN_MAILBOX_IDLE, memory_order_release);
    return RADIO_GAIN_CLOSED;
  }
  owner->mailbox_request = *request;
  atomic_store_explicit(&owner->mailbox_state, RADIO_GAIN_MAILBOX_PENDING, memory_order_release);
  return RADIO_GAIN_OK;
}

bool radio_gain_take_result(radio_gain_owner_t *owner, radio_gain_result_t *result)
{
  if (owner == NULL || result == NULL)
    return false;
  int expected = RADIO_GAIN_MAILBOX_DONE;
  if (!atomic_compare_exchange_strong_explicit(&owner->mailbox_state,
                                               &expected,
                                               RADIO_GAIN_MAILBOX_READING,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return false;
  *result = owner->mailbox_result;
  atomic_store_explicit(&owner->mailbox_state, RADIO_GAIN_MAILBOX_IDLE, memory_order_release);
  return true;
}

bool radio_gain_snapshot(const radio_gain_owner_t *owner, radio_gain_result_t *result)
{
  return owner != NULL && result != NULL && radio_gain_snapshot_load(owner, result);
}

bool radio_gain_channels(const radio_gain_owner_t *owner, radio_gain_channel_t *rx, radio_gain_channel_t *tx)
{
  if (owner == NULL || rx == NULL || tx == NULL)
    return false;
  struct radio_gain_owner *mutable_owner = (struct radio_gain_owner *)(void *)owner;
  pthread_mutex_lock(&mutable_owner->channels_mutex);
  const bool valid = mutable_owner->channel_profiles_valid;
  if (valid) {
    *rx = mutable_owner->rx_channel_info;
    *tx = mutable_owner->tx_channel_info;
  }
  pthread_mutex_unlock(&mutable_owner->channels_mutex);
  return valid;
}

bool radio_gain_set_tx_admission(radio_gain_owner_t *owner, bool open)
{
  if (owner == NULL)
    return false;
  if (!open) {
    atomic_store_explicit(&owner->tx_admission_open, false, memory_order_release);
    return true;
  }

  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&owner->tx_settings_inflight,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return false;
  atomic_store_explicit(&owner->tx_admission_open, true, memory_order_release);
  radio_gain_release_tx_gate(owner);
  return true;
}

void radio_gain_note_tx_end(radio_gain_owner_t *owner, int64_t end_ticks)
{
  if (owner == NULL)
    return;
  if (!atomic_load_explicit(&owner->tx_admission_open, memory_order_acquire)) {
    /* A producer that publishes after its coordinator closed admission breaks
     * the handoff to a potentially in-flight setting transaction. Keep all
     * later TX reconfiguration conservatively blocked rather than guessing. */
    atomic_store_explicit(&owner->tx_end_uncertain, true, memory_order_release);
    atomic_store_explicit(&owner->tx_end_known, true, memory_order_release);
    return;
  }
  int64_t observed = atomic_load_explicit(&owner->last_tx_end, memory_order_relaxed);
  for (unsigned int attempt = 0; attempt < RADIO_GAIN_TX_END_CAS_ATTEMPTS; ++attempt) {
    if (end_ticks <= observed) {
      atomic_store_explicit(&owner->tx_end_known, true, memory_order_release);
      return;
    }
    if (atomic_compare_exchange_weak_explicit(&owner->last_tx_end,
                                              &observed,
                                              end_ticks,
                                              memory_order_release,
                                              memory_order_relaxed)) {
      atomic_store_explicit(&owner->tx_end_known, true, memory_order_release);
      return;
    }
  }
  /* A producer never spins indefinitely. A missed maximum could admit a gain
   * change before an unknown future interval finishes, so reject future TX
   * settings conservatively until this owner is destroyed. */
  atomic_store_explicit(&owner->tx_end_uncertain, true, memory_order_release);
  atomic_store_explicit(&owner->tx_end_known, true, memory_order_release);
}

void radio_gain_owner_set_failure_handler(radio_gain_owner_t *owner, void (*handler)(int))
{
  if (owner)
    owner->failure_handler = handler;
}

void radio_gain_owner_report_failure(radio_gain_owner_t *owner, int failure)
{
  if (owner && failure > 0) {
    int expected = 0;
    (void)atomic_compare_exchange_strong_explicit(&owner->terminal_failure,
                                                  &expected,
                                                  failure,
                                                  memory_order_release,
                                                  memory_order_relaxed);
  }
}
