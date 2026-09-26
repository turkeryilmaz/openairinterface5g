/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_gain_device.h"
#include "common_lib.h"
#include "openair1/PHY/impl_defs_top.h"
#include "radio_gain_policy.h"
#include "executables/agc_options.h"
#include "common/utils/LOG/log.h"
#include "common/utils/LOG/flight_recorder.h"
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <time.h>

/* NR initially qualifies one physical radio and one logical RX/TX stream.
 * Retain the binding until process exit so concurrent fatal shutdown cannot
 * free the lookup handle underneath a late producer. */
static struct {
  openair0_device_t *device;
  radio_gain_owner_t *owner;
  radio_gain_sample_history_t *history;
  radio_gain_channel_t rx;
  radio_gain_channel_t tx;
  double rx_peak_refresh_fs;
  /* Profile contents are immutable after startup. Retunes may only retain the
   * same qualified operating range; they never publish a different mapper. */
  radio_tx_profile_t tx_profile;
  radio_tx_relative_config_t tx_relative;
  _Atomic(bool) tx_mapping_valid;
  _Atomic(bool) tx_fault;
  _Atomic(uint64_t) next_relative_gate_ns;
  bool gnb_reference_ready;
  double gnb_sss_dbm;
  int16_t gnb_amplitude;
  _Atomic(bool) closed;
  _Atomic(unsigned int) users;
  _Atomic(uint64_t) next_request;
  _Atomic(unsigned int) read_calls;
  /* Serialized by tx_writer_active; accessed only when recording is enabled. */
  unsigned int tx_level_calls;
  /* One synchronous controller consumes the one-result mailbox. */
  _Atomic(bool) request_consumer;
  /* One atomic phase/floor snapshot: bit 0 is tracking, bit 1 blocks policy
   * submissions until an earlier policy result is recorded, higher bits are
   * the host-monotonic transition time. */
  _Atomic(uint64_t) rx_phase_token;
  /* Fast producer-visible maintenance hint. The owner result itself remains
   * consumed only under request_consumer. */
  _Atomic(bool) policy_result_outstanding;
  /* These fields are accessed only under request_consumer's try-admission. */
  bool policy_request_pending;
  radio_rx_policy_state_t rx_policy;
  /* Raw reads run ahead of decoded references. Order each source separately,
   * while every completed physical action updates both cooldowns. */
  radio_rx_policy_state_t rx_headroom_policy;
  /* Both source-specific filters observe one physical input-referred peak
   * history. request_consumer serializes all accesses. */
  radio_rx_peak_envelope_t rx_peak_envelope;
  radio_rx_reason_t last_reason;
  uint64_t last_decision_log_ns;
  /* The callback interface has no producer reference object. Refuse a second
   * concurrent write rather than close the owner admission below a writer. */
  _Atomic(bool) tx_writer_active;
  /* A retune owns this gate before it closes owner admission. */
  _Atomic(bool) tx_reconfiguration_active;
  /* Host/device queues are not jointly observable through the legacy callback.
   * After the first admission, only a caller-wide quiescence contract can make
   * retune safe; this intermediate binding has no such contract. */
  _Atomic(bool) tx_seen;
  /* An inline fatal callback closes admission but leaves storage to an
   * external coordinator after this callback frame has unwound. */
  _Atomic(bool) shutdown_deferred;
  int (*read)(openair0_device_t *, openair0_timestamp_t *, void **, int, int);
  int (*write)(openair0_device_t *, openair0_timestamp_t, void **, int, int, int);
  int (*stop)(openair0_device_t *);
  void (*end)(openair0_device_t *);
} binding;
static pthread_mutex_t shutdown_mutex = PTHREAD_MUTEX_INITIALIZER;
/* A wrapped legacy callback can invoke the global fatal path synchronously. */
static _Thread_local unsigned int binding_callback_depth;

static bool binding_atomics_lock_free(void)
{
  return atomic_is_lock_free(&binding.closed) && atomic_is_lock_free(&binding.users) && atomic_is_lock_free(&binding.next_request)
         && atomic_is_lock_free(&binding.read_calls) && atomic_is_lock_free(&binding.request_consumer)
         && atomic_is_lock_free(&binding.rx_phase_token) && atomic_is_lock_free(&binding.policy_result_outstanding)
         && atomic_is_lock_free(&binding.tx_writer_active) && atomic_is_lock_free(&binding.tx_reconfiguration_active)
         && atomic_is_lock_free(&binding.tx_seen) && atomic_is_lock_free(&binding.shutdown_deferred)
         && atomic_is_lock_free(&binding.tx_mapping_valid) && atomic_is_lock_free(&binding.tx_fault)
         && atomic_is_lock_free(&binding.next_relative_gate_ns);
}

static bool enter(openair0_device_t *device)
{
  if (device == NULL || device != binding.device || atomic_load_explicit(&binding.closed, memory_order_acquire))
    return false;
  atomic_fetch_add_explicit(&binding.users, 1, memory_order_acq_rel);
  if (atomic_load_explicit(&binding.closed, memory_order_acquire)) {
    atomic_fetch_sub_explicit(&binding.users, 1, memory_order_release);
    return false;
  }
  ++binding_callback_depth;
  return true;
}

static void leave(void)
{
  if (binding_callback_depth != 0)
    --binding_callback_depth;
  atomic_fetch_sub_explicit(&binding.users, 1, memory_order_release);
}

static void wait_for_users(void)
{
  const struct timespec interval = {.tv_nsec = 100000};
  while (atomic_load_explicit(&binding.users, memory_order_acquire) != 0)
    (void)nanosleep(&interval, NULL);
}

static uint64_t monotonic_ns(void)
{
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0)
    return 0;
  return (uint64_t)now.tv_sec * 1000000000ULL + now.tv_nsec;
}

enum {
  RADIO_RX_PHASE_TRACKING_BIT = 1,
  RADIO_RX_PHASE_PENDING_BIT = 2,
  RADIO_RX_PHASE_TIME_SHIFT = 2,
};

static uint64_t rx_phase_token(radio_gain_device_rx_phase_t phase, bool pending, uint64_t started_ns)
{
  if (started_ns > (UINT64_MAX >> RADIO_RX_PHASE_TIME_SHIFT))
    started_ns = UINT64_MAX >> RADIO_RX_PHASE_TIME_SHIFT;
  return (started_ns << RADIO_RX_PHASE_TIME_SHIFT) | (pending ? RADIO_RX_PHASE_PENDING_BIT : 0)
         | (phase == RADIO_GAIN_RX_PHASE_TRACKING ? RADIO_RX_PHASE_TRACKING_BIT : 0);
}

static radio_gain_device_rx_phase_t rx_phase_from_token(uint64_t token)
{
  return token & RADIO_RX_PHASE_TRACKING_BIT ? RADIO_GAIN_RX_PHASE_TRACKING : RADIO_GAIN_RX_PHASE_ACQUISITION;
}

static bool rx_phase_pending(uint64_t token)
{
  return token & RADIO_RX_PHASE_PENDING_BIT;
}

static uint64_t rx_phase_started_ns(uint64_t token)
{
  return token >> RADIO_RX_PHASE_TIME_SHIFT;
}

/* Initial engineering settings. Tracking uses an occupancy-independent serving
 * reference level; peaks refer to sparse raw time-domain samples. Neither is
 * calibrated connector power or a sample-rate ADC protection guarantee. */
static const radio_rx_policy_config_t rx_policy_config = {
    .target_dbfs = -18,
    .deadband_db = 3,
    .maximum_step_db = 3,
    .search_step_db = 3,
    .peak_ceiling_dbfs = -3,
    .peak_release_db_per_second = 3,
    .near_rail_fraction = 0.01,
    .filter_weight = 0.25,
    .minimum_interval_ns = 200000000,
    .maximum_age_ns = 200000000,
};

static int64_t milli_db(double value, bool valid)
{
  return valid && isfinite(value) && fabs(value) < 1e6 ? llround(value * 1000) : INT64_MIN;
}

static void record_result(const radio_gain_result_t *result)
{
  const unsigned int valid = result->rx_gain_valid | (result->tx_gain_valid << 1) | (result->device_time_valid << 2);
  flight_recorder_emit(
      FLIGHT_EVENT_RADIO_GAIN,
      0,
      result->generation,
      milli_db(result->request.gain_db, result->request.request_id != 0 && result->request.operation != RADIO_GAIN_RETUNE),
      milli_db(result->reported_rx_db, result->rx_gain_valid),
      milli_db(result->reported_tx_db, result->tx_gain_valid),
      result->status | ((int64_t)result->request.operation << 8) | ((int64_t)valid << 16));
  flight_recorder_emit(FLIGHT_EVENT_RADIO_GAIN_TIME,
                       0,
                       result->generation,
                       result->request.request_id,
                       result->device_time_valid ? result->begin_device_ticks : INT64_MIN,
                       result->device_time_valid ? result->end_device_ticks : INT64_MIN,
                       valid);
}

static bool continuous_rx_capable(const radio_gain_api_t *api)
{
  const size_t needed = offsetof(radio_gain_api_t, device_ticks) + sizeof(api->device_ticks);
  return api != NULL && api->abi_version == OAI_RADIO_GAIN_ABI && api->struct_size >= needed && api->query != NULL
         && api->set_gain != NULL && api->set_rx_agc != NULL && api->retune != NULL && api->device_ticks != NULL;
}

static int status_to_errno(radio_gain_status_t status)
{
  switch (status) {
    case RADIO_GAIN_OK:
      return 0;
    case RADIO_GAIN_BUSY:
    case RADIO_GAIN_TX_PENDING:
      return -EBUSY;
    case RADIO_GAIN_UNSUPPORTED:
      return -ENOTSUP;
    case RADIO_GAIN_INVALID:
      return -EINVAL;
    case RADIO_GAIN_STALE:
      return -ESTALE;
    case RADIO_GAIN_CLOSED:
      return -ESHUTDOWN;
    case RADIO_GAIN_BACKEND_ERROR:
      return -EIO;
  }
  return -EIO;
}

/* request_consumer serializes every take_result call. A policy completion is
 * therefore drained only by its submitting policy path or a synchronous legacy
 * control path that already owns this token. */
static bool consume_policy_result(uint64_t now_ns)
{
  if (!binding.policy_request_pending)
    return true;

  radio_gain_result_t result;
  if (!radio_gain_take_result(binding.owner, &result))
    return false;
  binding.policy_request_pending = false;
  atomic_store_explicit(&binding.policy_result_outstanding, false, memory_order_release);
  record_result(&result);
  if (result.status == RADIO_GAIN_OK) {
    radio_rx_action_completed(&binding.rx_policy, result.generation, now_ns);
    radio_rx_action_completed(&binding.rx_headroom_policy, result.generation, now_ns);
  }
  return true;
}

static void reset_policy_observations(radio_rx_policy_state_t *state)
{
  state->filter_valid = false;
  state->filtered_linear_power = 0;
  state->generation = 0;
  state->last_observation_ns = 0;
  state->last_observation_valid = false;
  /* Preserve completed-action cooldown across sources and handoff. */
}

/* Finish a phase change only after an earlier policy request has been recorded.
 * A concurrent transition writes a newer token; one CAS attempt leaves that
 * newer transition pending for the next bounded producer/control entry. */
static void finish_rx_phase_transition(void)
{
  const uint64_t token = atomic_load_explicit(&binding.rx_phase_token, memory_order_acquire);
  if (!rx_phase_pending(token) || binding.policy_request_pending)
    return;

  reset_policy_observations(&binding.rx_policy);
  reset_policy_observations(&binding.rx_headroom_policy);
  binding.rx_peak_envelope = (radio_rx_peak_envelope_t){0};
  binding.last_reason = RADIO_RX_HOLD_INVALID;
  binding.last_decision_log_ns = 0;
  uint64_t expected = token;
  (void)atomic_compare_exchange_strong_explicit(&binding.rx_phase_token,
                                                &expected,
                                                token & ~(uint64_t)RADIO_RX_PHASE_PENDING_BIT,
                                                memory_order_release,
                                                memory_order_acquire);
}

static bool policy_allowed_for_phase(const agc_options_t *options, radio_gain_device_rx_phase_t phase, radio_rx_source_t source)
{
  const bool search = source == RADIO_RX_SOURCE_UE_SEARCH;
  if (options->mode == AGC_MODE_OBSERVE)
    return search ? options->rx_acquisition == AGC_RX_ACQUISITION_NEW : options->rx_tracking == AGC_RX_TRACKING_NEW;
  if (options->mode != AGC_MODE_CONTINUOUS)
    return false;
  if (phase == RADIO_GAIN_RX_PHASE_ACQUISITION)
    return options->rx_acquisition == AGC_RX_ACQUISITION_NEW
           && (source == RADIO_RX_SOURCE_UE_SEARCH || source == RADIO_RX_SOURCE_HEADROOM);
  return options->rx_tracking == AGC_RX_TRACKING_NEW && !search;
}

/* This runs only from legacy acquisition/retune control, never an RX/TX
 * producer. It may wait under the existing synchronous owner contract. Before
 * submitting, it drains and records a preceding asynchronous policy completion
 * while retaining the sole result-consumer token. */
static int synchronous_request(radio_gain_request_t *request, radio_gain_result_t *result)
{
  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&binding.request_consumer,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return -EBUSY;

  int status = -ESHUTDOWN;
  const struct timespec interval = {.tv_nsec = 100000};
  while (binding.policy_request_pending && !atomic_load_explicit(&binding.closed, memory_order_acquire)) {
    if (consume_policy_result(monotonic_ns()))
      break;
    (void)nanosleep(&interval, NULL);
  }
  if (binding.policy_request_pending)
    goto done;
  finish_rx_phase_transition();

  request->request_id = atomic_fetch_add_explicit(&binding.next_request, 1, memory_order_relaxed) + 1;
  const radio_gain_status_t submitted = radio_gain_submit(binding.owner, request);
  if (submitted != RADIO_GAIN_OK) {
    status = status_to_errno(submitted);
    goto done;
  }

  while (!atomic_load_explicit(&binding.closed, memory_order_acquire)) {
    if (radio_gain_take_result(binding.owner, result)) {
      record_result(result);
      status = status_to_errno(result->status);
      goto done;
    }
    (void)nanosleep(&interval, NULL);
  }

done:
  atomic_store_explicit(&binding.request_consumer, false, memory_order_release);
  return status;
}

/* Legacy acquisition must drain an older tracking result before it snapshots
 * gain/generation. It holds request_consumer throughout that sequence, so no
 * producer can submit a second policy request between the drain and request. */
static int synchronous_legacy_adjust(double delta_db, double *applied_delta_db, double *reported_gain_db)
{
  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&binding.request_consumer,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire))
    return -EBUSY;

  int status = -ESHUTDOWN;
  const struct timespec interval = {.tv_nsec = 100000};
  while (binding.policy_request_pending && !atomic_load_explicit(&binding.closed, memory_order_acquire)) {
    if (consume_policy_result(monotonic_ns()))
      break;
    (void)nanosleep(&interval, NULL);
  }
  if (binding.policy_request_pending)
    goto done;
  finish_rx_phase_transition();

  radio_gain_result_t before;
  if (!radio_gain_snapshot(binding.owner, &before) || !before.rx_gain_valid) {
    status = -EAGAIN;
    goto done;
  }
  const double selected = fmax(binding.rx.minimum_db, fmin(binding.rx.maximum_db, before.reported_rx_db + delta_db));
  if (selected == before.reported_rx_db) {
    *reported_gain_db = before.reported_rx_db;
    status = 0;
    goto done;
  }
  radio_gain_request_t request = {
      .operation = RADIO_GAIN_SET_RX,
      .generation = before.generation,
      .request_id = atomic_fetch_add_explicit(&binding.next_request, 1, memory_order_relaxed) + 1,
      .gain_db = selected,
  };
  const radio_gain_status_t submitted = radio_gain_submit(binding.owner, &request);
  if (submitted != RADIO_GAIN_OK) {
    status = status_to_errno(submitted);
    goto done;
  }
  radio_gain_result_t after;
  while (!atomic_load_explicit(&binding.closed, memory_order_acquire)) {
    if (radio_gain_take_result(binding.owner, &after)) {
      record_result(&after);
      status = status_to_errno(after.status);
      if (status == 0) {
        *applied_delta_db = after.reported_rx_db - before.reported_rx_db;
        *reported_gain_db = after.reported_rx_db;
      }
      goto done;
    }
    (void)nanosleep(&interval, NULL);
  }

done:
  atomic_store_explicit(&binding.request_consumer, false, memory_order_release);
  return status;
}

int radio_gain_device_adjust_rx(openair0_device_t *device, double delta_db, double *applied_delta_db, double *reported_gain_db)
{
  if (applied_delta_db != NULL)
    *applied_delta_db = 0;
  if (reported_gain_db != NULL)
    *reported_gain_db = NAN;
  const agc_options_t *options = get_agc_options();
  if (reported_gain_db == NULL || applied_delta_db == NULL || !isfinite(delta_db) || !options->rx_actuation
      || options->rx_acquisition != AGC_RX_ACQUISITION_LEGACY || !enter(device))
    return -ENOTSUP;

  const uint64_t phase_token = atomic_load_explicit(&binding.rx_phase_token, memory_order_acquire);
  if (options->mode == AGC_MODE_CONTINUOUS && rx_phase_from_token(phase_token) != RADIO_GAIN_RX_PHASE_ACQUISITION) {
    leave();
    return -EAGAIN;
  }
  const int status = synchronous_legacy_adjust(delta_db, applied_delta_db, reported_gain_db);
  leave();
  return status;
}

void radio_gain_device_set_rx_phase(radio_gain_device_rx_phase_t phase)
{
  if ((phase != RADIO_GAIN_RX_PHASE_ACQUISITION && phase != RADIO_GAIN_RX_PHASE_TRACKING) || binding.device == NULL
      || !enter(binding.device))
    return;

  const uint64_t previous = atomic_load_explicit(&binding.rx_phase_token, memory_order_acquire);
  if (!rx_phase_pending(previous) && rx_phase_from_token(previous) == phase) {
    leave();
    return;
  }
  atomic_store_explicit(&binding.rx_phase_token, rx_phase_token(phase, true, monotonic_ns()), memory_order_release);

  bool expected = false;
  if (atomic_compare_exchange_strong_explicit(&binding.request_consumer,
                                              &expected,
                                              true,
                                              memory_order_acq_rel,
                                              memory_order_acquire)) {
    if (consume_policy_result(monotonic_ns()))
      finish_rx_phase_transition();
    atomic_store_explicit(&binding.request_consumer, false, memory_order_release);
  }
  leave();
}

void radio_gain_device_observe_rx(const radio_gain_sample_context_t *context,
                                  double reference_bin_power,
                                  bool activity_valid,
                                  bool search_failed,
                                  radio_rx_source_t source)
{
  const agc_options_t *options = get_agc_options();
  const bool search = source == RADIO_RX_SOURCE_UE_SEARCH;
  if (context == NULL || !context->present)
    return;
  /* Ordinary raw reads need only a peak/count comparison. A pending result or
   * phase handoff still enters once to record/finish it; this prevents a quiet
   * headroom stream from stranding the one-result mailbox. */
  const uint64_t phase_before = atomic_load_explicit(&binding.rx_phase_token, memory_order_acquire);
  const bool maintenance_pending =
      rx_phase_pending(phase_before) || atomic_load_explicit(&binding.policy_result_outstanding, memory_order_acquire);
  if (source == RADIO_RX_SOURCE_HEADROOM && !maintenance_pending
      && (!context->level_valid
          || (context->peak_component_fs < binding.rx_peak_refresh_fs
              && context->near_rail_components < rx_policy_config.near_rail_fraction * context->sampled_components)))
    return;
  if (!enter(binding.device))
    return;
  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&binding.request_consumer,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    leave();
    return;
  }
  const uint64_t now = monotonic_ns();
  if (!consume_policy_result(now))
    goto done;
  finish_rx_phase_transition();
  const uint64_t phase_token = atomic_load_explicit(&binding.rx_phase_token, memory_order_acquire);
  const radio_gain_device_rx_phase_t phase = rx_phase_from_token(phase_token);
  if (rx_phase_pending(phase_token) || !policy_allowed_for_phase(options, phase, source)
      || (options->mode == AGC_MODE_CONTINUOUS && context->observation_ns < rx_phase_started_ns(phase_token)))
    goto done;
  radio_gain_result_t current;
  if (!radio_gain_snapshot(binding.owner, &current) || !current.rx_gain_valid)
    goto done;
  const double scale = binding.rx.component_full_scale;
  const double power = activity_valid ? reference_bin_power / scale / scale : context->mean_power_fs;
  radio_rx_observation_t observation = {
      .generation = context->generation,
      .observation_ns = context->observation_ns,
      .now_ns = now,
      .reported_gain_db = context->rx_gain_db,
      .mean_power_dbfs = power > 0 ? 10 * log10(power) : -200,
      .peak_component_dbfs = context->peak_component_fs > 0 ? 20 * log10(context->peak_component_fs) : -200,
      .sampled_components = context->sampled_components,
      .near_rail_components = context->near_rail_components,
      .power_valid = context->level_valid && isfinite(power) && power >= 0,
      .gain_valid = context->valid && context->generation == current.generation && now != 0,
      .settled = context->valid,
      .activity_valid = activity_valid && reference_bin_power > 0,
      .search_failed = search && search_failed,
  };
  radio_rx_policy_state_t *state = source == RADIO_RX_SOURCE_HEADROOM ? &binding.rx_headroom_policy : &binding.rx_policy;
  const radio_rx_decision_t decision =
      radio_rx_decide(&rx_policy_config, state, &binding.rx_peak_envelope, &binding.rx, &observation);
  bool submitted = false;
  if (decision.change && options->rx_actuation) {
    radio_gain_request_t request = {
        .operation = RADIO_GAIN_SET_RX,
        .generation = current.generation,
        .request_id = atomic_fetch_add_explicit(&binding.next_request, 1, memory_order_relaxed) + 1,
        .gain_db = decision.gain_db,
    };
    submitted = radio_gain_submit(binding.owner, &request) == RADIO_GAIN_OK;
    binding.policy_request_pending = submitted;
    if (submitted)
      atomic_store_explicit(&binding.policy_result_outstanding, true, memory_order_release);
  }
  if (decision.change || decision.reason != binding.last_reason || now - binding.last_decision_log_ns >= 1000000000ULL) {
    const int64_t flags = decision.reason | ((int64_t)decision.change << 8) | ((int64_t)submitted << 9)
                          | ((int64_t)observation.activity_valid << 10) | ((int64_t)observation.search_failed << 11) | (1LL << 12)
                          | ((int64_t)(phase == RADIO_GAIN_RX_PHASE_TRACKING) << 13);
    flight_recorder_emit(FLIGHT_EVENT_RADIO_RX_DECISION,
                         source,
                         context->generation,
                         context->end_sample,
                         milli_db(observation.mean_power_dbfs, observation.power_valid),
                         milli_db(decision.gain_db, observation.gain_valid),
                         flags);
    flight_recorder_emit(
        FLIGHT_EVENT_RADIO_RX_DECISION_INPUT,
        source,
        context->generation,
        context->end_sample,
        milli_db(observation.reported_gain_db, observation.gain_valid),
        milli_db(observation.peak_component_dbfs, context->level_valid),
        milli_db(decision.error_db, decision.reason == RADIO_RX_TRACK_LEVEL || decision.reason == RADIO_RX_HOLD_DEADBAND));
    if (decision.peak_bound_valid)
      flight_recorder_emit(FLIGHT_EVENT_RADIO_RX_PEAK_ENVELOPE,
                           source,
                           context->generation,
                           context->end_sample,
                           milli_db(decision.peak_bound_dbfs, true),
                           milli_db(observation.reported_gain_db, observation.gain_valid),
                           milli_db(rx_policy_config.peak_release_db_per_second, true));
    binding.last_reason = decision.reason;
    binding.last_decision_log_ns = now;
  }
done:
  atomic_store_explicit(&binding.request_consumer, false, memory_order_release);
  leave();
}

static int observe_read(openair0_device_t *device, openair0_timestamp_t *timestamp, void **buffers, int count, int antennas)
{
  if (!enter(device))
    return -ESHUTDOWN;

  radio_gain_result_t before = {0}, after = {0};
  const bool before_valid = radio_gain_snapshot(binding.owner, &before);
  const int received = binding.read ? binding.read(device, timestamp, buffers, count, antennas) : -ENOTSUP;
  const bool after_valid = radio_gain_snapshot(binding.owner, &after);
  if (received > 0 && timestamp != NULL && *timestamp <= INT64_MAX - received) {
    /* The initial settings precede streaming. Following changes, exclude a
     * configurable engineering guard after the device-call bracket. This is
     * not a sample-exact analog settling measurement or an RF calibration. */
    const unsigned int settle_us = get_agc_options()->rx_settle_us ? get_agc_options()->rx_settle_us : AGC_RX_SETTLE_DEFAULT_US;
    const double guard = ceil(binding.rx.sample_rate_hz * settle_us / 1000000.0);
    int64_t settled_after = INT64_MIN;
    bool boundary_valid = after_valid && after.generation == 1;
    if (!boundary_valid && after_valid && after.status == RADIO_GAIN_OK && after.device_time_valid
        && after.end_device_ticks >= after.begin_device_ticks && isfinite(guard) && guard >= 0 && guard < INT64_MAX
        && after.end_device_ticks <= INT64_MAX - (int64_t)guard) {
      settled_after = after.end_device_ticks + (int64_t)guard;
      boundary_valid = true;
    }
    radio_gain_sample_context_t context = radio_gain_sample_context(before_valid ? &before : NULL,
                                                                    after_valid ? &after : NULL,
                                                                    *timestamp,
                                                                    *timestamp + received,
                                                                    boundary_valid,
                                                                    settled_after);
    if (antennas == 1 && buffers && buffers[0]) {
      const int16_t *iq = buffers[0];
      const unsigned int samples = received < 64 ? (unsigned int)received : 64;
      uint64_t energy = 0;
      uint32_t peak = 0, near_rail = 0;
      const uint32_t rail = binding.rx.component_full_scale;
      for (unsigned int i = 0; i < samples; ++i) {
        const size_t index = (uint64_t)i * received / samples;
        for (unsigned int component = 0; component < 2; ++component) {
          const int32_t value = iq[2 * index + component];
          const uint32_t magnitude = value < 0 ? -value : value;
          energy += (int64_t)value * value;
          if (magnitude > peak)
            peak = magnitude;
          near_rail += (uint64_t)magnitude * 100 >= (uint64_t)rail * 98;
        }
      }
      context.level_valid = true;
      context.mean_power_fs = (double)energy / samples / rail / rail;
      context.peak_component_fs = (double)peak / rail;
      context.sampled_components = 2 * samples;
      context.near_rail_components = near_rail;
      context.observation_ns = monotonic_ns();
      if (((atomic_fetch_add_explicit(&binding.read_calls, 1, memory_order_relaxed) + 1) & 31U) == 0 && flight_recorder_enabled()) {
        flight_recorder_emit(FLIGHT_EVENT_RADIO_RX_LEVEL,
                             0,
                             context.valid ? (int64_t)context.generation : -(int64_t)context.generation,
                             *timestamp,
                             milli_db(energy ? 10 * log10(context.mean_power_fs) : 0, energy != 0),
                             milli_db(peak ? 20 * log10(context.peak_component_fs) : 0, peak != 0),
                             ((uint64_t)near_rail << 32) | (2 * samples));
      }
    }
    radio_gain_sample_publish(binding.history, &context);
    radio_gain_device_observe_rx(&context, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  }
  leave();
  return received;
}

/* The wrapper provides an explicit bounded producer handoff to the owner. A
 * second concurrent caller gets a truthful -EBUSY result before its samples
 * reach the legacy device. The ordinary NR path has one TX producer; lifting
 * this restriction needs a producer reference/count contract, not a spinlock. */
static bool relative_tx_selected(void)
{
  return radio_gain_device_tx_selected() && get_agc_options()->tx_power_mode == AGC_TX_POWER_RELATIVE;
}

bool radio_gain_device_tx_relative_actuating(void)
{
  return relative_tx_selected() && radio_gain_device_tx_actuating();
}

bool radio_gain_device_relative_tx_bounds(int *minimum, int *maximum)
{
  if (!relative_tx_selected() || !minimum || !maximum || !atomic_load_explicit(&binding.tx_mapping_valid, memory_order_acquire)
      || atomic_load_explicit(&binding.tx_fault, memory_order_acquire))
    return false;
  *minimum = binding.tx_relative.nominal_min;
  *maximum = binding.tx_relative.nominal_max;
  return true;
}

/* A relative mapping has no RF calibration epoch. A retune before the first
 * TX may retain it only if the connector, converter and fixed gain still agree. */
static bool relative_operating_point_matches(const radio_gain_channel_t *tx)
{
  return tx->component_full_scale == binding.tx.component_full_scale && isfinite(tx->reported_db)
         && fabs(tx->reported_db - binding.tx.reported_db) < 1e-6 && tx->sample_rate_hz == binding.tx.sample_rate_hz
         && tx->bandwidth_hz == binding.tx.bandwidth_hz && !strcmp(tx->identity, binding.tx.identity)
         && !strcmp(tx->antenna, binding.tx.antenna);
}

static void erase_relative_tx(c16_t *samples, uint32_t count, int frame, int slot, unsigned channel, int reason)
{
  /* A bounded whole-occasion erasure preserves phase/channel relationships in
   * every emitted occasion. It is not clipping, an accepted power mapping, or
   * a reason to restart the radio. The zero buffer still advances device time. */
  memset(samples, 0, (size_t)count * sizeof(*samples));
  flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_RELATIVE_ERASURE, channel, (int64_t)frame * 1000 + slot, reason, count, 0, 0);
}

static int observe_write(openair0_device_t *device,
                         openair0_timestamp_t timestamp,
                         void **buffers,
                         int count,
                         int antennas,
                         int flags)
{
  if (!enter(device))
    return -ESHUTDOWN;

  if (radio_gain_device_tx_actuating()
      && (atomic_load_explicit(&binding.tx_fault, memory_order_acquire)
          || !atomic_load_explicit(&binding.tx_mapping_valid, memory_order_acquire))) {
    leave();
    return -ERANGE;
  }

  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&binding.tx_writer_active,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    leave();
    return -EBUSY;
  }
  if (atomic_load_explicit(&binding.tx_reconfiguration_active, memory_order_acquire)) {
    atomic_store_explicit(&binding.tx_writer_active, false, memory_order_release);
    leave();
    return -EBUSY;
  }

  if (radio_gain_device_tx_actuating()) {
    const bool relative = relative_tx_selected();
    const uint32_t full_scale = relative ? binding.tx_relative.component_full_scale : binding.tx_profile.component_full_scale;
    const double peak_limit = relative ? binding.tx_relative.peak_limit_fs : binding.tx_profile.peak_limit_fs;
    const radio_tx_sample_level_t level =
        antennas == 1 && buffers && count > 0 ? radio_tx_sample_level(buffers[0], count, full_scale) : (radio_tx_sample_level_t){0};
    if (!level.valid || level.over_range_components || level.peak_component > peak_limit * full_scale) {
      if (relative && level.valid)
        erase_relative_tx(buffers[0], count, -1, -1, 0, RADIO_TX_POWER_HEADROOM);
      else {
        radio_gain_device_reject_tx(-1, -1, 0, RADIO_TX_REJECT_POWER_LIMIT);
        atomic_store_explicit(&binding.tx_writer_active, false, memory_order_release);
        leave();
        return -ERANGE;
      }
    }
  }

  int sent = binding.write == NULL ? -ENOTSUP : -EBUSY;
  if (binding.write != NULL && radio_gain_set_tx_admission(binding.owner, true)) {
    /* This is before the legacy callback: even a callback error cannot prove
     * that a host/device queue did not retain an interval. */
    atomic_store_explicit(&binding.tx_seen, true, memory_order_release);
    radio_tx_sample_level_t level = {0};
    radio_gain_result_t snapshot = {0};
    bool snapshot_valid = false;
    if (flight_recorder_enabled() && ((++binding.tx_level_calls & 63U) == 1U) && antennas == 1 && buffers != NULL && count > 0
        && binding.tx.component_full_scale >= 1 && binding.tx.component_full_scale <= 32768) {
      level = radio_tx_sample_level(buffers[0], count, (uint32_t)binding.tx.component_full_scale);
      if (level.valid)
        snapshot_valid = radio_gain_snapshot(binding.owner, &snapshot) && snapshot.tx_gain_valid;
    }
    sent = binding.write(device, timestamp, buffers, count, antennas, flags);
    if (level.valid) {
      flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_LEVEL,
                           0,
                           timestamp,
                           level.sample_count,
                           level.component_full_scale,
                           level.sum_squared_components,
                           ((uint64_t)level.peak_component << 32) | level.over_range_components);
      flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_LEVEL_STATE,
                           0,
                           timestamp,
                           count,
                           sent,
                           milli_db(snapshot.reported_tx_db, snapshot_valid),
                           snapshot_valid ? snapshot.generation : INT64_MIN);
    }
    if (sent > 0 && timestamp >= 0 && sent <= INT64_MAX - timestamp)
      radio_gain_note_tx_end(binding.owner, timestamp + sent);
    (void)radio_gain_set_tx_admission(binding.owner, false);
  }

  atomic_store_explicit(&binding.tx_writer_active, false, memory_order_release);
  leave();
  return sent;
}

/* The saved legacy set-frequency callback is deliberately never invoked in
 * managed modes. The local request copies all mutable configuration fields
 * before it enters the owner mailbox. This intermediate binding supports
 * retune only before the caller has admitted any TX; a caller racing its first
 * TX gets -EBUSY and must preserve its prior radio/configuration state. */
static int owner_set_frequency(openair0_device_t *device, openair0_config_t *config)
{
  if (config == NULL)
    return -EINVAL;
  if (!enter(device))
    return -ESHUTDOWN;

  bool expected = false;
  if (!atomic_compare_exchange_strong_explicit(&binding.tx_reconfiguration_active,
                                               &expected,
                                               true,
                                               memory_order_acq_rel,
                                               memory_order_acquire)) {
    leave();
    return -EBUSY;
  }

  int status = -EBUSY;
  if (!atomic_load_explicit(&binding.tx_seen, memory_order_acquire)
      && !atomic_load_explicit(&binding.tx_writer_active, memory_order_acquire)
      && radio_gain_set_tx_admission(binding.owner, false)) {
    radio_gain_result_t snapshot;
    if (radio_gain_snapshot(binding.owner, &snapshot)) {
      radio_gain_request_t request = {
          .operation = RADIO_GAIN_RETUNE,
          .generation = snapshot.generation,
          .rx_frequency_hz = config->rx_freq[0],
          .tx_frequency_hz = config->tx_freq[0],
          .tune_offset_hz = config->tune_offset,
      };
      atomic_store_explicit(&binding.tx_mapping_valid, false, memory_order_release);
      status = synchronous_request(&request, &snapshot);
      radio_gain_channel_t rx, tx;
      const bool valid =
          status == 0 && radio_gain_channels(binding.owner, &rx, &tx)
          && (relative_tx_selected() ? relative_operating_point_matches(&tx) : radio_tx_profile_matches(&binding.tx_profile, &tx));
      atomic_store_explicit(&binding.tx_mapping_valid, valid, memory_order_release);
      if (radio_gain_device_tx_actuating() && !valid) {
        atomic_store_explicit(&binding.tx_fault, true, memory_order_release);
        radio_gain_owner_report_failure(binding.owner, RADIO_TX_REJECT_PROFILE);
        status = -ERANGE;
      }
    } else {
      status = -EAGAIN;
    }
  }
  atomic_store_explicit(&binding.tx_reconfiguration_active, false, memory_order_release);
  leave();
  return status;
}

radio_gain_sample_context_t radio_gain_device_samples(openair0_device_t *device, int64_t first, int64_t end)
{
  radio_gain_sample_context_t result = {.present = device == binding.device, .first_sample = first, .end_sample = end};
  if (enter(device)) {
    result = radio_gain_sample_lookup(binding.history, first, end);
    leave();
  }
  return result;
}

static int reject_combined_gain(openair0_device_t *device, openair0_config_t *config)
{
  (void)device;
  (void)config;
  LOG_E(HW, "Combined gain setter is unavailable while the NR gain owner is active; use a direction-specific request\n");
  return -ENOTSUP;
}

static int owner_stop(openair0_device_t *device)
{
  if (device != binding.device)
    return -ENOTSUP;
  if (binding_callback_depth != 0) {
    /* A legacy callback can call exit_function(), which calls stop/end inline.
     * Do not lock, wait for ourselves, or release owner/backend storage under
     * that callback. A fatal non-return path intentionally leaks to process
     * exit; a normal returning path is completed by a later coordinator. */
    atomic_store_explicit(&binding.closed, true, memory_order_release);
    radio_gain_owner_close(binding.owner);
    atomic_store_explicit(&binding.shutdown_deferred, true, memory_order_release);
    return 0;
  }

  pthread_mutex_lock(&shutdown_mutex);
  atomic_store_explicit(&binding.closed, true, memory_order_release);
  radio_gain_owner_close(binding.owner);
  /* Stop backend activity only after callbacks that entered before close have
   * returned. This is non-real-time shutdown coordination. */
  wait_for_users();
  const int result = binding.owner && binding.stop ? binding.stop(device) : 0;
  pthread_mutex_unlock(&shutdown_mutex);
  return result;
}

static void owner_end(openair0_device_t *device)
{
  if (device != binding.device)
    return;
  if (binding_callback_depth != 0) {
    atomic_store_explicit(&binding.closed, true, memory_order_release);
    radio_gain_owner_close(binding.owner);
    atomic_store_explicit(&binding.shutdown_deferred, true, memory_order_release);
    return;
  }

  pthread_mutex_lock(&shutdown_mutex);
  atomic_store_explicit(&binding.closed, true, memory_order_release);
  radio_gain_owner_close(binding.owner);
  wait_for_users();
  if (binding.owner) {
    radio_gain_owner_destroy(binding.owner);
    binding.owner = NULL;
    radio_gain_sample_history_destroy(binding.history);
    binding.history = NULL;
    if (binding.end)
      binding.end(device);
  }
  pthread_mutex_unlock(&shutdown_mutex);
}

static void terminal_tx_failure(int failure)
{
  /* Runs on the normal-priority settings worker, never a PHY producer. The
   * existing ITTI signal path performs coordinated process/radio teardown. */
  LOG_E(HW, "[AGC] TX admission closed after power/profile failure %d; requesting coordinated shutdown\n", failure);
  (void)kill(getpid(), SIGTERM);
}

int radio_gain_device_attach(openair0_device_t *device, openair0_config_t *config, const radio_gain_api_t *api)
{
  const agc_options_t *options = get_agc_options();
  if (options->mode == AGC_MODE_OFF)
    return 0;
  if (device == NULL || config == NULL || !binding_atomics_lock_free()) {
    LOG_E(HW, "Managed AGC requires lock-free producer atomics and initialized radio/configuration storage\n");
    return -ENOTSUP;
  }
  if (options->mode == AGC_MODE_CONTINUOUS && options->rx_actuation && !continuous_rx_capable(api)) {
    LOG_E(HW, "Managed continuous RX requires versioned query, RX set-gain, hardware-AGC disable, and device-tick capabilities.\n");
    return -ENOTSUP;
  }
  if (binding.device || config->rx_num_channels != 1 || config->tx_num_channels != 1 || !api) {
    LOG_E(HW, "Requested AGC mode requires one radio, one RX/TX stream, and the versioned gain capability interface\n");
    return -ENOTSUP;
  }
  if (device->trx_read_func2 != NULL || device->trx_write_func2 != NULL) {
    LOG_E(HW, "Managed AGC does not support alternate radio callbacks that bypass the owned RX/TX handoff\n");
    return -ENOTSUP;
  }

  radio_gain_owner_t *owner = radio_gain_owner_create(api, device, 0, 0, options->rx_actuation);
  radio_gain_channel_t rx, tx;
  if (!owner || !radio_gain_channels(owner, &rx, &tx)) {
    radio_gain_owner_destroy(owner);
    LOG_E(HW, "Cannot establish verified direction-specific radio gain ownership\n");
    return -ENOTSUP;
  }
  LOG_I(HW,
        "[AGC] RX %s antenna=%s range=%.3f..%.3f step=%.3f readback=%.3f dB converter_fs=%u; "
        "TX %s antenna=%s range=%.3f..%.3f step=%.3f readback=%.3f dB power_reference=%s\n",
        rx.identity,
        rx.antenna,
        rx.minimum_db,
        rx.maximum_db,
        rx.step_db,
        rx.reported_db,
        rx.component_full_scale,
        tx.identity,
        tx.antenna,
        tx.minimum_db,
        tx.maximum_db,
        tx.step_db,
        tx.reported_db,
        tx.power_reference_valid ? "available-unqualified" : "unavailable");
  radio_tx_relative_config_t relative_config = {0};
  const bool relative = relative_tx_selected();
  const bool tx_mapping_valid =
      relative ? isfinite(tx.reported_db) && radio_tx_relative_configure(tx.component_full_scale, AMP, 6.0, &relative_config)
               : radio_tx_profile_matches(&options->tx_profile, &tx);
  if (options->tx_actuation && !tx_mapping_valid) {
    LOG_E(HW,
          "Managed TX mapping is incompatible with the actual connector, gain or converter (absolute mode also requires a matching "
          "profile).\n");
    radio_gain_owner_destroy(owner);
    return -ENOTSUP;
  }
  if (relative)
    LOG_I(HW,
          "[AGC] TX relative: fixed gain=%.3f dB, reference=%.3f dBFS at nominal %.0f, nominal range=%d..%d; "
          "RF output power is uncalibrated\n",
          tx.reported_db,
          relative_config.reference_dbfs,
          relative_config.nominal_reference,
          relative_config.nominal_min,
          relative_config.nominal_max);
  else if (options->tx_policy == AGC_TX_POLICY_MANAGED)
    LOG_I(HW,
          "[AGC] TX profile=%s valid=%s evidence=%s reference=%.3f dBm uncertainty=%.3f dB range=%.3f..%.3f dBm; analog gain held "
          "fixed\n",
          options->tx_profile.id,
          tx_mapping_valid ? "yes" : "no",
          options->tx_profile.provenance,
          options->tx_profile.power.reference_dbm,
          options->tx_profile.power.uncertainty_db,
          options->tx_profile.power.minimum_dbm,
          options->tx_profile.power.maximum_dbm);

  radio_gain_sample_history_t *history = radio_gain_sample_history_create();
  if (!history || rx.component_full_scale == 0) {
    radio_gain_sample_history_destroy(history);
    radio_gain_owner_destroy(owner);
    return -ENOMEM;
  }

  /* Initial adoption precedes all RX processing. Keep configuration bookkeeping
   * consistent with the value actually accepted by the device. */
  config->rx_gain[0] = rx.reported_db + config->rx_gain_offset[0];
  atomic_store_explicit(&binding.closed, false, memory_order_relaxed);
  atomic_store_explicit(&binding.users, 0, memory_order_relaxed);
  atomic_store_explicit(&binding.next_request, 0, memory_order_relaxed);
  atomic_store_explicit(&binding.read_calls, 0, memory_order_relaxed);
  binding.tx_level_calls = 0;
  atomic_store_explicit(&binding.request_consumer, false, memory_order_relaxed);
  atomic_store_explicit(&binding.policy_result_outstanding, false, memory_order_relaxed);
  const radio_gain_device_rx_phase_t initial_phase =
      options->role == AGC_ROLE_GNB ? RADIO_GAIN_RX_PHASE_TRACKING : RADIO_GAIN_RX_PHASE_ACQUISITION;
  atomic_store_explicit(&binding.rx_phase_token, rx_phase_token(initial_phase, false, monotonic_ns()), memory_order_relaxed);
  atomic_store_explicit(&binding.tx_writer_active, false, memory_order_relaxed);
  atomic_store_explicit(&binding.tx_reconfiguration_active, false, memory_order_relaxed);
  atomic_store_explicit(&binding.tx_seen, false, memory_order_relaxed);
  atomic_store_explicit(&binding.shutdown_deferred, false, memory_order_relaxed);
  binding.policy_request_pending = false;
  binding.rx_policy = (radio_rx_policy_state_t){0};
  binding.rx_headroom_policy = (radio_rx_policy_state_t){0};
  binding.rx_peak_envelope = (radio_rx_peak_envelope_t){0};
  binding.last_reason = RADIO_RX_HOLD_INVALID;
  binding.last_decision_log_ns = 0;
  binding.device = device;
  binding.owner = owner;
  radio_gain_owner_set_failure_handler(owner, terminal_tx_failure);
  binding.history = history;
  binding.rx = rx;
  binding.tx = tx;
  binding.tx_profile = options->tx_profile;
  binding.tx_relative = relative_config;
  atomic_store_explicit(&binding.tx_mapping_valid, tx_mapping_valid, memory_order_release);
  atomic_store_explicit(&binding.tx_fault, false, memory_order_release);
  atomic_store_explicit(&binding.next_relative_gate_ns, 0, memory_order_relaxed);
  binding.rx_peak_refresh_fs = pow(10, (rx_policy_config.peak_ceiling_dbfs - rx_policy_config.deadband_db) / 20);
  binding.read = device->trx_read_func;
  binding.write = device->trx_write_func;
  binding.stop = device->trx_stop_func;
  binding.end = device->trx_end_func;
  device->trx_read_func = observe_read;
  device->trx_write_func = observe_write;
  device->trx_set_freq_func = owner_set_frequency;
  device->trx_stop_func = owner_stop;
  device->trx_end_func = owner_end;
  device->trx_set_gains_func = reject_combined_gain;

  if (relative)
    flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_RELATIVE_CONFIG,
                         options->role,
                         tx.component_full_scale,
                         milli_db(relative_config.reference_dbfs, tx_mapping_valid),
                         relative_config.nominal_min,
                         relative_config.nominal_max,
                         milli_db(tx.reported_db, true));
  radio_gain_result_t initial;
  if (radio_gain_snapshot(owner, &initial))
    record_result(&initial);
  return 0;
}

bool radio_gain_device_tx_selected(void)
{
  return get_agc_options()->tx_policy == AGC_TX_POLICY_MANAGED;
}

bool radio_gain_device_tx_actuating(void)
{
  return get_agc_options()->tx_actuation;
}

void radio_gain_device_reject_tx(int frame, int slot, unsigned channel, int reason)
{
  if (radio_gain_device_tx_actuating()) {
    atomic_store_explicit(&binding.tx_fault, true, memory_order_release);
    if (enter(binding.device)) {
      radio_gain_owner_report_failure(binding.owner, reason);
      leave();
    }
  }
  if (flight_recorder_enabled())
    flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_REJECT,
                         0,
                         (int64_t)frame * 1000 + slot,
                         channel,
                         reason,
                         radio_gain_device_tx_actuating(),
                         0);
}

bool radio_gain_device_tx_cancelled(int result)
{
  return (result == -ERANGE || result == -ESHUTDOWN) && radio_gain_device_tx_actuating()
         && atomic_load_explicit(&binding.tx_fault, memory_order_acquire);
}

bool radio_gain_device_apply_tx(c16_t *samples, uint32_t count, double requested_dbm, int frame, int slot, unsigned channel)
{
  if (!radio_gain_device_tx_selected())
    return true;
  const bool apply = radio_gain_device_tx_actuating();
  if (!enter(binding.device))
    return !apply;
  const bool valid = atomic_load_explicit(&binding.tx_mapping_valid, memory_order_acquire)
                     && !atomic_load_explicit(&binding.tx_fault, memory_order_acquire);
  const radio_tx_profile_t *p = &binding.tx_profile;
  const bool relative = relative_tx_selected();
  const radio_tx_power_result_t result =
      relative ? radio_tx_apply_relative_power(samples, count, valid ? &binding.tx_relative : NULL, requested_dbm, apply)
               : radio_tx_apply_power(samples,
                                      count,
                                      p->component_full_scale,
                                      valid ? &p->power : NULL,
                                      requested_dbm,
                                      p->power.maximum_dbm,
                                      p->peak_limit_fs,
                                      p->maximum_quantization_error_db,
                                      p->maximum_quantization_evm,
                                      apply);
  /* Normal nominal saturation was already decided in MAC. A valid buffer
   * rejected by exact relative preflight is an explicitly logged erasure. */
  const bool erased = relative && apply && valid && samples && count > 0 && count <= RADIO_TX_POWER_MAX_SAMPLES
                      && result.status != RADIO_TX_POWER_OK;
  if (erased)
    erase_relative_tx(samples, count, frame, slot, channel, result.status);
  if (apply && result.status != RADIO_TX_POWER_OK && !erased) {
    atomic_store_explicit(&binding.tx_fault, true, memory_order_release);
    radio_gain_owner_report_failure(binding.owner, 100 + result.status);
  }
  if (flight_recorder_enabled()) {
    if (relative) {
      flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_RELATIVE_POWER,
                           channel,
                           (int64_t)frame * 1000 + slot,
                           result.status | ((int64_t)result.applied << 8),
                           milli_db(requested_dbm, isfinite(requested_dbm)),
                           milli_db(result.requested_power_dbfs, isfinite(result.requested_power_dbfs)),
                           milli_db(result.realized_power_dbfs, result.status == RADIO_TX_POWER_OK));
      flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_RELATIVE_QUALITY,
                           channel,
                           (int64_t)frame * 1000 + slot,
                           result.sample_count,
                           milli_db(result.quantization_error_db, result.status == RADIO_TX_POWER_OK),
                           result.status == RADIO_TX_POWER_OK ? llround(result.quantization_evm * 1e9) : INT64_MIN,
                           result.status == RADIO_TX_POWER_OK ? llround(result.mapping.amplitude_scale * 1073741824.0) : INT64_MIN);
    } else {
      flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_POWER,
                           channel,
                           (int64_t)frame * 1000 + slot,
                           result.status | ((int64_t)result.applied << 8),
                           milli_db(requested_dbm, true),
                           milli_db(result.estimated_output_dbm, result.status == RADIO_TX_POWER_OK),
                           result.status == RADIO_TX_POWER_OK ? llround(result.mapping.amplitude_scale * 1073741824.0) : INT64_MIN);
    }
    flight_recorder_emit(relative ? FLIGHT_EVENT_RADIO_TX_RELATIVE_SAMPLES : FLIGHT_EVENT_RADIO_TX_POWER_SAMPLES,
                         channel,
                         (int64_t)frame * 1000 + slot,
                         result.sample_count,
                         result.input_energy,
                         result.status == RADIO_TX_POWER_OK ? (int64_t)result.output_energy : INT64_MIN,
                         ((uint64_t)result.input_peak_component << 32) | result.output_peak_component);
    if (!relative)
      flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_POWER_QUALITY,
                           channel,
                           (int64_t)frame * 1000 + slot,
                           milli_db(result.quantization_error_db, result.status == RADIO_TX_POWER_OK),
                           result.status == RADIO_TX_POWER_OK ? llround(result.quantization_evm * 1000000000.0) : INT64_MIN,
                           milli_db(p->power.uncertainty_db, valid),
                           p->component_full_scale);
  }
  leave();
  return !apply || result.status == RADIO_TX_POWER_OK || erased;
}

bool radio_gain_device_validate_ue_power_limit(int p_max, int p_max_alt, int frame, int slot)
{
  if (!radio_gain_device_tx_actuating())
    return true;
  /* A latched first fault has already closed TX and requested shutdown. Do
   * not misreport every subsequent slot as a new network power-limit error. */
  if (atomic_load_explicit(&binding.tx_fault, memory_order_acquire))
    return false;
  /* Existing NR MAC supports power class 3 here. Reuse its network p-Max
   * ceiling; do not invent a lower PHY-only cap while advertising more PHR.
   * The alternate FR1 limit is not yet consumed by nr_get_Pcmax. */
  const double network_ceiling = p_max == INT_MIN ? 23.0 : fmin(23.0, p_max);
  if (relative_tx_selected()) {
    int minimum = 0, maximum = 0;
    const bool ready = radio_gain_device_relative_tx_bounds(&minimum, &maximum);
    const bool allowed = p_max_alt == INT_MIN && ready && network_ceiling >= minimum;
    /* The scheduler may never run in slot zero of a TDD frame. Throttle on
     * actual blocked calls, with one non-waiting publication attempt. */
    if (!allowed && flight_recorder_enabled()) {
      const uint64_t now = monotonic_ns();
      uint64_t next = atomic_load_explicit(&binding.next_relative_gate_ns, memory_order_relaxed);
      if (now && now >= next && now <= UINT64_MAX - 1000000000ULL
          && atomic_compare_exchange_strong_explicit(&binding.next_relative_gate_ns,
                                                     &next,
                                                     now + 1000000000ULL,
                                                     memory_order_relaxed,
                                                     memory_order_relaxed))
        flight_recorder_emit(FLIGHT_EVENT_RADIO_TX_RELATIVE_GATE,
                             0,
                             (int64_t)frame * 1000 + slot,
                             p_max == INT_MIN ? INT64_MIN : p_max,
                             p_max_alt == INT_MIN ? INT64_MIN : p_max_alt,
                             ready ? minimum : INT64_MIN,
                             ready ? maximum : INT64_MIN);
    }
    return allowed;
  }
  if (p_max_alt != INT_MIN || !atomic_load_explicit(&binding.tx_mapping_valid, memory_order_acquire)
      || network_ceiling > binding.tx_profile.power.maximum_dbm) {
    radio_gain_device_reject_tx(frame, slot, 0, RADIO_TX_REJECT_POWER_LIMIT);
    return false;
  }
  return true;
}

bool radio_gain_device_configure_gnb_tx(double requested_sss_dbm, uint32_t fft_size, int16_t *amplitude)
{
  if (!radio_gain_device_tx_selected())
    return true;
  if (relative_tx_selected()) {
    const bool valid =
        amplitude && *amplitude > 0 && fft_size > 0 && atomic_load_explicit(&binding.tx_mapping_valid, memory_order_acquire);
    if (!valid)
      return !radio_gain_device_tx_actuating();
    const int16_t candidate = lround(*amplitude * pow(10.0, -binding.tx_relative.backoff_db / 20.0));
    LOG_I(HW,
          "[AGC] gNB relative TX: common amplitude %d -> %d, fixed %.1f dB digital backoff; SSS setting %.3f is nominal\n",
          *amplitude,
          candidate,
          binding.tx_relative.backoff_db,
          requested_sss_dbm);
    if (radio_gain_device_tx_actuating()) {
      *amplitude = candidate;
      binding.gnb_sss_dbm = requested_sss_dbm;
      binding.gnb_amplitude = candidate;
      binding.gnb_reference_ready = true;
    }
    return true;
  }
  int16_t candidate = 0;
  double estimated = NAN;
  const bool valid = atomic_load_explicit(&binding.tx_mapping_valid, memory_order_acquire)
                     && radio_tx_sss_amplitude(&binding.tx_profile, fft_size, requested_sss_dbm, &candidate, &estimated);
  LOG_I(HW,
        "[AGC] gNB SSS request=%.3f dBm/RE candidate amplitude=%d estimate=%.3f dBm/RE valid=%s apply=%s\n",
        requested_sss_dbm,
        candidate,
        estimated,
        valid ? "yes" : "no",
        radio_gain_device_tx_actuating() ? "yes" : "no");
  if (radio_gain_device_tx_actuating()) {
    if (!valid || !amplitude) {
      radio_gain_device_reject_tx(-1, -1, 0, RADIO_TX_REJECT_PROFILE);
      return false;
    }
    *amplitude = candidate;
    binding.gnb_sss_dbm = requested_sss_dbm;
    binding.gnb_amplitude = candidate;
    binding.gnb_reference_ready = true;
  }
  return true;
}

bool radio_gain_device_validate_gnb_reference(double requested_sss_dbm, int16_t amplitude, int frame, int slot)
{
  if (!radio_gain_device_tx_actuating())
    return true;
  if (!binding.gnb_reference_ready || requested_sss_dbm != binding.gnb_sss_dbm || amplitude != binding.gnb_amplitude) {
    radio_gain_device_reject_tx(frame, slot, 0, RADIO_TX_REJECT_PROFILE);
    return false;
  }
  return true;
}

bool radio_gain_device_validate_gnb_tx(c16_t *samples, uint32_t count, int frame, int slot)
{
  if (!radio_gain_device_tx_actuating())
    return true;
  if (relative_tx_selected()) {
    const radio_tx_sample_level_t level =
        radio_tx_sample_level((const int16_t *)samples, count, binding.tx_relative.component_full_scale);
    if (!binding.gnb_reference_ready || !level.valid) {
      radio_gain_device_reject_tx(frame, slot, 0, RADIO_TX_REJECT_SPAN);
      return false;
    }
    if (level.over_range_components
        || level.peak_component > binding.tx_relative.peak_limit_fs * binding.tx_relative.component_full_scale)
      erase_relative_tx(samples, count, frame, slot, 0, RADIO_TX_POWER_HEADROOM);
    return true;
  }
  const radio_tx_profile_t *p = &binding.tx_profile;
  const radio_tx_sample_level_t level = radio_tx_sample_level((const int16_t *)samples, count, p->component_full_scale);
  const double maximum_energy = (double)count * p->component_full_scale * p->component_full_scale
                                * pow(10.0, (p->power.maximum_dbm - p->power.reference_dbm) / 10.0);
  if (!binding.gnb_reference_ready || !level.valid || level.over_range_components
      || level.peak_component > p->peak_limit_fs * p->component_full_scale || level.sum_squared_components > maximum_energy) {
    radio_gain_device_reject_tx(frame, slot, 0, RADIO_TX_REJECT_POWER_LIMIT);
    return false;
  }
  return true;
}
