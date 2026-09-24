/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "radio_gain.h"

#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(condition)                                                               \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      fprintf(stderr, "check failed at %s:%d: %s\\n", __FILE__, __LINE__, #condition); \
      return EXIT_FAILURE;                                                             \
    }                                                                                  \
  } while (0)

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_t creator;
  bool query_on_creator;
  bool query_scheduler_valid;
  int query_scheduler_policy;
  int query_scheduler_priority;
  unsigned int query_calls;
  unsigned int fail_query_call;
  unsigned int rx_set_calls;
  unsigned int tx_set_calls;
  unsigned int rx_agc_calls;
  bool rx_agc_enabled;
  bool fail_set;
  bool bad_readback;
  bool block_set;
  bool set_entered;
  bool release_set;
  unsigned int retune_calls;
  bool fail_retune;
  int64_t ticks;
  int64_t tick_sequence[2];
  unsigned int tick_sequence_count;
  unsigned int tick_sequence_index;
  unsigned int tick_calls;
  bool fail_ticks;
  double rx_gain;
  double tx_gain;
  double rx_frequency_hz;
  double tx_frequency_hz;
  double last_requested_rx_db;
  double last_requested_tx_db;
} fake_radio_t;

static void fake_init(fake_radio_t *radio)
{
  memset(radio, 0, sizeof(*radio));
  if (pthread_mutex_init(&radio->mutex, NULL) != 0 || pthread_cond_init(&radio->cond, NULL) != 0)
    abort();
  radio->creator = pthread_self();
  radio->rx_gain = 10.0;
  radio->tx_gain = 11.0;
  radio->rx_frequency_hz = 3500000000.0;
  radio->tx_frequency_hz = 3600000000.0;
  radio->ticks = 100;
}

static void fake_fini(fake_radio_t *radio)
{
  if (pthread_cond_destroy(&radio->cond) != 0 || pthread_mutex_destroy(&radio->mutex) != 0)
    abort();
}

static void fake_set_tick_sequence(fake_radio_t *radio, int64_t first, int64_t second)
{
  pthread_mutex_lock(&radio->mutex);
  radio->tick_sequence[0] = first;
  radio->tick_sequence[1] = second;
  radio->tick_sequence_count = 2;
  radio->tick_sequence_index = 0;
  pthread_mutex_unlock(&radio->mutex);
}

static int fake_query(void *opaque, radio_gain_direction_t direction, unsigned int channel, radio_gain_channel_t *result)
{
  fake_radio_t *radio = opaque;
  int scheduler_policy = 0;
  struct sched_param scheduler_parameters = {0};
  const int scheduler_status = pthread_getschedparam(pthread_self(), &scheduler_policy, &scheduler_parameters);
  pthread_mutex_lock(&radio->mutex);
  radio->query_calls++;
  if (pthread_equal(pthread_self(), radio->creator))
    radio->query_on_creator = true;
  radio->query_scheduler_valid = scheduler_status == 0;
  radio->query_scheduler_policy = scheduler_policy;
  radio->query_scheduler_priority = scheduler_parameters.sched_priority;
  const bool fail = radio->fail_query_call != 0 && radio->query_calls == radio->fail_query_call;
  const double gain = direction == RADIO_GAIN_RX ? radio->rx_gain : radio->tx_gain;
  const double frequency = direction == RADIO_GAIN_RX ? radio->rx_frequency_hz : radio->tx_frequency_hz;
  pthread_mutex_unlock(&radio->mutex);
  if (fail)
    return -1;

  *result = (radio_gain_channel_t){
      .minimum_db = 0.0,
      .maximum_db = 30.0,
      .step_db = 1.0,
      .reported_db = gain,
      .frequency_hz = frequency,
      .sample_rate_hz = 30720000.0,
      .component_full_scale = 2048,
  };
  (void)snprintf(result->identity, sizeof(result->identity), "fake-radio");
  (void)snprintf(result->antenna, sizeof(result->antenna), "%s", channel == 0 ? "RX0TX0" : "mapped");
  return 0;
}

static int fake_set_gain(void *opaque, radio_gain_direction_t direction, unsigned int channel, double gain_db, double *reported_db)
{
  fake_radio_t *radio = opaque;
  (void)channel;
  pthread_mutex_lock(&radio->mutex);
  if (direction == RADIO_GAIN_RX) {
    radio->rx_set_calls++;
    radio->last_requested_rx_db = gain_db;
  } else {
    radio->tx_set_calls++;
    radio->last_requested_tx_db = gain_db;
  }
  radio->set_entered = true;
  pthread_cond_broadcast(&radio->cond);
  while (radio->block_set && !radio->release_set)
    pthread_cond_wait(&radio->cond, &radio->mutex);
  const bool fail = radio->fail_set;
  const bool bad_readback = radio->bad_readback;
  const double coerced = round(gain_db);
  if (direction == RADIO_GAIN_RX)
    radio->rx_gain = coerced;
  else
    radio->tx_gain = coerced;
  pthread_mutex_unlock(&radio->mutex);
  if (fail)
    return -1; /* The device may still have applied the setting. */
  *reported_db = bad_readback ? 31.0 : coerced;
  return 0;
}

static int fake_set_rx_agc(void *opaque, unsigned int channel, bool enable)
{
  fake_radio_t *radio = opaque;
  (void)channel;
  pthread_mutex_lock(&radio->mutex);
  radio->rx_agc_calls++;
  radio->rx_agc_enabled = enable;
  pthread_mutex_unlock(&radio->mutex);
  return 0;
}

static int fake_retune(void *opaque, unsigned int rx_channel, unsigned int tx_channel, double rx_hz, double tx_hz, double offset_hz)
{
  fake_radio_t *radio = opaque;
  (void)rx_channel;
  (void)tx_channel;
  (void)offset_hz;
  pthread_mutex_lock(&radio->mutex);
  radio->retune_calls++;
  radio->rx_frequency_hz = rx_hz;
  if (!radio->fail_retune)
    radio->tx_frequency_hz = tx_hz;
  const bool fail = radio->fail_retune;
  pthread_mutex_unlock(&radio->mutex);
  return fail ? -1 : 0;
}

static int fake_ticks(void *opaque, double sample_rate_hz, int64_t *ticks)
{
  fake_radio_t *radio = opaque;
  (void)sample_rate_hz;
  pthread_mutex_lock(&radio->mutex);
  const bool fail = radio->fail_ticks;
  radio->tick_calls++;
  if (radio->tick_sequence_index < radio->tick_sequence_count)
    *ticks = radio->tick_sequence[radio->tick_sequence_index++];
  else
    *ticks = radio->ticks;
  pthread_mutex_unlock(&radio->mutex);
  return fail ? -1 : 0;
}

static radio_gain_api_t fake_api(void)
{
  return (radio_gain_api_t){
      .abi_version = OAI_RADIO_GAIN_ABI,
      .struct_size = sizeof(radio_gain_api_t),
      .query = fake_query,
      .set_gain = fake_set_gain,
      .set_rx_agc = fake_set_rx_agc,
      .retune = fake_retune,
      .device_ticks = fake_ticks,
  };
}

static bool take_result_wait(radio_gain_owner_t *owner, radio_gain_result_t *result)
{
  const struct timespec interval = {.tv_sec = 0, .tv_nsec = 1000000L};
  for (unsigned int attempt = 0; attempt < 1000; ++attempt) {
    if (radio_gain_take_result(owner, result))
      return true;
    (void)nanosleep(&interval, NULL);
  }
  return false;
}

static radio_gain_result_t submit_and_take(radio_gain_owner_t *owner, radio_gain_request_t request)
{
  radio_gain_result_t result = {0};
  if (radio_gain_submit(owner, &request) != RADIO_GAIN_OK || !take_result_wait(owner, &result))
    return (radio_gain_result_t){.status = RADIO_GAIN_BACKEND_ERROR};
  return result;
}

static void fake_wait_set_entered(fake_radio_t *radio)
{
  pthread_mutex_lock(&radio->mutex);
  while (!radio->set_entered)
    pthread_cond_wait(&radio->cond, &radio->mutex);
  pthread_mutex_unlock(&radio->mutex);
}

typedef struct {
  radio_gain_owner_t *owner;
  _Atomic(bool) stop;
  _Atomic(bool) failed;
} snapshot_reader_t;

static void *snapshot_reader(void *opaque)
{
  snapshot_reader_t *reader = opaque;
  while (!atomic_load_explicit(&reader->stop, memory_order_acquire)) {
    radio_gain_result_t snapshot;
    if (!radio_gain_snapshot(reader->owner, &snapshot))
      continue;
    if (snapshot.status == RADIO_GAIN_OK && snapshot.request.request_id != 0
        && (snapshot.request.request_id != snapshot.request.generation * UINT64_C(37) + UINT64_C(5)
            || snapshot.generation != snapshot.request.generation + 1 || !snapshot.rx_gain_valid)) {
      atomic_store_explicit(&reader->failed, true, memory_order_release);
      break;
    }
  }
  return NULL;
}

static _Atomic(int) terminal_calls;
static _Atomic(int) terminal_code;
static void record_terminal_failure(int code)
{
  atomic_fetch_add_explicit(&terminal_calls, 1, memory_order_relaxed);
  atomic_store_explicit(&terminal_code, code, memory_order_release);
}

int main(void)
{
  fake_radio_t invalid_radio;
  fake_init(&invalid_radio);
  radio_gain_api_t api = fake_api();
  api.abi_version = OAI_RADIO_GAIN_ABI + 1;
  CHECK(radio_gain_owner_create(&api, &invalid_radio, 0, 1, false) == NULL);
  api = fake_api();
  api.struct_size = 0;
  CHECK(radio_gain_owner_create(&api, &invalid_radio, 0, 1, false) == NULL);
  invalid_radio.fail_query_call = 2;
  CHECK(radio_gain_owner_create(&api, &invalid_radio, 0, 1, false) == NULL);
  CHECK(invalid_radio.rx_agc_calls == 0);
  fake_fini(&invalid_radio);

  fake_radio_t radio;
  fake_init(&radio);
  api = fake_api();
  radio_gain_owner_t *owner = radio_gain_owner_create(&api, &radio, 2, 3, false);
  CHECK(owner != NULL);
  CHECK(radio.query_calls == 2);
  CHECK(!radio.query_on_creator);
  CHECK(radio.query_scheduler_valid);
  CHECK(radio.query_scheduler_policy == SCHED_OTHER);
  CHECK(radio.query_scheduler_priority == 0);
  CHECK(radio.rx_agc_calls == 0);

  radio_gain_result_t snapshot;
  CHECK(radio_gain_snapshot(owner, &snapshot));
  CHECK(snapshot.status == RADIO_GAIN_OK);
  CHECK(snapshot.generation == 1);
  CHECK(snapshot.rx_gain_valid && snapshot.tx_gain_valid);
  radio_gain_channel_t rx_channel;
  radio_gain_channel_t tx_channel;
  CHECK(radio_gain_channels(owner, &rx_channel, &tx_channel));
  CHECK(rx_channel.frequency_hz == 3500000000.0 && tx_channel.frequency_hz == 3600000000.0);

  fake_set_tick_sequence(&radio, 100, 101);
  unsigned int tick_calls = radio.tick_calls;
  radio_gain_result_t result = submit_and_take(owner,
                                               (radio_gain_request_t){
                                                   .operation = RADIO_GAIN_SET_RX,
                                                   .generation = 1,
                                                   .request_id = 101,
                                                   .gain_db = 12.6,
                                               });
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 2);
  CHECK(result.request.gain_db == 12.6 && result.reported_rx_db == 13.0);
  CHECK(result.rx_gain_valid && result.tx_gain_valid && result.device_time_valid);
  CHECK(result.begin_device_ticks == 100 && result.end_device_ticks == 101 && radio.tick_calls == tick_calls + 2);
  CHECK(radio.rx_set_calls == 1 && radio.tx_set_calls == 0);

  radio.fail_ticks = true;
  tick_calls = radio.tick_calls;
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_RX,
                               .generation = 2,
                               .request_id = 102,
                               .gain_db = 100.0,
                           });
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 3 && result.reported_rx_db == 30.0);
  CHECK(!result.device_time_valid && radio.tick_calls == tick_calls + 1);
  CHECK(radio.last_requested_rx_db == 30.0);
  radio.fail_ticks = false;

  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_RX,
                               .generation = 1,
                               .request_id = 103,
                               .gain_db = 5.0,
                           });
  CHECK(result.status == RADIO_GAIN_STALE && result.generation == 3);

  radio.fail_set = true;
  fake_set_tick_sequence(&radio, 200, 201);
  tick_calls = radio.tick_calls;
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_RX,
                               .generation = 3,
                               .request_id = 104,
                               .gain_db = 5.0,
                           });
  CHECK(result.status == RADIO_GAIN_BACKEND_ERROR && result.generation == 4 && !result.rx_gain_valid && result.tx_gain_valid);
  CHECK(result.device_time_valid && result.begin_device_ticks == 200 && result.end_device_ticks == 201
        && radio.tick_calls == tick_calls + 2);
  CHECK(radio_gain_snapshot(owner, &snapshot));
  CHECK(snapshot.generation == 4 && !snapshot.rx_gain_valid && snapshot.tx_gain_valid && isnan(snapshot.reported_rx_db));
  radio.fail_set = false;

  radio.bad_readback = true;
  fake_set_tick_sequence(&radio, 300, 299);
  tick_calls = radio.tick_calls;
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_RX,
                               .generation = 4,
                               .request_id = 105,
                               .gain_db = 6.0,
                           });
  CHECK(result.status == RADIO_GAIN_BACKEND_ERROR && result.generation == 5 && !result.rx_gain_valid);
  CHECK(!result.device_time_valid && radio.tick_calls == tick_calls + 2);
  radio.bad_readback = false;

  fake_set_tick_sequence(&radio, 400, 405);
  radio.block_set = true;
  radio.set_entered = false;
  radio.release_set = false;
  tick_calls = radio.tick_calls;
  CHECK(radio_gain_submit(owner,
                          &(radio_gain_request_t){
                              .operation = RADIO_GAIN_SET_RX,
                              .generation = 5,
                              .request_id = 106,
                              .gain_db = 7.0,
                          })
        == RADIO_GAIN_OK);
  fake_wait_set_entered(&radio);
  CHECK(radio.tick_calls == tick_calls + 1);
  CHECK(radio_gain_snapshot(owner, &snapshot));
  CHECK(snapshot.status == RADIO_GAIN_BUSY && snapshot.generation == 6 && !snapshot.rx_gain_valid && snapshot.device_time_valid
        && snapshot.begin_device_ticks == 400);
  pthread_mutex_lock(&radio.mutex);
  radio.release_set = true;
  radio.block_set = false;
  pthread_cond_broadcast(&radio.cond);
  pthread_mutex_unlock(&radio.mutex);
  CHECK(take_result_wait(owner, &result));
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 6 && result.rx_gain_valid && result.device_time_valid);
  CHECK(result.begin_device_ticks == 400 && result.end_device_ticks == 405 && radio.tick_calls == tick_calls + 2);

  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_TX,
                               .generation = 6,
                               .request_id = 107,
                               .gain_db = 8.4,
                           });
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 7 && result.reported_tx_db == 8.0);
  CHECK(radio.rx_set_calls == 5 && radio.tx_set_calls == 1);

  CHECK(radio_gain_set_tx_admission(owner, true));
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_TX,
                               .generation = 7,
                               .request_id = 108,
                               .gain_db = 9.0,
                           });
  CHECK(result.status == RADIO_GAIN_TX_PENDING && result.generation == 7 && radio.tx_set_calls == 1);
  radio_gain_note_tx_end(owner, 500);
  CHECK(radio_gain_set_tx_admission(owner, false));
  radio.ticks = 499;
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_TX,
                               .generation = 7,
                               .request_id = 109,
                               .gain_db = 9.0,
                           });
  CHECK(result.status == RADIO_GAIN_TX_PENDING && result.generation == 7 && radio.tx_set_calls == 1);
  radio.ticks = 500;
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_SET_TX,
                               .generation = 7,
                               .request_id = 110,
                               .gain_db = 9.0,
                           });
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 8 && result.device_time_valid && result.begin_device_ticks == 500
        && radio.tx_set_calls == 2);

  radio.block_set = true;
  radio.set_entered = false;
  radio.release_set = false;
  CHECK(radio_gain_submit(owner,
                          &(radio_gain_request_t){
                              .operation = RADIO_GAIN_SET_TX,
                              .generation = 8,
                              .request_id = 111,
                              .gain_db = 10.0,
                          })
        == RADIO_GAIN_OK);
  fake_wait_set_entered(&radio);
  CHECK(!radio_gain_set_tx_admission(owner, true));
  CHECK(radio_gain_submit(owner,
                          &(radio_gain_request_t){
                              .operation = RADIO_GAIN_SET_RX,
                              .generation = 8,
                              .request_id = 112,
                              .gain_db = 11.0,
                          })
        == RADIO_GAIN_BUSY);
  pthread_mutex_lock(&radio.mutex);
  radio.release_set = true;
  radio.block_set = false;
  pthread_cond_broadcast(&radio.cond);
  pthread_mutex_unlock(&radio.mutex);
  CHECK(take_result_wait(owner, &result));
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 9);

  CHECK(radio_gain_set_tx_admission(owner, true));
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_RETUNE,
                               .generation = 9,
                               .request_id = 113,
                               .rx_frequency_hz = 3700000000.0,
                               .tx_frequency_hz = 3800000000.0,
                           });
  CHECK(result.status == RADIO_GAIN_TX_PENDING && result.generation == 9 && radio.retune_calls == 0);
  CHECK(radio_gain_set_tx_admission(owner, false));
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_RETUNE,
                               .generation = 9,
                               .request_id = 114,
                               .rx_frequency_hz = 3700000000.0,
                               .tx_frequency_hz = 3800000000.0,
                               .tune_offset_hz = 1.0,
                           });
  CHECK(result.status == RADIO_GAIN_OK && result.generation == 10 && result.rx_gain_valid && result.tx_gain_valid);
  CHECK(radio_gain_channels(owner, &rx_channel, &tx_channel));
  CHECK(rx_channel.frequency_hz == 3700000000.0 && tx_channel.frequency_hz == 3800000000.0);

  radio.fail_retune = true;
  result = submit_and_take(owner,
                           (radio_gain_request_t){
                               .operation = RADIO_GAIN_RETUNE,
                               .generation = 10,
                               .request_id = 115,
                               .rx_frequency_hz = 3900000000.0,
                               .tx_frequency_hz = 4000000000.0,
                           });
  CHECK(result.status == RADIO_GAIN_BACKEND_ERROR && result.generation == 11 && !result.rx_gain_valid && !result.tx_gain_valid);
  CHECK(!radio_gain_channels(owner, &rx_channel, &tx_channel));
  CHECK(radio_gain_snapshot(owner, &snapshot));
  CHECK(snapshot.generation == 11 && !snapshot.rx_gain_valid && !snapshot.tx_gain_valid);

  radio_gain_owner_close(owner);
  CHECK(radio_gain_submit(owner,
                          &(radio_gain_request_t){
                              .operation = RADIO_GAIN_SET_RX,
                              .generation = 11,
                              .request_id = 116,
                              .gain_db = 1.0,
                          })
        == RADIO_GAIN_CLOSED);
  radio_gain_owner_destroy(owner);
  fake_fini(&radio);

  fake_radio_t agc_radio;
  fake_init(&agc_radio);
  owner = radio_gain_owner_create(&api, &agc_radio, 0, 0, true);
  CHECK(owner != NULL && agc_radio.rx_agc_calls == 1 && !agc_radio.rx_agc_enabled);
  radio_gain_owner_destroy(owner);
  fake_fini(&agc_radio);

  fake_radio_t race_radio;
  fake_init(&race_radio);
  owner = radio_gain_owner_create(&api, &race_radio, 0, 0, false);
  CHECK(owner != NULL);
  snapshot_reader_t reader = {.owner = owner};
  atomic_init(&reader.stop, false);
  atomic_init(&reader.failed, false);
  pthread_t reader_thread;
  CHECK(pthread_create(&reader_thread, NULL, snapshot_reader, &reader) == 0);
  for (uint64_t generation = 1; generation <= 128; ++generation) {
    const radio_gain_request_t request = {
        .operation = RADIO_GAIN_SET_RX,
        .generation = generation,
        .request_id = generation * UINT64_C(37) + UINT64_C(5),
        .gain_db = (double)(generation % 30),
    };
    result = submit_and_take(owner, request);
    CHECK(result.status == RADIO_GAIN_OK && result.generation == generation + 1);
  }
  atomic_store_explicit(&reader.stop, true, memory_order_release);
  CHECK(pthread_join(reader_thread, NULL) == 0);
  CHECK(!atomic_load_explicit(&reader.failed, memory_order_acquire));
  radio_gain_owner_set_failure_handler(owner, record_terminal_failure);
  radio_gain_owner_report_failure(owner, 55);
  radio_gain_owner_report_failure(owner, 66);
  for (int attempt = 0; attempt < 100 && atomic_load_explicit(&terminal_code, memory_order_acquire) == 0; ++attempt) {
    const struct timespec wait = {.tv_nsec = 1000000};
    nanosleep(&wait, NULL);
  }
  CHECK(atomic_load_explicit(&terminal_code, memory_order_acquire) == 55);
  radio_gain_owner_destroy(owner);
  CHECK(atomic_load_explicit(&terminal_calls, memory_order_acquire) == 1);
  fake_fini(&race_radio);

  return EXIT_SUCCESS;
}
