/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_gain_device.h"
#include "radio_gain_policy.h"
#include "common_lib.h"
#include "executables/agc_options.h"
#include "common/utils/LOG/flight_recorder.h"
#include "common/utils/LOG/log.h"

#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(condition)                                                               \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      fprintf(stderr, "check failed at %s:%d: %s\\n", __FILE__, __LINE__, #condition); \
      return EXIT_FAILURE;                                                             \
    }                                                                                  \
  } while (0)

static agc_options_t test_options;
static log_t test_log;
log_t *g_log = &test_log;

const agc_options_t *get_agc_options(void)
{
  return &test_options;
}

void logRecord_mt(const char *file, const char *function, int line, int component, int level, const char *format, ...)
{
  (void)file;
  (void)function;
  (void)line;
  (void)component;
  (void)level;
  (void)format;
}

typedef struct {
  uint32_t event;
  int64_t a;
  int64_t b;
  int64_t c;
  int64_t d;
  int64_t e;
  int64_t f;
} recorder_event_t;

enum { TEST_RECORDER_EVENTS = 32 };
static bool test_recorder_enabled;
static unsigned int test_recorder_count;
static recorder_event_t test_recorder_events[TEST_RECORDER_EVENTS];

static void recorder_reset(bool enabled)
{
  test_recorder_enabled = enabled;
  test_recorder_count = 0;
  memset(test_recorder_events, 0, sizeof(test_recorder_events));
}

static unsigned int recorder_event_count(uint32_t event)
{
  unsigned int count = 0;
  for (unsigned int index = 0; index < test_recorder_count; ++index)
    count += test_recorder_events[index].event == event;
  return count;
}

static const recorder_event_t *recorder_last_event(uint32_t event)
{
  for (unsigned int index = test_recorder_count; index > 0; --index) {
    if (test_recorder_events[index - 1].event == event)
      return &test_recorder_events[index - 1];
  }
  return NULL;
}

bool flight_recorder_enabled(void)
{
  return test_recorder_enabled;
}

void flight_recorder_emit(uint32_t event, int64_t a, int64_t b, int64_t c, int64_t d, int64_t e, int64_t f)
{
  if (!test_recorder_enabled || test_recorder_count == TEST_RECORDER_EVENTS)
    return;
  test_recorder_events[test_recorder_count++] = (recorder_event_t){.event = event, .a = a, .b = b, .c = c, .d = d, .e = e, .f = f};
}

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t condition;
  unsigned int query_calls;
  unsigned int set_calls;
  unsigned int retune_calls;
  unsigned int write_calls;
  unsigned int legacy_frequency_calls;
  unsigned int end_calls;
  unsigned int stop_calls;
  bool rx_agc_enabled;
  bool fail_retune;
  bool block_retune;
  bool release_retune;
  bool retune_entered;
  bool block_write;
  bool release_write;
  bool write_entered;
  bool block_set;
  bool release_set;
  bool set_entered;
  bool inline_shutdown;
  bool inline_shutdown_returned;
  int inline_stop_result;
  int read_count;
  openair0_timestamp_t read_timestamp;
  int64_t ticks;
  double rx_gain;
  double tx_gain;
  double rx_frequency;
  double tx_frequency;
} fake_radio_t;

static fake_radio_t *radio_for(void *opaque)
{
  return ((openair0_device_t *)opaque)->priv;
}

static void fake_init(fake_radio_t *radio)
{
  memset(radio, 0, sizeof(*radio));
  if (pthread_mutex_init(&radio->mutex, NULL) != 0 || pthread_cond_init(&radio->condition, NULL) != 0)
    abort();
  radio->ticks = 100;
  radio->rx_gain = 10.0;
  radio->tx_gain = 11.0;
  radio->rx_frequency = 3500000000.0;
  radio->tx_frequency = 3600000000.0;
}

static void fake_fini(fake_radio_t *radio)
{
  if (pthread_cond_destroy(&radio->condition) != 0 || pthread_mutex_destroy(&radio->mutex) != 0)
    abort();
}

static int fake_query(void *opaque, radio_gain_direction_t direction, unsigned int channel, radio_gain_channel_t *result)
{
  fake_radio_t *radio = radio_for(opaque);
  (void)channel;
  pthread_mutex_lock(&radio->mutex);
  radio->query_calls++;
  const double gain = direction == RADIO_GAIN_RX ? radio->rx_gain : radio->tx_gain;
  const double frequency = direction == RADIO_GAIN_RX ? radio->rx_frequency : radio->tx_frequency;
  pthread_mutex_unlock(&radio->mutex);

  *result = (radio_gain_channel_t){
      .minimum_db = 0.0,
      .maximum_db = 30.0,
      .step_db = 1.0,
      .reported_db = gain,
      .frequency_hz = frequency,
      .sample_rate_hz = 30720000.0,
      .bandwidth_hz = 20000000,
      .component_full_scale = 2048,
  };
  (void)snprintf(result->identity, sizeof(result->identity), "fake-radio");
  (void)snprintf(result->antenna, sizeof(result->antenna), "RF0");
  return 0;
}

static int fake_set_gain(void *opaque, radio_gain_direction_t direction, unsigned int channel, double gain_db, double *reported_db)
{
  fake_radio_t *radio = radio_for(opaque);
  (void)channel;
  pthread_mutex_lock(&radio->mutex);
  radio->set_calls++;
  radio->set_entered = true;
  pthread_cond_broadcast(&radio->condition);
  while (radio->block_set && !radio->release_set)
    pthread_cond_wait(&radio->condition, &radio->mutex);
  if (direction == RADIO_GAIN_RX)
    radio->rx_gain = gain_db;
  else
    radio->tx_gain = gain_db;
  pthread_mutex_unlock(&radio->mutex);
  *reported_db = gain_db;
  return 0;
}

static int fake_set_rx_agc(void *opaque, unsigned int channel, bool enable)
{
  fake_radio_t *radio = radio_for(opaque);
  (void)channel;
  pthread_mutex_lock(&radio->mutex);
  radio->rx_agc_enabled = enable;
  pthread_mutex_unlock(&radio->mutex);
  return 0;
}

static int fake_retune(void *opaque,
                       unsigned int rx_channel,
                       unsigned int tx_channel,
                       double rx_frequency,
                       double tx_frequency,
                       double tune_offset)
{
  fake_radio_t *radio = radio_for(opaque);
  (void)rx_channel;
  (void)tx_channel;
  (void)tune_offset;
  pthread_mutex_lock(&radio->mutex);
  radio->retune_calls++;
  radio->retune_entered = true;
  pthread_cond_broadcast(&radio->condition);
  while (radio->block_retune && !radio->release_retune)
    pthread_cond_wait(&radio->condition, &radio->mutex);
  radio->rx_frequency = rx_frequency;
  if (!radio->fail_retune)
    radio->tx_frequency = tx_frequency;
  const bool fail = radio->fail_retune;
  pthread_mutex_unlock(&radio->mutex);
  return fail ? -1 : 0;
}

static int fake_ticks(void *opaque, double sample_rate, int64_t *ticks)
{
  fake_radio_t *radio = radio_for(opaque);
  (void)sample_rate;
  pthread_mutex_lock(&radio->mutex);
  *ticks = radio->ticks;
  pthread_mutex_unlock(&radio->mutex);
  return 0;
}

static int legacy_read(openair0_device_t *device, openair0_timestamp_t *timestamp, void **buffers, int count, int antennas)
{
  fake_radio_t *radio = device->priv;
  (void)buffers;
  (void)count;
  (void)antennas;
  if (radio->inline_shutdown) {
    radio->inline_stop_result = device->trx_stop_func(device);
    device->trx_end_func(device);
    radio->inline_shutdown_returned = true;
    if (timestamp != NULL)
      *timestamp = 100;
    return 1;
  }
  if (radio->read_count > 0) {
    if (timestamp != NULL)
      *timestamp = radio->read_timestamp;
    return radio->read_count;
  }
  return -ENOTSUP;
}

static int legacy_write(openair0_device_t *device,
                        openair0_timestamp_t timestamp,
                        void **buffers,
                        int count,
                        int antennas,
                        int flags)
{
  fake_radio_t *radio = device->priv;
  (void)timestamp;
  (void)buffers;
  (void)antennas;
  (void)flags;
  pthread_mutex_lock(&radio->mutex);
  radio->write_calls++;
  radio->write_entered = true;
  pthread_cond_broadcast(&radio->condition);
  while (radio->block_write && !radio->release_write)
    pthread_cond_wait(&radio->condition, &radio->mutex);
  pthread_mutex_unlock(&radio->mutex);
  return count;
}

static int legacy_write2(openair0_device_t *device,
                         openair0_timestamp_t timestamp,
                         void **buffers,
                         int fd_ind,
                         int count,
                         int flags,
                         int antennas)
{
  (void)device;
  (void)timestamp;
  (void)buffers;
  (void)fd_ind;
  (void)count;
  (void)flags;
  (void)antennas;
  return -ENOTSUP;
}

static int legacy_set_frequency(openair0_device_t *device, openair0_config_t *config)
{
  fake_radio_t *radio = device->priv;
  (void)config;
  pthread_mutex_lock(&radio->mutex);
  radio->legacy_frequency_calls++;
  pthread_mutex_unlock(&radio->mutex);
  return 0;
}

static int legacy_stop(openair0_device_t *device)
{
  fake_radio_t *radio = device->priv;
  pthread_mutex_lock(&radio->mutex);
  radio->stop_calls++;
  pthread_mutex_unlock(&radio->mutex);
  return 0;
}

static void legacy_end(openair0_device_t *device)
{
  fake_radio_t *radio = device->priv;
  pthread_mutex_lock(&radio->mutex);
  radio->end_calls++;
  pthread_mutex_unlock(&radio->mutex);
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

static void make_device(openair0_device_t *device, openair0_config_t *config, fake_radio_t *radio)
{
  memset(device, 0, sizeof(*device));
  memset(config, 0, sizeof(*config));
  config->rx_num_channels = 1;
  config->tx_num_channels = 1;
  config->rx_freq[0] = radio->rx_frequency;
  config->tx_freq[0] = radio->tx_frequency;
  device->priv = radio;
  device->trx_read_func = legacy_read;
  device->trx_write_func = legacy_write;
  device->trx_set_freq_func = legacy_set_frequency;
  device->trx_stop_func = legacy_stop;
  device->trx_end_func = legacy_end;
}

static void wait_retune_entered(fake_radio_t *radio)
{
  pthread_mutex_lock(&radio->mutex);
  while (!radio->retune_entered)
    pthread_cond_wait(&radio->condition, &radio->mutex);
  pthread_mutex_unlock(&radio->mutex);
}

static void wait_write_entered(fake_radio_t *radio)
{
  pthread_mutex_lock(&radio->mutex);
  while (!radio->write_entered)
    pthread_cond_wait(&radio->condition, &radio->mutex);
  pthread_mutex_unlock(&radio->mutex);
}

static void wait_set_entered(fake_radio_t *radio)
{
  pthread_mutex_lock(&radio->mutex);
  while (!radio->set_entered)
    pthread_cond_wait(&radio->condition, &radio->mutex);
  pthread_mutex_unlock(&radio->mutex);
}

typedef struct {
  openair0_device_t *device;
  openair0_config_t *config;
  int result;
} retune_thread_t;

static void *retune_thread(void *opaque)
{
  retune_thread_t *thread = opaque;
  thread->result = thread->device->trx_set_freq_func(thread->device, thread->config);
  return NULL;
}

typedef struct {
  openair0_device_t *device;
  openair0_timestamp_t timestamp;
  int count;
  int result;
} write_thread_t;

static void *write_thread(void *opaque)
{
  write_thread_t *thread = opaque;
  thread->result = thread->device->trx_write_func(thread->device, thread->timestamp, NULL, thread->count, 1, 0);
  return NULL;
}

static uint64_t test_monotonic_ns(void)
{
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec < 0)
    return 0;
  return (uint64_t)now.tv_sec * UINT64_C(1000000000) + now.tv_nsec;
}

static radio_gain_sample_context_t policy_context(uint64_t generation, int64_t first_sample, int64_t end_sample)
{
  return (radio_gain_sample_context_t){
      .present = true,
      .valid = true,
      .generation = generation,
      .rx_gain_db = 10.0,
      .first_sample = first_sample,
      .end_sample = end_sample,
      .level_valid = true,
      .mean_power_fs = 0.1,
      .peak_component_fs = 0.4,
      .sampled_components = 100,
      .near_rail_components = 0,
      .observation_ns = test_monotonic_ns(),
  };
}

static int test_observe_policy_candidates(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_OBSERVE,
      .directions = AGC_DIRECTIONS_BOTH,
      .rx_acquisition = AGC_RX_ACQUISITION_NEW,
      .rx_tracking = AGC_RX_TRACKING_NEW,
  };
  const radio_gain_api_t api = fake_api();
  recorder_reset(true);
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  recorder_reset(true); /* Initial adoption is not a policy decision. */

  radio_gain_sample_context_t stale = policy_context(0, 100, 200);
  CHECK(stale.observation_ns != 0);
  radio_gain_device_observe_rx(&stale, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  CHECK(radio.set_calls == 0 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION) == 1);
  const recorder_event_t *event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_DECISION);
  CHECK(event != NULL && event->a == RADIO_RX_SOURCE_UE_SSB && event->b == 0 && event->c == 200
        && (event->f & (INT64_C(1) << 8)) == 0 && (event->f & (INT64_C(1) << 9)) == 0);

  radio_gain_sample_context_t mixed = policy_context(1, 200, 300);
  CHECK(mixed.observation_ns != 0);
  mixed.valid = false;
  radio_gain_device_observe_rx(&mixed, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  CHECK(radio.set_calls == 0 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION) == 1);

  radio_gain_sample_context_t serving = policy_context(1, 300, 400);
  CHECK(serving.observation_ns != 0);
  radio_gain_device_observe_rx(&serving, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  CHECK(radio.set_calls == 0 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION) == 2
        && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION_INPUT) == 2);
  event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_DECISION);
  CHECK(event != NULL && event->a == RADIO_RX_SOURCE_UE_SSB && event->b == 1 && event->c == 400 && event->d == -10000
        && event->e == 7000 && (event->f & 0xff) == RADIO_RX_TRACK_LEVEL && (event->f & (INT64_C(1) << 8)) != 0
        && (event->f & (INT64_C(1) << 9)) == 0);

  const struct timespec interval = {.tv_nsec = 1000000};
  CHECK(nanosleep(&interval, NULL) == 0);
  radio_gain_sample_context_t headroom = policy_context(1, 400, 500);
  CHECK(headroom.observation_ns != 0);
  headroom.mean_power_fs = 0.5;
  headroom.peak_component_fs = 0.99;
  headroom.near_rail_components = 2;
  radio_gain_device_observe_rx(&headroom, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  CHECK(radio.set_calls == 0 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION) == 3
        && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION_INPUT) == 3);
  event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_DECISION);
  CHECK(event != NULL && event->a == RADIO_RX_SOURCE_HEADROOM && event->b == 1 && event->c == 500 && event->e == 7000
        && (event->f & 0xff) == RADIO_RX_REDUCE_OVERLOAD && (event->f & (INT64_C(1) << 8)) != 0
        && (event->f & (INT64_C(1) << 9)) == 0);

  /* A raw reader has already observed newer samples when delayed PBCH
   * processing reports its still-fresh serving reference. */
  radio_gain_sample_context_t delayed = policy_context(1, 450, 550);
  delayed.observation_ns = serving.observation_ns + (headroom.observation_ns - serving.observation_ns) / 2;
  radio_gain_device_observe_rx(&delayed, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_DECISION);
  CHECK(event != NULL && event->a == RADIO_RX_SOURCE_UE_SSB && (event->f & 0xff) == RADIO_RX_TRACK_LEVEL);
  CHECK(radio.set_calls == 0);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_off_and_acquisition_policy_gates(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  const radio_gain_api_t api = fake_api();
  test_options = (agc_options_t){.mode = AGC_MODE_OFF};
  recorder_reset(true);
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  CHECK(device.trx_read_func == legacy_read && device.trx_set_freq_func == legacy_set_frequency);
  radio_gain_sample_context_t context = policy_context(1, 100, 200);
  CHECK(context.observation_ns != 0);
  radio_gain_device_observe_rx(&context, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  CHECK(radio.set_calls == 0 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION) == 0);

  test_options = (agc_options_t){
      .mode = AGC_MODE_ACQUISITION,
      .rx_acquisition = AGC_RX_ACQUISITION_LEGACY,
      .rx_tracking = AGC_RX_TRACKING_HOLD,
      .rx_actuation = true,
  };
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  recorder_reset(true); /* Initial adoption is outside the new RX policy. */
  context = policy_context(1, 200, 300);
  CHECK(context.observation_ns != 0);
  radio_gain_device_observe_rx(&context, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  CHECK(radio.set_calls == 0 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_DECISION) == 0);
  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_raw_read_summary_and_default_settle_guard(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_ACQUISITION,
      .rx_acquisition = AGC_RX_ACQUISITION_LEGACY,
      .rx_tracking = AGC_RX_TRACKING_HOLD,
      .rx_actuation = true,
      .rx_settle_us = 0,
  };
  const radio_gain_api_t api = fake_api();
  recorder_reset(false);
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  double applied = 0, reported = 0;
  CHECK(radio_gain_device_adjust_rx(&device, 1.0, &applied, &reported) == 0 && applied == 1.0 && radio.set_calls == 1);

  int16_t iq[2] = {1024, -1024};
  void *buffers[] = {iq};
  radio.read_count = 1;
  radio.read_timestamp = 614499;
  openair0_timestamp_t timestamp = 0;
  CHECK(device.trx_read_func(&device, &timestamp, buffers, 1, 1) == 1 && timestamp == 614499);
  radio_gain_sample_context_t context = radio_gain_device_samples(&device, 614499, 614500);
  CHECK(context.present && !context.valid && context.generation == 0 && context.level_valid && context.mean_power_fs == 0.5
        && context.peak_component_fs == 0.5 && context.sampled_components == 2 && context.near_rail_components == 0
        && context.observation_ns != 0 && test_recorder_count == 0);

  radio.read_timestamp = 614500;
  CHECK(device.trx_read_func(&device, &timestamp, buffers, 1, 1) == 1 && timestamp == 614500);
  context = radio_gain_device_samples(&device, 614500, 614501);
  CHECK(context.present && context.valid && context.generation == 2 && context.rx_gain_db == 11.0 && context.level_valid
        && context.mean_power_fs == 0.5 && context.peak_component_fs == 0.5 && context.sampled_components == 2
        && context.near_rail_components == 0 && context.observation_ns != 0 && test_recorder_count == 0);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_continuous_rejected(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){.mode = AGC_MODE_CONTINUOUS, .rx_actuation = true, .tx_actuation = true};
  const radio_gain_api_t api = fake_api();
  CHECK(radio_gain_device_attach(&device, &config, &api) == -ENOTSUP);
  CHECK(device.trx_set_freq_func == legacy_set_frequency);
  CHECK(device.trx_write_func == legacy_write);

  test_options = (agc_options_t){.mode = AGC_MODE_CONTINUOUS, .directions = AGC_DIRECTIONS_TX, .tx_actuation = true};
  CHECK(radio_gain_device_attach(&device, &config, &api) == -ENOTSUP);
  CHECK(device.trx_set_freq_func == legacy_set_frequency);
  CHECK(device.trx_write_func == legacy_write);

  radio_gain_api_t missing_retune = api;
  missing_retune.retune = NULL;
  test_options = (agc_options_t){
      .mode = AGC_MODE_CONTINUOUS,
      .directions = AGC_DIRECTIONS_RX,
      .rx_acquisition = AGC_RX_ACQUISITION_LEGACY,
      .rx_tracking = AGC_RX_TRACKING_NEW,
      .rx_actuation = true,
  };
  CHECK(radio_gain_device_attach(&device, &config, &missing_retune) == -ENOTSUP);
  CHECK(device.trx_set_freq_func == legacy_set_frequency);
  CHECK(device.trx_write_func == legacy_write);

  test_options = (agc_options_t){.mode = AGC_MODE_OBSERVE};
  device.trx_write_func2 = legacy_write2;
  CHECK(radio_gain_device_attach(&device, &config, &api) == -ENOTSUP);
  CHECK(device.trx_set_freq_func == legacy_set_frequency);
  CHECK(device.trx_write_func == legacy_write);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_continuous_legacy_handoff(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_CONTINUOUS,
      .directions = AGC_DIRECTIONS_RX,
      .rx_acquisition = AGC_RX_ACQUISITION_LEGACY,
      .rx_tracking = AGC_RX_TRACKING_NEW,
      .rx_actuation = true,
  };
  const radio_gain_api_t api = fake_api();
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);

  int16_t clipped_iq[2] = {2047, -2047};
  void *buffers[] = {clipped_iq};
  radio.read_count = 1;
  radio.read_timestamp = 100;
  openair0_timestamp_t timestamp = 0;
  CHECK(device.trx_read_func(&device, &timestamp, buffers, 1, 1) == 1);
  CHECK(radio.set_calls == 0); /* Legacy is the sole acquisition policy. */

  radio_gain_sample_context_t stale = policy_context(1, 100, 101);
  CHECK(stale.observation_ns != 0);
  const struct timespec interval = {.tv_nsec = 1000000};
  CHECK(nanosleep(&interval, NULL) == 0);
  radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_TRACKING);
  radio_gain_device_observe_rx(&stale, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  CHECK(radio.set_calls == 0); /* Delayed acquisition context cannot track. */

  pthread_mutex_lock(&radio.mutex);
  radio.block_set = true;
  radio.release_set = false;
  radio.set_entered = false;
  pthread_mutex_unlock(&radio.mutex);
  radio.read_timestamp = 101;
  CHECK(device.trx_read_func(&device, &timestamp, buffers, 1, 1) == 1);
  wait_set_entered(&radio);
  CHECK(radio.set_calls == 1);

  /* Loss returns to legacy acquisition without waiting in the phase call. */
  radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_ACQUISITION);
  radio_gain_device_observe_rx(&stale, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  CHECK(radio.set_calls == 1);
  pthread_mutex_lock(&radio.mutex);
  radio.release_set = true;
  pthread_cond_broadcast(&radio.condition);
  pthread_mutex_unlock(&radio.mutex);

  double applied = 0, reported = 0;
  CHECK(radio_gain_device_adjust_rx(&device, 1.0, &applied, &reported) == 0 && applied == 1.0 && reported == 8.0
        && radio.set_calls == 2);
  radio_gain_device_observe_rx(&stale, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  CHECK(radio.set_calls == 2);
  CHECK(radio_gain_device_adjust_rx(&device, -100.0, &applied, &reported) == 0 && applied == -8.0 && reported == 0.0
        && radio.set_calls == 3);
  CHECK(radio_gain_device_adjust_rx(&device, -1.0, &applied, &reported) == 0 && applied == 0.0 && reported == 0.0
        && radio.set_calls == 3);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_continuous_new_acquisition_shared_cooldown(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_CONTINUOUS,
      .directions = AGC_DIRECTIONS_RX,
      .rx_acquisition = AGC_RX_ACQUISITION_NEW,
      .rx_tracking = AGC_RX_TRACKING_NEW,
      .rx_actuation = true,
  };
  const radio_gain_api_t api = fake_api();
  recorder_reset(true);
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  recorder_reset(true);

  radio_gain_sample_context_t headroom = policy_context(1, 100, 200);
  CHECK(headroom.observation_ns != 0);
  headroom.peak_component_fs = 0.99;
  headroom.near_rail_components = headroom.sampled_components;
  radio_gain_device_observe_rx(&headroom, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  wait_set_entered(&radio);
  CHECK(radio.set_calls == 1);

  radio_gain_device_set_rx_phase(RADIO_GAIN_RX_PHASE_TRACKING);
  const struct timespec interval = {.tv_nsec = 1000000};
  CHECK(nanosleep(&interval, NULL) == 0);
  radio_gain_sample_context_t reference = policy_context(2, 200, 300);
  CHECK(reference.observation_ns != 0);
  reference.rx_gain_db = 7.0;
  radio_gain_device_observe_rx(&reference, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_UE_SSB);
  CHECK(radio.set_calls == 1 && recorder_event_count(FLIGHT_EVENT_RADIO_GAIN) == 1);
  const recorder_event_t *event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_DECISION);
  CHECK(event != NULL && (event->f & 0xff) == RADIO_RX_HOLD_COOLDOWN);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_continuous_gnb_tracking(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .role = AGC_ROLE_GNB,
      .mode = AGC_MODE_CONTINUOUS,
      .directions = AGC_DIRECTIONS_RX,
      .rx_acquisition = AGC_RX_ACQUISITION_NEW,
      .rx_tracking = AGC_RX_TRACKING_NEW,
      .rx_actuation = true,
  };
  const radio_gain_api_t api = fake_api();
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  radio_gain_sample_context_t pusch = policy_context(1, 100, 200);
  CHECK(pusch.observation_ns != 0);
  radio_gain_device_observe_rx(&pusch, 0.1 * 2048.0 * 2048.0, true, false, RADIO_RX_SOURCE_GNB_PUSCH);
  wait_set_entered(&radio);
  CHECK(radio.set_calls == 1);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_shared_peak_envelope_binding(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  radio.rx_gain = 27.0;
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .role = AGC_ROLE_GNB,
      .mode = AGC_MODE_CONTINUOUS,
      .directions = AGC_DIRECTIONS_RX,
      .rx_acquisition = AGC_RX_ACQUISITION_NEW,
      .rx_tracking = AGC_RX_TRACKING_NEW,
      .rx_actuation = true,
  };
  const radio_gain_api_t api = fake_api();
  recorder_reset(true);
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  recorder_reset(true);

  radio_gain_sample_context_t headroom = policy_context(1, 100, 200);
  headroom.rx_gain_db = 27.0;
  headroom.peak_component_fs = pow(10.0, -1.7 / 20.0);
  radio_gain_device_observe_rx(&headroom, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  wait_set_entered(&radio);
  CHECK(radio.set_calls == 1);

  const struct timespec release_interval = {.tv_nsec = 220000000};
  CHECK(nanosleep(&release_interval, NULL) == 0);
  radio_gain_sample_context_t pusch = policy_context(2, 200, 300);
  pusch.rx_gain_db = 24.0;
  pusch.peak_component_fs = pow(10.0, -26.0 / 20.0);
  const double weak_pusch_power = pow(10.0, -22.0 / 10.0) * 2048.0 * 2048.0;
  radio_gain_device_observe_rx(&pusch, weak_pusch_power, true, false, RADIO_RX_SOURCE_GNB_PUSCH);
  CHECK(radio.set_calls == 1 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_PEAK_ENVELOPE) == 2);
  const recorder_event_t *event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_PEAK_ENVELOPE);
  CHECK(event != NULL && event->a == RADIO_RX_SOURCE_GNB_PUSCH && event->b == 2 && event->c == 300 && event->d <= -5300
        && event->d > -6000 && event->e == 24000 && event->f == 3000);

  /* -5 dBFS lies below the raw overload ceiling but above the precomputed -6
   * dBFS refresh threshold, so the quiet-read fast path must still refresh the
   * shared input-referred envelope. */
  headroom = policy_context(2, 300, 400);
  headroom.rx_gain_db = 24.0;
  headroom.peak_component_fs = pow(10.0, -5.0 / 20.0);
  radio_gain_device_observe_rx(&headroom, 0, false, false, RADIO_RX_SOURCE_HEADROOM);
  CHECK(radio.set_calls == 1 && recorder_event_count(FLIGHT_EVENT_RADIO_RX_PEAK_ENVELOPE) == 3);
  event = recorder_last_event(FLIGHT_EVENT_RADIO_RX_PEAK_ENVELOPE);
  CHECK(event != NULL && event->a == RADIO_RX_SOURCE_HEADROOM && event->b == 2 && event->c == 400 && event->d == -5000
        && event->e == 24000 && event->f == 3000);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_observe_no_gain_write(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){.mode = AGC_MODE_OBSERVE};
  const radio_gain_api_t api = fake_api();
  double applied = 123.0, reported = 0;
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  config.rx_freq[0] = 3510000000.0;
  config.tx_freq[0] = 3610000000.0;
  CHECK(device.trx_set_freq_func(&device, &config) == 0);
  CHECK(radio.retune_calls == 1 && radio.legacy_frequency_calls == 0);
  CHECK(radio_gain_device_adjust_rx(&device, 1.0, &applied, &reported) == -ENOTSUP);
  CHECK(applied == 0.0);
  CHECK(radio.set_calls == 0);
  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_tx_level_records_and_disabled_path(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){.mode = AGC_MODE_OBSERVE};
  const radio_gain_api_t api = fake_api();
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  int16_t data[] = {-2048, 2047, 2048, -2049};
  void *buffers[] = {data};
  recorder_reset(false);
  CHECK(device.trx_write_func(&device, 100, buffers, 2, 1, 0) == 2);
  CHECK(test_recorder_count == 0);
  recorder_reset(true);
  CHECK(device.trx_write_func(&device, 200, buffers, 2, 1, 0) == 2);
  const recorder_event_t *level = recorder_last_event(FLIGHT_EVENT_RADIO_TX_LEVEL);
  const recorder_event_t *state = recorder_last_event(FLIGHT_EVENT_RADIO_TX_LEVEL_STATE);
  CHECK(level && state && level->b == 200 && level->c == 2 && level->d == 2048 && level->e == 16777218
        && level->f == ((INT64_C(2049) << 32) | 2));
  CHECK(state->b == 200 && state->c == 2 && state->d == 2 && state->e == 11000 && state->f > 0);
  CHECK(data[0] == -2048 && data[1] == 2047 && data[2] == 2048 && data[3] == -2049);
  for (unsigned int i = 0; i < 63; ++i)
    CHECK(device.trx_write_func(&device, 300 + i, buffers, 2, 1, 0) == 2);
  CHECK(recorder_event_count(FLIGHT_EVENT_RADIO_TX_LEVEL) == 1);
  CHECK(device.trx_write_func(&device, 400, buffers, 2, 1, 0) == 2);
  CHECK(recorder_event_count(FLIGHT_EVENT_RADIO_TX_LEVEL) == 2 && radio.set_calls == 0);
  device.trx_end_func(&device);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int run_child(int (*test)(void))
{
  const pid_t child = fork();
  if (child < 0)
    return EXIT_FAILURE;
  if (child == 0)
    _exit(test());
  int status;
  return waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == EXIT_SUCCESS ? EXIT_SUCCESS
                                                                                                         : EXIT_FAILURE;
}

static int test_acquisition_retune_and_tx_fence(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_ACQUISITION,
      .rx_acquisition = AGC_RX_ACQUISITION_LEGACY,
      .rx_actuation = true,
  };
  const radio_gain_api_t api = fake_api();

  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  CHECK(device.trx_set_freq_func != legacy_set_frequency);
  CHECK(device.trx_write_func != legacy_write);
  CHECK(!radio.rx_agc_enabled);

  config.rx_freq[0] = 3510000000.0;
  config.tx_freq[0] = 3610000000.0;
  config.tune_offset = 250000.0;
  CHECK(device.trx_set_freq_func(&device, &config) == 0);
  CHECK(radio.retune_calls == 1);
  CHECK(radio.legacy_frequency_calls == 0);
  CHECK(radio.rx_frequency == config.rx_freq[0] && radio.tx_frequency == config.tx_freq[0]);

  /* A pre-TX retune must establish its own RX settling boundary. No gain
   * adjustment is needed to recover valid measurements in the new epoch. */
  int16_t iq[2] = {1024, -1024};
  void *buffers[] = {iq};
  radio.read_count = 1;
  radio.read_timestamp = 614499;
  openair0_timestamp_t timestamp = 0;
  CHECK(device.trx_read_func(&device, &timestamp, buffers, 1, 1) == 1);
  radio_gain_sample_context_t context = radio_gain_device_samples(&device, 614499, 614500);
  CHECK(context.present && !context.valid);
  radio.read_timestamp = 614500;
  CHECK(device.trx_read_func(&device, &timestamp, buffers, 1, 1) == 1);
  context = radio_gain_device_samples(&device, 614500, 614501);
  CHECK(context.present && context.valid && context.generation == 2 && context.rx_gain_db == 10.0);
  CHECK(radio.set_calls == 0);

  radio.block_retune = true;
  radio.release_retune = false;
  radio.retune_entered = false;
  config.rx_freq[0] = 3520000000.0;
  config.tx_freq[0] = 3620000000.0;
  retune_thread_t retune = {.device = &device, .config = &config};
  pthread_t retune_id;
  CHECK(pthread_create(&retune_id, NULL, retune_thread, &retune) == 0);
  wait_retune_entered(&radio);
  CHECK(device.trx_set_freq_func(&device, &config) == -EBUSY);
  pthread_mutex_lock(&radio.mutex);
  radio.release_retune = true;
  pthread_cond_broadcast(&radio.condition);
  pthread_mutex_unlock(&radio.mutex);
  CHECK(pthread_join(retune_id, NULL) == 0);
  CHECK(retune.result == 0);
  CHECK(radio.retune_calls == 2);

  radio.block_retune = false;

  CHECK(device.trx_write_func(&device, 100, NULL, 16, 1, 0) == 16);
  CHECK(radio.write_calls == 1);
  radio.ticks = INT64_MAX;
  config.rx_freq[0] = 3530000000.0;
  config.tx_freq[0] = 3630000000.0;
  CHECK(device.trx_set_freq_func(&device, &config) == -EBUSY);
  CHECK(radio.retune_calls == 2);
  CHECK(radio.legacy_frequency_calls == 0);

  radio.block_write = true;
  radio.release_write = false;
  radio.write_entered = false;
  write_thread_t writer = {.device = &device, .timestamp = 200, .count = 10};
  pthread_t writer_id;
  CHECK(pthread_create(&writer_id, NULL, write_thread, &writer) == 0);
  wait_write_entered(&radio);
  CHECK(device.trx_write_func(&device, 201, NULL, 10, 1, 0) == -EBUSY);
  pthread_mutex_lock(&radio.mutex);
  radio.release_write = true;
  pthread_cond_broadcast(&radio.condition);
  pthread_mutex_unlock(&radio.mutex);
  CHECK(pthread_join(writer_id, NULL) == 0);
  CHECK(writer.result == 10);
  CHECK(radio.write_calls == 2);

  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  CHECK(device.trx_stop_func(&device) == 0);
  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_inline_callback_shutdown_deferred(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){.mode = AGC_MODE_OBSERVE};
  const radio_gain_api_t api = fake_api();
  radio.inline_shutdown = true;

  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  openair0_timestamp_t timestamp = 0;
  CHECK(device.trx_read_func(&device, &timestamp, NULL, 1, 1) == 1);
  CHECK(radio.inline_stop_result == 0 && radio.inline_shutdown_returned);
  CHECK(radio.stop_calls == 0 && radio.end_calls == 0);

  CHECK(device.trx_stop_func(&device) == 0);
  CHECK(radio.stop_calls == 1 && radio.end_calls == 0);
  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_failed_retune_invalidates(void)
{
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_ACQUISITION,
      .rx_acquisition = AGC_RX_ACQUISITION_LEGACY,
      .rx_actuation = true,
  };
  const radio_gain_api_t api = fake_api();

  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  radio.fail_retune = true;
  config.rx_freq[0] = 3520000000.0;
  config.tx_freq[0] = 3620000000.0;
  CHECK(device.trx_set_freq_func(&device, &config) == -EIO);
  CHECK(radio.retune_calls == 1);
  double applied = 0.0, reported = 0;
  CHECK(radio_gain_device_adjust_rx(&device, 1.0, &applied, &reported) == -EAGAIN);
  CHECK(applied == 0.0);
  device.trx_end_func(&device);
  CHECK(radio.end_calls == 1);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

static int test_managed_tx_profile_and_fault(void)
{
  CHECK(signal(SIGTERM, SIG_IGN) != SIG_ERR); /* admit worker notification without terminating this fixture */
  fake_radio_t radio;
  openair0_device_t device;
  openair0_config_t config;
  fake_init(&radio);
  make_device(&device, &config, &radio);
  test_options = (agc_options_t){
      .mode = AGC_MODE_CONTINUOUS,
      .directions = AGC_DIRECTIONS_TX,
      .tx_policy = AGC_TX_POLICY_MANAGED,
      .tx_actuation = true,
      .tx_profile = {.power = {.qualified = true, .reference_dbm = 0, .uncertainty_db = 0.5, .minimum_dbm = -40, .maximum_dbm = 0},
                     .id = "fake-profile",
                     .identity = "fake-radio",
                     .antenna = "RF0",
                     .provenance = "synthetic-test",
                     .minimum_frequency_hz = 3599999999,
                     .maximum_frequency_hz = 3600000001,
                     .sample_rate_hz = 30720000,
                     .bandwidth_hz = 20000000,
                     .reported_gain_db = 11,
                     .component_full_scale = 2048,
                     .peak_limit_fs = 0.9,
                     .maximum_quantization_error_db = 0.01,
                     .maximum_quantization_evm = 0.03}};
  const radio_gain_api_t api = fake_api();
  CHECK(radio_gain_device_attach(&device, &config, &api) == 0);
  recorder_reset(true);
  c16_t samples[] = {{1024, 0}, {-1024, 0}};
  void *buffers[] = {samples};
  CHECK(!radio_gain_device_tx_cancelled(-ERANGE));
  CHECK(!radio_gain_device_tx_cancelled(-ESHUTDOWN));
  CHECK(radio_gain_device_apply_tx(samples, 2, -12.041199826559248, 1, 2, 2));
  CHECK(samples[0].r == 512 && samples[1].r == -512);
  CHECK(recorder_event_count(FLIGHT_EVENT_RADIO_TX_POWER) == 1);
  CHECK(device.trx_write_func(&device, 200, buffers, 2, 1, 0) == 2);
  CHECK(radio.write_calls == 1 && radio.set_calls == 0);
  CHECK(!radio_gain_device_apply_tx(samples, 2, 10, 1, 3, 2));
  CHECK(radio_gain_device_tx_cancelled(-ERANGE));
  CHECK(radio_gain_device_tx_cancelled(-ESHUTDOWN));
  CHECK(!radio_gain_device_tx_cancelled(-EIO));
  CHECK(!radio_gain_device_tx_cancelled(0));
  CHECK(!radio_gain_device_tx_cancelled(1));
  const unsigned int records_before_cancel = test_recorder_count;
  for (int slot = 0; slot < 10; ++slot)
    CHECK(!radio_gain_device_validate_ue_power_limit(0, INT_MIN, 2, slot));
  CHECK(test_recorder_count == records_before_cancel);
  CHECK(samples[0].r == 512 && samples[1].r == -512);
  CHECK(device.trx_write_func(&device, 202, buffers, 2, 1, 0) == -ERANGE);
  CHECK(radio.write_calls == 1);
  device.trx_end_func(&device);
  fake_fini(&radio);
  return EXIT_SUCCESS;
}

int main(void)
{
  memset(&test_log, 0, sizeof(test_log));
  for (unsigned int index = 0; index < MAX_LOG_COMPONENTS; ++index)
    test_log.log_component[index].level = OAILOG_DISABLE;

  CHECK(run_child(test_managed_tx_profile_and_fault) == EXIT_SUCCESS);
  CHECK(run_child(test_continuous_rejected) == EXIT_SUCCESS);
  CHECK(run_child(test_continuous_legacy_handoff) == EXIT_SUCCESS);
  CHECK(run_child(test_continuous_new_acquisition_shared_cooldown) == EXIT_SUCCESS);
  CHECK(run_child(test_continuous_gnb_tracking) == EXIT_SUCCESS);
  CHECK(run_child(test_shared_peak_envelope_binding) == EXIT_SUCCESS);
  CHECK(run_child(test_observe_no_gain_write) == EXIT_SUCCESS);
  CHECK(run_child(test_tx_level_records_and_disabled_path) == EXIT_SUCCESS);
  CHECK(run_child(test_observe_policy_candidates) == EXIT_SUCCESS);
  CHECK(run_child(test_off_and_acquisition_policy_gates) == EXIT_SUCCESS);
  CHECK(run_child(test_raw_read_summary_and_default_settle_guard) == EXIT_SUCCESS);
  CHECK(run_child(test_inline_callback_shutdown_deferred) == EXIT_SUCCESS);
  CHECK(run_child(test_failed_retune_invalidates) == EXIT_SUCCESS);
  CHECK(test_acquisition_retune_and_tx_fence() == EXIT_SUCCESS);
  puts("radio gain device glue tests passed");
  return EXIT_SUCCESS;
}
