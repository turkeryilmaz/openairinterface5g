/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "radio_health.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                               \
  do {                                                                                 \
    if (!(condition)) {                                                                \
      fprintf(stderr, "check failed at %s:%d: %s\\n", __FILE__, __LINE__, #condition); \
      return EXIT_FAILURE;                                                             \
    }                                                                                  \
  } while (0)

enum { producer_iterations = 100000 };

typedef struct {
  radio_health_device_t *device;
} producer_args_t;

static void *produce_tx_calls(void *opaque)
{
  producer_args_t *args = opaque;
  for (unsigned int i = 0; i < producer_iterations; ++i)
    if (!radio_health_counter_add(args->device, RADIO_HEALTH_METRIC_TX_SEND_CALLS, 1))
      return (void *)1;
  return NULL;
}

static void *produce_tx_metadata(void *opaque)
{
  producer_args_t *args = opaque;
  for (uint64_t raw_code = 1000; raw_code < 11000; ++raw_code)
    radio_health_observe_tx_async(args->device, raw_code + 1, raw_code, true, raw_code + 2, true, raw_code + 3);
  return NULL;
}

int main(void)
{
  radio_health_snapshot_t snapshot;
  const uint64_t tx_async_tuple_metrics =
      RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS)
      | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE)
      | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL)
      | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID)
      | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS)
      | RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID);
  radio_health_set_enabled(false);
  CHECK(radio_health_register(RADIO_HEALTH_BACKEND_UHD, 1, RADIO_HEALTH_CAP_TX_SEND) == NULL);
  CHECK(radio_health_device_count() == 0);

  radio_health_set_enabled(true);
  for (radio_health_metric_t metric = 0; metric < RADIO_HEALTH_METRIC_COUNT; ++metric) {
    CHECK(radio_health_metric_name(metric) != NULL);
    CHECK(radio_health_metric_capabilities(metric) != 0);
  }
  CHECK(radio_health_metric_name(RADIO_HEALTH_METRIC_COUNT) == NULL);
  CHECK(radio_health_metric_capabilities(RADIO_HEALTH_METRIC_COUNT) == 0);

  uint64_t rate_microhz = UINT64_C(123456789);
  CHECK(radio_health_rate_microhz(7680000.0, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(7680000000000));
  CHECK(radio_health_rate_microhz(7680000.0059838388, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(7680000005984));
  rate_microhz = UINT64_C(123456789);
  CHECK(!radio_health_rate_microhz(0.0, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(123456789));
  CHECK(!radio_health_rate_microhz(-1.0, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(123456789));
  CHECK(!radio_health_rate_microhz(NAN, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(123456789));
  CHECK(!radio_health_rate_microhz(INFINITY, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(123456789));
  CHECK(!radio_health_rate_microhz(18446744073710.0, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(123456789));
  CHECK(!radio_health_rate_microhz(0x1p-80, &rate_microhz));
  CHECK(rate_microhz == UINT64_C(123456789));
  CHECK(!radio_health_rate_microhz(7680000.0, NULL));

  radio_health_device_t *tx =
      radio_health_register(RADIO_HEALTH_BACKEND_UHD, 17, RADIO_HEALTH_CAP_TX_SEND | RADIO_HEALTH_CAP_TX_ASYNC);
  CHECK(tx != NULL);
  CHECK(radio_health_device_count() == 1);
  CHECK(radio_health_snapshot(0, &snapshot));
  CHECK(snapshot.device_id == 0);
  CHECK(snapshot.backend == RADIO_HEALTH_BACKEND_UHD);
  CHECK(snapshot.device_type == 17);
  CHECK(snapshot.lifecycle == RADIO_HEALTH_LIFECYCLE_ACTIVE);
  CHECK(radio_health_metric_supported(snapshot.capabilities, RADIO_HEALTH_METRIC_TX_SEND_CALLS));
  CHECK(!radio_health_metric_supported(snapshot.capabilities, RADIO_HEALTH_METRIC_RX_RECV_CALLS));
  CHECK((snapshot.observed_metrics & RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_SEND_CALLS)) != 0);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_SEND_CALLS] == 0);
  CHECK((snapshot.observed_metrics & RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_ASYNC_LAST_POLL_MONO_NS)) == 0);
  CHECK((snapshot.observed_metrics & tx_async_tuple_metrics) == 0);
  CHECK(radio_health_gauge_set(tx, RADIO_HEALTH_METRIC_TX_SEND_INFLIGHT, 0));
  CHECK(radio_health_snapshot(0, &snapshot));
  CHECK((snapshot.observed_metrics & RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_SEND_INFLIGHT)) != 0);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_SEND_INFLIGHT] == 0);
  CHECK(!radio_health_counter_add(tx, RADIO_HEALTH_METRIC_RX_RECV_CALLS, 1));
  CHECK(!radio_health_gauge_set(tx, RADIO_HEALTH_METRIC_TX_SEND_CALLS, 1));

  radio_health_observe_tx_async(tx, 100, 0x8, true, 2, true, 1234);
  CHECK(radio_health_snapshot(0, &snapshot));
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS] == 100);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE] == 0x8);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID] == 1);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID] == 1);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS] == 1234);

  radio_health_observe_tx_async(tx, 200, 0x2, false, 0, false, 9999);
  CHECK(radio_health_snapshot(0, &snapshot));
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS] == 200);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL_VALID] == 0);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID] == 0);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS] == 9999);

  radio_health_device_t *rx_queue =
      radio_health_register(RADIO_HEALTH_BACKEND_UHD, 23, RADIO_HEALTH_CAP_RX_STREAM | RADIO_HEALTH_CAP_TX_QUEUE);
  CHECK(rx_queue != NULL);
  CHECK(radio_health_snapshot(1, &snapshot));
  CHECK(snapshot.device_id == 1);
  CHECK(snapshot.device_type == 23);
  CHECK((snapshot.observed_metrics & RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_RX_RECV_CALLS)) != 0);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_RX_RECV_CALLS] == 0);
  CHECK((snapshot.observed_metrics & RADIO_HEALTH_METRIC_BIT(RADIO_HEALTH_METRIC_TX_QUEUE_DEPTH)) == 0);

  radio_health_observe_rx_metadata(rx_queue, 0x8, true, 666);
  CHECK(radio_health_gauge_set(rx_queue, RADIO_HEALTH_METRIC_TX_QUEUE_DEPTH, 2));
  CHECK(radio_health_counter_add(rx_queue, RADIO_HEALTH_METRIC_TX_QUEUE_ENQUEUES, 2));
  CHECK(radio_health_snapshot(1, &snapshot));
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_RX_LAST_ERROR_RAW_CODE] == 0x8);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TIME_VALID] == 1);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_RX_LAST_DEVICE_TICKS] == 666);
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_QUEUE_DEPTH] == 2);

  producer_args_t args = {.device = tx};
  pthread_t metadata_producer;
  unsigned int available_metadata_snapshots = 0;
  unsigned int unavailable_metadata_snapshots = 0;
  CHECK(pthread_create(&metadata_producer, NULL, produce_tx_metadata, &args) == 0);
  for (unsigned int i = 0; i < 1000; ++i) {
    CHECK(radio_health_snapshot(0, &snapshot));
    if ((snapshot.observed_metrics & tx_async_tuple_metrics) == tx_async_tuple_metrics) {
      ++available_metadata_snapshots;
      if (snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TIME_VALID] == 1) {
        const uint64_t raw_code = snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_RAW_CODE];
        CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_MONO_NS] == raw_code + 1);
        CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_CHANNEL] == raw_code + 2);
        CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_ASYNC_LAST_EVENT_DEVICE_TICKS] == raw_code + 3);
      }
    } else {
      ++unavailable_metadata_snapshots;
    }
  }
  CHECK(available_metadata_snapshots + unavailable_metadata_snapshots == 1000);
  void *metadata_result = NULL;
  CHECK(pthread_join(metadata_producer, &metadata_result) == 0);
  CHECK(metadata_result == NULL);

  pthread_t first;
  pthread_t second;
  CHECK(pthread_create(&first, NULL, produce_tx_calls, &args) == 0);
  CHECK(pthread_create(&second, NULL, produce_tx_calls, &args) == 0);
  void *first_result = NULL;
  void *second_result = NULL;
  CHECK(pthread_join(first, &first_result) == 0);
  CHECK(pthread_join(second, &second_result) == 0);
  CHECK(first_result == NULL);
  CHECK(second_result == NULL);
  for (unsigned int i = 0; i < 1000; ++i)
    CHECK(radio_health_snapshot(0, &snapshot));
  CHECK(radio_health_snapshot(0, &snapshot));
  CHECK(snapshot.values[RADIO_HEALTH_METRIC_TX_SEND_CALLS] == 2 * producer_iterations);

  radio_health_close(tx);
  CHECK(!radio_health_counter_add(tx, RADIO_HEALTH_METRIC_TX_SEND_CALLS, 1));
  CHECK(radio_health_snapshot(0, &snapshot));
  CHECK(snapshot.lifecycle == RADIO_HEALTH_LIFECYCLE_CLOSED);

  radio_health_device_t *third = radio_health_register(RADIO_HEALTH_BACKEND_UHD, 31, RADIO_HEALTH_CAP_TX_SEND);
  radio_health_device_t *fourth = radio_health_register(RADIO_HEALTH_BACKEND_UHD, 37, RADIO_HEALTH_CAP_TX_SEND);
  CHECK(third != NULL);
  CHECK(fourth != NULL);
  CHECK(radio_health_device_count() == RADIO_HEALTH_MAX_DEVICES);
  CHECK(radio_health_snapshot(2, &snapshot));
  CHECK(snapshot.device_type == 31);
  CHECK(radio_health_register(RADIO_HEALTH_BACKEND_UHD, 41, RADIO_HEALTH_CAP_TX_SEND) == NULL);
  CHECK(radio_health_device_count() == RADIO_HEALTH_MAX_DEVICES);

  radio_health_close(rx_queue);
  radio_health_close(third);
  radio_health_close(fourth);
  return EXIT_SUCCESS;
}
