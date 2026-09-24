/* SPDX-License-Identifier: LicenseRef-CSSL-1.0 */
#include "radio_gain_samples.h"
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#define CHECK(x)                                              \
  do {                                                        \
    if (!(x)) {                                               \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); \
      return EXIT_FAILURE;                                    \
    }                                                         \
  } while (0)

enum { concurrent_generations = 1024 };

static radio_gain_sample_context_t level_context(uint64_t generation,
                                                 int64_t first_sample,
                                                 int64_t end_sample,
                                                 double mean_power_fs,
                                                 double peak_component_fs,
                                                 uint32_t sampled_components,
                                                 uint32_t near_rail_components,
                                                 uint64_t observation_ns)
{
  return (radio_gain_sample_context_t){
      .present = true,
      .valid = true,
      .generation = generation,
      .rx_gain_db = 40.25,
      .first_sample = first_sample,
      .end_sample = end_sample,
      .level_valid = true,
      .mean_power_fs = mean_power_fs,
      .peak_component_fs = peak_component_fs,
      .sampled_components = sampled_components,
      .near_rail_components = near_rail_components,
      .observation_ns = observation_ns,
  };
}

typedef struct {
  radio_gain_sample_history_t *history;
  _Atomic(uint64_t) published_generation;
  _Atomic(bool) producer_done;
  _Atomic(bool) failed;
  _Atomic(unsigned int) valid_observations;
} concurrent_history_test_t;

static void *publish_homogeneous_ranges(void *opaque)
{
  concurrent_history_test_t *test = opaque;
  for (uint64_t generation = 1; generation <= concurrent_generations; ++generation) {
    const int64_t first = (int64_t)generation * 200;
    const radio_gain_sample_context_t first_half = {
        .present = true,
        .valid = true,
        .generation = generation,
        .rx_gain_db = (double)generation + 0.25,
        .first_sample = first,
        .end_sample = first + 100,
        .level_valid = true,
        .mean_power_fs = 0.2,
        .peak_component_fs = 0.5,
        .sampled_components = 100,
        .near_rail_components = 0,
        .observation_ns = generation * 10 + 1,
    };
    radio_gain_sample_context_t second_half = first_half;
    second_half.first_sample += 100;
    second_half.end_sample += 100;
    second_half.mean_power_fs = 0.1;
    second_half.peak_component_fs = 0.99;
    second_half.sampled_components = 200;
    second_half.near_rail_components = 1;
    second_half.observation_ns = generation * 10 + 2;
    radio_gain_sample_publish(test->history, &first_half);
    radio_gain_sample_publish(test->history, &second_half);
    atomic_store_explicit(&test->published_generation, generation, memory_order_release);
  }
  atomic_store_explicit(&test->producer_done, true, memory_order_release);
  return NULL;
}

static void *read_homogeneous_ranges(void *opaque)
{
  concurrent_history_test_t *test = opaque;
  unsigned int drain_attempts = 0;
  do {
    const uint64_t generation = atomic_load_explicit(&test->published_generation, memory_order_acquire);
    if (generation != 0) {
      const int64_t first = (int64_t)generation * 200;
      const radio_gain_sample_context_t result = radio_gain_sample_lookup(test->history, first, first + 200);
      if (result.valid) {
        if (result.generation != generation || result.rx_gain_db != (double)generation + 0.25 || result.first_sample != first
            || result.end_sample != first + 200 || !result.level_valid || fabs(result.mean_power_fs - (2.0 / 15.0)) > 1e-12
            || result.peak_component_fs != 0.99 || result.sampled_components != 300 || result.near_rail_components != 1
            || result.observation_ns != generation * 10 + 2) {
          atomic_store_explicit(&test->failed, true, memory_order_release);
          return NULL;
        }
        atomic_fetch_add_explicit(&test->valid_observations, 1, memory_order_relaxed);
      }
    }
    if (atomic_load_explicit(&test->producer_done, memory_order_acquire))
      ++drain_attempts;
  } while (!atomic_load_explicit(&test->producer_done, memory_order_acquire) || drain_attempts < 512);
  return NULL;
}

static bool test_concurrent_homogeneous_ranges(void)
{
  concurrent_history_test_t test = {.history = radio_gain_sample_history_create()};
  if (!test.history)
    return false;
  atomic_init(&test.published_generation, 0);
  atomic_init(&test.producer_done, false);
  atomic_init(&test.failed, false);
  atomic_init(&test.valid_observations, 0);
  pthread_t producer;
  pthread_t reader;
  const int producer_created = pthread_create(&producer, NULL, publish_homogeneous_ranges, &test);
  const int reader_created = producer_created == 0 ? pthread_create(&reader, NULL, read_homogeneous_ranges, &test) : -1;
  if (producer_created != 0 || reader_created != 0) {
    if (producer_created == 0)
      (void)pthread_join(producer, NULL);
    radio_gain_sample_history_destroy(test.history);
    return false;
  }
  const int producer_joined = pthread_join(producer, NULL);
  const int reader_joined = pthread_join(reader, NULL);
  const bool success = producer_joined == 0 && reader_joined == 0 && !atomic_load_explicit(&test.failed, memory_order_acquire)
                       && atomic_load_explicit(&test.valid_observations, memory_order_acquire) != 0;
  radio_gain_sample_history_destroy(test.history);
  return success;
}
static bool test_measurement_qualification(void)
{
  radio_gain_sample_context_t context = level_context(1, 0, 100, 0.25, 0.5, 200, 0, 1);
  if (!radio_gain_sample_measurement_valid(&context) || !context.valid)
    return false;

  context.level_valid = false;
  if (radio_gain_sample_measurement_valid(&context) || !context.valid)
    return false;

  context = level_context(1, 0, 100, 0.25, 0.99, 200, 1, 1);
  if (radio_gain_sample_measurement_valid(&context) || !context.valid)
    return false;

  context = level_context(1, 0, 100, 0.25, 0.5, 200, 0, 1);
  context.rx_gain_db = NAN;
  if (radio_gain_sample_measurement_valid(&context) || !context.valid)
    return false;

  context = level_context(1, 0, 100, 0.25, 0.5, 200, 0, 1);
  context.present = false;
  return !radio_gain_sample_measurement_valid(&context) && context.valid;
}

static bool test_level_summary_aggregation(void)
{
  radio_gain_sample_history_t *history = radio_gain_sample_history_create();
  if (!history)
    return false;

  radio_gain_sample_context_t first = level_context(7, 0, 100, 0.5, 0.6, 200, 0, 100);
  radio_gain_sample_context_t second = level_context(7, 100, 200, 0.25, 0.99, 100, 1, 200);
  radio_gain_sample_publish(history, &first);
  radio_gain_sample_publish(history, &second);
  radio_gain_sample_context_t result = radio_gain_sample_lookup(history, 80, 160);
  if (!result.valid || !result.level_valid || radio_gain_sample_measurement_valid(&result) || result.generation != 7
      || fabs(result.mean_power_fs - (5.0 / 12.0)) > 1e-12 || result.peak_component_fs != 0.99 || result.sampled_components != 300
      || result.near_rail_components != 1 || result.observation_ns != 200) {
    radio_gain_sample_history_destroy(history);
    return false;
  }

  first = level_context(8, 200, 300, 0.2, 0.5, 100, 0, 300);
  second = level_context(8, 300, 400, NAN, 0.5, 100, 0, 400);
  radio_gain_sample_publish(history, &first);
  radio_gain_sample_publish(history, &second);
  result = radio_gain_sample_lookup(history, 250, 350);
  if (!result.valid || result.generation != 8 || result.level_valid) {
    radio_gain_sample_history_destroy(history);
    return false;
  }

  first = level_context(9, 400, 500, 0.1, 0.5, UINT32_MAX - 1U, 0, 500);
  second = level_context(9, 500, 600, 0.1, 0.5, UINT32_MAX - 1U, 0, 600);
  radio_gain_sample_publish(history, &first);
  radio_gain_sample_publish(history, &second);
  result = radio_gain_sample_lookup(history, 450, 550);
  const bool success = result.valid && result.generation == 9 && !result.level_valid && result.sampled_components == 0
                       && result.near_rail_components == 0;
  radio_gain_sample_history_destroy(history);
  return success;
}

int main(void)
{
  const int16_t range_samples[] = {-2048, 2047, 2048, -2049};
  radio_tx_sample_level_t tx = radio_tx_sample_level(range_samples, 2, 2048);
  CHECK(tx.valid && tx.sample_count == 2 && tx.sum_squared_components == UINT64_C(16777218) && tx.peak_component == 2049
        && tx.over_range_components == 2);
  const int16_t integer_limits[] = {INT16_MIN, INT16_MAX};
  tx = radio_tx_sample_level(integer_limits, 1, 32768);
  CHECK(tx.valid && tx.sum_squared_components == UINT64_C(2147418113) && tx.peak_component == 32768
        && tx.over_range_components == 0);
  CHECK(!radio_tx_sample_level(NULL, 1, 2048).valid);
  CHECK(!radio_tx_sample_level(range_samples, 0, 2048).valid);
  CHECK(!radio_tx_sample_level(range_samples, RADIO_TX_LEVEL_MAX_SAMPLES + 1, 2048).valid);
  CHECK(!radio_tx_sample_level(range_samples, 1, 32769).valid);
  static int16_t maximum_buffer[RADIO_TX_LEVEL_MAX_SAMPLES * 2];
  for (unsigned int i = 0; i < RADIO_TX_LEVEL_MAX_SAMPLES * 2; ++i)
    maximum_buffer[i] = INT16_MIN;
  tx = radio_tx_sample_level(maximum_buffer, RADIO_TX_LEVEL_MAX_SAMPLES, 2048);
  CHECK(tx.valid && tx.sum_squared_components == (UINT64_C(1) << 47) && tx.over_range_components == RADIO_TX_LEVEL_MAX_SAMPLES * 2);

  radio_gain_sample_history_t *history = radio_gain_sample_history_create();
  CHECK(history);
  const radio_gain_sample_context_t absent = radio_gain_sample_lookup(history, 0, 10);
  CHECK(!absent.valid && !absent.level_valid);
  radio_gain_result_t before = {.generation = 1, .rx_gain_valid = true, .reported_rx_db = 40.25};
  radio_gain_result_t after = before;
  radio_gain_sample_context_t a = radio_gain_sample_context(&before, &after, 0, 100, true, 0);
  a.level_valid = true;
  a.mean_power_fs = 0.25;
  a.peak_component_fs = 0.5;
  a.sampled_components = 200;
  a.near_rail_components = 0;
  a.observation_ns = 1;
  CHECK(a.present && a.valid && a.rx_gain_db == 40.25);
  radio_gain_sample_publish(history, &a);
  a.first_sample = 100;
  a.end_sample = 200;
  radio_gain_sample_publish(history, &a);
  a = radio_gain_sample_lookup(history, 80, 160);
  CHECK(a.valid && a.generation == 1 && a.rx_gain_db == 40.25); // lookahead spans two reads
  CHECK(!radio_gain_sample_lookup(history, -1, 160).valid);
  CHECK(!radio_gain_sample_lookup(history, 80, 201).valid);
  after.generation = 2;
  after.reported_rx_db = 43.25;
  a = radio_gain_sample_context(&before, &after, 200, 300, true, 0);
  CHECK(!a.valid);
  radio_gain_sample_publish(history, &a);
  CHECK(!radio_gain_sample_lookup(history, 150, 250).valid);
  before = after;
  a = radio_gain_sample_context(&before, &after, 300, 400, false, 0);
  CHECK(!a.valid); // readback does not prove analog settling
  a = radio_gain_sample_context(&before, &after, 300, 400, true, 350);
  CHECK(!a.valid); // a partly unsettled read cannot normalize a whole slot
  a = radio_gain_sample_context(&before, &after, 300, 400, true, 300);
  CHECK(a.valid);
  a.level_valid = true;
  a.mean_power_fs = 0.25;
  a.peak_component_fs = 0.5;
  a.sampled_components = 200;
  a.near_rail_components = 0;
  a.observation_ns = 2;
  radio_gain_sample_publish(history, &a);
  CHECK(radio_gain_sample_lookup(history, 300, 350).valid);
  a.first_sample = 450;
  a.end_sample = 550;
  radio_gain_sample_publish(history, &a);
  CHECK(!radio_gain_sample_lookup(history, 350, 500).valid); // timestamp gap
  for (unsigned i = 0; i < RADIO_GAIN_SAMPLE_HISTORY + 2; ++i) {
    a.first_sample = 1000 + i * 100;
    a.end_sample = a.first_sample + 100;
    radio_gain_sample_publish(history, &a);
  }
  CHECK(!radio_gain_sample_lookup(history, 80, 160).valid); // delayed actor lost history
  a = radio_gain_sample_lookup(history, a.first_sample, a.end_sample);
  CHECK(a.valid && a.level_valid); // overwritten history still retains a coherent newest summary
  radio_gain_sample_history_destroy(history);
  CHECK(test_measurement_qualification());
  CHECK(test_level_summary_aggregation());
  CHECK(test_concurrent_homogeneous_ranges());
  puts("radio gain sample-history tests passed");
  return 0;
}
