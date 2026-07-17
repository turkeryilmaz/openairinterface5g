/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include <assert.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include "common/utils/oai_profiler_pmu.h"

static uint64_t monotonic_raw_ns(void)
{
  struct timespec now = {0};
  assert(clock_gettime(CLOCK_MONOTONIC_RAW, &now) == 0);
  return (uint64_t)now.tv_sec * 1000000000ULL + (uint64_t)now.tv_nsec;
}

static void exercise_cpu(void)
{
  volatile uint64_t value = 1;
  for (uint64_t i = 1; i < 1000000; i++)
    value = value * 33U + i;
  assert(value != 0);
}

int main(void)
{
  const size_t descriptor_count = oai_profile_pmu_descriptor_count();
  assert(descriptor_count > 0);
  assert(descriptor_count <= OAI_PROFILE_PMU_MAX_EVENTS);
  uint16_t previous_id = 0;
  size_t software_event_count = 0;
  for (size_t i = 0; i < descriptor_count; i++) {
    const oai_profile_pmu_descriptor_t *descriptor = oai_profile_pmu_descriptor(i);
    assert(descriptor != NULL);
    assert(descriptor->event_id > previous_id);
    assert(descriptor->name != NULL && descriptor->name[0] != '\0');
    assert(descriptor->domain != NULL && descriptor->domain[0] != '\0');
    assert(descriptor->unit != NULL && descriptor->unit[0] != '\0');
    previous_id = descriptor->event_id;
    software_event_count += strcmp(descriptor->domain, "software") == 0;
  }
  assert(oai_profile_pmu_descriptor(descriptor_count) == NULL);

  assert(oai_profile_pmu_parse_mode(NULL) == OAI_PROFILE_PMU_AUTO);
  assert(oai_profile_pmu_parse_mode("off") == OAI_PROFILE_PMU_OFF);
  assert(oai_profile_pmu_parse_mode("sw") == OAI_PROFILE_PMU_SOFTWARE);
  assert(oai_profile_pmu_parse_mode("hardware") == OAI_PROFILE_PMU_HARDWARE);
  assert(oai_profile_pmu_parse_mode("all") == OAI_PROFILE_PMU_ALL);
  assert(strcmp(oai_profile_pmu_mode_name(OAI_PROFILE_PMU_AUTO), "auto") == 0);
  assert(oai_profile_pmu_open((pid_t)syscall(SYS_gettid), OAI_PROFILE_PMU_OFF) == NULL);

  oai_profile_pmu_state_t *state = oai_profile_pmu_open((pid_t)syscall(SYS_gettid), OAI_PROFILE_PMU_SOFTWARE);
  assert(state != NULL);
  oai_profile_pmu_availability_t availability[OAI_PROFILE_PMU_MAX_EVENTS];
  const size_t availability_count = oai_profile_pmu_get_availability(state, availability, OAI_PROFILE_PMU_MAX_EVENTS);
  assert(availability_count == descriptor_count);
  size_t requested = 0;
  size_t available = 0;
  for (size_t i = 0; i < availability_count; i++) {
    const oai_profile_pmu_descriptor_t *descriptor = oai_profile_pmu_descriptor(i);
    const bool is_software = strcmp(descriptor->domain, "software") == 0;
    assert(availability[i].event_id == descriptor->event_id);
    assert(availability[i].requested == is_software);
    assert(availability[i].status != NULL && availability[i].status[0] != '\0');
    requested += availability[i].requested;
    available += availability[i].available;
    if (!availability[i].requested)
      assert(strcmp(availability[i].status, "not_requested") == 0);
  }
  assert(requested == software_event_count);
  assert(available == oai_profile_pmu_available_event_count(state));
  assert(oai_profile_pmu_active_group_count(state) <= 1);

  oai_profile_pmu_observation_t observations[OAI_PROFILE_PMU_MAX_EVENTS];
  const uint64_t sample_time_ns = monotonic_raw_ns();
  const oai_profile_pmu_collect_result_t first =
      oai_profile_pmu_collect(state, sample_time_ns, observations, OAI_PROFILE_PMU_MAX_EVENTS);
  assert(first.observation_count <= available);
  exercise_cpu();
  usleep(10000);
  const oai_profile_pmu_collect_result_t second =
      oai_profile_pmu_collect(state, sample_time_ns, observations, OAI_PROFILE_PMU_MAX_EVENTS);
  assert(second.observation_count <= available);
  for (size_t i = 0; i < second.observation_count; i++) {
    assert(observations[i].event_id > 0);
    assert(observations[i].status != NULL && observations[i].status[0] != '\0');
    if (observations[i].delta_valid)
      assert(observations[i].interval_ns > 0);
  }
  if (available > 0 && first.read_errors == 0 && second.read_errors == 0) {
    assert(second.observation_count == available);
    for (size_t i = 0; i < second.observation_count; i++) {
      assert(strcmp(observations[i].status, "clock_regression") == 0);
      assert(!observations[i].delta_valid);
      assert(observations[i].interval_ns == 0);
    }
  }

  oai_profile_pmu_close(state);
  return 0;
}
