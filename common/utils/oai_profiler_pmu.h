/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#ifndef OAI_PROFILER_PMU_H
#define OAI_PROFILER_PMU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OAI_PROFILE_PMU_MAX_EVENTS 24U

typedef enum {
  OAI_PROFILE_PMU_OFF = 0,
  OAI_PROFILE_PMU_AUTO,
  OAI_PROFILE_PMU_SOFTWARE,
  OAI_PROFILE_PMU_HARDWARE,
  OAI_PROFILE_PMU_ALL,
} oai_profile_pmu_mode_t;

typedef struct oai_profile_pmu_state_s oai_profile_pmu_state_t;

typedef struct {
  uint16_t event_id;
  const char *name;
  const char *domain;
  const char *unit;
  uint32_t type;
  uint64_t config;
  uint8_t group_id;
} oai_profile_pmu_descriptor_t;

typedef struct {
  uint16_t event_id;
  bool requested;
  bool available;
  int error_code;
  const char *status;
} oai_profile_pmu_availability_t;

typedef struct {
  uint16_t event_id;
  uint64_t raw_value;
  uint64_t delta_raw;
  uint64_t time_enabled_ns;
  uint64_t time_running_ns;
  uint64_t delta_enabled_ns;
  uint64_t delta_running_ns;
  uint64_t interval_ns;
  double scaled_value;
  double delta_scaled;
  double multiplex_ratio;
  bool delta_valid;
  bool scaling_valid;
  int error_code;
  const char *status;
} oai_profile_pmu_observation_t;

typedef struct {
  size_t observation_count;
  uint16_t group_reads;
  uint16_t read_errors;
} oai_profile_pmu_collect_result_t;

oai_profile_pmu_mode_t oai_profile_pmu_parse_mode(const char *value);
const char *oai_profile_pmu_mode_name(oai_profile_pmu_mode_t mode);
size_t oai_profile_pmu_descriptor_count(void);
const oai_profile_pmu_descriptor_t *oai_profile_pmu_descriptor(size_t index);

oai_profile_pmu_state_t *oai_profile_pmu_open(pid_t tid, oai_profile_pmu_mode_t mode);
void oai_profile_pmu_close(oai_profile_pmu_state_t *state);
size_t oai_profile_pmu_get_availability(const oai_profile_pmu_state_t *state,
                                        oai_profile_pmu_availability_t *availability,
                                        size_t capacity);
oai_profile_pmu_collect_result_t oai_profile_pmu_collect(oai_profile_pmu_state_t *state,
                                                         uint64_t monotonic_raw_ns,
                                                         oai_profile_pmu_observation_t *observations,
                                                         size_t capacity);
size_t oai_profile_pmu_available_event_count(const oai_profile_pmu_state_t *state);
size_t oai_profile_pmu_active_group_count(const oai_profile_pmu_state_t *state);

#ifdef __cplusplus
}
#endif

#endif /* OAI_PROFILER_PMU_H */
