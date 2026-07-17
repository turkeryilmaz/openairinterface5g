/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "oai_profiler_pmu.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#define OAI_PROFILE_PMU_MAX_GROUPS 5U
#define OAI_PROFILE_PMU_MAX_GROUP_MEMBERS 8U

#if defined(__linux__)
#define HW_CACHE_CONFIG(cache, operation, result) ((uint64_t)(cache) | ((uint64_t)(operation) << 8U) | ((uint64_t)(result) << 16U))

static const oai_profile_pmu_descriptor_t pmu_descriptors[] = {
    {1, "cpu_cycles", "hardware", "count", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, 0},
    {2, "instructions", "hardware", "count", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, 0},
    {3, "branches", "hardware", "count", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS, 0},
    {4, "branch_misses", "hardware", "count", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, 0},
    {5, "cache_references", "hardware", "count", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES, 1},
    {6, "cache_misses", "hardware", "count", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, 1},
    {7, "stalled_cycles_frontend", "hardware", "cycle", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND, 1},
    {8, "stalled_cycles_backend", "hardware", "cycle", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND, 1},
    {9,
     "l1d_read_accesses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
     2},
    {10,
     "l1d_read_misses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
     2},
    {11,
     "llc_read_accesses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
     2},
    {12,
     "llc_read_misses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
     2},
    {13,
     "dtlb_read_accesses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
     3},
    {14,
     "dtlb_read_misses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_DTLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
     3},
    {15,
     "itlb_read_accesses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_ITLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_ACCESS),
     3},
    {16,
     "itlb_read_misses",
     "hardware_cache",
     "count",
     PERF_TYPE_HW_CACHE,
     HW_CACHE_CONFIG(PERF_COUNT_HW_CACHE_ITLB, PERF_COUNT_HW_CACHE_OP_READ, PERF_COUNT_HW_CACHE_RESULT_MISS),
     3},
    {17, "task_clock", "software", "nanosecond", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, 4},
    {18, "context_switches", "software", "count", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, 4},
    {19, "cpu_migrations", "software", "count", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS, 4},
    {20, "page_faults", "software", "count", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, 4},
    {21, "minor_faults", "software", "count", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN, 4},
    {22, "major_faults", "software", "count", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ, 4},
};
#else
static const oai_profile_pmu_descriptor_t pmu_descriptors[] = {
    {1, "cpu_cycles", "hardware", "count", 0, 0, 0},
    {2, "instructions", "hardware", "count", 0, 0, 0},
    {3, "branches", "hardware", "count", 0, 0, 0},
    {4, "branch_misses", "hardware", "count", 0, 0, 0},
    {5, "cache_references", "hardware", "count", 0, 0, 1},
    {6, "cache_misses", "hardware", "count", 0, 0, 1},
    {7, "stalled_cycles_frontend", "hardware", "cycle", 0, 0, 1},
    {8, "stalled_cycles_backend", "hardware", "cycle", 0, 0, 1},
    {9, "l1d_read_accesses", "hardware_cache", "count", 0, 0, 2},
    {10, "l1d_read_misses", "hardware_cache", "count", 0, 0, 2},
    {11, "llc_read_accesses", "hardware_cache", "count", 0, 0, 2},
    {12, "llc_read_misses", "hardware_cache", "count", 0, 0, 2},
    {13, "dtlb_read_accesses", "hardware_cache", "count", 0, 0, 3},
    {14, "dtlb_read_misses", "hardware_cache", "count", 0, 0, 3},
    {15, "itlb_read_accesses", "hardware_cache", "count", 0, 0, 3},
    {16, "itlb_read_misses", "hardware_cache", "count", 0, 0, 3},
    {17, "task_clock", "software", "nanosecond", 0, 0, 4},
    {18, "context_switches", "software", "count", 0, 0, 4},
    {19, "cpu_migrations", "software", "count", 0, 0, 4},
    {20, "page_faults", "software", "count", 0, 0, 4},
    {21, "minor_faults", "software", "count", 0, 0, 4},
    {22, "major_faults", "software", "count", 0, 0, 4},
};
#endif

typedef struct {
  int fd;
  uint64_t kernel_id;
  bool requested;
  bool available;
  int error_code;
  const char *status;
  uint64_t previous_raw;
  uint64_t previous_enabled;
  uint64_t previous_running;
  uint64_t previous_sample_ns;
  bool previous_valid;
} pmu_event_state_t;

typedef struct {
  int leader_fd;
  size_t event_index[OAI_PROFILE_PMU_MAX_GROUP_MEMBERS];
  size_t event_count;
  bool active;
} pmu_group_state_t;

struct oai_profile_pmu_state_s {
  pid_t tid;
  oai_profile_pmu_mode_t mode;
  pmu_event_state_t event[OAI_PROFILE_PMU_MAX_EVENTS];
  pmu_group_state_t group[OAI_PROFILE_PMU_MAX_GROUPS];
};

static bool is_software_event(const oai_profile_pmu_descriptor_t *descriptor)
{
  return strcmp(descriptor->domain, "software") == 0;
}

static bool mode_requests_event(oai_profile_pmu_mode_t mode, const oai_profile_pmu_descriptor_t *descriptor)
{
  switch (mode) {
    case OAI_PROFILE_PMU_SOFTWARE:
      return is_software_event(descriptor);
    case OAI_PROFILE_PMU_HARDWARE:
      return !is_software_event(descriptor);
    case OAI_PROFILE_PMU_AUTO:
    case OAI_PROFILE_PMU_ALL:
      return true;
    default:
      return false;
  }
}

oai_profile_pmu_mode_t oai_profile_pmu_parse_mode(const char *value)
{
  if (value == NULL || value[0] == '\0' || strcasecmp(value, "auto") == 0)
    return OAI_PROFILE_PMU_AUTO;
  if (strcasecmp(value, "off") == 0 || strcasecmp(value, "none") == 0 || strcmp(value, "0") == 0)
    return OAI_PROFILE_PMU_OFF;
  if (strcasecmp(value, "software") == 0 || strcasecmp(value, "sw") == 0)
    return OAI_PROFILE_PMU_SOFTWARE;
  if (strcasecmp(value, "hardware") == 0 || strcasecmp(value, "hw") == 0)
    return OAI_PROFILE_PMU_HARDWARE;
  if (strcasecmp(value, "all") == 0 || strcmp(value, "1") == 0)
    return OAI_PROFILE_PMU_ALL;
  return OAI_PROFILE_PMU_AUTO;
}

const char *oai_profile_pmu_mode_name(oai_profile_pmu_mode_t mode)
{
  switch (mode) {
    case OAI_PROFILE_PMU_OFF:
      return "off";
    case OAI_PROFILE_PMU_AUTO:
      return "auto";
    case OAI_PROFILE_PMU_SOFTWARE:
      return "software";
    case OAI_PROFILE_PMU_HARDWARE:
      return "hardware";
    case OAI_PROFILE_PMU_ALL:
      return "all";
    default:
      return "unknown";
  }
}

size_t oai_profile_pmu_descriptor_count(void)
{
  return sizeof(pmu_descriptors) / sizeof(pmu_descriptors[0]);
}

const oai_profile_pmu_descriptor_t *oai_profile_pmu_descriptor(size_t index)
{
  return index < oai_profile_pmu_descriptor_count() ? &pmu_descriptors[index] : NULL;
}

static const char *open_error_status(int error_code)
{
  switch (error_code) {
    case EACCES:
    case EPERM:
      return "permission_denied";
    case EINVAL:
    case ENOENT:
    case ENOSYS:
#ifdef EOPNOTSUPP
    case EOPNOTSUPP:
#endif
      return "unsupported";
    default:
      return "open_error";
  }
}

#if defined(__linux__)
static int perf_event_open(struct perf_event_attr *attr, pid_t tid, int group_fd)
{
  return (int)syscall(__NR_perf_event_open, attr, tid, -1, group_fd, PERF_FLAG_FD_CLOEXEC);
}

static void mark_opened_group_error(oai_profile_pmu_state_t *state, size_t group_id, int error_code, const char *status)
{
  pmu_group_state_t *group = &state->group[group_id];
  for (size_t i = 0; i < group->event_count; i++) {
    pmu_event_state_t *event = &state->event[group->event_index[i]];
    event->error_code = error_code;
    event->status = status;
  }
}

static void close_group(oai_profile_pmu_state_t *state, size_t group_id)
{
  pmu_group_state_t *group = &state->group[group_id];
  if (group->leader_fd >= 0)
    ioctl(group->leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
  for (size_t i = 0; i < group->event_count; i++) {
    pmu_event_state_t *event = &state->event[group->event_index[i]];
    if (event->fd >= 0)
      close(event->fd);
    event->fd = -1;
    event->available = false;
  }
  group->leader_fd = -1;
  group->event_count = 0;
  group->active = false;
}

static void open_group(oai_profile_pmu_state_t *state, size_t group_id)
{
  pmu_group_state_t *group = &state->group[group_id];
  group->leader_fd = -1;
  for (size_t i = 0; i < oai_profile_pmu_descriptor_count(); i++) {
    if (pmu_descriptors[i].group_id != group_id || !state->event[i].requested)
      continue;
    struct perf_event_attr attr = {0};
    attr.type = pmu_descriptors[i].type;
    attr.size = sizeof(attr);
    attr.config = pmu_descriptors[i].config;
    attr.disabled = group->leader_fd < 0;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    const int fd = perf_event_open(&attr, state->tid, group->leader_fd);
    if (fd < 0) {
      const int error_code = errno;
      state->event[i].error_code = error_code;
      state->event[i].status = open_error_status(error_code);
      continue;
    }
    uint64_t kernel_id = 0;
    if (ioctl(fd, PERF_EVENT_IOC_ID, &kernel_id) != 0) {
      const int error_code = errno;
      close(fd);
      state->event[i].error_code = error_code;
      state->event[i].status = "id_error";
      continue;
    }
    if (group->event_count >= OAI_PROFILE_PMU_MAX_GROUP_MEMBERS) {
      close(fd);
      state->event[i].error_code = E2BIG;
      state->event[i].status = "group_capacity_exceeded";
      continue;
    }

    if (group->leader_fd < 0)
      group->leader_fd = fd;
    state->event[i].fd = fd;
    state->event[i].kernel_id = kernel_id;
    state->event[i].available = true;
    state->event[i].status = "available";
    group->event_index[group->event_count++] = i;
  }

  if (group->leader_fd < 0 || group->event_count == 0)
    return;
  if (ioctl(group->leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0
      || ioctl(group->leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
    const int error_code = errno;
    mark_opened_group_error(state, group_id, error_code, "enable_error");
    close_group(state, group_id);
    return;
  }
  group->active = true;
}
#endif

oai_profile_pmu_state_t *oai_profile_pmu_open(pid_t tid, oai_profile_pmu_mode_t mode)
{
  if (mode == OAI_PROFILE_PMU_OFF)
    return NULL;
  oai_profile_pmu_state_t *state = calloc(1, sizeof(*state));
  if (state == NULL)
    return NULL;
  state->tid = tid;
  state->mode = mode;
  for (size_t i = 0; i < OAI_PROFILE_PMU_MAX_EVENTS; i++)
    state->event[i].fd = -1;
  for (size_t i = 0; i < OAI_PROFILE_PMU_MAX_GROUPS; i++)
    state->group[i].leader_fd = -1;
  for (size_t i = 0; i < oai_profile_pmu_descriptor_count(); i++) {
    state->event[i].requested = mode_requests_event(mode, &pmu_descriptors[i]);
    state->event[i].status = state->event[i].requested ? "not_opened" : "not_requested";
  }

#if defined(__linux__)
  for (size_t group_id = 0; group_id < OAI_PROFILE_PMU_MAX_GROUPS; group_id++)
    open_group(state, group_id);
#else
  for (size_t i = 0; i < oai_profile_pmu_descriptor_count(); i++) {
    if (state->event[i].requested)
      state->event[i].status = "unsupported_platform";
  }
#endif
  return state;
}

void oai_profile_pmu_close(oai_profile_pmu_state_t *state)
{
  if (state == NULL)
    return;
#if defined(__linux__)
  for (size_t group_id = 0; group_id < OAI_PROFILE_PMU_MAX_GROUPS; group_id++)
    close_group(state, group_id);
#endif
  free(state);
}

size_t oai_profile_pmu_get_availability(const oai_profile_pmu_state_t *state,
                                        oai_profile_pmu_availability_t *availability,
                                        size_t capacity)
{
  if (state == NULL || availability == NULL)
    return 0;
  const size_t count = oai_profile_pmu_descriptor_count() < capacity ? oai_profile_pmu_descriptor_count() : capacity;
  for (size_t i = 0; i < count; i++) {
    availability[i] = (oai_profile_pmu_availability_t){
        .event_id = pmu_descriptors[i].event_id,
        .requested = state->event[i].requested,
        .available = state->event[i].available,
        .error_code = state->event[i].error_code,
        .status = state->event[i].status,
    };
  }
  return count;
}

size_t oai_profile_pmu_available_event_count(const oai_profile_pmu_state_t *state)
{
  if (state == NULL)
    return 0;
  size_t count = 0;
  for (size_t i = 0; i < oai_profile_pmu_descriptor_count(); i++)
    count += state->event[i].available;
  return count;
}

size_t oai_profile_pmu_active_group_count(const oai_profile_pmu_state_t *state)
{
  if (state == NULL)
    return 0;
  size_t count = 0;
  for (size_t i = 0; i < OAI_PROFILE_PMU_MAX_GROUPS; i++)
    count += state->group[i].active;
  return count;
}

#if defined(__linux__)
static pmu_event_state_t *find_event_by_kernel_id(oai_profile_pmu_state_t *state, uint64_t kernel_id, size_t *event_index)
{
  for (size_t i = 0; i < oai_profile_pmu_descriptor_count(); i++) {
    if (state->event[i].available && state->event[i].kernel_id == kernel_id) {
      *event_index = i;
      return &state->event[i];
    }
  }
  return NULL;
}

static oai_profile_pmu_observation_t make_observation(pmu_event_state_t *event,
                                                      size_t event_index,
                                                      uint64_t raw_value,
                                                      uint64_t enabled,
                                                      uint64_t running,
                                                      uint64_t monotonic_raw_ns)
{
  const bool interval_valid = event->previous_valid && monotonic_raw_ns > event->previous_sample_ns;
  oai_profile_pmu_observation_t observation = {
      .event_id = pmu_descriptors[event_index].event_id,
      .raw_value = raw_value,
      .time_enabled_ns = enabled,
      .time_running_ns = running,
      .interval_ns = interval_valid ? monotonic_raw_ns - event->previous_sample_ns : 0,
      .status = event->previous_valid ? "ok" : "warmup",
  };
  observation.scaling_valid = running > 0 && enabled >= running;
  if (observation.scaling_valid) {
    observation.multiplex_ratio = enabled == 0 ? 0.0 : (double)running / (double)enabled;
    observation.scaled_value = (double)raw_value * (double)enabled / (double)running;
  }
  if (interval_valid && raw_value >= event->previous_raw && enabled >= event->previous_enabled
      && running >= event->previous_running) {
    observation.delta_valid = true;
    observation.delta_raw = raw_value - event->previous_raw;
    observation.delta_enabled_ns = enabled - event->previous_enabled;
    observation.delta_running_ns = running - event->previous_running;
    if (observation.delta_running_ns > 0 && observation.delta_enabled_ns >= observation.delta_running_ns) {
      observation.delta_scaled =
          (double)observation.delta_raw * (double)observation.delta_enabled_ns / (double)observation.delta_running_ns;
      observation.multiplex_ratio =
          observation.delta_enabled_ns == 0 ? 0.0 : (double)observation.delta_running_ns / (double)observation.delta_enabled_ns;
      observation.scaling_valid = true;
    } else {
      observation.scaling_valid = false;
      observation.status = "not_running";
    }
  } else if (event->previous_valid && !interval_valid) {
    observation.status = "clock_regression";
  } else if (event->previous_valid) {
    observation.status = "counter_reset_or_reconfigured";
  }
  event->previous_raw = raw_value;
  event->previous_enabled = enabled;
  event->previous_running = running;
  event->previous_sample_ns = monotonic_raw_ns;
  event->previous_valid = true;
  return observation;
}

static void append_group_read_error(oai_profile_pmu_state_t *state,
                                    const pmu_group_state_t *group,
                                    uint64_t monotonic_raw_ns,
                                    oai_profile_pmu_observation_t *observations,
                                    size_t capacity,
                                    oai_profile_pmu_collect_result_t *result,
                                    int error_code,
                                    const char *status)
{
  result->read_errors++;
  for (size_t i = 0; i < group->event_count && result->observation_count < capacity; i++) {
    const pmu_event_state_t *event = &state->event[group->event_index[i]];
    const uint64_t interval_ns =
        event->previous_valid && monotonic_raw_ns > event->previous_sample_ns ? monotonic_raw_ns - event->previous_sample_ns : 0;
    observations[result->observation_count++] = (oai_profile_pmu_observation_t){
        .event_id = pmu_descriptors[group->event_index[i]].event_id,
        .interval_ns = interval_ns,
        .error_code = error_code,
        .status = status,
    };
  }
}

#endif

oai_profile_pmu_collect_result_t oai_profile_pmu_collect(oai_profile_pmu_state_t *state,
                                                         uint64_t monotonic_raw_ns,
                                                         oai_profile_pmu_observation_t *observations,
                                                         size_t capacity)
{
  oai_profile_pmu_collect_result_t result = {0};
  if (state == NULL || observations == NULL || capacity == 0)
    return result;

#if defined(__linux__)
  for (size_t group_id = 0; group_id < OAI_PROFILE_PMU_MAX_GROUPS; group_id++) {
    pmu_group_state_t *group = &state->group[group_id];
    if (!group->active)
      continue;
    result.group_reads++;
    uint64_t values[3 + 2 * OAI_PROFILE_PMU_MAX_GROUP_MEMBERS] = {0};
    const size_t expected_size = (3 + 2 * group->event_count) * sizeof(uint64_t);
    const ssize_t bytes = read(group->leader_fd, values, expected_size);
    const bool complete_read = bytes >= 0 && (size_t)bytes == expected_size && values[0] == group->event_count;
    if (!complete_read) {
      const int error_code = bytes < 0 ? errno : EIO;
      append_group_read_error(state, group, monotonic_raw_ns, observations, capacity, &result, error_code, "read_error");
      continue;
    }
    const size_t returned = values[0];
    size_t event_index_by_position[OAI_PROFILE_PMU_MAX_GROUP_MEMBERS] = {0};
    bool event_seen[OAI_PROFILE_PMU_MAX_EVENTS] = {false};
    bool id_set_valid = true;
    for (size_t i = 0; i < returned; i++) {
      size_t event_index = 0;
      const uint64_t kernel_id = values[4 + 2 * i];
      pmu_event_state_t *event = find_event_by_kernel_id(state, kernel_id, &event_index);
      if (event == NULL || pmu_descriptors[event_index].group_id != group_id || event_seen[event_index]) {
        id_set_valid = false;
        break;
      }
      event_seen[event_index] = true;
      event_index_by_position[i] = event_index;
    }
    for (size_t i = 0; id_set_valid && i < group->event_count; i++)
      id_set_valid = event_seen[group->event_index[i]];
    if (!id_set_valid) {
      append_group_read_error(state, group, monotonic_raw_ns, observations, capacity, &result, EIO, "malformed_group_read");
      continue;
    }

    const uint64_t enabled = values[1];
    const uint64_t running = values[2];
    for (size_t i = 0; i < returned && result.observation_count < capacity; i++) {
      const uint64_t raw_value = values[3 + 2 * i];
      const size_t event_index = event_index_by_position[i];
      pmu_event_state_t *event = &state->event[event_index];
      observations[result.observation_count++] =
          make_observation(event, event_index, raw_value, enabled, running, monotonic_raw_ns);
    }
  }
#else
  (void)monotonic_raw_ns;
#endif
  return result;
}
