/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "oai_profiler_system.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define OAI_PROFILE_PROC_LINE_MAX 65536U
#define OAI_PROFILE_ACTIVITY_INITIAL_CAPACITY 1024U

static const char *const softirq_class_names[OAI_PROFILE_SOFTIRQ_CLASSES] =
    {"HI", "TIMER", "NET_TX", "NET_RX", "BLOCK", "IRQ_POLL", "TASKLET", "SCHED", "HRTIMER", "RCU"};

static void set_status(char *destination, size_t destination_size, const char *status)
{
  if (destination_size > 0)
    snprintf(destination, destination_size, "%s", status);
}

static void preserve_error(int *error_code, int candidate)
{
  if (*error_code == 0 && candidate != 0)
    *error_code = candidate;
}

static bool read_first_line(const char *path, char *line, size_t line_size, int *error_code)
{
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    preserve_error(error_code, errno);
    return false;
  }
  const bool valid = fgets(line, line_size, file) != NULL;
  if (!valid)
    preserve_error(error_code, ferror(file) ? errno : EPROTO);
  fclose(file);
  return valid;
}

static bool parse_thread_schedstat(pid_t tid, oai_profile_thread_metrics_snapshot_t *snapshot, int *error_code)
{
  char path[128];
  snprintf(path, sizeof(path), "/proc/self/task/%ld/schedstat", (long)tid);
  char line[256];
  if (!read_first_line(path, line, sizeof(line), error_code))
    return false;
  if (sscanf(line, "%" SCNu64 " %" SCNu64 " %" SCNu64, &snapshot->runtime_ns, &snapshot->runqueue_wait_ns, &snapshot->timeslices)
      != 3) {
    preserve_error(error_code, EPROTO);
    return false;
  }
  return true;
}

static bool parse_thread_stat(pid_t tid, oai_profile_thread_metrics_snapshot_t *snapshot, int *error_code)
{
  char path[128];
  snprintf(path, sizeof(path), "/proc/self/task/%ld/stat", (long)tid);
  char line[4096];
  if (!read_first_line(path, line, sizeof(line), error_code))
    return false;

  char *cursor = strrchr(line, ')');
  if (cursor == NULL) {
    preserve_error(error_code, EPROTO);
    return false;
  }
  cursor++;
  for (int field = 3; field <= 41; field++) {
    while (isspace((unsigned char)*cursor))
      cursor++;
    if (*cursor == '\0') {
      preserve_error(error_code, EPROTO);
      return false;
    }
    if (field == 3) {
      snapshot->state = *cursor;
      while (*cursor != '\0' && !isspace((unsigned char)*cursor))
        cursor++;
      continue;
    }

    char *end = cursor;
    while (*end != '\0' && !isspace((unsigned char)*end))
      end++;
    switch (field) {
      case 10:
      case 12:
      case 14:
      case 15:
      case 40:
      case 41: {
        errno = 0;
        char *parsed_end = NULL;
        const uint64_t value = strtoull(cursor, &parsed_end, 10);
        if (errno != 0 || parsed_end != end) {
          preserve_error(error_code, errno != 0 ? errno : EPROTO);
          return false;
        }
        if (field == 10)
          snapshot->minor_faults = value;
        else if (field == 12)
          snapshot->major_faults = value;
        else if (field == 14)
          snapshot->user_ticks = value;
        else if (field == 15)
          snapshot->system_ticks = value;
        else if (field == 40)
          snapshot->rt_priority = (uint32_t)value;
        else
          snapshot->policy = (uint32_t)value;
        break;
      }
      case 18:
      case 19:
      case 39: {
        errno = 0;
        char *parsed_end = NULL;
        const int64_t value = strtoll(cursor, &parsed_end, 10);
        if (errno != 0 || parsed_end != end) {
          preserve_error(error_code, errno != 0 ? errno : EPROTO);
          return false;
        }
        if (field == 18)
          snapshot->priority = (int32_t)value;
        else if (field == 19)
          snapshot->nice = (int32_t)value;
        else
          snapshot->processor = (int32_t)value;
        break;
      }
      default:
        break;
    }
    cursor = end;
  }
  return true;
}

static bool parse_status_counter(const char *line, const char *key, uint64_t *value)
{
  const size_t key_length = strlen(key);
  if (strncmp(line, key, key_length) != 0)
    return false;
  const char *cursor = line + key_length;
  while (isspace((unsigned char)*cursor))
    cursor++;
  errno = 0;
  char *end = NULL;
  const unsigned long long parsed = strtoull(cursor, &end, 10);
  if (errno != 0 || end == cursor)
    return false;
  *value = (uint64_t)parsed;
  return true;
}

static bool parse_thread_status(pid_t tid, oai_profile_thread_metrics_snapshot_t *snapshot, int *error_code)
{
  char path[128];
  snprintf(path, sizeof(path), "/proc/self/task/%ld/status", (long)tid);
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    preserve_error(error_code, errno);
    return false;
  }
  bool voluntary_valid = false;
  bool involuntary_valid = false;
  char line[256];
  while (fgets(line, sizeof(line), file) != NULL) {
    voluntary_valid |= parse_status_counter(line, "voluntary_ctxt_switches:", &snapshot->voluntary_context_switches);
    involuntary_valid |= parse_status_counter(line, "nonvoluntary_ctxt_switches:", &snapshot->involuntary_context_switches);
  }
  if (ferror(file))
    preserve_error(error_code, errno);
  fclose(file);
  if (!voluntary_valid || !involuntary_valid) {
    preserve_error(error_code, EPROTO);
    return false;
  }
  return true;
}

static bool read_cpu_frequency(int32_t cpu, int64_t *frequency_khz)
{
  if (cpu < 0)
    return false;
  char path[PATH_MAX];
  snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);
  FILE *file = fopen(path, "r");
  if (file == NULL)
    return false;
  const bool valid = fscanf(file, "%" SCNd64, frequency_khz) == 1;
  fclose(file);
  return valid;
}

static bool thread_counters_monotonic(const oai_profile_thread_metrics_snapshot_t *current,
                                      const oai_profile_thread_metrics_snapshot_t *previous)
{
  return current->runtime_ns >= previous->runtime_ns && current->runqueue_wait_ns >= previous->runqueue_wait_ns
         && current->timeslices >= previous->timeslices && current->minor_faults >= previous->minor_faults
         && current->major_faults >= previous->major_faults && current->user_ticks >= previous->user_ticks
         && current->system_ticks >= previous->system_ticks
         && current->voluntary_context_switches >= previous->voluntary_context_switches
         && current->involuntary_context_switches >= previous->involuntary_context_switches;
}

void oai_profile_read_thread_metrics(pid_t tid,
                                     uint64_t monotonic_ns,
                                     oai_profile_thread_metrics_state_t *state,
                                     oai_profile_thread_metrics_observation_t *observation)
{
  memset(observation, 0, sizeof(*observation));
  observation->current.processor = -1;
  observation->current.cpu_frequency_khz = -1;

  if (parse_thread_schedstat(tid, &observation->current, &observation->error_code))
    observation->current.valid_mask |= OAI_PROFILE_THREAD_METRIC_SCHEDSTAT;
  if (parse_thread_stat(tid, &observation->current, &observation->error_code))
    observation->current.valid_mask |= OAI_PROFILE_THREAD_METRIC_STAT;
  if (parse_thread_status(tid, &observation->current, &observation->error_code))
    observation->current.valid_mask |= OAI_PROFILE_THREAD_METRIC_STATUS;
  if ((observation->current.valid_mask & OAI_PROFILE_THREAD_METRIC_STAT)
      && read_cpu_frequency(observation->current.processor, &observation->current.cpu_frequency_khz))
    observation->current.valid_mask |= OAI_PROFILE_THREAD_METRIC_CPU_FREQUENCY;

  if ((observation->current.valid_mask & OAI_PROFILE_THREAD_METRIC_CORE_MASK) != OAI_PROFILE_THREAD_METRIC_CORE_MASK) {
    set_status(observation->status,
               sizeof(observation->status),
               observation->current.valid_mask == 0 ? "thread_unavailable" : "partial");
    return;
  }

  if (!state->previous_valid) {
    set_status(observation->status, sizeof(observation->status), "warmup");
  } else if (monotonic_ns <= state->previous_monotonic_ns) {
    set_status(observation->status, sizeof(observation->status), "clock_regression");
  } else if (!thread_counters_monotonic(&observation->current, &state->previous)) {
    set_status(observation->status, sizeof(observation->status), "counter_reset");
  } else {
    observation->interval_ns = monotonic_ns - state->previous_monotonic_ns;
    observation->delta_runtime_ns = observation->current.runtime_ns - state->previous.runtime_ns;
    observation->delta_runqueue_wait_ns = observation->current.runqueue_wait_ns - state->previous.runqueue_wait_ns;
    observation->delta_timeslices = observation->current.timeslices - state->previous.timeslices;
    observation->delta_minor_faults = observation->current.minor_faults - state->previous.minor_faults;
    observation->delta_major_faults = observation->current.major_faults - state->previous.major_faults;
    observation->delta_user_ticks = observation->current.user_ticks - state->previous.user_ticks;
    observation->delta_system_ticks = observation->current.system_ticks - state->previous.system_ticks;
    observation->delta_voluntary_context_switches =
        observation->current.voluntary_context_switches - state->previous.voluntary_context_switches;
    observation->delta_involuntary_context_switches =
        observation->current.involuntary_context_switches - state->previous.involuntary_context_switches;
    observation->delta_valid = true;
    observation->cpu_changed_since_previous = observation->current.processor != state->previous.processor;
    set_status(observation->status, sizeof(observation->status), "ok");
  }

  state->previous = observation->current;
  state->previous_monotonic_ns = monotonic_ns;
  state->previous_valid = true;
}

static bool parse_u64_values(const char *cursor, uint64_t *values, size_t count)
{
  for (size_t i = 0; i < count; i++) {
    while (isspace((unsigned char)*cursor))
      cursor++;
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = strtoull(cursor, &end, 10);
    if (errno != 0 || end == cursor)
      return false;
    values[i] = (uint64_t)parsed;
    cursor = end;
  }
  return true;
}

static bool kernel_counters_monotonic(const oai_profile_kernel_activity_snapshot_t *current,
                                      const oai_profile_kernel_activity_snapshot_t *previous)
{
  if (current->interrupts < previous->interrupts || current->context_switches < previous->context_switches
      || current->processes_created < previous->processes_created || current->softirqs < previous->softirqs)
    return false;
  for (size_t i = 0; i < OAI_PROFILE_SOFTIRQ_CLASSES; i++) {
    if (current->softirq_classes[i] < previous->softirq_classes[i])
      return false;
  }
  return true;
}

void oai_profile_read_kernel_activity(uint64_t monotonic_ns,
                                      oai_profile_kernel_activity_state_t *state,
                                      oai_profile_kernel_activity_observation_t *observation)
{
  memset(observation, 0, sizeof(*observation));
  FILE *file = fopen("/proc/stat", "r");
  if (file == NULL) {
    observation->error_code = errno;
    set_status(observation->status, sizeof(observation->status), "unavailable");
    return;
  }

  char line[4096];
  while (fgets(line, sizeof(line), file) != NULL) {
    uint64_t value = 0;
    if (sscanf(line, "intr %" SCNu64, &value) == 1) {
      observation->current.interrupts = value;
      observation->current.valid_mask |= OAI_PROFILE_KERNEL_ACTIVITY_INTERRUPTS;
    } else if (sscanf(line, "ctxt %" SCNu64, &value) == 1) {
      observation->current.context_switches = value;
      observation->current.valid_mask |= OAI_PROFILE_KERNEL_ACTIVITY_CONTEXT_SWITCHES;
    } else if (sscanf(line, "processes %" SCNu64, &value) == 1) {
      observation->current.processes_created = value;
      observation->current.valid_mask |= OAI_PROFILE_KERNEL_ACTIVITY_PROCESSES;
    } else if (sscanf(line, "procs_running %" SCNu64, &value) == 1) {
      observation->current.processes_running = value;
      observation->current.valid_mask |= OAI_PROFILE_KERNEL_ACTIVITY_RUNNING;
    } else if (sscanf(line, "procs_blocked %" SCNu64, &value) == 1) {
      observation->current.processes_blocked = value;
      observation->current.valid_mask |= OAI_PROFILE_KERNEL_ACTIVITY_BLOCKED;
    } else if (strncmp(line, "softirq ", 8) == 0) {
      uint64_t values[OAI_PROFILE_SOFTIRQ_CLASSES + 1];
      if (parse_u64_values(line + 8, values, OAI_PROFILE_SOFTIRQ_CLASSES + 1)) {
        observation->current.softirqs = values[0];
        memcpy(observation->current.softirq_classes, &values[1], sizeof(observation->current.softirq_classes));
        observation->current.valid_mask |= OAI_PROFILE_KERNEL_ACTIVITY_SOFTIRQS;
      }
    }
  }
  if (ferror(file))
    observation->error_code = errno;
  fclose(file);

  if (observation->current.valid_mask != OAI_PROFILE_KERNEL_ACTIVITY_ALL_MASK) {
    preserve_error(&observation->error_code, EPROTO);
    set_status(observation->status, sizeof(observation->status), "partial");
    return;
  }
  if (!state->previous_valid) {
    set_status(observation->status, sizeof(observation->status), "warmup");
  } else if (monotonic_ns <= state->previous_monotonic_ns) {
    set_status(observation->status, sizeof(observation->status), "clock_regression");
  } else if (!kernel_counters_monotonic(&observation->current, &state->previous)) {
    set_status(observation->status, sizeof(observation->status), "counter_reset");
  } else {
    observation->interval_ns = monotonic_ns - state->previous_monotonic_ns;
    observation->delta_interrupts = observation->current.interrupts - state->previous.interrupts;
    observation->delta_context_switches = observation->current.context_switches - state->previous.context_switches;
    observation->delta_processes_created = observation->current.processes_created - state->previous.processes_created;
    observation->delta_softirqs = observation->current.softirqs - state->previous.softirqs;
    for (size_t i = 0; i < OAI_PROFILE_SOFTIRQ_CLASSES; i++)
      observation->delta_softirq_classes[i] = observation->current.softirq_classes[i] - state->previous.softirq_classes[i];
    observation->delta_valid = true;
    set_status(observation->status, sizeof(observation->status), "ok");
  }
  state->previous = observation->current;
  state->previous_monotonic_ns = monotonic_ns;
  state->previous_valid = true;
}

const char *oai_profile_softirq_class_name(size_t index)
{
  return index < OAI_PROFILE_SOFTIRQ_CLASSES ? softirq_class_names[index] : "UNKNOWN";
}

typedef struct {
  bool used;
  char source[16];
  char label[48];
  int32_t cpu;
  uint64_t previous_count;
  uint64_t previous_monotonic_ns;
} activity_entry_t;

struct oai_profile_activity_state_s {
  activity_entry_t *entries;
  size_t capacity;
  size_t count;
};

static uint64_t hash_activity_key(const char *source, const char *label, int32_t cpu)
{
  uint64_t hash = UINT64_C(1469598103934665603);
  for (const unsigned char *p = (const unsigned char *)source; *p != '\0'; p++)
    hash = (hash ^ *p) * UINT64_C(1099511628211);
  for (const unsigned char *p = (const unsigned char *)label; *p != '\0'; p++)
    hash = (hash ^ *p) * UINT64_C(1099511628211);
  return (hash ^ (uint32_t)cpu) * UINT64_C(1099511628211);
}

static bool activity_state_resize(oai_profile_activity_state_t *state, size_t new_capacity)
{
  activity_entry_t *new_entries = calloc(new_capacity, sizeof(*new_entries));
  if (new_entries == NULL)
    return false;
  for (size_t i = 0; i < state->capacity; i++) {
    activity_entry_t entry = state->entries[i];
    if (!entry.used)
      continue;
    size_t index = hash_activity_key(entry.source, entry.label, entry.cpu) % new_capacity;
    while (new_entries[index].used)
      index = (index + 1) % new_capacity;
    new_entries[index] = entry;
  }
  free(state->entries);
  state->entries = new_entries;
  state->capacity = new_capacity;
  return true;
}

static activity_entry_t *get_activity_entry(oai_profile_activity_state_t *state, const char *source, const char *label, int32_t cpu)
{
  if (state == NULL)
    return NULL;
  if (state->count * 10 >= state->capacity * 7 && !activity_state_resize(state, state->capacity * 2))
    return NULL;
  size_t index = hash_activity_key(source, label, cpu) % state->capacity;
  while (state->entries[index].used) {
    activity_entry_t *entry = &state->entries[index];
    if (entry->cpu == cpu && strcmp(entry->source, source) == 0 && strcmp(entry->label, label) == 0)
      return entry;
    index = (index + 1) % state->capacity;
  }
  activity_entry_t *entry = &state->entries[index];
  entry->used = true;
  snprintf(entry->source, sizeof(entry->source), "%s", source);
  snprintf(entry->label, sizeof(entry->label), "%s", label);
  entry->cpu = cpu;
  state->count++;
  return entry;
}

oai_profile_activity_state_t *oai_profile_activity_state_create(void)
{
  oai_profile_activity_state_t *state = calloc(1, sizeof(*state));
  if (state == NULL)
    return NULL;
  state->capacity = OAI_PROFILE_ACTIVITY_INITIAL_CAPACITY;
  state->entries = calloc(state->capacity, sizeof(*state->entries));
  if (state->entries == NULL) {
    free(state);
    return NULL;
  }
  return state;
}

void oai_profile_activity_state_destroy(oai_profile_activity_state_t *state)
{
  if (state == NULL)
    return;
  free(state->entries);
  free(state);
}

static uint32_t count_cpu_columns(char *line)
{
  uint32_t count = 0;
  char *save = NULL;
  for (char *token = strtok_r(line, " \t\r\n", &save); token != NULL; token = strtok_r(NULL, " \t\r\n", &save)) {
    if (strncmp(token, "CPU", 3) != 0 || token[3] == '\0')
      continue;
    bool digits = true;
    for (const char *p = token + 3; *p != '\0'; p++)
      digits &= isdigit((unsigned char)*p) != 0;
    count += digits;
  }
  return count;
}

static void trim_label(char *label)
{
  char *start = label;
  while (isspace((unsigned char)*start))
    start++;
  if (start != label)
    memmove(label, start, strlen(start) + 1);
  size_t length = strlen(label);
  while (length > 0 && isspace((unsigned char)label[length - 1]))
    label[--length] = '\0';
}

static bool is_radio_relevant(const char *label, const char *description)
{
  static const char *const terms[] = {"xhci", "usb", "dwc", "usrp", "b205", "b210"};
  for (size_t i = 0; i < sizeof(terms) / sizeof(terms[0]); i++) {
    if (strcasestr(label, terms[i]) != NULL || strcasestr(description, terms[i]) != NULL)
      return true;
  }
  return false;
}

static bool is_global_interrupt_scalar(const char *source, const char *label, const char *values, uint32_t cpu_count)
{
  if (cpu_count <= 1 || strcmp(source, "hardirq") != 0 || (strcasecmp(label, "ERR") != 0 && strcasecmp(label, "MIS") != 0))
    return false;

  while (isspace((unsigned char)*values))
    values++;
  if (!isdigit((unsigned char)*values))
    return false;
  errno = 0;
  char *end = NULL;
  (void)strtoull(values, &end, 10);
  if (errno != 0 || end == values)
    return false;
  while (isspace((unsigned char)*end))
    end++;
  return *end == '\0';
}

static void emit_activity(oai_profile_activity_state_t *state,
                          const char *source,
                          const char *label,
                          const char *description,
                          int32_t cpu,
                          uint64_t count,
                          uint64_t monotonic_ns,
                          oai_profile_activity_callback_t callback,
                          void *opaque)
{
  oai_profile_activity_observation_t observation = {
      .source = source,
      .label = label,
      .description = description,
      .cpu = cpu,
      .raw_count = count,
      .radio_relevant = is_radio_relevant(label, description),
      .status = "warmup",
  };
  activity_entry_t *entry = get_activity_entry(state, source, label, cpu);
  if (entry == NULL) {
    observation.status = "state_unavailable";
  } else if (entry->previous_monotonic_ns == 0) {
    entry->previous_count = count;
    entry->previous_monotonic_ns = monotonic_ns;
  } else if (monotonic_ns <= entry->previous_monotonic_ns) {
    observation.status = "clock_regression";
    entry->previous_count = count;
    entry->previous_monotonic_ns = monotonic_ns;
  } else if (count < entry->previous_count) {
    observation.status = "counter_reset";
    entry->previous_count = count;
    entry->previous_monotonic_ns = monotonic_ns;
  } else {
    observation.delta_count = count - entry->previous_count;
    observation.interval_ns = monotonic_ns - entry->previous_monotonic_ns;
    observation.delta_valid = true;
    observation.status = "ok";
    entry->previous_count = count;
    entry->previous_monotonic_ns = monotonic_ns;
  }
  callback(&observation, opaque);
}

static oai_profile_activity_result_t collect_activity(const char *path,
                                                      const char *source,
                                                      oai_profile_activity_state_t *state,
                                                      uint64_t monotonic_ns,
                                                      oai_profile_activity_callback_t callback,
                                                      void *opaque)
{
  oai_profile_activity_result_t result = {0};
  set_status(result.status, sizeof(result.status), "unavailable");
  if (path == NULL || source == NULL || callback == NULL) {
    result.error_code = EINVAL;
    return result;
  }
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    result.error_code = errno;
    return result;
  }
  char *line = malloc(OAI_PROFILE_PROC_LINE_MAX);
  if (line == NULL) {
    result.error_code = ENOMEM;
    fclose(file);
    return result;
  }
  if (fgets(line, OAI_PROFILE_PROC_LINE_MAX, file) == NULL) {
    result.error_code = ferror(file) ? errno : EPROTO;
    free(line);
    fclose(file);
    return result;
  }
  result.cpu_count = count_cpu_columns(line);
  if (result.cpu_count == 0) {
    result.error_code = EPROTO;
    free(line);
    fclose(file);
    return result;
  }

  uint64_t *counts = calloc(result.cpu_count, sizeof(*counts));
  if (counts == NULL) {
    result.error_code = ENOMEM;
    free(line);
    fclose(file);
    return result;
  }
  while (fgets(line, OAI_PROFILE_PROC_LINE_MAX, file) != NULL) {
    char *colon = strchr(line, ':');
    if (colon == NULL)
      continue;
    *colon = '\0';
    trim_label(line);
    if (line[0] == '\0')
      continue;
    char *cursor = colon + 1;
    if (is_global_interrupt_scalar(source, line, cursor, result.cpu_count))
      continue;
    bool parsed = true;
    for (uint32_t cpu = 0; cpu < result.cpu_count; cpu++) {
      while (isspace((unsigned char)*cursor))
        cursor++;
      errno = 0;
      char *end = NULL;
      const unsigned long long value = strtoull(cursor, &end, 10);
      if (errno != 0 || end == cursor) {
        parsed = false;
        break;
      }
      counts[cpu] = (uint64_t)value;
      cursor = end;
    }
    if (!parsed) {
      result.parse_errors++;
      preserve_error(&result.error_code, EPROTO);
      continue;
    }
    while (isspace((unsigned char)*cursor))
      cursor++;
    cursor[strcspn(cursor, "\r\n")] = '\0';
    for (uint32_t cpu = 0; cpu < result.cpu_count; cpu++) {
      emit_activity(state, source, line, cursor, (int32_t)cpu, counts[cpu], monotonic_ns, callback, opaque);
      result.rows++;
    }
  }
  if (ferror(file))
    result.error_code = errno;
  free(counts);
  free(line);
  fclose(file);
  set_status(result.status, sizeof(result.status), result.parse_errors == 0 && result.error_code == 0 ? "ok" : "partial");
  return result;
}

oai_profile_activity_result_t oai_profile_collect_interrupts_path(const char *path,
                                                                  oai_profile_activity_state_t *state,
                                                                  uint64_t monotonic_ns,
                                                                  oai_profile_activity_callback_t callback,
                                                                  void *opaque)
{
  return collect_activity(path, "hardirq", state, monotonic_ns, callback, opaque);
}

oai_profile_activity_result_t oai_profile_collect_interrupts(oai_profile_activity_state_t *state,
                                                             uint64_t monotonic_ns,
                                                             oai_profile_activity_callback_t callback,
                                                             void *opaque)
{
  return oai_profile_collect_interrupts_path("/proc/interrupts", state, monotonic_ns, callback, opaque);
}

oai_profile_activity_result_t oai_profile_collect_softirqs(oai_profile_activity_state_t *state,
                                                           uint64_t monotonic_ns,
                                                           oai_profile_activity_callback_t callback,
                                                           void *opaque)
{
  return collect_activity("/proc/softirqs", "softirq", state, monotonic_ns, callback, opaque);
}
