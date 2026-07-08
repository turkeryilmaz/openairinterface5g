/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE
#include "oai_profiler.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "common/utils/LOG/log.h"

#define OAI_PROFILE_DEFAULT_BUFFER_RECORDS 131072U
#define OAI_PROFILE_DEFAULT_FLUSH_US 100000U
#define OAI_PROFILE_MIN_BUFFER_RECORDS 1024U
#define OAI_PROFILE_THREAD_NAME_LEN 16

volatile int oai_profiler_enabled = 0;

typedef struct {
  uint64_t seq;
  uint64_t start_tick;
  uint64_t duration_tick;
  uint32_t event_id;
  int32_t frame;
  int32_t slot;
  int64_t aux0;
  int64_t aux1;
  int64_t aux2;
  int64_t aux3;
  uint32_t flags;
} oai_profile_record_t;

typedef struct {
  bool active;
  pid_t tid;
  char name[OAI_PROFILE_THREAD_NAME_LEN];
  oai_profile_record_t *records;
  uint32_t capacity;
  volatile uint64_t write_count;
  volatile uint64_t read_count;
  uint64_t dropped_records;
} oai_profile_thread_buffer_t;

static const char *const event_names[OAI_PROFILE_EVENT_MAX] = {
    [OAI_PROFILE_EVENT_UNSPEC] = "UNSPEC",
    [OAI_PROFILE_EVENT_UE_SLOT_LOOP] = "UE_SLOT_LOOP",
    [OAI_PROFILE_EVENT_UE_RF_READ] = "UE_RF_READ",
    [OAI_PROFILE_EVENT_UE_RF_READ_DRIFT] = "UE_RF_READ_DRIFT",
    [OAI_PROFILE_EVENT_UE_SCOPE_COPY] = "UE_SCOPE_COPY",
    [OAI_PROFILE_EVENT_UE_TIMING_COMPUTE] = "UE_TIMING_COMPUTE",
    [OAI_PROFILE_EVENT_UE_DL_PREPROCESS] = "UE_DL_PREPROCESS",
    [OAI_PROFILE_EVENT_UE_DL_PROCESSING] = "UE_DL_PROCESSING",
    [OAI_PROFILE_EVENT_UE_DL_ACTOR_DISPATCH] = "UE_DL_ACTOR_DISPATCH",
    [OAI_PROFILE_EVENT_UE_NTN_CONFIG_APPLY] = "UE_NTN_CONFIG_APPLY",
    [OAI_PROFILE_EVENT_UE_TX_SCHEDULE] = "UE_TX_SCHEDULE",
    [OAI_PROFILE_EVENT_UE_TX_SLOT] = "UE_TX_SLOT",
    [OAI_PROFILE_EVENT_UE_TX_UL_INDICATION] = "UE_TX_UL_INDICATION",
    [OAI_PROFILE_EVENT_UE_TX_BARRIER_WAIT] = "UE_TX_BARRIER_WAIT",
    [OAI_PROFILE_EVENT_UE_TX_PHY_PROCEDURES] = "UE_TX_PHY_PROCEDURES",
    [OAI_PROFILE_EVENT_UE_TX_RU_WRITE] = "UE_TX_RU_WRITE",
    [OAI_PROFILE_EVENT_UE_RF_WRITE] = "UE_RF_WRITE",
    [OAI_PROFILE_EVENT_UE_TX_DEADLINE_MISS] = "UE_TX_DEADLINE_MISS",
    [OAI_PROFILE_EVENT_GNB_SLOT_INDICATION] = "GNB_SLOT_INDICATION",
    [OAI_PROFILE_EVENT_GNB_RX_TRIGGER] = "GNB_RX_TRIGGER",
    [OAI_PROFILE_EVENT_GNB_PHY_TX] = "GNB_PHY_TX",
    [OAI_PROFILE_EVENT_GNB_RU_TX] = "GNB_RU_TX",
    [OAI_PROFILE_EVENT_GNB_L1_TX_JOB] = "GNB_L1_TX_JOB",
    [OAI_PROFILE_EVENT_GNB_L1_RX_JOB] = "GNB_L1_RX_JOB",
    [OAI_PROFILE_EVENT_GNB_PRACH_QUEUE_DRAIN] = "GNB_PRACH_QUEUE_DRAIN",
    [OAI_PROFILE_EVENT_GNB_PHASE_COMP] = "GNB_PHASE_COMP",
    [OAI_PROFILE_EVENT_GNB_PHY_UESPEC_RX] = "GNB_PHY_UESPEC_RX",
    [OAI_PROFILE_EVENT_GNB_UL_INDICATION] = "GNB_UL_INDICATION",
    [OAI_PROFILE_EVENT_GNB_RF_READ] = "GNB_RF_READ",
    [OAI_PROFILE_EVENT_GNB_RF_READ_ALIGN] = "GNB_RF_READ_ALIGN",
    [OAI_PROFILE_EVENT_GNB_RF_WRITE] = "GNB_RF_WRITE",
};

static oai_profile_thread_buffer_t thread_buffers[OAI_PROFILE_MAX_THREADS];
static pthread_mutex_t registry_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t lifecycle_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_t writer_thread;
static bool profiler_initialized;
static bool writer_started;
static volatile bool profiler_shutdown_requested;
static __thread int thread_buffer_index = -1;
static uint32_t global_buffer_records = OAI_PROFILE_DEFAULT_BUFFER_RECORDS;
static uint32_t global_flush_us = OAI_PROFILE_DEFAULT_FLUSH_US;
static uint64_t global_seq;
static uint64_t counter_hz;
static char output_dir[PATH_MAX];
static FILE *events_file;
static FILE *sync_file;
static FILE *drops_file;

const char *oai_profiler_event_name(oai_profile_event_id_t event_id)
{
  if (event_id <= OAI_PROFILE_EVENT_UNSPEC || event_id >= OAI_PROFILE_EVENT_MAX || event_names[event_id] == NULL)
    return "UNKNOWN";
  return event_names[event_id];
}

static uint64_t read_counter_hz(void)
{
#if defined(__aarch64__)
  uint64_t hz = 0;
  asm volatile("mrs %0, cntfrq_el0" : "=r"(hz));
  return hz;
#else
  return (uint64_t)(get_cpu_freq_GHz() * 1000000000.0);
#endif
}

static uint32_t parse_u32_or_default(const char *value, uint32_t default_value)
{
  if (value == NULL || value[0] == '\0')
    return default_value;
  char *end = NULL;
  errno = 0;
  unsigned long parsed = strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed > UINT32_MAX)
    return default_value;
  return (uint32_t)parsed;
}

static uint64_t timespec_to_ns(const struct timespec *ts)
{
  return (uint64_t)ts->tv_sec * 1000000000ULL + (uint64_t)ts->tv_nsec;
}

static bool parse_env_enable(bool cli_enabled)
{
  const char *env = getenv("OAI_PROFILE");
  if (env == NULL || env[0] == '\0')
    return cli_enabled;
  if (!strcmp(env, "0") || !strcasecmp(env, "false") || !strcasecmp(env, "off"))
    return false;
  return true;
}

static int mkdir_p(const char *path)
{
  char tmp[PATH_MAX];
  if (path == NULL || path[0] == '\0')
    return -1;
  int ret = snprintf(tmp, sizeof(tmp), "%s", path);
  if (ret < 0 || (size_t)ret >= sizeof(tmp))
    return -1;
  size_t len = strlen(tmp);
  if (len == 0)
    return -1;
  if (tmp[len - 1] == '/')
    tmp[len - 1] = '\0';
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir(tmp, 0775) != 0 && errno != EEXIST)
        return -1;
      *p = '/';
    }
  }
  if (mkdir(tmp, 0775) != 0 && errno != EEXIST)
    return -1;
  return 0;
}

static int open_profile_file(FILE **file, const char *name, const char *mode)
{
  char path[PATH_MAX];
  int ret = snprintf(path, sizeof(path), "%s/%s", output_dir, name);
  if (ret < 0 || (size_t)ret >= sizeof(path))
    return -1;
  *file = fopen(path, mode);
  if (*file == NULL)
    return -1;
  setvbuf(*file, NULL, _IOFBF, 1 << 20);
  return 0;
}

static void write_event_catalog(void)
{
  FILE *file = NULL;
  if (open_profile_file(&file, "event_catalog.csv", "w") != 0)
    return;
  fprintf(file, "event_id,event_name\n");
  for (int i = 1; i < OAI_PROFILE_EVENT_MAX; i++)
    fprintf(file, "%d,%s\n", i, oai_profiler_event_name(i));
  fclose(file);
}

static void write_metadata(const char *process_name, int argc, char **argv, uint32_t buffer_records, uint32_t flush_us)
{
  FILE *file = NULL;
  if (open_profile_file(&file, "metadata.txt", "w") != 0)
    return;
  struct timespec rt = {0};
  struct timespec mt = {0};
  clock_gettime(CLOCK_REALTIME, &rt);
  clock_gettime(CLOCK_MONOTONIC_RAW, &mt);
  char hostname[256] = {0};
  gethostname(hostname, sizeof(hostname) - 1);
  fprintf(file, "process_name=%s\n", process_name ? process_name : "unknown");
  fprintf(file, "pid=%ld\n", (long)getpid());
  fprintf(file, "hostname=%s\n", hostname);
  fprintf(file, "counter_hz=%" PRIu64 "\n", counter_hz);
  fprintf(file, "start_realtime_ns=%" PRIu64 "\n", timespec_to_ns(&rt));
  fprintf(file, "start_monotonic_raw_ns=%" PRIu64 "\n", timespec_to_ns(&mt));
  fprintf(file, "buffer_records_per_thread=%u\n", buffer_records);
  fprintf(file, "flush_us=%u\n", flush_us);
  fprintf(file, "max_threads=%u\n", OAI_PROFILE_MAX_THREADS);
  fprintf(file, "cmdline=");
  for (int i = 0; i < argc; i++)
    fprintf(file, "%s%s", i == 0 ? "" : " ", argv[i]);
  fprintf(file, "\n");
  fclose(file);
}

static void write_sync_sample(void)
{
  if (sync_file == NULL)
    return;
  struct timespec rt = {0};
  struct timespec mt = {0};
  clock_gettime(CLOCK_REALTIME, &rt);
  clock_gettime(CLOCK_MONOTONIC_RAW, &mt);
  uint64_t tick = rdtsc_oai();
  fprintf(sync_file, "%" PRIu64 ",%" PRIu64 ",%" PRIu64 "\n", timespec_to_ns(&rt), timespec_to_ns(&mt), tick);
}

static void drain_thread_buffer(oai_profile_thread_buffer_t *tb)
{
  uint64_t read_count = tb->read_count;
  const uint64_t write_count = tb->write_count;
  while (read_count < write_count) {
    const oai_profile_record_t *r = &tb->records[read_count % tb->capacity];
    const double duration_us = counter_hz == 0 ? 0.0 : ((double)r->duration_tick * 1000000.0) / (double)counter_hz;
    fprintf(events_file,
            "%" PRIu64 ",%ld,%s,%u,%s,%d,%d,%u,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRIu64 ",%" PRIu64 ",%.3f\n",
            r->seq,
            (long)tb->tid,
            tb->name,
            r->event_id,
            oai_profiler_event_name((oai_profile_event_id_t)r->event_id),
            r->frame,
            r->slot,
            r->flags,
            r->aux0,
            r->aux1,
            r->aux2,
            r->aux3,
            r->start_tick,
            r->duration_tick,
            duration_us);
    read_count++;
  }
  tb->read_count = read_count;
}

static void drain_all_buffers(void)
{
  pthread_mutex_lock(&registry_mutex);
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    if (thread_buffers[i].active)
      drain_thread_buffer(&thread_buffers[i]);
  }
  pthread_mutex_unlock(&registry_mutex);
  if (events_file != NULL)
    fflush(events_file);
  if (sync_file != NULL)
    fflush(sync_file);
}

static void write_drop_summary(void)
{
  if (drops_file == NULL)
    return;
  fprintf(drops_file, "thread_index,tid,thread_name,dropped_records\n");
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    if (thread_buffers[i].active)
      fprintf(drops_file,
              "%d,%ld,%s,%" PRIu64 "\n",
              i,
              (long)thread_buffers[i].tid,
              thread_buffers[i].name,
              thread_buffers[i].dropped_records);
  }
  fflush(drops_file);
}

static void *profiler_writer_thread(void *arg)
{
  (void)arg;
  pthread_setname_np(pthread_self(), "oai_profile");
  while (!profiler_shutdown_requested) {
    drain_all_buffers();
    write_sync_sample();
    usleep(global_flush_us);
  }
  drain_all_buffers();
  write_sync_sample();
  write_drop_summary();
  return NULL;
}

static int register_thread_buffer(void)
{
  if (thread_buffer_index >= 0)
    return thread_buffer_index;

  pthread_mutex_lock(&registry_mutex);
  int idx = -1;
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    if (!thread_buffers[i].active) {
      idx = i;
      break;
    }
  }
  if (idx < 0) {
    pthread_mutex_unlock(&registry_mutex);
    return -1;
  }

  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  memset(tb, 0, sizeof(*tb));
  tb->records = calloc(global_buffer_records, sizeof(*tb->records));
  if (tb->records == NULL) {
    pthread_mutex_unlock(&registry_mutex);
    return -1;
  }
  tb->capacity = global_buffer_records;
  tb->tid = (pid_t)syscall(SYS_gettid);
  if (pthread_getname_np(pthread_self(), tb->name, sizeof(tb->name)) != 0 || tb->name[0] == '\0')
    snprintf(tb->name, sizeof(tb->name), "tid-%ld", (long)tb->tid);
  tb->active = true;
  thread_buffer_index = idx;
  pthread_mutex_unlock(&registry_mutex);
  return idx;
}

void oai_profiler_register_thread(void)
{
  if (!oai_profiler_enabled)
    return;
  (void)register_thread_buffer();
}

void oai_profiler_record_duration(oai_profile_event_id_t event_id,
                                  uint64_t start_tick,
                                  int frame,
                                  int slot,
                                  int64_t aux0,
                                  int64_t aux1,
                                  int64_t aux2,
                                  int64_t aux3,
                                  uint32_t flags)
{
  if (!oai_profiler_enabled || start_tick == 0 || event_id <= OAI_PROFILE_EVENT_UNSPEC || event_id >= OAI_PROFILE_EVENT_MAX)
    return;
  const uint64_t end_tick = rdtsc_oai();
  const int idx = register_thread_buffer();
  if (idx < 0)
    return;
  oai_profile_thread_buffer_t *tb = &thread_buffers[idx];
  const uint64_t write_count = tb->write_count;
  const uint64_t read_count = tb->read_count;
  if (write_count - read_count >= tb->capacity) {
    tb->dropped_records++;
    return;
  }
  oai_profile_record_t *r = &tb->records[write_count % tb->capacity];
  r->seq = __sync_fetch_and_add(&global_seq, 1);
  r->start_tick = start_tick;
  r->duration_tick = end_tick - start_tick;
  r->event_id = event_id;
  r->frame = frame;
  r->slot = slot;
  r->aux0 = aux0;
  r->aux1 = aux1;
  r->aux2 = aux2;
  r->aux3 = aux3;
  r->flags = flags;
  __sync_synchronize();
  tb->write_count = write_count + 1;
}

void oai_profiler_record_instant(oai_profile_event_id_t event_id,
                                 int frame,
                                 int slot,
                                 int64_t aux0,
                                 int64_t aux1,
                                 int64_t aux2,
                                 int64_t aux3,
                                 uint32_t flags)
{
  uint64_t tick = oai_profiler_start();
  oai_profiler_record_duration(event_id, tick, frame, slot, aux0, aux1, aux2, aux3, flags);
}

void oai_profiler_init(const char *process_name,
                       int argc,
                       char **argv,
                       bool enable_from_cli,
                       const char *profile_dir,
                       uint32_t buffer_records,
                       uint32_t flush_us)
{
  pthread_mutex_lock(&lifecycle_mutex);
  if (profiler_initialized) {
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }

  const bool enabled = parse_env_enable(enable_from_cli);
  if (!enabled) {
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }

  const char *env_dir = getenv("OAI_PROFILE_DIR");
  const char *dir = env_dir != NULL && env_dir[0] != '\0' ? env_dir : profile_dir;
  if (dir != NULL && dir[0] != '\0') {
    int ret = snprintf(output_dir, sizeof(output_dir), "%s", dir);
    if (ret < 0 || (size_t)ret >= sizeof(output_dir)) {
      pthread_mutex_unlock(&lifecycle_mutex);
      return;
    }
  } else {
    int ret =
        snprintf(output_dir, sizeof(output_dir), "./oai_profile_%s_%ld", process_name ? process_name : "softmodem", (long)getpid());
    if (ret < 0 || (size_t)ret >= sizeof(output_dir)) {
      pthread_mutex_unlock(&lifecycle_mutex);
      return;
    }
  }

  global_buffer_records = buffer_records ? buffer_records : OAI_PROFILE_DEFAULT_BUFFER_RECORDS;
  global_buffer_records = parse_u32_or_default(getenv("OAI_PROFILE_BUFFER_RECORDS"), global_buffer_records);
  if (global_buffer_records < OAI_PROFILE_MIN_BUFFER_RECORDS)
    global_buffer_records = OAI_PROFILE_MIN_BUFFER_RECORDS;
  global_flush_us = flush_us ? flush_us : OAI_PROFILE_DEFAULT_FLUSH_US;
  global_flush_us = parse_u32_or_default(getenv("OAI_PROFILE_FLUSH_US"), global_flush_us);
  if (global_flush_us == 0)
    global_flush_us = OAI_PROFILE_DEFAULT_FLUSH_US;

  if (mkdir_p(output_dir) != 0) {
    LOG_W(UTIL, "OAI profiler disabled: cannot create %s: %s\n", output_dir, strerror(errno));
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }

  counter_hz = read_counter_hz();
  write_event_catalog();
  write_metadata(process_name, argc, argv, global_buffer_records, global_flush_us);
  if (open_profile_file(&events_file, "events.csv", "w") != 0 || open_profile_file(&sync_file, "sync.csv", "w") != 0
      || open_profile_file(&drops_file, "drops.csv", "w") != 0) {
    LOG_W(UTIL, "OAI profiler disabled: cannot open output files under %s\n", output_dir);
    if (events_file != NULL)
      fclose(events_file);
    if (sync_file != NULL)
      fclose(sync_file);
    if (drops_file != NULL)
      fclose(drops_file);
    events_file = NULL;
    sync_file = NULL;
    drops_file = NULL;
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }
  fprintf(events_file,
          "seq,tid,thread_name,event_id,event_name,frame,slot,flags,aux0,aux1,aux2,aux3,start_tick,duration_tick,duration_us\n");
  fprintf(sync_file, "realtime_ns,monotonic_raw_ns,tick\n");
  write_sync_sample();

  profiler_shutdown_requested = false;
  profiler_initialized = true;
  oai_profiler_enabled = 1;
  if (pthread_create(&writer_thread, NULL, profiler_writer_thread, NULL) == 0) {
    writer_started = true;
    LOG_I(UTIL, "OAI profiler enabled, writing to %s\n", output_dir);
  } else {
    oai_profiler_enabled = 0;
    profiler_initialized = false;
    LOG_W(UTIL, "OAI profiler disabled: cannot create writer thread\n");
    fclose(events_file);
    fclose(sync_file);
    fclose(drops_file);
    events_file = NULL;
    sync_file = NULL;
    drops_file = NULL;
  }
  pthread_mutex_unlock(&lifecycle_mutex);
}

void oai_profiler_shutdown(void)
{
  pthread_mutex_lock(&lifecycle_mutex);
  if (!profiler_initialized) {
    pthread_mutex_unlock(&lifecycle_mutex);
    return;
  }
  oai_profiler_enabled = 0;
  profiler_shutdown_requested = true;
  const bool join_writer = writer_started;
  pthread_t writer = writer_thread;
  pthread_mutex_unlock(&lifecycle_mutex);

  if (join_writer)
    pthread_join(writer, NULL);

  pthread_mutex_lock(&lifecycle_mutex);
  for (int i = 0; i < OAI_PROFILE_MAX_THREADS; i++) {
    free(thread_buffers[i].records);
    memset(&thread_buffers[i], 0, sizeof(thread_buffers[i]));
  }
  if (events_file != NULL)
    fclose(events_file);
  if (sync_file != NULL)
    fclose(sync_file);
  if (drops_file != NULL)
    fclose(drops_file);
  events_file = NULL;
  sync_file = NULL;
  drops_file = NULL;
  writer_started = false;
  profiler_initialized = false;
  profiler_shutdown_requested = false;
  pthread_mutex_unlock(&lifecycle_mutex);
}
