/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int test_fstat(int file_descriptor, struct stat *status);

#define fstat test_fstat
#define main oai_memprof_archive_append_main
#include "../oai_memprof_archive_append.c"
#undef main
#undef fstat

static int selected_auxiliary_fd = -1;
static unsigned int selected_auxiliary_fstat_calls;
static int replacement_directory_fd = -1;
static const char *replacement_leaf_name;
static const char *replacement_source_name;
static int replacement_rename_result = -1;

static int test_fstat(int file_descriptor, struct stat *status)
{
  const int result = fstat(file_descriptor, status);
  if (result != 0)
    return result;
  if (selected_auxiliary_fd < 0)
    selected_auxiliary_fd = file_descriptor;
  if (file_descriptor == selected_auxiliary_fd) {
    ++selected_auxiliary_fstat_calls;
    if (selected_auxiliary_fstat_calls == 2)
      replacement_rename_result =
          renameat(replacement_directory_fd, replacement_source_name, replacement_directory_fd, replacement_leaf_name);
  }
  return result;
}

static bool expect(bool condition, const char *message)
{
  if (condition)
    return true;
  fprintf(stderr, "test_oai_memprof_archive_append: %s\n", message);
  return false;
}

static bool write_full(int file_descriptor, const uint8_t *bytes, size_t size)
{
  size_t offset = 0;
  while (offset != size) {
    const ssize_t count = write(file_descriptor, bytes + offset, size - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0)
      return false;
    offset += (size_t)count;
  }
  return true;
}

static bool write_leaf(int directory_fd, const char *name, const uint8_t *bytes, size_t size)
{
  const int file_descriptor = openat(directory_fd, name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
  if (file_descriptor < 0)
    return false;
  bool ok = write_full(file_descriptor, bytes, size);
  if (close(file_descriptor) != 0)
    ok = false;
  return ok;
}

static bool read_leaf_exact(int directory_fd, const char *name, uint8_t *bytes, size_t size)
{
  const int file_descriptor = openat(directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (file_descriptor < 0)
    return false;
  bool ok = true;
  size_t offset = 0;
  while (offset != size) {
    const ssize_t count = read(file_descriptor, bytes + offset, size - offset);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      ok = false;
      break;
    }
    offset += (size_t)count;
  }
  uint8_t extra_byte = 0;
  if (ok && read(file_descriptor, &extra_byte, sizeof(extra_byte)) != 0)
    ok = false;
  if (close(file_descriptor) != 0)
    ok = false;
  return ok;
}

static bool capture_pipe_text(int file_descriptor, char *text, size_t capacity)
{
  if (text == NULL || capacity == 0)
    return false;
  size_t offset = 0;
  for (;;) {
    char buffer[64];
    const ssize_t count = read(file_descriptor, buffer, sizeof(buffer));
    if (count < 0 && errno == EINTR)
      continue;
    if (count < 0)
      return false;
    if (count == 0)
      break;
    if ((size_t)count >= capacity - offset)
      return false;
    memcpy(text + offset, buffer, (size_t)count);
    offset += (size_t)count;
  }
  text[offset] = '\0';
  return true;
}

static bool capture_contract_probe(int *return_code,
                                   char *stdout_text,
                                   size_t stdout_capacity,
                                   char *stderr_text,
                                   size_t stderr_capacity)
{
  int stdout_pipe[2] = {-1, -1};
  int stderr_pipe[2] = {-1, -1};
  int saved_stdout = -1;
  int saved_stderr = -1;
  bool stdout_redirected = false;
  bool stderr_redirected = false;
  bool ok = return_code != NULL && pipe(stdout_pipe) == 0 && pipe(stderr_pipe) == 0;
  if (ok && (fflush(stdout) == EOF || fflush(stderr) == EOF))
    ok = false;
  if (ok) {
    saved_stdout = dup(STDOUT_FILENO);
    saved_stderr = dup(STDERR_FILENO);
    ok = saved_stdout >= 0 && saved_stderr >= 0;
  }
  if (ok) {
    stdout_redirected = dup2(stdout_pipe[1], STDOUT_FILENO) >= 0;
    stderr_redirected = dup2(stderr_pipe[1], STDERR_FILENO) >= 0;
    ok = stdout_redirected && stderr_redirected;
  }
  if (stdout_pipe[1] >= 0 && close(stdout_pipe[1]) != 0)
    ok = false;
  stdout_pipe[1] = -1;
  if (stderr_pipe[1] >= 0 && close(stderr_pipe[1]) != 0)
    ok = false;
  stderr_pipe[1] = -1;
  if (ok) {
    char executable_name[] = "oai_memprof_archive_append";
    char probe_argument[] = ARCHIVE_APPEND_CONTRACT_PROBE_ARGUMENT;
    char *arguments[] = {executable_name, probe_argument, NULL};
    *return_code = oai_memprof_archive_append_main(2, arguments);
    if (fflush(stdout) == EOF || fflush(stderr) == EOF)
      ok = false;
  }
  if (stdout_redirected && dup2(saved_stdout, STDOUT_FILENO) < 0)
    ok = false;
  if (stderr_redirected && dup2(saved_stderr, STDERR_FILENO) < 0)
    ok = false;
  if (saved_stdout >= 0 && close(saved_stdout) != 0)
    ok = false;
  if (saved_stderr >= 0 && close(saved_stderr) != 0)
    ok = false;
  if (stdout_pipe[0] >= 0) {
    if (!capture_pipe_text(stdout_pipe[0], stdout_text, stdout_capacity))
      ok = false;
    if (close(stdout_pipe[0]) != 0)
      ok = false;
  } else {
    ok = false;
  }
  if (stderr_pipe[0] >= 0) {
    if (!capture_pipe_text(stderr_pipe[0], stderr_text, stderr_capacity))
      ok = false;
    if (close(stderr_pipe[0]) != 0)
      ok = false;
  } else {
    ok = false;
  }
  return ok;
}

int main(void)
{
  static const uint8_t accepted_leaf[] = "AUXILIARY-OBJECT-A";
  static const uint8_t replacement_leaf[] = "AUXILIARY-OBJECT-B";
  static const uint8_t prefooter_sentinel[] = "PREFOOTER-SENTINEL";
  static uint8_t digest_input[] = "AUTHENTICATED-HANDOFF";
  static const char expected_digest_hex[] = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
  static const char auxiliary_name[] = "handoff.bin";
  static const char replacement_name[] = "handoff-replacement.bin";
  static const char prefooter_name[] = "prefooter.stream";
  char directory_path[] = "/tmp/oai_memprof_archive_append.XXXXXX";
  char *directory_path_result = NULL;
  int directory_fd = -1;
  immutable_file_t result = {0};
  struct stat accepted_status = {0};
  struct stat named_status = {0};
  uint8_t observed_replacement[sizeof(replacement_leaf)] = {0};
  uint8_t observed_prefooter[sizeof(prefooter_sentinel)] = {0};
  uint8_t expected_digest[32] = {0};
  uint8_t decoded_digest[32] = {0};
  immutable_file_t digest_file = {.bytes = digest_input, .size = sizeof(digest_input)};
  int contract_probe_status = -1;
  char contract_probe_stdout[128] = {0};
  char contract_probe_stderr[128] = {0};
  bool ok = true;

  if (ok)
    ok = expect(capture_contract_probe(&contract_probe_status,
                                       contract_probe_stdout,
                                       sizeof(contract_probe_stdout),
                                       contract_probe_stderr,
                                       sizeof(contract_probe_stderr)),
                "could not capture appender contract probe");
  if (ok)
    ok = expect(contract_probe_status == 0, "appender contract probe did not succeed");
  if (ok)
    ok = expect(strcmp(contract_probe_stdout, ARCHIVE_APPEND_CONTRACT_PROBE_OUTPUT) == 0, "appender contract probe stdout differs");
  if (ok)
    ok = expect(contract_probe_stderr[0] == '\0', "appender contract probe emitted stderr");
  if (ok) {
    directory_path_result = mkdtemp(directory_path);
    ok = directory_path_result != NULL;
  }

  if (ok)
    ok = expect(decode_sha256_hex(expected_digest_hex, decoded_digest), "expected digest hex did not decode");
  if (ok)
    ok = expect(decoded_digest[0] == UINT8_C(0x01) && decoded_digest[31] == UINT8_C(0xef), "decoded digest bytes differ");
  if (ok)
    ok = expect(!decode_sha256_hex("not-a-sha256", decoded_digest), "malformed digest was accepted");
  if (ok)
    ok = expect(oai_memprof_container_v1_sha256(digest_file.bytes, digest_file.size, expected_digest) == OAI_MEMPROF_CONTAINER_V1_OK
                    && immutable_file_sha256_matches(&digest_file, expected_digest),
                "exact handoff digest was not accepted");
  if (ok) {
    expected_digest[0] ^= UINT8_C(1);
    ok = expect(!immutable_file_sha256_matches(&digest_file, expected_digest), "substituted handoff digest was accepted");
  }

  if (ok) {
    directory_fd = open(directory_path, O_RDONLY | O_CLOEXEC | O_DIRECTORY | O_NOFOLLOW);
    ok = directory_fd >= 0;
  }
  if (ok)
    ok = write_leaf(directory_fd, auxiliary_name, accepted_leaf, sizeof(accepted_leaf));
  if (ok)
    ok = write_leaf(directory_fd, replacement_name, replacement_leaf, sizeof(replacement_leaf));
  if (ok)
    ok = write_leaf(directory_fd, prefooter_name, prefooter_sentinel, sizeof(prefooter_sentinel));
  if (ok)
    ok = fstatat(directory_fd, auxiliary_name, &accepted_status, AT_SYMLINK_NOFOLLOW) == 0;

  replacement_directory_fd = directory_fd;
  replacement_leaf_name = auxiliary_name;
  replacement_source_name = replacement_name;
  const bool read_ok = ok && read_immutable_leaf(directory_fd, auxiliary_name, sizeof(accepted_leaf), &result);
  if (ok)
    ok = expect(selected_auxiliary_fstat_calls == 2, "expected two descriptor fstat calls");
  if (ok)
    ok = expect(replacement_rename_result == 0, "deterministic replacement rename failed");
  if (ok)
    ok = expect(!read_ok, "detached auxiliary descriptor bytes were accepted");
  if (ok)
    ok = expect(result.bytes == NULL && result.size == 0, "failed read returned auxiliary bytes");
  if (ok)
    ok = expect(fstatat(directory_fd, auxiliary_name, &named_status, AT_SYMLINK_NOFOLLOW) == 0, "replacement leaf is unavailable");
  if (ok)
    ok = expect(named_status.st_dev != accepted_status.st_dev || named_status.st_ino != accepted_status.st_ino,
                "named auxiliary leaf still identifies the accepted descriptor");
  if (ok)
    ok = expect(read_leaf_exact(directory_fd, auxiliary_name, observed_replacement, sizeof(observed_replacement)),
                "could not read replaced auxiliary leaf");
  if (ok)
    ok = expect(memcmp(observed_replacement, replacement_leaf, sizeof(replacement_leaf)) == 0,
                "named auxiliary leaf does not contain replacement bytes");
  if (ok)
    ok = expect(read_leaf_exact(directory_fd, prefooter_name, observed_prefooter, sizeof(observed_prefooter)),
                "could not read sentinel prefooter stream");
  if (ok)
    ok = expect(memcmp(observed_prefooter, prefooter_sentinel, sizeof(prefooter_sentinel)) == 0,
                "unrelated prefooter stream changed");

  free(result.bytes);
  if (directory_fd >= 0) {
    (void)unlinkat(directory_fd, auxiliary_name, 0);
    (void)unlinkat(directory_fd, replacement_name, 0);
    (void)unlinkat(directory_fd, prefooter_name, 0);
    if (close(directory_fd) != 0)
      ok = false;
  }
  if (directory_path_result != NULL && rmdir(directory_path) != 0)
    ok = false;
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
