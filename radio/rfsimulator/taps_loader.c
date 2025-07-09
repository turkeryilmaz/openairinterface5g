#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <common/utils/LOG/log.h>
#include "openair1/SIMULATION/TOOLS/sim.h"
#include "radio/rfsimulator/taps_loader.h"

#define TAPSLOADER_LOG_MAXSNAP 5
// To debug tap selection, enable this macro to print selected taps per snapshot
// #define TAPSLOADER_LOG_TAPS

/**
 * @brief Loads external CIR snapshots from a binary float32 dataset and selects top-N taps per snapshot.
 * See taps_loader.h for binary format and parameter description.
 *
 * @param[in]  bin_path   Path to binary file (float32, real/imag interleaved per snapshot)
 * @param[out] desc       Pointer to channel descriptor to populate
 * @return     0 on success, negative error code on failure
 */
static int cmp_idx_desc(const void *a, const void *b, void *energies_v)
{
  const double *E = energies_v;
  int ia = *(const int *)a, ib = *(const int *)b;
  double eA = E[ia], eB = E[ib];
  return (eA < eB) - (eA > eB);
}

static int compare_ints(const void *a, const void *b)
{
  return (*(int *)a - *(int *)b);
}

int load_external_taps_binary(const char *bin_path, channel_desc_t *desc)
{
  if (!desc) {
    LOG_E(HW, "Error: desc pointer is NULL\n");
    return -9;
  }

  // TEMP: For debugging, override bin_path with known absolute path.
  // TODO: Remove this hardcode after filesystem/path issues are resolved.
  // Original bin_path: %s
  bin_path = "../../cir_datasets/urban_macro/SionnaRT_CIR.bin";

  FILE *f = fopen(bin_path, "rb");
  if (!f) {
    LOG_E(HW, "Error -1: Cannot open CIR file: %s (errno=%d, %s)\n", bin_path, errno, strerror(errno));
    return -1;
  }

  if (fseek(f, 0, SEEK_END) != 0 || ftell(f) < 0) {
    LOG_E(HW, "Error -2: Failed to determine file size\n");
    fclose(f);
    return -2;
  }

  long bytes = ftell(f);
  rewind(f);

  uint8_t L = desc->external_cir_len;
  uint32_t M = desc->max_loaded_taps;
  if (L == 0) {
    LOG_E(HW, "Error -3: external_cir_len is zero\n");
    fclose(f);
    return -3;
  }

  if (M == 0 || M > L)
    M = L;

  long total_float = bytes / sizeof(float);
  if (total_float % (2 * L)) {
    LOG_E(HW, "Error -4: File size is not a multiple of 2 * L (L = %u)\n", L);
    fclose(f);
    return -4;
  }

  uint32_t count = total_float / (2 * L);
  LOG_I(HW, "CIR binary '%s' contains %u snapshots of %u taps\n", bin_path, count, L);

  float *real_buf = malloc(sizeof(*real_buf) * count * L);
  float *imag_buf = malloc(sizeof(*imag_buf) * count * L);
  if (!real_buf || !imag_buf) {
    LOG_E(HW, "Error -5: Memory allocation failed\n");
    free(real_buf);
    free(imag_buf);
    fclose(f);
    return -5;
  }

  if (fread(real_buf, sizeof(float), count * L, f) != count * L || fread(imag_buf, sizeof(float), count * L, f) != count * L) {
    LOG_E(HW, "Error -6: Failed to read CIR data\n");
    free(real_buf);
    free(imag_buf);
    fclose(f);
    return -6;
  }

  fclose(f);

  struct complexd *tap_data = calloc(count * M, sizeof(*tap_data));
  if (!tap_data) {
    LOG_E(HW, "Error -7: Failed to allocate tap_data\n");
    free(real_buf);
    free(imag_buf);
    return -7;
  }

  double *energies = malloc(sizeof(*energies) * L);
  int *indices = malloc(sizeof(*indices) * L);
  int *top_idx = malloc(sizeof(*top_idx) * M);
  if (!energies || !indices || !top_idx) {
    LOG_E(HW, "Error -8: Failed to allocate sorting buffers\n");
    free(real_buf);
    free(imag_buf);
    free(tap_data);
    free(energies);
    free(indices);
    free(top_idx);
    return -8;
  }

  for (uint32_t s = 0; s < count; s++) {
    for (uint8_t t = 0; t < L; t++) {
      double r = (double)real_buf[s * L + t];
      double i = (double)imag_buf[s * L + t];
      energies[t] = r * r + i * i;
      indices[t] = t;
    }

    qsort_r(indices, L, sizeof(int), cmp_idx_desc, energies); // Sort indices by descending energy
    memcpy(top_idx, indices, sizeof(int) * M); // Select top M indices
    qsort(top_idx, M, sizeof(int), compare_ints); // Restore original tap order (causality)

    for (uint8_t m = 0; m < M; m++) {
      int t = top_idx[m];
      tap_data[s * M + m].r = real_buf[s * L + t];
      tap_data[s * M + m].i = imag_buf[s * L + t];
    }

    /* --------------- TAP LOGGING ---------------
     * By default, all CIR tap logs are disabled for performance and clarity.
     * Developers can enable logging for debugging by uncommenting the macro above.
     * Example log will output the selected taps and energies for each snapshot.
     */
#ifdef TAPSLOADER_LOG_TAPS
    if (s < TAPSLOADER_LOG_MAXSNAP) {
      LOG_I(HW, "\n================= [TAPS] Snapshot #%u: Selected Top-%u CIR Taps =================", s, M);
      for (uint8_t m = 0; m < M; m++) {
        int t = top_idx[m];
        double r = real_buf[s * L + t];
        double i = imag_buf[s * L + t];
        double e = r * r + i * i;
        LOG_I(HW, "[TAPS]   Tap[%3d] = ( %+1.6f + %+1.6fj ) | Energy = %.3e", t, r, i, e);
      }
      LOG_I(HW, "[TAPS]   --> Stored taps (Causal Order):");
      for (int m = 0; m < M; m++)
        LOG_I(HW, "[TAPS]       [%2d] ( %+1.6f + %+1.6fj )", m, tap_data[s * M + m].r, tap_data[s * M + m].i);
      LOG_I(HW, "=================================================================================\n");
    }
#endif
  }

  free(real_buf);
  free(imag_buf);
  free(energies);
  free(indices);
  free(top_idx);

  desc->external_cir_count = count;
  desc->external_cir_len = M;
  desc->external_cir_idx = 0;
  desc->external_cir_data = tap_data;

  if (desc->external_cir_file)
    free(desc->external_cir_file);
  desc->external_cir_file = strdup(bin_path);

  LOG_I(HW, "Successfully loaded %u snapshots, %u taps each (top %u retained)\n", count, L, M);
  return 0;
}
