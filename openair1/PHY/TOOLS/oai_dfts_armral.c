/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include <armral.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <execinfo.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define debug_msg
#define ONE_OVER_SQRT2_Q15 23170
#define ONE_OVER_SQRT3_Q15 18919

#include "assertions.h"
#include "LOG/log.h"
#include "time_meas.h"
#include "tools_defs.h"
#include "../sse_intrin.h"

#define DEFAULT_NEON

#ifdef DEFAULT_NEON
#include "common/config/config_userapi.h" 
#include "common/utils/load_module_shlib.h" 
static loader_shlibfunc_t shlib_fdesc[4];
static char *arg[64] = {"phytest", "-O", "cmdlineonly::dbgl0"};
dftfunc_t dft_neon;
idftfunc_t idft_neon;
dfts_start_t dfts_start_neon;
dfts_stop_t dfts_stop_neon;
#endif

#define SZ_PTR(Sz) Sz,
const int dft_stab[] = {FOREACH_DFTSZ(SZ_PTR)};

#define SZ_iPTR(Sz)  Sz,
const int idft_stab[] = {FOREACH_IDFTSZ(SZ_iPTR)};

// Pre-allocated plans for PRACH
armral_fft_plan_t *idft_256_cs16_plan_p;
armral_fft_plan_t *idft_1024_cs16_plan_p;

void armral_dft(uint8_t sizeidx, int16_t *x, int16_t *y, unsigned char scale_flag)
{
#ifdef DEFAULT_NEON
  dft_neon(sizeidx, x, y, scale_flag);
#else
  armral_status status;
  switch (sizeidx) {
    default:
      armral_fft_plan_t *dft_cs16_plan_p;
      status = armral_fft_create_plan_cs16(&dft_cs16_plan_p, idft_stab[sizeidx], ARMRAL_FFT_FORWARDS);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT create plan\n");
      status = armral_fft_execute_cs16(dft_cs16_plan_p, (armral_cmplx_int16_t *)x, (armral_cmplx_int16_t *)y);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT\n");
      status = armral_fft_destroy_plan_cs16(&dft_cs16_plan_p);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT destroy plan\n");
      if (scale_flag > 0) {
        simd_q15_t *y128 = (simd_q15_t *)y;
        int sz = dft_stab[sizeidx];
        for (int i = 0; i < (sz / 4); i++) {
          y128[i] = shiftright_int16(y128[i], 1);
        }
      }
      break;
  }
#endif
}

void armral_idft(uint8_t sizeidx, int16_t *x, int16_t *y, unsigned char scale_flag)
{
  armral_status status;
  switch (sizeidx) {
    case IDFT_256:
      status = armral_fft_execute_cs16(idft_256_cs16_plan_p, (armral_cmplx_int16_t *)x, (armral_cmplx_int16_t *)y);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT\n");
      if (scale_flag > 0) {
        simd_q15_t *y128 = (simd_q15_t *)y;
        for (int i = 0; i < 4; i++) {
          y128[0] = shiftright_int16(y128[0], 1);
          y128[1] = shiftright_int16(y128[1], 1);
          y128[2] = shiftright_int16(y128[2], 1);
          y128[3] = shiftright_int16(y128[3], 1);
          y128[4] = shiftright_int16(y128[4], 1);
          y128[5] = shiftright_int16(y128[5], 1);
          y128[6] = shiftright_int16(y128[6], 1);
          y128[7] = shiftright_int16(y128[7], 1);
          y128[8] = shiftright_int16(y128[8], 1);
          y128[9] = shiftright_int16(y128[9], 1);
          y128[10] = shiftright_int16(y128[10], 1);
          y128[11] = shiftright_int16(y128[11], 1);
          y128[12] = shiftright_int16(y128[12], 1);
          y128[13] = shiftright_int16(y128[13], 1);
          y128[14] = shiftright_int16(y128[14], 1);
          y128[15] = shiftright_int16(y128[15], 1);
          y128 += 16;
        }
      }
      break;
    case IDFT_1024:
      status = armral_fft_execute_cs16(idft_1024_cs16_plan_p, (armral_cmplx_int16_t *)x, (armral_cmplx_int16_t *)y);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT\n");
      if (scale_flag > 0) {
        simd_q15_t *y128 = (simd_q15_t *)y;
        for (int i = 0; i < 16; i++) {
          y128[0] = shiftright_int16(y128[0], 1);
          y128[1] = shiftright_int16(y128[1], 1);
          y128[2] = shiftright_int16(y128[2], 1);
          y128[3] = shiftright_int16(y128[3], 1);
          y128[4] = shiftright_int16(y128[4], 1);
          y128[5] = shiftright_int16(y128[5], 1);
          y128[6] = shiftright_int16(y128[6], 1);
          y128[7] = shiftright_int16(y128[7], 1);
          y128[8] = shiftright_int16(y128[8], 1);
          y128[9] = shiftright_int16(y128[9], 1);
          y128[10] = shiftright_int16(y128[10], 1);
          y128[11] = shiftright_int16(y128[11], 1);
          y128[12] = shiftright_int16(y128[12], 1);
          y128[13] = shiftright_int16(y128[13], 1);
          y128[14] = shiftright_int16(y128[14], 1);
          y128[15] = shiftright_int16(y128[15], 1);
          y128 += 16;
        }
      }
      break;
    default:
#ifdef DEFAULT_NEON
      idft_neon(sizeidx, x, y, scale_flag);
#else
      armral_fft_plan_t *idft_cs16_plan_p;
      status = armral_fft_create_plan_cs16(&idft_cs16_plan_p, idft_stab[sizeidx], ARMRAL_FFT_BACKWARDS);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT create plan\n");
      status = armral_fft_execute_cs16(idft_cs16_plan_p, (armral_cmplx_int16_t *)x, (armral_cmplx_int16_t *)y);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT\n");
      status = armral_fft_destroy_plan_cs16(&idft_cs16_plan_p);
      AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT destroy plan\n");
      if (scale_flag > 0) {
        simd_q15_t *y128 = (simd_q15_t *)y;
        int sz = idft_stab[sizeidx];
        for (int i = 0; i < (sz / 4); i++) {
          y128[i] = shiftright_int16(y128[i], 1);
        }
      }
#endif
      break;
  }
}

void dft_implementation(uint8_t sizeidx, int16_t *input, int16_t *output, unsigned char scale_flag)
{
  AssertFatal((sizeidx >= 0 && sizeidx < DFT_SIZE_IDXTABLESIZE), "Invalid dft size index %i\n", sizeidx);
  int algn = 0xF;
  AssertFatal(((intptr_t)output & algn) == 0, "Buffers should be aligned %p", output);

  if (((intptr_t)input) & algn) {
    int sz = dft_stab[sizeidx];
    LOG_D(PHY, "DFT called with input not aligned, add a memcpy, size %d\n", sz);
    if (sizeidx == DFT_12) // This case does 8 DFTs in //
      sz *= 8;
    int16_t tmp[sz * 2] __attribute__((aligned(32))); // input and output are not in right type (int16_t instead of c16_t)
    memcpy(tmp, input, sizeof(tmp));
    armral_dft(sizeidx, tmp, output, scale_flag);
  } else {
    armral_dft(sizeidx, input, output, scale_flag);
  }
}

void idft_implementation(uint8_t sizeidx, int16_t *input, int16_t *output, unsigned char scale_flag)
{
  AssertFatal((sizeidx >= 0 && sizeidx < DFT_SIZE_IDXTABLESIZE), "Invalid idft size index %i\n", sizeidx);
  int algn = 0xF;
  AssertFatal(((intptr_t)output & algn) == 0, "Buffers should be 16 bytes aligned %p", output);

  if (((intptr_t)input) & algn) {
    int sz = idft_stab[sizeidx];
    LOG_D(PHY, "IDFT called with input not aligned, add a memcpy, size %d\n", sz);
    int16_t tmp[sz * 2] __attribute__((aligned(32))); // input and output are not in right type (int16_t instead of c16_t)
    memcpy(tmp, input, sizeof(tmp));
    armral_idft(sizeidx, tmp, output, scale_flag);
  } else {
    armral_idft(sizeidx, input, output, scale_flag);
  }
}

void dfts_start()
{
  armral_status status;
  status = armral_fft_create_plan_cs16(&idft_256_cs16_plan_p, 256, ARMRAL_FFT_BACKWARDS);
  AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT create plan\n");
  status = armral_fft_create_plan_cs16(&idft_1024_cs16_plan_p, 1024, ARMRAL_FFT_BACKWARDS);
  AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT create plan\n");
#ifdef DEFAULT_NEON
  char *ptr = (char *)config_get_if();
  if (ptr == NULL) { // phy simulators, config module possibly not loaded
    uniqCfg = load_configmodule(3, (char **)arg, CONFIG_ENABLECMDLINEONLY);
    logInit();
  }
  shlib_fdesc[0].fname = "dft_implementation";
  shlib_fdesc[1].fname = "idft_implementation";
  shlib_fdesc[2].fname = "dfts_start";
  shlib_fdesc[3].fname = "dfts_stop";
  int ret = load_module_version_shlib("dfts", "", shlib_fdesc, sizeof(shlib_fdesc) / sizeof(loader_shlibfunc_t), NULL);
  AssertFatal((ret >= 0), "Error loading dfts decoder");

  dft_neon = (dftfunc_t)shlib_fdesc[0].fptr;
  idft_neon = (idftfunc_t)shlib_fdesc[1].fptr;
  dfts_start_neon = (dfts_start_t)shlib_fdesc[2].fptr;
  dfts_stop_neon = (dfts_stop_t)shlib_fdesc[3].fptr;

  dfts_start_neon();
#endif
}

void dfts_stop()
{
  armral_status status;
  status = armral_fft_destroy_plan_cs16(&idft_256_cs16_plan_p);
  AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT destroy plan\n");
  status = armral_fft_destroy_plan_cs16(&idft_1024_cs16_plan_p);
  AssertFatal(status == ARMRAL_SUCCESS, "Failure in ArmRAL FFT destroy plan\n");
#ifdef DEFAULT_NEON
  dfts_stop_neon();
#endif
}
