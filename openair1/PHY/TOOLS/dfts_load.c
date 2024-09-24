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

/*! \file dfts_load.c
 * \brief: load library implementing coding/decoding algorithms
 * \author Francois TABURET
 * \date 2020
 * \version 0.1
 * \company NOKIA BellLabs France
 * \email: francois.taburet@nokia-bell-labs.com
 * \note
 * \warning
 */
#define _GNU_SOURCE 
#include <sys/types.h>
#include <stdlib.h>
#include <malloc.h>
#include "assertions.h"
#include "common/utils/LOG/log.h"
#define OAIDFTS_LOADER
#include "tools_defs.h"
#include "common/config/config_userapi.h" 
#include "common/utils/load_module_shlib.h" 

uint32_t DFT_SCALING_128[2] = {2,2};
uint32_t DFT_SCALING_256[2] = {2,2};
uint32_t DFT_SCALING_512[3] = {1,2,2};
uint32_t DFT_SCALING_768[3] = {1,2,2};
uint32_t DFT_SCALING_1024[3] = {1,2,2};
uint32_t DFT_SCALING_1536[3] = {1,2,2};
uint32_t DFT_SCALING_2048[4] = {1,0,3,2};
uint32_t DFT_SCALING_3072[4] = {1,0,3,2};
uint32_t DFT_SCALING_4096[4] = {0,0,3,3};
uint32_t DFT_SCALING_6144[5] = {1,0,0,3,3};
uint32_t DFT_SCALING_8192[5] = {1,0,0,3,3};
uint32_t DFT_SCALING_9216[5] = {1,0,0,3,3};
uint32_t DFT_SCALING_12288[5] = {1,0,0,3,3};
uint32_t DFT_SCALING_16384[5] = {0,0,1,3,3};
uint32_t DFT_SCALING_18432[6] = {1,1,0,0,3,3};
uint32_t DFT_SCALING_24576[6] = {1,1,0,0,3,3};
uint32_t DFT_SCALING_32768[6] = {1,0,0,1,3,3};
uint32_t DFT_SCALING_36864[6] = {1,1,0,0,3,3};
uint32_t DFT_SCALING_49152[6] = {1,0,0,1,3,3};
uint32_t DFT_SCALING_65536[6] = {0,0,0,2,3,3};
uint32_t DFT_SCALING_73728[7] = {1,1,1,0,0,3,3};
uint32_t DFT_SCALING_98304[7] = {1,1,0,0,1,3,3};

uint32_t IDFT_SCALING_128[2] = {2,2};
uint32_t IDFT_SCALING_256[2] = {2,2};
uint32_t IDFT_SCALING_512[3] = {1,2,2};
uint32_t IDFT_SCALING_768[3] = {1,2,2};
uint32_t IDFT_SCALING_1024[3] = {1,2,2};
uint32_t IDFT_SCALING_1536[3] = {1,2,2};
uint32_t IDFT_SCALING_2048[4] = {1,0,3,2};
uint32_t IDFT_SCALING_3072[4] = {1,0,3,2};
uint32_t IDFT_SCALING_4096[4] = {0,0,3,3};
uint32_t IDFT_SCALING_6144[5] = {1,0,0,3,3};
uint32_t IDFT_SCALING_8192[5] = {1,0,0,3,3};
uint32_t IDFT_SCALING_9216[5] = {1,0,0,3,3};
uint32_t IDFT_SCALING_12288[5] = {1,0,0,3,3};
uint32_t IDFT_SCALING_16384[5] = {0,0,1,3,3};
uint32_t IDFT_SCALING_18432[6] = {1,1,0,0,3,3};
uint32_t IDFT_SCALING_24576[6] = {1,1,0,0,3,3};
uint32_t IDFT_SCALING_32768[6] = {1,0,0,1,3,3};
uint32_t IDFT_SCALING_36864[6] = {1,1,0,0,3,3};
uint32_t IDFT_SCALING_49152[6] = {1,0,0,1,3,3};
uint32_t IDFT_SCALING_65536[6] = {0,0,0,2,3,3};
uint32_t IDFT_SCALING_73728[7] = {1,1,1,0,0,3,3};
uint32_t IDFT_SCALING_98304[7] = {1,1,0,0,1,3,3};

/* function description array, to be used when loading the dfts/idfts lib */
static loader_shlibfunc_t shlib_fdesc[2];
static char *arg[64] = {"phytest", "-O", "cmdlineonly::dbgl0"};
int load_dftslib(void)
{
  char *ptr = (char *)config_get_if();
  if (ptr == NULL) { // phy simulators, config module possibly not loaded
    uniqCfg = load_configmodule(3, (char **)arg, CONFIG_ENABLECMDLINEONLY);
    logInit();
  }
  shlib_fdesc[0].fname = "dft";
  shlib_fdesc[1].fname = "idft";
  int ret = load_module_shlib("dfts", shlib_fdesc, sizeof(shlib_fdesc) / sizeof(loader_shlibfunc_t), NULL);
  AssertFatal((ret >= 0), "Error loading dftsc decoder");
  dft = (dftfunc_t)shlib_fdesc[0].fptr;
  idft = (idftfunc_t)shlib_fdesc[1].fptr;
  return 0;
}
