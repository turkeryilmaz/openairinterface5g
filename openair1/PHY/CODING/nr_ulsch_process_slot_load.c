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

/*! \file openair1/PHY/CODING/coding_nr_load.c
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
#define LDPC_LOADER
#include "common/config/config_userapi.h" 
#include "common/utils/load_module_shlib.h" 
#include "PHY/CODING/nr_ulsch_process_slot.h"

int load_nr_ulsch_process_slot(void) {

     loader_shlibfunc_t shlib_nr_ulsch_process_slot_fdesc; 
     shlib_nr_ulsch_process_slot_fdesc.fname = "nr_ulsch_process_slot";
     int ret=load_module_shlib("_nr_ulsch_process_slot",&shlib_nr_ulsch_process_slot_fdesc,1,NULL);
     AssertFatal( (ret >= 0),"Error loading nr_ulsch_process_slot");
     nr_ulsch_process_slot = (nr_ulsch_process_slot_t)shlib_nr_ulsch_process_slot_fdesc.fptr;

  return 0;
}

