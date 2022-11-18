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

/**********************************************************************
*
* FILENAME    :  pss_util_test.h
*
* MODULE      :  UE utilities for test bench of synchronisation
*
* DESCRIPTION :  it allows unitary tests of UE synchronisation on host machine
*
************************************************************************/

#ifndef PSS_UTIL_TEST_H
#define PSS_UTIL_TEST_H

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include "openair1/PHY/defs_nr_UE.h"
#include "openair1/PHY/NR_REFSIG/ss_pbch_nr.h"
#include "openair1/PHY/NR_REFSIG/pss_nr.h"

/************** DEFINE *******************************************/
//#define DEBUG_TEST_PSS
#define PSS_DETECTION_MARGIN_MAX    (4)

/*************** TYPE*********************************************/

typedef struct {
  const char *test_current;
  int number_of_tests;
  int number_of_pass;
  int number_of_pass_warning;
  int number_of_fail;
} test_t;

int test_pss_sl(PHY_VARS_NR_UE *UE);

void test_synchro_pss_sl_nr(PHY_VARS_NR_UE *PHY_VARS_NR_UE, int position_symbol, int pss_sequence_number, test_t *test);

void set_sequence_pss_sl(PHY_VARS_NR_UE *PHY_vars_UE, int position_symbol, int pss_sequence_number);

void set_random_sl_rx_buffer(PHY_VARS_NR_UE *UE, int amp);

int set_pss_sl_in_rx_buffer(PHY_VARS_NR_UE * UE, int position_symbol, int pss_sequence_number);

void display_sl_data(int pss_sequence_number, int16_t *rxdata, int position);
#endif /* PSS_UTIL_TEST_H */