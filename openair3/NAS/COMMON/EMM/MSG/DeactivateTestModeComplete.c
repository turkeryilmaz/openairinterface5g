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

/*! \file DeactivateTestModeComplete.c

\brief test mode procedures for eNB/gNB
\author
\email:
\date 2022
\version 0.1
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "nas_log.h"

#include "DeactivateTestModeComplete.h"

int encode_deactivate_test_mode_complete(deactivate_test_mode_complete_msg *deactivate_test_mode_complete, uint8_t *buffer, uint32_t len)
{
  int encoded = 0;
  /* *(buffer + encoded) = deactivate_test_mode_complete->protocoldiscriminator; */
  /* encoded++; */
  /* *(buffer + encoded) = deactivate_test_mode_complete->messagetype; */
  /* encoded++; */
  return encoded;
}
