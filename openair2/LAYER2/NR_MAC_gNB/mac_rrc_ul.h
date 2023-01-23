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
 *      conmnc_digit_lengtht@openairinterface.org
 */

<<<<<<< HEAD:openair2/LAYER2/NR_MAC_gNB/mac_rrc_ul.h
#ifndef MAC_RRC_UL_H
#define MAC_RRC_UL_H

#include "platform_types.h"
#include "f1ap_messages_types.h"

typedef void (*ue_context_setup_response_func_t)(const f1ap_ue_context_setup_t* req, const f1ap_ue_context_setup_t *resp);

typedef void (*initial_ul_rrc_message_transfer_func_t)(module_id_t module_id, const f1ap_initial_ul_rrc_message_t *ul_rrc);

struct nr_mac_rrc_ul_if_s;
void mac_rrc_ul_direct_init(struct nr_mac_rrc_ul_if_s *mac_rrc);
void mac_rrc_ul_f1ap_init(struct nr_mac_rrc_ul_if_s *mac_rrc);

#endif /* MAC_RRC_UL_H */
=======
#include <stdio.h>
#include <stdint.h>
#define NB_R  3
void nrLDPC_cnProc_BG1_generator_AVX2(const char*, int);
void nrLDPC_cnProc_BG2_generator_AVX2(const char*, int);

const char *__asan_default_options()
{
  /* don't do leak checking in nr_ulsim, creates problems in the CI */
  return "detect_leaks=0";
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s <output-dir>\n", argv[0]);
    return 1;
  }
  const char *dir = argv[1];

  int R[NB_R]={0,1,2};
  for(int i=0; i<NB_R;i++) {
    nrLDPC_cnProc_BG1_generator_AVX2(dir, R[i]);
    nrLDPC_cnProc_BG2_generator_AVX2(dir, R[i]);
  }

  return(0);
}

>>>>>>> 41ca5b1258... Rebasing openair1 folder:openair1/PHY/CODING/nrLDPC_decoder/nrLDPC_tools/generator_cnProc/main.c
