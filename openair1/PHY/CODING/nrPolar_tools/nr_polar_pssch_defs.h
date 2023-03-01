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

#ifndef __NR_POLAR_PSSCH_DEFS__H__
#define __NR_POLAR_PSSCH_DEFS__H__

#define NR_POLAR_PSSCH_AGGREGATION_LEVEL 0 //uint8_t
#define NR_POLAR_PSSCH_MESSAGE_TYPE 4      //int8_t
#define NR_POLAR_PSSCH_PAYLOAD_BITS 32     //uint16_t
#define NR_POLAR_PSSCH_CRC_PARITY_BITS 24
#define NR_POLAR_PSSCH_CRC_ERROR_CORRECTION_BITS 3
//Assumed 3 by 3GPP when NR_POLAR_PSSCH_L>8 to meet false alarm rate requirements.

//Sec. 7.1.4: Channel Coding
#define NR_POLAR_PSSCH_N_MAX 9   //uint8_t
#define NR_POLAR_PSSCH_I_IL 1    //uint8_t
#define NR_POLAR_PSSCH_I_SEG 0   //uint8_t
#define NR_POLAR_PSSCH_N_PC 0    //uint8_t
#define NR_POLAR_PSSCH_N_PC_WM 0 //uint8_t
//#define NR_POLAR_PSSCH_N 512     //uint16_t

//Sec. 7.1.5: Rate Matching
#define NR_POLAR_PSSCH_I_BIL 0 //uint8_t
#define NR_POLAR_PSSCH_E 1792   //uint16_t
#define NR_POLAR_PSSCH_E_DWORD 56 // NR_POLAR_PSSCH_E/32

#endif
