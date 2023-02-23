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

/*! \file ss_config.h
* \brief System Simulator configuration struct definitions and function prototypes
* \author Anurag Asokan
* \date 2022
* \version 0.1
* \company Firecell
* \email: anurag.ashokan@firecell.io
*/

#ifndef __SS_CONFIG_H__
#define __SS_CONFIG_H__

#include "openair2/LAYER2/PDCP_v10.1.0/pdcp.h"

#if 0 /** TODO: FC */
typedef enum ss_mode {
  SS_ENB = 0x00,
  SS_VT,
  SS_VT_SRB,
} ss_mode_e;
#else
#define SS_ENB  		(0x00)
#define SS_GNB                  (0x00)
#define SS_SOFTMODEM    	(0x01)
#define SS_SOFTMODEM_SRB        (0x02)
#endif

typedef struct ss_config_s {
  /** SS mode of operation */
  uint8_t mode; /* 0: Normal eNB/gNB, 1: SS-Mode */
  /** IP Address where the TTCN engine is running */
  const char* hostIp;
  /** SYS port number */
  int Sysport;
  /** SRB port number */
  int Srbport;
  /** VNG port number */
  int Vngport;
  /** VTP port number */
  int Vtpport;
  /** SYSIND port number */
  int SysIndport;
  /** DRB port number */
  int Drbport;
  /** SYS Port NR number */
  int SysportNR;
  uint8_t configured;
  /** State info */
  int State;
  int vtp_ready;
  int CBRA_flag;
  int mac_rlc_data_ind_frame;
  int mac_rlc_data_ind_subframe;
  ss_l1macind_ctrl_t l1macind[MAX_NUM_CCs];
  uint8_t CC_conf_flag[MAX_NUM_CCs];
  ss_rrc_pdcp_api_t *ss_pdcp_api;
} ss_config_t;
/**
typedef enum {
  SS_STATE_NOT_CONFIGURED = 0,
  SS_STATE_CELL_CONFIGURED,
  SS_STATE_CELL_ACTIVE,
  SS_STATE_AS_SECURITY_ACTIVE,
  SS_STATE_AS_RBS_ACTIVE,
  SS_STATE_CELL_BROADCASTING,
  SS_STATE_MAX_STATE
} SS_STATE_t;
*/

#endif /** __SS_CONFIG_H__ */
/** @} */
