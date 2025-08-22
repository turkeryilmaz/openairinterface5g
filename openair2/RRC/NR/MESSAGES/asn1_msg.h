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

/*! \file asn1_msg.h
* \brief primitives to build the asn1 messages
* \author Raymond Knopp and Navid Nikaein, WIE-TAI CHEN
* \date 2011, 2018
* \version 1.0
* \company Eurecom, NTUST
* \email: raymond.knopp@eurecom.fr and  navid.nikaein@eurecom.fr, kroempa@gmail.com
*/

#ifndef __RRC_NR_MESSAGES_ASN1_MSG__H__
#define __RRC_NR_MESSAGES_ASN1_MSG__H__

#include <common/utils/assertions.h>
#include <stdint.h>
#include <stdio.h>
#include "NR_ARFCN-ValueNR.h"
#include "NR_CellGroupConfig.h"
#include "NR_CipheringAlgorithm.h"
#include "NR_DRB-ToAddModList.h"
#include "NR_DRB-ToReleaseList.h"
#include "NR_IntegrityProtAlgorithm.h"
#include "NR_LogicalChannelConfig.h"
#include "NR_MeasConfig.h"
#include "NR_MeasTiming.h"
#include "NR_RLC-BearerConfig.h"
#include "NR_RLC-Config.h"
#include "NR_RRC-TransactionIdentifier.h"
#include "NR_RadioBearerConfig.h"
#include "NR_ReestablishmentCause.h"
#include "NR_SRB-ToAddModList.h"
#include "NR_SecurityConfig.h"
#include "NR_MeasurementTimingConfiguration.h"
#include "ds/seq_arr.h"
#include "ds/byte_array.h"
#include "rrc_messages_types.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_configuration.h"
struct asn_TYPE_descriptor_s;

typedef struct {
  uint8_t transaction_id;
  NR_SRB_ToAddModList_t *srb_config_list;
  NR_DRB_ToAddModList_t *drb_config_list;
  NR_DRB_ToReleaseList_t *drb_release_list;
  NR_SecurityConfig_t *security_config;
  NR_MeasConfig_t *meas_config;
  byte_array_t dedicated_NAS_msg_list[MAX_DRBS_PER_UE];
  int num_nas_msg;
  NR_CellGroupConfig_t *cell_group_config;
  bool masterKeyUpdate;
  int nextHopChainingCount;
} nr_rrc_reconfig_param_t;

/*
 * The variant of the above function which dumps the BASIC-XER (XER_F_BASIC)
 * output into the chosen string buffer.
 * RETURN VALUES:
 *       0: The structure is printed.
 *      -1: Problem printing the structure.
 * WARNING: No sensible errno value is returned.
 */
int xer_sprint_NR(char *string, size_t string_size, struct asn_TYPE_descriptor_s *td, void *sptr);

byte_array_t do_SIB2_NR(NR_SSB_MTC_t *ssbmtc);

byte_array_t do_RRCReject();

NR_RadioBearerConfig_t *get_default_rbconfig(int eps_bearer_id,
                                             int rb_id,
                                             e_NR_CipheringAlgorithm ciphering_algorithm,
                                             e_NR_SecurityConfig__keyToUse key_to_use,
                                             const nr_pdcp_configuration_t *pdcp_config);

int do_RRCSetup(uint8_t *const buffer,
                size_t buffer_size,
                const uint8_t transaction_id,
                const uint8_t *masterCellGroup,
                int masterCellGroup_len,
                const gNB_RrcConfigurationReq *configuration,
                NR_SRB_ToAddModList_t *SRBs);

byte_array_t do_NR_SecurityModeCommand(const uint8_t Transaction_id,
                                       const uint8_t cipheringAlgorithm,
                                       NR_IntegrityProtAlgorithm_t integrityProtAlgorithm);

byte_array_t do_NR_SA_UECapabilityEnquiry(const uint8_t Transaction_id, const uint8_t band);

byte_array_t do_NR_RRCRelease(uint8_t Transaction_id);

byte_array_t do_RRCReconfiguration(const nr_rrc_reconfig_param_t *params);

byte_array_t do_RRCSetupComplete(const uint8_t Transaction_id,
                                 uint8_t sel_plmn_id,
                                 bool is_rrc_connection_setup,
                                 uint64_t fiveG_S_TMSI,
                                 const int dedicatedInfoNASLength,
                                 const char *dedicatedInfoNAS);

byte_array_t do_NR_HandoverPreparationInformation(const byte_array_t uecap);

int do_NR_MeasConfig(const NR_MeasConfig_t *measconfig, uint8_t *buf, int buf_size);

int do_NR_MeasurementTimingConfiguration(const NR_MeasurementTimingConfiguration_t *mtc, uint8_t *buf, int buf_size);

byte_array_t do_RRCSetupRequest(uint8_t *rv, uint64_t fiveG_S_TMSI_part1);

byte_array_t do_NR_RRCReconfigurationComplete_for_nsa(NR_RRC_TransactionIdentifier_t Transaction_id);

byte_array_t do_NR_RRCReconfigurationComplete(const uint8_t Transaction_id);

byte_array_t do_NR_DLInformationTransfer(uint8_t transaction_id, uint32_t pdu_length, uint8_t *pdu_buffer);

byte_array_t do_NR_ULInformationTransfer(uint32_t pdu_length, uint8_t *pdu_buffer);

byte_array_t do_RRCReestablishmentRequest(NR_ReestablishmentCause_t cause, uint32_t pci, uint16_t c_rnti, byte_array_t shortMAC_I);

byte_array_t do_RRCReestablishment(int8_t nh_ncc, const uint8_t Transaction_id);

byte_array_t do_RRCReestablishmentComplete(int64_t rrc_TransactionIdentifier);

NR_MeasConfig_t *get_MeasConfig(const NR_MeasTiming_t *mt,
                                int band,
                                int scs,
                                NR_ReportConfigToAddMod_t *rc_PER,
                                NR_ReportConfigToAddMod_t *rc_A2,
                                seq_arr_t *rc_A3_seq,
                                seq_arr_t *neigh_seq);
void free_MeasConfig(NR_MeasConfig_t *mc);
byte_array_t do_NR_Paging(uint8_t Mod_id, uint32_t tmsi);

byte_array_t generate_VarShortMAC_input(uint32_t pci, uint16_t c_rnti, uint32_t cell_id);

#endif  /* __RRC_NR_MESSAGES_ASN1_MSG__H__ */
