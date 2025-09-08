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

/*! \file e1ap_lib_test.c
 * \brief Unit tests for E1AP libraries
 * \author Guido Casati
 * \date 2024
 * \version 0.1
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include "common/utils/utils.h"
#include "e1ap_bearer_context_management.h"
#include "e1ap_interface_management.h"
#include "e1ap_lib_includes.h"
#include "common/utils/assertions.h"

void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  printf("detected error at %s:%d:%s: %s\n", file, line, function, s);
  abort();
}

static E1AP_E1AP_PDU_t *e1ap_encode_decode(const E1AP_E1AP_PDU_t *enc_pdu)
{
  // xer_fprint(stdout, &asn_DEF_E1AP_E1AP_PDU, enc_pdu);

  DevAssert(enc_pdu != NULL);
  char errbuf[2048];
  size_t errlen = sizeof(errbuf);
  int ret = asn_check_constraints(&asn_DEF_E1AP_E1AP_PDU, enc_pdu, errbuf, &errlen);
  AssertFatal(ret == 0, "asn_check_constraints() failed: %s\n", errbuf);

  uint8_t msgbuf[16384];
  asn_enc_rval_t enc = aper_encode_to_buffer(&asn_DEF_E1AP_E1AP_PDU, NULL, enc_pdu, msgbuf, sizeof(msgbuf));
  AssertFatal(enc.encoded > 0, "aper_encode_to_buffer() failed\n");

  E1AP_E1AP_PDU_t *dec_pdu = NULL;
  asn_codec_ctx_t st = {.max_stack_size = 100 * 1000};
  asn_dec_rval_t dec = aper_decode(&st, &asn_DEF_E1AP_E1AP_PDU, (void **)&dec_pdu, msgbuf, enc.encoded, 0, 0);
  AssertFatal(dec.code == RC_OK, "aper_decode() failed\n");

  // xer_fprint(stdout, &asn_DEF_E1AP_E1AP_PDU, dec_pdu);
  return dec_pdu;
}

static void e1ap_msg_free(E1AP_E1AP_PDU_t *pdu)
{
  ASN_STRUCT_FREE(asn_DEF_E1AP_E1AP_PDU, pdu);
}

static UP_TL_information_t create_up_tl_info(void)
{
  UP_TL_information_t tl_info;
  tl_info.tlAddress = htonl(0xC0A90001); // 192.169.0.1
  tl_info.teId = 0x2345;
  return tl_info;
}

/**
 * @brief Test E1AP Bearer Context Setup Request encoding/decoding
 */
static void test_bearer_context_setup_request(void)
{
  bearer_context_pdcp_config_t pdcp = {
    .discardTimer = E1AP_DiscardTimer_ms10,
    .pDCP_Reestablishment = true,
    .pDCP_SN_Size_DL = E1AP_PDCP_SN_Size_s_12,
    .pDCP_SN_Size_UL = E1AP_PDCP_SN_Size_s_12,
    .reorderingTimer = 10,
    .rLC_Mode = E1AP_RLC_Mode_rlc_am,
  };

  bearer_context_sdap_config_t sdap = {
    .defaultDRB = 1,
    .sDAP_Header_DL = true,
    .sDAP_Header_UL = true,
  };

  qos_flow_to_setup_t qos = {
    .qfi = 1,
    .qos_params.alloc_reten_priority.preemption_capability = E1AP_Pre_emptionCapability_shall_not_trigger_pre_emption,
    .qos_params.alloc_reten_priority.preemption_vulnerability = E1AP_Pre_emptionVulnerability_not_pre_emptable,
    .qos_params.alloc_reten_priority.priority_level = E1AP_PriorityLevel_highest,
    .qos_params.qos_characteristics.non_dynamic.fiveqi = 9,
  };

  security_indication_t security = {
    .confidentialityProtectionIndication = E1AP_ConfidentialityProtectionIndication_required,
    .integrityProtectionIndication = E1AP_IntegrityProtectionIndication_required,
    .maxIPrate = E1AP_MaxIPrate_max_UErate,
  };

  // Step 1: Initialize the E1AP Bearer Context Setup Request
  e1ap_bearer_setup_req_t orig = {
      .gNB_cu_cp_ue_id = 1234,
      .secInfo.cipheringAlgorithm = 0x01,
      .secInfo.integrityProtectionAlgorithm = 0x01,
      .ueDlAggMaxBitRate = 1000000000,
      .bearerContextStatus = 0,
      .servingPLMNid.mcc = 001,
      .servingPLMNid.mnc = 01,
      .servingPLMNid.mnc_digit_length = 0x02,
      .numPDUSessions = 1,
      .pduSession[0].sessionId = 1,
      .pduSession[0].sessionType = E1AP_PDU_Session_Type_ipv4,
      .pduSession[0].nssai.sd = 0x01,
      .pduSession[0].nssai.sst = 0x01,
      .pduSession[0].securityIndication = security,
      .pduSession[0].numDRB2Setup = 1,
      .pduSession[0].UP_TL_information = create_up_tl_info(),
      .pduSession[0].DRBnGRanList[0].id = 1,
      .pduSession[0].DRBnGRanList[0].sdap_config = sdap,
      .pduSession[0].DRBnGRanList[0].pdcp_config = pdcp,
      .pduSession[0].DRBnGRanList[0].id = 1,
      .pduSession[0].DRBnGRanList[0].numCellGroups = 1,
      .pduSession[0].DRBnGRanList[0].cellGroupList[0] = MCG,
      .pduSession[0].DRBnGRanList[0].numQosFlow2Setup = 1,
      .pduSession[0].DRBnGRanList[0].qosFlows[0] = qos,
  };
  memset(orig.secInfo.encryptionKey, 0xAB, sizeof(orig.secInfo.encryptionKey));
  memset(orig.secInfo.integrityProtectionKey, 0xCD, sizeof(orig.secInfo.integrityProtectionKey));
  // E1AP encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_setup_request(&orig);
  // E1AP decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);
  // Free the E1AP encoded message
  e1ap_msg_free(enc);
  // E1 message decode
  e1ap_bearer_setup_req_t decoded = {0};
  bool ret = decode_E1_bearer_context_setup_request(dec, &decoded);
  AssertFatal(ret, "decode_E1_bearer_context_setup_request(): could not decode message\n");
  // Free the E1AP decoded message
  e1ap_msg_free(dec);
  // Equality check original/decoded
  ret = eq_bearer_context_setup_request(&orig, &decoded);
  AssertFatal(ret, "eq_bearer_context_setup_request(): decoded message doesn't match\n");
  // Free the memory for the decoded message
  free_e1ap_context_setup_request(&decoded);
  // Deep copy and equality check of the original message
  e1ap_bearer_setup_req_t cp = cp_bearer_context_setup_request(&orig);
  ret = eq_bearer_context_setup_request(&orig, &cp);
  AssertFatal(ret, "eq_bearer_context_setup_request(): copied message doesn't match\n");
  // Free the copied message and original
  free_e1ap_context_setup_request(&cp);
  free_e1ap_context_setup_request(&orig);
}

/**
 * @brief Test E1AP Bearer Context Setup Response encoding/decoding
 */
static void test_bearer_context_setup_response(void)
{
  // Step 1: Initialize the E1AP Bearer Context Setup Response
  e1ap_bearer_setup_resp_t orig = {
      .gNB_cu_cp_ue_id = 1234,
      .gNB_cu_up_ue_id = 5678,
      .numPDUSessions = 1,
      .pduSession[0].id = 1,
      .pduSession[0].tl_info = create_up_tl_info(),
      .pduSession[0].numDRBSetup = 1,
      .pduSession[0].numDRBFailed = 0,
      .pduSession[0].DRBnGRanList[0].id = 1,
      .pduSession[0].DRBnGRanList[0].numUpParam = 1,
      .pduSession[0].DRBnGRanList[0].numQosFlowSetup = 1,
      .pduSession[0].DRBnGRanList[0].qosFlows[0].qfi = 1,
      .pduSession[0].DRBnGRanList[0].UpParamList[0].cell_group_id = MCG,
      .pduSession[0].DRBnGRanList[0].UpParamList[0].tl_info = create_up_tl_info(),
  };
  // E1AP encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_setup_response(&orig);

  // E1AP decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the E1AP encoded message
  e1ap_msg_free(enc);

  // E1 message decode
  e1ap_bearer_setup_resp_t decoded = {0};
  bool ret = decode_E1_bearer_context_setup_response(dec, &decoded);
  AssertFatal(ret, "decode_E1_bearer_context_setup_response(): could not decode message\n");

  // Free the E1AP decoded message
  e1ap_msg_free(dec);

  // Equality check original/decoded
  ret = eq_bearer_context_setup_response(&orig, &decoded);
  AssertFatal(ret, "eq_bearer_context_setup_response(): decoded message doesn't match\n");

  // Free the memory for the decoded message
  free_e1ap_context_setup_response(&decoded);

  // Deep copy and equality check of the original message
  e1ap_bearer_setup_resp_t cp = cp_bearer_context_setup_response(&orig);
  ret = eq_bearer_context_setup_response(&orig, &cp);
  AssertFatal(ret, "eq_bearer_context_setup_response(): copied message doesn't match\n");

  // Free the copied message and original
  free_e1ap_context_setup_response(&cp);
  free_e1ap_context_setup_response(&orig);
}

/** @brief Test E1AP Bearer Context Setup Failure encoding/decoding */
static void test_bearer_context_setup_failure(void)
{
  e1ap_bearer_context_setup_failure_t orig = {
      .gNB_cu_cp_ue_id = 0x1234,
      .gNB_cu_up_ue_id = malloc_or_fail(sizeof(*orig.gNB_cu_up_ue_id)),
      .cause.type = E1AP_Cause_PR_radioNetwork,
      .cause.value = 11,
  };
  *orig.gNB_cu_up_ue_id = 0x5678;

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_setup_failure(&orig);

  // E1AP decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the E1AP encoded message
  e1ap_msg_free(enc);

  // Decode the encoded message
  e1ap_bearer_context_setup_failure_t decoded = {0};
  bool ret = decode_E1_bearer_context_setup_failure(&decoded, dec);
  AssertFatal(ret, "decode_E1_bearer_context_setup_failure failed");

  // Free the E1AP decoded message
  e1ap_msg_free(dec);

  // Equality check original/decoded
  AssertFatal(eq_bearer_context_setup_failure(&orig, &decoded), "eq_bearer_context_mod_failure: decoded message does not match original");

  // Free the memory for the decoded message
  free_e1_bearer_context_setup_failure(&decoded);

  // Deep copy and equality check of the original message
  e1ap_bearer_context_setup_failure_t cp = cp_bearer_context_setup_failure(&orig);
  AssertFatal(eq_bearer_context_setup_failure(&orig, &cp), "eq_bearer_context_setup_failure(): copied message doesn't match\n");

  // Cleanup
  free_e1_bearer_context_setup_failure(&orig);
  free_e1_bearer_context_setup_failure(&cp);
}

/** @brief Test E1AP Bearer Context Release Command encoding/decoding */
static void test_bearer_context_release_command(void)
{
  e1ap_bearer_release_cmd_t orig = {
      .gNB_cu_cp_ue_id = 0x1234,
      .gNB_cu_up_ue_id = 0x5678,
      .cause.type = E1AP_CAUSE_RADIO_NETWORK,
      .cause.value = E1AP_RADIO_CAUSE_MULTIPLE_QOS_FLOW_ID_INSTANCES,
  };

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_e1_bearer_context_release_command(&orig);

  // E1AP decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the E1AP encoded message
  e1ap_msg_free(enc);

  // Decode the encoded message
  e1ap_bearer_release_cmd_t decoded = {0};
  AssertFatal(decode_e1_bearer_context_release_command(&decoded, dec), "decode_e1_bearer_context_release_command failed");

  // Free the E1AP decoded message
  e1ap_msg_free(dec);

  // Equality check original/decoded
  AssertFatal(eq_bearer_context_release_command(&orig, &decoded), "eq_bearer_context_release_command: decoded message does not match original");

  // Free the memory for the decoded message
  free_e1_bearer_context_release_command(&decoded);

  // Deep copy and equality check of the original message
  e1ap_bearer_release_cmd_t cp = cp_bearer_context_release_command(&orig);
  AssertFatal(eq_bearer_context_release_command(&orig, &cp), "eq_bearer_context_release_command(): copied message doesn't match\n");

  // Cleanup
  free_e1_bearer_context_release_command(&orig);
  free_e1_bearer_context_release_command(&cp);
}

/** @brief Test E1AP Bearer Context Release Complete encoding/decoding */
static void test_bearer_context_release_complete(void)
{
  e1ap_bearer_release_cplt_t orig = {
      .gNB_cu_cp_ue_id = 0x1234,
      .gNB_cu_up_ue_id = 0x5678,
  };

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_e1_bearer_context_release_complete(&orig);

  // E1AP decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the E1AP encoded message
  e1ap_msg_free(enc);

  // Decode the encoded message
  e1ap_bearer_release_cplt_t decoded = {0};
  AssertFatal(decode_e1_bearer_context_release_complete(&decoded, dec), "decode_e1_bearer_context_release_complete failed");

  // Free the E1AP decoded message
  e1ap_msg_free(dec);

  // Equality check original/decoded
  AssertFatal(eq_bearer_context_release_complete(&orig, &decoded), "eq_bearer_context_release_complete: decoded message does not match original");

  // Free the memory for the decoded message
  free_e1_bearer_context_release_complete(&decoded);

  // Deep copy and equality check of the original message
  e1ap_bearer_release_cplt_t cp = cp_bearer_context_release_complete(&orig);
  AssertFatal(eq_bearer_context_release_complete(&orig, &cp), "eq_bearer_context_release_complete(): copied message doesn't match\n");

  // Cleanup
  free_e1_bearer_context_release_complete(&orig);
  free_e1_bearer_context_release_complete(&cp);
}

/**
 * @brief Test CU-UP Setup Request encoding/decoding
 */
static void test_e1_cuup_setup_request(void)
{
  e1ap_setup_req_t orig = {.gNB_cu_up_id = 1234,
                           .gNB_cu_up_name = strdup("OAI CU-UP"),
                           .transac_id = 42,
                           .supported_plmns = 1,
                           .cn_support = cn_support_5GC,
                           .plmn[0] = {.id = {.mcc = 001, .mnc = 01, .mnc_digit_length = 2},
                                       .supported_slices = 1}};
  orig.plmn[0].slice = malloc_or_fail(sizeof(*orig.plmn[0].slice));
  orig.plmn[0].slice->sst = 0x01;
  orig.plmn[0].slice->sd = 0x01;

  E1AP_E1AP_PDU_t *encoded = encode_e1ap_cuup_setup_request(&orig);
  E1AP_E1AP_PDU_t *decoded_msg = e1ap_encode_decode(encoded);
  e1ap_msg_free(encoded);

  e1ap_setup_req_t decoded = {0};
  bool ret = decode_e1ap_cuup_setup_request(decoded_msg, &decoded);
  AssertFatal(ret, "Failed to decode setup request");
  e1ap_msg_free(decoded_msg);

  ret = eq_e1ap_cuup_setup_request(&orig, &decoded);
  AssertFatal(ret, "Decoded setup request doesn't match original");
  free_e1ap_cuup_setup_request(&decoded);

  e1ap_setup_req_t cp = cp_e1ap_cuup_setup_request(&orig);
  ret = eq_e1ap_cuup_setup_request(&orig, &cp);
  AssertFatal(ret, "eq_e1ap_cuup_setup_request(): copied message doesn't match\n");

  free_e1ap_cuup_setup_request(&orig);
  free_e1ap_cuup_setup_request(&cp);
}

/**
 * @brief Test CU-UP Setup Response encoding/decoding
 */
static void test_e1_cuup_setup_response(void)
{
  e1ap_setup_resp_t orig = {.transac_id = 42,
                            .gNB_cu_cp_name = strdup("OAI CU-CP"),
                            .tnla_info = malloc(sizeof(tnl_address_info_t))};
  orig.tnla_info->num_addresses_to_add = 1;
  orig.tnla_info->num_addresses_to_remove = 1;
  orig.tnla_info->addresses_to_add[0].num_gtp_tl_addresses = 1;
  orig.tnla_info->addresses_to_add[0].ipsec_tl_address = 0xC0A80001;
  orig.tnla_info->addresses_to_add[0].gtp_tl_addresses[0] = 0xC0A80002;
  orig.tnla_info->num_addresses_to_remove = 1;
  orig.tnla_info->addresses_to_remove[0].num_gtp_tl_addresses = 1;
  orig.tnla_info->addresses_to_remove[0].ipsec_tl_address = 0xC0A80003;
  orig.tnla_info->addresses_to_remove[0].gtp_tl_addresses[0] = 0xC0A80004;

  E1AP_E1AP_PDU_t *encoded = encode_e1ap_cuup_setup_response(&orig);
  E1AP_E1AP_PDU_t *decoded_msg = e1ap_encode_decode(encoded);
  e1ap_msg_free(encoded);

  e1ap_setup_resp_t decoded = {0};
  bool ret = decode_e1ap_cuup_setup_response(decoded_msg, &decoded);
  AssertFatal(ret, "Failed to decode setup response");
  e1ap_msg_free(decoded_msg);

  ret = eq_e1ap_cuup_setup_response(&orig, &decoded);
  AssertFatal(ret, "Decoded setup response doesn't match original");
  free_e1ap_cuup_setup_response(&decoded);

  e1ap_setup_resp_t cp = cp_e1ap_cuup_setup_response(&orig);
  ret = eq_e1ap_cuup_setup_response(&orig, &cp);
  AssertFatal(ret, "eq_e1ap_cuup_setup_response(): copied message doesn't match\n");

  free_e1ap_cuup_setup_response(&orig);
  free_e1ap_cuup_setup_response(&cp);
}

// Test for E1AP CU-UP Setup Failure
static void test_e1_cuup_setup_failure(void)
{
  e1ap_setup_fail_t orig = {.transac_id = 42,
                            .cause.type = E1AP_CAUSE_RADIO_NETWORK,
                            .cause.value = E1AP_RADIO_CAUSE_NORMAL_RELEASE,
                            .time_to_wait = malloc_or_fail(sizeof(long)),
                            .crit_diag = malloc_or_fail(sizeof(criticality_diagnostics_t))};
  *orig.time_to_wait = 5;
  orig.crit_diag->procedure_code = malloc_or_fail(sizeof(*orig.crit_diag->procedure_code));
  *orig.crit_diag->procedure_code = 99;
  orig.crit_diag->triggering_msg = malloc_or_fail(sizeof(*orig.crit_diag->triggering_msg));
  *orig.crit_diag->triggering_msg = TRIGGERING_MSG_SUCCESSFUL_OUTCOME;
  orig.crit_diag->procedure_criticality = malloc_or_fail(sizeof(*orig.crit_diag->procedure_criticality));
  *orig.crit_diag->procedure_criticality = CRITICALITY_IGNORE;
  orig.crit_diag->num_errors = 1;
  orig.crit_diag->errors[0].ie_id = 66;
  orig.crit_diag->errors[0].error_type = ERROR_TYPE_MISSING;
  orig.crit_diag->errors[0].criticality = CRITICALITY_IGNORE;

  E1AP_E1AP_PDU_t *encoded = encode_e1ap_cuup_setup_failure(&orig);
  E1AP_E1AP_PDU_t *decoded_msg = e1ap_encode_decode(encoded);
  e1ap_msg_free(encoded);

  e1ap_setup_fail_t decoded = {0};
  bool ret = decode_e1ap_cuup_setup_failure(decoded_msg, &decoded);
  AssertFatal(ret, "Failed to decode setup failure");
  e1ap_msg_free(decoded_msg);

  ret = eq_e1ap_cuup_setup_failure(&orig, &decoded);
  AssertFatal(ret, "Decoded setup failure doesn't match original");
  free_e1ap_cuup_setup_failure(&decoded);

  e1ap_setup_fail_t cp = cp_e1ap_cuup_setup_failure(&orig);
  ret = eq_e1ap_cuup_setup_failure(&orig, &cp);
  AssertFatal(ret, "eq_e1ap_cuup_setup_failure(): copied message doesn't match\n");

  free_e1ap_cuup_setup_failure(&cp);
  free_e1ap_cuup_setup_failure(&orig);
}

/**
 * @brief Test E1AP Bearer Context Modification Request encoding/decoding
 */
static void test_bearer_context_modification_request(void)
{
  const bearer_context_sdap_config_t dummy_sdap_config = {
      .defaultDRB = 0,
      .sDAP_Header_DL = false,
      .sDAP_Header_UL = false,
  };

  const bearer_context_pdcp_config_t dummy_pdcp_config = {
      .discardTimer = E1AP_DiscardTimer_ms100,
      .pDCP_Reestablishment = true,
      .pDCP_SN_Size_DL = E1AP_PDCP_SN_Size_s_12,
      .pDCP_SN_Size_UL = E1AP_PDCP_SN_Size_s_12,
      .reorderingTimer = 5,
      .rLC_Mode = E1AP_RLC_Mode_rlc_um_bidirectional,
  };

  const qos_flow_to_setup_t dummy_qos_flows = {
      .qfi = 9,
      .qos_params.alloc_reten_priority.preemption_capability = E1AP_Pre_emptionCapability_may_trigger_pre_emption,
      .qos_params.alloc_reten_priority.preemption_vulnerability = E1AP_Pre_emptionVulnerability_pre_emptable,
      .qos_params.alloc_reten_priority.priority_level = E1AP_PriorityLevel_no_priority,
      .qos_params.qos_characteristics.non_dynamic.fiveqi = 9,
  };

  const e1_pdcp_status_info_t dummy_pdcp_status = {
      .dl_count.hfn = 12,
      .dl_count.sn = 34,
      .ul_count.hfn = 56,
      .ul_count.sn = 78,
  };

  DRB_nGRAN_to_mod_t drb_to_mod = {
    .numDlUpParam = 1,
    .DlUpParamList[0].cell_group_id = MCG,
    .DlUpParamList[0].tl_info = create_up_tl_info(),
    .id = 1,
    .pdcp_config = malloc_or_fail(sizeof(*drb_to_mod.pdcp_config)),
    .pdcp_sn_status_requested = true,
    .pdcp_status = malloc_or_fail(sizeof(*drb_to_mod.pdcp_status)),
    .numQosFlow2Setup = 1,
    .qosFlows[0] = dummy_qos_flows,
  };
  *drb_to_mod.pdcp_config = dummy_pdcp_config;
  *drb_to_mod.pdcp_status = dummy_pdcp_status;

  pdu_session_to_mod_t pdusession_mod_item = {
      .sessionId = 1,
      .numDRB2Modify = 1,
      .DRBnGRanModList[0] = drb_to_mod,
  };

  DRB_nGRAN_to_setup_t drb_to_setup = {
      .id = 1,
      .cellGroupList[0] = MCG,
      .numCellGroups = 1,
      .pdcp_config = dummy_pdcp_config,
      .sdap_config = dummy_sdap_config,
      .numQosFlow2Setup = 1,
      .qosFlows[0] = dummy_qos_flows,
      .drb_inactivity_timer = malloc_or_fail(sizeof(*drb_to_setup.drb_inactivity_timer)),
  };
  *drb_to_setup.drb_inactivity_timer = 500;

  pdu_session_to_setup_t pdusession_setup_item = {
      .numDRB2Setup = 1,
      .nssai.sd = 0x01,
      .nssai.sst = 0x01,
      .UP_TL_information.teId = 0x12345,
      .UP_TL_information.tlAddress = 167772161,
      .DRBnGRanList[0] = drb_to_setup,
  };

  // Initialize the Bearer Context Modification Request
  e1ap_bearer_mod_req_t orig = {
      .gNB_cu_cp_ue_id = 0x1234,
      .gNB_cu_up_ue_id = 0x5678,
      .bearerContextStatus = malloc_or_fail(sizeof(*orig.bearerContextStatus)),
      .inactivityTimer = malloc_or_fail(sizeof(*orig.inactivityTimer)),
      .numPDUSessions = 1,
      .pduSession[0] = pdusession_setup_item,
      .numPDUSessionsMod = 1,
      .pduSessionMod[0] = pdusession_mod_item,
  };
  *orig.bearerContextStatus = BEARER_SUSPEND;
  *orig.inactivityTimer = 1000;

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_mod_request(&orig);

  // Decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the encoded message
  e1ap_msg_free(enc);

  // Decode the message into a new struct
  e1ap_bearer_mod_req_t decoded = {0};
  AssertFatal(decode_E1_bearer_context_mod_request(dec, &decoded), "decode_E1_bearer_context_mod_request(): could not decode message\n");

  // Free the decoded E1AP message
  e1ap_msg_free(dec);

  // Compare the original and decoded structs
  AssertFatal(eq_bearer_context_mod_request(&orig, &decoded), "eq_bearer_context_mod_request(): decoded message doesn't match\n");

  // Free the memory for the decoded message
  free_e1ap_context_mod_request(&decoded);

  // Deep copy the original message
  e1ap_bearer_mod_req_t cp = cp_bearer_context_mod_request(&orig);

  // Verify the deep copy matches the original
  AssertFatal(eq_bearer_context_mod_request(&orig, &cp), "eq_bearer_context_mod_request(): copied message doesn't match\n");

  // Free the copied and original message
  free_e1ap_context_mod_request(&cp);
  free_e1ap_context_mod_request(&orig);
}

const e1ap_cause_t dummy_cause = {
  .value = E1AP_RADIO_CAUSE_UNSPECIFIED,
  .type = E1AP_CAUSE_RADIO_NETWORK,
};

const DRB_nGRAN_failed_t dummy_drb_failed = {
    .id = 2,
    .cause.value = E1AP_RADIO_CAUSE_UNKNOWN_DRB_ID,
    .cause.type = E1AP_CAUSE_RADIO_NETWORK,
};

/**
 * @brief Test E1AP Bearer Context Modification Response encoding/decoding
 */
static void test_bearer_context_modification_response(void)
{
  const e1_pdcp_status_info_t dummy_pdcp_status = {
      .dl_count.hfn = 12,
      .dl_count.sn = 2,
      .ul_count.hfn = 12,
      .ul_count.sn = 3,
  };

  // DRB Modified List
  DRB_nGRAN_modified_t drb_mod = {
      .id = 1,
      .numQosFlowSetup = 1,
      .qosFlows[0].qfi = 1,
      .pdcp_status = malloc_or_fail(sizeof(*drb_mod.pdcp_status)),
  };
  *drb_mod.pdcp_status = dummy_pdcp_status;

  // DRB Setup List
  DRB_nGRAN_setup_t setup = {
      .id = 1,
      .numUpParam = 1,
      .UpParamList[0].cell_group_id = MCG,
      .UpParamList[0].tl_info = create_up_tl_info(),
      .numQosFlowSetup = 1,
      .qosFlows[0].qfi = 1,
      .numQosFlowFailed = 1,
      .qosFlowsFailed[0].qfi = 9,
      .qosFlowsFailed[0].cause = dummy_cause,
  };

  // PDU Session Modified
  pdu_session_modif_t pdu_mod = {
      .id = 1,
      .integrityProtectionIndication = malloc_or_fail(sizeof(*pdu_mod.integrityProtectionIndication)),
      .confidentialityProtectionIndication = malloc_or_fail(sizeof(*pdu_mod.confidentialityProtectionIndication)),
      .ng_DL_UP_TL_info = malloc_or_fail(sizeof(*pdu_mod.ng_DL_UP_TL_info)),
      .numDRBModified = 1,
      .DRBnGRanModList[0] = drb_mod,
      .numDRBFailedToMod = 1,
      .DRBnGRanFailedModList[0] = dummy_drb_failed,
      .numDRBSetup = 1,
      .DRBnGRanSetupList[0] = setup,
      .numDRBFailed = 1,
      .DRBnGRanFailedList[0] = dummy_drb_failed,
  };

  *pdu_mod.integrityProtectionIndication = SECURITY_PREFERRED;
  *pdu_mod.confidentialityProtectionIndication = SECURITY_PREFERRED;
  *pdu_mod.ng_DL_UP_TL_info = create_up_tl_info();

  e1ap_bearer_modif_resp_t orig = {
      .gNB_cu_cp_ue_id = 0x1234,
      .gNB_cu_up_ue_id = 0x5678,
      .numPDUSessionsMod = 1,
      .pduSessionMod[0] = pdu_mod,
  };

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_mod_response(&orig);

  // Decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the encoded message
  e1ap_msg_free(enc);

  // Decode the message into a new struct
  e1ap_bearer_modif_resp_t decoded = {0};
  bool ret = decode_E1_bearer_context_mod_response(&decoded, dec);
  AssertFatal(ret, "decode_E1_bearer_context_mod_response(): could not decode message\n");

  // Free the decoded E1AP message
  e1ap_msg_free(dec);

  // Compare the original and decoded structs
  ret = eq_bearer_context_mod_response(&orig, &decoded);
  AssertFatal(ret, "eq_bearer_context_mod_response(): decoded message doesn't match\n");

  // Free the memory for the decoded message
  free_e1ap_context_mod_response(&decoded);

  // Deep copy the original message
  e1ap_bearer_modif_resp_t cp = cp_bearer_context_mod_response(&orig);

  // Verify the deep copy matches the original
  ret = eq_bearer_context_mod_response(&orig, &cp);
  AssertFatal(ret, "eq_bearer_context_mod_response(): copied message doesn't match\n");

  // Free the copied and original message
  free_e1ap_context_mod_response(&cp);
  free_e1ap_context_mod_response(&orig);
}

/** @brief Test E1AP Bearer Context Modification Response encoding/decoding
 *         with failed DRBs (setup and modification) */
static void test_bearer_context_modification_response_fail(void)
{
  const e1ap_cause_t cause = {
      .value = E1AP_PROTOCOL_CAUSE_SEMANTIC_ERROR,
      .type = E1AP_CAUSE_PROTOCOL,
  };

  DRB_nGRAN_failed_t failed_setup_drb = {
      .id = 5,
      .cause = cause,
  };

  DRB_nGRAN_failed_t failed_mod_drb = {
      .id = 6,
      .cause = cause,
  };

  pdu_session_modif_t pdu_mod = {
      .id = 2,
      .numDRBModified = 0,
      .numDRBFailedToMod = 1,
      .DRBnGRanFailedModList[0] = failed_mod_drb,
      .numDRBSetup = 0,
      .numDRBFailed = 1,
      .DRBnGRanFailedList[0] = failed_setup_drb,
      .integrityProtectionIndication = NULL,
      .confidentialityProtectionIndication = NULL,
      .ng_DL_UP_TL_info = NULL,
  };

  e1ap_bearer_modif_resp_t orig = {
      .gNB_cu_cp_ue_id = 0xABCD,
      .gNB_cu_up_ue_id = 0xDCBA,
      .numPDUSessionsMod = 1,
      .pduSessionMod[0] = pdu_mod,
  };

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_mod_response(&orig);

  // Decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the encoded message
  e1ap_msg_free(enc);

  // Decode into a new struct
  e1ap_bearer_modif_resp_t decoded = {0};
  bool ret = decode_E1_bearer_context_mod_response(&decoded, dec);
  AssertFatal(ret, "decode_E1_bearer_context_mod_response(): could not decode failed DRB setup+mod case\n");

  // Free the decoded E1AP message
  e1ap_msg_free(dec);

  // Compare the original and decoded structs
  ret = eq_bearer_context_mod_response(&orig, &decoded);
  AssertFatal(ret, "eq_bearer_context_mod_response(): failed DRB setup+mod case mismatch\n");

  // Free all memory
  free_e1ap_context_mod_response(&decoded);
  free_e1ap_context_mod_response(&orig);
}

/** @brief Test E1AP Bearer Context Modification Failure encoding/decoding */
static void test_bearer_context_modification_failure(void)
{
  // Create the original failure struct
  e1ap_bearer_context_mod_failure_t orig = {
      .gNB_cu_cp_ue_id = 0x1111,
      .gNB_cu_up_ue_id = 0x2222,
      .cause.type = E1AP_CAUSE_TRANSPORT,
      .cause.value = E1AP_TRANSPORT_CAUSE_RESOURCE_UNAVAILABLE,
  };

  // Encode the original message
  E1AP_E1AP_PDU_t *enc = encode_E1_bearer_context_mod_failure(&orig);

  // Decode the encoded message
  E1AP_E1AP_PDU_t *dec = e1ap_encode_decode(enc);

  // Free the encoded message
  e1ap_msg_free(enc);

  // Decode into a new struct
  e1ap_bearer_context_mod_failure_t decoded = {0};
  AssertFatal(decode_E1_bearer_context_mod_failure(&decoded, dec),
              "decode_E1_bearer_context_mod_failure(): could not decode message\n");

  // Free the decoded PDU
  e1ap_msg_free(dec);

  // Compare original and decoded messages
  AssertFatal(eq_E1_bearer_context_mod_failure(&orig, &decoded),
              "eq_E1_bearer_context_mod_failure(): decoded message doesn't match\n");

  // Deep copy the original message
  e1ap_bearer_context_mod_failure_t cp = cp_E1_bearer_context_mod_failure(&orig);

  // Compare original and copied messages
  AssertFatal(eq_E1_bearer_context_mod_failure(&orig, &cp), "eq_E1_bearer_context_mod_failure(): copied message doesn't match\n");

  // Free the decoded and copied messages
  free_E1_bearer_context_mod_failure(&decoded);
  free_E1_bearer_context_mod_failure(&cp);
}

int main()
{
  // E1 Bearer Context Setup
  test_bearer_context_setup_request();
  test_bearer_context_setup_response();
  test_bearer_context_setup_failure();
  // E1 Interface Management
  test_e1_cuup_setup_request();
  test_e1_cuup_setup_response();
  test_e1_cuup_setup_failure();
  // E1 Bearer Context Modification
  test_bearer_context_modification_request();
  test_bearer_context_modification_response();
  test_bearer_context_modification_response_fail();
  test_bearer_context_modification_failure();
  // Bearer Context Release
  test_bearer_context_release_command();
  test_bearer_context_release_complete();
  return 0;
}
