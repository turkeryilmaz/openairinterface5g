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

#include "e1ap_helpers.h"
#include "5g_platform_types.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_asn1_utils.h"
#include "E1AP_RLC-Mode.h"

/// @brief set PDCP configuration in E1 Bearer Context Management message
bearer_context_pdcp_config_t e1_fill_bearer_context_pdcp_config(const nr_pdcp_configuration_t *pdcp,
                                                                bool um_on_default_drb,
                                                                const nr_redcap_ue_cap_t *redcap_cap)
{
  bearer_context_pdcp_config_t out = {0};
  if (redcap_cap && redcap_cap->support_of_redcap_r17 && !redcap_cap->pdcp_drb_long_sn_redcap_r17) {
    LOG_I(NR_RRC, "UE is RedCap without long PDCP SN support: overriding PDCP SN size to 12\n");
    out.pDCP_SN_Size_DL = NR_PDCP_Config__drb__pdcp_SN_SizeDL_len12bits;
    out.pDCP_SN_Size_UL = NR_PDCP_Config__drb__pdcp_SN_SizeUL_len12bits;
  } else {
    out.pDCP_SN_Size_DL = encode_sn_size_dl(pdcp->drb.sn_size);
    out.pDCP_SN_Size_UL = encode_sn_size_ul(pdcp->drb.sn_size);
  }
  out.discardTimer = encode_discard_timer(pdcp->drb.discard_timer);
  out.reorderingTimer = encode_t_reordering(pdcp->drb.t_reordering);
  out.rLC_Mode = um_on_default_drb ? E1AP_RLC_Mode_rlc_um_bidirectional : E1AP_RLC_Mode_rlc_am;
  out.pDCP_Reestablishment = false;
  return out;
}

/// @brief set QoS Flows to Setup in E1 DRB To Setup List
qos_flow_to_setup_t e1_fill_qos_flow_to_setup(const pdusession_level_qos_parameter_t *qos)
{
  qos_flow_to_setup_t qos_flow = {0};
  // QFI
  qos_flow.qfi = qos->qfi;
  // ARP
  qos_flow.qos_params.alloc_reten_priority.preemption_capability = qos->arp.pre_emp_capability;
  qos_flow.qos_params.alloc_reten_priority.preemption_vulnerability = qos->arp.pre_emp_vulnerability;
  qos_flow.qos_params.alloc_reten_priority.priority_level = qos->arp.priority_level;
  // QoS Characteristics
  qos_characteristics_t *qos_characteristics = &qos_flow.qos_params.qos_characteristics;
  if (qos->fiveQI_type == NON_DYNAMIC) {
    qos_characteristics->non_dynamic.fiveqi = qos->fiveQI;
    qos_characteristics->non_dynamic.qos_priority_level = qos->qos_priority;
  } else {
    qos_characteristics->dynamic.fiveqi = qos->fiveQI;
    qos_characteristics->dynamic.qos_priority_level = qos->qos_priority;
    // NOTE: missing packet error rate and delay budget
  }
  qos_characteristics->qos_type = qos->fiveQI_type;

  return qos_flow;
}