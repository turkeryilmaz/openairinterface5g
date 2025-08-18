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

#ifndef E1AP_HELPERS_H_
#define E1AP_HELPERS_H_

#include "stdbool.h"
#include "e1ap_messages_types.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_configuration.h"
#include "openair2/RRC/NR/nr_rrc_defs.h"

/// @brief set PDCP configuration in E1 Bearer Context Management message
bearer_context_pdcp_config_t e1_fill_bearer_context_pdcp_config(const nr_pdcp_configuration_t *pdcp,
                                                                bool um_on_default_drb,
                                                                const nr_redcap_ue_cap_t *redcap_cap);

/// @brief set QoS Flows to Setup in E1 DRB To Setup List
qos_flow_to_setup_t e1_fill_qos_flow_to_setup(const pdusession_level_qos_parameter_t *qos);

#endif /* E1AP_HELPERS_H_ */
