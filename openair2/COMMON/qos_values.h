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
 * Author and copyright: Laurent Thomas, open-cells.com
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

#ifndef QOS_VALUES_H_
#define QOS_VALUES_H_

void get_drb_characteristics(qos_flow_to_setup_t *qos_flows_in, int num_qos_flows, qos_flow_level_qos_parameters_t *dRB_QoS);

// based on the 5QI value, its corresponding parameters are searched from the standarized table of 5QI to QoS mapping
uint64_t get_5QI_id(uint64_t fiveqi);

int get_non_dynamic_priority(int fiveqi);

extern const standard_5QI_characteristics_t params_5QI[];

#endif /* E1AP_COMMON_H_ */
