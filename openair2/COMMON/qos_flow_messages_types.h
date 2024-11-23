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

#ifndef QOSFLOW_MESSAGES_TYPES_H_
#define QOSFLOW_MESSAGES_TYPES_H_

#define STANDARIZED_5QI_NUM 26

typedef enum { NON_DYNAMIC_5QI, DYNAMIC_5QI } fiveQI_t;

typedef enum { GBR, NON_GBR, DELAY_CRITICAL_GBR } qos_flow_type_t;

typedef enum qos_priority_e {
  PRIORITY_20 = 20,
  PRIORITY_40 = 40,
  PRIORITY_30 = 30,
  PRIORITY_50 = 50,
  PRIORITY_7 = 7,
  PRIORITY_15 = 15,
  PRIORITY_56 = 56,
  PRIORITY_10 = 10,
  PRIORITY_60 = 60,
  PRIORITY_70 = 70,
  PRIORITY_80 = 80,
  PRIORITY_90 = 90,
  PRIORITY_5 = 5,
  PRIORITY_55 = 55,
  PRIORITY_65 = 65,
  PRIORITY_68 = 68,
  PRIORITY_19 = 19,
  PRIORITY_22 = 22,
  PRIORITY_24 = 24,
  PRIORITY_21 = 21,
  PRIORITY_18 = 18
} qos_priority_t;

typedef enum qos_fiveqi_e {
  FIVEQI_1 = 1,
  FIVEQI_2 = 2,
  FIVEQI_3 = 3,
  FIVEQI_4 = 4,
  FIVEQI_65 = 65,
  FIVEQI_66 = 66,
  FIVEQI_67 = 67,
  FIVEQI_71 = 71,
  FIVEQI_72 = 72,
  FIVEQI_73 = 73,
  FIVEQI_74 = 74,
  FIVEQI_76 = 76,
  FIVEQI_5 = 5,
  FIVEQI_6 = 6,
  FIVEQI_7 = 7,
  FIVEQI_8 = 8,
  FIVEQI_9 = 9,
  FIVEQI_69 = 69,
  FIVEQI_70 = 70,
  FIVEQI_79 = 79,
  FIVEQI_80 = 80,
  FIVEQI_82 = 82,
  FIVEQI_83 = 83,
  FIVEQI_84 = 84,
  FIVEQI_85 = 85,
  FIVEQI_86 = 86
} qos_fiveqi_t;

typedef enum preemption_capability_e {
  SHALL_NOT_TRIGGER_PREEMPTION,
  MAY_TRIGGER_PREEMPTION,
} preemption_capability_t;

typedef enum preemption_vulnerability_e {
  NOT_PREEMPTABLE,
  PREEMPTABLE,
} preemption_vulnerability_t;

typedef struct {
  long fiveqi;
  long qos_priority_level;
} non_dynamic_t;

typedef struct {
  long per_scalar;
  long per_exponent;
} packet_error_rate_t;

typedef struct {
  long fiveqi; // -1 -> optional
  long qos_priority_level;
  long packet_delay_budget;
  packet_error_rate_t packet_error_rate;
} dynamic_t;

typedef struct qos_characteristics_s {
  union {
    non_dynamic_t non_dynamic;
    dynamic_t dynamic;
  };
  fiveQI_t qos_type;
} qos_characteristics_t;

typedef struct ngran_allocation_retention_priority_s {
  uint16_t priority_level;
  preemption_capability_t preemption_capability;
  preemption_vulnerability_t preemption_vulnerability;
} ngran_allocation_retention_priority_t;

typedef struct gbr_qos_flow_information_s {
  long mbr_dl;
  long mbr_ul;
  long gbr_dl;
  long gbr_ul;
} gbr_qos_flow_information_t;

typedef struct qos_flow_level_qos_parameters_s {
  qos_characteristics_t qos_characteristics;
  ngran_allocation_retention_priority_t alloc_reten_priority;
  gbr_qos_flow_information_t *gbr_qos_flow_info;
} qos_flow_level_qos_parameters_t;

typedef struct standard_5QI_characteristics_e {
  uint64_t five_QI;
  uint64_t priority_level;
  uint64_t resource_type;
} standard_5QI_characteristics_t;

#endif /* QOSFLOW_MESSAGES_TYPES_H_ */