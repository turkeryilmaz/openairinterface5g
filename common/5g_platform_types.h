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

#ifndef FIVEG_PLATFORM_TYPES_H__
#define FIVEG_PLATFORM_TYPES_H__

#include <stdint.h>

typedef struct plmn_id_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t mnc_digit_length;
} plmn_id_t;

typedef struct nssai_s {
  uint8_t sst;
  uint32_t sd;
} nssai_t;

// Globally Unique AMF Identifier
typedef struct nr_guami_s {
  uint16_t mcc;
  uint16_t mnc;
  uint8_t mnc_len;
  uint8_t amf_region_id;
  uint16_t amf_set_id;
  uint8_t amf_pointer;
} nr_guami_t;

typedef enum {
  PDUSessionType_ipv4 = 0,
  PDUSessionType_ipv6 = 1,
  PDUSessionType_ipv4v6 = 2,
  PDUSessionType_ethernet = 3,
  PDUSessionType_unstructured = 4
} pdu_session_type_t;

typedef enum { NON_DYNAMIC, DYNAMIC } fiveQI_t;

typedef struct {
  union {
    struct {
      uint16_t fiveqi;
      uint8_t qos_priority_level;
    } non_dynamic;
    struct {
      // Range 5QI [0 - 255]
      uint16_t fiveqi;
      /* Range [0 - 15]
       15 = "no priority," 1-14 = decreasing priority (1 highest), 0 = logical error if received */
      uint8_t qos_priority_level;
      // Range [0, 1023]: Upper bound for packet delay in 0.5ms units
      uint16_t packet_delay_budget;
      struct {
        // PER = Scalar x 10^-k (k: 0-9)
        uint8_t per_scalar;
        uint8_t per_exponent;
      } packet_error_rate;
    } dynamic;
  };
  fiveQI_t qos_type;
} qos_characteristics_t;

typedef struct {
  uint16_t priority_level;
  // Pre-emption capability on other QoS flows
  uint8_t preemption_capability;
  // Vulnerability of the QoS flow to pre-emption of other QoS flows
  uint8_t preemption_vulnerability;
} ngran_allocation_retention_priority_t;

typedef struct {
  qos_characteristics_t qos_characteristics;
  ngran_allocation_retention_priority_t arp;
} qos_flow_level_qos_parameters_t;

#endif
