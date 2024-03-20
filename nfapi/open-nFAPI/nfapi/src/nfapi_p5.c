/*
 * Copyright 2017 Cisco Systems, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


#include <signal.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sched.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>

#include <nfapi_interface.h>
#include <nfapi.h>
#include "nfapi_nr_interface.h"
#include "nfapi_nr_interface_scf.h"
#include "nfapi/oai_integration/vendor_ext.h"
#include <debug.h>

// Pack routines
//TODO: Add pacl/unpack fns for uint32 and uint64

uint8_t pack_pnf_param_general_value(void *tlv, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_pnf_param_general_t *value = (nfapi_pnf_param_general_t *)tlv;
  return ( push8(value->nfapi_sync_mode, ppWritePackedMsg, end) &&
           push8(value->location_mode, ppWritePackedMsg, end) &&
           push16(value->location_coordinates_length, ppWritePackedMsg, end) &&
           pusharray8(value->location_coordinates, NFAPI_PNF_PARAM_GENERAL_LOCATION_LENGTH, value->location_coordinates_length, ppWritePackedMsg, end) &&
           push32(value->dl_config_timing, ppWritePackedMsg, end) &&
           push32(value->tx_timing, ppWritePackedMsg, end) &&
           push32(value->ul_config_timing, ppWritePackedMsg, end) &&
           push32(value->hi_dci0_timing, ppWritePackedMsg, end) &&
           push16(value->maximum_number_phys, ppWritePackedMsg, end) &&
           push16(value->maximum_total_bandwidth, ppWritePackedMsg, end) &&
           push8(value->maximum_total_number_dl_layers, ppWritePackedMsg, end) &&
           push8(value->maximum_total_number_ul_layers, ppWritePackedMsg, end) &&
           push8(value->shared_bands, ppWritePackedMsg, end) &&
           push8(value->shared_pa, ppWritePackedMsg, end) &&
           pushs16(value->maximum_total_power, ppWritePackedMsg, end) &&
           pusharray8(value->oui, NFAPI_PNF_PARAM_GENERAL_OUI_LENGTH, NFAPI_PNF_PARAM_GENERAL_OUI_LENGTH, ppWritePackedMsg, end));
}

uint8_t pack_rf_config_info(void *elem, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_rf_config_info_t *rf = (nfapi_rf_config_info_t *)elem;
  return (push16(rf->rf_config_index, ppWritePackedMsg, end));
}


static uint8_t pack_pnf_phy_info(void *elem, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_pnf_phy_info_t *phy = (nfapi_pnf_phy_info_t *)elem;
  return (  push16(phy->phy_config_index, ppWritePackedMsg, end) &&
            push16(phy->number_of_rfs, ppWritePackedMsg, end) &&
            packarray(phy->rf_config, sizeof(nfapi_rf_config_info_t), NFAPI_MAX_PNF_PHY_RF_CONFIG, phy->number_of_rfs, ppWritePackedMsg, end, &pack_rf_config_info) &&
            push16(phy->number_of_rf_exclusions, ppWritePackedMsg, end) &&
            packarray(phy->excluded_rf_config, sizeof(nfapi_rf_config_info_t), NFAPI_MAX_PNF_PHY_RF_CONFIG, phy->number_of_rf_exclusions, ppWritePackedMsg, end, &pack_rf_config_info) &&
            push16(phy->downlink_channel_bandwidth_supported, ppWritePackedMsg, end) &&
            push16(phy->uplink_channel_bandwidth_supported, ppWritePackedMsg, end) &&
            push8(phy->number_of_dl_layers_supported, ppWritePackedMsg, end) &&
            push8(phy->number_of_ul_layers_supported, ppWritePackedMsg, end) &&
            push16(phy->maximum_3gpp_release_supported, ppWritePackedMsg, end) &&
            push8(phy->nmm_modes_supported, ppWritePackedMsg, end));
}

uint8_t pack_pnf_phy_value(void *tlv, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_pnf_phy_t *value = (nfapi_pnf_phy_t *)tlv;
  return ( push16(value->number_of_phys, ppWritePackedMsg, end) &&
           packarray(value->phy, sizeof(nfapi_pnf_phy_info_t), NFAPI_MAX_PNF_PHY, value->number_of_phys, ppWritePackedMsg, end, &pack_pnf_phy_info));
}

static uint8_t pack_phy_rf_config_info(void *elem, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_phy_rf_config_info_t *rf = (nfapi_phy_rf_config_info_t *)elem;
  return (push16(rf->phy_id, ppWritePackedMsg, end) &&
          push16(rf->phy_config_index, ppWritePackedMsg, end) &&
          push16(rf->rf_config_index, ppWritePackedMsg, end));
}


uint8_t pack_pnf_phy_rf_config_value(void *tlv, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_pnf_phy_rf_config_t *value = (nfapi_pnf_phy_rf_config_t *)tlv;
  return(push16(value->number_phy_rf_config_info, ppWritePackedMsg, end) &&
         packarray(value->phy_rf_config, sizeof(nfapi_phy_rf_config_info_t), NFAPI_MAX_PHY_RF_INSTANCES, value->number_phy_rf_config_info, ppWritePackedMsg, end, &pack_phy_rf_config_info));
}


uint8_t pack_ipv4_address_value(void *tlv, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_ipv4_address_t *value = (nfapi_ipv4_address_t *)tlv;
  return pusharray8(value->address, NFAPI_IPV4_ADDRESS_LENGTH, NFAPI_IPV4_ADDRESS_LENGTH, ppWritePackedMsg, end);
}
uint8_t unpack_ipv4_address_value(void *tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_ipv4_address_t *value = (nfapi_ipv4_address_t *)tlv;
  return pullarray8(ppReadPackedMsg, value->address, NFAPI_IPV4_ADDRESS_LENGTH, NFAPI_IPV4_ADDRESS_LENGTH, end);
}
uint8_t pack_ipv6_address_value(void *tlv, uint8_t **ppWritePackedMsg, uint8_t *end) {
  nfapi_ipv6_address_t *value = (nfapi_ipv6_address_t *)tlv;
  return pusharray8(value->address, NFAPI_IPV6_ADDRESS_LENGTH, NFAPI_IPV6_ADDRESS_LENGTH, ppWritePackedMsg, end);
}
uint8_t unpack_ipv6_address_value(void *tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_ipv4_address_t *value = (nfapi_ipv4_address_t *)tlv;
  return pullarray8(ppReadPackedMsg, value->address, NFAPI_IPV6_ADDRESS_LENGTH, NFAPI_IPV6_ADDRESS_LENGTH, end);
}

uint8_t pack_stop_response(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  nfapi_stop_response_t *pNfapiMsg = (nfapi_stop_response_t *)msg;
  return ( push32(pNfapiMsg->error_code, ppWritePackedMsg, end) &&
           pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config) );
}

// helper function for message length calculation -
// takes the pointers to the start of message to end of message

uint32_t get_packed_msg_len(uintptr_t msgHead, uintptr_t msgEnd) {
  if (msgEnd < msgHead) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "get_packed_msg_len: Error in pointers supplied %p, %p\n", &msgHead, &msgEnd);
    return 0;
  }

  return (msgEnd - msgHead);
}

uint8_t unpack_pnf_param_general_value(void *tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_pnf_param_general_t *value = (nfapi_pnf_param_general_t *)tlv;
  return( pull8(ppReadPackedMsg, &value->nfapi_sync_mode, end) &&
          pull8(ppReadPackedMsg, &value->location_mode, end) &&
          pull16(ppReadPackedMsg, &value->location_coordinates_length, end) &&
          pullarray8(ppReadPackedMsg, value->location_coordinates, NFAPI_PNF_PARAM_GENERAL_LOCATION_LENGTH, value->location_coordinates_length, end) &&
          pull32(ppReadPackedMsg, &value->dl_config_timing, end) &&
          pull32(ppReadPackedMsg, &value->tx_timing, end) &&
          pull32(ppReadPackedMsg, &value->ul_config_timing, end) &&
          pull32(ppReadPackedMsg, &value->hi_dci0_timing, end) &&
          pull16(ppReadPackedMsg, &value->maximum_number_phys, end) &&
          pull16(ppReadPackedMsg, &value->maximum_total_bandwidth, end) &&
          pull8(ppReadPackedMsg, &value->maximum_total_number_dl_layers, end) &&
          pull8(ppReadPackedMsg, &value->maximum_total_number_ul_layers, end) &&
          pull8(ppReadPackedMsg, &value->shared_bands, end) &&
          pull8(ppReadPackedMsg, &value->shared_pa, end) &&
          pulls16(ppReadPackedMsg, &value->maximum_total_power, end) &&
          pullarray8(ppReadPackedMsg, value->oui, NFAPI_PNF_PARAM_GENERAL_OUI_LENGTH, NFAPI_PNF_PARAM_GENERAL_OUI_LENGTH, end));
}

static uint8_t unpack_rf_config_info(void *elem, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_rf_config_info_t *info = (nfapi_rf_config_info_t *)elem;
  return pull16(ppReadPackedMsg, &info->rf_config_index, end);
}

static uint8_t unpack_pnf_phy_info(void *elem, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_pnf_phy_info_t *phy = (nfapi_pnf_phy_info_t *)elem;
  return ( pull16(ppReadPackedMsg, &phy->phy_config_index, end) &&
           pull16(ppReadPackedMsg, &phy->number_of_rfs, end) &&
           unpackarray(ppReadPackedMsg, phy->rf_config, sizeof(nfapi_rf_config_info_t), NFAPI_MAX_PNF_PHY_RF_CONFIG, phy->number_of_rfs, end, &unpack_rf_config_info) &&
           pull16(ppReadPackedMsg, &phy->number_of_rf_exclusions, end) &&
           unpackarray(ppReadPackedMsg, phy->excluded_rf_config, sizeof(nfapi_rf_config_info_t), NFAPI_MAX_PNF_PHY_RF_CONFIG, phy->number_of_rf_exclusions, end, &unpack_rf_config_info) &&
           pull16(ppReadPackedMsg, &phy->downlink_channel_bandwidth_supported, end) &&
           pull16(ppReadPackedMsg, &phy->uplink_channel_bandwidth_supported, end) &&
           pull8(ppReadPackedMsg, &phy->number_of_dl_layers_supported, end) &&
           pull8(ppReadPackedMsg, &phy->number_of_ul_layers_supported, end) &&
           pull16(ppReadPackedMsg, &phy->maximum_3gpp_release_supported, end) &&
           pull8(ppReadPackedMsg, &phy->nmm_modes_supported, end));
}


uint8_t unpack_pnf_phy_value(void *tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_pnf_phy_t *value = (nfapi_pnf_phy_t *)tlv;
  return ( pull16(ppReadPackedMsg, &value->number_of_phys, end) &&
           unpackarray(ppReadPackedMsg, value->phy, sizeof(nfapi_pnf_phy_info_t), NFAPI_MAX_PNF_PHY, value->number_of_phys, end, &unpack_pnf_phy_info));
}
static uint8_t unpack_phy_rf_config_info(void *elem, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_phy_rf_config_info_t *rf = (nfapi_phy_rf_config_info_t *)elem;
  return( pull16(ppReadPackedMsg, &rf->phy_id, end) &&
          pull16(ppReadPackedMsg, &rf->phy_config_index, end) &&
          pull16(ppReadPackedMsg, &rf->rf_config_index, end));
}

uint8_t unpack_pnf_phy_rf_config_value(void *tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_pnf_phy_rf_config_t *value = (nfapi_pnf_phy_rf_config_t *)tlv;
  return ( pull16(ppReadPackedMsg, &value->number_phy_rf_config_info, end) &&
           unpackarray(ppReadPackedMsg, value->phy_rf_config, sizeof(nfapi_phy_rf_config_info_t), NFAPI_MAX_PHY_RF_INSTANCES, value->number_phy_rf_config_info, end, &unpack_phy_rf_config_info));
}
