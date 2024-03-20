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
/*! \file nfapi/open-nFAPI/nfapi/src/nr_nfapi_p5.c
 * \brief Defines the NR specific packing/unpacking procedures for NR nFAPI messages (SCF 225)
 * \author Ruben S. Silva
 * \date 2024
 * \version 0.1
 * \company OpenAirInterface Software Alliance
 * \email: contact@openairinterface.org, rsilva@allbesmart.pt
 * \note
 * \warning
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

#include "nfapi_interface.h"
#include <nfapi.h>
#include "nfapi_nr_interface.h"
#include "nfapi_nr_interface_scf.h"
#include "nfapi/oai_integration/vendor_ext.h"
#include <debug.h>
#include "nr_fapi_p5.h"

static uint8_t pack_nr_pnf_param_request(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_param_request_t *request = (nfapi_nr_pnf_param_request_t *)msg;
  return pack_vendor_extension_tlv(request->vendor_extension, ppWritePackedMsg, end, config);
}

static uint8_t pack_nr_pnf_param_response(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_param_response_t *pNfapiMsg = (nfapi_nr_pnf_param_response_t *)msg;
  return (
      push8(pNfapiMsg->error_code, ppWritePackedMsg, end) && push8(pNfapiMsg->num_tlvs, ppWritePackedMsg, end) && // numTLVs
      pack_nr_tlv(NFAPI_PNF_PARAM_GENERAL_TAG, &pNfapiMsg->pnf_param_general, ppWritePackedMsg, end, &pack_pnf_param_general_value)
      && pack_nr_tlv(NFAPI_PNF_PHY_TAG, &pNfapiMsg->pnf_phy, ppWritePackedMsg, end, &pack_pnf_phy_value)
      && pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config));
}

static uint8_t pack_nr_pnf_config_request(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_config_request_t *pNfapiMsg = (nfapi_nr_pnf_config_request_t *)msg;
  return (push8(pNfapiMsg->num_tlvs, ppWritePackedMsg, end) && // numTLVs
          pack_nr_tlv(NFAPI_PNF_PHY_RF_TAG, &pNfapiMsg->pnf_phy_rf_config, ppWritePackedMsg, end, &pack_pnf_phy_rf_config_value)
          && pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config));
}

static uint8_t pack_nr_pnf_config_response(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_config_response_t *pNfapiMsg = (nfapi_nr_pnf_config_response_t *)msg;
  return (push8(pNfapiMsg->error_code, ppWritePackedMsg, end)
          && pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config));
}

static uint8_t pack_nr_pnf_start_request(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_start_request_t *pNfapiMsg = (nfapi_nr_pnf_start_request_t *)msg;
  return ( pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config));
}

static uint8_t pack_nr_pnf_start_response(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_start_response_t *pNfapiMsg = (nfapi_nr_pnf_start_response_t *)msg;
  return( push32(pNfapiMsg->error_code, ppWritePackedMsg, end) &&
          pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config));
}

static uint8_t pack_nr_pnf_stop_request(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_stop_request_t *pNfapiMsg = (nfapi_nr_pnf_stop_request_t *)msg;
  return ( pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config) );
}

static uint8_t pack_nr_pnf_stop_response(void *msg, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_stop_response_t *pNfapiMsg = (nfapi_nr_pnf_stop_response_t *)msg;
  return ( push32(pNfapiMsg->error_code, ppWritePackedMsg, end) &&
          pack_vendor_extension_tlv(pNfapiMsg->vendor_extension, ppWritePackedMsg, end, config));
}

static uint8_t pack_nr_p5_message_body(nfapi_p4_p5_message_header_t *header, uint8_t **ppWritePackedMsg, uint8_t *end, nfapi_p4_p5_codec_config_t *config) {
  uint8_t result = 0;

  // look for the specific message
  switch (header->message_id) {
    case NFAPI_NR_PHY_MSG_TYPE_PNF_PARAM_REQUEST:
      result = pack_nr_pnf_param_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_PNF_PARAM_RESPONSE:
      result = pack_nr_pnf_param_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_CONFIG_REQUEST:
      result = pack_nr_pnf_config_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_CONFIG_RESPONSE:
      result = pack_nr_pnf_config_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_START_REQUEST:
      result = pack_nr_pnf_start_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_START_RESPONSE:
      result = pack_nr_pnf_start_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_STOP_REQUEST:
      result = pack_nr_pnf_stop_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_PNF_STOP_RESPONSE:
      result = pack_nr_pnf_stop_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PARAM_REQUEST:
      result = pack_nr_param_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PARAM_RESPONSE:
      result = pack_nr_param_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_CONFIG_REQUEST:
      result = pack_nr_config_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_CONFIG_RESPONSE:
      result = pack_nr_config_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_START_REQUEST:
      result = pack_nr_start_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_START_RESPONSE:
      result = pack_nr_start_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_STOP_REQUEST:
      result = pack_nr_stop_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_STOP_RESPONSE:
      result = pack_stop_response(header, ppWritePackedMsg, end, config);
      break;

    default: {
      if(header->message_id >= NFAPI_VENDOR_EXT_MSG_MIN &&
          header->message_id <= NFAPI_VENDOR_EXT_MSG_MAX) {
        if(config && config->pack_p4_p5_vendor_extension) {
          result = (config->pack_p4_p5_vendor_extension)(header, ppWritePackedMsg, end, config);
        } else {
          NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s VE NFAPI message ID %d. No ve ecoder provided\n", __FUNCTION__, header->message_id);
        }
      } else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s NFAPI Unknown message ID %d\n", __FUNCTION__, header->message_id);
      }
    }
    break;
  }

  return result;
}


int nfapi_nr_p5_message_pack(void *pMessageBuf,
                             uint32_t messageBufLen,
                             void *pPackedBuf,
                             uint32_t packedBufLen,
                             nfapi_p4_p5_codec_config_t *config)
{
  nfapi_p4_p5_message_header_t *pMessageHeader = pMessageBuf;
  uint8_t *pWritePackedMessage = pPackedBuf;
  uint32_t packedMsgLen;
  uint32_t packedBodyLen;
  uint16_t packedMsgLen16;

  if (pMessageBuf == NULL || pPackedBuf == NULL) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P5 Pack supplied pointers are null\n");
    return -1;
  }

  uint8_t *pPackMessageEnd = pPackedBuf + packedBufLen;
  uint8_t *pPackedLengthField = &pWritePackedMessage[4];
  uint8_t *pPacketBodyFieldStart = &pWritePackedMessage[8];
  uint8_t *pPacketBodyField = &pWritePackedMessage[8];

  // pack the message
  if (push16(pMessageHeader->phy_id, &pWritePackedMessage, pPackMessageEnd)
      && push16(pMessageHeader->message_id, &pWritePackedMessage, pPackMessageEnd)
      && push16(0, &pWritePackedMessage, pPackMessageEnd) && push16(pMessageHeader->spare, &pWritePackedMessage, pPackMessageEnd)
      && pack_nr_p5_message_body(pMessageHeader, &pPacketBodyField, pPackMessageEnd, config)) {
    // to check if whole message is bigger than the buffer provided
    packedMsgLen = get_packed_msg_len((uintptr_t)pPackedBuf, (uintptr_t)pPacketBodyField);
    // obtain the length of the message body to pack
    packedBodyLen = get_packed_msg_len((uintptr_t)pPacketBodyFieldStart, (uintptr_t)pPacketBodyField);

    if (packedMsgLen > 0xFFFF || packedMsgLen > packedBufLen) {
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "Packed message length error %d, buffer supplied %d\n", packedMsgLen, packedBufLen);
      return -1;
    } else {
      packedMsgLen16 = (uint16_t)packedBodyLen;
    }

    // Update the message length in the header
    if (!push16(packedMsgLen16, &pPackedLengthField, pPackMessageEnd))
      return -1;

    // return the packed length
    return (packedMsgLen);
  } else {
    // Failed to pack the meassage
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P5 Failed to pack message\n");
    return -1;
  }
}


static int check_nr_unpack_length(nfapi_nr_phy_msg_type_e msgId, uint32_t unpackedBufLen) {
  int retLen = 0;

  switch (msgId) {
    case NFAPI_NR_PHY_MSG_TYPE_PNF_PARAM_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_pnf_param_request_t))
        retLen = sizeof(nfapi_pnf_param_request_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_PARAM_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_param_response_t))
        retLen = sizeof(nfapi_nr_pnf_param_response_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_CONFIG_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_config_request_t))
        retLen = sizeof(nfapi_nr_pnf_config_request_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_CONFIG_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_config_response_t))
        retLen = sizeof(nfapi_nr_pnf_config_response_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_START_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_start_request_t))
        retLen = sizeof(nfapi_nr_pnf_start_request_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_START_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_start_response_t))
        retLen = sizeof(nfapi_nr_pnf_start_response_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_STOP_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_stop_request_t))
        retLen = sizeof(nfapi_nr_pnf_stop_request_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_STOP_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_pnf_stop_response_t))
        retLen = sizeof(nfapi_nr_pnf_stop_response_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PARAM_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_nr_param_request_scf_t))
        retLen = sizeof(nfapi_nr_param_request_scf_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_PARAM_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_param_response_scf_t))
        retLen = sizeof(nfapi_nr_param_response_scf_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_CONFIG_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_nr_config_request_scf_t))
        retLen = sizeof(nfapi_nr_config_request_scf_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_CONFIG_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_config_response_scf_t))
        retLen = sizeof(nfapi_nr_config_response_scf_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_START_REQUEST:
      if (unpackedBufLen >= sizeof( nfapi_nr_start_request_scf_t))
        retLen = sizeof( nfapi_nr_start_request_scf_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_START_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_nr_start_response_scf_t))
        retLen = sizeof(nfapi_nr_start_response_scf_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_STOP_REQUEST:
      if (unpackedBufLen >= sizeof(nfapi_stop_request_t))
        retLen = sizeof(nfapi_stop_request_t);

      break;

    case NFAPI_NR_PHY_MSG_TYPE_STOP_RESPONSE:
      if (unpackedBufLen >= sizeof(nfapi_stop_response_t))
        retLen = sizeof(nfapi_stop_response_t);

      break;

    default:
      NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s Unknown message ID %d\n", __FUNCTION__, msgId);
      break;
  }

  return retLen;
}

static uint8_t unpack_nr_stop_response(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config) {
  nfapi_stop_response_t *pNfapiMsg = (nfapi_stop_response_t *)msg;
  return ( pull32(ppReadPackedMsg, &pNfapiMsg->error_code, end) &&
          unpack_nr_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension)));
}
static uint8_t unpack_nr_measurement_request(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config) {
  nfapi_measurement_request_t *pNfapiMsg = (nfapi_measurement_request_t *)msg;
  unpack_tlv_t unpack_fns[] = {
      { NFAPI_MEASUREMENT_REQUEST_DL_RS_XTX_POWER_TAG, &pNfapiMsg->dl_rs_tx_power, &unpack_uint16_tlv_value},
      { NFAPI_MEASUREMENT_REQUEST_RECEIVED_INTERFERENCE_POWER_TAG, &pNfapiMsg->received_interference_power, &unpack_uint16_tlv_value},
      { NFAPI_MEASUREMENT_REQUEST_THERMAL_NOISE_POWER_TAG, &pNfapiMsg->thermal_noise_power, &unpack_uint16_tlv_value},
  };
  return unpack_nr_tlv_list(unpack_fns, sizeof(unpack_fns)/sizeof(unpack_tlv_t), ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension));
}

static uint8_t unpack_received_interference_power_measurement_value(void *tlv, uint8_t **ppReadPackedMsg, uint8_t *end) {
  nfapi_received_interference_power_measurement_t *value = (nfapi_received_interference_power_measurement_t *)tlv;
  return ( pull16(ppReadPackedMsg, &value->number_of_resource_blocks, end) &&
          pullarrays16(ppReadPackedMsg, value->received_interference_power,  NFAPI_MAX_RECEIVED_INTERFERENCE_POWER_RESULTS, value->number_of_resource_blocks, end));
}

static uint8_t unpack_nr_measurement_response(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config) {
  nfapi_measurement_response_t *pNfapiMsg = (nfapi_measurement_response_t *)msg;
  unpack_tlv_t unpack_fns[] = {
      { NFAPI_MEASUREMENT_RESPONSE_DL_RS_POWER_MEASUREMENT_TAG, &pNfapiMsg->dl_rs_tx_power_measurement, &unpack_int16_tlv_value},
      { NFAPI_MEASUREMENT_RESPONSE_RECEIVED_INTERFERENCE_POWER_MEASUREMENT_TAG, &pNfapiMsg->received_interference_power_measurement, &unpack_received_interference_power_measurement_value},
      { NFAPI_MEASUREMENT_RESPONSE_THERMAL_NOISE_MEASUREMENT_TAG, &pNfapiMsg->thermal_noise_power_measurement, &unpack_int16_tlv_value},
  };
  return ( pull32(ppReadPackedMsg, &pNfapiMsg->error_code, end) &&
          unpack_nr_tlv_list(unpack_fns, sizeof(unpack_fns)/sizeof(unpack_tlv_t), ppReadPackedMsg, end, config, &pNfapiMsg->vendor_extension));
}

static uint8_t unpack_nr_pnf_param_request(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_param_request_t *pNfapiMsg = (nfapi_nr_pnf_param_request_t *)msg;
  return unpack_nr_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension));
}

static uint8_t unpack_nr_pnf_param_response(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_param_response_t *pNfapiMsg = (nfapi_nr_pnf_param_response_t *)msg;
  unpack_tlv_t unpack_fns[] = {
      {NFAPI_PNF_PARAM_GENERAL_TAG, &pNfapiMsg->pnf_param_general, &unpack_pnf_param_general_value},
      {NFAPI_PNF_PHY_TAG, &pNfapiMsg->pnf_phy, &unpack_pnf_phy_value},
  };
  return (pull8(ppReadPackedMsg, (uint8_t *)&pNfapiMsg->error_code, end)
          && pull8(ppReadPackedMsg, (uint8_t *)&pNfapiMsg->num_tlvs, end)
          && unpack_nr_tlv_list(unpack_fns,
                                sizeof(unpack_fns) / sizeof(unpack_tlv_t),
                                ppReadPackedMsg,
                                end,
                                config,
                                &pNfapiMsg->vendor_extension));
}

static uint8_t unpack_nr_pnf_config_request(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_config_request_t *pNfapiMsg = (nfapi_nr_pnf_config_request_t *)msg;
  pull8(ppReadPackedMsg, &pNfapiMsg->num_tlvs, end);
  unpack_tlv_t unpack_fns[] = {
      {NFAPI_PNF_PHY_RF_TAG, &pNfapiMsg->pnf_phy_rf_config, &unpack_pnf_phy_rf_config_value},
  };
  return unpack_nr_tlv_list(unpack_fns,
                            sizeof(unpack_fns) / sizeof(unpack_tlv_t),
                            ppReadPackedMsg,
                            end,
                            config,
                            &pNfapiMsg->vendor_extension);
}

static uint8_t unpack_nr_pnf_config_response(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_config_response_t *pNfapiMsg = (nfapi_nr_pnf_config_response_t *)msg;
  return (pull8(ppReadPackedMsg, &pNfapiMsg->error_code, end)
          && unpack_nr_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension)));
}

static uint8_t unpack_nr_pnf_start_request(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_start_request_t *pNfapiMsg = (nfapi_nr_pnf_start_request_t *)msg;
  return unpack_nr_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension));
}

static uint8_t unpack_nr_pnf_start_response(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config)
{
  nfapi_nr_pnf_start_response_t *pNfapiMsg = (nfapi_nr_pnf_start_response_t *)msg;
  return (pull32(ppReadPackedMsg, &pNfapiMsg->error_code, end)
          && unpack_nr_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension)));
}

static uint8_t unpack_nr_pnf_stop_request(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_stop_request_t *pNfapiMsg = (nfapi_nr_pnf_stop_request_t *)msg;
  return unpack_nr_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension));
}

static uint8_t unpack_nr_pnf_stop_response(uint8_t **ppReadPackedMsg, uint8_t *end, void *msg, nfapi_p4_p5_codec_config_t *config) {
  nfapi_nr_pnf_stop_response_t *pNfapiMsg = (nfapi_nr_pnf_stop_response_t *)msg;
  return ( pull32(ppReadPackedMsg, &pNfapiMsg->error_code, end) &&
          unpack_tlv_list(NULL, 0, ppReadPackedMsg, end, config, &(pNfapiMsg->vendor_extension)));
}

int nfapi_nr_p5_message_unpack(void *pMessageBuf,
                               uint32_t messageBufLen,
                               void *pUnpackedBuf,
                               uint32_t unpackedBufLen,
                               nfapi_p4_p5_codec_config_t *config)
{
  nfapi_p4_p5_message_header_t *pMessageHeader = pUnpackedBuf;
  uint8_t *pReadPackedMessage = pMessageBuf;

  if (pMessageBuf == NULL || pUnpackedBuf == NULL) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P5 unpack supplied pointers are null\n");
    return -1;
  }

  uint8_t *end = (uint8_t *)pMessageBuf + messageBufLen;

  if (messageBufLen < NFAPI_HEADER_LENGTH || unpackedBufLen < sizeof(nfapi_p4_p5_message_header_t)) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P5 unpack supplied message buffer is too small %d, %d\n", messageBufLen, unpackedBufLen);
    return -1;
  }

  uint8_t *ptr = pReadPackedMessage;
  printf("\n Read NR message unpack: ");

  while (ptr < end) {
    printf(" %02x ", *ptr);
    ptr++;
  }

  printf("\n");
  // clean the supplied buffer for - tag value blanking
  (void)memset(pUnpackedBuf, 0, unpackedBufLen);

  // process the header
  if (!(pull16(&pReadPackedMessage, &pMessageHeader->phy_id, end) && pull16(&pReadPackedMessage, &pMessageHeader->message_id, end)
        && pull16(&pReadPackedMessage, &pMessageHeader->message_length, end)
        && pull16(&pReadPackedMessage, &pMessageHeader->spare, end))) {
    // failed to read the header
    return -1;
  }

  int result = -1;

  if (check_nr_unpack_length(pMessageHeader->message_id, unpackedBufLen) == 0) {
    // the unpack buffer is not big enough for the struct
    return -1;
  }

  // look for the specific message
  switch (pMessageHeader->message_id) {
    case NFAPI_NR_PHY_MSG_TYPE_PNF_PARAM_REQUEST:
      result = unpack_nr_pnf_param_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_PARAM_RESPONSE:
      result = unpack_nr_pnf_param_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_CONFIG_REQUEST:
      result = unpack_nr_pnf_config_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_CONFIG_RESPONSE:
      result = unpack_nr_pnf_config_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_START_REQUEST:
      result = unpack_nr_pnf_start_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_START_RESPONSE:
      result = unpack_nr_pnf_start_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_STOP_REQUEST:
      result = unpack_nr_pnf_stop_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PNF_STOP_RESPONSE:
      result = unpack_nr_pnf_stop_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PARAM_REQUEST:
      result = unpack_nr_param_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_PARAM_RESPONSE:
      result = unpack_nr_param_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_CONFIG_REQUEST:
      result = unpack_nr_config_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_CONFIG_RESPONSE:
      result = unpack_nr_config_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_START_REQUEST:
      result = unpack_nr_start_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_START_RESPONSE:
      result = unpack_nr_start_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_STOP_REQUEST:
      result = unpack_nr_stop_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_STOP_RESPONSE:
      result = unpack_nr_stop_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_MEASUREMENT_REQUEST:
      result = unpack_nr_measurement_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_MEASUREMENT_RESPONSE:
      result = unpack_nr_measurement_response(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    default:
      if (pMessageHeader->message_id >= NFAPI_VENDOR_EXT_MSG_MIN && pMessageHeader->message_id <= NFAPI_VENDOR_EXT_MSG_MAX) {
        if (config && config->unpack_p4_p5_vendor_extension) {
          result = (config->unpack_p4_p5_vendor_extension)(pMessageHeader, &pReadPackedMessage, end, config);
        } else {
          NFAPI_TRACE(NFAPI_TRACE_ERROR,
                      "%s VE NFAPI message ID %d. No ve decoder provided\n",
                      __FUNCTION__,
                      pMessageHeader->message_id);
        }
      } else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s NFAPI Unknown P5 message ID %d\n", __FUNCTION__, pMessageHeader->message_id);
      }

      break;
  }

  return result;
}


