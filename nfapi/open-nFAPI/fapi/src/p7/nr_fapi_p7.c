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
/*! \file nfapi/open-nFAPI/fapi/src/p7/nr_fapi_p7.c
 * \brief
 * \author Ruben S. Silva
 * \date 2024
 * \version 0.1
 * \company OpenAirInterface Software Alliance
 * \email: contact@openairinterface.org, rsilva@allbesmart.pt
 * \note
 * \warning
 */
#include "nr_fapi.h"
#include "nr_fapi_p7.h"
#include "nr_nfapi_p7.h"
#include "debug.h"

uint8_t fapi_nr_p7_message_body_pack(nfapi_p7_message_header_t *header,
                                     uint8_t **ppWritePackedMsg,
                                     uint8_t *end,
                                     nfapi_p7_codec_config_t *config)
{
  // look for the specific message
  uint8_t result = 0;
  switch (header->message_id) {
    case NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST:
      result = pack_dl_tti_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST:
      result = pack_ul_tti_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST:
      result = pack_tx_data_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST:
      result = pack_ul_dci_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_UE_RELEASE_REQUEST:
      result = pack_ue_release_request(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_UE_RELEASE_RESPONSE:
      result = pack_ue_release_response(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_SLOT_INDICATION:
      result = pack_nr_slot_indication(header, ppWritePackedMsg, end, config);

    case NFAPI_NR_PHY_MSG_TYPE_RX_DATA_INDICATION:
      result = pack_nr_rx_data_indication(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_CRC_INDICATION:
      result = pack_nr_crc_indication(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_UCI_INDICATION:
      result = pack_nr_uci_indication(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_SRS_INDICATION:
      result = pack_nr_srs_indication(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_RACH_INDICATION:
      result = pack_nr_rach_indication(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_DL_NODE_SYNC:
      result = pack_nr_dl_node_sync(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_UL_NODE_SYNC:
      result = pack_nr_ul_node_sync(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_TIMING_INFO:
      result = pack_nr_timing_info(header, ppWritePackedMsg, end, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_VENDOR_EXT_SLOT_RESPONSE:
      result = pack_nr_slot_indication(header, ppWritePackedMsg, end, config);
      break;

    default: {
      if (header->message_id >= NFAPI_VENDOR_EXT_MSG_MIN && header->message_id <= NFAPI_VENDOR_EXT_MSG_MAX) {
        if (config && config->pack_p7_vendor_extension) {
          result = (config->pack_p7_vendor_extension)(header, ppWritePackedMsg, end, config);
        } else {
          NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s VE NFAPI message ID %d. No ve decoder provided\n", __FUNCTION__, header->message_id);
        }
      } else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s NFAPI Unknown message ID %d\n", __FUNCTION__, header->message_id);
      }
    } break;
  }

  if (result == 0) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 Pack failed to pack message\n");
    return -1;
  }
}

int fapi_nr_p7_message_pack(void *pMessageBuf, void *pPackedBuf, uint32_t packedBufLen, nfapi_p7_codec_config_t *config)
{
  if (pMessageBuf == NULL || pPackedBuf == NULL) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 Pack supplied pointers are null\n");
    return -1;
  }

  nfapi_p7_message_header_t *pMessageHeader = pMessageBuf;
  uint8_t *end = pPackedBuf + packedBufLen;
  uint8_t *pWritePackedMessage = pPackedBuf;
  uint8_t *pPackMessageEnd = pPackedBuf + packedBufLen;
  uint8_t *pPackedLengthField = &pWritePackedMessage[4];
  uint8_t *pPacketBodyField = &pWritePackedMessage[8];
  uint8_t *pPacketBodyFieldStart = &pWritePackedMessage[8];

  uint8_t result = fapi_nr_p7_message_body_pack(pMessageHeader, &pPacketBodyField, pPackMessageEnd, config);
  AssertFatal(result >= 0, "fapi_nr_p7_message_body_pack error packing message body %d\n", result);

  // PHY API message header
  // Number of messages [0]
  // Opaque handle [1]
  // PHY API Message structure
  // Message type ID [2,3]
  // Message Length [4,5,6,7]
  // Message Body [8,...]
  if (!(push8(1, &pWritePackedMessage, pPackMessageEnd) && push8(0, &pWritePackedMessage, pPackMessageEnd)
        && push16(pMessageHeader->message_id, &pWritePackedMessage, pPackMessageEnd))) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 Pack header failed\n");
    return -1;
  }

  // check for a valid message length
  uintptr_t msgHead = (uintptr_t)pPacketBodyFieldStart;
  uintptr_t msgEnd = (uintptr_t)pPacketBodyField;
  uint32_t packedMsgLen = msgEnd - msgHead;
  uint16_t packedMsgLen16;
  if (packedMsgLen > 0xFFFF || packedMsgLen > packedBufLen) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "Packed message length error %d, buffer supplied %d\n", packedMsgLen, packedBufLen);
    return -1;
  } else {
    packedMsgLen16 = (uint16_t)packedMsgLen;
  }

  // Update the message length in the header
  pMessageHeader->message_length = packedMsgLen16;

  // Update the message length in the header
  if (!push32(packedMsgLen, &pPackedLengthField, pPackMessageEnd))
    return -1;

  if (1) {
    // quick test
    if (pMessageHeader->message_length != packedMsgLen) {
      NFAPI_TRACE(NFAPI_TRACE_ERROR,
                  "nfapi packedMsgLen(%d) != message_length(%d) id %d\n",
                  packedMsgLen,
                  pMessageHeader->message_length,
                  pMessageHeader->message_id);
    }
  }

  return (packedMsgLen16);
}

int fapi_nr_p7_message_unpack(void *pMessageBuf,
                              uint32_t messageBufLen,
                              void *pUnpackedBuf,
                              uint32_t unpackedBufLen,
                              nfapi_p7_codec_config_t *config)
{
  int result = 0;
  nfapi_p7_message_header_t *pMessageHeader = (nfapi_p7_message_header_t *)pUnpackedBuf;
  uint8_t *pReadPackedMessage = pMessageBuf;

  if (pMessageBuf == NULL || pUnpackedBuf == NULL) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 unpack supplied pointers are null\n");
    return -1;
  }

  uint8_t *end = (uint8_t *)pMessageBuf + messageBufLen;

  if (messageBufLen < NFAPI_P7_HEADER_LENGTH || unpackedBufLen < sizeof(nfapi_p7_message_header_t)) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 unpack supplied message buffer is too small %d, %d\n", messageBufLen, unpackedBufLen);
    return -1;
  }

  // clean the supplied buffer for - tag value blanking
  (void)memset(pUnpackedBuf, 0, unpackedBufLen);

  // process the header
  if (!(pull16(&pReadPackedMessage, &pMessageHeader->phy_id, end) && pull16(&pReadPackedMessage, &pMessageHeader->message_id, end)
        && pull16(&pReadPackedMessage, &pMessageHeader->message_length, end)
        && pull16(&pReadPackedMessage, &pMessageHeader->m_segment_sequence, end)
        && pull32(&pReadPackedMessage, &pMessageHeader->checksum, end)
        && pull32(&pReadPackedMessage, &pMessageHeader->transmit_timestamp, end))) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 unpack header failed\n");
    return -1;
  }

  if ((uint8_t *)(pMessageBuf + pMessageHeader->message_length) > end) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 unpack message length is greater than the message buffer \n");
    return -1;
  }

  /*
  if(check_unpack_length(pMessageHeader->message_id, unpackedBufLen) == 0)
  {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 unpack unpack buffer is not large enough \n");
    return -1;
  }
  */

  // look for the specific message
  switch (pMessageHeader->message_id) {
    case NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_DL_TTI_REQUEST, unpackedBufLen))
        result = unpack_dl_tti_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;

    case NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_UL_TTI_REQUEST, unpackedBufLen))
        result = unpack_ul_tti_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;
    case NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_TX_DATA_REQUEST, unpackedBufLen))
        result = unpack_tx_data_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;
    case NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_UL_DCI_REQUEST, unpackedBufLen))
        result = unpack_ul_dci_request(&pReadPackedMessage, end, pMessageHeader, config);
      break;
    case NFAPI_NR_PHY_MSG_TYPE_SLOT_INDICATION:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_SLOT_INDICATION, unpackedBufLen)) {
        nfapi_nr_slot_indication_scf_t *msg = (nfapi_nr_slot_indication_scf_t *)pMessageHeader;
        result = unpack_nr_slot_indication(&pReadPackedMessage, end, msg, config);
      }
      break;

    case NFAPI_NR_PHY_MSG_TYPE_RX_DATA_INDICATION:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_RX_DATA_INDICATION, unpackedBufLen)) {
        nfapi_nr_rx_data_indication_t *msg = (nfapi_nr_rx_data_indication_t *)pMessageHeader;
        msg->pdu_list = (nfapi_nr_rx_data_pdu_t *)malloc(sizeof(nfapi_nr_rx_data_pdu_t));
        msg->pdu_list->pdu = (uint8_t *)malloc(sizeof(uint8_t));
        result = unpack_nr_rx_data_indication(&pReadPackedMessage, end, msg, config);
      }
      break;

    case NFAPI_NR_PHY_MSG_TYPE_CRC_INDICATION:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_CRC_INDICATION, unpackedBufLen)) {
        nfapi_nr_crc_indication_t *msg = (nfapi_nr_crc_indication_t *)pMessageHeader;
        msg->crc_list = (nfapi_nr_crc_t *)malloc(sizeof(nfapi_nr_crc_t));
        result = unpack_nr_crc_indication(&pReadPackedMessage, end, msg, config);
      }
      break;

    case NFAPI_NR_PHY_MSG_TYPE_UCI_INDICATION:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_UCI_INDICATION, unpackedBufLen)) {
        nfapi_nr_uci_indication_t *msg = (nfapi_nr_uci_indication_t *)pMessageHeader;
        msg->uci_list = (nfapi_nr_uci_t *)malloc(sizeof(nfapi_nr_uci_t));
        result = unpack_nr_uci_indication(&pReadPackedMessage, end, msg, config);
      }
      break;

    case NFAPI_NR_PHY_MSG_TYPE_SRS_INDICATION:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_SRS_INDICATION, unpackedBufLen)) {
        nfapi_nr_srs_indication_t *msg = (nfapi_nr_srs_indication_t *)pMessageHeader;
        msg->pdu_list = (nfapi_nr_srs_indication_pdu_t *)malloc(sizeof(nfapi_nr_srs_indication_pdu_t));
        result = unpack_nr_srs_indication(&pReadPackedMessage, end, msg, config);
      }
      break;

    case NFAPI_NR_PHY_MSG_TYPE_RACH_INDICATION:
      if (check_nr_fapi_unpack_length(NFAPI_NR_PHY_MSG_TYPE_RACH_INDICATION, unpackedBufLen)) {
        nfapi_nr_rach_indication_t *msg = (nfapi_nr_rach_indication_t *)pMessageHeader;
        result = unpack_nr_rach_indication(&pReadPackedMessage, end, msg, config);
      }
      break;
    default:

      if (pMessageHeader->message_id >= NFAPI_VENDOR_EXT_MSG_MIN && pMessageHeader->message_id <= NFAPI_VENDOR_EXT_MSG_MAX) {
        if (config && config->unpack_p7_vendor_extension) {
          result = (config->unpack_p7_vendor_extension)(pMessageHeader, &pReadPackedMessage, end, config);
        } else {
          NFAPI_TRACE(NFAPI_TRACE_ERROR,
                      "%s VE NFAPI message ID %d. No ve decoder provided\n",
                      __FUNCTION__,
                      pMessageHeader->message_id);
        }
      } else {
        NFAPI_TRACE(NFAPI_TRACE_ERROR, "%s NFAPI Unknown message ID %d\n", __FUNCTION__, pMessageHeader->message_id);
      }
      break;
  }

  if (result == 0) {
    NFAPI_TRACE(NFAPI_TRACE_ERROR, "P7 Pack failed to pack message\n");
    return -1;
  }
  return 0;
}
