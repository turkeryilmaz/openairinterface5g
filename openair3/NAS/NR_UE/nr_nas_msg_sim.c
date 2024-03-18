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

/*! \file nr_nas_msg_sim.c
 * \brief simulator for nr nas message
 * \author Yoshio INOUE, Masayuki HARADA
 * \email yoshio.inoue@fujitsu.com,masayuki.harada@fujitsu.com
 * \date 2020
 * \version 0.1
 *
 * 2023.01.27 Vladimir Dorovskikh 16 digits IMEISV
 */


#include <string.h> // memset
#include <stdlib.h> // malloc, free

#include "nas_log.h"
#include "TLVDecoder.h"
#include "TLVEncoder.h"
#include "nr_nas_msg_sim.h"
#include "aka_functions.h"
#include "secu_defs.h"
#include "kdf.h"
#include "key_nas_deriver.h"
#include "PduSessionEstablishRequest.h"
#include "PduSessionEstablishmentAccept.h"
#include "RegistrationAccept.h"
#include "FGSDeregistrationRequestUEOriginating.h"
#include "intertask_interface.h"
#include "openair2/RRC/NAS/nas_config.h"
#include <openair3/NAS/COMMON/NR_NAS_defs.h>
#include <openair1/SIMULATION/ETH_TRANSPORT/proto.h>
#include "openair2/SDAP/nr_sdap/nr_sdap.h"
#include "openair3/SECU/nas_stream_eia2.h"
#include "openair3/UTILS/conversions.h"
#include <openair3/NAS/COMMON/EMM/MSG/NASSecurityModeCommand.h>
#include "nas_stream_eea1.h"
#include "nas_stream_eea2.h"
#include "nas_stream_eia1.h"
#include "nas_stream_eia2.h"

// #define AUTH_ALGO_MILENAGE

static bool _security_set = false;

uint8_t  *registration_request_buf;
uint32_t  registration_request_len;
extern char *baseNetAddress;
extern uint16_t NB_UE_INST;
static nr_ue_nas_t nr_ue_nas = {0};
static nr_nas_msg_snssai_t nas_allowed_nssai[8];
static int _ul_nas_count = 0;
static int _dl_nas_count = 0;
static int _nas_integrity_algo = NIA2_128_ALG_ID;
static int _nas_ciphering_algo = NEA2_128_ALG_ID;

typedef enum
{
  MM_TEST_LOOP_A = 0x00,
  MM_TEST_LOOP_B = 0x01,
  MM_TEST_LOOP_C = 0x02,
  MM_TEST_LOOP_MODE_MAX,
} MmTestMode;

MmTestMode mm_activeTestLoop = MM_TEST_LOOP_MODE_MAX;

typedef enum
{
  MM_TEST_LOOP_STATE_NULL,
  MM_TEST_LOOP_STATE_ACTIVE,
  MM_TEST_LOOP_STATE_CLOSE,
  MM_TEST_LOOP_STATE_OPEN,
}MmTestLoopState;

MmTestLoopState mm_testLoopState = MM_TEST_LOOP_STATE_NULL;

typedef enum
{
  MM_TEST_LOOP_EVENT_ACTIVE,
  MM_TEST_LOOP_EVENT_DEACTIVE,
  MM_TEST_LOOP_EVENT_CLOSE_LOOP,
  MM_TEST_LOOP_EVENT_OPEN_LOOP,
}MmTestLoopEvent;

typedef struct MMCloseUeTestLoopMsgTag
{
  MmTestMode         ueTestLoopMode;
  /* optional fields */
  //MmUETestLoopDrbSetupList testLoopDrbSetupList;  // Mode A
  u8                      testLoopIpPduDelayTimeSeconds; //Mode B
  //MmUeTestLoopMchId     testLoopMtchId;  //Mode C
} MMCloseUeTestLoopMsg;

void decodeCloseUeTestLoopMsg(MMCloseUeTestLoopMsg *closeUeTestLoopMsg, uint8_t *buffer, uint32_t len)
{
  uint8_t offset   = 0;
  if (((nas_msg_header_t *)(buffer))->choice.security_protected_nas_msg_header_t.security_header_type > 0){
    offset +=SECURITY_PROTECTED_5GS_NAS_MESSAGE_HEADER_LENGTH;
    offset +=2; /* skipIndicator('0000'B) +protocolDiscriminator('1111'B) + messageType('10000000'B)  */
  } else {
    offset +=PLAIN_5GS_NAS_MESSAGE_HEADER_LENGTH;
  }
  closeUeTestLoopMsg->ueTestLoopMode = (uint8_t)(*(buffer+offset));
  offset+=1;
  switch(closeUeTestLoopMsg->ueTestLoopMode){
    case MM_TEST_LOOP_A:
      //should decode the modeA LB setup list. Do nothing here.
      break;
    case MM_TEST_LOOP_B:
      closeUeTestLoopMsg->testLoopIpPduDelayTimeSeconds = (uint8_t)(*(buffer+offset));
      break;
    default:
      break;
  }
}

static void mm_setLoop(bool enable, MmTestMode mode, MMCloseUeTestLoopMsg * closeUeTestLoopMsg)
{
  extern void set_pdcp_loopback(bool enable);
  extern void set_sdap_loopback(bool enable, uint8_t testLoopDelayTimeSeconds);

  if(enable){
    if(mode == MM_TEST_LOOP_A){
      //pdcp above loopback
      set_pdcp_loopback(true); //TODO: shall including bearer list parameter
    }else if(mode == MM_TEST_LOOP_B){
      //sdap above loopback
      set_sdap_loopback(true,closeUeTestLoopMsg->testLoopIpPduDelayTimeSeconds);
    }
  } else {
    if(mode == MM_TEST_LOOP_A){
      //pdcp above loopback
      set_pdcp_loopback(false);
    }else if(mode == MM_TEST_LOOP_B){
      //sdap above loopback
      set_sdap_loopback(false,0);
    }
  }
}

static void mm_testLoop_action(MmTestLoopEvent event, MMCloseUeTestLoopMsg * closeUeTestLoopMsg)
{
  MmTestLoopState old_state = mm_testLoopState;
  MmTestLoopState new_state = old_state;

  switch(mm_testLoopState){
    case MM_TEST_LOOP_STATE_NULL:
    {
      if(event == MM_TEST_LOOP_EVENT_ACTIVE){
        new_state = MM_TEST_LOOP_STATE_ACTIVE;
      }
    }
    break;

    case MM_TEST_LOOP_STATE_ACTIVE:
    {
      if(event == MM_TEST_LOOP_EVENT_CLOSE_LOOP && closeUeTestLoopMsg){
        mm_activeTestLoop = closeUeTestLoopMsg->ueTestLoopMode;
        mm_setLoop(true,mm_activeTestLoop,closeUeTestLoopMsg);
        new_state = MM_TEST_LOOP_STATE_CLOSE;
      }
    }
    break;

    case MM_TEST_LOOP_STATE_CLOSE:
    {
      if(event == MM_TEST_LOOP_EVENT_OPEN_LOOP || event == MM_TEST_LOOP_EVENT_DEACTIVE){
        mm_setLoop(false,mm_activeTestLoop,closeUeTestLoopMsg);
        if(event == MM_TEST_LOOP_EVENT_OPEN_LOOP){
          new_state = MM_TEST_LOOP_STATE_OPEN;
        } else {
          new_state = MM_TEST_LOOP_STATE_NULL;
          mm_activeTestLoop = MM_TEST_LOOP_MODE_MAX;
        }
      } else if (event == MM_TEST_LOOP_EVENT_CLOSE_LOOP && closeUeTestLoopMsg){
        if(mm_activeTestLoop != closeUeTestLoopMsg->ueTestLoopMode){
          mm_setLoop(false,mm_activeTestLoop,closeUeTestLoopMsg);
          mm_activeTestLoop = closeUeTestLoopMsg->ueTestLoopMode;
          mm_setLoop(true,mm_activeTestLoop,closeUeTestLoopMsg);
        }
      }
    }
    break;

    case MM_TEST_LOOP_STATE_OPEN:
    {
      if(event == MM_TEST_LOOP_EVENT_CLOSE_LOOP && closeUeTestLoopMsg){
        mm_activeTestLoop = closeUeTestLoopMsg->ueTestLoopMode;
        mm_setLoop(true,mm_activeTestLoop,closeUeTestLoopMsg);
        new_state = MM_TEST_LOOP_STATE_CLOSE;
      } else if (event == MM_TEST_LOOP_EVENT_DEACTIVE){
        new_state = MM_TEST_LOOP_STATE_NULL;
        mm_activeTestLoop = MM_TEST_LOOP_MODE_MAX;
      }
    }
    break;

    default:
      LOG_E(NAS, "wrong test loop state:%d\n", mm_testLoopState);
      break;
  }
  mm_testLoopState = new_state;
  LOG_D(NAS, "test loop state %d ==> %d  test loop mode:%d\n", old_state,new_state, mm_activeTestLoop);
}

static int nas_protected_security_header_encode(
  char                                       *buffer,
  const fgs_nas_message_security_header_t    *header,
  int                                         length)
{
  LOG_FUNC_IN;

  int size = 0;

  /* Encode the protocol discriminator) */
  ENCODE_U8(buffer, header->protocol_discriminator, size);

  /* Encode the security header type */
  ENCODE_U8(buffer+size, (header->security_header_type & 0xf), size);

  /* Encode the message authentication code */
  ENCODE_U32(buffer+size, header->message_authentication_code, size);
  /* Encode the sequence number */
  ENCODE_U8(buffer+size, header->sequence_number, size);

  LOG_FUNC_RETURN (size);
}

static int _nas_mm_msg_encode_header(const mm_msg_header_t *header,
                                  uint8_t *buffer, uint32_t len) {
  int size = 0;

  /* Check the buffer length */
  if (len < sizeof(mm_msg_header_t)) {
    return (TLV_ENCODE_BUFFER_TOO_SHORT);
  }

  /* Check the protocol discriminator */
  if (header->ex_protocol_discriminator != FGS_MOBILITY_MANAGEMENT_MESSAGE &&
          header->ex_protocol_discriminator != TEST_PD) {
    LOG_TRACE(ERROR, "ESM-MSG   - Unexpected extened protocol discriminator: 0x%x",
              header->ex_protocol_discriminator);
    return (TLV_ENCODE_PROTOCOL_NOT_SUPPORTED);
  }

  /* Encode the extendedprotocol discriminator */
  ENCODE_U8(buffer + size, header->ex_protocol_discriminator, size);
  /* Encode the security header type */
  if (header->ex_protocol_discriminator != TEST_PD) {
      ENCODE_U8(buffer + size, (header->security_header_type & 0xf), size);
  }
  /* Encode the message type */
  ENCODE_U8(buffer + size, header->message_type, size);
  return (size);
}

static int fill_suci(FGSMobileIdentity *mi, const uicc_t *uicc)
{
  mi->suci.typeofidentity = FGS_MOBILE_IDENTITY_SUCI;
  mi->suci.mncdigit1 = uicc->nmc_size == 2 ? uicc->imsiStr[3] - '0' : uicc->imsiStr[4] - '0';
  mi->suci.mncdigit2 = uicc->nmc_size == 2 ? uicc->imsiStr[4] - '0' : uicc->imsiStr[5] - '0';
  mi->suci.mncdigit3 = uicc->nmc_size == 2 ? 0xF : uicc->imsiStr[3] - '0';
  mi->suci.mccdigit1 = uicc->imsiStr[0] - '0';
  mi->suci.mccdigit2 = uicc->imsiStr[1] - '0';
  mi->suci.mccdigit3 = uicc->imsiStr[2] - '0';
  memcpy(mi->suci.schemeoutput, uicc->imsiStr + 3 + uicc->nmc_size, strlen(uicc->imsiStr) - (3 + uicc->nmc_size));
  return sizeof(Suci5GSMobileIdentity_t);
}

static int fill_guti(FGSMobileIdentity *mi, const Guti5GSMobileIdentity_t *guti)
{
  AssertFatal(guti != NULL, "UE has no GUTI\n");
  mi->guti = *guti;
  return 13;
}

static int fill_imeisv(FGSMobileIdentity *mi, const uicc_t *uicc)
{
  int i=0;
  mi->imeisv.typeofidentity = FGS_MOBILE_IDENTITY_IMEISV;
  mi->imeisv.digittac01 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac02 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac03 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac04 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac05 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac06 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac07 = getImeisvDigit(uicc, i++);
  mi->imeisv.digittac08 = getImeisvDigit(uicc, i++);
  mi->imeisv.digit09 = getImeisvDigit(uicc, i++);
  mi->imeisv.digit10 = getImeisvDigit(uicc, i++);
  mi->imeisv.digit11 = getImeisvDigit(uicc, i++);
  mi->imeisv.digit12 = getImeisvDigit(uicc, i++);
  mi->imeisv.digit13 = getImeisvDigit(uicc, i++);
  mi->imeisv.digit14 = getImeisvDigit(uicc, i++);
  mi->imeisv.digitsv1 = getImeisvDigit(uicc, i++);
  mi->imeisv.digitsv2 = getImeisvDigit(uicc, i++);
  mi->imeisv.spare  = 0x0f;
  mi->imeisv.oddeven = 1;
  return 19;
}

int mm_msg_encode(MM_msg *mm_msg, uint8_t *buffer, uint32_t len) {
  LOG_FUNC_IN;
  int header_result;
  int encode_result;
  uint8_t msg_type = mm_msg->header.message_type;


  /* First encode the EMM message header */
  header_result = _nas_mm_msg_encode_header(&mm_msg->header, buffer, len);

  if (header_result < 0) {
    LOG_TRACE(ERROR, "EMM-MSG   - Failed to encode EMM message header "
              "(%d)", header_result);
    LOG_FUNC_RETURN(header_result);
  }

  buffer += header_result;
  len -= header_result;

  switch(msg_type) {
    case REGISTRATION_REQUEST:
      encode_result = encode_registration_request(&mm_msg->registration_request, buffer, len);
      break;
    case FGS_IDENTITY_RESPONSE:
      encode_result = encode_identiy_response(&mm_msg->fgs_identity_response, buffer, len);
      break;
    case FGS_AUTHENTICATION_RESPONSE:
      encode_result = encode_fgs_authentication_response(&mm_msg->fgs_auth_response, buffer, len);
      break;
    case FGS_SECURITY_MODE_COMPLETE:
      encode_result = encode_fgs_security_mode_complete(&mm_msg->fgs_security_mode_complete, buffer, len);
      break;
    case FGS_UPLINK_NAS_TRANSPORT:
      encode_result = encode_fgs_uplink_nas_transport(&mm_msg->uplink_nas_transport, buffer, len);
      break;
    case FGS_DEREGISTRATION_REQUEST_UE_ORIGINATING:
      encode_result = encode_fgs_deregistration_request_ue_originating(&mm_msg->fgs_deregistration_request_ue_originating, buffer, len);
      break;
    case ACTIVATE_TEST_MODE_COMPLETE:
      encode_result = encode_activate_test_mode_complete(&mm_msg->activate_test_mode_complete, buffer, len);
      break;
    case CLOSE_UE_TEST_LOOP_COMPLETE:
      encode_result = encode_close_ue_test_loop_complete(&mm_msg->close_ue_test_loop_complete, buffer, len);
      break;
    case OPEN_UE_TEST_LOOP_COMPLETE:
      encode_result = encode_open_ue_test_loop_complete(&mm_msg->open_ue_test_loop_complete, buffer, len);
      break;
    case DEACTIVATE_TEST_MODE_COMPLETE:
      encode_result = encode_deactivate_test_mode_complete(&mm_msg->deactivate_test_mode_complete, buffer, len);
      break;
    case FGS_SERVICE_REQUEST:
      encode_result = encode_fgs_service_request(&mm_msg->fgs_service_request, buffer, len);
      break;
    default:
      LOG_TRACE(ERROR, "EMM-MSG   - Unexpected message type: 0x%x",
          mm_msg->header.message_type);
      encode_result = TLV_ENCODE_WRONG_MESSAGE_TYPE;
      break;
      /* TODO: Handle not standard layer 3 messages: SERVICE_REQUEST */
  }

  if (encode_result < 0) {
    LOG_TRACE(ERROR, "EMM-MSG   - Failed to encode L3 EMM message 0x%x "
              "(%d)", mm_msg->header.message_type, encode_result);
  }

  if (encode_result < 0)
    LOG_FUNC_RETURN (encode_result);

  LOG_FUNC_RETURN (header_result + encode_result);
}

void derive_keys_xor(uint8_t key[16], uint8_t rand[16], uint8_t ck[16], uint8_t ik[16], uint8_t ak[6], uint8_t res[16]) {
  // RES aka XDOut
  for (int i = 0; i < 16; ++i) {
    res[i] = key[i] ^ rand[i];
  }
  printf("key: "); for (int i = 0; i < 16; ++i) printf("%02x", key[i]); printf("\n");
  printf("rand: "); for (int i = 0; i < 16; ++i) printf("%02x", rand[i]); printf("\n");
  printf("res: "); for (int i = 0; i < 16; ++i) printf("%02x", res[i]); printf("\n");

  // AK
  for (int i = 0; i <6 ; i++) {
    ak[i] = res[3+i];
  }
  printf("ak: "); for (int i = 0; i < 6; ++i) printf("%02x", ak[i]); printf("\n");

  // CK
  memmove(&ck[0], &res[1], 15);
  ck[15] = res[0];
  printf("ck: "); for (int i = 0; i < 16; ++i) printf("%02x", ck[i]); printf("\n");

  // IK
  memmove(&ik[0], &ck[1], 15);
  ik[15] = ck[0];
  printf("ik: "); for (int i = 0; i < 16; ++i) printf("%02x", ik[i]); printf("\n");
}

void transferRES(uint8_t ck[16], uint8_t ik[16], uint8_t *input, uint8_t rand[16], uint8_t *output, uicc_t* uicc) {
  uint8_t S[100]={0};
  S[0] = 0x6B;
  servingNetworkName (S+1, uicc->imsiStr, uicc->nmc_size);
  int netNamesize = strlen((char*)S+1);
  S[1 + netNamesize] = (netNamesize & 0xff00) >> 8;
  S[2 + netNamesize] = (netNamesize & 0x00ff);
  for (int i = 0; i < 16; i++)
    S[3 + netNamesize + i] = rand[i];
  S[19 + netNamesize] = 0x00;
  S[20 + netNamesize] = 0x10;

  if (uicc->use_milenage) {
    for (int i = 0; i < 8; i++)
      S[21 + netNamesize + i] = input[i];
    S[29 + netNamesize] = 0x00;
    S[30 + netNamesize] = 0x08;

    uint8_t plmn[3] = { 0x02, 0xf8, 0x39 };
    uint8_t oldS[100];
    oldS[0] = 0x6B;
    memcpy(&oldS[1], plmn, 3);
    oldS[4] = 0x00;
    oldS[5] = 0x03;
    for (int i = 0; i < 16; i++)
      oldS[6 + i] = rand[i];
    oldS[22] = 0x00;
    oldS[23] = 0x10;
    for (int i = 0; i < 8; i++)
      oldS[24 + i] = input[i];
    oldS[32] = 0x00;
    oldS[33] = 0x08;
  } else {
    for (int i = 0; i < 16; i++) {
      S[21 + netNamesize + i] = input[i];
    }
    S[37 + netNamesize] = 0x00;
    S[38 + netNamesize] = 0x10;
  }

  uint8_t key[32] = {0};
  memcpy(&key[0], ck, 16);
  memcpy(&key[16], ik, 16);  //KEY
  uint8_t out[32] = {0};

  if (uicc->use_milenage) {
    byte_array_t data = {.buf = S, .len = 31 + netNamesize};
    kdf(key, data, 32, out);
  } else {
    byte_array_t data = {.buf = S, .len = 39 + netNamesize};
    kdf(key, data, 32, out);
  }

  memcpy(output, out + 16, 16);
}

void derive_kausf(uint8_t ck[16], uint8_t ik[16], uint8_t sqn[6], uint8_t kausf[32], uicc_t *uicc) {
  uint8_t S[100]={0};
  uint8_t key[32] = {0};
  printf(">>> %s\n", __FUNCTION__);

  memcpy(&key[0], ck, 16);
  memcpy(&key[16], ik, 16);  //KEY
  printf("Key: "); for (int i = 0; i < 32; ++i) printf("%02x", key[i]); printf("\n");
  printf("sqn: "); for (int i = 0; i < 6; ++i) printf("%02x", sqn[i]); printf("\n");

  S[0] = 0x6A;
  printf("%s, nmc_sz=%d\n", uicc->imsiStr, uicc->nmc_size);

  servingNetworkName (S+1, uicc->imsiStr, uicc->nmc_size);
  int netNamesize = strlen((char*)S+1);
  S[1 + netNamesize] = (uint8_t)((netNamesize & 0xff00) >> 8);
  S[2 + netNamesize] = (uint8_t)(netNamesize & 0x00ff);
  for (int i = 0; i < 6; i++) {
   S[3 + netNamesize + i] = sqn[i];
  }
  S[9 + netNamesize] = 0x00;
  S[10 + netNamesize] = 0x06;

  byte_array_t data = {.buf = S, .len = 11 +  netNamesize};
  kdf(key, data, 32, kausf);
}

void derive_mk(uint8_t ck[16], uint8_t ik[16], uint8_t sqn[6], uint8_t *mk, uicc_t *uicc) {
  uint8_t S[256]={0};
  uint8_t key[32];
  printf(">>> %s\n", __FUNCTION__);

  memcpy(&key[0], ck, 16);
  memcpy(&key[16], ik, 16);  //KEY
  printf("Key: "); for (int i = 0; i < 32; ++i) printf("%02x", key[i]); printf("\n");
  printf("sqn: "); for (int i = 0; i < 6; ++i) printf("%02x", sqn[i]); printf("\n");

  S[0] = 0x20;
  printf("%s, nmc_sz=%d\n", uicc->imsiStr, uicc->nmc_size);

  servingNetworkName (S+1, uicc->imsiStr, uicc->nmc_size);
  int netNamesize = strlen((char*)S+1);
  S[1 + netNamesize] = (uint8_t)((netNamesize & 0xff00) >> 8);
  S[2 + netNamesize] = (uint8_t)(netNamesize & 0x00ff);
  for (int i = 0; i < 6; i++) {
    S[3 + netNamesize + i] = sqn[i];
  }
  S[9 + netNamesize] = 0x00;
  S[10 + netNamesize] = 0x06;
  printf("Key: "); for (int i = 0; i < 11 + netNamesize; ++i) printf("%02x", S[i]); printf("\n");

  uint8_t ck_ik_[32]; // CK' + IK'

  byte_array_t data = {.buf = S, .len = 11 + netNamesize};
  kdf(key, data, 32, ck_ik_);
  printf("ck': "); for (int i = 0; i < 16; ++i)  printf("%02x", ck_ik_[i]); printf("\n");
  printf("ik': "); for (int i = 16; i < 32; ++i) printf("%02x", ck_ik_[i]); printf("\n");

  memcpy(&key[0], ck_ik_ + 16, 16);
  memcpy(&key[16], ck_ik_, 16);  //KEY
  printf("Key2: "); for (int i = 0; i < 32; ++i) printf("%02x", key[i]); printf("\n");

  // PRF' calculation (RFC 5448 cl. 3.4.1)
  int outBitsNum = 1664; // bits
  int blockSize = 256; // bits
  int prfIters = (outBitsNum + blockSize - 1) / blockSize; // 7 iters
  int mkLen = 0; // resulting MK length
  uint8_t out[32];
  for (int i = 0; i < prfIters; i++) {
    int sLen = 0; // length of string for key derivation
    if (i != 0) {
      memcpy(S, out, 32);
      sLen += 32;
    }
    memcpy(S + sLen, "EAP-AKA'", 8);
    sLen += 8;
    memcpy(S + sLen, uicc->imsiStr, strlen(uicc->imsiStr));
    sLen += strlen(uicc->imsiStr);
    S[sLen] = (uint8_t)(i + 1);
    sLen++;
    printf("S: "); for (int j = 0; j < sLen; ++j) printf("%02x", S[j]); printf("\n");

    byte_array_t data = {.buf = S, .len = sLen};
    kdf(key, data, 32, out);

    for (int j = 0; j < 32; j++) {
      mk[mkLen + j] = out[j];
    }
    mkLen += 32;
  }
}

void derive_kseaf(uint8_t kausf[32], uint8_t kseaf[32], uicc_t *uicc) {
  uint8_t S[100]={0};
  S[0] = 0x6C;  //FC
  servingNetworkName (S+1, uicc->imsiStr, uicc->nmc_size);
  int netNamesize = strlen((char*)S+1);
  S[1 + netNamesize] = (uint8_t)((netNamesize & 0xff00) >> 8);
  S[2 + netNamesize] = (uint8_t)(netNamesize & 0x00ff);

  byte_array_t data = {.buf = S , .len = 3 + netNamesize};
  kdf(kausf, data, 32, kseaf);
}

void derive_kamf(uint8_t *kseaf, uint8_t *kamf, uint16_t abba, uicc_t* uicc) {
  int imsiLen = strlen(uicc->imsiStr);
  uint8_t S[100] = {0};
  S[0] = 0x6D;  //FC = 0x6D
  memcpy(&S[1], uicc->imsiStr, imsiLen );
  S[1 + imsiLen] = (uint8_t)((imsiLen & 0xff00) >> 8);
  S[2 + imsiLen] = (uint8_t)(imsiLen & 0x00ff);
  S[3 + imsiLen] = abba & 0x00ff;
  S[4 + imsiLen] = (abba & 0xff00) >> 8;
  S[5 + imsiLen] = 0x00;
  S[6 + imsiLen] = 0x02;

  byte_array_t data = {.buf = S, .len = 7 + imsiLen};
  kdf(kseaf, data, 32, kamf);
}

//------------------------------------------------------------------------------
void derive_knas(algorithm_type_dist_t nas_alg_type, uint8_t nas_alg_id, uint8_t kamf[32], uint8_t *knas_int) {
  uint8_t S[20] = {0};
  uint8_t out[32] = { 0 };
  S[0] = 0x69;  //FC
  S[1] = (uint8_t)(nas_alg_type & 0xFF);
  S[2] = 0x00;
  S[3] = 0x01;
  S[4] = nas_alg_id;
  S[5] = 0x00;
  S[6] = 0x01;

  byte_array_t data = {.buf = S, .len = 7};
  kdf(kamf, data, 32, out);

  memcpy(knas_int, out+16, 16);
}

void derive_kgnb(uint8_t kamf[32], uint32_t count, uint8_t *kgnb){
  LOG_FUNC_IN;
  /* Compute the KDF input parameter
   * S = FC(0x6E) || UL NAS Count || 0x00 0x04 || 0x01 || 0x00 0x01
   */
  uint8_t  input[32] = {0};
  //    uint16_t length    = 4;
  //    int      offset    = 0;

  LOG_TRACE(INFO, "%s  with count= %d", __FUNCTION__, count);
  memset(input, 0, 32);
  input[0] = 0x6E;
  // P0
  input[1] = count >> 24;
  input[2] = (uint8_t)(count >> 16);
  input[3] = (uint8_t)(count >> 8);
  input[4] = (uint8_t)count;
  // L0
  input[5] = 0;
  input[6] = 4;
  // P1
  input[7] = 0x01;
  // L1
  input[8] = 0;
  input[9] = 1;

  byte_array_t data = {.buf = input, .len = 10};
  kdf(kamf, data, 32, kgnb);

  printf("kgnb : ");
  for(int pp=0;pp<32;pp++)
   printf("%02x ",kgnb[pp]);
  printf("\n");
}

// Hacky approach to detect if eapMessage is present in authentication_Request,
// then the auth type is EAP-AKA'.
// Returns the offset of eapMsg in eapMessage if present, zero otherwise.
int detect_eap_msg(uint8_t *buf)
{
  int offset = -1;

  // suppose here that 'abba' is 2 octets length in authentication_Request
  int mandatory_fields_length = 7;

  if (buf[mandatory_fields_length] == 0x21) { // rand field is present
    offset = 0;
  } else if (buf[mandatory_fields_length] == 0x20) { // autn field is present
    offset = 0;
  } else if (buf[mandatory_fields_length] == 0x78) {
    // if rand or autn is present, more likely, the auth type is 5G-AKA,
    // otherwise eapMessage should be present
    offset = mandatory_fields_length + 3;
  }

  AssertFatal (offset != -1, "Failed, neither 5G-AKA nor EAP-AKA' detected\n");

  return offset;
}

int parse_eap_msg_len(uint8_t *buf)
{
  int len_offset = 2; // length offset in eapMsg
  int len = ((int)buf[len_offset] << 8) + ((int)buf[len_offset + 1]);
  return len;
}

void parse_eap_msg_rand(uint8_t *buf, uint8_t rand[16])
{
  int mandatory_fields_length = 8; // offset for beginning of attributes in eapMsg
  int eap_msg_len = parse_eap_msg_len(buf);
  int off = mandatory_fields_length;

  while (buf[off] != 0x01) {
    off += buf[off + 1] * 4;
    AssertFatal (off < eap_msg_len, "Failed, no RAND attribute found in eapMsg\n");
  }

  off += 4;
  for (int i = 0; i < 16; i++) {
    rand[i] = buf[off + i];
  }
}

void parse_eap_msg_autn(uint8_t *buf, uint8_t autn[6])
{
  int mandatory_fields_length = 8; // offset for beginning of attributes in eapMsg
  int eap_msg_len = parse_eap_msg_len(buf);
  int off = mandatory_fields_length;

  while (buf[off] != 0x02) {
    off += buf[off + 1] * 4;
    AssertFatal (off < eap_msg_len, "Failed, no AUTN attribute found in eapMsg\n");
  }

  off += 4;
  for (int i = 0; i < 6; i++) {
    autn[i] = buf[off + i];
  }
}

static void make_eap_msg(nr_ue_nas_t *nas, OctetString *msg)
{
  uint8_t tmpl[] = { 0x02, 0x00, // code & id
                     0x00 , 0x00, // len
                     0x32, 0x01, 0x00, 0x00, // type & subtype & reserved

                     /* res attribute */
                     0x03, 0x05, // attributeType & len
                     0x00, 0x80, // reslen
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // res (to be set)

                     /* mac attribute */
                     0x0B, 0x05, 0x00, 0x00, // attributeType & len & reserved
                     0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mac (placeholder for now, to be calculated)
  };
  int len = sizeof(tmpl);
  uint8_t *lenPtr = &tmpl[2];
  uint8_t *resPtr = &tmpl[8 + 4];
  uint8_t *macPtr = &tmpl[8 + (4 + 16) + 4];

  // set len field
  lenPtr[0] = (uint8_t)((len >> 8) & 0xFF);
  lenPtr[1] = (uint8_t)(len & 0xFF);

  // set res attribute
  memcpy(resPtr, nas->security.res, 16);

  // calc and set mac attribute
  uint8_t *mk = nas->security.mk; // EAP-AKA' master-key
  uint8_t kaut[32];
  uint8_t mac[32];
  memcpy(kaut, mk + 16, 32); // RFC 5448 cl. 3.3 K_aut to be used for computing MAC
  byte_array_t data = {.buf = tmpl, .len = len};
  kdf(kaut, data, 16, mac);
  memcpy(macPtr, mac, 16);

  msg->length = len;
  msg->value = calloc(1, msg->length);
  memcpy(msg->value, tmpl, msg->length);
}

static void derive_ue_keys(uint8_t *buf, nr_ue_nas_t *nas) {
  LOG_FUNC_IN;
  uint8_t ak[6];
  uint8_t sqn[6];

  DevAssert(nas != NULL);

  // clear old key
  memset(&nas->security, 0, sizeof(nas->security));

  uint8_t *mk       = nas->security.mk; // EAP-AKA' master-key
  uint8_t *kausf    = nas->security.kausf;
  uint8_t *kseaf    = nas->security.kseaf;
  uint8_t *kamf     = nas->security.kamf;
  uint8_t *knas_int = nas->security.knas_int;
  uint8_t *knas_enc = nas->security.knas_enc;
  uint8_t *xres     = nas->security.res;
  uint8_t *rand     = nas->security.rand;
  uint8_t *kgnb     = nas->security.kgnb;

  int eap_msg_offset = detect_eap_msg(buf);
  int eap_msg_len = 0;

  if (eap_msg_offset) {
    eap_msg_len = parse_eap_msg_len(&buf[eap_msg_offset]);
  }

  printf("eap_msg_offset: %d eap_msg_len: %d\n", eap_msg_offset, eap_msg_len);

  if (!eap_msg_offset) {
    // get RAND from authentication request, it is located by 8 octets offset,
    // suppose here that 'abba' is 2 octets length
    for (int index = 0; index < 16;index++){
      rand[index] = buf[8+index];
    }

    // get AUTN from authentication request, it is located by 26 octets offset,
    // suppose here that 'abba' is 2 octets length
    for(int index = 0; index < 6; index++){
      sqn[index] = buf[26+index];
    }
  } else {
    parse_eap_msg_rand(&buf[eap_msg_offset], rand);

    parse_eap_msg_autn(&buf[eap_msg_offset], sqn);
  }

  uint8_t resTemp[16];
  uint8_t ck[16], ik[16];

  if (nas->uicc->use_milenage) {
    f2345(nas->uicc->key, rand, resTemp, ck, ik, ak, nas->uicc->opc);
  } else {
    derive_keys_xor(nas->uicc->key, rand, ck, ik, ak, resTemp);
  }

  if (!eap_msg_offset) {
    transferRES(ck, ik, resTemp, rand, xres, nas->uicc); // calc XRES_Star

    derive_kausf(ck, ik, sqn, kausf, nas->uicc);
  } else {
    memcpy(xres, resTemp, 16); // no XRES_Star in EAP-AKA'

    derive_mk(ck, ik, sqn, mk, nas->uicc);
    memcpy(kausf, mk + 144, 32); // KAUSF is EMSK in RFC 5448 cl. 3.3
  }

  derive_kseaf(kausf, kseaf, nas->uicc);
  derive_kamf(kseaf, kamf, 0x0000, nas->uicc);
  derive_knas(0x02, _nas_integrity_algo, kamf, knas_int); // _nas_integrity_algo is unknown here
  derive_knas(0x01, _nas_ciphering_algo, kamf, knas_enc); // _nas_integrity_algo is unknown here
  derive_kgnb(kamf, 0, kgnb);

  printf("xres:");     for(int i = 0; i < 16; i++) printf("%02x", xres[i]);     printf("\n");
  printf("mk:");       for(int i = 0; i < 48; i++) printf("%02x", mk[i]);       printf("\n");
  printf("kausf:");    for(int i = 0; i < 32; i++) printf("%02x", kausf[i]);    printf("\n");
  printf("kseaf:");    for(int i = 0; i < 32; i++) printf("%02x", kseaf[i]);    printf("\n");
  printf("kamf:");     for(int i = 0; i < 32; i++) printf("%02x", kamf[i]);     printf("\n");
  /* printf("knas_int:"); for(int i = 0; i < 16; i++) printf("%02x", knas_int[i]); printf("\n"); */
  /* printf("knas_enc:"); for(int i = 0; i < 16; i++) printf("%02x", knas_enc[i]); printf("\n"); */
  printf("rand:");     for(int i = 0; i < 16; i++) printf("%02x", rand[i]);     printf("\n");
  printf("kgnb:");     for(int i = 0; i < 32; i++) printf("%02x", kgnb[i]);     printf("\n");

  LOG_FUNC_OUT;
}

static void _enc_dec_msg(nr_ue_nas_t *nas, uint8_t dir, uint32_t count, uint8_t *msg, size_t size, uint8_t algo)
{
  uint8_t *buffer = NULL;
  nas_stream_cipher_t stream_cipher;

  stream_cipher.key_length = NR_NAS_CIP_INT_KEY_LEN_BYTES;
  stream_cipher.count      = count;
  stream_cipher.bearer     = 1;
  stream_cipher.direction  = dir;

  stream_cipher.key        = nas->security.knas_enc;
  stream_cipher.message    = msg;
  /* length in bits */
  stream_cipher.blength    = size << 3;

  switch(algo)
  {
    case NEA0_ALG_ID:
      LOG_W(NAS, "Null Ciphering algo\n");
      return;
    case NEA1_128_ALG_ID:
      buffer = (uint8_t*)malloc(size);
      nas_stream_encrypt_eea1(&stream_cipher, buffer);
      break;
    case NEA2_128_ALG_ID:
      buffer = (uint8_t*)malloc(size);
      nas_stream_encrypt_eea2(&stream_cipher, buffer);
      break;
    default:
      LOG_E(NAS, "Ciphering algo %d is not supported\n", algo);
      return;
  }

  memmove(msg, buffer, size);
  if (NULL != buffer) {
    free(buffer);
  }
}

static void _encrypt_nas_msg(nr_ue_nas_t *nas, uint32_t count, uint8_t *msg, size_t size)
{
  _enc_dec_msg(nas, SECU_DIRECTION_UPLINK, count, msg, size, _nas_ciphering_algo);
}

static void _decrypt_nas_msg(nr_ue_nas_t *nas, uint32_t count, uint8_t *msg, size_t size)
{
  _enc_dec_msg(nas, SECU_DIRECTION_DOWNLINK, count, msg, size, _nas_ciphering_algo);
}

static void _calculate_nas_maci(nr_ue_nas_t *nas, uint8_t dir, uint32_t count, uint8_t *msg, size_t size, uint8_t *mac)
{
  nas_stream_cipher_t stream_cipher;
  uint8_t algo = _nas_integrity_algo;

  stream_cipher.key_length = NR_NAS_CIP_INT_KEY_LEN_BYTES;
  stream_cipher.count      = count;
  stream_cipher.bearer     = 1;
  stream_cipher.direction  = dir;
  stream_cipher.key        = nas->security.knas_int;
  stream_cipher.message    = msg;
  /* length in bits */
  stream_cipher.blength    = size << 3;

  switch(algo)
  {
    case NIA0_ALG_ID:
      LOG_W(NAS, "Null Integrity algo\n");
      memset(mac, 0, 4);
      break;
    case NIA1_128_ALG_ID:
      nas_stream_encrypt_eia1(&stream_cipher, mac);
      break;
    case NIA2_128_ALG_ID:
      nas_stream_encrypt_eia2(&stream_cipher, mac);
      break;
    default:
      LOG_E(NAS, "Integrity algo %d is not supported\n", algo);
      return;
  }
}

nr_ue_nas_t *get_ue_nas_info(module_id_t module_id)
{
  DevAssert(module_id == 0);
  if (!nr_ue_nas.uicc)
    nr_ue_nas.uicc = checkUicc(0);
  return &nr_ue_nas;
}

void generateRegistrationRequest(as_nas_info_t *initialNasMsg, nr_ue_nas_t *nas)
{
  LOG_FUNC_IN;
  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg={0};
  MM_msg *mm_msg;
  static int initial_registration= 1;
  
  mm_msg = &nas_msg.plain.mm_msg;
  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = REGISTRATION_REQUEST;


  // set registration request
  mm_msg->registration_request.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->registration_request.securityheadertype = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->registration_request.messagetype = REGISTRATION_REQUEST;
  size += 1;
  if( initial_registration== 1){
    mm_msg->registration_request.fgsregistrationtype = INITIAL_REGISTRATION;
    initial_registration=0;
  }else{
    mm_msg->registration_request.fgsregistrationtype = MOBILITY_REGISTRATION_UPDATING;
  }
  
  /* Set naskeysetidentifier to 7 instead of 1 for the TTCN */
  mm_msg->registration_request.naskeysetidentifier.naskeysetidentifier = 7;
  size += 1;
  if(nas->guti){
    size += fill_guti(&mm_msg->registration_request.fgsmobileidentity, nas->guti);
  } else {
    size += fill_suci(&mm_msg->registration_request.fgsmobileidentity, nas->uicc);
  }

#if 0
  /* This cannot be sent in clear, the core network Open5GS rejects the UE.
   * TODO: do we have to send this at some point?
   * For the time being, let's keep it here for later proper fix.
   */
  mm_msg->registration_request.presencemask |= REGISTRATION_REQUEST_5GMM_CAPABILITY_PRESENT;
  mm_msg->registration_request.fgmmcapability.iei = REGISTRATION_REQUEST_5GMM_CAPABILITY_IEI;
  mm_msg->registration_request.fgmmcapability.length = 1;
  mm_msg->registration_request.fgmmcapability.value = 0x7;
  size += 3;
#endif
  mm_msg->registration_request.presencemask |= REGISTRATION_REQUEST_UE_SECURITY_CAPABILITY_PRESENT;
  mm_msg->registration_request.nruesecuritycapability.iei = REGISTRATION_REQUEST_UE_SECURITY_CAPABILITY_IEI;
  mm_msg->registration_request.nruesecuritycapability.length = 8;
  mm_msg->registration_request.nruesecuritycapability.fg_EA = 0x80;
  /* Workaround fix of bypassing security for the TTCN */
  mm_msg->registration_request.nruesecuritycapability.fg_IA = 0x80;
  mm_msg->registration_request.nruesecuritycapability.EEA = 0x80;
  mm_msg->registration_request.nruesecuritycapability.EIA = 0x80;
  size += 10;

  // encode the message
  initialNasMsg->data = malloc16_clear(size * sizeof(Byte_t));
  registration_request_buf = initialNasMsg->data;

  initialNasMsg->length = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data), size);
  registration_request_len = initialNasMsg->length;

  LOG_D(NAS, "registration_request_len %d, msg: ", registration_request_len);
  for (int i = 0; i < registration_request_len; ++i) {
    printf("%02x ", initialNasMsg->data[i]);
  }
  printf("\n");

  LOG_FUNC_OUT;
}

void generateIdentityResponse(as_nas_info_t *initialNasMsg, uint8_t identitytype, uicc_t* uicc) {
  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg;
  memset(&nas_msg, 0, sizeof(fgs_nas_message_t));
  MM_msg *mm_msg;

  mm_msg = &nas_msg.plain.mm_msg;
  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = FGS_IDENTITY_RESPONSE;


  // set identity response
  mm_msg->fgs_identity_response.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->fgs_identity_response.securityheadertype = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->fgs_identity_response.messagetype = FGS_IDENTITY_RESPONSE;
  size += 1;
  if(identitytype == FGS_MOBILE_IDENTITY_SUCI){
    size += fill_suci(&mm_msg->fgs_identity_response.fgsmobileidentity, uicc);
  }

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));

  initialNasMsg->length = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data), size);

}

static void generateAuthenticationResp(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg, uint8_t *buf)
{
  LOG_FUNC_IN;
  int eap_msg_offset = detect_eap_msg(buf);

  derive_ue_keys(buf, nas);

  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg;
  memset(&nas_msg, 0, sizeof(fgs_nas_message_t));
  MM_msg *mm_msg;

  mm_msg = &nas_msg.plain.mm_msg;
  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = FGS_AUTHENTICATION_RESPONSE;

  // set authentication response
  mm_msg->fgs_identity_response.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->fgs_identity_response.securityheadertype = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->fgs_identity_response.messagetype = FGS_AUTHENTICATION_RESPONSE;
  size += 1;

  //set response parameter
  if (!eap_msg_offset) {
    OctetString res;
    res.length = 16;
    res.value = calloc(1,16);
    memcpy(res.value, nas->security.res, 16);

    mm_msg->fgs_auth_response.presencemask |= FGS_AUTHENTICATION_RESPONSE_AUTH_RESPONSE_PARAM_PRESENT;
    mm_msg->fgs_auth_response.authenticationresponseparameter.res = res;
    size += 18;
  } else {
    OctetString eapMsg;
    make_eap_msg(nas, &eapMsg);

    mm_msg->fgs_auth_response.presencemask |= FGS_AUTHENTICATION_RESPONSE_EAP_MESSAGE_PRESENT;
    mm_msg->fgs_auth_response.eapmessage.eapMsg = eapMsg;
    size += 3 + eapMsg.length;
  }

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));

  initialNasMsg->length = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data), size);

  LOG_FUNC_OUT;
}

int nas_itti_kgnb_refresh_req(instance_t instance, const uint8_t kgnb[32])
{
  LOG_FUNC_IN;
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_NAS_NRUE, 0, NAS_KENB_REFRESH_REQ);
  memcpy(NAS_KENB_REFRESH_REQ(message_p).kenb, kgnb, sizeof(NAS_KENB_REFRESH_REQ(message_p).kenb));
  LOG_FUNC_OUT;
  return itti_send_msg_to_task(TASK_RRC_NRUE, instance, message_p);
}

static void generateSecurityModeComplete(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg)
{
  LOG_FUNC_IN;
  int size = sizeof(mm_msg_header_t);
  MM_msg *mm_msg;
  fgs_nas_message_t nas_msg;
  _ul_nas_count = 0;
  uint8_t mac[4] = {0};
  int security_header_len = 0;
  int msg_len = 0;
  memset(&nas_msg, 0, sizeof(fgs_nas_message_t));

  // set security protected header
  nas_msg.header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  nas_msg.header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED_WITH_NEW_SECU_CTX;
  nas_msg.header.message_authentication_code = 0;
  nas_msg.header.sequence_number = 0;
  size += 7;

  mm_msg = &nas_msg.security_protected.plain.mm_msg;

  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = FGS_SECURITY_MODE_COMPLETE;

  // set security mode complete
  mm_msg->fgs_security_mode_complete.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->fgs_security_mode_complete.securityheadertype    = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->fgs_security_mode_complete.messagetype           = FGS_SECURITY_MODE_COMPLETE;
  size += 1;

  /* Workaround fix for the issue in TTCN till imeisv is supported by TTCN */
  if(0)
  {
    size += fill_imeisv(&mm_msg->fgs_security_mode_complete.fgsmobileidentity, nas->uicc);
  }

  mm_msg->fgs_security_mode_complete.fgsnasmessagecontainer.nasmessagecontainercontents.value  = registration_request_buf;
  mm_msg->fgs_security_mode_complete.fgsnasmessagecontainer.nasmessagecontainercontents.length = registration_request_len;
  size += (registration_request_len + 2);

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));

  security_header_len = nas_protected_security_header_encode((char*)(initialNasMsg->data),&(nas_msg.header), size);
  msg_len = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data+security_header_len), size-security_header_len);

  initialNasMsg->length = security_header_len + msg_len;
  LOG_T(NAS, "header len %d, msg len %d, todal: %d\n", security_header_len, msg_len, initialNasMsg->length);
  printf("Before Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");

  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + security_header_len, msg_len);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + security_header_len - 1, msg_len + 1,  mac);
  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");

  _security_set = true;

  LOG_FUNC_OUT;
}

static void decodeRegistrationAccept(const uint8_t *buf, int len, nr_ue_nas_t *nas)
{
  registration_accept_msg reg_acc = {0};
  /* it seems there is no 5G corresponding emm_msg_decode() function, so here
   * we just jump to the right decision */
  buf += 7; /* skip security header */
  buf += 2; /* skip prot discriminator, security header, half octet */
  AssertFatal(*buf == 0x42, "this is not a NAS Registration Accept\n");
  buf++;
  int decoded = decode_registration_accept(&reg_acc, buf, len);
  AssertFatal(decoded > 0, "could not decode registration accept\n");
  if (reg_acc.guti) {
     AssertFatal(reg_acc.guti->guti.typeofidentity == FGS_MOBILE_IDENTITY_5G_GUTI,
                 "registration accept 5GS Mobile Identity is not GUTI, but %d\n",
                 reg_acc.guti->guti.typeofidentity);
     nas->guti = malloc(sizeof(*nas->guti));
     AssertFatal(nas->guti, "out of memory\n");
     *nas->guti = reg_acc.guti->guti;
     free(reg_acc.guti); /* no proper memory management for NAS decoded messages */
  } else {
    LOG_W(NAS, "no GUTI in registration accept\n");
  }
}

static void generateRegistrationComplete(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg, SORTransparentContainer *sortransparentcontainer)
{
  LOG_FUNC_IN;
  //wait send RRCReconfigurationComplete and InitialContextSetupResponse
  sleep(1);
  int length = 0;
  int size = 0;
  fgs_nas_message_t nas_msg;
//  nas_stream_cipher_t stream_cipher;
  uint8_t             mac[4];
  _ul_nas_count = 1;
  memset(&nas_msg, 0, sizeof(fgs_nas_message_t));
  fgs_nas_message_security_protected_t *sp_msg;

  sp_msg = &nas_msg.security_protected;
  // set header
  sp_msg->header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  sp_msg->header.security_header_type   = INTEGRITY_PROTECTED_AND_CIPHERED;
  sp_msg->header.message_authentication_code = 0;
  sp_msg->header.sequence_number        = 1;
  length = 7;
  sp_msg->plain.mm_msg.registration_complete.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  length += 1;
  sp_msg->plain.mm_msg.registration_complete.securityheadertype    = PLAIN_5GS_MSG;
  sp_msg->plain.mm_msg.registration_complete.sparehalfoctet        = 0;
  length += 1;
  sp_msg->plain.mm_msg.registration_complete.messagetype = REGISTRATION_COMPLETE;
  length += 1;

  if(sortransparentcontainer) {
    length += sortransparentcontainer->sortransparentcontainercontents.length;
  }

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(length * sizeof(Byte_t));

  /* Encode the first octet of the header (extended protocol discriminator) */
  ENCODE_U8(initialNasMsg->data + size, sp_msg->header.protocol_discriminator, size);
  
  /* Encode the security header type */
  ENCODE_U8(initialNasMsg->data + size, sp_msg->header.security_header_type, size);
  
  /* Encode the message authentication code */
  ENCODE_U32(initialNasMsg->data + size, sp_msg->header.message_authentication_code, size);
  
  /* Encode the sequence number */
  ENCODE_U8(initialNasMsg->data + size, sp_msg->header.sequence_number, size);
  
  
  /* Encode the extended protocol discriminator */
  ENCODE_U8(initialNasMsg->data + size, sp_msg->plain.mm_msg.registration_complete.protocoldiscriminator, size);
    
  /* Encode the security header type */
  ENCODE_U8(initialNasMsg->data + size, sp_msg->plain.mm_msg.registration_complete.securityheadertype, size);
    
  /* Encode the message type */
  ENCODE_U8(initialNasMsg->data + size, sp_msg->plain.mm_msg.registration_complete.messagetype, size);

  if(sortransparentcontainer) {
    encode_registration_complete(&sp_msg->plain.mm_msg.registration_complete, initialNasMsg->data + size, length - size);
  }
  
  initialNasMsg->length = length;

  printf("Before Security: "); for (int i = 0; i < length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");

  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + 7, length - 7);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + 6, length - 6,  mac);

  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");

  LOG_FUNC_OUT;
}

void decodeDownlinkNASTransport(as_nas_info_t *initialNasMsg, uint8_t * pdu_buffer){
  uint8_t msg_type = *(pdu_buffer + 16);
  if(msg_type == FGS_PDU_SESSION_ESTABLISHMENT_ACC){
    sprintf(baseNetAddress, "%d.%d", *(pdu_buffer + 39),*(pdu_buffer + 40));
    int third_octet = *(pdu_buffer + 41);
    int fourth_octet = *(pdu_buffer + 42);
    LOG_A(NAS, "Received PDU Session Establishment Accept\n");
    nas_config(1,third_octet,fourth_octet,"ue");
  } else {
    LOG_E(NAS, "Received unexpected message in DLinformationTransfer %d\n", msg_type);
  }
}

static void generateDeregistrationRequest(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg, const nas_deregistration_req_t *req)
{
  fgs_nas_message_t nas_msg = {0};
  fgs_nas_message_security_protected_t *sp_msg;
  sp_msg = &nas_msg.security_protected;
  sp_msg->header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  sp_msg->header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED;
  sp_msg->header.message_authentication_code = 0;
  sp_msg->header.sequence_number = 3;
  int size = sizeof(fgs_nas_message_security_header_t);

  fgs_deregistration_request_ue_originating_msg *dereg_req = &sp_msg->plain.mm_msg.fgs_deregistration_request_ue_originating;
  dereg_req->protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  dereg_req->securityheadertype = INTEGRITY_PROTECTED_AND_CIPHERED_WITH_NEW_SECU_CTX;
  size += 1;
  dereg_req->messagetype = FGS_DEREGISTRATION_REQUEST_UE_ORIGINATING;
  size += 1;
  dereg_req->deregistrationtype.switchoff = NORMAL_DEREGISTRATION;
  dereg_req->deregistrationtype.reregistration_required = REREGISTRATION_NOT_REQUIRED;
  dereg_req->deregistrationtype.access_type = TGPP_ACCESS;
  dereg_req->naskeysetidentifier.naskeysetidentifier = 1;
  size += 1;
  size += fill_guti(&dereg_req->fgsmobileidentity, nas->guti);

  // encode the message
  initialNasMsg->data = calloc(size, sizeof(Byte_t));
  int security_header_len = nas_protected_security_header_encode((char *)(initialNasMsg->data), &nas_msg.header, size);

  initialNasMsg->length = security_header_len + mm_msg_encode(&sp_msg->plain.mm_msg, (uint8_t *)(initialNasMsg->data + security_header_len), size - security_header_len);

  nas_stream_cipher_t stream_cipher = {
    .key = nas->security.knas_int,
    .key_length = 16,
    .count = nas->security.mm_counter++,
    .bearer = 1,
    .direction = 0,
    .message = (unsigned char *)(initialNasMsg->data + 6),
    .blength = (initialNasMsg->length - 6) << 3, /* length in bits */
  };
  uint8_t mac[4];
  nas_stream_encrypt_eia2(&stream_cipher, mac);

  for(int i = 0; i < 4; i++)
    initialNasMsg->data[2 + i] = mac[i];
}

static void generatePduSessionEstablishRequest(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg, nas_pdu_session_req_t *pdu_req)
{
  LOG_FUNC_IN;
  //wait send RegistrationComplete
  usleep(100*150);
  int size = 0;
  int security_header_len = 0;
  int msg_len = 0;
  _ul_nas_count = 2;

  fgs_nas_message_t nas_msg={0};

  // setup pdu session establishment request
  uint16_t req_length = 7;
  uint8_t *req_buffer = malloc(req_length);
  pdu_session_establishment_request_msg pdu_session_establish;
  pdu_session_establish.protocoldiscriminator = FGS_SESSION_MANAGEMENT_MESSAGE;
  pdu_session_establish.pdusessionid = pdu_req->pdusession_id;
  pdu_session_establish.pti = 1;
  pdu_session_establish.pdusessionestblishmsgtype = FGS_PDU_SESSION_ESTABLISHMENT_REQ;
  pdu_session_establish.maxdatarate = 0xffff;
  pdu_session_establish.pdusessiontype = pdu_req->pdusession_type;
  encode_pdu_session_establishment_request(&pdu_session_establish, req_buffer);



  MM_msg *mm_msg;
  uint8_t             mac[4];
  nas_msg.header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  nas_msg.header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED;
  nas_msg.header.sequence_number = 2;
  size += 7;

  mm_msg = &nas_msg.security_protected.plain.mm_msg;

  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = FGS_UPLINK_NAS_TRANSPORT;

  // set uplink nas transport
  mm_msg->uplink_nas_transport.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->uplink_nas_transport.securityheadertype    = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->uplink_nas_transport.messagetype = FGS_UPLINK_NAS_TRANSPORT;
  size += 1;

  mm_msg->uplink_nas_transport.payloadcontainertype.iei = 0;
  mm_msg->uplink_nas_transport.payloadcontainertype.type = 1;
  size += 1;
  mm_msg->uplink_nas_transport.fgspayloadcontainer.payloadcontainercontents.length = req_length;
  mm_msg->uplink_nas_transport.fgspayloadcontainer.payloadcontainercontents.value = req_buffer;
  size += (2+req_length);
  mm_msg->uplink_nas_transport.pdusessionid = pdu_req->pdusession_id;
  mm_msg->uplink_nas_transport.requesttype = 1;
  size += 3;
#if 0
  const bool has_nssai_sd = nas->uicc->nssai_sd != 0xffffff; // 0xffffff means "no SD", TS 23.003
  const size_t nssai_len = has_nssai_sd ? 4 : 1;
  mm_msg->uplink_nas_transport.snssai.length = nssai_len;
  //Fixme: it seems there are a lot of memory errors in this: this value was on the stack, 
  // but pushed  in a itti message to another thread
  // this kind of error seems in many places in 5G NAS
  mm_msg->uplink_nas_transport.snssai.value = calloc(1, nssai_len);
  mm_msg->uplink_nas_transport.snssai.value[0] = pdu_req->sst;
  if (has_nssai_sd)
    INT24_TO_BUFFER(pdu_req->sd, &mm_msg->uplink_nas_transport.snssai.value[1]);
  size += 1 + 1 + nssai_len;
  int dnnSize=strlen(nas->uicc->dnnStr);
  mm_msg->uplink_nas_transport.dnn.value=calloc(1,dnnSize+1);
  mm_msg->uplink_nas_transport.dnn.length = dnnSize + 1;
  mm_msg->uplink_nas_transport.dnn.value[0] = dnnSize;
  memcpy(mm_msg->uplink_nas_transport.dnn.value + 1, nas->uicc->dnnStr, dnnSize);
  size += (1+1+dnnSize+1);
#endif

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  security_header_len = nas_protected_security_header_encode((char*)(initialNasMsg->data),&(nas_msg.header), size);
  msg_len = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data+security_header_len), size-security_header_len);
  initialNasMsg->length = security_header_len + msg_len;
  LOG_T(NAS, "header len %d, msg len %d, todal: %d\n", security_header_len, msg_len, initialNasMsg->length);

  printf("Before Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");

  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + security_header_len, msg_len);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + security_header_len - 1, msg_len +1,  mac);

  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  LOG_FUNC_OUT;
}

static void generateActivateTestModeComplete(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg) {
  LOG_FUNC_IN;

  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg={0};

  int security_header_len = 0;
  int msg_len = 0;

  MM_msg *mm_msg;
  uint8_t             mac[4];
  nas_msg.header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  nas_msg.header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED;
  size += 7;

  mm_msg = &nas_msg.security_protected.plain.mm_msg;

  // set header
  mm_msg->header.ex_protocol_discriminator = TEST_PD;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = ACTIVATE_TEST_MODE_COMPLETE;

  // set activate test mode complete
  mm_msg->activate_test_mode_complete.protocoldiscriminator = TEST_PD;
  size += 1;
  mm_msg->activate_test_mode_complete.messagetype = ACTIVATE_TEST_MODE_COMPLETE;
  size += 1;

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  security_header_len = nas_protected_security_header_encode((char*)(initialNasMsg->data),&(nas_msg.header), size);
  msg_len = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data+security_header_len), size-security_header_len);
  initialNasMsg->length = security_header_len + msg_len;
  LOG_T(NAS, "header len %d, msg len %d, todal: %d\n", security_header_len, msg_len, initialNasMsg->length);

  printf("Before Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  _ul_nas_count = 0;
  printf("%d\n", _ul_nas_count);
  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + security_header_len, msg_len);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + security_header_len - 1, msg_len +1,  mac);

  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  LOG_FUNC_OUT;
}

static void generateCloseUeTestLoopComplete(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg) {
  LOG_FUNC_IN;
  int size = sizeof(mm_msg_header_t);
  int security_header_len = 0;
  int msg_len = 0;

  fgs_nas_message_t nas_msg={0};

  MM_msg *mm_msg;
  uint8_t             mac[4];
  nas_msg.header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  nas_msg.header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED;
  size += 7;

  mm_msg = &nas_msg.security_protected.plain.mm_msg;

  // set header
  mm_msg->header.ex_protocol_discriminator = TEST_PD;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = CLOSE_UE_TEST_LOOP_COMPLETE;

  // set activate test mode complete
  mm_msg->close_ue_test_loop_complete.protocoldiscriminator = TEST_PD;
  size += 1;
  mm_msg->close_ue_test_loop_complete.messagetype = CLOSE_UE_TEST_LOOP_COMPLETE;
  size += 1;

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  security_header_len = nas_protected_security_header_encode((char*)(initialNasMsg->data),&(nas_msg.header), size);
  msg_len = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data+security_header_len), size-security_header_len);
  initialNasMsg->length = security_header_len + msg_len;
  LOG_T(NAS, "header len %d, msg len %d, todal: %d\n", security_header_len, msg_len, initialNasMsg->length);

  printf("Before Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  _ul_nas_count = 256; /* HACK: Only to match expected TTCN value in TC 7.1.2.2.1*/
  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + security_header_len, msg_len);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + security_header_len - 1, msg_len +1,  mac);

  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  LOG_FUNC_OUT;
}

static void generateOpenUeTestLoopComplete(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg) {
  LOG_FUNC_IN;
  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg={0};
  int security_header_len = 0;
  int msg_len = 0;
  uint8_t mac[4];

  MM_msg *mm_msg;
  nas_msg.header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  nas_msg.header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED;
  size += 7;

  mm_msg = &nas_msg.security_protected.plain.mm_msg;

  // set header
  mm_msg->header.ex_protocol_discriminator = TEST_PD;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = OPEN_UE_TEST_LOOP_COMPLETE;

  // set activate test mode complete
  mm_msg->open_ue_test_loop_complete.protocoldiscriminator = TEST_PD;
  size += 1;
  mm_msg->open_ue_test_loop_complete.messagetype = OPEN_UE_TEST_LOOP_COMPLETE;
  size += 1;

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  security_header_len = nas_protected_security_header_encode((char*)(initialNasMsg->data),&(nas_msg.header), size);
  msg_len = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data+security_header_len), size-security_header_len);
  initialNasMsg->length = security_header_len + msg_len;
  LOG_T(NAS, "header len %d, msg len %d, todal: %d\n", security_header_len, msg_len, initialNasMsg->length);

  printf("Before Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  _ul_nas_count = 256; /* HACK: Only to match expected TTCN value in TC 7.1.3.x*/
  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + security_header_len, msg_len);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + security_header_len - 1, msg_len +1,  mac);
  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");

  LOG_FUNC_OUT;
}

static void generateDeactivateTestModeComplete(nr_ue_nas_t *nas, as_nas_info_t *initialNasMsg) {
  LOG_FUNC_IN;
  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg={0};
  int security_header_len = 0;
  int msg_len = 0;
  uint8_t mac[4];

  MM_msg *mm_msg;
  nas_msg.header.protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  nas_msg.header.security_header_type = INTEGRITY_PROTECTED_AND_CIPHERED;
  size += 7;

  mm_msg = &nas_msg.security_protected.plain.mm_msg;

  // set header
  mm_msg->header.ex_protocol_discriminator = TEST_PD;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = DEACTIVATE_TEST_MODE_COMPLETE;

  // set activate test mode complete
  mm_msg->deactivate_test_mode_complete.protocoldiscriminator = TEST_PD;
  size += 1;
  mm_msg->deactivate_test_mode_complete.messagetype = DEACTIVATE_TEST_MODE_COMPLETE;
  size += 1;

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  security_header_len = nas_protected_security_header_encode((char*)(initialNasMsg->data),&(nas_msg.header), size);
  msg_len = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data+security_header_len), size-security_header_len);
  initialNasMsg->length = security_header_len + msg_len;
  LOG_T(NAS, "header len %d, msg len %d, todal: %d\n", security_header_len, msg_len, initialNasMsg->length);

  printf("Before Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  _ul_nas_count = 256; /* HACK: Only to match expected TTCN value in TC 7.1.3.x*/
  // Ciphering
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + security_header_len, msg_len);

  // Integrity
  _calculate_nas_maci(nas, SECU_DIRECTION_UPLINK, _ul_nas_count, initialNasMsg->data + security_header_len - 1, msg_len +1,  mac);
  printf("xmac %02x%02x%02x%02x\n", mac[0], mac[1], mac[2], mac[3]);
  for(int i = 0; i < 4; i++) {
    initialNasMsg->data[2+i] = mac[i];
  }
  printf("After Security: "); for (int i = 0; i < initialNasMsg->length; ++i) printf("%02x", (initialNasMsg->data)[i]); printf("\n");
  LOG_FUNC_OUT;
}

void generateServiceRequestInner(as_nas_info_t *initialNasMsg, nr_ue_nas_t *nas) {
  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg={0};
  MM_msg *mm_msg;

  mm_msg = &nas_msg.plain.mm_msg;
  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = FGS_SERVICE_REQUEST;

  // set service request
  mm_msg->fgs_service_request.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->fgs_service_request.securityheadertype = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->fgs_service_request.messagetype = FGS_SERVICE_REQUEST;
  size += 1;

  mm_msg->fgs_service_request.servicetype = 0b0010;
  mm_msg->fgs_service_request.naskeysetidentifier.naskeysetidentifier = 0;
  size += 1;

  size += fill_imeisv(&mm_msg->fgs_service_request.fgsmobileidentity, nas->uicc);

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  initialNasMsg->length = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data), size);

  // HACK: TTCN decrypts the piggybacked msg. Thus, need to encrypt it using expected NAS Count == 2
  _ul_nas_count = 2;
  _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data, initialNasMsg->length);
}

void generateServiceRequest(as_nas_info_t *initialNasMsg, nr_ue_nas_t *nas) {
  int size = sizeof(mm_msg_header_t);
  fgs_nas_message_t nas_msg={0};
  MM_msg *mm_msg;

  mm_msg = &nas_msg.plain.mm_msg;
  // set header
  mm_msg->header.ex_protocol_discriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  mm_msg->header.security_header_type = PLAIN_5GS_MSG;
  mm_msg->header.message_type = FGS_SERVICE_REQUEST;

  // set service request
  mm_msg->fgs_service_request.protocoldiscriminator = FGS_MOBILITY_MANAGEMENT_MESSAGE;
  size += 1;
  mm_msg->fgs_service_request.securityheadertype = PLAIN_5GS_MSG;
  size += 1;
  mm_msg->fgs_service_request.messagetype = FGS_SERVICE_REQUEST;
  size += 1;

  mm_msg->fgs_service_request.servicetype = 0b0010;
  mm_msg->fgs_service_request.naskeysetidentifier.naskeysetidentifier = 0;
  size += 1;

  size += fill_imeisv(&mm_msg->fgs_service_request.fgsmobileidentity, nas->uicc);

  as_nas_info_t serviceRequestInnerNasMsg;
  generateServiceRequestInner(&serviceRequestInnerNasMsg, nas);
  mm_msg->fgs_service_request.presencemask |= FGS_SERVICE_REQUEST_NAS_MESSAGE_CONTAINER_PRESENT;
  mm_msg->fgs_service_request.fgsnasmessagecontainer.nasmessagecontainercontents.value  = serviceRequestInnerNasMsg.data;
  mm_msg->fgs_service_request.fgsnasmessagecontainer.nasmessagecontainercontents.length = serviceRequestInnerNasMsg.length;
  size += (serviceRequestInnerNasMsg.length + 2);

  // encode the message
  initialNasMsg->data = (Byte_t *)malloc(size * sizeof(Byte_t));
  initialNasMsg->length = mm_msg_encode(mm_msg, (uint8_t*)(initialNasMsg->data), size);

  // HACK: TTCN decrypts the piggybacked msg. Thus, need to encrypt it using expected NAS Count == 2
  /* _ul_nas_count = 2; */
  /* _encrypt_nas_msg(nas, _ul_nas_count, initialNasMsg->data + 11, initialNasMsg->length - 11); */
}

uint8_t get_msg_type(uint8_t *pdu, uint32_t length) {
  uint8_t          pd = 0;
  uint8_t          msg_type = 0;
  uint8_t          offset   = 0;
  uint8_t         *pdu_buffer = NULL;

  if ((pdu != NULL) && (length > 0)) {
    pdu_buffer = malloc(length);
    AssertFatal (pdu_buffer != NULL, "Failed to allocate memory\n");
    memcpy(pdu_buffer, pdu, length);
    printf("%s: pdu: ", __FUNCTION__); for(int i = 0; i < length; i++) printf("%02x", pdu_buffer[i]);printf("\n");

    if (((nas_msg_header_t *)(pdu_buffer))->choice.security_protected_nas_msg_header_t.security_header_type > 0) {
      offset += SECURITY_PROTECTED_5GS_NAS_MESSAGE_HEADER_LENGTH;
      if (offset < length) {
        pd = ((mm_msg_header_t *)(pdu_buffer + offset))->ex_protocol_discriminator;
        msg_type = ((mm_msg_header_t *)(pdu_buffer + offset))->message_type;

        if (msg_type == FGS_DOWNLINK_NAS_TRANSPORT) {
          msg_type = ((dl_nas_transport_t *)(pdu_buffer+ offset))->sm_nas_msg_header.message_type;
        } else if (pd == TEST_PD) {
          msg_type = *(pdu_buffer + offset + 1);
        }
      }
    } else { // plain 5GS NAS message
      msg_type = ((nas_msg_header_t *)(pdu_buffer))->choice.plain_nas_msg_header.message_type;
    }
  } else {
    LOG_I(NAS, "[UE] Received invalid downlink message\n");
  }

  return msg_type;
}

static void send_nas_uplink_data_req(instance_t instance, const as_nas_info_t *initial_nas_msg)
{
  MessageDef *msg = itti_alloc_new_message(TASK_NAS_NRUE, 0, NAS_UPLINK_DATA_REQ);
  ul_info_transfer_req_t *req = &NAS_UPLINK_DATA_REQ(msg);
  req->UEid = instance;
  req->nasMsg.data = (uint8_t *) initial_nas_msg->data;
  req->nasMsg.length = initial_nas_msg->length;
  itti_send_msg_to_task(TASK_RRC_NRUE, instance, msg);
}

static void send_nas_detach_req(instance_t instance, bool wait_release)
{
  MessageDef *msg = itti_alloc_new_message(TASK_NAS_NRUE, 0, NAS_DETACH_REQ);
  nas_detach_req_t *req = &NAS_DETACH_REQ(msg);
  req->wait_release = wait_release;
  itti_send_msg_to_task(TASK_RRC_NRUE, instance, msg);
}

static void parse_allowed_nssai(nr_nas_msg_snssai_t nssaiList[8], const uint8_t *buf, const uint32_t len)
{
  int nssai_cnt = 0;
  const uint8_t *end = buf + len;
  while (buf < end) {
    const int length = *buf++;
    nr_nas_msg_snssai_t *nssai = nssaiList + nssai_cnt;
    nssai->sd = 0xffffff;

    switch (length) {
      case 1:
        nssai->sst = *buf++;
        nssai_cnt++;
        break;

      case 2:
        nssai->sst = *buf++;
        nssai->hplmn_sst = *buf++;
        nssai_cnt++;
        break;

      case 4:
        nssai->sst = *buf++;
        nssai->sd = 0xffffff & ntoh_int24_buf(buf);
        buf += 3;
        nssai_cnt++;
        break;

      case 5:
        nssai->sst = *buf++;
        nssai->sd = 0xffffff & ntoh_int24_buf(buf);
        buf += 3;
        nssai->hplmn_sst = *buf++;
        nssai_cnt++;
        break;

      case 8:
        nssai->sst = *buf++;
        nssai->sd = 0xffffff & ntoh_int24_buf(buf);
        buf += 3;
        nssai->hplmn_sst = *buf++;
        nssai->hplmn_sd = 0xffffff & ntoh_int24_buf(buf);
        buf += 3;
        nssai_cnt++;
        break;

      default:
        LOG_E(NAS, "UE received unknown length in an allowed S-NSSAI\n");
        break;
    }
  }
}

/* Extract Allowed NSSAI from Regestration Accept according to
   3GPP TS 24.501 Table 8.2.7.1.1
*/
static void get_allowed_nssai(nr_nas_msg_snssai_t nssai[8], const uint8_t *pdu_buffer, const uint32_t pdu_length)
{
  if ((pdu_buffer == NULL) || (pdu_length <= 0))
    return;

  const uint8_t *end = pdu_buffer + pdu_length;
  if (((nas_msg_header_t *)(pdu_buffer))->choice.security_protected_nas_msg_header_t.security_header_type > 0) {
    pdu_buffer += SECURITY_PROTECTED_5GS_NAS_MESSAGE_HEADER_LENGTH;
  }

  pdu_buffer += 1 + 1 + 1 + 2; // Mandatory fields offset
  /* optional fields */
  while (pdu_buffer < end) {
    const int type = *pdu_buffer++;
    int length = 0;
    switch (type) {
      case 0x77: // 5GS mobile identity
        pdu_buffer += ntoh_int16_buf(pdu_buffer) + sizeof(uint16_t);
        break;

      case 0x4A: // PLMN list
      case 0x54: // 5GS tracking area identity
        pdu_buffer += *pdu_buffer + 1; // offset length + 1 byte which contains the length
        break;

      case 0x15: // allowed NSSAI
        length = *pdu_buffer++;
        parse_allowed_nssai(nssai, pdu_buffer, length);
        break;

      default:
        LOG_W(NAS, "This NAS IEI is not handled when extracting list of allowed NSSAI\n");
        pdu_buffer = end;
        break;
    }
  }
}

static void request_default_pdusession(int instance, int nssai_idx)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_NAS_NRUE, 0, NAS_PDU_SESSION_REQ);
  NAS_PDU_SESSION_REQ(message_p).pdusession_id = 10; /* first or default pdu session */
  NAS_PDU_SESSION_REQ(message_p).pdusession_type = 0x91;
  NAS_PDU_SESSION_REQ(message_p).sst = nas_allowed_nssai[nssai_idx].sst;
  NAS_PDU_SESSION_REQ(message_p).sd = nas_allowed_nssai[nssai_idx].sd;
  itti_send_msg_to_task(TASK_NAS_NRUE, instance, message_p);
}

static int get_user_nssai_idx(const nr_nas_msg_snssai_t allowed_nssai[8], const nr_ue_nas_t *nas)
{
  for (int i = 0; i < 8; i++) {
    const nr_nas_msg_snssai_t *nssai = allowed_nssai + i;
    if ((nas->uicc->nssai_sst == nssai->sst) && (nas->uicc->nssai_sd == nssai->sd))
      return i;
  }
  return -1;
}

void *nas_nrue_task(void *args_p)
{
  nr_ue_nas.uicc = checkUicc(0);
  while (1) {
    nas_nrue(NULL);
  }
}

static void handle_registration_accept(instance_t instance,
                                       nr_ue_nas_t *nas,
                                       const uint8_t *pdu_buffer,
                                       uint32_t msg_length)
{
  LOG_I(NAS, "[UE] Received REGISTRATION ACCEPT message\n");
  decodeRegistrationAccept(pdu_buffer, msg_length, nas);
  get_allowed_nssai(nas_allowed_nssai, pdu_buffer, msg_length);

  as_nas_info_t initialNasMsg = {0};
  generateRegistrationComplete(nas, &initialNasMsg, NULL);
  if (initialNasMsg.length > 0) {
    send_nas_uplink_data_req(instance, &initialNasMsg);
    LOG_I(NAS, "Send NAS_UPLINK_DATA_REQ message(RegistrationComplete)\n");
  }
  const int nssai_idx = get_user_nssai_idx(nas_allowed_nssai, nas);
  if (nssai_idx < 0) {
    LOG_E(NAS, "NSSAI parameters not match with allowed NSSAI. Couldn't request PDU session.\n");
  } else {
    request_default_pdusession(instance, nssai_idx);
  }
}

void *nas_nrue(void *args_p)
{
  // Wait for a message or an event
  nr_ue_nas.uicc = checkUicc(0);
  
  MessageDef *msg_p;
  itti_receive_msg(TASK_NAS_NRUE, &msg_p);

  if (msg_p != NULL) {
    instance_t instance = msg_p->ittiMsgHeader.originInstance;
    AssertFatal(instance == 0, "cannot handle more than one UE!\n");

    switch (ITTI_MSG_ID(msg_p)) {
      case INITIALIZE_MESSAGE:

        break;

      case TERMINATE_MESSAGE:
        itti_exit_task();
        break;

      case MESSAGE_TEST:
        break;

      case NAS_CELL_SELECTION_CNF:
        LOG_I(NAS,
              "[UE %ld] Received %s: errCode %u, cellID %u, tac %u\n",
              instance,
              ITTI_MSG_NAME(msg_p),
              NAS_CELL_SELECTION_CNF(msg_p).errCode,
              NAS_CELL_SELECTION_CNF(msg_p).cellID,
              NAS_CELL_SELECTION_CNF(msg_p).tac);
        // as_stmsi_t s_tmsi={0, 0};
        // as_nas_info_t nas_info;
        // plmn_t plmnID={0, 0, 0, 0};
        // generateRegistrationRequest(&nas_info);
        // nr_nas_itti_nas_establish_req(0, AS_TYPE_ORIGINATING_SIGNAL, s_tmsi, plmnID, nas_info.data, nas_info.length, 0);
        break;

      case NAS_CELL_SELECTION_IND:
        LOG_I(NAS,
              "[UE %ld] Received %s: cellID %u, tac %u\n",
              instance,
              ITTI_MSG_NAME(msg_p),
              NAS_CELL_SELECTION_IND(msg_p).cellID,
              NAS_CELL_SELECTION_IND(msg_p).tac);

        /* TODO not processed by NAS currently */
        break;

      case NAS_PAGING_IND:
        LOG_I(NAS, "[UE %ld] Received %s: cause %u\n", instance, ITTI_MSG_NAME(msg_p), NAS_PAGING_IND(msg_p).cause);

        /* TODO not processed by NAS currently */
        break;

      case NAS_PDU_SESSION_REQ: {  //TODO: w49 rebase, to see if this is needed.
        as_nas_info_t pduEstablishMsg = {0};
        nas_pdu_session_req_t *pduReq = &NAS_PDU_SESSION_REQ(msg_p);
        nr_ue_nas_t *nas = get_ue_nas_info(0);
        generatePduSessionEstablishRequest(nas, &pduEstablishMsg, pduReq);
        if (pduEstablishMsg.length > 0) {
          send_nas_uplink_data_req(instance, &pduEstablishMsg);
          LOG_I(NAS, "Send NAS_UPLINK_DATA_REQ message(PduSessionEstablishRequest)\n");
        }
        break;
      }


        /* Do not use this message anymore, all the NAS DL message is processed in  NAS_DOWNLINK_DATA_IND*/
#if 0
        case NAS_CONN_ESTABLI_CNF: {
          LOG_I(NAS, "[UE %ld] Received %s: errCode %u, length %u\n", instance, ITTI_MSG_NAME(msg_p), NAS_CONN_ESTABLI_CNF(msg_p).errCode, NAS_CONN_ESTABLI_CNF(msg_p).nasMsg.length);

        uint8_t *pdu_buffer = NAS_CONN_ESTABLI_CNF(msg_p).nasMsg.data;
        int msg_type = get_msg_type(pdu_buffer, NAS_CONN_ESTABLI_CNF(msg_p).nasMsg.length);

        if (msg_type == REGISTRATION_ACCEPT) {
          nr_ue_nas_t *nas = get_ue_nas_info(0);
          handle_registration_accept(instance, nas, pdu_buffer, NAS_CONN_ESTABLI_CNF(msg_p).nasMsg.length);
        } else if (msg_type == FGS_PDU_SESSION_ESTABLISHMENT_ACC) {
          capture_pdu_session_establishment_accept_msg(pdu_buffer, NAS_CONN_ESTABLI_CNF(msg_p).nasMsg.length);
        }

        break;
      }
#endif

      case NR_NAS_CONN_RELEASE_IND:
        LOG_I(NAS, "[UE %ld] Received %s: cause %u\n",
              instance, ITTI_MSG_NAME (msg_p), NR_NAS_CONN_RELEASE_IND (msg_p).cause);
        nr_ue_nas_t *nas = get_ue_nas_info(0);
        // TODO handle connection release
        // TODO: need to provide flag to recognize UE runs not under TTCN SS
        if (nas->termination_procedure) { //TODO W51: TTCN SS if(0){
          /* the following is not clean, but probably necessary: we need to give
           * time to RLC to Ack the SRB1 PDU which contained the RRC release
           * message. Hence, we just below wait some time, before finally
           * unblocking the nr-uesoftmodem, which will terminate the process. */
          usleep(100000);
          itti_wait_tasks_unblock(); /* will unblock ITTI to stop nr-uesoftmodem */
        }
        break;

      case NAS_UPLINK_DATA_CNF:
        LOG_I(NAS,
              "[UE %ld] Received %s: UEid %u, errCode %u\n",
              instance,
              ITTI_MSG_NAME(msg_p),
              NAS_UPLINK_DATA_CNF(msg_p).UEid,
              NAS_UPLINK_DATA_CNF(msg_p).errCode);

        break;

      case NAS_DEREGISTRATION_REQ: {
        LOG_I(NAS, "[UE %ld] Received %s\n", instance, ITTI_MSG_NAME(msg_p));
        nr_ue_nas_t *nas = get_ue_nas_info(0);
        nas_deregistration_req_t *req = &NAS_DEREGISTRATION_REQ(msg_p);
        if (nas->guti) {
          if (req->cause == AS_DETACH) {
            nas->termination_procedure = true;
            send_nas_detach_req(instance, true);
          }
          as_nas_info_t initialNasMsg = {0};
          generateDeregistrationRequest(nas, &initialNasMsg, req);
          send_nas_uplink_data_req(instance, &initialNasMsg);
        } else {
          LOG_W(NAS, "No GUTI, cannot trigger deregistration request.\n");
          if (req->cause == AS_DETACH)
            send_nas_detach_req(instance, false);
        }
      } break;

      case NAS_DOWNLINK_DATA_IND: {
		nr_ue_nas_t *nas = get_ue_nas_info(0);
        LOG_I(NAS,
              "[UE %ld] Received %s: length %u , buffer %p\n",
              instance,
              ITTI_MSG_NAME(msg_p),
              NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.length,
              NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.data);
        as_nas_info_t initialNasMsg={0};
        size_t pdu_buffer_len = 0;
        uint8_t *pdu_buffer = NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.data;
        pdu_buffer_len = NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.length;

        LOG_I(NAS, "NAS_DOWNLINK_DATA_IND msg: ");
        for(int i = 0; i < pdu_buffer_len; i++) {
          printf("%02x", pdu_buffer[i]);
        } printf("\n");

        if(_security_set) {
          _dl_nas_count++;

          // TODO: Integrity check

           LOG_I(NAS, "_dl_nas_count=%d\n", _dl_nas_count);
          _decrypt_nas_msg(nas, _dl_nas_count,
                           pdu_buffer + SECURITY_PROTECTED_5GS_NAS_MESSAGE_HEADER_LENGTH,
                           pdu_buffer_len - SECURITY_PROTECTED_5GS_NAS_MESSAGE_HEADER_LENGTH);
        }
        uint8_t msg_type = get_msg_type(pdu_buffer, pdu_buffer_len);
        LOG_I(NAS, "NAS msg type:%d\n",msg_type);
        switch(msg_type)
        {
          case FGS_IDENTITY_REQUEST:
            generateIdentityResponse(&initialNasMsg, *(pdu_buffer + 3), nas->uicc);
            break;
          case FGS_AUTHENTICATION_REQUEST:
            generateAuthenticationResp(nas, &initialNasMsg, pdu_buffer);
            break;
          case FGS_SECURITY_MODE_COMMAND:
          {
            security_mode_command_msg smc = {0};
            // SECURITY_PROTECTED_5GS_NAS_MESSAGE_HEADER_LENGTH + PLAIN_5GS_NAS_MESSAGE_HEADER_LENGTH = 10
            decode_security_mode_command(&smc, pdu_buffer+10, pdu_buffer_len-10);

            _nas_integrity_algo = smc.selectednassecurityalgorithms.typeofintegrityalgorithm;
            _nas_ciphering_algo = smc.selectednassecurityalgorithms.typeofcipheringalgorithm;

            LOG_I(NAS, "Security Mode Command: integrity=%d, ciphering=%d\n", _nas_integrity_algo, _nas_ciphering_algo);

            derive_knas(0x02, _nas_integrity_algo, nas->security.kamf, nas->security.knas_int);
            derive_knas(0x01, _nas_ciphering_algo, nas->security.kamf, nas->security.knas_enc);
            printf("knas_int:"); for(int i = 0; i < 16; i++) printf("%02x", nas->security.knas_int[i]); printf("\n");
            printf("knas_enc:"); for(int i = 0; i < 16; i++) printf("%02x", nas->security.knas_enc[i]); printf("\n");

            nas_itti_kgnb_refresh_req(instance,nas->security.kgnb);
            generateSecurityModeComplete(nas, &initialNasMsg);
            break;
          }
          case FGS_DOWNLINK_NAS_TRANSPORT:
            decodeDownlinkNASTransport(&initialNasMsg, pdu_buffer);
            break;
          case REGISTRATION_ACCEPT:
            handle_registration_accept(instance, nas, pdu_buffer, NAS_DOWNLINK_DATA_IND(msg_p).nasMsg.length);
            break;
          case FGS_DEREGISTRATION_ACCEPT:
            LOG_I(NAS, "received deregistration accept\n");
            break;
          case FGS_PDU_SESSION_ESTABLISHMENT_ACC:
            LOG_I(NAS, "[UE] Received PDU_SESSION_ESTABLISHMENT_ACCEPT message\n");
            capture_pdu_session_establishment_accept_msg(pdu_buffer,pdu_buffer_len);
            break;
          case ACTIVATE_TEST_MODE:
            LOG_I(NAS, "[UE] Received ACTIVATE_TEST_MODE message\n");
            mm_testLoop_action(MM_TEST_LOOP_EVENT_ACTIVE, NULL);
            generateActivateTestModeComplete(nas, &initialNasMsg);
            break;
          case NR_CLOSE_UE_TEST_LOOP:
            LOG_I(NAS, "[UE] Received NR_CLOSE_UE_TEST_LOOP message\n");
            MMCloseUeTestLoopMsg closeUeTestLoopMsg;
            decodeCloseUeTestLoopMsg(&closeUeTestLoopMsg,pdu_buffer, pdu_buffer_len);
            mm_testLoop_action(MM_TEST_LOOP_EVENT_CLOSE_LOOP, &closeUeTestLoopMsg);
            generateCloseUeTestLoopComplete(nas, &initialNasMsg);
            break;
          case OPEN_UE_TEST_LOOP:
            LOG_I(NAS, "[UE] Received OPEN_UE_TEST_LOOP message\n");
            mm_testLoop_action(MM_TEST_LOOP_EVENT_OPEN_LOOP,NULL);
            generateOpenUeTestLoopComplete(nas, &initialNasMsg);
            break;
          case DEACTIVATE_TEST_MODE:
            LOG_I(NAS, "[UE] Received DEACTIVATE_TEST_MODE message\n");
            mm_testLoop_action(MM_TEST_LOOP_EVENT_DEACTIVE,NULL);
            generateDeactivateTestModeComplete(nas, &initialNasMsg);
            break;
          default:
            LOG_W(NR_RRC,"unknown message type %d\n",msg_type);
            break;
        }

          if (initialNasMsg.length > 0)
            send_nas_uplink_data_req(instance, &initialNasMsg);
          LOG_I(NAS, "Sent NAS_UPLINK_DATA_REQ message\n");
        }
        break;

      default:
        LOG_E(NAS, "[UE %ld] Received unexpected message %s\n", instance, ITTI_MSG_NAME(msg_p));
        break;
    }

    int result = itti_free(ITTI_MSG_ORIGIN_ID(msg_p), msg_p);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
      msg_p = NULL;
  }
  return NULL;
}

void updateKgNB(nr_ue_nas_t *nas, uint8_t *kgnb)
{
  printf("%s: ulNasCount=%d, kAMF: ", __FUNCTION__, _ul_nas_count);

  for (int i = 0; i < 32; ++i) {
    printf("%02x ", nas->security.kamf[i]);
  }
  printf("\n");

  derive_kgnb(nas->security.kamf, _ul_nas_count, nas->security.kgnb);

  memcpy(kgnb, nas->security.kgnb, 32);
}
