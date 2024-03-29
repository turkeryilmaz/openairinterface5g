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

/*! \file FGSServiceRequest.h

\brief service request procedures for gNB
\author
\email:
\date 2022
\version 0.1
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "ExtendedProtocolDiscriminator.h"
#include "SecurityHeaderType.h"
#include "SpareHalfOctet.h"
#include "MessageType.h"
#include "ServiceType.h"
#include "NasKeySetIdentifier.h"
#include "FGSMobileIdentity.h"
#include "FGCNasMessageContainer.h"

#ifndef FGS_SERVICE_REQUEST_H_
#define FGS_SERVICE_REQUEST_H_

# define FGS_SERVICE_REQUEST_NAS_MESSAGE_CONTAINER_PRESENT                   (1<<3)

typedef enum fgs_service_request_iei_tag {
  FGS_SERVICE_REQUEST_NAS_MESSAGE_CONTAINER_IEI                                   = 0x71, /* 0x71 = 113  */
} fgs_service_request_iei;

typedef struct fgs_service_request_msg_tag {
  /* Mandatory fields */
  ExtendedProtocolDiscriminator           protocoldiscriminator;
  SecurityHeaderType                      securityheadertype:4;
  SpareHalfOctet                          sparehalfoctet:4;
  MessageType                             messagetype;
  ServiceType                             servicetype;
  NasKeySetIdentifier                     naskeysetidentifier;
  FGSMobileIdentity                       fgsmobileidentity;

  /* Optional fields */
  uint32_t                                presencemask;
  FGCNasMessageContainer                  fgsnasmessagecontainer;
} fgs_service_request_msg;

int encode_fgs_service_request(fgs_service_request_msg *fgs_service_request, uint8_t *buffer, uint32_t len);

#endif /* ! defined(FGS_SERVICE_REQUEST_H_) */
