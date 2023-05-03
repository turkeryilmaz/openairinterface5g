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

/*! \file nrppa_common.h
 * \brief nrppa common procedures
 * \author Adeel Malik
 * \email adeel.malik@eurecom.fr
 * \date 2023
 * \version 0.1
 */



#ifndef NRPPA_COMMON_H_
#define NRPPA_COMMON_H_


#include "common/utils/LOG/log.h"
#include "oai_asn1.h"


/* Start: ad**l todo add all nrppa ASN genrated header files here */
//#include "NRPPA-PDU.h"
/* END: ad**l todo add all nrppa ASN genrated header files here*/

/* Checking version of ASN1C compiler */
#if (ASN1C_ENVIRONMENT_VERSION < ASN1C_MINIMUM_VERSION)
# error "You are compiling nrppa with the wrong version of ASN1C"
#endif

extern int asn_debug;
extern int asn1_xer_print;

#if defined(ENB_MODE)
# include "common/utils/LOG/log.h"
# include "ngap_gNB_default_values.h"
# define NRPPA_ERROR(x, args...) LOG_E(NRPPA, x, ##args)
# define NRPPA_WARN(x, args...)  LOG_W(NRPPA, x, ##args)
# define NRPPA_TRAF(x, args...)  LOG_I(NRPPA, x, ##args)
# define NRPPA_INFO(x, args...) LOG_I(NRPPA, x, ##args)
# define NRPPA_DEBUG(x, args...) LOG_I(NRPPA, x, ##args)
#else
# define NRPPA_ERROR(x, args...) do { fprintf(stdout, "[NRPPA][E]"x, ##args); } while(0)
# define NRPPA_WARN(x, args...)  do { fprintf(stdout, "[NRPPA][W]"x, ##args); } while(0)
# define NRPPA_TRAF(x, args...)  do { fprintf(stdout, "[NRPPA][T]"x, ##args); } while(0)
# define NRPPA_INFO(x, args...) do { fprintf(stdout, "[NRPPA][I]"x, ##args); } while(0)
# define NRPPA_DEBUG(x, args...) do { fprintf(stdout, "[NRPPA][D]"x, ##args); } while(0)
#endif


#define NRPPA_FIND_PROTOCOLIE_BY_ID(IE_TYPE, ie, container, IE_ID, mandatory)                                                            \
  do {                                                                                                                                  \
    IE_TYPE **ptr;                                                                                                                      \
    ie = NULL;                                                                                                                          \
    for (ptr = container->protocolIEs.list.array; ptr < &container->protocolIEs.list.array[container->protocolIEs.list.count]; ptr++) { \
      if ((*ptr)->id == IE_ID) {                                                                                                        \
        ie = *ptr;                                                                                                                      \
        break;                                                                                                                          \
      }                                                                                                                                 \
    }                                                                                                                                   \
    if (ie == NULL) {                                                                                                                   \
      if (mandatory) {                                                                                                                  \
        AssertFatal(NRPPA, "NRPPA_FIND_PROTOCOLIE_BY_ID ie is NULL (searching for ie: %ld)\n", IE_ID);                                    \
      } else {                                                                                                                          \
        NRPPA_INFO("NRPPA_FIND_PROTOCOLIE_BY_ID ie is NULL (searching for ie: %ld)\n", IE_ID);                                            \
      }                                                                                                                                 \
    }                                                                                                                                   \
  } while (0);                                                                                                                          \
  if (mandatory && !ie)                                                                                                                 \
  return -1

/** \brief Function callback prototype.
 **/
/* ad**l todo
typedef int (*ngap_message_decoded_callback)(
    uint32_t         assoc_id,
    uint32_t         stream,
    NGAP_NGAP_PDU_t *pdu
);*/


/** \brief Handle criticality
 \param criticality Criticality of the IE
 @returns void
 **/

/* ad**l
void ngap_handle_criticality(NGAP_Criticality_t criticality);
*/

#endif /* NRPPA_COMMON_H_ */
