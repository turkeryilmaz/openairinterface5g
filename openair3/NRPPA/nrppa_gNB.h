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

/*! \file nrppa_gNB.h
 * \brief NRPPA gNB task
 * \author  Adeel Maik
 * \date 2023
 * \email: adeel.malik@eurecom.fr
 * \version 1.0
 * @ingroup _nrppa
 */


#include <stdio.h>
#include <stdint.h>

#ifndef NRPPA_GNB_H_
#define NRPPA_GNB_H_

//Processing DownLINK UE ASSOCIATED NRPPA TRANSPORT
int nrppa_process_DownlinkUEAssociatedNRPPaTransport(instance_t instance, ngap_DownlinkUEAssociatedNRPPa_t *ngap_DownlinkUEAssociatedNRPPa_p);
//Processing DOWNLINK NON UE ASSOCIATED NRPPA TRANSPORT (9.2.9.4 of TS 38.413 Version 16.0.0.0 Release 16)
int nrppa_process_DownlinkNonUEAssociatedNRPPaTransport(instance_t instance, ngap_DownlinkNonUEAssociatedNRPPa_t *ngap_DownlinkNonUEAssociatedNRPPa_p);

void nrppa_gNB_init(void) ;
void *nrppa_gNB_process_itti_msg(void *notUsed) ;
void *nrppa_gNB_task(void *arg) ;

#endif /* NGAP_GNB_H_ */
