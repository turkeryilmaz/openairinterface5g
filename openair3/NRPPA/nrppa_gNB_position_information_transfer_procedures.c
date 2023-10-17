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

/*! \file nrppa_gNB_position_information_transfer_procedures.c
 * \brief NRPPA gNB tasks related to position information transfer
 * \author Adeel Malik
 * \email adeel.malik@eurecom.fr
 *\date 2023
 * \version 1.0
 * @ingroup _nrppa
 */


#include "intertask_interface.h"

/* todo */
#include "nrppa_common.h"
#include "nrppa_gNB_position_information_transfer_procedures.h"
#include "nrppa_gNB_itti_messaging.h"
/* todo */

/*to access SRS config*/
#include "PHY/impl_defs_nr.h"   // SRS_Resource_t; SRS_ResourceSet_t;
#include "common/ran_context.h"
#include "NR_MAC_gNB/nr_mac_gNB.h"
#include "RRC/NR/nr_rrc_defs.h"
#include "PHY/defs_gNB.h"
//#include "F1AP/f1ap_common.h"
//#include "F1AP/f1ap_id.h"
//extern RAN_CONTEXT_t RC;
/*to access SRS config*/

/* PositioningInformationExchange (Parent) procedure for  PositioningInformationRequest, PositioningInformationResponse, and PositioningInformationFailure*/
//int nrppa_gNB_handle_PositioningInformationExchange(instance_t instance, uint32_t gNB_ue_ngap_id, uint64_t amf_ue_ngap_id,  uint8_t *routingId_buffer, uint32_t routingId_buffer_length, NRPPA_NRPPA_PDU_t *pdu){
int nrppa_gNB_handle_PositioningInformationExchange( nrppa_gnb_ue_info_t *nrppa_msg_info, NRPPA_NRPPA_PDU_t *rx_pdu)
{
    LOG_D(NRPPA, "Processing Received PositioningInformationRequest \n");
    //printf("Test 1 Adeel:  Processing Received PositioningInformationRequest \n");
// Processing Received PositioningInformationRequest
    NRPPA_PositioningInformationRequest_t     *container;
    NRPPA_PositioningInformationRequest_IEs_t *ie;
    uint32_t                         nrppa_transaction_id;
    NRPPA_RequestedSRSTransmissionCharacteristics_t req_SRS_info;

    DevAssert(rx_pdu != NULL);

    container = &rx_pdu->choice.initiatingMessage->value.choice.PositioningInformationRequest; //IE 9.2.3 Message type (M)
    nrppa_transaction_id = rx_pdu->choice.initiatingMessage->nrppatransactionID; // IE 9.2.4 nrppatransactionID (M)

    /* IE 9.2.27 RequestedSRSTransmissionCharacteristics (O)*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_PositioningInformationRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_RequestedSRSTransmissionCharacteristics, true);
    req_SRS_info = ie->value.choice.RequestedSRSTransmissionCharacteristics;
   // printf("[NRPPA] PIE Test Adeel:  RequestedSRSTransmissionCharacteristics \n");
   // xer_fprint(stdout, &asn_DEF_NRPPA_RequestedSRSTransmissionCharacteristics, &req_SRS_info); // test adeel
    /* TODO Decide if gNB able to provide the information response or not */


    //    struct PHY_VARS_gNB_s *gNB = RC.gNB[0];
//    int module_id= gNB->Mod_id;
//    gNB_MAC_INST *nrmac = RC.nrmac[module_id];
//    gNB_MAC_INST *nrmac = RC.nrmac[0];
    // Retrieve SRS configuration information from RAN Context

//          uint32_t gNB_ue_ngap_id = 0;
// protocol_ctxt_t ctxt;
// gNB_ue_ngap_id = NGAP_UE_CONTEXT_RELEASE_COMMAND(msg_p).gNB_ue_ngap_id;
    /*rrc_gNB_ue_context_t *ue_context_p = rrc_gNB_get_ue_context(RC.nrrrc[nrppa_msg_info->instance], nrppa_msg_info->gNB_ue_ngap_id);
    const gNB_RRC_UE_t *UE = &ue_context_p->ue_context;
    //  f1_ue_data_t ue_data = cu_get_f1_ue_data(UE->rrc_ue_id);

      gNB_RRC_INST *rrc = RC.nrrrc[nrppa_msg_info->instance];

      f1ap_positioning_information_req_t req={
       .gNB_CU_ue_id= nrppa_msg_info->gNB_ue_ngap_id, //UE->rrc_ue_id,
       .gNB_DU_ue_id= 0, //UE->rrc_ue_id //ue_data.secondary_ue uncomment after synch with latest develop
       .nrppa_msg_info.nrppa_transaction_id=nrppa_transaction_id,
       .nrppa_msg_info.instance=nrppa_msg_info->instance,
       .nrppa_msg_info.gNB_ue_ngap_id=nrppa_msg_info->gNB_ue_ngap_id,
       .nrppa_msg_info.amf_ue_ngap_id=nrppa_msg_info->amf_ue_ngap_id,
       .nrppa_msg_info.routing_id_buffer=nrppa_msg_info->routing_id_buffer,
       .nrppa_msg_info.routing_id_length=nrppa_msg_info->routing_id_length,
       };
    //    f1ap_positioning_information_resp_t resp;
      rrc->mac_rrc.positioning_information_request(&req);*/

//printf("NRPPa Monolithic mode procesing PositioningInformationRequest gNB_CU_ue_id=%d, gNB_DU_ue_id=%d \n", f1ap_req->gNB_CU_ue_id, f1ap_req->gNB_DU_ue_id);
    MessageDef *msg = itti_alloc_new_message (TASK_RRC_GNB, 0, F1AP_POSITIONING_INFORMATION_REQ);
    f1ap_positioning_information_req_t *f1ap_req = &F1AP_POSITIONING_INFORMATION_REQ(msg);
    f1ap_req->gNB_CU_ue_id = nrppa_msg_info->gNB_ue_ngap_id;
    f1ap_req->gNB_DU_ue_id = 0;
    f1ap_req->nrppa_msg_info.nrppa_transaction_id=nrppa_transaction_id;
    f1ap_req->nrppa_msg_info.instance=nrppa_msg_info->instance;
    f1ap_req->nrppa_msg_info.gNB_ue_ngap_id=nrppa_msg_info->gNB_ue_ngap_id;
    f1ap_req->nrppa_msg_info.amf_ue_ngap_id=nrppa_msg_info->amf_ue_ngap_id;
    f1ap_req->nrppa_msg_info.routing_id_buffer=nrppa_msg_info->routing_id_buffer;
    f1ap_req->nrppa_msg_info.routing_id_length=nrppa_msg_info->routing_id_length;

    LOG_I(NRPPA,"Procesing PositioningInformationRequest gNB_CU_ue_id=%d, gNB_DU_ue_id=%d \n", f1ap_req->gNB_CU_ue_id, f1ap_req->gNB_DU_ue_id);
    itti_send_msg_to_task(TASK_RRC_GNB, 0, msg);





    // printf("[NRPPA]monolithic Mode :Processing Received PositioningInformationRequest \n");
    // Preparing Response for the Received PositioningInformationRequest
    bool response_type_indicator = 1;  // 1 = send Position Information transfer Response, 0 = send Position Information transfer Failure

//   uint8_t *tx_nrppa_pdu = NULL;
    uint32_t nrppa_pdu_length;
//   NRPPA_NRPPA_PDU_t tx_pdu;  // TODO rename
    //  uint8_t  *buffer= NULL;
//    uint32_t  length=0;
    /*if (response_type_indicator)
    {
        printf("[NRPPA] PIR Test Adeel:  Processing Received PositioningInformationRequest \n");
        LOG_D(NRPPA, "Preparing PositioningInformationResponse message \n");
        //nrppa_pdu_length= nrppa_gNB_PositioningInformationResponse(nrppa_transaction_id, tx_nrppa_pdu);
        nrppa_pdu_length= nrppa_gNB_PositioningInformationResponse(nrppa_transaction_id, nrppa_msg_info);
        printf("[NRPPA] Test Adeel:  Response prepared PositioningInformationRequest calling itti with NRPPa PDU length %d \n", nrppa_pdu_length);
        return nrppa_pdu_length;
    }
    else
    {
        printf("[NRPPA] PIF Test Adeel:  Processing Received PositioningInformationRequest \n");
        LOG_D(NRPPA, "Preparing PositioningInformationFailure  message\n");
        //nrppa_pdu_length= nrppa_gNB_PositioningInformationFailure(nrppa_transaction_id, &tx_nrppa_pdu);
        nrppa_pdu_length= nrppa_gNB_PositioningInformationFailure(nrppa_transaction_id, nrppa_msg_info);
        printf("[NRPPA] Test Adeel:  Response prepared PositioningInformationRequest calling itti with NRPPa PDU length %d \n", nrppa_pdu_length);

        return nrppa_pdu_length;

}


//int nrppa_gNB_PositioningInformationResponse( uint32_t nrppa_transaction_id,  nrppa_gnb_ue_info_t *nrppa_msg_info  )
int nrppa_gNB_PositioningInformationResponse(instance_t instance, MessageDef *msg_p)
{

    f1ap_positioning_information_resp_t *resp = &F1AP_POSITIONING_INFORMATION_RESP(msg_p);
    LOG_I(NRPPA, "NRPPA Received PositioningInformationResponse info gNB_CU_ue_id=%d, gNB_DU_ue_id=%d \n", resp->gNB_CU_ue_id, resp->gNB_DU_ue_id);

// Prepare NRPPA Position Information transfer Response
    NRPPA_NRPPA_PDU_t tx_pdu;  // TODO rename
    uint8_t  *buffer= NULL;
    uint32_t  length=0;

    /* Prepare the NRPPA message to encode for successfulOutcome PositioningInformationResponse */

    //IE: 9.2.3 Message Type successfulOutcome PositioningInformationResponse /* mandatory */
    memset(&tx_pdu, 0, sizeof(tx_pdu));
    tx_pdu.present = NRPPA_NRPPA_PDU_PR_successfulOutcome;
    asn1cCalloc(tx_pdu.choice.successfulOutcome, head);
    head->procedureCode = NRPPA_ProcedureCode_id_positioningInformationExchange;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_SuccessfulOutcome__value_PR_PositioningInformationResponse;
    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID = resp->nrppa_msg_info.nrppa_transaction_id;//nrppa_transaction_id;

    NRPPA_PositioningInformationResponse_t *out = &head->value.choice.PositioningInformationResponse;

    //printf("Test 2 Adeel:  PositioningInformationResponse \n");
    //IE 9.2.28 SRS Configuration (O)
    // TODO  set the SRS configuration as requested in PositioningInformationRequest
    //  Currently changing SRS configuration on the fly is not possible, therefore we add the predefined SRS configuration in NRPPA pdu
    if (0)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_SRSConfiguration;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationResponse_IEs__value_PR_SRSConfiguration;

        /* printf("Test 3 Adeel:  PositioningInformationResponse \n");
         //start test debug adeel
         asn1cSequenceAdd(ie->value.choice.SRSConfiguration.sRSCarrier_List.list, NRPPA_SRSCarrier_List_Item_t, item); //test
         item->pointA=1;  // IE of SRSCarrier_List_Item //TODO adeel retrieve and add //test
         item->activeULBWP.locationAndBandwidth= 0;// long //TODO adeel retrieve and add
         item->activeULBWP.subcarrierSpacing= 0;// long //TODO adeel retrieve and add
         item->activeULBWP.cyclicPrefix= 0;// long //TODO adeel retrieve and add
         item->activeULBWP.txDirectCurrentLocation= 0;// long //TODO adeel retrieve and add*/
        //end test debug adeel
        // Retrieve SRS configuration information from RAN Context
        /*    struct PHY_VARS_gNB_s *gNB = RC.gNB[0];
            int module_id= gNB->Mod_id;
            printf("[NRRPA Building PIR] module_id is %d \n", module_id);
            gNB_MAC_INST *nrmac = RC.nrmac[module_id];
            NR_UEs_t *UE_info = &nrmac->UE_info;
            UE_iterator(UE_info->list, UE)
            {
                int UE_SUPI = 1;  // TODO adeel add a condition to select configuration of the target UE of positioning
                if(UE_SUPI==1)
                {
                    printf("[NRRPA  Building  PIR] UE_id is %d \n", &UE->uid); ////uid_t uid = &UE->uid;
                    NR_UE_UL_BWP_t *current_BWP = &UE->current_UL_BWP;
                    NR_SRS_Config_t *srs_config = current_BWP->srs_Config;

                    // IE 9.2.28 SRS Configuration Preparing SRS Configuration IE of PositioningInformationResponse
                    int nb_of_srscarrier= 1;//gNB->max_nb_srs; // TODO find the acutal number for carrier and add here
                    for (int i = 0; i < nb_of_srscarrier; i++)
                    {
                        asn1cSequenceAdd(ie->value.choice.SRSConfiguration.sRSCarrier_List.list, NRPPA_SRSCarrier_List_Item_t, item);
                        item->pointA=1;  // IE of SRSCarrier_List_Item //TODO adeel retrieve and add
                        //item->PCI=1;   // IE of SRSCarrier_List_Item Optional Physical cell ID of the cell that contians the SRS carrier

                        //     Preparing Active UL BWP information IE of SRSCarrier_List
                        item->activeULBWP.locationAndBandwidth= 0;// long //TODO adeel retrieve and add
                        item->activeULBWP.subcarrierSpacing= 0;// long //TODO adeel retrieve and add
                        item->activeULBWP.cyclicPrefix= 0;// long //TODO adeel retrieve and add
                        item->activeULBWP.txDirectCurrentLocation= 0;// long //TODO adeel retrieve and add
                       // item->activeULBWP.shift7dot5kHz= NULL;// long  Optional //TODO adeel retrieve and add

                        //Preparing SRS Config IE of activeULBWP
                       int nb_srsresource =srs_config->srs_ResourceToAddModList->list.count;
                        asn1cCalloc(item->activeULBWP.sRSConfig.sRSResource_List, srsresource_list);
                        for(int k=0; k < nb_srsresource; k++)  //Preparing SRS Resource List
                        {
                            printf("Test 4 Adeel:  PositioningInformationResponse nb_srsresource=%d \n", nb_srsresource);
                            //NRPPA_NRPPA_PDU_t pdu;  // TODO rename
        //uint8_t  *buffer;
        //uint32_t  length;
        //memset(&pdu, 0, sizeof(pdu));
        //pdu.present = NRPPA_NRPPA_PDU_PR_successfulOutcome;
        //asn1cCalloc(pdu.choice.successfulOutcome, head);
                            asn1cSequenceAdd(srsresource_list->list, NRPPA_SRSResource_t, resource_item);
                            //asn1cSequenceAdd(item->activeULBWP.sRSConfig.sRSResource_List->list, NRPPA_SRSResource_t, resource_item);
                            printf("Test 4.1 Adeel:  PositioningInformationResponse nb_srsresource=%d \n", nb_srsresource);
                            //resource_item = srs_config->srs_ResourceToAddModList->list.array[k];
                            resource_item->nrofSymbols=1;
                        } //for(int k=0; k < nb_srsresource; k++)
                        //}


                        int nb_srsresourceset =srs_config->srs_ResourceSetToAddModList->list.count;
                        for(int y=0; y < nb_srsresourceset; y++) //Preparing SRS Resource Set List
                        {
                            printf("Test 5 Adeel:  PositioningInformationResponse nb_srsresourceset=%d \n", nb_srsresourceset);
                            asn1cCalloc(item->activeULBWP.sRSConfig.sRSResourceSet_List, srsresourceset_list);
                            asn1cSequenceAdd(srsresourceset_list->list, NRPPA_SRSResourceSet_t, srsresourceset_item);
                            //asn1cSequenceAdd(item->activeULBWP.sRSConfig.sRSResourceSet_List->list, NRPPA_SRSResourceSet_t, srsresourceset_item);
                            srsresourceset_item->sRSResourceSetID= srs_config->srs_ResourceSetToAddModList->list.array[y]->srs_ResourceSetId;
                        }
                        //srsresource_item-> = initial_ctxt_resp_p->pdusessions[i].associated_qos_flows[j].qfi;
                        //}

        //	item->activeULBWP.sRSConfig.posSRSResource_List;
                        //item->activeULBWP.sRSConfig.posSRSResourceSet_List;
                        //item->activeULBWP.sRSConfig.iE_Extensions;
                        // OPTIONAL TODO struct NRPPA_ProtocolExtensionContainer	*iE_Extensions;
                        //item->activeULBWP.IE_Extensions;

        // ie->value.choice.SRSConfiguration.sRSCarrier_List.list= current_BWP;
                        //asn1cCalloc(ie->value.choice.SRSConfiguration, &srs_config);

                        //   TODO  Preparing Uplink Channel BW Per SCS List information
                        int size_SCS_list =1;  //TODO adeel retrieve and add
                        for(int a=0; a < size_SCS_list; a++)
                        {
                            asn1cSequenceAdd(item->uplinkChannelBW_PerSCS_List, NRPPA_UplinkChannelBW_PerSCS_List_t, SCS_list_item);

                            int size_SpecificCarrier_list =1;  //TODO adeel retrieve and add
                            for(int b=0; b < size_SpecificCarrier_list; b++)
                            {
                                asn1cSequenceAdd(SCS_list_item->list, NRPPA_SCS_SpecificCarrier_t, SpecificCarrier_item);
                                SpecificCarrier_item->offsetToCarrier= 0;  //TODO adeel retrieve and add
                                SpecificCarrier_item->subcarrierSpacing=0; //TODO adeel retrieve and add
                                SpecificCarrier_item->carrierBandwidth=0;  //TODO adeel retrieve and add
                                //SpecificCarrier_item->iE_Extenstions // OPtional TODO
                            } // for(int b=0; b < size_SpecificCarrier_list; b++)
                        } // for(int a=0; a < size_SCS_list; a++)
                    } //for (int i = 0; i < nb_of_srscarrier; i++)
                }  //Condition for the target UE of positioning
            } // UE_iterator*/
    }  //IE 9.2.28 SRS Configuration

//IE 9.2.36 SFN Initialisation Time (O)
    if (0)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_SFNInitialisationTime;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationResponse_IEs__value_PR_SFNInitialisationTime;
        // TODO Retreive SFN Initialisation Time and assign
        //ie->value.choice.SFNInitialisationTime.buf = NULL ; //TODO adeel retrieve and add TYPE typedef struct BIT_STRING_s {uint8_t *buf;	size_t size;	int bits_unused;} BIT_STRING_t;
        //ie->value.choice.SFNInitialisationTime.size =4;
        //ie->value.choice.SFNInitialisationTime.bits_unused =0;
    }

//  TODO IE 9.2.2 CriticalityDiagnostics (O)
    if (1)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_CriticalityDiagnostics;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationResponse_IEs__value_PR_CriticalityDiagnostics;
        // TODO Retreive CriticalityDiagnostics information and assign
        // ie->value.choice.CriticalityDiagnostics.procedureCode = 9; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.triggeringMessage = 1; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.procedureCriticality; = NRPPA_Criticality_reject; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.nrppatransactionID =10;//nrppa_transaction_id ;
        //ie->value.choice.CriticalityDiagnostics.iEsCriticalityDiagnostics = ; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.iE_Extensions = ; //TODO adeel retrieve and add
    }

    LOG_I(NRPPA, "Calling encoder for PositioningInformationResponse \n");
    xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, &tx_pdu); // test adeel

    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&tx_pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa PositioningInformationResponse\n");
        /* Encode procedure has failed... */
        return -1;
    }

//    printf("Test 2 Adeel: dummy nrppa pdu of PositioningInformationResponse length=%d \n", length);
//    xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, &buffer); // test adeel
    //printf("Test 2 Adeel:  PositioningInformationResponse length=%d \n", length);

    /* Forward the NRPPA PDU to NGAP */

    if ( 1)
    {
        //printf("[NRPPA] Sending ITTI UplinkUEAssociatedNRPPa pdu (PositioningInformationResponse/Failure) to NGAP nrppa_pdu_length=%d \n", length);
        //xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, &tx_pdu); // test adeel
        LOG_D(NRPPA, "Sending UplinkUEAssociatedNRPPa (PositioningInformationResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkUEAssociatedNRPPa(resp->nrppa_msg_info.instance,
                resp->nrppa_msg_info.gNB_ue_ngap_id,
                resp->nrppa_msg_info.amf_ue_ngap_id,
                resp->nrppa_msg_info.routing_id_buffer,
                resp->nrppa_msg_info.routing_id_length,
                buffer, length);
        return length;
    }
//    else if (nrppa_msg_info->gNB_ue_ngap_id ==-1 && nrppa_msg_info->amf_ue_ngap_id == -1) //
    else if (resp->nrppa_msg_info.gNB_ue_ngap_id ==-1 && resp->nrppa_msg_info.amf_ue_ngap_id == -1) //
    {
        LOG_D(NRPPA, "Sending UplinkNonUEAssociatedNRPPa (PositioningInformationResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkNonUEAssociatedNRPPa(resp->nrppa_msg_info.instance,
                resp->nrppa_msg_info.routing_id_buffer,
                resp->nrppa_msg_info.routing_id_length,
                buffer, length);
        return length;
    }
    else
    {
        NRPPA_ERROR("Failed to find context for Uplink NonUE/UE Associated NRPPa PositioningInformationRequest\n");
        return -1;
    }



}


int nrppa_gNB_PositioningInformationFailure( uint32_t nrppa_transaction_id, nrppa_gnb_ue_info_t *nrppa_msg_info )
{
// Prepare NRPPA Position Information failure
    NRPPA_NRPPA_PDU_t tx_pdu;  // TODO rename
    uint8_t  *buffer= NULL;
    uint32_t  length=0;

    //IE: 9.2.3 Message Type unsuccessfulOutcome PositioningInformationFaliure /* mandatory */
    //IE 9.2.3 Message type (M)
    memset(&tx_pdu, 0, sizeof(tx_pdu));
    tx_pdu.present = NRPPA_NRPPA_PDU_PR_unsuccessfulOutcome;
    asn1cCalloc(tx_pdu.choice.unsuccessfulOutcome, head);
    head->procedureCode = NRPPA_ProcedureCode_id_positioningInformationExchange;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_UnsuccessfulOutcome__value_PR_PositioningInformationFailure;



    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;
    NRPPA_PositioningInformationFailure_t *out = &head->value.choice.PositioningInformationFailure;
// TODO IE 9.2.1 Cause (M)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationFailure_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_Cause;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationFailure_IEs__value_PR_Cause;
        //TODO Reteive Cause and assign
        ie->value.choice.Cause.present = NRPPA_Cause_PR_misc ; //IE 1
        //ie->value.choice.Cause.present = NRPPA_Cause_PR_NOTHING ; //IE 1
        ie->value.choice.Cause.choice.misc=0; //TODO dummay response
        //ie->value.choice.Cause. =;  // IE 2 and so on
    }


//  TODO IE 9.2.2 CriticalityDiagnostics (O)
    if (1)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationFailure_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_CriticalityDiagnostics;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationFailure_IEs__value_PR_CriticalityDiagnostics;
        // TODO Retreive CriticalityDiagnostics information and assign
        // ie->value.choice.CriticalityDiagnostics.procedureCode = 9; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.triggeringMessage = 1; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.procedureCriticality; = NRPPA_Criticality_reject; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.nrppatransactionID =10;//nrppa_transaction_id ;
        //ie->value.choice.CriticalityDiagnostics.iEsCriticalityDiagnostics = ; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.iE_Extensions = ; //TODO adeel retrieve and add
        // TODO Retreive CriticalityDiagnostics information and assign
        //ie->value.choice.CriticalityDiagnostics. = ;
        //ie->value.choice.CriticalityDiagnostics. = ;
    }

//printf("Test Adeel: nrppa pdu\n");
//xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, &pdu); // test adeel
    printf("Test 2 Adeel: PIF  calling encoder \n");
    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&tx_pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa PositioningInformationFailure \n");
        /* Encode procedure has failed... */
        return -1;
    }
    printf("Test 3 Adeel: PIF  pdu encoded pdu_length=%d \n", length);
// adeel end test

    if ( 1)
    {
        printf("[NRPPA] Test  4.2  Sending ITTI UplinkUEAssociatedNRPPa pdu (PositioningInformationResponse/Failure) to NGAP nrppa_pdu_length=%d \n", length);
        xer_fprint(stdout, &asn_DEF_NRPPA_NRPPA_PDU, &tx_pdu); // test adeel
        LOG_D(NRPPA, "Sending UplinkUEAssociatedNRPPa (PositioningInformationResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkUEAssociatedNRPPa(nrppa_msg_info->instance,
                nrppa_msg_info->gNB_ue_ngap_id,
                nrppa_msg_info->amf_ue_ngap_id,
                nrppa_msg_info->routing_id_buffer,
                nrppa_msg_info->routing_id_length,
                buffer, length); //tx_nrppa_pdu=buffer, nrppa_pdu_length=length
        return length;
    }
    else if (nrppa_msg_info->gNB_ue_ngap_id ==-1 && nrppa_msg_info->amf_ue_ngap_id == -1) //
    {
        LOG_D(NRPPA, "Sending UplinkNonUEAssociatedNRPPa (PositioningInformationResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkNonUEAssociatedNRPPa(nrppa_msg_info->instance,
                nrppa_msg_info->routing_id_buffer,
                nrppa_msg_info->routing_id_length,
                buffer, length);
        return length;
    }
    else
    {
        NRPPA_ERROR("Failed to find context for Uplink NonUE/UE Associated NRPPa PositioningInformationRequest\n");

        return -1;
    }

}


int nrppa_gNB_PositioningInformationUpdate( uint32_t nrppa_transaction_id, uint8_t *buffer ) // TODO adeel define when and where to call this function and setup corresponding ITTI exchange to NGAP
{
    LOG_D(NRPPA, "Preparing PositioningInformationUpdate \n");
    // Prepare NRPPA Position Information Update
    NRPPA_NRPPA_PDU_t pdu;  // TODO rename
    //uint8_t  *buffer;
    uint32_t  length;

    /* Prepare the NRPPA message to encode for initiating message PositioningInformationUpdate */

    //IE: 9.2.3 Message Type initiatingMessage PositioningInformationUpdate /* mandatory */
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_initiatingMessage;
    asn1cCalloc(pdu.choice.initiatingMessage, head);
    head->procedureCode = NRPPA_ProcedureCode_id_positioningInformationUpdate;
    head->criticality = NRPPA_Criticality_ignore;
    head->value.present = NRPPA_InitiatingMessage__value_PR_PositioningInformationUpdate;

    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;

    NRPPA_PositioningInformationUpdate_t *out = &head->value.choice.PositioningInformationUpdate;

    //IE 9.2.28 SRS Configuration (O)
    /*  Currently changing SRS configuration on the fly is not possible, therefore we add the predefined SRS configuration in NRPPA pdu */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationUpdate_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_SRSConfiguration;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationUpdate_IEs__value_PR_SRSConfiguration;

        // Retrieve SRS configuration information from RAN Context
        struct PHY_VARS_gNB_s *gNB = RC.gNB[0];
        int module_id= gNB->Mod_id;
        printf("[NRRPA Building PIR] module_id is %d", module_id);
        gNB_MAC_INST *nrmac = RC.nrmac[module_id];
        NR_UEs_t *UE_info = &nrmac->UE_info;
        UE_iterator(UE_info->list, UE)
        {
            int UE_SUPI = 1;  // TODO adeel add a condition to select configuration of the target UE of positioning
            if(UE_SUPI==1)
            {
                //printf("[NRRPA  Building  PIR] UE_id is %d", &UE->uid); ////uid_t uid = &UE->uid;
                NR_UE_UL_BWP_t *current_BWP = &UE->current_UL_BWP;
                NR_SRS_Config_t *srs_config = current_BWP->srs_Config;

                /* IE 9.2.28 SRS Configuration Preparing SRS Configuration IE of PositioningInformationResponse*/
                int nb_of_srscarrier= gNB->max_nb_srs; // TODO find the acutal number for carrier and add here
                for (int i = 0; i < nb_of_srscarrier; i++)
                {
                    asn1cSequenceAdd(ie->value.choice.SRSConfiguration.sRSCarrier_List.list, NRPPA_SRSCarrier_List_Item_t, item);
                    item->pointA=1;  // IE of SRSCarrier_List_Item //TODO adeel retrieve and add
                    //item->PCI=1;   // IE of SRSCarrier_List_Item Optional Physical cell ID of the cell that contians the SRS carrier

                    //     Preparing Active UL BWP information IE of SRSCarrier_List
                    item->activeULBWP.locationAndBandwidth= 0;// long //TODO adeel retrieve and add
                    item->activeULBWP.subcarrierSpacing= 0;// long //TODO adeel retrieve and add
                    item->activeULBWP.cyclicPrefix= 0;// long //TODO adeel retrieve and add
                    item->activeULBWP.txDirectCurrentLocation= 0;// long //TODO adeel retrieve and add
                    item->activeULBWP.shift7dot5kHz= NULL;// long  Optional //TODO adeel retrieve and add

                    //Preparing SRS Config IE of activeULBWP
                    int nb_srsresource =srs_config->srs_ResourceToAddModList->list.count;
                    for(int k=0; k < nb_srsresource; k++) //Preparing SRS Resource List;	/* OPTIONAL */
                    {
                        asn1cSequenceAdd(item->activeULBWP.sRSConfig.sRSResource_List->list, NRPPA_SRSResource_t, resource_item);
                        resource_item = srs_config->srs_ResourceToAddModList->list.array[k];
                    } //for(int k=0; k < nb_srsresource; k++)
                    //}

                    //Preparing SRS Resource Set List	/* OPTIONAL */
                    //int size_srsresourceset_list =srs_config->srs_ResourceSetToAddModList->list.count; // TODO confirm if correct
                    // srs_config->srs_ResourceSetToAddModList->list.count;
                    //NR_SRS_ResourceSet_t *srs_resource_set = srs_config->srs_ResourceSetToAddModList->list.array[rs];
                    //for(int x=0; x < size_srsresourceset_list; x++)
                    //{
                    //asn1cSequenceAdd(item->activeULBWP.sRSConfig.sRSResourceSet_List, NRPPA_SRSResourceSet_List_t, resourcesetlist_item);

                    //Preparing SRS Resource Set /* OPTIONAL */
                    int nb_srsresourceset =srs_config->srs_ResourceSetToAddModList->list.count;
                    for(int y=0; y < nb_srsresourceset; y++)
                    {
                        //asn1cSequenceAdd(resourcesetlist_item->list, NRPPA_SRSResourceSet_t, srsresourceset_item);
                        asn1cSequenceAdd(item->activeULBWP.sRSConfig.sRSResourceSet_List->list, NRPPA_SRSResourceSet_t, srsresourceset_item);
                        //srsresourceset_item= srs_config->srs_ResourceSetToAddModList->list.array[y];
                        srsresourceset_item->sRSResourceSetID= srs_config->srs_ResourceSetToAddModList->list.array[y]->srs_ResourceSetId;
                    }
                    //	item->activeULBWP.sRSConfig.posSRSResource_List;	/* OPTIONAL */
                    //item->activeULBWP.sRSConfig.posSRSResourceSet_List;	/* OPTIONAL */
                    //item->activeULBWP.sRSConfig.iE_Extensions;	/* OPTIONAL */
                    // OPTIONAL TODO struct NRPPA_ProtocolExtensionContainer	*iE_Extensions;
                    //item->activeULBWP.IE_Extensions;	/* OPTIONAL */

                    //   TODO  Preparing Uplink Channel BW Per SCS List information
                    int size_SCS_list =1;  //TODO adeel retrieve and add
                    for(int a=0; a < size_SCS_list; a++)
                    {
                        asn1cSequenceAdd(item->uplinkChannelBW_PerSCS_List, NRPPA_UplinkChannelBW_PerSCS_List_t, SCS_list_item);

                        int size_SpecificCarrier_list =1;  //TODO adeel retrieve and add
                        for(int b=0; b < size_SpecificCarrier_list; b++)
                        {
                            asn1cSequenceAdd(SCS_list_item->list, NRPPA_SCS_SpecificCarrier_t, SpecificCarrier_item);
                            SpecificCarrier_item->offsetToCarrier= 0;  //TODO adeel retrieve and add
                            SpecificCarrier_item->subcarrierSpacing=0; //TODO adeel retrieve and add
                            SpecificCarrier_item->carrierBandwidth=0;  //TODO adeel retrieve and add
                            //SpecificCarrier_item->iE_Extenstions // OPtional TODO
                        } // for(int b=0; b < size_SpecificCarrier_list; b++)
                    } // for(int a=0; a < size_SCS_list; a++)
                } //for (int i = 0; i < nb_of_srscarrier; i++)
            }  //Condition for the target UE of positioning
        } // UE_iterator
    } //IE 9.2.28 SRS Configuration

    //IE 9.2.36 SFN Initialisation Time (O)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningInformationUpdate_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_SFNInitialisationTime;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningInformationUpdate_IEs__value_PR_SFNInitialisationTime;
        // TODO Retreive SFN Initialisation Time and assign
//        ie->value.choice.SFNInitialisationTime = "1253486"; //TODO adeel retrieve and add TYPE typedef struct BIT_STRING_s {uint8_t *buf;	size_t size;	int bits_unused;} BIT_STRING_t;
    }



    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa PositioningInformationUpdate\n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;



}


/* PositioningActivation (Parent) procedure for  PositioningActivationRequest, PositioningActivationResponse, and PositioningActivationFailure*/
int nrppa_gNB_handle_PositioningActivation(nrppa_gnb_ue_info_t *nrppa_msg_info, NRPPA_NRPPA_PDU_t *pdu)
{
    LOG_D(NRPPA, "Processing Received PositioningActivation \n");
// Processing Received PositioningActivation
    NRPPA_PositioningActivationRequest_t     *container;
    NRPPA_PositioningActivationRequestIEs_t *ie;

    uint32_t                        nrppa_transaction_id;
    uint32_t                        srs_resource_set_id;
    //NRPPA_SRSType_t                 srs_type;
    //NRPPA_SRSSpatialRelation_t      srs_spatial_relation_info;
    //NRPPA_SRSResourceTrigger_t      srs_resource_trigger;
    //NRPPA_SFNInitialisationTime_t   sfn_initialisation_time;

    DevAssert(pdu != NULL);

    container = &pdu->choice.initiatingMessage->value.choice.PositioningActivationRequest; //IE 9.2.3 Message type (M)
    nrppa_transaction_id = pdu->choice.initiatingMessage->nrppatransactionID; // IE 9.2.4 nrppatransactionID (M)


    /* IE  SRSType (O)*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_PositioningActivationRequestIEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_SRSType, true);
    //srs_type = ie->value.choice.SRSType;

    //struct NRPPA_SemipersistentSRS	*semipersistentSRS;
    //srs_resource_set_id = srs_type.present
    //NRPPA_SpatialRelationInfo_t     spatial_relation_info;

    //struct NRPPA_AperiodicSRS	*aperiodicSRS;
    //NRPPA_SRSResourceTrigger_t      srs_resource_trigger;

    //struct NRPPA_ProtocolIE_Single_Container	*sRSType_extension;

    /* IE  Activation Time (O)*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_PositioningActivationRequestIEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_ActivationTime, true);



// TODO process activation request and generate corresponding response

// Preparing Response for the Received PositioningActivationRequest

    bool response_type_indicator = 1;  // 1 = send Position Activtion Response, 0 = send Position Activation Failure

    uint8_t *nrppa_pdu;
    uint32_t nrppa_pdu_length;

    if (response_type_indicator)
    {
        LOG_D(NRPPA, "Preparing PositioningActivationResponse message \n");
        nrppa_pdu_length= nrppa_gNB_PositioningActivationResponse(nrppa_transaction_id, nrppa_pdu);
//nrppa_gNB_itti_send_downlink_ind(ngap_gNB_instance->instance, ue_desc_p->gNB_ue_ngap_id, ie->value.choice.NAS_PDU.buf, ie->value.choice.NAS_PDU.size);

    }
    else
    {
        LOG_D(NRPPA, "Preparing PositioningActivationFailure message \n");
        nrppa_pdu_length= nrppa_gNB_PositioningActivationFailure(nrppa_transaction_id, nrppa_pdu);
    }

    /* Forward the NRPPA PDU to NGAP */
    if (nrppa_msg_info->gNB_ue_ngap_id >0 && nrppa_msg_info->amf_ue_ngap_id > 0)   // TODO ad**l check if the condition is valid
    {
        LOG_D(NRPPA, "Sending UplinkUEAssociatedNRPPa (PositioningActivationResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkUEAssociatedNRPPa(nrppa_msg_info->instance,
                nrppa_msg_info->gNB_ue_ngap_id,
                nrppa_msg_info->amf_ue_ngap_id,
                nrppa_msg_info->routing_id_buffer,
                nrppa_msg_info->routing_id_length,
                nrppa_pdu, nrppa_pdu_length);
    }
    else
    {
        LOG_D(NRPPA, "Sending UplinkNonUEAssociatedNRPPa (PositioningActivationResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkNonUEAssociatedNRPPa(nrppa_msg_info->instance,
                nrppa_msg_info->routing_id_buffer,
                nrppa_msg_info->routing_id_length,
                nrppa_pdu, nrppa_pdu_length);
    }

    return 0;
}


int nrppa_gNB_PositioningActivationResponse(uint32_t nrppa_transaction_id, uint8_t *buffer)
{
// Prepare NRPPA Positioning  Activation Response
    NRPPA_NRPPA_PDU_t pdu;  // TODO rename
    //uint8_t  *buffer;
    uint32_t  length;

    /* Prepare the NRPPA message to encode for successfulOutcome PositioningActivationResponse */

    //IE: 9.2.3 Message Type successfulOutcome PositioningActivationResponse /* mandatory */
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_successfulOutcome;
    asn1cCalloc(pdu.choice.successfulOutcome, head);
    head->procedureCode = NRPPA_ProcedureCode_id_positioningActivation;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_SuccessfulOutcome__value_PR_PositioningActivationResponse;

    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;


//  TODO IE 9.2.2 CriticalityDiagnostics (O)
    NRPPA_PositioningActivationResponse_t *out = &head->value.choice.PositioningActivationResponse;
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningActivationResponseIEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_CriticalityDiagnostics;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningActivationResponseIEs__value_PR_CriticalityDiagnostics;
        // TODO Retreive CriticalityDiagnostics information and assign
        //ie->value.choice.CriticalityDiagnostics.procedureCode = ; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.triggeringMessage; = ; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.procedureCriticality; = ; //TODO adeel retrieve and add
        ie->value.choice.CriticalityDiagnostics.nrppatransactionID =nrppa_transaction_id ;
        //ie->value.choice.CriticalityDiagnostics.iEsCriticalityDiagnostics = ; //TODO adeel retrieve and add
        //ie->value.choice.CriticalityDiagnostics.iE_Extensions = ; //TODO adeel retrieve and add
    }


    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa PositioningActivationResponse\n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;


}
int nrppa_gNB_PositioningActivationFailure(uint32_t nrppa_transaction_id, uint8_t *buffer)
{
    // Prepare NRPPA Positioning Activation Failure

    NRPPA_NRPPA_PDU_t pdu;  // TODO rename
    uint32_t  length;

    /* Prepare the NRPPA message to encode for unsuccessfulOutcome PositioningActivationFailure */
    //IE: 9.2.3 Message Type unsuccessfulOutcome PositioningActivationFailure /* mandatory */

    //IE 9.2.3 Message type (M)
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_unsuccessfulOutcome;
    asn1cCalloc(pdu.choice.unsuccessfulOutcome, head);
    head->procedureCode = NRPPA_ProcedureCode_id_positioningActivation;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_UnsuccessfulOutcome__value_PR_PositioningActivationFailure;



    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;

    NRPPA_PositioningInformationFailure_t *out = &head->value.choice.PositioningInformationFailure;


    // TODO IE 9.2.1 Cause (M)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningActivationFailureIEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_Cause;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningActivationFailureIEs__value_PR_Cause;
        //TODO Reteive Cause and assign
        //ie->value.choice.Cause. = ; //IE 1
        //ie->value.choice.Cause. =;  // IE 2 and so on
        /* Send a dummy cause */
//sample
//    ie->value.present = NGAP_NASNonDeliveryIndication_IEs__value_PR_Cause;
//   ie->value.choice.Cause.present = NGAP_Cause_PR_radioNetwork;
        //  ie->value.choice.Cause.choice.radioNetwork = NGAP_CauseRadioNetwork_radio_connection_with_ue_lost;
    }


//  TODO IE 9.2.2 CriticalityDiagnostics (O)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_PositioningActivationFailureIEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_CriticalityDiagnostics;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_PositioningActivationFailureIEs__value_PR_CriticalityDiagnostics;
        // TODO Retreive CriticalityDiagnostics information and assign
        //ie->value.choice.CriticalityDiagnostics. = ;
        //ie->value.choice.CriticalityDiagnostics. = ;
    }


    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa PositioningActivationFailure \n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;

}


int nrppa_gNB_handle_PositioningDeactivation(nrppa_gnb_ue_info_t *nrppa_msg_info,NRPPA_NRPPA_PDU_t *pdu)
{
    LOG_D(NRPPA, "Processing Received PositioningDeActivation \n");
// Processing Received PositioningDeActivation
    NRPPA_PositioningDeactivation_t     *container;
    NRPPA_PositioningDeactivationIEs_t *ie;

    uint32_t                        nrppa_transaction_id;
    uint32_t                        srs_resource_set_id;


    DevAssert(pdu != NULL);

    container = &pdu->choice.initiatingMessage->value.choice.PositioningDeactivation; //IE 9.2.3 Message type (M)
    nrppa_transaction_id = pdu->choice.initiatingMessage->nrppatransactionID; // IE 9.2.4 nrppatransactionID (M)


    /* IE  Abort Transmission(O)*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_PositioningDeactivationIEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_AbortTransmission, true);
    // ie->value.choice.AbortTransmission

    //srs_resource_set_id = ie->value.choice.AbortTransmission.S
    //Release_all = ie->value.choice.AbortTransmission.Re


// TODO process daactivation request and stop the corresponding positioning process

    return 0;
}



