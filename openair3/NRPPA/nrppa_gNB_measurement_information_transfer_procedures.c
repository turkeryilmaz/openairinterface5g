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

/*! \file nrppa_gNB_measurement_information_transfer_procedures.c
 * \brief NRPPA gNB tasks related to measurement information transfer
 * \author Adeel Malik
 * \email adeel.malik@eurecom.fr
 *\date 2023
 * \version 1.0
 * @ingroup _nrppa
 */


#include "intertask_interface.h"

#include "nrppa_common.h"
#include "nrppa_gNB_measurement_information_transfer_procedures.h"
#include "nrppa_gNB_itti_messaging.h"




int nrppa_gNB_handle_Measurement(nrppa_gnb_ue_info_t *nrppa_msg_info, NRPPA_NRPPA_PDU_t *pdu) /* Measurement (Parent) procedure for  MeasurementRequest, MeasurementResponse, and MeasurementFailure*/
{
    LOG_D(NRPPA, "Processing Received MeasurementRequest \n");
// Processing Received MeasurmentRequest
    NRPPA_MeasurementRequest_t     *container;
    NRPPA_MeasurementRequest_IEs_t *ie;
    uint32_t                         nrppa_transaction_id;

    DevAssert(pdu != NULL);

    container = &pdu->choice.initiatingMessage->value.choice.MeasurementRequest; //IE 9.2.3 Message type (M)
    nrppa_transaction_id = pdu->choice.initiatingMessage->nrppatransactionID; // IE 9.2.4 nrppatransactionID (M)

    /* IE TRP Measurement Request List */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_TRP_MeasurementRequestList, true);
    //NRPPA_TRP_MeasurementRequestList_t measurement_request_list = ie->value.choice.TRP_MeasurementRequestList; // TODO process this information

    /* IE Report Characteristics */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_ReportCharacteristics, true);
    //NRPPA_ReportCharacteristics_t report_characteristics = ie->value.choice.ReportCharacteristics; // TODO process this information

    /* IE Measurement Periodicity */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_MeasurementPeriodicity, true);
    //NRPPA_MeasurementPeriodicity_t measurement_periodicity = ie->value.choice.MeasurementPeriodicity; // TODO process this information


    /* IE TRP Measurement Quantities*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_TRPMeasurementQuantities, true);
    //NRPPA_TRPMeasurementQuantities_t measurement_quantities = ie->value.choice.TRPMeasurementQuantities; // TODO process this information


    /* IE SFNInitialisationTime*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_SFNInitialisationTime, true);
    //NRPPA_SFNInitialisationTime_t sfn_time = ie->value.choice.SFNInitialisationTime; // TODO process this information

    /* IE SRSConfiguration*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_SRSConfiguration, true);
    //NRPPA_SRSConfiguration_t srs_config = ie->value.choice.SRSConfiguration; // TODO process this information

    /* IE MeasurementBeamInfoRequest*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_MeasurementBeamInfoRequest, true);
    //NRPPA_MeasurementBeamInfoRequest_t measurement_beam_info_request = ie->value.choice.MeasurementBeamInfoRequest; // TODO process this information

    /* IE SystemFrameNumber*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_SystemFrameNumber, true);
    //NRPPA_SystemFrameNumber_t frame_num = ie->value.choice.SystemFrameNumber; // TODO process this information

    /* IE SlotNumber*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_SlotNumber, true);
    //NRPPA_SlotNumber_t slot_num = ie->value.choice.SlotNumber; // TODO process this information


// TODO process the received data  and generate the corresponding request
// Preparing Response for the Received MeasurementRequest

    /*TODO define conditions that will decide the response_type_indicator */
    bool response_type_indicator = 1;  // 1 = send Measurement Response, 0 = send Measurement Failure
    uint8_t *nrppa_pdu;
    uint32_t nrppa_pdu_length;

    if (response_type_indicator)
    {
        LOG_D(NRPPA, "Preparing MeasurementResponse message \n");
        nrppa_pdu_length= nrppa_gNB_MeasurementResponse(nrppa_transaction_id, nrppa_pdu);
    }
    else
    {
        LOG_D(NRPPA, "Preparing MeasurementFailure message \n");
        nrppa_pdu_length= nrppa_gNB_MeasurementFailure(nrppa_transaction_id, nrppa_pdu);
    }

    /* Forward the NRPPA PDU to NGAP */
    if (nrppa_msg_info->gNB_ue_ngap_id >0 && nrppa_msg_info->amf_ue_ngap_id > 0)   // TODO ad**l check if the condition is valid
    {
        LOG_D(NRPPA, "Sending UplinkUEAssociatedNRPPa (MeasurementResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkUEAssociatedNRPPa(nrppa_msg_info->instance,
                nrppa_msg_info->gNB_ue_ngap_id,
                nrppa_msg_info->amf_ue_ngap_id,
                nrppa_msg_info->routing_id_buffer,
                nrppa_msg_info->routing_id_length,
                nrppa_pdu, nrppa_pdu_length);
    }
    else
    {
        LOG_D(NRPPA, "Sending UplinkNonUEAssociatedNRPPa (MeasurementResponse/Failure) to NGAP  \n");
        nrppa_gNB_itti_send_UplinkNonUEAssociatedNRPPa(nrppa_msg_info->instance,
                nrppa_msg_info->routing_id_buffer,
                nrppa_msg_info->routing_id_length,
                nrppa_pdu, nrppa_pdu_length);
    }

    return 0;
}

int nrppa_gNB_MeasurementResponse(uint32_t nrppa_transaction_id, uint8_t *buffer)
{

    // Prepare NRPPA Measurement Response
    NRPPA_NRPPA_PDU_t pdu;
    uint32_t  length;

    /* Prepare the NRPPA message to encode for successfulOutcome MeasurementResponse */

    //IE: 9.2.3 Message Type successfulOutcome MeasurementResponse /* mandatory */
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_successfulOutcome;
    asn1cCalloc(pdu.choice.successfulOutcome, head);
    head->procedureCode = NRPPA_ProcedureCode_id_Measurement;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_SuccessfulOutcome__value_PR_MeasurementResponse;

    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;


    NRPPA_MeasurementResponse_t *out = &head->value.choice.MeasurementResponse;


    //IE = LMF  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_LMF_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementResponse_IEs__value_PR_Measurement_ID;
        ie->value.choice.Measurement_ID=0;   //dummy value TODO  define and change
    }
    // IE = RAN  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_RAN_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementResponse_IEs__value_PR_Measurement_ID_1; //TODO adeel check if it is with Measurement_ID_1 or Measurement_ID
        ie->value.choice.Measurement_ID_1=0;   //dummy value TODO  define and change
    }
    //IE = TRP Measurement Response List
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_TRP_MeasurementResponseList;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementResponse_IEs__value_PR_TRP_MeasurementResponseList;

        // TODO Retrieve Measurement info
        // TODO USE ITTI based apparoch to Retrieve Measurement info
        int nb_meas_TRPs= 1;  // TODO find the acutal number for TRP and add here
        for (int i = 0; i < nb_meas_TRPs; i++)
        {
            asn1cSequenceAdd(ie->value.choice.TRP_MeasurementResponseList.list, NRPPA_TRP_MeasurementResponseItem_t, item);
            item->tRP_ID= 0;  // IE 9.2.24 long NRPPA_TRP_ID_t  //  dummy value // TODO adeel retrive relevent info and add

            //Preparing measurementResult list an IE of MeasurementResponseItem

            // TODO adding uL_RTOA in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_uLRTOA);
            measItem_uLRTOA->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_uL_RTOA;
            measItem_uLRTOA->measuredResultsValue.choice.uL_RTOA= NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add

            // TODO adding gNB_RxTxTimeDiff in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_RxTxTimeDiff);
            measItem_RxTxTimeDiff->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_gNB_RxTxTimeDiff;
            measItem_RxTxTimeDiff->measuredResultsValue.choice.gNB_RxTxTimeDiff= NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add


            // TODO adding uL_AngleOfArrival in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_uLAoA);
            measItem_uLAoA->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_uL_AngleOfArrival;
            measItem_uLAoA->measuredResultsValue.choice.uL_AngleOfArrival= NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add


            // TODO adding uL_SRS_RSRP in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_SrsRSRP);
            measItem_SrsRSRP->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_uL_SRS_RSRP;
            measItem_SrsRSRP->measuredResultsValue.choice.uL_SRS_RSRP=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add

        } //for (int i = 0; i < nb_meas_TRPs; i++)

    } //IE = TRP Measurement Response List

//  TODO IE 9.2.2 CriticalityDiagnostics (O)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementResponse_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_CriticalityDiagnostics;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_MeasurementResponse_IEs__value_PR_CriticalityDiagnostics;
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
        NRPPA_ERROR("Failed to encode Uplink NRPPa MeasurementResponse\n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;
}


int nrppa_gNB_MeasurementFailure(uint32_t nrppa_transaction_id, uint8_t *buffer)
{

    // Prepare NRPPA Measurement Failure
    NRPPA_NRPPA_PDU_t pdu;
    uint32_t  length;
    /* Prepare the NRPPA message to encode for unsuccessfulOutcome MeasurementFailure */

    //IE: 9.2.3 Message Type unsuccessfulOutcome MeasurementFaliure /* mandatory */
    //IE 9.2.3 Message type (M)
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_unsuccessfulOutcome;
    asn1cCalloc(pdu.choice.unsuccessfulOutcome, head);
    head->procedureCode = NRPPA_ProcedureCode_id_Measurement;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_UnsuccessfulOutcome__value_PR_MeasurementFailure;

    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;

    NRPPA_MeasurementFailure_t *out = &head->value.choice.MeasurementFailure;

    //IE = LMF  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementFailure_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_LMF_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementFailure_IEs__value_PR_Measurement_ID;
        ie->value.choice.Measurement_ID=0;   //dummy value TODO  define and change
    }

    // TODO IE 9.2.1 Cause (M)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementFailure_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_Cause;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_MeasurementFailure_IEs__value_PR_Cause;
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
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementFailure_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_CriticalityDiagnostics;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_MeasurementFailure_IEs__value_PR_CriticalityDiagnostics;
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
        NRPPA_ERROR("Failed to encode Uplink NRPPa MeasurementFailure \n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;
}

int nrppa_gNB_MeasurementReport(uint32_t nrppa_transaction_id, uint8_t *buffer)  // adeel TODO when and where to call this function
{
// Prepare NRPPA Measurement Report
LOG_D(NRPPA, "Preparing MeasurementReport message \n");
    NRPPA_NRPPA_PDU_t pdu;
    uint32_t  length;

    /* Prepare the NRPPA message to encode for initiatingMessage MeasurementReport */

    //IE: 9.2.3 Message Type initiatingMessage MeasurementReport /* mandatory */
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_initiatingMessage;
    asn1cCalloc(pdu.choice.initiatingMessage, head);
    head->procedureCode = NRPPA_ProcedureCode_id_MeasurementReport;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_InitiatingMessage__value_PR_MeasurementReport;

    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;


    NRPPA_MeasurementReport_t *out = &head->value.choice.MeasurementReport;


    //IE = LMF  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementReport_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_LMF_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementReport_IEs__value_PR_Measurement_ID;
        ie->value.choice.Measurement_ID=0;   //dummy value TODO  define and change
    }

    // IE = RAN  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementReport_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_RAN_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementReport_IEs__value_PR_Measurement_ID_1; //TODO adeel check if it is with Measurement_ID_1 or Measurement_ID
        ie->value.choice.Measurement_ID_1=0;   //dummy value TODO  define and change
    }
    //IE = TRP Measurement Report List  (= TRP_MeasurementResponseList)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementReport_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_TRP_MeasurementReportList;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementReport_IEs__value_PR_TRP_MeasurementResponseList; // TRP_MeasurementReportList = TRP_MeasurementResponseList

        // TODO Retrieve Measurement info
        int nb_meas_TRPs= 1;  // TODO find the acutal number for TRP and add here
        for (int i = 0; i < nb_meas_TRPs; i++)
        {
            asn1cSequenceAdd(ie->value.choice.TRP_MeasurementResponseList.list, NRPPA_TRP_MeasurementResponseItem_t, item); // NRPPA_TRP_MeasurementReportItem_t=NRPPA_TRP_MeasurementResponseItem_t
            item->tRP_ID= 0;  // IE 9.2.24 long NRPPA_TRP_ID_t  //  dummy value // TODO adeel retrive relevent info and add

            //Preparing IE 9.2.37 measurementResult list an IE of MeasurementReportItem (=MeasurementResponseItem)

            // TODO adding uL_RTOA in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_uLRTOA);
            measItem_uLRTOA->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_uL_RTOA;
            measItem_uLRTOA->measuredResultsValue.choice.uL_RTOA= NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLRTOA->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add

            // TODO adding gNB_RxTxTimeDiff in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_RxTxTimeDiff);
            measItem_RxTxTimeDiff->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_gNB_RxTxTimeDiff;
            measItem_RxTxTimeDiff->measuredResultsValue.choice.gNB_RxTxTimeDiff= NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_RxTxTimeDiff->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add


            // TODO adding uL_AngleOfArrival in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_uLAoA);
            measItem_uLAoA->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_uL_AngleOfArrival;
            measItem_uLAoA->measuredResultsValue.choice.uL_AngleOfArrival= NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_uLAoA->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add


            // TODO adding uL_SRS_RSRP in measurment results
            asn1cSequenceAdd(item->measurementResult.list, NRPPA_TrpMeasurementResultItem_t, measItem_SrsRSRP);
            measItem_SrsRSRP->measuredResultsValue.present = NRPPA_TrpMeasuredResultsValue_PR_uL_SRS_RSRP;
            measItem_SrsRSRP->measuredResultsValue.choice.uL_SRS_RSRP=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.systemFrameNumber=0;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.slotIndex.present=NRPPA_TimeStampSlotIndex_PR_sCS_30;  //  dummy value check NRPPA_TimeStampSlotIndex_t// TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.slotIndex.choice.sCS_30=0; //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->timeStamp.measurementTime=NULL;  //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->measurementQuality=NULL; //  dummy value // TODO adeel retrive relevent info and add
            measItem_SrsRSRP->measurementBeamInfo=NULL; //  dummy value // TODO adeel retrive relevent info and add

        } //for (int i = 0; i < nb_meas_TRPs; i++)

    } //IE TRP Measurement Report List


    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa MeasurementReport\n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;
}

int nrppa_gNB_handle_MeasurementUpdate(nrppa_gnb_ue_info_t *nrppa_msg_info, NRPPA_NRPPA_PDU_t *pdu)  // called via handler
{
    LOG_D(NRPPA, "Processing Received MeasurementUpdate \n");
// Processing Received MeasurementUpdate
    NRPPA_MeasurementUpdate_t     *container;
    NRPPA_MeasurementUpdate_IEs_t *ie;
    uint32_t                         nrppa_transaction_id;

    DevAssert(pdu != NULL);

    container = &pdu->choice.initiatingMessage->value.choice.MeasurementUpdate; //IE 9.2.3 Message type (M)
    nrppa_transaction_id = pdu->choice.initiatingMessage->nrppatransactionID; // IE 9.2.4 nrppatransactionID (M)

    /* IE LMF_Measurement_ID */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementUpdate_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_LMF_Measurement_ID, true);
    NRPPA_Measurement_ID_t LMF_Meas_ID = ie->value.choice.Measurement_ID; // TODO process this information

    /* IE RAN_Measurement_ID */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementUpdate_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_RAN_Measurement_ID, true);
    NRPPA_Measurement_ID_t RAN_Meas_ID = ie->value.choice.Measurement_ID_1; // TODO process this information  //TODO adeel check if it is with Measurement_ID_1 or Measurement_ID

    /* IE SRSConfiguration*/
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementRequest_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_SRSConfiguration, true);
    NRPPA_SRSConfiguration_t srs_config = ie->value.choice.SRSConfiguration; // TODO process this information


// TODO process the received data  and Overwrite the previously received measurement configuration

    LOG_D(NRPPA, "MeasurementUpdate: Overwriting Previously received measurement configuration \n");
// NO response to LMF required
    return 0;

}

int nrppa_gNB_handle_MeasurementAbort(nrppa_gnb_ue_info_t *nrppa_msg_info, NRPPA_NRPPA_PDU_t *pdu)  // called via handler
{
    LOG_D(NRPPA, "Processing Received MeasurementAbort \n");
// Processing Received MeasurementAbort
    NRPPA_MeasurementAbort_t     *container;
    NRPPA_MeasurementAbort_IEs_t *ie;
    uint32_t                         nrppa_transaction_id;

    DevAssert(pdu != NULL);

    container = &pdu->choice.initiatingMessage->value.choice.MeasurementAbort; //IE 9.2.3 Message type (M)
    nrppa_transaction_id = pdu->choice.initiatingMessage->nrppatransactionID; // IE 9.2.4 nrppatransactionID (M)

    /* IE LMF_Measurement_ID */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementAbort_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_LMF_Measurement_ID, true);
    NRPPA_Measurement_ID_t LMF_Meas_ID = ie->value.choice.Measurement_ID; // TODO process this information

    /* IE RAN_Measurement_ID */
    NRPPA_FIND_PROTOCOLIE_BY_ID(NRPPA_MeasurementAbort_IEs_t, ie, container,
                                NRPPA_ProtocolIE_ID_id_RAN_Measurement_ID, true);
    NRPPA_Measurement_ID_t RAN_Meas_ID = ie->value.choice.Measurement_ID; // TODO process this information


// TODO process the received data  and Release the previously allocated resource for measurement

    LOG_D(NRPPA, "MeasurementAbort: Releasing previously allocated resource for measurement %d\n", LMF_Meas_ID );
// NO response to LMF required
    return 0;

}

int nrppa_gNB_MeasurementFailureIndication(uint32_t nrppa_transaction_id, uint8_t *buffer)  // adeel TODO when and where to call this function
{
    LOG_D(NRPPA, "Preparing Measurement Failure Indication message \n");
    // Prepare NRPPA Measurement Failure Indication
    NRPPA_NRPPA_PDU_t pdu;
    uint32_t  length;
    /* Prepare the NRPPA message to encode for initiatingMessage MeasurementFailureIndication */

    //IE: 9.2.3 Message Type initiatingMessage MeasurementFaliureIndication /* mandatory */
    //IE 9.2.3 Message type (M)
    memset(&pdu, 0, sizeof(pdu));
    pdu.present = NRPPA_NRPPA_PDU_PR_initiatingMessage;
    asn1cCalloc(pdu.choice.initiatingMessage, head);
    head->procedureCode = NRPPA_ProcedureCode_id_MeasurementFailureIndication;
    head->criticality = NRPPA_Criticality_reject;
    head->value.present = NRPPA_InitiatingMessage__value_PR_MeasurementFailureIndication;

    //IE 9.2.4 nrppatransactionID  /* mandatory */
    head->nrppatransactionID =nrppa_transaction_id;

    NRPPA_MeasurementFailureIndication_t *out = &head->value.choice.MeasurementFailureIndication;

    //IE = LMF  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementFailureIndication_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_LMF_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementFailureIndication_IEs__value_PR_Measurement_ID;
        ie->value.choice.Measurement_ID=0;   //dummy value TODO  define and change
    }

    //IE = RAN  Measurement ID  /* mandatory */
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementFailureIndication_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_RAN_Measurement_ID;
        ie->criticality = NRPPA_Criticality_reject;
        ie->value.present = NRPPA_MeasurementFailureIndication_IEs__value_PR_Measurement_ID_1; //TODO adeel check if it is with Measurement_ID_1 or Measurement_ID
        ie->value.choice.Measurement_ID_1=0;   //dummy value TODO  define and change
    }

    // TODO IE 9.2.1 Cause (M)
    {
        asn1cSequenceAdd(out->protocolIEs.list, NRPPA_MeasurementFailureIndication_IEs_t, ie);
        ie->id = NRPPA_ProtocolIE_ID_id_Cause;
        ie->criticality = NRPPA_Criticality_ignore;
        ie->value.present = NRPPA_MeasurementFailureIndication_IEs__value_PR_Cause;
        //TODO Reteive Cause and assign
        //ie->value.choice.Cause. = ; //IE 1
        //ie->value.choice.Cause. =;  // IE 2 and so on
        /* Send a dummy cause */
//sample
//    ie->value.present = NGAP_NASNonDeliveryIndication_IEs__value_PR_Cause;
//   ie->value.choice.Cause.present = NGAP_Cause_PR_radioNetwork;
        //  ie->value.choice.Cause.choice.radioNetwork = NGAP_CauseRadioNetwork_radio_connection_with_ue_lost;
    }


    /* Encode NRPPA message */
    if (nrppa_gNB_encode_pdu(&pdu, &buffer, &length) < 0)
    {
        NRPPA_ERROR("Failed to encode Uplink NRPPa MeasurementFailureIndication \n");
        /* Encode procedure has failed... */
        return -1;
    }

    return length;
}



