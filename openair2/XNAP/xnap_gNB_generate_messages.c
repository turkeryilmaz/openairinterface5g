/* Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
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

#include "intertask_interface.h"
#include "xnap_common.h"
#include "xnap_gNB_task.h"
#include "xnap_gNB_generate_messages.h"
#include "XNAP_ProtocolIE-Field.h"
#include "XNAP_GlobalgNB-ID.h"
#include "XNAP_ServedCells-NR-Item.h"
#include "XNAP_ServedCellInformation-NR.h"
#include "XNAP_NRFrequencyBandItem.h"
#include "xnap_gNB_itti_messaging.h"
#include "XNAP_ServedCells-NR.h"
#include "assertions.h"
#include "conversions.h"
#include "XNAP_BroadcastPLMNinTAISupport-Item.h"
#include "XNAP_TAISupport-Item.h"
#include "XNAP_GlobalAMF-Region-Information.h"
#include "XNAP_TargetCellList-Item.h"
#include "XNAP_GlobalAMF-Region-Information.h"
#include "XNAP_QoSFlowsToBeSetup-Item.h"
#include "XNAP_NRModeInfoFDD.h"
#include "XNAP_NRModeInfoTDD.h"
#include "openair2/RRC/NR/nr_rrc_defs.h"
#include "xnap_gNB_defs.h"

int xnap_gNB_generate_xn_setup_request(sctp_assoc_t assoc_id, xnap_setup_req_t *req)
{
  XNAP_XnAP_PDU_t pdu;
  XNAP_XnSetupRequest_t *out;
  XNAP_XnSetupRequest_IEs_t *ie;
  XNAP_BroadcastPLMNinTAISupport_Item_t *e_BroadcastPLMNinTAISupport_ItemIE;
  XNAP_TAISupport_Item_t *TAISupport_ItemIEs;
  XNAP_S_NSSAI_t *e_S_NSSAI_ItemIE;
  XNAP_GlobalAMF_Region_Information_t *e_GlobalAMF_Region_Information_ItemIEs;
  XNAP_ServedCells_NR_Item_t *servedCellMember;
  XNAP_NRFrequencyBandItem_t *nrfreqbanditemul;
  XNAP_NRFrequencyBandItem_t *nrfreqbanditemdl;
  XNAP_NRFrequencyBandItem_t *nrfreqbanditem;
  XNAP_PLMN_Identity_t *plmn;

  uint8_t *buffer;
  uint32_t len;
  int ret = 0;

  /* Prepare the XnAP message to encode */
  memset(&pdu, 0, sizeof(pdu));

  pdu.present = XNAP_XnAP_PDU_PR_initiatingMessage;
  // pdu.choice.initiatingMessage = &initiating_msg;
  pdu.choice.initiatingMessage = (XNAP_InitiatingMessage_t *)calloc(1, sizeof(XNAP_InitiatingMessage_t));
  pdu.choice.initiatingMessage->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.initiatingMessage->criticality = XNAP_Criticality_reject;
  pdu.choice.initiatingMessage->value.present = XNAP_InitiatingMessage__value_PR_XnSetupRequest;

  out = &pdu.choice.initiatingMessage->value.choice.XnSetupRequest;

  /* mandatory */
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_GlobalNG_RAN_node_ID;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_GlobalNG_RANNode_ID;
  ie->value.choice.GlobalNG_RANNode_ID.present = XNAP_GlobalNG_RANNode_ID_PR_gNB;
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB = (XNAP_GlobalgNB_ID_t *)calloc(1, sizeof(XNAP_GlobalgNB_ID_t));
  MCC_MNC_TO_PLMNID(req->info.plmn.mcc,
                    req->info.plmn.mnc,
                    req->info.plmn.mnc_digit_length,
                    &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id);

  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.present = XNAP_GNB_ID_Choice_PR_gnb_ID;

  MACRO_GNB_ID_TO_BIT_STRING(req->gNB_id, &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID); // 28 bits
  LOG_I(XNAP,
        "%lu -> %02x%02x%02x\n",
        req->gNB_id,
        ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[0],
        ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[1],
        ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[2]);

  asn1cSeqAdd(&out->protocolIEs.list, ie);

  /* mandatory */ // TAI Support list
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_TAISupport_list;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_TAISupport_List;

  // for (int i=0;i<1;i++)
  {
    TAISupport_ItemIEs = (XNAP_TAISupport_Item_t *)calloc(1, sizeof(XNAP_TAISupport_Item_t));
    INT24_TO_OCTET_STRING(req->tai_support, &TAISupport_ItemIEs->tac);
    {
      for (int j = 0; j < 1; j++) {
        e_BroadcastPLMNinTAISupport_ItemIE =
            (XNAP_BroadcastPLMNinTAISupport_Item_t *)calloc(1, sizeof(XNAP_BroadcastPLMNinTAISupport_Item_t));

        MCC_MNC_TO_PLMNID(req->info.plmn.mcc,
                          req->info.plmn.mnc,
                          req->info.plmn.mnc_digit_length,
                          &e_BroadcastPLMNinTAISupport_ItemIE->plmn_id);

        for (int k = 0; k < 1; k++) {
          e_S_NSSAI_ItemIE = (XNAP_S_NSSAI_t *)calloc(1, sizeof(XNAP_S_NSSAI_t));
          INT8_TO_OCTET_STRING(req->snssai[k].sst, &e_S_NSSAI_ItemIE->sst);
          if (req->snssai[k].sd != 0xffffff && req->snssai[k].sd != 0) {
            e_S_NSSAI_ItemIE->sd = calloc(3, sizeof(OCTET_STRING_t));
            INT24_TO_OCTET_STRING(req->snssai[k].sd, e_S_NSSAI_ItemIE->sd);
          }
          asn1cSeqAdd(&e_BroadcastPLMNinTAISupport_ItemIE->tAISliceSupport_List.list, e_S_NSSAI_ItemIE);
        }
        asn1cSeqAdd(&TAISupport_ItemIEs->broadcastPLMNs.list, e_BroadcastPLMNinTAISupport_ItemIE);
      }
    }
    asn1cSeqAdd(&ie->value.choice.TAISupport_List.list, TAISupport_ItemIEs);
  }
  //}

  asn1cSeqAdd(&out->protocolIEs.list, ie);

  /* mandatory */
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_List_of_served_cells_NR;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_ServedCells_NR;
  {
    servedCellMember = (XNAP_ServedCells_NR_Item_t *)calloc(1, sizeof(XNAP_ServedCells_NR_Item_t));
    {
      servedCellMember->served_cell_info_NR.nrPCI = req->info.nr_pci; // long

      MCC_MNC_TO_PLMNID(req->info.plmn.mcc,
                        req->info.plmn.mnc,
                        req->info.plmn.mnc_digit_length,
                        &servedCellMember->served_cell_info_NR.cellID.plmn_id); // octet string
      NR_CELL_ID_TO_BIT_STRING(req->gNB_id,
                               &servedCellMember->served_cell_info_NR.cellID.nr_CI); // bit string

      INT24_TO_OCTET_STRING(req->tai_support, &servedCellMember->served_cell_info_NR.tac); // octet string
      for (int k = 0; k < 1; k++) {
        plmn = (XNAP_PLMN_Identity_t *)calloc(1, sizeof(XNAP_PLMN_Identity_t));
        {
          MCC_MNC_TO_PLMNID(req->info.plmn.mcc, req->info.plmn.mnc, req->info.plmn.mnc_digit_length, plmn);
          asn1cSeqAdd(&servedCellMember->served_cell_info_NR.broadcastPLMN.list, plmn);
        }
      }
      if (req->info.mode == XNAP_MODE_FDD) { // FDD
        const xnap_fdd_info_t *fdd = &req->info.fdd;
        servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_fdd;
        servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd =
            (XNAP_NRModeInfoFDD_t *)calloc(1, sizeof(XNAP_NRModeInfoFDD_t));
        servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.nrARFCN = fdd->ul_freqinfo.arfcn;
        for (int j = 0; j < 1; j++) { // fdd ul number of available freq_Bands = 1
          nrfreqbanditemul = (XNAP_NRFrequencyBandItem_t *)calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
          nrfreqbanditemul->nr_frequency_band = fdd->ul_freqinfo.band;
          asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list,
                      nrfreqbanditemul);
        }

        servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRFrequencyInfo.nrARFCN = fdd->dl_freqinfo.arfcn;
        for (int j = 0; j < 1; j++) { ////fdd dl number of available freq_Bands = 1
          nrfreqbanditemdl = (XNAP_NRFrequencyBandItem_t *)calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
          nrfreqbanditemdl->nr_frequency_band = fdd->dl_freqinfo.band;
          asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list,
                      nrfreqbanditemdl);
        }

        switch (fdd->ul_tbw.scs) {
          case 15:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs15;
            break;

          case 30:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs30;
            break;

          case 60:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs60;
            break;

          case 120:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs120;
            break;
        }

        switch (fdd->ul_tbw.nrb) {
          case 11:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb11;
            break;

          case 18:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb18;
            break;

          case 24:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb24;
            break;

          case 78:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb78;
            break;

          case 106:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb106;
            break;

          case 162:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb162;
            break;
          case 217:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb217;
            break;
          case 273:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb273;
            break;
          default:
            AssertFatal(0, "Failed: Check value for N_RB_DL/N_RB_UL");
            break;
        }
        switch (fdd->dl_tbw.scs) {
          case 15:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs15;
            break;

          case 30:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs30;
            break;

          case 60:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs60;
            break;

          case 120:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs120;
            break;
        }

        switch (fdd->dl_tbw.nrb) {
          case 11:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb11;
            break;

          case 18:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb18;
            break;

          case 24:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb24;
            break;

          case 78:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb78;
            break;

          case 106:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb106;
            break;

          case 162:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb162;
            break;
          case 217:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb217;
            break;
          case 273:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb273;
            break;
          default:
            AssertFatal(0, "Failed: Check value for N_RB_DL/N_RB_UL");
            break;
        }
      } else if (req->info.mode == XNAP_MODE_TDD) { // TDD
        const xnap_tdd_info_t *tdd = &req->info.tdd;
        servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_tdd;
        servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd =
            (XNAP_NRModeInfoTDD_t *)calloc(1, sizeof(XNAP_NRModeInfoTDD_t));
        servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.nrARFCN = tdd->freqinfo.arfcn;
        for (int j = 0; j < 1; j++) { // number of available bands = j = 1
          nrfreqbanditem = (XNAP_NRFrequencyBandItem_t *)calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
          nrfreqbanditem->nr_frequency_band = tdd->freqinfo.band;
          asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.frequencyBand_List.list,
                      nrfreqbanditem);
        }
        switch (tdd->tbw.scs) {
          case 15:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs15;
            break;

          case 30:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs30;
            break;

          case 60:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs60;
            break;

          case 120:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs120;
            break;
        }
        switch (tdd->tbw.nrb) {
          case 11:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb11;
            break;

          case 18:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb18;
            break;

          case 24:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb24;
            break;

          case 78:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb78;
            break;

          case 106:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb106;
            break;

          case 162:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb162;
            break;
          case 217:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb217;
            break;
          case 273:
            servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb273;
            break;
          default:
            AssertFatal(0, "Failed: Check value for N_RB_DL/N_RB_UL");
            break;
        }
      }
      // Setting MTC to 0 now. Will be handled later.
      INT8_TO_OCTET_STRING(0, &servedCellMember->served_cell_info_NR.measurementTimingConfiguration);
      servedCellMember->served_cell_info_NR.connectivitySupport.eNDC_Support = 1;
    }
    asn1cSeqAdd(&ie->value.choice.ServedCells_NR.list, servedCellMember);
  }
  asn1cSeqAdd(&out->protocolIEs.list, ie);

  /* mandatory */ // AMFRegion
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_AMF_Region_Information;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_AMF_Region_Information;
  // for (int i=0;i<1;i++)
  {
    e_GlobalAMF_Region_Information_ItemIEs =
        (XNAP_GlobalAMF_Region_Information_t *)calloc(1, sizeof(XNAP_GlobalAMF_Region_Information_t));

    MCC_MNC_TO_PLMNID(req->info.plmn.mcc,
                      req->info.plmn.mnc,
                      req->info.plmn.mnc_digit_length,
                      &e_GlobalAMF_Region_Information_ItemIEs->plmn_ID);
    e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.size = 1;
    e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.buf =
        calloc(1, e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.size);
    e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.buf[0] = 80; // TODO: Hardcoded for now
    e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.bits_unused = 0;

    asn1cSeqAdd(&ie->value.choice.AMF_Region_Information.list, e_GlobalAMF_Region_Information_ItemIEs);
  }
  asn1cSeqAdd(&out->protocolIEs.list, ie);

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP, "Failed to encode Xn setup request\n");
    return -1;
  }

  xnap_gNB_itti_send_sctp_data_req(assoc_id, buffer, len, 0);

  return ret;
}

int xnap_gNB_generate_xn_setup_failure(sctp_assoc_t assoc_id, xnap_setup_failure_t *fail)
{
  XNAP_XnAP_PDU_t pdu;
  XNAP_XnSetupFailure_t *out;
  XNAP_XnSetupFailure_IEs_t *ie;

  uint8_t *buffer;
  uint32_t len;
  int ret = 0;

  /* Prepare the XnAP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = XNAP_XnAP_PDU_PR_unsuccessfulOutcome;
  pdu.choice.unsuccessfulOutcome = (XNAP_UnsuccessfulOutcome_t *)calloc(1, sizeof(XNAP_UnsuccessfulOutcome_t));
  pdu.choice.unsuccessfulOutcome->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.unsuccessfulOutcome->criticality = XNAP_Criticality_reject;
  pdu.choice.unsuccessfulOutcome->value.present = XNAP_UnsuccessfulOutcome__value_PR_XnSetupFailure;
  out = &pdu.choice.unsuccessfulOutcome->value.choice.XnSetupFailure;

  /* mandatory */
  ie = (XNAP_XnSetupFailure_IEs_t *)calloc(1, sizeof(XNAP_XnSetupFailure_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_Cause;
  ie->criticality = XNAP_Criticality_ignore;
  ie->value.present = XNAP_XnSetupFailure_IEs__value_PR_Cause;

  xnap_gNB_set_cause(&ie->value.choice.Cause, fail->cause_type, fail->cause_value);

  asn1cSeqAdd(&out->protocolIEs.list, ie);

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP, "Failed to encode Xn setup failure\n");
    return -1;
  }

  xnap_gNB_itti_send_sctp_data_req(assoc_id, buffer, len, 0);

  return ret;
}

int xnap_gNB_set_cause(XNAP_Cause_t *cause_p, XNAP_Cause_PR cause_type, long cause_value)
{
  DevAssert(cause_p != NULL);

  switch (cause_type) {
    case XNAP_CAUSE_RADIO_NETWORK:
      cause_p->present = XNAP_Cause_PR_radioNetwork;
      cause_p->choice.radioNetwork = cause_value;
      break;

    case XNAP_CAUSE_TRANSPORT:
      cause_p->present = XNAP_Cause_PR_transport;
      cause_p->choice.transport = cause_value;
      break;

    case XNAP_CAUSE_PROTOCOL:
      cause_p->present = XNAP_Cause_PR_protocol;
      cause_p->choice.protocol = cause_value;
      break;

    case XNAP_CAUSE_MISC:
      cause_p->present = XNAP_Cause_PR_misc;
      cause_p->choice.misc = cause_value;
      break;

    case XNAP_CAUSE_NOTHING:
    default:
      cause_p->present = XNAP_Cause_PR_NOTHING;
      break;
  }

  return 0;
}

int xnap_gNB_generate_xn_setup_response(sctp_assoc_t assoc_id, xnap_setup_resp_t *resp)
// xnap_gNB_instance_t *instance_p, xnap_gNB_data_t *xnap_gNB_data_p)
{
  XNAP_XnAP_PDU_t pdu;
  uint8_t *buffer = NULL;
  uint32_t len = 0;
  int ret = 0;
  XNAP_XnSetupResponse_t *out;
  XNAP_XnSetupResponse_IEs_t *ie;
  XNAP_PLMN_Identity_t *plmn;
  XNAP_BroadcastPLMNinTAISupport_Item_t *e_BroadcastPLMNinTAISupport_ItemIE;
  XNAP_TAISupport_Item_t *TAISupport_ItemIEs;
  XNAP_S_NSSAI_t *e_S_NSSAI_ItemIE;
  // XNAP_GlobalAMF_Region_Information_t   *e_GlobalAMF_Region_Information_ItemIEs;
  XNAP_ServedCells_NR_Item_t *servedCellMember;
  // XNAP_ServedCells_NR_t       *ServedCells_NR;
  XNAP_NRFrequencyBandItem_t *nrfreqbanditemul;
  XNAP_NRFrequencyBandItem_t *nrfreqbanditemdl;
  XNAP_NRFrequencyBandItem_t *nrfreqbanditem;

  /* Prepare the XNAP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = XNAP_XnAP_PDU_PR_successfulOutcome;
  pdu.choice.successfulOutcome = (XNAP_SuccessfulOutcome_t *)calloc(1, sizeof(XNAP_SuccessfulOutcome_t));
  pdu.choice.successfulOutcome->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.successfulOutcome->criticality = XNAP_Criticality_reject;
  pdu.choice.successfulOutcome->value.present = XNAP_SuccessfulOutcome__value_PR_XnSetupResponse;
  out = &pdu.choice.successfulOutcome->value.choice.XnSetupResponse;

  /* mandatory */
  ie = (XNAP_XnSetupResponse_IEs_t *)calloc(1, sizeof(XNAP_XnSetupResponse_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_GlobalNG_RAN_node_ID;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupResponse_IEs__value_PR_GlobalNG_RANNode_ID;
  ie->value.choice.GlobalNG_RANNode_ID.present = XNAP_GlobalNG_RANNode_ID_PR_gNB;
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB = (XNAP_GlobalgNB_ID_t *)calloc(1, sizeof(XNAP_GlobalgNB_ID_t));
  MCC_MNC_TO_PLMNID(resp->info.plmn.mcc,
                    resp->info.plmn.mnc,
                    resp->info.plmn.mnc_digit_length,
                    &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id);
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.present = XNAP_GNB_ID_Choice_PR_gnb_ID;
  MACRO_GNB_ID_TO_BIT_STRING(resp->gNB_id, &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID);
  LOG_I(XNAP,
        "%ld -> %02x%02x%02x\n",
        resp->gNB_id,
        ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[0],
        ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[1],
        ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[2]);
  asn1cSeqAdd(&out->protocolIEs.list, ie);

  /* mandatory */ // TAI Support list
  ie = (XNAP_XnSetupResponse_IEs_t *)calloc(1, sizeof(XNAP_XnSetupResponse_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_TAISupport_list;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupResponse_IEs__value_PR_TAISupport_List;

  for (int i = 0; i < 1; i++) {
    TAISupport_ItemIEs = (XNAP_TAISupport_Item_t *)calloc(1, sizeof(XNAP_TAISupport_Item_t));
    INT24_TO_OCTET_STRING(resp->tai_support, &TAISupport_ItemIEs->tac);
    {
      for (int j = 0; j < 1; j++) {
        e_BroadcastPLMNinTAISupport_ItemIE =
            (XNAP_BroadcastPLMNinTAISupport_Item_t *)calloc(1, sizeof(XNAP_BroadcastPLMNinTAISupport_Item_t));

        MCC_MNC_TO_PLMNID(resp->info.plmn.mcc,
                          resp->info.plmn.mnc,
                          resp->info.plmn.mnc_digit_length,
                          &e_BroadcastPLMNinTAISupport_ItemIE->plmn_id);

        {
          for (int k = 0; k < 1; k++) {
            e_S_NSSAI_ItemIE = (XNAP_S_NSSAI_t *)calloc(1, sizeof(XNAP_S_NSSAI_t));
            e_S_NSSAI_ItemIE->sst.size = 1; // OCTET STRING(SIZE(1))
            e_S_NSSAI_ItemIE->sst.buf = calloc(e_S_NSSAI_ItemIE->sst.size, sizeof(OCTET_STRING_t));
            e_S_NSSAI_ItemIE->sst.buf[0] = 1;

            asn1cSeqAdd(&e_BroadcastPLMNinTAISupport_ItemIE->tAISliceSupport_List.list, e_S_NSSAI_ItemIE);
          }
        }
        asn1cSeqAdd(&TAISupport_ItemIEs->broadcastPLMNs.list, e_BroadcastPLMNinTAISupport_ItemIE);
      }
    }
    asn1cSeqAdd(&ie->value.choice.TAISupport_List.list, TAISupport_ItemIEs);
  }
  //}

  asn1cSeqAdd(&out->protocolIEs.list, ie);

  /* mandatory */
  ie = (XNAP_XnSetupResponse_IEs_t *)calloc(1, sizeof(XNAP_XnSetupResponse_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_List_of_served_cells_NR;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupResponse_IEs__value_PR_ServedCells_NR;
  {
    servedCellMember = (XNAP_ServedCells_NR_Item_t *)calloc(1, sizeof(XNAP_ServedCells_NR_Item_t));
    servedCellMember->served_cell_info_NR.nrPCI = resp->info.nr_pci; // long

    MCC_MNC_TO_PLMNID(resp->info.plmn.mcc,
                      resp->info.plmn.mnc,
                      resp->info.plmn.mnc_digit_length,
                      &servedCellMember->served_cell_info_NR.cellID.plmn_id); // octet string
    NR_CELL_ID_TO_BIT_STRING(resp->gNB_id,
                             &servedCellMember->served_cell_info_NR.cellID.nr_CI); // bit string

    INT24_TO_OCTET_STRING(resp->tai_support, &servedCellMember->served_cell_info_NR.tac); // octet string
    for (int k = 0; k < 1; k++) {
      plmn = (XNAP_PLMN_Identity_t *)calloc(1, sizeof(XNAP_PLMN_Identity_t));
      {
        MCC_MNC_TO_PLMNID(resp->info.plmn.mcc, resp->info.plmn.mnc, resp->info.plmn.mnc_digit_length, plmn);
        asn1cSeqAdd(&servedCellMember->served_cell_info_NR.broadcastPLMN.list, plmn);
      }
    }

    if (resp->info.mode == XNAP_MODE_FDD) { // FDD
      const xnap_fdd_info_t *fdd = &resp->info.fdd;
      servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_fdd;
      servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd = (XNAP_NRModeInfoFDD_t *)calloc(1, sizeof(XNAP_NRModeInfoFDD_t));
      servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.nrARFCN = fdd->ul_freqinfo.arfcn;
      for (int j = 0; j < 1; j++) { // fdd ul number of available freq_Bands = 1
        nrfreqbanditemul = (XNAP_NRFrequencyBandItem_t *)calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
        nrfreqbanditemul->nr_frequency_band = fdd->ul_freqinfo.band;
        asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list,
                    nrfreqbanditemul);
      }
      servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRFrequencyInfo.nrARFCN = fdd->dl_freqinfo.arfcn;
      for (int j = 0; j < 1; j++) { ////fdd dl number of available freq_Bands = 1
        nrfreqbanditemdl = (XNAP_NRFrequencyBandItem_t *)calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
        nrfreqbanditemdl->nr_frequency_band = fdd->dl_freqinfo.band;
        asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list,
                    nrfreqbanditemdl);
      }
      switch (fdd->ul_tbw.scs) {
        case 15:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs15;
          break;
        case 30:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs30;
          break;
        case 60:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs60;
          break;
        case 120:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs120;
          break;
      }

      switch (fdd->ul_tbw.nrb) {
        case 0:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb11;
          break;
        case 1:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb18;
          break;
        case 2:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb24;
          break;
        case 11:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb78;
          break;
        case 14:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb106;
          break;
        case 21:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb162;
          break;
        case 24:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb217;
          break;
        case 28:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb273;
          break;
        default:
          AssertFatal(0, "Failed: Check value for N_RB_DL/N_RB_UL");
          break;
      }
      switch (fdd->dl_tbw.scs) {
        case 15:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs15;
          break;
        case 30:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs30;
          break;
        case 60:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs60;
          break;
        case 120:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs120;
          break;
      }
      switch (fdd->dl_tbw.nrb) {
        case 0:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb11;
          break;
        case 1:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb18;
          break;
        case 2:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb24;
          break;
        case 11:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb78;
          break;
        case 14:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb106;
          break;
        case 21:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb162;
          break;
        case 24:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb217;
          break;
        case 28:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb273;
          break;
        default:
          AssertFatal(0, "Failed: Check value for N_RB_DL/N_RB_UL"); // TODO: Add all values or function to convert
          break;
      }
    } else if (resp->info.mode == XNAP_MODE_TDD) { // TDD
      const xnap_tdd_info_t *tdd = &resp->info.tdd;
      servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_tdd;
      servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd = (XNAP_NRModeInfoTDD_t *)calloc(1, sizeof(XNAP_NRModeInfoTDD_t));
      servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.nrARFCN = tdd->freqinfo.arfcn;
      for (int j = 0; j < 1; j++) { // number of available bands = j = 1
        nrfreqbanditem = (XNAP_NRFrequencyBandItem_t *)calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
        nrfreqbanditem->nr_frequency_band = tdd->freqinfo.band;
        asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.frequencyBand_List.list,
                    nrfreqbanditem);
      }
      switch (tdd->tbw.scs) {
        case 15:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs15;
          break;
        case 30:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs30;
          break;
        case 60:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs60;
          break;
        case 120:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS = XNAP_NRSCS_scs120;
          break;
      }
      switch (tdd->tbw.nrb) {
        case 0:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb11;
          break;
        case 1:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb18;
          break;
        case 2:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb24;
          break;
        case 11:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb78;
          break;
        case 14:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb106;
          break;
        case 21:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb162;
          break;
        case 24:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb217;
          break;
        case 28:
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB = XNAP_NRNRB_nrb273;
          break;
        default:
          AssertFatal(0, "Failed: Check value for N_RB_DL/N_RB_UL"); // TODO: Add all values or function to convert.
  break;
      }
    }
    // Setting MTC to 0 now. Will be handled later.
    INT8_TO_OCTET_STRING(0, &servedCellMember->served_cell_info_NR.measurementTimingConfiguration);
    servedCellMember->served_cell_info_NR.connectivitySupport.eNDC_Support = 1;
    asn1cSeqAdd(&ie->value.choice.ServedCells_NR.list, servedCellMember);
  }
  asn1cSeqAdd(&out->protocolIEs.list, ie);

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP, "Failed to encode Xn setup response\n");
    return -1;
  }
  xnap_gNB_itti_send_sctp_data_req(assoc_id, buffer, len, 0);

  return ret;
}

//int xnap_gNB_generate_xn_handover_request (sctp_assoc_t assoc_id, instance_t instance,
 //                                          xnap_handover_req_t *xnap_handover_req, int ue_id)
 int xnap_gNB_generate_xn_handover_request (sctp_assoc_t assoc_id, xnap_handover_req_t *xnap_handover_req)                                         
{

  XNAP_XnAP_PDU_t                     pdu;
  XNAP_HandoverRequest_t              *xnhandoverreq;
  XNAP_HandoverRequest_IEs_t          *ie;
  instance_t  instance;

  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;
//get intance from tree if needed
//  DevAssert(instance != NULL); //not pointer
//  DevAssert(xnap_gNB_data_p != NULL);// not defined

 //  DevAssert(pdu != NULL);
   xnhandoverreq = &pdu.choice.initiatingMessage->value.choice.HandoverRequest;
    /* Send a xn setup failure with protocol cause unspecified */
/*    MessageDef *message_p = itti_alloc_new_message(TASK_XNAP, 0, XNAP_FAILURE_REQ);// failure????
    message_p->ittiMsgHeader.originInstance = assoc_id;
    itti_send_msg_to_task(TASK_XNAP, 0, message_p);       */
  
  LOG_D(XNAP, "Received a new XN setup request\n");

  /* Prepare the XnAP handover message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = XNAP_XnAP_PDU_PR_initiatingMessage;
  pdu.choice.initiatingMessage->procedureCode = XNAP_ProcedureCode_id_handoverPreparation;
  pdu.choice.initiatingMessage->criticality = XNAP_Criticality_reject;
  pdu.choice.initiatingMessage->value.present = XNAP_InitiatingMessage__value_PR_HandoverRequest;
  xnhandoverreq = &pdu.choice.initiatingMessage->value.choice.HandoverRequest;

  /* mandatory */
  ie = (XNAP_HandoverRequest_IEs_t *)calloc(1, sizeof(XNAP_HandoverRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_sourceNG_RANnodeUEXnAPID;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_HandoverRequest_IEs__value_PR_NG_RANnodeUEXnAPID;
 // ie->value.choice.UE_XnAP_ID = xnap_id_get_id_source(&instance_p->id_manager, ue_id);//// value to be added.
  asn1cSeqAdd(&xnhandoverreq->protocolIEs.list, ie);

  /* mandatory */
  ie = (XNAP_HandoverRequest_IEs_t *)calloc(1, sizeof(XNAP_HandoverRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_Cause;
  ie->criticality = XNAP_Criticality_ignore;
  ie->value.present = XNAP_HandoverRequest_IEs__value_PR_Cause;
  ie->value.choice.Cause.present = XNAP_Cause_PR_radioNetwork;
  ie->value.choice.Cause.choice.radioNetwork = 1; //Xnap_CauseRadioNetwork_handover_desirable_for_radio_reasons;
  asn1cSeqAdd(&xnhandoverreq->protocolIEs.list, ie);

  /* mandatory */
  ie = (XNAP_HandoverRequest_IEs_t *)calloc(1, sizeof(XNAP_HandoverRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_TargetCellCGI;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_HandoverRequest_IEs__value_PR_Target_CGI;
  ie->value.choice.Target_CGI.present = XNAP_Target_CGI_PR_nr;
  ie->value.choice.Target_CGI.choice.nr = (XNAP_NR_CGI_t *)calloc(1, sizeof(XNAP_NR_CGI_t));
  MCC_MNC_TO_PLMNID(xnap_handover_req->plmn_id.mcc,/// correct
                    xnap_handover_req->plmn_id.mnc,
                    xnap_handover_req->plmn_id.mnc_digit_length,
                    &ie->value.choice.Target_CGI.choice.nr->plmn_id); 
   NR_CELL_ID_TO_BIT_STRING(xnap_handover_req->target_cgi.cgi,
                               &ie->value.choice.Target_CGI.choice.nr->nr_CI); // bit string
   asn1cSeqAdd(&xnhandoverreq->protocolIEs.list, ie); 

  /* mandatory */
/**  ie = (XNAP_HandoverRequest_IEs_t *)calloc(1, sizeof(XNAP_HandoverRequest_IEs_t));
  ie->id = Xnap_ProtocolIE_ID_id_GUAMI;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = Xnap_HandoverRequest_IEs__value_PR_GUAMI;
  ie->value.choice.GUAMI.present = XNAP_Cause_PR_GUAMI;
  //ie->value.choice.GUAMI = (XNAP_GUAMI_t *)calloc(1, sizeof(XNAP_GUAMI_t));// what is the need? guami not pointer i think
  MCC_MNC_TO_PLMNID(req->info.plmn.mcc,
                    req->info.plmn.mnc,
                    req->info.plmn.mnc_digit_length,
                    &ie->value.choice.GUAMI.choice.plmn_ID);////handover req structure
  ie->value.choice.GUAMI.choice.amf_region_id = 1;//TODO: hardcoded
  ie->value.choice.GUAMI.choice.amf_set_id = 1;
  ie->value.choice.GUAMI.choice.amf_pointer = 1;
  asn1cSeqAdd(&out->protocolIEs.list, ie); **/

  /* mandatory */
/**  ie = (XNAP_HandoverRequest_IEs_t *)calloc(1, sizeof(XNAP_HandoverRequest_IEs_t));
  ie->id = Xnap_ProtocolIE_ID_id_UE_ContextInformation;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = Xnap_ProtocolIE_ID_id_UEContextInfoHORequest;
  //@TODO: consider to update this value
  ie->value.choice.UEContextInfoHORequest.ng_c_UE_reference = xnap_handover_req->ngc_ue_sig_ref; //what about other mandatory things in this IE? **/

  /* mandatory */
  /*ie = (Xnap_CPTransportLayerInformation_t *)calloc(4, sizeof(XNAP_HandoverRequest_IEs_t));
  cptlinfo->present = Xnap_CPTransportLayerInformation_PR_endpointIPAddress;
  cptlinfo->choice.endpointIPAddress.size = 4 ; //endian ness has to be verified here
  cptlinfo->value.choice.cp_TNL_info_source = xnap_handover_req->cp_TNL_info_source;*/ /// where is this storing now??

 // KGNB_STAR_TO_BIT_STRING(xnap_handover_req->kgnb,&ie->value.choice.UE_ContextInformation.aS_SecurityInformation.key_gNB_star); //function not there

/**  if (xnap_handover_req->kgnb_ncc >=0) { // Check this condition
    ie->value.choice.UEContextInfoHORequest.aS_SecurityInformation.nextHopChainingCount = xnap_handover_req->ue_context.as_security_ncc;
  }
  else {
    ie->value.choice.UEContextInfoHORequest.aS_SecurityInformation.nextHopChainingCount = 1;
  }   **/

   /* id-NRUESecurityCapabilities */
  /*XNAP_FIND_PROTOCOLIE_BY_ID(Xnap_HandoverRequest_IEs_t, ie, container,
                             XNAP_ProtocolIE_ID_id_NRUESecurityCapabilities, false);
  if (ie != NULL) {
    xnap_handover_req.nr_security_capabilities.encryption_algorithms =
      BIT_STRING_to_uint16(&ie->value.choice.NRUESecurityCapabilities.nRencryptionAlgorithms);
    xnap_handover_req.nr_security_capabilities.integrity_algorithms =
      BIT_STRING_to_uint16(&ie->value.choice.NRUESecurityCapabilities.nRintegrityProtectionAlgorithms);
  } else {
    xnap_handover_req.nr_security_capabilities.encryption_algorithms = 0;
    xnap_handover_req.nr_security_capabilities.integrity_algorithms = 0;
  }*/ //// why this is here???? not filling ie 


  OCTET_STRING_fromBuf(&ie->value.choice.UEContextInfoHORequest.rrc_Context, (char*) xnap_handover_req->ue_context.rrc_buffer, 8192);// added 8192- need to remove hard coding

  asn1cSeqAdd(&xnhandoverreq->protocolIEs.list, ie);

  /* mandatory */
/**  ie = (XNAP_HandoverRequest_IEs_t *)calloc(1, sizeof(XNAP_HandoverRequest_IEs_t));
  ie->id = Xnap_ProtocolIE_ID_id_UE_HistoryInformation;
  ie->criticality = XNAP_Criticality_ignore;
  ie->value.present = Xnap_HandoverRequest_IEs__value_PR_UE_HistoryInformation;
  //@TODO: consider to update this value
  {
   lastVisitedCell_Item = (XNAP_LastVisitedCell_Item *)calloc(1, sizeof(XNAP_LastVisitedCell_Item));
   lastVisitedCell_Item->present = XNAP_LastVisitedCell_Item_PR_nG_RAN_Cell;  **/
  /* MCC_MNC_TO_PLMNID(xnap_handover_req->plmn_id.mcc, xnap_handover_req->plmn_id.mnc, xnap_handover_req->plmn_id.mnc_digit_length,
                       &lastVisitedCell_Item->choice.nG_RAN_Cell.global_Cell_ID.pLMN_Identity);
   MACRO_ENB_ID_TO_CELL_IDENTITY(0, 0, &lastVisitedCell_Item->choice.e_UTRAN_Cell.global_Cell_ID.ng_RANcellIdentifier);
  lastVisitedCell_Item->choice.ngRAN_Cell.cellType.cell_Size = XNAP_Cell_Size_small;
   lastVisitedCell_Item->choice.ngRAN_Cell.time_UE_StayedInCell = 2;
   asn1cSeqAdd(&ie->value.choice.UE_HistoryInformation.list, lastVisitedCell_Item);*/ ///its octet string ryt??
 /** } **/

//  asn1cSeqAdd(&out->protocolIEs.list, ie);

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP,"Failed to encode XN handover request\n");
    return -1;
  }

  xnap_gNB_itti_send_sctp_data_req(assoc_id, buffer, len, 0); /////where do you get the assoc id from??

  return ret;
}

/** getting assoc-id- sharing this clip from F1AP
 * int f1ap_assoc_id(F1_t isCu, instance_t instanceP) {
  f1ap_setup_req_t *f1_inst=f1ap_req(isCu, instanceP);
  return f1_inst->assoc_id;
}
If method that we have implemented to fetch assoc_id is not correct, need to try this 

**/

/**
void rrc_gNB_process_HandoverPreparationInformation(//// why is this here? should be in RRC
     rrc_gNB_ue_context_t ue_context_p,
     uint8_t    *buffer,
     int        *size)
{
  memset(buffer, 0, 8192);
  char *ho_buf = (char *) buffer;
  int ho_size;
  ho_size = do_NRHandoverPreparation(ho_buf, 8192, ue_context_p.ue_context.UE_Capability_nr, ue_context_p.ue_context.UE_Capability_size);
  *size = ho_size;


} **/
