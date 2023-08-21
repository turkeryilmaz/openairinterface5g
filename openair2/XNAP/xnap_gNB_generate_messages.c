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
#include "xnap_gNB_encoder.h"
#include "xnap_gNB_decoder.h"
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
#include "XNAP_NRModeInfoFDD.h"
#include "XNAP_NRModeInfoTDD.h"

int xnap_gNB_generate_xn_setup_request(xnap_gNB_instance_t *instance_p, xnap_gNB_data_t *xnap_gNB_data_p)
{
  XNAP_XnAP_PDU_t                     pdu;
  XNAP_XnSetupRequest_t              *out;
  XNAP_XnSetupRequest_IEs_t          *ie;
  XNAP_BroadcastPLMNinTAISupport_Item_t   *e_BroadcastPLMNinTAISupport_ItemIE;
  XNAP_TAISupport_Item_t      *TAISupport_ItemIEs;
  XNAP_S_NSSAI_t           *e_S_NSSAI_ItemIE ;
  XNAP_GlobalAMF_Region_Information_t   *e_GlobalAMF_Region_Information_ItemIEs;
  XNAP_ServedCells_NR_Item_t  *servedCellMember;
 // XNAP_ServedCells_NR_t       *ServedCells_NR;
  XNAP_NRFrequencyBandItem_t  *nrfreqbanditemul;
  XNAP_NRFrequencyBandItem_t  *nrfreqbanditemdl;
  XNAP_NRFrequencyBandItem_t  *nrfreqbanditem;
  XNAP_PLMN_Identity_t        *plmn ;
  

  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  DevAssert(instance_p != NULL);
  DevAssert(xnap_gNB_data_p != NULL);

  xnap_gNB_data_p->state = XNAP_GNB_STATE_WAITING;

  /* Prepare the XnAP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  
  pdu.present = XNAP_XnAP_PDU_PR_initiatingMessage;
  //pdu.choice.initiatingMessage = &initiating_msg;
  pdu.choice.initiatingMessage 	= (XNAP_InitiatingMessage_t *) calloc(1, sizeof(XNAP_InitiatingMessage_t)); 
  pdu.choice.initiatingMessage->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.initiatingMessage->criticality = XNAP_Criticality_reject;
  pdu.choice.initiatingMessage->value.present = XNAP_InitiatingMessage__value_PR_XnSetupRequest;
 
  out = &pdu.choice.initiatingMessage->value.choice.XnSetupRequest;
  
  /* mandatory */
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_GlobalNG_RAN_node_ID ;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_GlobalNG_RANNode_ID;
  ie->value.choice.GlobalNG_RANNode_ID.present=XNAP_GlobalNG_RANNode_ID_PR_gNB;
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB =(XNAP_GlobalgNB_ID_t *)calloc(1,sizeof(XNAP_GlobalgNB_ID_t));
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id);
  
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.present=XNAP_GNB_ID_Choice_PR_gnb_ID;

  MACRO_GNB_ID_TO_BIT_STRING(instance_p->gNB_id, &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID); //28 bits
  LOG_I(XNAP, "%d -> %02x%02x%02x\n", instance_p->gNB_id,
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[0],
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[1],
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[2]);

  asn1cSeqAdd(&out->protocolIEs.list, ie);


  /* mandatory */ //TAI Support list
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_TAISupport_list ;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_TAISupport_List;
	
  	//{
    	//for (int i=0;i<1;i++)
    		{
    		TAISupport_ItemIEs = (XNAP_TAISupport_Item_t *)calloc(1,sizeof(XNAP_TAISupport_Item_t));
      		INT24_TO_OCTET_STRING(instance_p->tac, &TAISupport_ItemIEs->tac);	
		{
		for (int j=0; j<1; j++) 
			{
			e_BroadcastPLMNinTAISupport_ItemIE = (XNAP_BroadcastPLMNinTAISupport_Item_t *)calloc(1, sizeof(XNAP_BroadcastPLMNinTAISupport_Item_t));
			
			MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &e_BroadcastPLMNinTAISupport_ItemIE->plmn_id);
			
			{
			for (int k=0;k<1;k++)
				{
				e_S_NSSAI_ItemIE = (XNAP_S_NSSAI_t *)calloc(1, sizeof(XNAP_S_NSSAI_t));
				e_S_NSSAI_ItemIE->sst.size=1; //OCTET STRING(SIZE(1))
				e_S_NSSAI_ItemIE->sst.buf=calloc(e_S_NSSAI_ItemIE->sst.size,sizeof(OCTET_STRING_t));
				e_S_NSSAI_ItemIE->sst.buf[0]=1;
				
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
  ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_List_of_served_cells_NR;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_ServedCells_NR;
  {
      servedCellMember = (XNAP_ServedCells_NR_Item_t *)calloc(1,sizeof(XNAP_ServedCells_NR_Item_t));
      {
        servedCellMember->served_cell_info_NR.nrPCI = instance_p->Nid_cell; //long

        MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length,
                      &servedCellMember->served_cell_info_NR.cellID.plmn_id); //octet string
        NR_CELL_ID_TO_BIT_STRING(instance_p->gNB_id,
                                   &servedCellMember->served_cell_info_NR.cellID.nr_CI); //bit string

        INT24_TO_OCTET_STRING(instance_p->tac, &servedCellMember->served_cell_info_NR.tac); //octet string
        for (int k=0;k<1;k++)
        {
        	plmn = (XNAP_PLMN_Identity_t *)calloc(1,sizeof(XNAP_PLMN_Identity_t));
        	{
         	 MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, plmn);
          	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.broadcastPLMN.list, plmn);
        	}
	}
	if (instance_p->frame_type == FDD) 
	{
          servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_fdd;
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd=(XNAP_NRModeInfoFDD_t *)calloc(1,sizeof(XNAP_NRModeInfoFDD_t));
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.nrARFCN= instance_p->fdd_earfcn_UL;
	  for(int j=0;j<1;j++)
	  {
	  	nrfreqbanditemul=(XNAP_NRFrequencyBandItem_t *) calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
	  	nrfreqbanditemul->nr_frequency_band=78; //how to fill ?
	  	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list, nrfreqbanditemul);
	  }
	  
	  servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRFrequencyInfo.nrARFCN= instance_p->fdd_earfcn_DL;
	  for(int j=0;j<1;j++)
	  {
	  	nrfreqbanditemdl=(XNAP_NRFrequencyBandItem_t *) calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
	  	nrfreqbanditemdl->nr_frequency_band=78; //how to fill ?
	  	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list, nrfreqbanditemdl);
	  }
	  
	  switch (instance_p->nr_SCS) 
	  {
	  case 15:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs15;
          	break;  
          	
          case 30:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs30;
          	break;
          	
          case 60:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs60;
          	break;
          	
          case 120:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs120;
          	break;
          }
          
          
          switch (instance_p->N_RB_UL) 
	  {
	  case 11:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb11;
          	break;  
          	
          case 18:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb18;
          	break;
          	
          case 24:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb24;
          	break;
          	
          case 78:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb78;
          	break;
          	
          case 106:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb106;
          	break;
          	
          case 162:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb162;
          	break;
          case 217:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb217;
          	break;
          case 273:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb273;
          	break;
          default:
              AssertFatal(0,"Failed: Check value for N_RB_DL/N_RB_UL");
              break;
          }
          switch (instance_p->nr_SCS) 
	  {
	  case 15:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs15;
          	break;  
          	
          case 30:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs30;
          	break;
          	
          case 60:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs60;
          	break;
          	
          case 120:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs120;
          	break;
          }
          
          switch (instance_p->N_RB_DL) 
	  {
	  case 11:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb11;
          	break;  
          	
          case 18:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb18;
          	break;
          	
          case 24:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb24;
          	break;
          	
          case 78:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb78;
          	break;
          	
          case 106:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb106;
          	break;
          	
          case 162:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb162;
          	break;
          case 217:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb217;
          	break;
          case 273:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb273;
          	break;
          default:
              AssertFatal(0,"Failed: Check value for N_RB_DL/N_RB_UL");
              break;
          }
        }
        else
        {
          servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_tdd;
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd=(XNAP_NRModeInfoTDD_t *) calloc(1,sizeof(XNAP_NRModeInfoTDD_t));
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.nrARFCN= 640008;//instance_p->nrARFCN[i];
        for(int j=0;j<1;j++)
	  {
	  	nrfreqbanditem=(XNAP_NRFrequencyBandItem_t *) calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
	  	nrfreqbanditem->nr_frequency_band=106;//instance_p->nr_band; //how to fill ?
	  	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.frequencyBand_List.list, nrfreqbanditem);
	  }
	  switch (instance_p->nr_SCS) 
	  {
	  case 15:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs15;
          	break;  
          	
          case 30:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs30;
          	break;
          	
          case 60:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs60;
          	break;
          	
          case 120:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs120;
          	break;
          }
          
          
          switch (instance_p->N_RB_DL) 
	  {
	  case 11:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb11;
          	break;  
          	
          case 18:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb18;
          	break;
          	
          case 24:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb24;
          	break;
          	
          case 78:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb78;
          	break;
          	
          case 106:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb106;
          	break;
          	
          case 162:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb162;
          	break;
          case 217:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb217;
          	break;
          case 273:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb273;
          	break;
          default:
              AssertFatal(0,"Failed: Check value for N_RB_DL/N_RB_UL");
              break;
          }
        }
        //Setting MTC to 0 now. Will be handled later.
        INT8_TO_OCTET_STRING(0,&servedCellMember->served_cell_info_NR.measurementTimingConfiguration);
        servedCellMember->served_cell_info_NR.connectivitySupport.eNDC_Support=1;
      }
      asn1cSeqAdd(&ie->value.choice.ServedCells_NR.list, servedCellMember);
  }
  asn1cSeqAdd(&out->protocolIEs.list, ie);

	/* mandatory */ //AMFRegion
	ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  	ie->id = XNAP_ProtocolIE_ID_id_AMF_Region_Information ;
  	ie->criticality = XNAP_Criticality_reject;
  	ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_AMF_Region_Information;
  	//{
    	//for (int i=0;i<1;i++)
    		{
    		e_GlobalAMF_Region_Information_ItemIEs = (XNAP_GlobalAMF_Region_Information_t *)calloc(1,sizeof(XNAP_GlobalAMF_Region_Information_t));
      		
      	MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &e_GlobalAMF_Region_Information_ItemIEs->plmn_ID);
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.size=1;
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.buf=calloc(1,e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.size);
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.buf[0]=80;
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.bits_unused=0;
			
    		asn1cSeqAdd(&ie->value.choice.AMF_Region_Information.list, e_GlobalAMF_Region_Information_ItemIEs);
  		}
  	//}
	asn1cSeqAdd(&out->protocolIEs.list, ie);
	


  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP, "Failed to encode Xn setup request\n");
    return -1;
  }

  xnap_gNB_itti_send_sctp_data_req(instance_p->instance, xnap_gNB_data_p->assoc_id, buffer, len, 0);

  return ret;
}

int xnap_gNB_generate_xn_setup_failure(instance_t instance,
                                       uint32_t assoc_id,
                                       XNAP_Cause_PR cause_type,
                                       long cause_value,
                                       long time_to_wait)
{
  XNAP_XnAP_PDU_t                     pdu;
  XNAP_XnSetupFailure_t              *out;
  XNAP_XnSetupFailure_IEs_t          *ie;

  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  /* Prepare the XnAP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = XNAP_XnAP_PDU_PR_unsuccessfulOutcome;
  pdu.choice.unsuccessfulOutcome 	= (XNAP_UnsuccessfulOutcome_t *) calloc(1, sizeof(XNAP_UnsuccessfulOutcome_t)); 
  pdu.choice.unsuccessfulOutcome->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.unsuccessfulOutcome->criticality = XNAP_Criticality_reject;
  pdu.choice.unsuccessfulOutcome->value.present = XNAP_UnsuccessfulOutcome__value_PR_XnSetupFailure;
  out = &pdu.choice.unsuccessfulOutcome->value.choice.XnSetupFailure;

  /* mandatory */
  ie = (XNAP_XnSetupFailure_IEs_t *)calloc(1, sizeof(XNAP_XnSetupFailure_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_Cause;
  ie->criticality = XNAP_Criticality_ignore;
  ie->value.present = XNAP_XnSetupFailure_IEs__value_PR_Cause;

  xnap_gNB_set_cause (&ie->value.choice.Cause, cause_type, cause_value);

  asn1cSeqAdd(&out->protocolIEs.list, ie);

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP, "Failed to encode Xn setup failure\n");
    return -1;
  }

  xnap_gNB_itti_send_sctp_data_req(instance, assoc_id, buffer, len, 0);

  return ret;
}

int xnap_gNB_set_cause (XNAP_Cause_t * cause_p,
                        XNAP_Cause_PR cause_type,
                        long cause_value)
{

  DevAssert (cause_p != NULL);
  cause_p->present = cause_type;

  switch (cause_type) {
  case XNAP_Cause_PR_radioNetwork:
    cause_p->choice.misc = cause_value;
    break;

  case XNAP_Cause_PR_transport:
    cause_p->choice.misc = cause_value;
    break;

  case XNAP_Cause_PR_protocol:
    cause_p->choice.misc = cause_value;
    break;

  case XNAP_Cause_PR_misc:
    cause_p->choice.misc = cause_value;
    break;

  default:
    return -1;
  }

  return 0;
}

int xnap_gNB_generate_xn_setup_response(xnap_gNB_instance_t *instance_p, xnap_gNB_data_t *xnap_gNB_data_p)
{
  XNAP_XnAP_PDU_t                     pdu;
  XNAP_XnSetupResponse_t              *out;
  XNAP_XnSetupResponse_IEs_t          *ie;
  XNAP_PLMN_Identity_t                *plmn;
  XNAP_BroadcastPLMNinTAISupport_Item_t   *e_BroadcastPLMNinTAISupport_ItemIE;
  XNAP_TAISupport_Item_t      *TAISupport_ItemIEs;
  XNAP_S_NSSAI_t           *e_S_NSSAI_ItemIE ;
  //XNAP_GlobalAMF_Region_Information_t   *e_GlobalAMF_Region_Information_ItemIEs;
  XNAP_ServedCells_NR_Item_t  *servedCellMember;
  //XNAP_ServedCells_NR_t       *ServedCells_NR;
  XNAP_NRFrequencyBandItem_t  *nrfreqbanditemul;
  XNAP_NRFrequencyBandItem_t  *nrfreqbanditemdl;
  XNAP_NRFrequencyBandItem_t  *nrfreqbanditem;

  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  DevAssert(instance_p != NULL);
  DevAssert(xnap_gNB_data_p != NULL);

  /* Prepare the XNAP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = XNAP_XnAP_PDU_PR_successfulOutcome;
  pdu.choice.successfulOutcome 	= (XNAP_SuccessfulOutcome_t *) calloc(1, sizeof(XNAP_SuccessfulOutcome_t)); 
  pdu.choice.successfulOutcome->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.successfulOutcome->criticality = XNAP_Criticality_reject;
  pdu.choice.successfulOutcome->value.present = XNAP_SuccessfulOutcome__value_PR_XnSetupResponse;
  out = &pdu.choice.successfulOutcome->value.choice.XnSetupResponse;

  /* mandatory */
  ie = (XNAP_XnSetupResponse_IEs_t *)calloc(1, sizeof(XNAP_XnSetupResponse_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_GlobalNG_RAN_node_ID;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupResponse_IEs__value_PR_GlobalNG_RANNode_ID;
  ie->value.choice.GlobalNG_RANNode_ID.present=XNAP_GlobalNG_RANNode_ID_PR_gNB;
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB =(XNAP_GlobalgNB_ID_t *)calloc(1,sizeof(XNAP_GlobalgNB_ID_t));
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length,
                    &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id);
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.present=XNAP_GNB_ID_Choice_PR_gnb_ID;
  MACRO_GNB_ID_TO_BIT_STRING(instance_p->gNB_id,
                             &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID);
  LOG_I(XNAP, "%d -> %02x%02x%02x\n", instance_p->gNB_id,
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[0],
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[1],
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[2]);
  asn1cSeqAdd(&out->protocolIEs.list, ie);

 /* mandatory */ //TAI Support list
  ie = (XNAP_XnSetupResponse_IEs_t *)calloc(1, sizeof(XNAP_XnSetupResponse_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_TAISupport_list ;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupResponse_IEs__value_PR_TAISupport_List;
	
  	//{
    	//for (int i=0;i<1;i++)
    		{
    		TAISupport_ItemIEs = (XNAP_TAISupport_Item_t *)calloc(1,sizeof(XNAP_TAISupport_Item_t));
      		INT24_TO_OCTET_STRING(instance_p->tac, &TAISupport_ItemIEs->tac);
		{
		for (int j=0; j<1; j++) 
			{
			e_BroadcastPLMNinTAISupport_ItemIE = (XNAP_BroadcastPLMNinTAISupport_Item_t *)calloc(1, sizeof(XNAP_BroadcastPLMNinTAISupport_Item_t));
			
			MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &e_BroadcastPLMNinTAISupport_ItemIE->plmn_id);
			
			{
			for (int k=0;k<1;k++)
				{
				e_S_NSSAI_ItemIE = (XNAP_S_NSSAI_t *)calloc(1, sizeof(XNAP_S_NSSAI_t));
				e_S_NSSAI_ItemIE->sst.size=1; //OCTET STRING(SIZE(1))
				e_S_NSSAI_ItemIE->sst.buf=calloc(e_S_NSSAI_ItemIE->sst.size,sizeof(OCTET_STRING_t));
				e_S_NSSAI_ItemIE->sst.buf[0]=1;
				
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
      servedCellMember = (XNAP_ServedCells_NR_Item_t *)calloc(1,sizeof(XNAP_ServedCells_NR_Item_t));
      {
        servedCellMember->served_cell_info_NR.nrPCI = instance_p->Nid_cell; //long

        MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length,
                      &servedCellMember->served_cell_info_NR.cellID.plmn_id); //octet string
        NR_CELL_ID_TO_BIT_STRING(instance_p->gNB_id,
                                   &servedCellMember->served_cell_info_NR.cellID.nr_CI); //bit string

        INT24_TO_OCTET_STRING(instance_p->tac, &servedCellMember->served_cell_info_NR.tac); //octet string
        for (int k=0;k<1;k++)
        {
        	plmn = (XNAP_PLMN_Identity_t *)calloc(1,sizeof(XNAP_PLMN_Identity_t));
        	{
         	 MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, plmn);
          	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.broadcastPLMN.list, plmn);
        	}
	}
	if (instance_p->frame_type == FDD) 
	{
          servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_fdd;
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd=(XNAP_NRModeInfoFDD_t *)calloc(1,sizeof(XNAP_NRModeInfoFDD_t));
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.nrARFCN= instance_p->fdd_earfcn_UL;
	  for(int j=0;j<1;j++)
	  {
	  	nrfreqbanditemul=(XNAP_NRFrequencyBandItem_t *) calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
	  	nrfreqbanditemul->nr_frequency_band=78; //how to fill ?
	  	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list, nrfreqbanditemul);
	  }
	  
	  servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRFrequencyInfo.nrARFCN= instance_p->fdd_earfcn_DL;
	  for(int j=0;j<1;j++)
	  {
	  	nrfreqbanditemdl=(XNAP_NRFrequencyBandItem_t *) calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
	  	nrfreqbanditemdl->nr_frequency_band=78; //how to fill ?
	  	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRFrequencyInfo.frequencyBand_List.list, nrfreqbanditemdl);
	  }
	  
	  switch (instance_p->nr_SCS) 
	  {
	  case 15:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs15;
          	break;  
          	
          case 30:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs30;
          	break;
          	
          case 60:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs60;
          	break;
          	
          case 120:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs120;
          	break;
          }
          
          
          switch (instance_p->N_RB_UL) 
	  {
	  case 11:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb11;
          	break;  
          	
          case 18:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb18;
          	break;
          	
          case 24:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb24;
          	break;
          	
          case 78:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb78;
          	break;
          	
          case 106:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb106;
          	break;
          	
          case 162:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb162;
          	break;
          case 217:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb217;
          	break;
          case 273:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->ulNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb273;
          	break;
          default:
              AssertFatal(0,"Failed: Check value for N_RB_DL/N_RB_UL");
              break;
          }
          switch (instance_p->nr_SCS) 
	  {
	  case 15:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs15;
          	break;  
          	
          case 30:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs30;
          	break;
          	
          case 60:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs60;
          	break;
          	
          case 120:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs120;
          	break;
          }
          
          switch (instance_p->N_RB_DL) 
	  {
	  case 11:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb11;
          	break;  
          	
          case 18:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb18;
          	break;
          	
          case 24:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb24;
          	break;
          	
          case 78:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb78;
          	break;
          	
          case 106:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb106;
          	break;
          	
          case 162:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb162;
          	break;
          case 217:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb217;
          	break;
          case 273:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.fdd->dlNRTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb273;
          	break;
          default:
              AssertFatal(0,"Failed: Check value for N_RB_DL/N_RB_UL");
              break;
          }
        }
        else
        {
          servedCellMember->served_cell_info_NR.nrModeInfo.present = XNAP_NRModeInfo_PR_tdd;
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd=(XNAP_NRModeInfoTDD_t *) calloc(1,sizeof(XNAP_NRModeInfoTDD_t));
          servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.nrARFCN= 640008;//instance_p->nrARFCN[i];
        for(int j=0;j<1;j++)
	  {
	  	nrfreqbanditem=(XNAP_NRFrequencyBandItem_t *) calloc(1, sizeof(XNAP_NRFrequencyBandItem_t));
	  	nrfreqbanditem->nr_frequency_band=106;//instance_p->nr_band; //how to fill ?
	  	asn1cSeqAdd(&servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrFrequencyInfo.frequencyBand_List.list, nrfreqbanditem);
	  }
	  switch (instance_p->nr_SCS) 
	  {
	  case 15:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs15;
          	break;  
          	
          case 30:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs30;
          	break;
          	
          case 60:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs60;
          	break;
          	
          case 120:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRSCS =XNAP_NRSCS_scs120;
          	break;
          }
          
          
          switch (instance_p->N_RB_DL) 
	  {
	  case 11:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb11;
          	break;  
          	
          case 18:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb18;
          	break;
          	
          case 24:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb24;
          	break;
          	
          case 78:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb78;
          	break;
          	
          case 106:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb106;
          	break;
          	
          case 162:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb162;
          	break;
          case 217:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb217;
          	break;
          case 273:
          	servedCellMember->served_cell_info_NR.nrModeInfo.choice.tdd->nrTransmissonBandwidth.nRNRB =XNAP_NRNRB_nrb273;
          	break;
          default:
              AssertFatal(0,"Failed: Check value for N_RB_DL/N_RB_UL");
              break;
          }
        }
        //Setting MTC to 0 now. Will be handled later.
        INT8_TO_OCTET_STRING(0,&servedCellMember->served_cell_info_NR.measurementTimingConfiguration);
        servedCellMember->served_cell_info_NR.connectivitySupport.eNDC_Support=1;
      }
      asn1cSeqAdd(&ie->value.choice.ServedCells_NR.list, servedCellMember);
  }
  asn1cSeqAdd(&out->protocolIEs.list, ie);


  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    LOG_E(XNAP, "Failed to encode Xn setup response\n");
    return -1;
  }
  xnap_gNB_data_p->state = XNAP_GNB_STATE_READY;
  xnap_gNB_itti_send_sctp_data_req(instance_p->instance, xnap_gNB_data_p->assoc_id, buffer, len, 0);

  return ret;
}







