
#include "intertask_interface.h"

#include "xnap_common.h"
#include "xnap_gNB_task.h"
#include "xnap_gNB_generate_messages.h"
#include "xnap_gNB_encoder.h"
#include "xnap_gNB_decoder.h"
#include "XNAP_GlobalgNB-ID.h"

#include "xnap_gNB_itti_messaging.h"

#include "assertions.h"
#include "conversions.h"

int xnap_gNB_generate_xn_setup_request(xnap_gNB_instance_t *instance_p, xnap_gNB_data_t *xnap_gNB_data_p)
{
  XNAP_XnAP_PDU_t                     pdu;
  XNAP_XnSetupRequest_t              *out;
  XNAP_XnSetupRequest_IEs_t          *ie;
  //XNAP_PLMN_Identity_t               *e_BroadcastPLMN_ItemIE;;
  //XNAP_ServedCells_NR_Item_t   *ServedCells_NR_ItemIEs;
  //X2AP_GU_Group_ID_t                 *gu;
  //XNAP_BroadcastPLMNinTAISupport_Item_t   *e_BroadcastPLMNinTAISupport_ItemIE;
  //XNAP_TAISupport_Item_t      *TAISupport_ItemIEs;
  //XNAP_S_NSSAI_t           *e_S_NSSAI_ItemIE ;
  //GlobalAMF_Region_Information_t   *e_GlobalAMF_Region_Information_ItemIEs;
  //XNAP_NRFrequencyBandItem_t    *e_ulNRFrequencyBand_ItemIE;
  //XNAP_NRFrequencyBandItem_t    *e_dlNRFrequencyBand_ItemIE; 

  uint8_t  *buffer;
  uint32_t  len;
  int       ret = 0;

  DevAssert(instance_p != NULL);
  DevAssert(xnap_gNB_data_p != NULL);

  xnap_gNB_data_p->state = XNAP_GNB_STATE_WAITING;

  /* Prepare the X2AP message to encode */
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
  /*ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id.size=3;  //OCTET STRING (SIZE(3)) 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id.buf=calloc(ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id.size, sizeof(uint8_t)); 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id.buf[0]=208; 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id.buf[1]=95; 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id.buf[2]=2; */
  MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->plmn_id);
  
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.present=XNAP_GNB_ID_Choice_PR_gnb_ID;
  /*ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.size=4;  //BIT STRING (SIZE(22..32))
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf=calloc(1,ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.size); 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[0]=3; 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[1]=3; 
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[2]=1;
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[3]=1;
  ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.bits_unused=2;*/
  MACRO_GNB_ID_TO_BIT_STRING(instance_p->gNB_id, &ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID); //28 bits
  XNAP_INFO("%d -> %02x%02x%02x\n", instance_p->gNB_id,
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[0],
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[1],
            ie->value.choice.GlobalNG_RANNode_ID.choice.gNB->gnb_id.choice.gnb_ID.buf[2]);

  asn1cSeqAdd(&out->protocolIEs.list, ie);


  /* mandatory */ //TAI Support list
  /*ie = (XNAP_XnSetupRequest_IEs_t *)calloc(1, sizeof(XNAP_XnSetupRequest_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_TAISupport_list ;
  ie->criticality = XNAP_Criticality_reject;
  ie->value.present = XNAP_XnSetupRequest_IEs__value_PR_TAISupport_List;
	
  	//{
    	//for (int i=0;i<1;i++)
    		{
    		TAISupport_ItemIEs = (TAISupport_Item_t *)calloc(1,sizeof(TAISupport_Item_t));
      		/*TAISupport_ItemIEs->tac.size = 3;//octet string
      		
      		TAISupport_ItemIEs->tac.buf=calloc(TAISupport_ItemIEs->tac.size,sizeof(OCTET_STRING_t));
      		TAISupport_ItemIEs->tac.buf[0]=208;
      		TAISupport_ItemIEs->tac.buf[1]=95;
      		TAISupport_ItemIEs->tac.buf[2]=2;*/
      		/*INT24_TO_OCTET_STRING(instance_p->tac, &TAISupport_ItemIEs->tac);
		// NR_FIVEGS_TAC_ID_TO_BIT_STRING	
		{
		for (int j=0; j<1; j++) 
			{
			e_BroadcastPLMNinTAISupport_ItemIE = (BroadcastPLMNinTAISupport_Item_t *)calloc(1, sizeof(BroadcastPLMNinTAISupport_Item_t));
			/*e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.size = 3;
			//e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.buf=calloc(1,e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.size);
			e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.buf=calloc(e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.size,sizeof(OCTET_STRING_t));
			e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.buf[0]=208;
			e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.buf[1]=95;
			e_BroadcastPLMNinTAISupport_ItemIE->plmn_id.buf[2]=2;*/
			/*MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &e_BroadcastPLMNinTAISupport_ItemIE->plmn_id);
			
			{
			for (int k=0;k<1;k++)
				{
				e_S_NSSAI_ItemIE = (S_NSSAI_t *)calloc(1, sizeof(S_NSSAI_t));
				e_S_NSSAI_ItemIE->sst.size=1; //OCTET STRING(SIZE(1))
				//e_S_NSSAI_ItemIE->sst.buf=calloc(1,e_S_NSSAI_ItemIE->sst.size);
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
	
	asn1cSeqAdd(&out->protocolIEs.list, ie);*/


	/* mandatory */ //AMFRegion
/*  	ie = (XnSetupRequest_IEs_t *)calloc(1, sizeof(XnSetupRequest_IEs_t));
  	ie->id = ProtocolIE_ID_id_AMF_Region_Information ;
  	ie->criticality = Criticality_reject;
  	ie->value.present = XnSetupRequest_IEs__value_PR_AMF_Region_Information;
  	

  	//{
    	//for (int i=0;i<1;i++)
    		{
    		e_GlobalAMF_Region_Information_ItemIEs = (GlobalAMF_Region_Information_t *)calloc(1,sizeof(GlobalAMF_Region_Information_t));
      		/*e_GlobalAMF_Region_Information_ItemIEs->plmn_ID.size = 3;//octet string
      		e_GlobalAMF_Region_Information_ItemIEs->plmn_ID.buf=calloc(1,e_GlobalAMF_Region_Information_ItemIEs->plmn_ID.size);
      		e_GlobalAMF_Region_Information_ItemIEs->plmn_ID.buf[0]=208;
      		e_GlobalAMF_Region_Information_ItemIEs->plmn_ID.buf[1]=95;
      		e_GlobalAMF_Region_Information_ItemIEs->plmn_ID.buf[2]=2;*/ 
/*      	MCC_MNC_TO_PLMNID(instance_p->mcc, instance_p->mnc, instance_p->mnc_digit_length, &e_GlobalAMF_Region_Information_ItemIEs->plmn_ID);
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.size=1;
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.buf=calloc(1,e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.size);
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.buf[0]=6;
      		e_GlobalAMF_Region_Information_ItemIEs->amf_region_id.bits_unused=0;
			
    		asn1cSeqAdd(&ie->value.choice.AMF_Region_Information.list, e_GlobalAMF_Region_Information_ItemIEs);
  		}
  	//}
	
	asn1cSeqAdd(&out->protocolIEs.list, ie);
	
*/

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    XNAP_ERROR("Failed to encode X2 setup request\n");
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

  /* Prepare the X2AP message to encode */
  memset(&pdu, 0, sizeof(pdu));
  pdu.present = XNAP_XnAP_PDU_PR_unsuccessfulOutcome;
  pdu.choice.unsuccessfulOutcome->procedureCode = XNAP_ProcedureCode_id_xnSetup;
  pdu.choice.unsuccessfulOutcome->criticality = XNAP_Criticality_reject;
  pdu.choice.unsuccessfulOutcome->value.present = XNAP_UnsuccessfulOutcome__value_PR_XnSetupFailure;
  out = &pdu.choice.unsuccessfulOutcome->value.choice.XnSetupFailure;

  /* mandatory */
  ie = (XNAP_XnSetupFailure_IEs_t *)calloc(1, sizeof(XNAP_XnSetupFailure_IEs_t));
  ie->id = XNAP_ProtocolIE_ID_id_Cause;
  ie->criticality = XNAP_Criticality_ignore;
  ie->value.present = XNAP_XnSetupFailure_IEs__value_PR_Cause;

  //xnap_gNB_set_cause (&ie->value.choice.Cause, cause_type, cause_value);

  //asn1cSeqAdd(&out->protocolIEs.list, ie);

  if (xnap_gNB_encode_pdu(&pdu, &buffer, &len) < 0) {
    XNAP_ERROR("Failed to encode Xn setup failure\n");
    return -1;
  }

  xnap_gNB_itti_send_sctp_data_req(instance, assoc_id, buffer, len, 0);

  return ret;
}






