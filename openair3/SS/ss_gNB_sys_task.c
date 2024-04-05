#include <pthread.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <sys/ioctl.h>
#include <net/if.h>

#include <netinet/in.h>
#include <netinet/sctp.h>

#include <arpa/inet.h>

#include "assertions.h"
#include "common/utils/system.h"
#include "queue.h"
#include "sctp_common.h"

#include "intertask_interface.h"
#include "common/ran_context.h"

#include "acpNrSys.h"
#include "gnb_config.h"
#include "ss_gNB_sys_task.h"
#include "ss_gNB_context.h"
#include "ss_gNB_proxy_iface.h"

#include "openair2/LAYER2/nr_pdcp/nr_pdcp_entity.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_ue_manager.h"

#include "constr_TYPE.h"
#include "OCTET_STRING.h"
#include "NR_DL-CCCH-Message.h"

#include "common/utils/LOG/ss-log.h"
#include "ss_utils.h"
#include "softmodem-common.h"
#include "../../openair2/LAYER2/NR_MAC_gNB/nr_mac_gNB.h"
#include "ss_gNB_multicell_helper.h"

#include "SidlCommon_NR_RachProcedureConfig_Type.h"


extern RAN_CONTEXT_t RC;
extern SSConfigContext_t SS_context;

extern pthread_cond_t cell_config_5G_done_cond;
extern pthread_mutex_t cell_config_5G_done_mutex;

static int sys_send_udp_msg(uint8_t *buffer, uint32_t buffer_len, uint32_t buffer_offset, uint32_t peerIpAddr, uint16_t peerPort);
char *local_5G_address = "127.0.0.1" ;
extern softmodem_params_t *get_softmodem_params(void);

static uint16_t paging_ue_index_g = 0;

typedef enum
{
  UndefinedMsg = 0,
  EnquireTiming = 1,
  CellConfig = 2
} sidl_msg_id;

int nr_cell_index = 0;
static bool reqCnfFlag_g = false;
static void ss_task_sys_nr_handle_deltaValues(struct NR_SYSTEM_CTRL_REQ *req);
int cell_config_5G_done=-1;
int cell_config_5G_done_indication();
bool ss_task_sys_nr_handle_cellConfig5G (struct NR_CellConfigRequest_Type *p_req,int cell_State);
bool ss_task_sys_nr_handle_cellConfigRadioBearer(struct NR_SYSTEM_CTRL_REQ *req);
bool ss_task_sys_nr_handle_cellConfigAttenuation(struct NR_SYSTEM_CTRL_REQ *req);
static int sys_5G_send_init_udp(const udpSockReq_t *req);
static void sys_5G_send_proxy(void *msg, int msgLen);
int proxy_5G_send_port = 7776;
int proxy_5G_recv_port = 7770;
bool ss_task_sys_nr_handle_pdcpCount(struct NR_SYSTEM_CTRL_REQ *req);


/*
 * Utility function to convert integer to binary
 *
 */
static void int_to_bin(uint32_t in, int count, uint8_t *out)
{
  /* assert: count <= sizeof(int)*CHAR_BIT */
  uint32_t mask = 1U << (count - 1);
  int i;
  for (i = 0; i < count; i++)
  {
    out[i] = (in & mask) ? 1 : 0;
    in <<= 1;
  }
}

static void process_RachProcedureMsg4RrcMsg(const struct NR_RachProcedureMsg4RrcMsg_Type *rrcMsg)
{
  switch (rrcMsg->d) {
    case NR_RachProcedureMsg4RrcMsg_Type_RrcCcchMsg:
      {
        NR_DL_CCCH_Message_t *dl_ccch_msg = NULL;
        asn_dec_rval_t dec_rval = uper_decode(NULL, &asn_DEF_NR_DL_CCCH_Message, (void **)&dl_ccch_msg, rrcMsg->v.RrcCcchMsg.v, rrcMsg->v.RrcCcchMsg.d, 0, 0);
        AssertFatal(dec_rval.code == RC_OK, "Failed to decode DL-CCCH-Message");

        OCTET_STRING_t *masterCellGroup = NULL;

        if (dl_ccch_msg->message.present == NR_DL_CCCH_MessageType_PR_c1 &&
            dl_ccch_msg->message.choice.c1->present == NR_DL_CCCH_MessageType__c1_PR_rrcSetup)
        {
          NR_RRCSetup_t *rrcSetup = dl_ccch_msg->message.choice.c1->choice.rrcSetup;
          if (rrcSetup->criticalExtensions.present == NR_RRCSetup__criticalExtensions_PR_rrcSetup)
          {
            masterCellGroup = &rrcSetup->criticalExtensions.choice.rrcSetup->masterCellGroup;
          }
        }

        if (masterCellGroup != NULL)
        {
          LOG_A(GNB_APP, "[SYS-GNB] Store CellGroupConfig for RrcSetup\n");
          if (RC.cellGroupConfig != NULL)
          {
            ASN_STRUCT_FREE(asn_DEF_OCTET_STRING, RC.cellGroupConfig);
          }
          RC.cellGroupConfig = OCTET_STRING_new_fromBuf(&asn_DEF_OCTET_STRING, (const char *)masterCellGroup->buf, masterCellGroup->size);
          AssertFatal(RC.cellGroupConfig != NULL, "Allocation for cellGroupConfig failed");
        }
        else
        {
          LOG_E(GNB_APP, "%s: masterCellGroup not found\n", __FUNCTION__);
        }

        ASN_STRUCT_FREE(asn_DEF_NR_DL_CCCH_Message, dl_ccch_msg);

        break;
      }
    default:
      {
        LOG_W(GNB_APP, "RachProcedure Msg4RrcMsg not handled\n");
        break;
      }
  }
}

/*
 * Function : send_sys_cnf
 * Description: Funtion to build and send the SYS_CNF
 * In :
 * resType - Result type of the requested command
 * resVal  - Result value Success/Fail for the command
 * cnfType - Confirmation type for the Request received
 *           needed by TTCN to map to the Request sent.
 */
static void send_sys_cnf(enum ConfirmationResult_Type_Sel resType,
                         bool resVal,
                         enum NR_SystemConfirm_Type_Sel cnfType,
                         void *msg)
{
  LOG_A(GNB_APP, "[SYS-GNB] Entry in fxn:%s\n", __FUNCTION__);
  struct NR_SYSTEM_CTRL_CNF *msgCnf = CALLOC(1, sizeof(struct NR_SYSTEM_CTRL_CNF));
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, SS_NR_SYS_PORT_MSG_CNF);
  assert(message_p);

  msgCnf->Common.CellId =  SS_context.SSCell_list[nr_cell_index].nr_cellId;
  msgCnf->Common.Result.d = resType;
  msgCnf->Common.Result.v.Success = resVal;
  msgCnf->Confirm.d = cnfType;
  switch (cnfType)
  {
    case NR_SystemConfirm_Type_Cell:
      {
        LOG_A(GNB_APP, "[SYS-GNB] Send confirm for cell configuration\n");
        msgCnf->Confirm.v.Cell = true;
      } break;
    case NR_SystemConfirm_Type_CellAttenuationList:
      {
        LOG_A(GNB_APP, "[SYS-GNB] Send confirm for cell configuration\n");
        msgCnf->Common.CellId = 0;
        msgCnf->Confirm.v.CellAttenuationList = true;
      } break;
    case NR_SystemConfirm_Type_PdcpCount:
      {
        if (msg)
        {
          LOG_A(GNB_APP, "[SYS-GNB] Send confirm for NR_SystemConfirm_Type_PdcpCount\n");
          memcpy(&msgCnf->Confirm.v.PdcpCount, msg, sizeof(struct NR_PDCP_CountCnf_Type));
        }
      } break;
    case NR_SystemConfirm_Type_AS_Security:
      {
        LOG_A(GNB_APP, "[SYS-GNB] Send confirm for cell configuration NR_SystemConfirm_Type_AS_Security\n");
        msgCnf->Confirm.v.AS_Security = true;
        break;
      }
    case NR_SystemConfirm_Type_DeltaValues:
      {
        LOG_A(GNB_APP, "[SYS-GNB] Send confirmation for 'DeltaValues'\n");
        if (msg)
        {
          memcpy(&msgCnf->Confirm.v.DeltaValues, msg, sizeof(struct UE_NR_DeltaValues_Type));
        }
      }; break;
    case NR_SystemConfirm_Type_RadioBearerList:
      {
        LOG_A(GNB_APP, "[SYS-GNB] Send confirmation for 'RadioBearerList'\n");
        msgCnf->Confirm.v.RadioBearerList = true;
      }; break;
    default:
      LOG_A(GNB_APP, "[SYS-GNB] Error not handled CNF TYPE to [SS-PORTMAN-GNB]\n");
  }

  SS_NR_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
  int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT, message_p);
  if (send_res < 0)
  {
    LOG_A(GNB_APP, "[SYS-GNB] Error sending to [SS-PORTMAN-GNB]\n");
  }
  else
  {
    LOG_A(GNB_APP, "[SYS-GNB] fxn:%s NR_SYSTEM_CTRL_CNF sent for cnfType:%d to Port Manager\n", __FUNCTION__, cnfType);
  }

  LOG_D(GNB_APP, "[SYS-GNB] Exit from fxn:%s\n", __FUNCTION__);
}
/*
 * Function : sys_handle_nr_enquire_timing
 * Description: Sends the NR enquire timing update to PORTMAN
 */
static void sys_handle_nr_enquire_timing(ss_nrset_timinfo_t *tinfo)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, SS_NRSET_TIM_INFO);
  if (message_p)
  {
    LOG_A(GNB_APP, "[SYS-GNB] Reporting info sfn:%d\t slot:%d.\n", tinfo->sfn, tinfo->slot);
    SS_NRSET_TIM_INFO(message_p).slot = tinfo->slot;
    SS_NRSET_TIM_INFO(message_p).sfn = tinfo->sfn;

    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(GNB_APP, "[SYS-GNB] Error sending to [SS-PORTMAN-GNB]");
    }
  }
}

/*
 * Function : sys_nr_cell_attn_update
 * Description: Sends the attenuation updates received from TTCN to proxy
 */
static void sys_nr_cell_attn_update(uint8_t cellId, uint8_t attnVal,int nr_cell_index)
{
  LOG_A(GNB_APP, "In sys_nr_cell_attn_update\n");
  attenuationConfigReq_t *attnConf = NULL;

  attnConf = (attenuationConfigReq_t *) calloc(1, sizeof(attenuationConfigReq_t));
  attnConf->header.preamble = 0xFEEDC0DE;
  attnConf->header.msg_id = SS_ATTN_LIST;
  attnConf->header.cell_id = cellId;
  attnConf->attnVal = attnVal;
  attnConf->header.cell_index = nr_cell_index;

  LOG_A(ENB_SS,"5G Cell Attenuation received for cell_id: %d AttnValue : %d cell_index: %d\n",attnConf->header.cell_id,attnConf->attnVal,attnConf->header.cell_index);
  /** Send to proxy */
  sys_5G_send_proxy((void *)attnConf, sizeof(attenuationConfigReq_t));
  LOG_A(GNB_APP, "Out sys_nr_cell_attn_update\n");
  return;
}

/*
 * Function : sys_handle_nr_cell_attn_req
 * Description: Handles the attenuation updates received from TTCN
 */
static void sys_handle_nr_cell_attn_req(struct NR_CellAttenuationConfig_Type_NR_CellAttenuationList_Type_Dynamic *CellAttenuationList)
{
  for(int i=0;i<CellAttenuationList->d;i++) {
    uint8_t cellId = (uint8_t)CellAttenuationList->v[i].CellId;
    uint8_t attnVal = 0; // default set it Off
    uint8_t NrCellIndex = get_cell_index(cellId, SS_context.SSCell_list);
    switch (CellAttenuationList->v[i].Attenuation.d)
    {
    case Attenuation_Type_Value:
      attnVal = CellAttenuationList->v[i].Attenuation.v.Value;
      LOG_A(GNB_APP, "[SYS-GNB] CellAttenuationList for Cell_id %d value %d dBm received\n",
            cellId, attnVal);
      sys_nr_cell_attn_update(cellId, attnVal, NrCellIndex);
      break;
    case Attenuation_Type_Off:
      attnVal = 53; /* TODO: attnVal hardcoded currently but Need to handle proper Attenuation_Type_Off */
      LOG_A(GNB_APP, "[SYS-GNB] CellAttenuationList turn off for Cell_id %d received with attnVal : %d\n",
            cellId,attnVal);
      sys_nr_cell_attn_update(cellId, attnVal, NrCellIndex);
      break;
    case Attenuation_Type_UNBOUND_VALUE:
      LOG_A(GNB_APP, "[SYS-GNB] CellAttenuationList Attenuation_Type_UNBOUND_VALUE received\n");
      break;
    default:
      LOG_A(GNB_APP, "[SYS-GNB] Invalid CellAttenuationList received\n");
    }
  }
}

/*
 * Function : sys_handle_nr_as_security_req
 * Description: Funtion handler of SYS_PORT. Handles the AS
 * Security command received from TTCN via the PORTMAN.
 * In :
 * req  - AS Security Request received from the TTCN via PORTMAN
 * Out:
 * newState: No impact on state machine.
 *
 */
static void sys_handle_nr_as_security_req(struct NR_AS_Security_Type *ASSecurity, int CC_id)
{
  MessageDef *msg_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, RRC_AS_SECURITY_CONFIG_REQ);
  if(msg_p)
  {
    LOG_I(GNB_APP,"[SYS-GNB] AS Security Request Received\n");
    RRC_AS_SECURITY_CONFIG_REQ(msg_p).rnti = SS_context.SSCell_list[CC_id].ss_rnti_g;

    switch(ASSecurity->d) {
    case NR_AS_Security_Type_StartRestart:
    {
      if(ASSecurity->v.StartRestart.Integrity.d == true)
      {
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).isIntegrityInfoPresent = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.integrity_algorithm = ASSecurity->v.StartRestart.Integrity.v.Algorithm;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint,0,16);
        bits_copy_from_array(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint, 0, ASSecurity->v.StartRestart.Integrity.v.KRRCint, 128);
        LOG_I(GNB_APP, "[SYS-GNB] kRRCint:\n");
        for(int i = 0; i < 16; i++) {
          LOG_I(GNB_APP, "%02x\n", RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint[i]);
        }

        if (ASSecurity->v.StartRestart.Integrity.v.KUPint.d == true) {
          RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.isUPIntegrityInfoPresent = true;
          RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kUPint = CALLOC(1, 16);
          memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kUPint, 0, 16);
          bits_copy_from_array(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kUPint, 0, ASSecurity->v.StartRestart.Integrity.v.KUPint.v, 128);

          LOG_I(GNB_APP, "[SYS-GNB] kUPint:\n");
          for(int i = 0; i < 16; i++) {
            LOG_I(GNB_APP, "%02x\n", RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kUPint[i]);
          }
        }

        if(ASSecurity->v.StartRestart.Integrity.v.ActTimeList.d == true)
        {
          for(int i=0;i < ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.d; i++)
          {
            switch(ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].RadioBearerId.d)
            {
              case RadioBearerId_Type_Srb:
                RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].RadioBearerId.v.Srb;
                break;
              case RadioBearerId_Type_Drb:
                RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].RadioBearerId.v.Drb;
                break;
              case RadioBearerId_Type_UNBOUND_VALUE:
                break;
              default:
              LOG_E(GNB_APP, "[SYS-GNB] AS Security Act time list is Invalid \n");
            }
            if (ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].UL.d == NR_PDCP_ActTime_Type_SQN)
            {
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].UL.format = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].UL.v.SQN.Format;
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].UL.sqn = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].UL.v.SQN.Value;
            }
            if (ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].DL.d == NR_PDCP_ActTime_Type_SQN)
            {
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].DL.format = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].DL.v.SQN.Format;
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].DL.sqn = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].DL.v.SQN.Value;
            }
          }
        }
      }

      if(ASSecurity->v.StartRestart.Ciphering.d == true)
      {
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).isCipheringInfoPresent = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ciphering_algorithm = ASSecurity->v.StartRestart.Ciphering.v.Algorithm;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc,0,16);
        bits_copy_from_array(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc, 0, ASSecurity->v.StartRestart.Ciphering.v.KRRCenc, 128);
        LOG_E(GNB_APP, "[SYS-GNB] kRRCenc:\n");
        for(int i = 0; i < 16; i++) {
          LOG_E(GNB_APP, "%02x\n", RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc[i]);
        }

        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc,0,16);
        bits_copy_from_array(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc, 0, ASSecurity->v.StartRestart.Ciphering.v.KUPenc, 128);
        LOG_E(GNB_APP, "[SYS-GNB] kUPenc:\n");
        for(int i = 0; i < 16; i++) {
          LOG_E(GNB_APP, "%02x\n", RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc[i]);
        }

        for(int i=0;i < ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.d; i++)
        {
          switch(ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].RadioBearerId.d)
          {
            case RadioBearerId_Type_Srb:
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].RadioBearerId.v.Srb;
              break;
            case RadioBearerId_Type_Drb:
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].RadioBearerId.v.Drb;
              break;
            case RadioBearerId_Type_UNBOUND_VALUE:
              break;
            default:
            LOG_E(GNB_APP, "[SYS-GNB] AS Security Act time list is Invalid \n");
          }
          if (ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].UL.d == NR_PDCP_ActTime_Type_SQN)
          {
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].UL.format = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].UL.v.SQN.Format;
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].UL.sqn = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].UL.v.SQN.Value;
          }
          if (ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].DL.d == NR_PDCP_ActTime_Type_SQN)
          {
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].DL.format = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].DL.v.SQN.Format;
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].DL.sqn = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].DL.v.SQN.Value;
          }
        }
      }
    }
    break;

    case NR_AS_Security_Type_Release:
    {
      // TODO
    }
    break;

    default:
      LOG_E(GNB_APP, "[TASK_SYS_GNB] unhandled message type %d\n", ASSecurity->d);
      return;
    }

    int send_res = itti_send_msg_to_task(TASK_RRC_GNB, INSTANCE_DEFAULT, msg_p);
    if (send_res < 0) {
      LOG_E(GNB_APP, "[SYS-GNB] Error sending RRC_AS_SECURITY_CONFIG_REQ to RRC_gNB task");
    }
  }
}

/*
 * Function : sys_handle_nr_paging_req
 * Description: Handles the attenuation updates received from TTCN
 */
static void sys_handle_nr_paging_req(struct NR_PagingTrigger_Type *pagingRequest, ss_nrset_timinfo_t tinfo)
{
    LOG_A(GNB_APP, "[SYS-GNB] Enter sys_handle_nr_paging_req Paging_IND for processing\n");

    /** TODO: Considering only one cell for now */
    uint8_t cellId = 0; //(uint8_t)pagingRequst->CellId;

    static uint8_t oneTimeProcessingFlag = 0;
    MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, 0, SS_SS_NR_PAGING_IND);
    if (message_p == NULL)
    {
        return;
    }

    SS_NR_PAGING_IND(message_p).cell_index = 0; // TODO: change to multicell index later
    SS_NR_PAGING_IND(message_p).sfn = tinfo.sfn;
    SS_NR_PAGING_IND(message_p).slot = tinfo.slot;
    SS_NR_PAGING_IND(message_p).paging_recordList = NULL;
    SS_NR_PAGING_IND(message_p).bSlotOffsetListPresent = false;

    switch (pagingRequest->Paging.message.d)
    {
        case SQN_NR_PCCH_MessageType_c1:
            if (pagingRequest->Paging.message.v.c1.d)
            {
                if (pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.d)
                {
                    struct SQN_NR_PagingRecord *p_sdl_msg = NULL;
                    p_sdl_msg = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v;
                    uint8_t numPagingRecord = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.d;
                    size_t pgSize = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.d * sizeof(ss_nr_paging_identity_t);
                    SS_NR_PAGING_IND(message_p).paging_recordList = CALLOC(1, pgSize);
                    ss_nr_paging_identity_t *p_record_msg = SS_NR_PAGING_IND(message_p).paging_recordList;
                    SS_NR_PAGING_IND(message_p).num_paging_record = numPagingRecord;
                    for (int count = 0; count < numPagingRecord; count++)
                    {
                        p_record_msg->bAccessTypePresent = false;
                        if (p_sdl_msg->accessType.d)
                        {
                            if (p_sdl_msg->accessType.v == SQN_NR_PagingRecord_accessType_e_non3GPP)
                            {
                                p_record_msg->bAccessTypePresent = true;
                                p_record_msg->access_type = ACCESS_TYPE_NON3GPP;
                            }
                        }

                        switch (p_sdl_msg->ue_Identity.d)
                        {
                            case SQN_NR_PagingUE_Identity_ng_5G_S_TMSI:
                                p_record_msg->ue_paging_identity.presenceMask = NR_UE_PAGING_IDENTITY_NG_5G_S_TMSI;
                                p_record_msg->ue_paging_identity.choice.ng_5g_s_tmsi.length = sizeof(p_sdl_msg->ue_Identity.v.ng_5G_S_TMSI);
                                memcpy((char *)p_record_msg->ue_paging_identity.choice.ng_5g_s_tmsi.buffer,
                                       (const char *)p_sdl_msg->ue_Identity.v.ng_5G_S_TMSI, sizeof(p_sdl_msg->ue_Identity.v.ng_5G_S_TMSI));
                                if (oneTimeProcessingFlag == 0)
                                {
                                  SS_NR_PAGING_IND(message_p).ue_index_value = paging_ue_index_g;
                                  paging_ue_index_g = ((paging_ue_index_g + 4) % MAX_MOBILES_PER_GNB);
                                  oneTimeProcessingFlag = 1;
                                }
                                break;
                            case SQN_NR_PagingUE_Identity_fullI_RNTI:
                                p_record_msg->ue_paging_identity.presenceMask = NR_UE_PAGING_IDENTITY_FULL_I_RNTI;
                                p_record_msg->ue_paging_identity.choice.full_i_rnti.length = sizeof(p_sdl_msg->ue_Identity.v.fullI_RNTI);
                                memcpy((char *)p_record_msg->ue_paging_identity.choice.full_i_rnti.buffer,
                                       (const char *)p_sdl_msg->ue_Identity.v.fullI_RNTI, sizeof(p_sdl_msg->ue_Identity.v.fullI_RNTI));
                                break;
                            case SQN_PagingUE_Identity_UNBOUND_VALUE:
                                LOG_A(GNB_APP, "[SYS-GNB] Error Unhandled Paging request\n");
                                break;
                            default :
                                LOG_A(GNB_APP, "[SYS-GNB] Invalid Paging request received\n");
                                break;

                        }
                        p_sdl_msg++;
                        p_record_msg++;
                    }
                }
            }
            if (pagingRequest->SlotOffsetList.d)
            {
                LOG_A(GNB_APP, "[SYS-GNB] Slot Offset List present in Paging request\n");
                SS_NR_PAGING_IND(message_p).bSlotOffsetListPresent = true;
                SS_NR_PAGING_IND(message_p).slotOffsetList.num = 0;
                for (int i=0; i < pagingRequest->SlotOffsetList.v.d; i++)
                {
                    SS_NR_PAGING_IND(message_p).slotOffsetList.slot_offset[i] = pagingRequest->SlotOffsetList.v.v[i];
                    SS_NR_PAGING_IND(message_p).slotOffsetList.num++;
                }
            }

            int send_res = itti_send_msg_to_task(TASK_RRC_GNB, 0, message_p);
            if (send_res < 0)
            {
                LOG_A(GNB_APP, "[SYS-GNB] Error sending Paging to RRC_GNB");
            }
            oneTimeProcessingFlag = 0;
            LOG_A(GNB_APP, "[SYS-GNB] Paging_IND for Cell_id %d sent to RRC\n", cellId);
            break;
        case SQN_NR_PCCH_MessageType_messageClassExtension:
            LOG_A(GNB_APP, "[SYS-GNB] NR_PCCH_MessageType_messageClassExtension for Cell_id %d received\n", cellId);
            break;
        case SQN_NR_PCCH_MessageType_UNBOUND_VALUE:
            LOG_A(GNB_APP, "[SYS-GNB] Invalid Pging request received Type_UNBOUND_VALUE received\n");
            break;
        default:
            LOG_A(GNB_APP, "[SYS-GNB] Invalid Pging request received\n");
            break;
    }
    LOG_A(GNB_APP, "[SYS-GNB] Exit sys_handle_nr_paging_req Paging_IND processing for Cell_id %d \n", cellId);
}

/*
 * ===========================================================================================================
 * Function Name: ss_task_sys_nr_handle_req
 * Parameter    : SYSTEM_CTRL_REQ *req, is the message having ASP Defination of NR_SYSTEM_CTRL_REQ (38.523-3)
 *                which is received on SIDL via TTCN.
 *                ss_set_timinfo_t *tinfo, is currently not used.
 * Description  : This function handles the SYS_PORT_NR configuration command received from TTCN via the PORTMAN.
 *                It applies the configuration on RAN Context for NR and sends the confirmation message to
 *                PORTMAN.
 * Returns      : Void
 * ==========================================================================================================
*/
static void ss_task_sys_nr_handle_req(struct NR_SYSTEM_CTRL_REQ *req, ss_nrset_timinfo_t *tinfo)
{
  int enterState =  SS_context.SSCell_list[nr_cell_index].State;
  if (req->Common.CellId > 0) {
    nr_cell_index = get_gNB_cell_index(req->Common.CellId, SS_context.SSCell_list);
      SS_context.SSCell_list[nr_cell_index].nr_cellId = req->Common.CellId;
    LOG_A(GNB_APP, "[SYS-GNB] Current SS_STATE %d received SystemRequest_Type %d nr_cellId %d cnf_flag %d cell_index: %d\n",
        SS_context.SSCell_list[nr_cell_index].State, 
        req->Request.d, 
        SS_context.SSCell_list[nr_cell_index].nr_cellId, 
        req->Common.ControlInfo.CnfFlag,nr_cell_index);
  }



  switch (SS_context.SSCell_list[nr_cell_index].State)
  {
    case SS_STATE_NOT_CONFIGURED:
      if (req->Request.d == NR_SystemRequest_Type_Cell)
      {
        LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_Cell received\n");
        init_gnb_cell_context(nr_cell_index);
        //bugz128620 rebase RC.nrrrc[0]->carrier[nr_cell_index].pdcch_ConfigSIB1 seems not being used 
        //LOG_I(GNB_APP,"controlResourceSetZero: %d controlResourceSetZero1: %d searchSpaceZero: %d searchSpaceZero1: %d\n",RC.nrrrc[0]->carrier[nr_cell_index].pdcch_ConfigSIB1->controlResourceSetZero,RC.nrrrc[0]->carrier[nr_cell_index].pdcch_ConfigSIB1->searchSpaceZero);
        SS_context.SSCell_list[nr_cell_index].PhysicalCellId = req->Request.v.Cell.v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.v;
        LOG_I(NR_RRC, "PhysicalCellId:%d for cell_index:%d\n", SS_context.SSCell_list[nr_cell_index].PhysicalCellId, nr_cell_index);
        if (false == ss_task_sys_nr_handle_cellConfig5G(&req->Request.v.Cell, SS_context.SSCell_list[nr_cell_index].State) )
        {
          LOG_A(GNB_APP, "[SYS-GNB] Error handling Cell Config 5G for NR_SystemRequest_Type_Cell \n");
          return;
        }
        cell_config_5G_done_indication();

        RC.ss.State  = SS_STATE_CELL_ACTIVE; //bugz128620 to clear this flag. and remove this comment
        if (SS_context.SSCell_list[nr_cell_index].State == SS_STATE_NOT_CONFIGURED)
        {
            //The flag is used to initilize the cell in the RRC layer during init_NR_SI funciton
            RC.ss.CC_conf_flag[nr_cell_index] = 1; 
            printf("fxn: %s cell_index: %d RC.ss.CC_conf_flag[cell_index]: %d\n",__FUNCTION__,nr_cell_index,RC.ss.CC_conf_flag[nr_cell_index]);

            LOG_A (GNB_APP,"current Configured CC are %d current CC_index %d nb_nr_mac_CC %d\n",
                  RC.nb_nr_CC[0],nr_cell_index,*RC.nb_nr_mac_CC);
            //Increment nb_cc only from 2nd cell as the initilization is done for 1 CC
            if (nr_cell_index)
            {
              //Increment the nb_CC supported as new cell is confiured
              RC.nb_nr_CC[0] ++;

              //Set the number of MAC_CC to current configured CC value
              *RC.nb_nr_mac_CC= RC.nb_nr_CC[0];

              LOG_A (GNB_APP,"CC-MGMT nb_cc is incremented current Configured CC are %d current CC_index %d nb_nr_mac_CC %d\n",
                  RC.nb_nr_CC[0],nr_cell_index,*RC.nb_nr_mac_CC);
            }

          SS_context.SSCell_list[nr_cell_index].State = SS_STATE_CELL_ACTIVE;
          LOG_A(GNB_APP, "[SYS-GNB] New State for cell:%d changed to %d\n", nr_cell_index, SS_context.SSCell_list[nr_cell_index].State);
        }

        send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_Cell, NULL);

       

        if (req->Request.v.Cell.d == NR_CellConfigRequest_Type_AddOrReconfigure)
        {
          CellConfig5GReq_t	*cellConfig = NULL;
          struct NR_CellConfigInfo_Type *p_cellConfig = NULL;
          p_cellConfig = &req->Request.v.Cell.v.AddOrReconfigure;
          SS_context.SSCell_list[nr_cell_index].maxRefPower = p_cellConfig->CellConfigCommon.v.InitialCellPower.v.MaxReferencePower;
          cellConfig = (CellConfig5GReq_t*)malloc(sizeof(CellConfig5GReq_t));
          if (cellConfig == NULL)
          {
		        AssertFatal(cellConfig != NULL , "[SYS-GNB] Failed to allocate memory for proxy cell config\n");
          }
          cellConfig->header.preamble = 0xFEEDC0DE;
          cellConfig->header.msg_id = SS_CELL_CONFIG;
          cellConfig->header.length = sizeof(proxy_ss_header_t);
          //cellConfig->initialAttenuation = 0;
          cellConfig->initialAttenuation = p_cellConfig->CellConfigCommon.v.InitialCellPower.v.Attenuation.v.Off==true?53:p_cellConfig->CellConfigCommon.v.InitialCellPower.v.Attenuation.v.Value;
          if (req->Request.v.Cell.v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.d == true)
          {
            cellConfig->header.cell_id = req->Request.v.Cell.v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.v;
          }
          cellConfig->maxRefPower= p_cellConfig->CellConfigCommon.v.InitialCellPower.v.MaxReferencePower;
          cellConfig->absoluteFrequencyPointA = p_cellConfig->PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.absoluteFrequencyPointA;
          cellConfig->header.cell_index = nr_cell_index;
          LOG_A(ENB_SS,"5G Cell configuration received for cell_id: %d Initial attenuation: %d \
              Max ref power: %d\n for absoluteFrequencyPointA : %ld cell_index: %d =================================== \n",
              cellConfig->header.cell_id,
              cellConfig->initialAttenuation, cellConfig->maxRefPower,
              cellConfig->absoluteFrequencyPointA, cellConfig->header.cell_index);
          sys_5G_send_proxy((void *)cellConfig, sizeof(CellConfig5GReq_t));
        }

      }
      else if (req->Request.d == NR_SystemRequest_Type_DeltaValues)
      {
        /* TBD: Sending the dummy confirmation for now*/
        ss_task_sys_nr_handle_deltaValues(req);
        LOG_A(GNB_APP, "[SYS-GNB] Sent SYS CNF for NR_SystemRequest_Type_DeltaValues\n");
      }
      else
      {
        LOG_E(GNB_APP, "[SYS-GNB] Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
            SS_context.SSCell_list[nr_cell_index].State , req->Request.d);
      }
      break;
    case SS_STATE_CELL_ACTIVE:
      {
        switch (req->Request.d)
        {
          case NR_SystemRequest_Type_Cell:
            {
              if (false == ss_task_sys_nr_handle_cellConfig5G(&req->Request.v.Cell, SS_context.SSCell_list[nr_cell_index].State) )
              {
                LOG_E(GNB_APP, "[SYS-GNB] Error handling Cell Config 5G for NR_SystemRequest_Type_Cell \n");
              }

              if ( req->Common.ControlInfo.CnfFlag) {
                send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_Cell, NULL);
              }
            }
            break;
          case NR_SystemRequest_Type_EnquireTiming:
            {
              sys_handle_nr_enquire_timing(tinfo);
              LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_EnquireTiming received\n");
            }
            break;
          case NR_SystemRequest_Type_RadioBearerList:
            {
              LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_RadioBearerList received\n");
              if (false == ss_task_sys_nr_handle_cellConfigRadioBearer(req) )
              {
                LOG_A(GNB_APP, "[SYS-GNB] Error handling Cell Config 5G for NR_SystemRequest_Type_Cell \n");
                return;
              }
            }
            break;
          case NR_SystemRequest_Type_CellAttenuationList:
            {
              LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_CellAttenuationList received\n");
              sys_handle_nr_cell_attn_req(&(req->Request.v.CellAttenuationList));
              if (false == ss_task_sys_nr_handle_cellConfigAttenuation(req) )
              {
                LOG_A(GNB_APP, "[SYS-GNB] Error handling Cell Config 5G for NR_SystemRequest_Type_Cell \n");
                return;
              }
            }
            break;
          case NR_SystemRequest_Type_PdcpCount:
            {
              LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_PdcpCount received\n");
              ss_task_sys_nr_handle_pdcpCount(req);
            }
            break;
          case NR_SystemRequest_Type_AS_Security:
            {
              LOG_A(GNB_APP, "[SYS-GNB] Handling for NR_SystemRequest_Type_AS_Security\n");
              sys_handle_nr_as_security_req(&(req->Request.v.AS_Security),nr_cell_index);
              if (req->Common.ControlInfo.CnfFlag) {
                send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_AS_Security, NULL);
              }
            }
            break;
          case NR_SystemRequest_Type_DeltaValues:
            {
              ss_task_sys_nr_handle_deltaValues(req);
              LOG_A(GNB_APP, "[SYS-GNB] Sent SYS CNF for NR_SystemRequest_Type_DeltaValues\n");
            }
            break;
          case NR_SystemRequest_Type_Paging:
            {
                LOG_A(GNB_APP, "[SYS-GNB] NR_SystemRequest_Type_Paging: received\n");
                ss_nrset_timinfo_t pg_timinfo;
                pg_timinfo.sfn = req->Common.TimingInfo.v.SubFrame.SFN.v.Number;
                pg_timinfo.slot = req->Common.TimingInfo.v.SubFrame.Subframe.v.Number; //TODO

                uint8_t slotsPerSubFrame = 1 << SS_context.mu;
                pg_timinfo.slot *= slotsPerSubFrame;

                if (req->Common.TimingInfo.v.SubFrame.Slot.d == SlotTimingInfo_Type_SlotOffset &&
                  req->Common.TimingInfo.v.SubFrame.Slot.v.SlotOffset.d >= SlotOffset_Type_Numerology1)
                {
                  pg_timinfo.slot += req->Common.TimingInfo.v.SubFrame.Slot.v.SlotOffset.v.Numerology1;
                }

                sys_handle_nr_paging_req(&(req->Request.v.Paging), pg_timinfo);
            }
            break;
          default:
            {
              LOG_E(GNB_APP, "[SYS-GNB] Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
                  SS_context.SSCell_list[nr_cell_index].State , req->Request.d);
            }
            break;
        }
      }
      break;
  }
  LOG_A(GNB_APP, "[SYS-GNB] SS_STATE %d New SS_STATE %d received SystemRequest_Type %d\n",
      enterState, SS_context.SSCell_list[nr_cell_index].State, req->Request.d);
}


/*
 * ===========================================================================================================
 * Function Name: ss_gNB_sys_process_itti_msg
 * Parameter    : notUsed, is a dummy parameter is not being used currently
 * Description  : This function is entry point function for TASK_SYS_5G_NR. This function process the received
 *                messages from other module and invokes respective handler function
 * Returns      : Void
 * ==========================================================================================================
*/

void *ss_gNB_sys_process_itti_msg(void *notUsed)
{
	MessageDef *received_msg = NULL;
	int result;
	static ss_nrset_timinfo_t tinfo = {.hsfn=0xFFFF, .sfn = 0xFFFF, .slot = 0xFFFFFFFF};

	itti_receive_msg(TASK_SYS_GNB, &received_msg);

	/* Check if there is a packet to handle */
	if (received_msg != NULL)
	{
		switch (ITTI_MSG_ID(received_msg))
		{
			case SS_NRUPD_TIM_INFO:
				{
					LOG_D(GNB_APP, "TASK_SYS_GNB received SS_NRUPD_TIM_INFO with sfn=%d slot=%d\n", SS_NRUPD_TIM_INFO(received_msg).sfn, SS_NRUPD_TIM_INFO(received_msg).slot);
					tinfo.slot = SS_NRUPD_TIM_INFO(received_msg).slot;
					tinfo.sfn = SS_NRUPD_TIM_INFO(received_msg).sfn;
					/*WA: calculate hsfn here */
					if(tinfo.hsfn == 0xFFFF){
						tinfo.hsfn = 0;
					} else if(tinfo.sfn == 1023 && SS_NRUPD_TIM_INFO(received_msg).slot == 0){
						tinfo.hsfn++;
						if(tinfo.hsfn == 1024){
							tinfo.hsfn = 0;
						}
					}
					SS_context.hsfn  = tinfo.hsfn;
					SS_context.sfn = tinfo.sfn;
					SS_context.slot  = tinfo.slot;
					g_log->sfn = tinfo.sfn;
					g_log->sf  = tinfo.slot;
				}
				break;

			case SS_NR_SYS_PORT_MSG_IND:
				{
						ss_task_sys_nr_handle_req(SS_NR_SYS_PORT_MSG_IND(received_msg).req, &tinfo);
				}
				break;
			case UDP_DATA_IND:
				{
					LOG_A(GNB_APP, "[TASK_SYS_GNB] received UDP_DATA_IND \n");
					proxy_ss_header_t hdr;
					memcpy(&hdr, (SS_SYS_PROXY_MSG_CNF(received_msg).buffer), sizeof(proxy_ss_header_t));
					LOG_A(GNB_APP, "[TASK_SYS_GNB] received msgId:%d\n", hdr.msg_id);
					switch (hdr.msg_id)
					{
						case SS_CELL_CONFIG_CNF:
							LOG_A(GNB_APP, "[TASK_SYS_GNB] received UDP_DATA_IND with Message SS_NR_SYS_PORT_MSG_CNF\n");
							break;
            case SS_ATTN_LIST_CNF:
              LOG_A(GNB_APP, "[TASK_SYS_GNB] received UDP_DATA_IND with Message SS_ATTN_LIST_CNF\n");
							break;

						default:
							LOG_E(GNB_APP, "[TASK_SYS_GNB] received unhandled message:%d \n",hdr.msg_id);
					}
				}
				break;
			case TERMINATE_MESSAGE:
				{
					itti_exit_task();
					break;
				}
			default:
				LOG_A(GNB_APP, "TASK_SYS_GNB: Received unhandled message %d:%s\n",
						ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
				break;
		}
		result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
		AssertFatal(result == EXIT_SUCCESS, "[SYS] Failed to free memory (%d)!\n", result);
		received_msg = NULL;
	}
	return NULL;
}

/*
 * Function : ss_gNB_sys_task
 * Description:  The SYS_TASK main function handler. Initilizes the UDP
 * socket towards the Proxy for the configuration updates. Initilizes
 * the SYS_TASK state machine Init_State. Invoke the itti message
 * handler for the SYY_PORT.
 */
void *ss_gNB_sys_task(void *arg)
{
  udpSockReq_t req;
  req.address = local_5G_address;
  req.port = proxy_5G_recv_port;
  sys_5G_send_init_udp(&req);
  sleep(5);
  // Set the state to NOT_CONFIGURED for Cell Config processing mode
  if (RC.ss.mode == SS_SOFTMODEM || RC.ss.mode == SS_HWTMODEM)
  {
    init_ss_gNB_context(SS_context.SSCell_list);
    LOG_A(GNB_APP, "TASK_SYS_GNB: fxn:%s line:%d RC.ss.mode:SS_STATE_NOT_CONFIGURED \n", __FUNCTION__, __LINE__);
    SS_context.SSCell_list[nr_cell_index].State = 0;
  }
  // Set the state to CELL_ACTIVE for SRB processing mode
  /*else if (RC.ss.mode == SS_HWTMODEM)
  {
    SS_context.SSCell_list[nr_cell_index].State = SS_STATE_CELL_ACTIVE;
    LOG_A(GNB_APP, "TASK_SYS_GNB: fxn:%s line:%d SS_STATE_CELL_ACTIVE \n", __FUNCTION__, __LINE__);
  }*/

  while (1)
  {
    ss_gNB_sys_process_itti_msg(NULL);
  }

  return NULL;
}

/*
 * Function   : ss_task_sys_nr_handle_deltaValues
 * Description: This function handles the NR_SYSTEM_CTRL_REQ for DeltaValues and updates the CNF structures as
 *              per cell's band configuration.
 * Returns    : None
 */
static void ss_task_sys_nr_handle_deltaValues(struct NR_SYSTEM_CTRL_REQ *req)
{
	LOG_A(GNB_APP, "[SYS-GNB] Entry in fxn:%s\n", __FUNCTION__);
  int CC_id = RC.nb_nr_CC[0];
  NR_UE_info_t *UE = NULL;
  if(req->Common.CellId){ 
    CC_id = get_gNB_cell_index(req->Common.CellId, SS_context.SSCell_list);
  }else{   // req->Common.CellId ==0  means no specific cell in this command
    for(CC_id = 0; CC_id < RC.nb_nr_CC[0];CC_id++){
      UE = find_nr_UE(&RC.nrmac[0]->UE_info,CC_id, SS_context.SSCell_list[CC_id].ss_rnti_g);
      if(UE!= NULL){
        break;
      }
    }

  }
  struct NR_SYSTEM_CTRL_CNF *msgCnf = CALLOC(1, sizeof(struct NR_SYSTEM_CTRL_CNF));
	MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, SS_NR_SYS_PORT_MSG_CNF);
  if (!message_p)
	{
		LOG_A(GNB_APP, "[SYS-GNB] Error Allocating Memory for message NR_SYSTEM_CTRL_CNF \n");
		return ;

	}
	msgCnf->Common.CellId = req->Common.CellId;
	msgCnf->Common.RoutingInfo.d = NR_RoutingInfo_Type_None;
	msgCnf->Common.RoutingInfo.v.None = true;
	msgCnf->Common.TimingInfo.d = TimingInfo_Type_None;
	msgCnf->Common.TimingInfo.v.None = true;
	msgCnf->Common.Result.d = ConfirmationResult_Type_Success;
	msgCnf->Common.Result.v.Success = true;

	msgCnf->Confirm.d = NR_SystemConfirm_Type_DeltaValues;
	struct UE_NR_DeltaValues_Type* vals = &msgCnf->Confirm.v.DeltaValues;
	struct DeltaValues_Type* deltaPrimaryBand = &vals->DeltaPrimaryBand;
	struct DeltaValues_Type* deltaSecondaryBand = &vals->DeltaSecondaryBand;

	deltaPrimaryBand->DeltaNRf1 = 0;
	deltaPrimaryBand->DeltaNRf2 = 0;
	deltaPrimaryBand->DeltaNRf3 = 0;
	deltaPrimaryBand->DeltaNRf4 = 0;

	deltaSecondaryBand->DeltaNRf1 = 0;
	deltaSecondaryBand->DeltaNRf2 = 0;
	deltaSecondaryBand->DeltaNRf3 = 0;
	deltaSecondaryBand->DeltaNRf4 = 0;
  if(CC_id != RC.nb_nr_CC[0]){
    LOG_A(GNB_APP, "[SYS-GNB] absoluteFrequencySSB:%ld\n",
      *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB);
  }
  if ((get_softmodem_params()->numerology >= 1 || get_softmodem_params()->numerology <= 2) && (SS_context.SSCell_list[CC_id].State == SS_STATE_CELL_ACTIVE) && CC_id != RC.nb_nr_CC[0])
  {
      UE = find_nr_UE(&RC.nrmac[0]->UE_info,CC_id, SS_context.SSCell_list[CC_id].ss_rnti_g);
      if(UE->rsrpReportStatus){
      LOG_A(GNB_APP, "[SYS-GNB] received SYSTEM_CTRL_REQ with DeltaValues in Active State for Primary Band \n");
      if (req->Request.v.DeltaValues.DeltaPrimary.Ssb_NRf1.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf1 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf1:%d for deltaPrimaryBand \n", deltaPrimaryBand->DeltaNRf1);
      }
      if (req->Request.v.DeltaValues.DeltaPrimary.Ssb_NRf2.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf2 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf2:%d for deltaPrimaryBand \n", deltaPrimaryBand->DeltaNRf2);
      }
      if (req->Request.v.DeltaValues.DeltaPrimary.Ssb_NRf3.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf3 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf3:%d for deltaPrimaryBand \n", deltaPrimaryBand->DeltaNRf3);
      }
      if (req->Request.v.DeltaValues.DeltaPrimary.Ssb_NRf4.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf4 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf4:%d for deltaPrimaryBand \n", deltaPrimaryBand->DeltaNRf4);
      }
    }
  }
  else if ((get_softmodem_params()->numerology >= 2 && SS_context.SSCell_list[CC_id].State == SS_STATE_CELL_ACTIVE) && (UE->rsrpReportStatus) && CC_id != RC.nb_nr_CC[0])
  {
      UE = find_nr_UE(&RC.nrmac[0]->UE_info, CC_id, SS_context.SSCell_list[CC_id].ss_rnti_g);
      if(UE->rsrpReportStatus){
      LOG_A(GNB_APP, "[SYS-GNB] received SYSTEM_CTRL_REQ with DeltaValues in Active State for Secondary Band \n");
      if (req->Request.v.DeltaValues.DeltaSecondary.Ssb_NRf1.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaSecondaryBand->DeltaNRf1 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf1:%d for deltaSecondaryBand \n", deltaSecondaryBand->DeltaNRf1);
      }
      if (req->Request.v.DeltaValues.DeltaSecondary.Ssb_NRf2.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf2 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf2:%d for deltaSecondaryBand \n", deltaSecondaryBand->DeltaNRf2);
      }
      if (req->Request.v.DeltaValues.DeltaSecondary.Ssb_NRf3.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf3 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf3:%d for deltaSecondaryBand \n", deltaSecondaryBand->DeltaNRf3);
      }
      if (req->Request.v.DeltaValues.DeltaSecondary.Ssb_NRf4.v.v.R15 == *RC.nrrrc[0]->configuration[CC_id].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB)
      {
        deltaPrimaryBand->DeltaNRf4 = UE->ssb_rsrp + 82;
        LOG_A(GNB_APP, "updated DeltaNRf4:%d for deltaSecondaryBand \n", deltaSecondaryBand->DeltaNRf4);
      }
    }
  }

	SS_NR_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
	int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT, message_p);
	if (send_res < 0)
	{
		LOG_A(GNB_APP, "[SYS-GNB] Error sending to [SS-PORTMAN-GNB]");
	}
	LOG_A(GNB_APP, "[SYS-GNB] Exit from fxn:%s\n", __FUNCTION__);

}

/*
 * Function   : ss_task_sys_nr_handle_cellConfig5G
 * Description: This function handles the NR_SYSTEM_CTRL_REQ for request type AddOrReconfigure. TTCN provides the values of for Cell Config Req 5G on SYS Port
 *              and those values are populated here in corresponding structures of NR RRC.
 * Returns    : None
 */

bool ss_task_sys_nr_handle_cellConfig5G(struct NR_CellConfigRequest_Type *p_req,int cell_State)
{
  uint32_t gnbId = 0;
  if (p_req->d == NR_CellConfigRequest_Type_AddOrReconfigure)
  {
    
    /* populate each config */
    /* 1. StaticResource Config */
    if(p_req->v.AddOrReconfigure.StaticResourceConfig.d)
    {

    }

    /* 2. CellConfigCommon: currently NR_InitialCellPower_Type is processed after cell config  */
    if(p_req->v.AddOrReconfigure.CellConfigCommon.d)
    {

    }
    /* 3.  PhysicalLayer */
    /* TODO: populate fields to
         RC.nrrrc[gnbId]->carrier.servingcellconfigcommon
         RC.nrrrc[gnbId]->carrier.cellConfigDedicated
    */
    if(p_req->v.AddOrReconfigure.PhysicalLayer.d)
    {

    }

    /* 4.  BcchConfig */
    /* TODO: populate all BcchConfig fields to
              RC.nrrrc[gnbId]->carrier.mib
              RC.nrrrc[gnbId]->carrier.siblock1
              RC.nrrrc[gnbId]->carrier.systemInformation
     */
    if(p_req->v.AddOrReconfigure.BcchConfig.d)  //W38 note: we are decoding mib/sib from ttcn and configure cell configuration to gnb cells, and then mac regenerate it 
    {
      if (p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.d == true )
      {
        RC.nrrrc[gnbId]->configuration[nr_cell_index].cellBarred = 
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.v.mib.cellBarred;
        LOG_I(GNB_APP, "Cell Barred Status:%d for cell_index:%d\n", 
            RC.nrrrc[gnbId]->configuration[nr_cell_index].cellBarred,
            nr_cell_index);
      }

      if (p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.d == true && 
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.d == SQN_NR_BCCH_DL_SCH_MessageType_c1_systemInformationBlockType1  &&
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellSelectionInfo.d == true)
      {
        RC.nrrrc[gnbId]->configuration[nr_cell_index].q_RxLevMinSIB1 = 
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellSelectionInfo.v.q_RxLevMin;
        LOG_I(GNB_APP, "SIB1 q_RxLevMin:%d for cell_index:%d\n", RC.nrrrc[gnbId]->configuration[nr_cell_index].q_RxLevMinSIB1, nr_cell_index);
      }
      if (p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.d == true &&
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.d == SQN_NR_BCCH_DL_SCH_MessageType_c1_systemInformationBlockType1  &&
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellAccessRelatedInfo.plmn_IdentityInfoList.d == true )
      {
       
        RC.nrrrc[gnbId]->configuration[nr_cell_index].cell_identity = 0;
        for (int i = 0; i < 36; i++)
        {
          RC.nrrrc[gnbId]->configuration[nr_cell_index].cell_identity += 
          (uint64_t)(p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellAccessRelatedInfo.plmn_IdentityInfoList.v->cellIdentity[i])<<(35-i);
        }
        
        if( p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellAccessRelatedInfo.plmn_IdentityInfoList.v->trackingAreaCode.d == true )
        RC.nrrrc[gnbId]->configuration[nr_cell_index].tac = 0;
        for (int i = 0; i < 24; i++)
          {
            RC.nrrrc[gnbId]->configuration[nr_cell_index].tac  +=p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellAccessRelatedInfo.plmn_IdentityInfoList.v->trackingAreaCode.v[i] << (23 - i);
          }
        //BIT_STRING_to_uint32(p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIB1.v.message.v.c1.v.systemInformationBlockType1.cellAccessRelatedInfo.plmn_IdentityInfoList.v->trackingAreaCode.v);
          
        LOG_I(GNB_APP, "SIB1 tac:%d for cell_index:%d RC.nrrrc[gnbId]->configuration[nr_cell_index].cell_identity %ld\n", RC.nrrrc[gnbId]->configuration[nr_cell_index].tac , nr_cell_index, RC.nrrrc[gnbId]->configuration[nr_cell_index].cell_identity);
      }
          
      if (p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIs.d == true && 
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIs.v.d == true &&
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIs.v.v->message.v.c1.d == SQN_NR_BCCH_DL_SCH_MessageType_c1_systemInformation &&
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIs.v.v->message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation.sib_TypeAndInfo.v->d ==
          SQN_NR_SystemInformation_IEs_sib_TypeAndInfo_s_sib2 )
      {
        RC.nrrrc[gnbId]->configuration[nr_cell_index].q_RxLevMinSIB2 = 
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.SIs.v.v->message.v.c1.v.systemInformation.criticalExtensions.
          v.systemInformation.sib_TypeAndInfo.v->v.sib2.intraFreqCellReselectionInfo.q_RxLevMin;
        LOG_I(ENB_SS, "SIB2 q_RxLevMin:%d for cell_index:%d\n", RC.nrrrc[gnbId]->configuration[nr_cell_index].q_RxLevMinSIB2, nr_cell_index);
      }

    }

    /* 5. PcchConfig */
    if(p_req->v.AddOrReconfigure.PcchConfig.d)
    {

    }

    /* 6. RachProcedureConfig */
    if(p_req->v.AddOrReconfigure.RachProcedureConfig.d)
    {
      if (p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.d
          && p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.v.d)
      {
        if (p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.v.d != 1)
        {
          LOG_W(GNB_APP, "RachProcedureList != 1 (not handled other elements)\n");
        }
        if (p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.v.v[0].ContentionResolution.d &&
            p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.v.v[0].ContentionResolution.v.d == NR_ContentionResolutionCtrl_Type_Msg4_Based &&
            p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.v.v[0].ContentionResolution.v.v.Msg4_Based.RrcPdu.d)
        {
          process_RachProcedureMsg4RrcMsg(&p_req->v.AddOrReconfigure.RachProcedureConfig.v.RachProcedureList.v.v[0].ContentionResolution.v.v.Msg4_Based.RrcPdu.v);
        }
      }
    }

    /* 7. DcchDtchConfig */
    if(p_req->v.AddOrReconfigure.DcchDtchConfig.d)
    {
      if(NULL == RC.nrrrc[gnbId]->carrier[nr_cell_index].dcchDtchConfig){
        RC.nrrrc[gnbId]->carrier[nr_cell_index].dcchDtchConfig = calloc(1,sizeof(NR_DcchDtchConfig_t));
      }
      NR_DcchDtchConfig_t * dcchDtchConfig = RC.nrrrc[gnbId]->carrier[nr_cell_index].dcchDtchConfig;
      if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.DL.d)
      {

      }

      if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.d)
      {
        if(NULL==dcchDtchConfig->ul){
          dcchDtchConfig->ul = calloc(1,sizeof(*(dcchDtchConfig->ul)));
        }
        if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.d){

          if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.d){
            if(NULL == dcchDtchConfig->ul->dci_info){
              dcchDtchConfig->ul->dci_info = calloc(1,sizeof(*(dcchDtchConfig->ul->dci_info)));
            }
            if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.d){
              if(NULL == dcchDtchConfig->ul->dci_info->resoure_assignment){
                 dcchDtchConfig->ul->dci_info->resoure_assignment = calloc(1,sizeof(*(dcchDtchConfig->ul->dci_info->resoure_assignment)));
              }
              if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.v.FreqDomain.d){
                dcchDtchConfig->ul->dci_info->resoure_assignment->FirstRbIndex =
                p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.v.FreqDomain.v.FirstRbIndex;
                //TODO bug #133016 with USRP board, system crashed if Nprb is 48 or 24 ie what is provided by TTCN which does make no sense. Need to understand why
                // if NPrb is set to a different value from 0, type0_PDCCH_CSS_config will be filled and such config is used for dcch_dtch config during rach procedure
                //if dcch_dtch config is not NULL, MAC will trigger rrc reconfiguration considering that UE under attachment has already been configured
                if (RC.ss.mode == SS_HWTMODEM) {
                  dcchDtchConfig->ul->dci_info->resoure_assignment->Nprb = 0;
                } else {
                  dcchDtchConfig->ul->dci_info->resoure_assignment->Nprb =
                          p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.v.FreqDomain.v.Nprb;
                }
	      }

              if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.v.TransportBlockScheduling.d) {
                if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.v.TransportBlockScheduling.v.d > 0){
                  struct NR_TransportBlockSingleTransmission_Type * tbst = &p_req->v.AddOrReconfigure.DcchDtchConfig.v.UL.v.SearchSpaceAndDci.v.DciInfo.v.ResoureAssignment.v.TransportBlockScheduling.v.v[0];
                  dcchDtchConfig->ul->dci_info->resoure_assignment->transportBlock_scheduling.imcs = tbst->ImcsValue;
                  dcchDtchConfig->ul->dci_info->resoure_assignment->transportBlock_scheduling.RedundancyVersion = tbst->RedundancyVersion;
                  dcchDtchConfig->ul->dci_info->resoure_assignment->transportBlock_scheduling.ToggleNDI = tbst->ToggleNDI;
                }
              }
            }
          }
        }
      }

      if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.DrxCtrl.d)
      {

      }

      if(p_req->v.AddOrReconfigure.DcchDtchConfig.v.MeasGapCtrl.d)
      {

      }
    }

    /* 8. ServingCellConfig */
    /* TODO: populate ServingCellConfig to
        RC.nrrrc[gnbId]->carrier.cell_GroupId
        RC.nrrrc[gnbId]->carrier.mac_cellGroupConfig
        RC.nrrrc[gnbId]->carrier.physicalCellGroupConfig
    */
    if(p_req->v.AddOrReconfigure.ServingCellConfig.d)
    {

    }
    #if 0
      if(cell_State != SS_STATE_NOT_CONFIGURED){
      /* Trigger RRC Cell reconfig when cell is active */
      MessageDef *msg_p = NULL;
      msg_p = itti_alloc_new_message (TASK_GNB_APP, 0, NRRRC_CONFIGURATION_REQ);
      LOG_I(GNB_APP,"ss_gNB Sending configuration message to NR_RRC task %lx\n", &RC.nrrrc[gnbId]->configuration);
      memcpy(&NRRRC_CONFIGURATION_REQ(msg_p), &RC.nrrrc[gnbId]->configuration,sizeof(NRRRC_CONFIGURATION_REQ(msg_p)));
      itti_send_msg_to_task (TASK_RRC_GNB, GNB_MODULE_ID_TO_INSTANCE(gnbId), msg_p);

 
      return true;
    }
    #endif
    /* following code shall be optimized and moved to "PhysicalLayer" populating*/
    /*****************************************************************************/
    /* Populating PhyCellId */
    if (p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.d == true && p_req->v.AddOrReconfigure.PhysicalLayer.d == true)
    {
      RC.nrrrc[gnbId]->carrier[nr_cell_index].physCellId = p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.v;
   // *(RC.nrrrc[gnbId]->carrier[nr_cell_index].servingcellconfigcommon->physCellId) = p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.v;
      LOG_A(GNB_APP," %s gnbId %d, nr_cell_index %d, physCellId %d \n",__FUNCTION__,gnbId,nr_cell_index,RC.nrrrc[gnbId]->carrier[nr_cell_index].physCellId );
    //note W38: scc and sib1 moved to mac. let's say rrc does not need physcellid => still needed, where rrc report message to ttcn, phycell id is needed.
      *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->physCellId = p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.PhysicalCellId.v;
    }

    /* Populating NR_ARFCN */
    /*
     TTCN configuraton: offsettopointA = 4, controlResourceSetZero =2 drived from absoluteFrequencyPointA and absoluteFrequencySSB
     json parameter: offsettopointA = 86, controlResourceSetZero =12
     since 7cca085ff0b16db04c53d24bf4dccbbae9f98486 above two are mixed: offsettoPointA = 4, controlResourceSetZero =12
     this caused vrb_map[...] out of bound, then various segment faults.
     previous WA: controlResourceSetZero to 2 caused regressions. this WA is to give up absoluteFrequencyPointA and absoluteFrequencySSB from TTCN, so offsettoPointA=86
  
    */
    if (RC.ss.mode == SS_HWTMODEM) { 
      if (p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.d == NR_ASN1_FrequencyInfoDL_Type_R15)
      {
        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA =
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.absoluteFrequencyPointA;
        LOG_A(GNB_APP, "fxn:%s DL absoluteFrequencyPointA :%ld\n", __FUNCTION__, 
            RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencyPointA);
      }

      /* Populating absoluteFrequencySSB */
      if (p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.absoluteFrequencySSB.d == true &&
          p_req->v.AddOrReconfigure.PhysicalLayer.d == true )
      {
        *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB =
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.absoluteFrequencySSB.v;
        LOG_A(GNB_APP, "fxn:%s DL absoluteFrequencySSB:%ld\n", __FUNCTION__, 
            *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->absoluteFrequencySSB);  //W38 note: we have only one copy of SCC, fortunately, ttcn does not update its item after cell activated.
        // note: no need to recover offsetToPointA as it can be deduced from absoluteFrequencySSB: get_ssb_offset_to_pointA API
      }
    }

    /* Populating frequency band list */
    if( p_req->v.AddOrReconfigure.PhysicalLayer.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.d == true && 0){
        for (int i = 0; i < p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.frequencyBandList.d; i++)
        {
          //LOG_I(GNB_APP,"mark: fxn:%s %d \n", __FUNCTION__,__LINE__);
         
          *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[i] =
            p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.frequencyBandList.v[i];
          

          if (p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.d == NR_DuplexMode_Type_TDD)
          {
            LOG_A(NR_MAC, "Duplex mode TDD\n");
            *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->frequencyBandList->list.array[i]= 
              p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.frequencyBandList.v[i];

          }
          LOG_A(GNB_APP, "fxn:%s DL band[%d]:%ld UL band[%d]:%ld\n", __FUNCTION__, i, 
              *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->frequencyBandList.list.array[i],
              i,
              *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->frequencyBandList->list.array[i]);
        
        }
	// LOG_I(GNB_APP,"mark: fxn:%s %d \n", __FUNCTION__,__LINE__);
        for (int i = 0; i < p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.scs_SpecificCarrierList.d; i++)
        {
            RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[i]->offsetToCarrier =
              p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.scs_SpecificCarrierList.v[i].offsetToCarrier;

            RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[i]->subcarrierSpacing =
              p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.scs_SpecificCarrierList.v[i].subcarrierSpacing;

            RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[i]->carrierBandwidth =
              p_req->v.AddOrReconfigure.PhysicalLayer.v.Downlink.v.FrequencyInfoDL.v.v.R15.scs_SpecificCarrierList.v[i].carrierBandwidth;

            *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->ssbSubcarrierSpacing = 
              RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[i]->subcarrierSpacing;
            SS_context.mu =      RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->frequencyInfoDL->scs_SpecificCarrierList.list.array[i]->subcarrierSpacing;
        }
    }
    
    /* Populating scs_SpecificCarrierList */
    if( p_req->v.AddOrReconfigure.PhysicalLayer.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.d == true)
    {
           
        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->tdd_UL_DL_ConfigurationCommon->referenceSubcarrierSpacing = 
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.v.v.R15.referenceSubcarrierSpacing;

        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->tdd_UL_DL_ConfigurationCommon->pattern1.dl_UL_TransmissionPeriodicity = 
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.v.v.R15.pattern1.dl_UL_TransmissionPeriodicity;

        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofDownlinkSlots = 
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.v.v.R15.pattern1.nrofDownlinkSlots;

        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofDownlinkSymbols = 
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.v.v.R15.pattern1.nrofDownlinkSymbols;

        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSlots = 
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.v.v.R15.pattern1.nrofUplinkSlots;

        RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->tdd_UL_DL_ConfigurationCommon->pattern1.nrofUplinkSymbols = 
          p_req->v.AddOrReconfigure.PhysicalLayer.v.Common.v.DuplexMode.v.v.TDD.v.Config.Common.v.v.R15.pattern1.nrofUplinkSymbols;
    }
    

    if (p_req->v.AddOrReconfigure.BcchConfig.d == true &&
        p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.d == true &&
      p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.d == true)
    {
      RC.nrrrc[gnbId]->configuration[nr_cell_index].ssb_SubcarrierOffset = 
        p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.v.mib.ssb_SubcarrierOffset;
    }
    

    if (p_req->v.AddOrReconfigure.BcchConfig.d == true &&
        p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.d == true &&
        p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.d == true &&
        p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.d ==true)
    {
        
        NR_ControlResourceSetZero_t *pNR_ControlResourceSetZero_t =RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero;
        LOG_D(NR_RRC,"mark %s line %d ccid %d \n  \
        %d pNR_ControlResourceSetZero_t[%d] %x \n \
        *pNR_ControlResourceSetZero_t[%d] %d \n \
        scc[%d]->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero %x \n",
            __FUNCTION__,
        __LINE__, 
        nr_cell_index,
        __LINE__,  nr_cell_index,pNR_ControlResourceSetZero_t,
        nr_cell_index, *pNR_ControlResourceSetZero_t,
        nr_cell_index,  p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.v.mib.pdcch_ConfigSIB1.controlResourceSetZero);
        //TODO setting up such params in simu causes unstabilities
        if (RC.ss.mode == SS_HWTMODEM){
	      RC.nrrrc[gnbId]->configuration[nr_cell_index].ssb_SubcarrierOffset =
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.v.mib.ssb_SubcarrierOffset;
          *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->controlResourceSetZero =
          p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.v.mib.pdcch_ConfigSIB1.controlResourceSetZero;
          *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->downlinkConfigCommon->initialDownlinkBWP->pdcch_ConfigCommon->choice.setup->searchSpaceZero =
           p_req->v.AddOrReconfigure.BcchConfig.v.BcchInfo.v.MIB.v.message.v.mib.pdcch_ConfigSIB1.searchSpaceZero;
        }
    }
   
    /* UL Absolute Frequency Population  */
    /* Populating NR_ARFCN */
    if (p_req->v.AddOrReconfigure.PhysicalLayer.d ==true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.d ==true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.d == NR_Uplink_Type_Config &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.d == true &&
        p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.v.d == NR_ASN1_FrequencyInfoUL_Type_R15)
    {

      /* Populating scs_SpecificCarrierList */
      for (int i = 0; i < p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.v.v.R15.scs_SpecificCarrierList.d; i++)
      {

        if (p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.d == true  &&
            p_req->v.AddOrReconfigure.PhysicalLayer.d == true )
        {
          RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]->carrierBandwidth=
            p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.v.v.R15.scs_SpecificCarrierList.v[i].carrierBandwidth;

          RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]->offsetToCarrier =
            p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.v.v.R15.scs_SpecificCarrierList.v[i].offsetToCarrier;

          RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]->subcarrierSpacing =
            p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.v.v.R15.scs_SpecificCarrierList.v[i].subcarrierSpacing;
          LOG_A(GNB_APP, 
              "fxn:%s UL scs_SpecificCarrierList.carrierBandwidth:%ld\n scs_SpecificCarrierList.offsetToCarrier:%ld\n scs_SpecificCarrierList.subcarrierSpacing:%ld", 
              __FUNCTION__, 
              RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]->carrierBandwidth,
              RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]->offsetToCarrier,
              RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->scs_SpecificCarrierList.list.array[i]->subcarrierSpacing);

          *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->p_Max =
            p_req->v.AddOrReconfigure.PhysicalLayer.v.Uplink.v.Uplink.v.v.Config.FrequencyInfoUL.v.v.R15.p_Max.v;
          LOG_A(GNB_APP, "fxn:%s UL p_Max :%ld\n", __FUNCTION__, 
              *RC.nrrrc[gnbId]->configuration[nr_cell_index].scc->uplinkConfigCommon->frequencyInfoUL->p_Max);
        }
      }


    }

   
    /*****************************************************************************/
    MessageDef *msg_p = NULL;
    msg_p = itti_alloc_new_message (TASK_SYS_GNB, 0, NRRRC_CONFIGURATION_REQ);
    LOG_I(GNB_APP,"ss_gNB Sending configuration message to NR_RRC task for cells\n");
    memcpy(&NRRRC_CONFIGURATION_REQ(msg_p), &RC.nrrrc[gnbId]->configuration,sizeof(NRRRC_CONFIGURATION_REQ(msg_p)));
    itti_send_msg_to_task (TASK_RRC_GNB, GNB_MODULE_ID_TO_INSTANCE(gnbId), msg_p);
  }
  return true;
}

/*
 * Function : cell_config_5G_done_indication
 * Description: Sends the cell_config_done_mutex signl to LTE_SOFTMODEM,
 * as in SS mode the eNB is waiting for the cell configration to be
 * received form TTCN. After receiving this signal only the eNB's init
 * is completed and its ready for processing.
 */
int cell_config_5G_done_indication()
{
  if (cell_config_5G_done < 0)
  {
    LOG_A(GNB_APP, "[SYS-GNB] fxn:%s Signal to TASK_GNB_APP about cell configuration complete", __FUNCTION__);
    pthread_mutex_lock(&cell_config_5G_done_mutex);
    cell_config_5G_done = 0;
    pthread_cond_broadcast(&cell_config_5G_done_cond);
    pthread_mutex_unlock(&cell_config_5G_done_mutex);
  }

  return 0;
}

/*
 * Function    : ss_task_sys_nr_handle_cellConfigRadioBearer
 * Description : This function handles the CellConfig 5G API on SYS Port and send processes the request.
 * Returns     : true/false
 */

bool ss_task_sys_nr_handle_cellConfigRadioBearer(struct NR_SYSTEM_CTRL_REQ *req)
{
  struct NR_RadioBearer_Type_NR_RadioBearerList_Type_Dynamic *BearerList =  &(req->Request.v.RadioBearerList);
  LOG_A(GNB_APP, "[SYS-GNB] Entry in fxn:%s\n", __FUNCTION__);
  MessageDef *msg_p = itti_alloc_new_message(TASK_SYS_GNB, 0, NRRRC_RBLIST_CFG_REQ);
  if (msg_p)
  {
    LOG_A(GNB_APP, "[SYS-GNB] BearerList size:%lu\n", BearerList->d);
    NRRRC_RBLIST_CFG_REQ(msg_p).rb_count = 0;
    NRRRC_RBLIST_CFG_REQ(msg_p).cell_index = get_gNB_cell_index(req->Common.CellId, SS_context.SSCell_list); //TODO: change to multicell index later
    
    for (int i = 0; i < BearerList->d; i++)
    {
      LOG_A(GNB_APP,"[SYS-GNB] RB Index i:%d\n", i);
      nr_rb_info * rb_info = &(NRRRC_RBLIST_CFG_REQ(msg_p).rb_list[i]);
      memset(rb_info, 0, sizeof(nr_rb_info));
      if (BearerList->v[i].Id.d == NR_RadioBearerId_Type_Srb)
      {
        rb_info->RbId = BearerList->v[i].Id.v.Srb;
      }
      else if (BearerList->v[i].Id.d == NR_RadioBearerId_Type_Drb)
      {
        rb_info->RbId = BearerList->v[i].Id.v.Drb + 2; // Added 2 for MAXSRB because DRB1 starts from index-3
      }

      NRRadioBearerConfig * rbConfig = &rb_info->RbConfig;
      if (BearerList->v[i].Config.d == NR_RadioBearerConfig_Type_AddOrReconfigure)
      {
        NRRRC_RBLIST_CFG_REQ(msg_p).rb_count++;
        /* Populate the SDAP Configuration for the radio Bearer */
        if(BearerList->v[i].Config.v.AddOrReconfigure.Sdap.d)
        {
          if(BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.d == SDAP_Configuration_Type_Config)
          {
            if(BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.d == SdapConfigInfo_Type_SdapConfig)
            {
              NR_SDAP_Config_t * sdap = CALLOC(1,sizeof(NR_SDAP_Config_t));
              sdap->pdu_Session =  BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.v.SdapConfig.Pdu_SessionId;
              if(BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.v.SdapConfig.Sdap_HeaderDL.d)
              {
                sdap->sdap_HeaderDL =  BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.v.SdapConfig.Sdap_HeaderDL.v;
              }
              if(BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.v.SdapConfig.MappedQoS_Flows.d)
              {
                sdap->mappedQoS_FlowsToAdd = CALLOC(1,sizeof(*sdap->mappedQoS_FlowsToAdd));
                for(int j=0; j < BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.v.SdapConfig.MappedQoS_Flows.v.d; j++)
                {
                  NR_QFI_t * qfi = CALLOC(1,sizeof(NR_QFI_t));
                  *qfi = BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.v.SdapConfig.MappedQoS_Flows.v.v[j];
                  ASN_SEQUENCE_ADD(&sdap->mappedQoS_FlowsToAdd->list,qfi);
                } 
              }
              rbConfig->Sdap = sdap;
            }else if(BearerList->v[i].Config.v.AddOrReconfigure.Sdap.v.v.Config.d == SdapConfigInfo_Type_TransparentMode){

            }
          }
         }

        /* Populate the PDCP Configuration for the radio Bearer */
        if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.d)
        {
          if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.d == NR_PDCP_Configuration_Type_RBTerminating)
          {
            if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.d)
            {
              if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.d == NR_PDCP_RbConfig_Type_Params)
              {
                if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.Params.Rb.d == NR_PDCP_RB_Config_Parameters_Type_Srb)
                {
                  LOG_A(GNB_APP,"[SYS-GNB] PDCP Config for Bearer Id: %d is Null\n", rb_info->RbId);
                }
                else if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.Params.Rb.d == NR_PDCP_RB_Config_Parameters_Type_Drb)
                {
                  NR_PDCP_Config_t *pdcp = CALLOC(1,sizeof(NR_PDCP_Config_t));
                  pdcp->drb = CALLOC(1, sizeof(*pdcp->drb));
                  pdcp->drb->pdcp_SN_SizeUL=CALLOC(1,sizeof(long));
                  *(pdcp->drb->pdcp_SN_SizeUL) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.Params.Rb.v.Drb.SN_SizeUL;
                  pdcp->drb->pdcp_SN_SizeDL=CALLOC(1,sizeof(long));
                  *(pdcp->drb->pdcp_SN_SizeDL) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.Params.Rb.v.Drb.SN_SizeDL;

                  if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.Params.Rb.v.Drb.IntegrityProtectionEnabled)
                  {
                    pdcp->drb->integrityProtection = CALLOC(1,sizeof(long));
                    *(pdcp->drb->integrityProtection) = 1;
                  }
                  //No rohc config for DRB from TTCN
                  if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.Params.Rb.v.Drb.HeaderCompression.d == NR_PDCP_DRB_HeaderCompression_Type_None)
                  {
                  }
                  rbConfig->Pdcp = pdcp;
                }
              }
              else if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.d == NR_PDCP_RbConfig_Type_TransparentMode)
              {
                rbConfig->pdcpTransparentSN_Size = CALLOC(1,sizeof(long));
                *(rbConfig->pdcpTransparentSN_Size) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.RbConfig.v.v.TransparentMode.SN_Size;
              }
            }

            if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.LinkToOtherCellGroup.d)
            {
              if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.LinkToOtherCellGroup.v.d == RlcBearerRouting_Type_EUTRA)
              {
                //BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.LinkToOtherCellGroup.v.v.EUTRA
              }
              else if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.LinkToOtherCellGroup.v.d == RlcBearerRouting_Type_NR)
              {
                //BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.RBTerminating.LinkToOtherCellGroup.v.v.NR
              }
            }
          }
          else if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.d == NR_PDCP_Configuration_Type_Proxy)
          {
            if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Proxy.LinkToOtherNode.d == RlcBearerRouting_Type_EUTRA)
            {
              //BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Proxy.LinkToOtherNode.v.EUTRA
            }
            else if(BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Proxy.LinkToOtherNode.d == RlcBearerRouting_Type_NR)
            {
              //BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Proxy.LinkToOtherNode.v.NR
            }
          }
        }

        /* Populate the RlcBearerConfig for the radio Bearer */
        if (BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.d)
        {
          NR_RLC_BearerConfig_t * rlcBearer = CALLOC(1,sizeof(NR_RLC_BearerConfig_t));
          if (BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.d == NR_RlcBearerConfig_Type_Config)
          {
            /* Populate the Rlc Config of RlcBearerConfig for the radio Bearer */
            if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.d)
            {
              if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.d)
              {
                NR_RLC_Config_t *rlc = CALLOC(1,sizeof(NR_RLC_Config_t));
                if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.d == NR_RLC_RbConfig_Type_AM)
                {
                  rlc->present = NR_RLC_Config_PR_am;
                  rlc->choice.am = CALLOC(1,sizeof(*rlc->choice.am));
                  if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.d)
                  {
                    if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.d == NR_ASN1_UL_AM_RLC_Type_R15)
                    {
                      if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.v.R15.sn_FieldLength.d)
                      {
                        rlc->choice.am->ul_AM_RLC.sn_FieldLength = CALLOC(1,sizeof(long));
                        *(rlc->choice.am->ul_AM_RLC.sn_FieldLength) =  BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.v.R15.sn_FieldLength.v;
                      }
                      rlc->choice.am->ul_AM_RLC.t_PollRetransmit = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.v.R15.t_PollRetransmit;
                      rlc->choice.am->ul_AM_RLC.pollPDU = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.v.R15.pollPDU;
                      rlc->choice.am->ul_AM_RLC.pollByte = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.v.R15.pollByte;
                      rlc->choice.am->ul_AM_RLC.maxRetxThreshold = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Tx.v.v.R15.maxRetxThreshold;
                    }
                  }
                  if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Rx.d)
                  {
                    if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Rx.v.d == NR_ASN1_DL_AM_RLC_Type_R15)
                    {
                      if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Rx.v.v.R15.sn_FieldLength.d)
                      {
                        rlc->choice.am->dl_AM_RLC.sn_FieldLength = CALLOC(1,sizeof(long));
                        *(rlc->choice.am->dl_AM_RLC.sn_FieldLength) = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Rx.v.v.R15.sn_FieldLength.v;
                      }
                      rlc->choice.am->dl_AM_RLC.t_Reassembly = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Rx.v.v.R15.t_Reassembly;
                      rlc->choice.am->dl_AM_RLC.t_StatusProhibit = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.AM.Rx.v.v.R15.t_StatusProhibit;
                    }
                  }
                }
                else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.d == NR_RLC_RbConfig_Type_UM)
                {
                  NR_UL_UM_RLC_t *ul_UM_RLC = NULL;
                  NR_DL_UM_RLC_t *dl_UM_RLC = NULL;
                  if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Tx.d && BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.d)
                  {
                    rlc->present = NR_RLC_Config_PR_um_Bi_Directional;
                    rlc->choice.um_Bi_Directional = CALLOC(1,sizeof(*rlc->choice.um_Bi_Directional));
                    ul_UM_RLC = &rlc->choice.um_Bi_Directional->ul_UM_RLC;
                    dl_UM_RLC = &rlc->choice.um_Bi_Directional->dl_UM_RLC;
                  }
                  else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Tx.d)
                  {
                    rlc->present = NR_RLC_Config_PR_um_Uni_Directional_UL;
                    rlc->choice.um_Uni_Directional_UL = CALLOC(1,sizeof(*rlc->choice.um_Uni_Directional_UL));
                    ul_UM_RLC = &rlc->choice.um_Uni_Directional_UL->ul_UM_RLC;
                  }
                  else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.d)
                  {
                    rlc->present = NR_RLC_Config_PR_um_Uni_Directional_DL;
                    rlc->choice.um_Uni_Directional_DL = CALLOC(1,sizeof(*rlc->choice.um_Uni_Directional_DL));
                    dl_UM_RLC = &rlc->choice.um_Uni_Directional_DL->dl_UM_RLC;
                  }

                  if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Tx.d)
                  {
                    if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Tx.v.d == NR_ASN1_UL_UM_RLC_Type_R15)
                    {
                      if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Tx.v.v.R15.sn_FieldLength.d)
                      {
                        ul_UM_RLC->sn_FieldLength = CALLOC(1,sizeof(long));
                        *(ul_UM_RLC->sn_FieldLength) = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Tx.v.v.R15.sn_FieldLength.v;
                      }
                    }
                  }
                  if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.d)
                  {
                     if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.v.d == NR_ASN1_DL_UM_RLC_Type_R15)
                    {
                      if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.v.v.R15.sn_FieldLength.d)
                      {
                        dl_UM_RLC->sn_FieldLength = CALLOC(1,sizeof(long));
                        *(dl_UM_RLC->sn_FieldLength) = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.v.v.R15.sn_FieldLength.v;
                      }
                      dl_UM_RLC->t_Reassembly = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.UM.Rx.v.v.R15.t_Reassembly;
                    }
                  }
                }
                else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.d == NR_RLC_RbConfig_Type_TM)
                {
                  //TODO: TransparentMode
                  // It is a workaround to provide SN size
                  if (BearerList->v[i].Id.d == NR_RadioBearerId_Type_Drb)
                  {
                    if (BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.Rb.v.v.TM && BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.d)
                    {
                      if (BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.v.Info.d == NR_RLC_TestModeInfo_Type_TransparentMode)
                      {
                        if (BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.v.Info.v.TransparentMode.d == NR_RLC_TransparentMode_Umd)
                        {
                          rlc->present = NR_RLC_Config_PR_um_Bi_Directional;
                          rlc->choice.um_Bi_Directional = CALLOC(1,sizeof(*rlc->choice.um_Bi_Directional));
                          NR_UL_UM_RLC_t *ul_UM_RLC = &rlc->choice.um_Bi_Directional->ul_UM_RLC;
                          NR_DL_UM_RLC_t *dl_UM_RLC = &rlc->choice.um_Bi_Directional->dl_UM_RLC;
                          ul_UM_RLC->sn_FieldLength = CALLOC(1,sizeof(long));
                          dl_UM_RLC->sn_FieldLength = CALLOC(1,sizeof(long));
                          *(ul_UM_RLC->sn_FieldLength) = *(dl_UM_RLC->sn_FieldLength) = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.v.Info.v.TransparentMode.v.Umd;
                        }
                      }
                    }
                  }
                }
                else
                {
                  rlc->present = NR_RLC_Config_PR_NOTHING;
                }
                rlcBearer->rlc_Config = rlc;
              }
              if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.d)
              {
                //TODO: RLC TestMode
                if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.d == NR_RLC_TestModeConfig_Type_Info)
                {
                  if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.v.Info.d == NR_RLC_TestModeInfo_Type_AckProhibit)
                  {
                  }
                  else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.v.Info.d == NR_RLC_TestModeInfo_Type_NotACK_NextRLC_PDU)
                  {
                  }
                  else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.v.Info.d == NR_RLC_TestModeInfo_Type_TransparentMode)
                  {
                  }
                }
                else if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Rlc.v.TestMode.v.d == NR_RLC_TestModeConfig_Type_None)
                {
                }
              }
            }
            /* Populate the LogicalChannelId of RlcBearerConfig for the radio Bearer */
            if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.LogicalChannelId.d)
            {
              rlcBearer->logicalChannelIdentity = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.LogicalChannelId.v;
            }

            /* Populate the LogicalChannelConfig of RlcBearerConfig for the radio Bearer */
            if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Mac.d)
            {
              if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Mac.v.LogicalChannel.d)
              {
                NR_LogicalChannelConfig_t * logicalChannelConfig = CALLOC(1,sizeof(NR_LogicalChannelConfig_t));
                logicalChannelConfig->ul_SpecificParameters = CALLOC(1,sizeof(*logicalChannelConfig->ul_SpecificParameters));
                logicalChannelConfig->ul_SpecificParameters->priority =  BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Mac.v.LogicalChannel.v.Priority;
                logicalChannelConfig->ul_SpecificParameters->prioritisedBitRate = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Mac.v.LogicalChannel.v.PrioritizedBitRate;
                rlcBearer->mac_LogicalChannelConfig = logicalChannelConfig;
              }
              if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.Mac.v.TestMode.d)
              {
                //TODO: MAC test mode
              }
            }

            /* Populate the DiscardULData of RlcBearerConfig for the radio Bearer */
            if(BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.DiscardULData.d)
            {
              //typically applicable for UM DRBs only: if true: SS shall discard any data in UL for this radio bearer
              rbConfig->DiscardULData =CALLOC(1,sizeof(bool));
              *(rbConfig->DiscardULData) = BearerList->v[i].Config.v.AddOrReconfigure.RlcBearer.v.v.Config.DiscardULData.v;
            }
          }
          rbConfig->RlcBearer = rlcBearer;
        }
      }
    }
    LOG_A(GNB_APP, "[SYS-GNB] Send NRRRC_RBLIST_CFG_REQ to TASK_RRC_GNB, RB Count : %d, Message: %s  \n", NRRRC_RBLIST_CFG_REQ(msg_p).rb_count, ITTI_MSG_NAME(msg_p));
    int send_res = itti_send_msg_to_task(TASK_RRC_GNB, 0, msg_p);
    if (send_res < 0)
    {
      LOG_A(GNB_APP, "[SYS-GNB] Error sending NRRRC_RBLIST_CFG_REQ to RRC_GNB \n");
    }

  }
  if ( req->Common.ControlInfo.CnfFlag) {
    send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_RadioBearerList, NULL);
  }
  LOG_A(GNB_APP, "[SYS-GNB] Exit from fxn:%s\n", __FUNCTION__);
  return true;
}

/*
 * Function    : ss_task_sys_nr_handle_cellConfigAttenuation
 * Description : This function handles the CellConfig 5G API on SYS Port for request type CellAttenuation and send processes the request.
 * Returns     : true/false
 */

bool ss_task_sys_nr_handle_cellConfigAttenuation(struct NR_SYSTEM_CTRL_REQ *req)
{
  LOG_A(GNB_APP, "[SYS-GNB] Entry in fxn:%s\n", __FUNCTION__);
  struct NR_SYSTEM_CTRL_CNF *msgCnf = CALLOC(1, sizeof(struct NR_SYSTEM_CTRL_CNF));
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS_GNB, INSTANCE_DEFAULT, SS_NR_SYS_PORT_MSG_CNF);

  if (message_p)
  {
    LOG_A(GNB_APP, "[SYS-GNB] Send SS_NR_SYS_PORT_MSG_CNF\n");
    msgCnf->Common.CellId = nr_Cell_NonSpecific;
    msgCnf->Common.Result.d = ConfirmationResult_Type_Success;
    msgCnf->Common.Result.v.Success = true;
    msgCnf->Confirm.d = NR_SystemConfirm_Type_CellAttenuationList;
    msgCnf->Confirm.v.CellAttenuationList = true;

    SS_NR_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN_GNB, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(GNB_APP, "[SYS-GNB] Error sending to [SS-PORTMAN-GNB]\n");
      return false;
    }
    else
    {
      LOG_A(GNB_APP, "[SYS-GNB] fxn:%s NR_SYSTEM_CTRL_CNF sent for cnfType:CellAttenuationList to Port Manager\n", __FUNCTION__);
    }
  }
  LOG_A(GNB_APP, "[SYS-GNB] Exit from fxn:%s\n", __FUNCTION__);
  return true;
}



extern nr_pdcp_ue_manager_t *nr_pdcp_ue_manager; /**< NR-PDCP doesn't suupport ITTI messages like it was done in eNB-PDCP*/


/**
 * @brief get rb by id from nr_pdcp_ue
 *
 * @param ue
 * @param srb
 * @param rb_id
 * @return nr_pdcp_entity_t*
 */
static nr_pdcp_entity_t * ss_task_sys_get_rb(nr_pdcp_ue_t *ue, bool srb, uint16_t rb_id)
{
    nr_pdcp_entity_t * rb;
    if (srb) {
      if (rb_id < 1 || rb_id > 2)
        rb = NULL;
      else
        rb = ue->srb[rb_id - 1];
    } else {
      if (rb_id < 1 || rb_id > 5)
        rb = NULL;
      else
        rb = ue->drb[rb_id - 1];
    }

    return rb;
}

/**
 * @brief Fill on PDCP count struct
 * @see struct NR_PdcpCountInfo_Type
 *
 * @param v
 * @param ue
 * @param isSrb
 * @param rbId
 */
static bool ss_task_sys_fill_pdcp_cnt_rb(struct NR_PdcpCountInfo_Type* v, nr_pdcp_ue_t *ue, bool isSrb, uint8_t rbId)
{
  if (rbId == 0) {
    return false;
  }

  nr_pdcp_entity_t* rb = ss_task_sys_get_rb(ue, isSrb, rbId);
  if (rb == NULL)
  {
    LOG_E(GNB_APP, "%s rbrId is NULL: %id\r\n", isSrb ? "SRB": "DRB", rbId);
    static nr_pdcp_entity_t _rb = {};
    rb = &_rb;
  }

  if (isSrb)
  {
    v->RadioBearerId.d = NR_RadioBearerId_Type_Srb;
    v->RadioBearerId.v.Srb = rbId;
    v->UL.v.Format = E_PdcpCount_Srb; // E_NrPdcpCount_Srb;
    v->DL.v.Format = E_PdcpCount_Srb; // E_NrPdcpCount_Srb;
  }
  else
  {
    v->RadioBearerId.d = NR_RadioBearerId_Type_Drb;
    v->RadioBearerId.v.Drb = rbId;
    v->UL.v.Format = E_PdcpCount_DrbShortSQN; // E_NrPdcpCount_DrbSQN12;
    v->DL.v.Format = E_PdcpCount_DrbShortSQN; // E_NrPdcpCount_DrbSQN12;
  }

  v->UL.d = true;
  v->DL.d = true;

  uint32_t ul_cnt = rb->rx_next - 1;
  uint32_t dl_cnt = rb->tx_next - 1;

  LOG_A(GNB_APP, "PDCP-Count srb:%d rb:%d SDU(UL)(rx): %d\n", isSrb, rbId, ul_cnt);
  LOG_A(GNB_APP, "PDCP-Count srb:%d rb:%d PDU(DL)(tx): %d\n", isSrb, rbId, dl_cnt);

  int_to_bin(ul_cnt, 32, v->UL.v.Value);
  int_to_bin(dl_cnt, 32, v->DL.v.Value);

  return true;
}

/**
 * @brief Send PDCP count confirmation
 *
 * @param req
 * @return true
 * @return false
 */
bool ss_task_sys_nr_handle_pdcpCount(struct NR_SYSTEM_CTRL_REQ *req)
{
  (void)req;

  uint16_t rnti = SS_context.SSCell_list[nr_cell_index].ss_rnti_g;
  //uint16_t rnti = SS_context.ss_rnti_g;
  if (!req->Common.ControlInfo.CnfFlag) {
    return true;
  }

  if (req->Request.v.PdcpCount.d == NR_PDCP_CountReq_Type_Get)
  {
      struct NR_PDCP_CountCnf_Type PdcpCount = {};
      PdcpCount.d = NR_PDCP_CountCnf_Type_Get;
      ue_id_t UEid = rnti;
      nr_pdcp_ue_t *ue = nr_pdcp_manager_get_ue_ex(nr_pdcp_ue_manager, UEid);

      if (ue == NULL)
      {
        LOG_E(GNB_APP, "could not found suitable UE with rnti: %d\r\n", rnti);

        // TODO: FIX
        PdcpCount.v.Get.d = 1;
        const size_t size = sizeof(struct NR_PdcpCountInfo_Type) * PdcpCount.v.Get.d;
        PdcpCount.v.Get.v = (struct NR_PdcpCountInfo_Type *)acpMalloc(size);
        PdcpCount.v.Get.v[0].RadioBearerId.d = NR_RadioBearerId_Type_Srb;
        PdcpCount.v.Get.v[0].RadioBearerId.v.Srb = 0;
        PdcpCount.v.Get.v[0].UL.d = true;
        PdcpCount.v.Get.v[0].DL.d = true;
        PdcpCount.v.Get.v[0].UL.v.Format = NR_PdcpCount_Srb;
        PdcpCount.v.Get.v[0].DL.v.Format = NR_PdcpCount_Srb;
        int_to_bin(0, 32, PdcpCount.v.Get.v[0].UL.v.Value);
        int_to_bin(0, 32, PdcpCount.v.Get.v[0].DL.v.Value);
        send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_PdcpCount, (void *)&PdcpCount);

        return false;
      }

      if (req->Request.v.PdcpCount.v.Get.d == NR_PdcpCountGetReq_Type_AllRBs)
      {
        PdcpCount.v.Get.d = 5;
        const size_t size = sizeof(struct NR_PdcpCountInfo_Type) * PdcpCount.v.Get.d;
        PdcpCount.v.Get.v =(struct NR_PdcpCountInfo_Type *)acpMalloc(size);
        if (!ss_task_sys_fill_pdcp_cnt_rb(PdcpCount.v.Get.v, ue, true, 1))
        {
          LOG_E(GNB_APP, "could not found suitable SRB RB \r\n");
          acpFree(PdcpCount.v.Get.v);
          return false;
        }

        for(uint8_t i = 1; i< PdcpCount.v.Get.d; i++) // about magic number 5 @see do_pdcp_data_ind() where it max_drb also 5
        {
          if(!ss_task_sys_fill_pdcp_cnt_rb(&PdcpCount.v.Get.v[i], ue, false, i))
          {
            LOG_E(GNB_APP, "DRB %i is null \r\n", i);
            acpFree(PdcpCount.v.Get.v);
            return false;
          }
        }

      }
      else if (req->Request.v.PdcpCount.v.Get.d == NR_PdcpCountGetReq_Type_SingleRB)
      {
          PdcpCount.v.Get.d = 1;

          PdcpCount.v.Get.v =(struct NR_PdcpCountInfo_Type *)acpMalloc(sizeof(struct NR_PdcpCountInfo_Type));
          uint8_t rbId = req->Request.v.PdcpCount.v.Get.v.SingleRB.d == NR_RadioBearerId_Type_Srb ? req->Request.v.PdcpCount.v.Get.v.SingleRB.v.Srb
                : req->Request.v.PdcpCount.v.Get.v.SingleRB.d == NR_RadioBearerId_Type_Drb ? req->Request.v.PdcpCount.v.Get.v.SingleRB.v.Drb : 0;

          if(!ss_task_sys_fill_pdcp_cnt_rb(PdcpCount.v.Get.v, ue, req->Request.v.PdcpCount.v.Get.v.SingleRB.d == NR_RadioBearerId_Type_Srb, rbId))
          {
            LOG_E(GNB_APP, "could not found suitable RB %d\r\n", rbId);
            acpFree(PdcpCount.v.Get.v);
            return false;
          }
      }
      else
      {
        LOG_E(GNB_APP, "%s line:%d it's not an PdcpCount.v.Get for single-rb not all-rbs cmd\r\n", __PRETTY_FUNCTION__, __LINE__);
        return false;
      }

      send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_PdcpCount, (void *)&PdcpCount);
      LOG_A(GNB_APP, "Exit from fxn:%s\n", __FUNCTION__);
      return true;
  }
  else if (req->Request.v.PdcpCount.d == NR_PDCP_CountReq_Type_Set)
  {
      send_sys_cnf(ConfirmationResult_Type_Success, true, NR_SystemConfirm_Type_PdcpCount, NULL);
      return true;
  }
  else
  {
    LOG_E(GNB_APP, "%s:%d it's nor a get nor set cmd\r\n", __PRETTY_FUNCTION__, __LINE__);
    return false;
  }

  return false;
}

/*
 * Function : sys_5G_send_proxy
 * Description: Sends the messages from SYS to proxy
 */
static void sys_5G_send_proxy(void *msg, int msgLen)
{
  LOG_A(GNB_APP, "Entry in %s\n", __FUNCTION__);
  uint32_t peerIpAddr = 0;
  uint16_t peerPort = proxy_5G_send_port;

  IPV4_STR_ADDR_TO_INT_NWBO(local_5G_address,peerIpAddr, " BAD IP Address");

  LOG_A(GNB_APP, "Sending CELL CONFIG 5G to Proxy\n");

  /** Send to proxy */
  sys_send_udp_msg((uint8_t *)msg, msgLen, 0, peerIpAddr, peerPort);
  LOG_A(GNB_APP, "Exit from %s\n", __FUNCTION__);
  return;
}


/*
 * Function : sys_send_udp_msg
 * Description: Sends the UDP_INIT message to UDP_TASK to create the listening socket
 */
static int sys_send_udp_msg(
    uint8_t *buffer,
    uint32_t buffer_len,
    uint32_t buffer_offset,
    uint32_t peerIpAddr,
    uint16_t peerPort)
{
  // Create and alloc new message
  MessageDef *message_p = NULL;
  udp_data_req_t *udp_data_req_p = NULL;
  message_p = itti_alloc_new_message(TASK_SYS_GNB, 0, UDP_DATA_REQ);

  if (message_p)
  {
    LOG_A(GNB_APP, "Sending UDP_DATA_REQ length %u offset %u buffer %d %d %d \n", buffer_len, buffer_offset, buffer[0], buffer[1], buffer[2]);
    udp_data_req_p = &message_p->ittiMsg.udp_data_req;
    udp_data_req_p->peer_address = peerIpAddr;
    udp_data_req_p->peer_port = peerPort;
    udp_data_req_p->buffer = buffer;
    udp_data_req_p->buffer_length = buffer_len;
    udp_data_req_p->buffer_offset = buffer_offset;
    return itti_send_msg_to_task(TASK_UDP, 0, message_p);
  }
  else
  {
    LOG_A(GNB_APP, "Failed Sending UDP_DATA_REQ length %u offset %u \n", buffer_len, buffer_offset);
    return -1;
  }
}

/*
 * Function : sys_send_init_udp
 * Description: Sends the UDP_INIT message to UDP_TASK to create the receiving socket
 * for the SYS_TASK from the Proxy for the configuration confirmations.
 */
static int sys_5G_send_init_udp(const udpSockReq_t *req)
{
  // Create and alloc new message
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_SYS_GNB, 0, UDP_INIT);
  if (message_p == NULL)
  {
    return -1;
  }
  UDP_INIT(message_p).port = req->port;
  //addr.s_addr = req->ss_ip_addr;
  UDP_INIT(message_p).address = req->address; //inet_ntoa(addr);
  LOG_A(GNB_APP, "Tx UDP_INIT IP addr %s (%x)\n", UDP_INIT(message_p).address, UDP_INIT(message_p).port);
  return itti_send_msg_to_task(TASK_UDP, 0, message_p);
}

