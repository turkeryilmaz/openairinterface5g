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

#include "acpSys.h"
#include "ss_gNB_sys_task.h"
#include "ss_gNB_context.h"

#include "udp_eNB_task.h"
#include "ss_eNB_proxy_iface.h"
#include "common/utils/LOG/ss-log.h"
#include "msc.h"

extern RAN_CONTEXT_t RC;
extern uint32_t from_earfcn(int eutra_bandP, uint32_t dl_earfcn);
extern pthread_cond_t cell_config_5g_done_cond;
extern pthread_mutex_t cell_config_5g_done_mutex;
extern int cell_config_5g_done;

extern uint16_t ss_rnti_nr_g;
static void sys_send_proxy(void *msg, int msgLen);
int cell_config_done_indication(void);
static uint16_t paging_ue_index_g = 0;
extern SSConfigContext_t SS_context;
typedef enum
{
  UndefinedMsg = 0,
  EnquireTiming = 1,
  CellConfig = 2
} sidl_msg_id;

char *local_address = "127.0.0.1";
int proxy_send_port = 7776;
int proxy_recv_port = 7770;
bool reqCnfFlag_g = false;
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

static int32_t bin_to_int(uint8_t array[], uint32_t len)
{
  int output = 0;
  int power = 1;

  for (int i = 0; i < len; i++)
  {
    output += array[(len - 1) - i] * power;
    // output goes 1*2^0 + 0*2^1 + 0*2^2 + ...
    power *= 2;
  }

  return output;
}

static void bitStrint_to_byteArray(unsigned char arr[], int bit_length, unsigned char *key, bool int_key)
{
  int len = 8;
  int byte_count = bit_length/len;
  int count = byte_count/2;
  if(int_key == TRUE)
  {
    for(int i=0;i<byte_count/2;i++)
    {
      unsigned long int output = 0;
      int power = 1;
      unsigned char *array = arr+8*i;
      for (int j = 0; j < len; j++)
      {
        output += array[(len - 1) - j] * power;
        // output goes 1*2^0 + 0*2^1 + 0*2^2 + ...
        power *= 2;
      }
      key[count] = output;
      count++;
    }
  }
  else
  {
    for(int i=0;i<byte_count;i++)
    {
      unsigned long int output = 0;
      int power = 1;
      unsigned char *array = arr+8*i;
      for (int j = 0; j < len; j++)
      {
        output += array[(len - 1) - j] * power;
        // output goes 1*2^0 + 0*2^1 + 0*2^2 + ...
        power *= 2;
      }
      key[i] = output;
    }
  }
}

/*
 * Function : cell_config_done_indication
 * Description: Sends the cell_config_done_mutex signl to LTE_SOFTMODEM,
 * as in SS mode the eNB is waiting for the cell configration to be
 * received form TTCN. After receiving this signal only the eNB's init
 * is completed and its ready for processing.
 */
int cell_config_5g_done_indication()
{

  if (cell_config_5g_done < 0)
  {
    printf("[SYS] Signal to OAI main code about cell config\n");
    pthread_mutex_lock(&cell_config_5g_done_mutex);
    cell_config_5g_done = 0;
    pthread_cond_broadcast(&cell_config_5g_done_cond);
    pthread_mutex_unlock(&cell_config_5g_done_mutex);
  }

  return 0;
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
  message_p = itti_alloc_new_message(TASK_SYS, INSTANCE_DEFAULT, UDP_DATA_REQ);

  if (message_p)
  {
    LOG_A(RRC, "Sending UDP_DATA_REQ length %u offset %u buffer %d %d %d", buffer_len, buffer_offset, buffer[0], buffer[1], buffer[2]);
    udp_data_req_p = &message_p->ittiMsg.udp_data_req;
    udp_data_req_p->peer_address = peerIpAddr;
    udp_data_req_p->peer_port = peerPort;
    udp_data_req_p->buffer = buffer;
    udp_data_req_p->buffer_length = buffer_len;
    udp_data_req_p->buffer_offset = buffer_offset;
    return itti_send_msg_to_task(TASK_UDP, INSTANCE_DEFAULT, message_p);
  }
  else
  {
    LOG_A(RRC, "Failed Sending UDP_DATA_REQ length %u offset %u", buffer_len, buffer_offset);
    return -1;
  }
}

/*
 * Function : sys_send_init_udp
 * Description: Sends the UDP_INIT message to UDP_TASK to create the receiving socket
 * for the SYS_TASK from the Proxy for the configuration confirmations.
 */
static int sys_send_init_udp(const udpSockReq_t *req)
{
  // Create and alloc new message
  MessageDef *message_p;
  message_p = itti_alloc_new_message(TASK_SYS, INSTANCE_DEFAULT, UDP_INIT);
  if (message_p == NULL)
  {
    return -1;
  }
  UDP_INIT(message_p).port = req->port;
  //addr.s_addr = req->ss_ip_addr;
  UDP_INIT(message_p).address = req->address; //inet_ntoa(addr);
  LOG_A(RRC, "Tx UDP_INIT IP addr %s (%x)\n", UDP_INIT(message_p).address, UDP_INIT(message_p).port);
  MSC_LOG_EVENT(
      MSC_GTPU_ENB,
      "0 UDP bind  %s:%u",
      UDP_INIT(message_p).address,
      UDP_INIT(message_p).port);
  return itti_send_msg_to_task(TASK_UDP, INSTANCE_DEFAULT, message_p);
}

static void ss_task_sys_handle_timing_info(ss_set_timinfo_t *tinfo)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, INSTANCE_DEFAULT, SS_SET_TIM_INFO);
  if (message_p)
  {
    LOG_A(RRC, "[SYS] Reporting info sfn:%d\t sf:%d.\n", tinfo->sfn, tinfo->sf);
    SS_SET_TIM_INFO(message_p).sf = tinfo->sf;
    SS_SET_TIM_INFO(message_p).sfn = tinfo->sfn;

    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending to [SS-PORTMAN]");
    }
  }
}
#if 0
/* 
 * =========================================================================================================== 
 * Function Name: sys_add_nr_reconfig_cell
 * Parameter    : CellConfigInfo_Type *AddOrReconfigure, is the structure having cell configuration information 
 *                (received from TTCN)as per ASP Defination of NR_CellConfigRequest_Type (38.523-3).
 * Description  : This function handler for handling NR Cell Configuration Request and updates the respective
 *                structure in RAN Context and builds the RRC config message with the received cell configruation 
 *                and sends itti message to RRC layer.
 * Returns      : TRUE/FALSE
 * ==========================================================================================================
*/
int sys_add_nr_reconfig_cell(struct CellConfigInfo_Type *AddOrReconfigure)
{

	if (AddOrReconfigure->Basic.d == false && AddOrReconfigure->Active.d == false)
		return false;

	uint8_t num_CC = 0; /** NOTE: Handling only one cell */
	for (int gnb_id = 0; gnb_id < RC.nb_nr_inst; gnb_id++)
	{
		MessageDef *msg_p = itti_alloc_new_message(TASK_SYS_5G_NR, GNB_MODULE_ID_TO_INSTANCE(gnb_id), NRRRC_CONFIGURATION_REQ);
		NRRRC_CONFIGURATION_REQ(msg_p) = RC.nrrrc[gnb_id]->configuration;

		if (AddOrReconfigure->Basic.d == true)
		{
			if (AddOrReconfigure->Basic.v.StaticCellInfo.d == true)
			{
				/** Handle Static Cell Info */
				/** TDD: 1 FDD: 0 in OAI */
				switch (AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.RAT.d)
				{
					case EUTRA_RAT_Type_FDD:
						RRC_CONFIGURATION_REQ(msg_p).frame_type[num_CC] = 0; /** FDD */
						break;
					case EUTRA_RAT_Type_TDD:
						RRC_CONFIGURATION_REQ(msg_p).frame_type[num_CC] = 1; /** TDD */
						break;
					case EUTRA_RAT_Type_HalfDuplexFDD:
					case EUTRA_RAT_Type_UNBOUND_VALUE:
						/* LOG */
						return false;
				}

				int band = AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.EutraBand;
				RRC_CONFIGURATION_REQ(msg_p).eutra_band[num_CC] = band;
				RRC_CONFIGURATION_REQ(msg_p).Nid_cell[num_CC] = AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.PhysicalCellId;
				/* TODO: 5G_cell_config start */
				//SS_context.cellId = AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.PhysicalCellId;
				/* 5G_cell_config end */

				/** TODO: Not filled now */
				/** eNB Cell ID: AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.eNB_CellId.v */
				/** CellTimingInfo: */


				uint32_t dl_Freq = from_earfcn(band, AddOrReconfigure->Basic.v.StaticCellInfo.v.Downlink.Earfcn);
				RRC_CONFIGURATION_REQ(msg_p).downlink_frequency[num_CC] = dl_Freq;
				if (AddOrReconfigure->Basic.v.StaticCellInfo.v.Uplink.d == true)
				{
					uint32_t ul_Freq = from_earfcn(band, AddOrReconfigure->Basic.v.StaticCellInfo.v.Uplink.v.Earfcn);
					int ul_Freq_off = ul_Freq - dl_Freq;
					RRC_CONFIGURATION_REQ(msg_p).uplink_frequency_offset[num_CC] = (unsigned int)ul_Freq_off;
				/* TODO: 5G_cell_config start */
					//SS_context.ul_earfcn = AddOrReconfigure->Basic.v.StaticCellInfo.v.Uplink.v.Earfcn;
					//SS_context.ul_freq = ul_Freq;
				/* 5G_cell_config end */
				}
				// Updated the SS context for the frequency related configuration
				/* TODO: 5G_cell_config start */
				//SS_context.dl_earfcn = AddOrReconfigure->Basic.v.StaticCellInfo.v.Downlink.Earfcn;
				//SS_context.dl_freq = dl_Freq;
				/* 5G_cell_config end */

				switch (AddOrReconfigure->Basic.v.StaticCellInfo.v.Downlink.Bandwidth)
				{
					case CarrierBandwidthEUTRA_dl_Bandwidth_e_n6:
						RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[num_CC] = 6;
						break;
					case CarrierBandwidthEUTRA_dl_Bandwidth_e_n15:
						RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[num_CC] = 15;
						break;
					case CarrierBandwidthEUTRA_dl_Bandwidth_e_n25:
						RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[num_CC] = 25;
						break;
					case CarrierBandwidthEUTRA_dl_Bandwidth_e_n50:
						RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[num_CC] = 50;
						break;
					case CarrierBandwidthEUTRA_dl_Bandwidth_e_n75:
						RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[num_CC] = 75;
						break;
					case CarrierBandwidthEUTRA_dl_Bandwidth_e_n100:
						RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[num_CC] = 100;
						break;
					default:
						/** LOG */
						LOG_A(RRC, "[SYS] CellConfigRequest Invalid DL Bandwidth configuration");
						return false;
				}
			}
#define BCCH_CONFIG AddOrReconfigure->Basic.v.BcchConfig
			if (AddOrReconfigure->Basic.v.BcchConfig.d == true)
			{
				LOG_A (RRC, "[SYS] BCCH Config update in Cell config \n");
				if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.d == true)
				{
					if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.d == true)
					{

						/** For MIB */
						LOG_A(RRC, "[SYS] CellConfigRequest PHICH Duration: %d\n", BCCH_CONFIG.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Duration);
						switch (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Duration)
						{
							case PHICH_Config_phich_Duration_e_normal:
								RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[num_CC].phich_duration = LTE_PHICH_Config__phich_Duration_normal;
								break;
							case PHICH_Config_phich_Duration_e_extended:
								RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[num_CC].phich_duration = LTE_PHICH_Config__phich_Duration_extended;
								break;
							default:
								LOG_A(RRC, "[SYS] CellConfigRequest Invalid PHICH Duration\n");
								return false;
						}

						LOG_A(RRC, "[SYS] CellConfigRequest PHICH Resource: %d\n", BCCH_CONFIG.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Resource);
						switch (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Resource)
						{
							case PHICH_Config_phich_Resource_e_oneSixth:
								RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[num_CC].phich_resource = LTE_PHICH_Config__phich_Resource_oneSixth;
								break;
							case PHICH_Config_phich_Resource_e_half:
								RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[num_CC].phich_resource = LTE_PHICH_Config__phich_Resource_half;
								break;
							case PHICH_Config_phich_Resource_e_one:
								RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[num_CC].phich_resource = LTE_PHICH_Config__phich_Resource_one;
								break;
							case PHICH_Config_phich_Resource_e_two:
								RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[num_CC].phich_resource = LTE_PHICH_Config__phich_Resource_two;
								break;
							default:
								LOG_A(RRC, "[SYS] CellConfigRequest Invalid PHICH Resource\n");
								return false;
						}

						RRC_CONFIGURATION_REQ(msg_p).schedulingInfoSIB1_BR_r13[num_CC] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.v.message.schedulingInfoSIB1_BR_r13;
					}
					/** TODO: FIXME: Possible bug if not checking boolean flag for presence */
#define SIDL_SIB1_VAL AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIB1.v.message.v
#define SIB1_CELL_ACCESS_REL_INFO SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.cellAccessRelatedInfo
#define SIB1_CELL_SEL_INFO SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.cellSelectionInfo
#define SIB1_CELL_NON_CE SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.nonCriticalExtension.v.nonCriticalExtension
#define SIB1_CELL_Q_QUALMIN SIB1_CELL_NON_CE.v.cellSelectionInfo_v920.v.q_QualMin_r9
					if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIB1.d == true)
					{
						LOG_A(RRC, "[SYS] [SIB1] q-RxLevMin: %d \n", SIB1_CELL_SEL_INFO.q_RxLevMin);
						RRC_CONFIGURATION_REQ(msg_p).q_RxLevMin = SIB1_CELL_SEL_INFO.q_RxLevMin;
						if (SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.nonCriticalExtension.d)
						{
							LOG_A(RRC, "[SYS] [SIB1] q-QualMin: %d \n", SIB1_CELL_Q_QUALMIN);
							RRC_CONFIGURATION_REQ(msg_p).q_QualMin = SIB1_CELL_Q_QUALMIN;
						}
					}

					RRC_CONFIGURATION_REQ(msg_p).num_plmn = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.d;

					for (int i = 0; i < RRC_CONFIGURATION_REQ(msg_p).num_plmn; ++i) {
						if(SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mcc.d == TRUE)
						{
							RRC_CONFIGURATION_REQ(msg_p).mcc[i] = (((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mcc.v[0])<<16) | ((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mcc.v[1])<<8) | ((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mcc.v[2])<<0));
						}
						if(SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.d == 2) {
							RRC_CONFIGURATION_REQ(msg_p).mnc[i] = (((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.v[0])<<8) | ((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.v[1])<<0));
						} else if(SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.d == 3) {
							RRC_CONFIGURATION_REQ(msg_p).mnc[i] = (((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.v[0])<<16) | ((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.v[1])<<8) | ((SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v->plmn_Identity.mnc.v[2])<<0));
						}
					}

				}
			}
			LOG_A(RRC, "Sending Cell configuration to RRC from SYSTEM_CTRL_REQ");
			itti_send_msg_to_task(TASK_RRC_ENB, ENB_MODULE_ID_TO_INSTANCE(gnb_id), msg_p);
		}
	}
	return true;
}
#endif
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
                         enum SystemConfirm_Type_Sel cnfType,
                         void *msg)
{
  struct SYSTEM_CTRL_CNF *msgCnf = CALLOC(1, sizeof(struct SYSTEM_CTRL_CNF));
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, INSTANCE_DEFAULT, SS_SYS_PORT_MSG_CNF);

  /* The request has send confirm flag flase so do nothing in this funciton */
  if (reqCnfFlag_g == FALSE)
  {
     LOG_A(RRC, "[SYS] No confirm required\n");
     return ;
  }

  if (message_p)
  {
    LOG_A(RRC, "[SYS] Send SS_SYS_PORT_MSG_CNF\n");
    msgCnf->Common.CellId = SS_context.eutra_cellId;
    msgCnf->Common.Result.d = resType;
    msgCnf->Common.Result.v.Success = resVal;
    msgCnf->Confirm.d = cnfType;
    switch (cnfType)
    {
    case SystemConfirm_Type_Cell:
    {
      LOG_A(RRC, "[SYS] Send confirm for cell configuration\n");
      msgCnf->Confirm.v.Cell = true;
      break;
    }
    case SystemConfirm_Type_CellAttenuationList:
    {
      msgCnf->Confirm.v.CellAttenuationList = true;
      break;
    }
    case SystemConfirm_Type_RadioBearerList:
      msgCnf->Confirm.v.RadioBearerList = true;
      break;
    case SystemConfirm_Type_AS_Security:
      msgCnf->Confirm.v.AS_Security = true;
      break;
    case SystemConfirm_Type_UE_Cat_Info:
      msgCnf->Confirm.v.UE_Cat_Info = true;
      break;
    case SystemConfirm_Type_PdcpCount:
#if 0
      if (msg)
      memcpy(&msgCnf->Confirm.v.PdcpCount, msg, sizeof(struct Pdcp_CountCnf_Type));
      else
      SS_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
#endif
      break;

    case SystemConfirm_Type_Paging:
      msgCnf->Confirm.v.Paging = true;
      break;
    case SystemConfirm_Type_Sps:
    case SystemConfirm_Type_L1MacIndCtrl:
    case SystemConfirm_Type_RlcIndCtrl:
    case SystemConfirm_Type_PdcpHandoverControl:
    case SystemConfirm_Type_L1_TestMode:
    case SystemConfirm_Type_PdcchOrder:
    case SystemConfirm_Type_ActivateScell:
    case SystemConfirm_Type_MbmsConfig:
    case SystemConfirm_Type_PDCCH_MCCH_ChangeNotification:
    case SystemConfirm_Type_MSI_Config:
    case SystemConfirm_Type_OCNG_Config:
    case SystemConfirm_Type_DirectIndicationInfo:
    default:
      LOG_A(RRC, "[SYS] Error not handled CNF TYPE to [SS-PORTMAN]");
    }
    SS_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending to [SS-PORTMAN]");
    }
  }
}
/* 
 * =========================================================================================================== 
 * Function Name: sys_handle_nr_cell_config_req
 * Parameter    : CellConfigRequest_Type *Cell, is the structure having cell configuration information 
 *                (received from TTCN)as per ASP Defination of NR_CellConfigRequest_Type (38.523-3).
 * Description  : This function handler for handling NR Cell Configuration Request and updates the respective
 *                structure in RAN Context.
 * Returns      : Cell State
 * ==========================================================================================================
*/

int sys_handle_nr_cell_config_req(struct CellConfigRequest_Type *Cell)
{
  int status = false;
  int returnState = RC.ss.State;
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_Cell;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;

  switch (Cell->d)
  {
  case CellConfigRequest_Type_AddOrReconfigure:

    LOG_A(NR_RRC, "TASK_SYS_5G_NR: CellConfigRequest_Type_AddOrReconfigure received\n");
    status = sys_add_nr_reconfig_cell(&(Cell->v.AddOrReconfigure));
    if (status)
    {
      /** TODO Signal to main thread */
      LOG_A(NR_RRC, "TASK_SYS_5G_NR: Signalling main thread for cell config done indication\n");
      cell_config_5g_done_indication();
    }
    //TODO Change it later to move to cell configuration
    if ( RC.ss.State == SS_STATE_NOT_CONFIGURED)
    {
    returnState = SS_STATE_CELL_CONFIGURED;
    }


    break;
  default:
    LOG_A(NR_RRC, "TASK_SYS_5G_NR: CellConfigRequest INVALID Type receivied\n");
  }
	/* TODO: 5G_cell_config start */
  //send_sys_cnf(resType, resVal, cnfType, NULL);
  return returnState;
}
#if 0
/*
 * Function : sys_handle_radiobearer_list
 * Description: Funtion handler of SYS_PORT. Handles the Radio
 * Bearer List configuration command received from TTCN via the PORTMAN.
 * Invokes the subroutinge to accept the configuration.
 * In :
 * req  - Radio Bearer List Request received from the TTCN via PORTMAN
 * Out:
 * newState: The next state for the SYS State machine
 *
 */
static int sys_handle_radiobearer_list(struct RadioBearer_Type_RadioBearerList_Dynamic *BearerList)
{
  int returnState = RC.ss.State;
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_RadioBearerList;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;
  MessageDef *msg_p = itti_alloc_new_message(TASK_SYS, 0, RRC_RBLIST_CFG_REQ);
  if (msg_p)
  {
    LOG_A(RRC, "[SYS] BearerList size:%lu\n", BearerList->d);
    RRC_RBLIST_CFG_REQ(msg_p).rb_count = 0;
    for (int i = 0; i < BearerList->d; i++)
    {
      LOG_A(RRC, "[SYS] RB Index i:%d\n", i);
      memset(&RRC_RBLIST_CFG_REQ(msg_p).rb_list[i], 0, sizeof(rb_info));
      if (BearerList->v[i].Id.d == RadioBearerId_Type_Srb)
      {
        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbId = BearerList->v[i].Id.v.Srb;
      }
      else if (BearerList->v[i].Id.d == RadioBearerId_Type_Drb)
      {
        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbId = BearerList->v[i].Id.v.Drb + 2; // Added 2 for MAXSRB because DRB1 starts from index-3
      }

      if (BearerList->v[i].Config.d == RadioBearerConfig_Type_AddOrReconfigure)
      {
        RRC_RBLIST_CFG_REQ(msg_p).rb_count++;
        /* Populate the PDCP Configuration for the radio Bearer */
        if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.d)
        {
          if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.d == PDCP_Configuration_Type_Config)
          {
            if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.d)
            {
              if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.d == PDCP_RBConfig_Type_Srb)
              {
                LOG_A(RRC, "[SYS] PDCP Config for Bearer Id: %d is Null\n", RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbId);
              }
              else if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.d == PDCP_RBConfig_Type_Drb)
              {
                RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isPDCPConfigValid = TRUE;
                if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.d == PDCP_Config_Type_R8)
                {
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.discardTimer.d)
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.discardTimer = CALLOC(1, sizeof(long));
                    *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.discardTimer) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.discardTimer.v;
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.rlc_AM.d)
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.rlc_AM = CALLOC(1, sizeof(struct LTE_PDCP_Config__rlc_AM));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.rlc_AM->statusReportRequired = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.rlc_AM.v.statusReportRequired;
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.rlc_UM.d)
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.rlc_UM = CALLOC(1, sizeof(struct LTE_PDCP_Config__rlc_UM));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.rlc_UM->pdcp_SN_Size = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.rlc_UM.v.pdcp_SN_Size;
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.d == PDCP_Config_headerCompression_rohc)
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.present = LTE_PDCP_Config__headerCompression_PR_rohc;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.maxCID.d)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.maxCID = CALLOC(1, sizeof(long));
                      *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.maxCID) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.maxCID.v;
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0001 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0001;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0002 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0002;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0003 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0003;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0004 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0004;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0006 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0006;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0101 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0101;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0102 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0102;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0103 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0103;
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.headerCompression.choice.rohc.profiles.profile0x0104 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.v.rohc.profiles.profile0x0104;
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.rn_IntegrityProtection_r10.d)
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext1 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext1));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext1->rn_IntegrityProtection_r10 = CALLOC(1, sizeof(long));
                    *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext1->rn_IntegrityProtection_r10) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.rn_IntegrityProtection_r10.v;
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_SN_Size_v1130.d)
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext2 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext2));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext2->pdcp_SN_Size_v1130 = CALLOC(1, sizeof(long));
                    *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext2->pdcp_SN_Size_v1130) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_SN_Size_v1130.v;
                  }
                  if ((BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitDRB_ViaSCG_r12.d) || (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.t_Reordering_r12.d))
                  {
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext3 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext3));
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitDRB_ViaSCG_r12.d)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext3->ul_DataSplitDRB_ViaSCG_r12 = CALLOC(1, sizeof(bool));
                      *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext3->ul_DataSplitDRB_ViaSCG_r12) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitDRB_ViaSCG_r12.v;
                    }
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.t_Reordering_r12.d)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext3->t_Reordering_r12 = CALLOC(1, sizeof(long));
                      *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext3->t_Reordering_r12) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.t_Reordering_r12.v;
                    }
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitThreshold_r13.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext4));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->ul_DataSplitThreshold_r13 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext4__ul_DataSplitThreshold_r13));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->ul_DataSplitThreshold_r13->present = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitThreshold_r13.v.d;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitThreshold_r13.v.d == PDCP_Config_ul_DataSplitThreshold_r13_setup)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->ul_DataSplitThreshold_r13->choice.setup = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitThreshold_r13.v.v.setup;
                    }
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_SN_Size_v1310.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext4));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->pdcp_SN_Size_v1310 = CALLOC(1, sizeof(long));
                    *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->pdcp_SN_Size_v1310) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_SN_Size_v1310.v;
                  }

                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext4));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext4__statusFeedback_r13));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->present = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.d;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.d == PDCP_Config_statusFeedback_r13_setup)
                    {
                      if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_TypeForPolling_r13.d)
                      {
                        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_TypeForPolling_r13 = CALLOC(1, sizeof(long));
                        *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_TypeForPolling_r13) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_TypeForPolling_r13.v;
                      }
                      if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_Periodicity_Type1_r13.d)
                      {
                        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_Periodicity_Type1_r13 = CALLOC(1, sizeof(long));
                        *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_Periodicity_Type1_r13) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_Periodicity_Type1_r13.v;
                      }
                      if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_Periodicity_Type2_r13.d)
                      {
                        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_Periodicity_Type2_r13 = CALLOC(1, sizeof(long));
                        *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_Periodicity_Type2_r13) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_Periodicity_Type2_r13.v;
                      }
                      if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_Periodicity_Offset_r13.d)
                      {
                        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_Periodicity_Offset_r13 = CALLOC(1, sizeof(long));
                        *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext4->statusFeedback_r13->choice.setup.statusPDU_Periodicity_Offset_r13) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.v.setup.statusPDU_Periodicity_Offset_r13.v;
                      }
                    }
                  }

                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext5));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->ul_LWA_Config_r14 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext5__ul_LWA_Config_r14));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->ul_LWA_Config_r14->present = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.v.d;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.v.d == PDCP_Config_ul_LWA_Config_r14_setup)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->ul_LWA_Config_r14->choice.setup.ul_LWA_DRB_ViaWLAN_r14 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.v.v.setup.ul_LWA_DRB_ViaWLAN_r14;
                      if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.v.v.setup.ul_LWA_DataSplitThreshold_r14.d)
                      {
                        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->ul_LWA_Config_r14->choice.setup.ul_LWA_DataSplitThreshold_r14 = CALLOC(1, sizeof(long));
                        *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->ul_LWA_Config_r14->choice.setup.ul_LWA_DataSplitThreshold_r14) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.v.v.setup.ul_LWA_DataSplitThreshold_r14.v;
                      }
                    }
                  }

                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext5));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->uplinkOnlyHeaderCompression_r14 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext5__uplinkOnlyHeaderCompression_r14));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->uplinkOnlyHeaderCompression_r14->present = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.v.d;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.v.d == PDCP_Config_uplinkOnlyHeaderCompression_r14_rohc_r14)
                    {
                      if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.v.v.rohc_r14.maxCID_r14.d)
                      {
                        RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->uplinkOnlyHeaderCompression_r14->choice.rohc_r14.maxCID_r14 = CALLOC(1, sizeof(long));
                        *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->uplinkOnlyHeaderCompression_r14->choice.rohc_r14.maxCID_r14) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.v.v.rohc_r14.maxCID_r14.v;
                      }
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext5->uplinkOnlyHeaderCompression_r14->choice.rohc_r14.profiles_r14.profile0x0006_r14 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.v.v.rohc_r14.profiles_r14.profile0x0006_r14;
                    }
                  }

                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkDataCompression_r15.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext6));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->uplinkDataCompression_r15 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext6__uplinkDataCompression_r15));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->uplinkDataCompression_r15->bufferSize_r15 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkDataCompression_r15.v.bufferSize_r15;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkDataCompression_r15.v.dictionary_r15.d)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->uplinkDataCompression_r15->dictionary_r15 = CALLOC(1, sizeof(long));
                      *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->uplinkDataCompression_r15->dictionary_r15) = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkDataCompression_r15.v.dictionary_r15.v;
                    }
                  }
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_DuplicationConfig_r15.d)
                  {
                    if (RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6 == NULL)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext6));
                    }
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->pdcp_DuplicationConfig_r15 = CALLOC(1, sizeof(struct LTE_PDCP_Config__ext6__pdcp_DuplicationConfig_r15));
                    RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->pdcp_DuplicationConfig_r15->present = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_DuplicationConfig_r15.v.d;
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_DuplicationConfig_r15.v.d == PDCP_Config_pdcp_DuplicationConfig_r15_setup)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->pdcp_DuplicationConfig_r15->choice.setup.pdcp_Duplication_r15 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_DuplicationConfig_r15.v.v.setup.pdcp_Duplication_r15;
                    }
                  }

                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ethernetHeaderCompression_r16.d)
                  {
                    LOG_A(RRC, "Unsupported IE: ethernetHeaderCompression_r16 \n");
                  }
                }
              }
            }
          }
        }

        /* Populate the RLC Configuration for the radio Bearer */
        if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.d)
        {
          if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.d)
          {
            RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isRLCConfigValid = TRUE;
            if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.d == RLC_RbConfig_Type_AM)
            {
              RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.present = LTE_RLC_Config_PR_am;
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Tx.d)
              {
                if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Tx.v.d == UL_AM_RLC_Type_R8)
                {
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.am.ul_AM_RLC.t_PollRetransmit = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Tx.v.v.R8.t_PollRetransmit;
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.am.ul_AM_RLC.pollPDU = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Tx.v.v.R8.pollPDU;
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.am.ul_AM_RLC.pollByte = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Tx.v.v.R8.pollByte;
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.am.ul_AM_RLC.maxRetxThreshold = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Tx.v.v.R8.maxRetxThreshold;
                }
              }
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Rx.d)
              {
                if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Rx.v.d == DL_AM_RLC_Type_R8)
                {
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.am.dl_AM_RLC.t_Reordering = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Rx.v.v.R8.t_Reordering;
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.am.dl_AM_RLC.t_StatusProhibit = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.Rx.v.v.R8.t_StatusProhibit;
                }
              }
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.AM.ExtendedLI.d)
              {
                //TODO
              }
            }
            if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.d == RLC_RbConfig_Type_UM)
            {
              RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.present = LTE_RLC_Config_PR_um_Bi_Directional;
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Tx.d)
              {
                if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Tx.v.d == UL_UM_RLC_Type_R8)
                {
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.um_Bi_Directional.ul_UM_RLC.sn_FieldLength = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Tx.v.v.R8.sn_FieldLength;
                }
              }
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Rx.d)
              {
                if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Rx.v.d == DL_UM_RLC_Type_R8)
                {
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.um_Bi_Directional.dl_UM_RLC.sn_FieldLength = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Rx.v.v.R8.sn_FieldLength;
                  RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.um_Bi_Directional.dl_UM_RLC.t_Reordering = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM.Rx.v.v.R8.t_Reordering;
                }
              }
            }

            if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyUL.Rx.d)
            {
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyUL.Rx.v.d == DL_UM_RLC_Type_R8)
              {
                // TTCN Configuration is based on the UE configuration that's why DL Configuration need to be read from Rx Configuration
                RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.um_Uni_Directional_DL.dl_UM_RLC.sn_FieldLength = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyUL.Rx.v.v.R8.sn_FieldLength;
                RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.um_Uni_Directional_DL.dl_UM_RLC.t_Reordering = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyUL.Rx.v.v.R8.t_Reordering;
              }
            }

            if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyDL.Tx.d)
            {
              if (BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyDL.Tx.v.d == UL_UM_RLC_Type_R8)
              {
                // TTCN Configuration is based on the UE configuration that's why UL Configuration need to be read from Tx Configuration
                RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Rlc.choice.um_Uni_Directional_UL.ul_UM_RLC.sn_FieldLength = BearerList->v[i].Config.v.AddOrReconfigure.Rlc.v.Rb.v.v.UM_OnlyDL.Tx.v.v.R8.sn_FieldLength;
              }
            }
          }
        }

        if (BearerList->v[i].Config.v.AddOrReconfigure.LogicalChannelId.d)
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isLogicalChannelIdValid = TRUE;
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.LogicalChannelId = BearerList->v[i].Config.v.AddOrReconfigure.LogicalChannelId.v;
        }
        else
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.LogicalChannelId = 0;
        }

        /* Populate the MAC Configuration for the radio Bearer */
        if (BearerList->v[i].Config.v.AddOrReconfigure.Mac.d)
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isMacConfigValid = TRUE;
          if (BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.LogicalChannel.d)
          {
            RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Mac.ul_SpecificParameters = CALLOC(1, sizeof(struct LTE_LogicalChannelConfig__ul_SpecificParameters));
            RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Mac.ul_SpecificParameters->priority = BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.LogicalChannel.v.Priority;
            RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Mac.ul_SpecificParameters->prioritisedBitRate = BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.LogicalChannel.v.PrioritizedBitRate;
            if (BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.LogicalChannel.v.LAA_UL_Allowed.d)
            {
              RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Mac.ext3 = CALLOC(1, sizeof(struct LTE_LogicalChannelConfig__ext3));
              RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Mac.ext3->laa_UL_Allowed_r14 = CALLOC(1, sizeof(bool));
              *(RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Mac.ext3->laa_UL_Allowed_r14) = BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.LogicalChannel.v.LAA_UL_Allowed.v;
            }
          }
        }

        if (BearerList->v[i].Config.v.AddOrReconfigure.DiscardULData.d)
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isDiscardULDataValid = TRUE;
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.DiscardULData = BearerList->v[i].Config.v.AddOrReconfigure.DiscardULData.v;
        }
        else
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.DiscardULData = FALSE;
        }
      }
    }
    LOG_A(RRC, "[SYS] Send RRC_RBLIST_CFG_REQ to TASK_RRC_ENB, RB Count : %d, Message: %s  \n", RRC_RBLIST_CFG_REQ(msg_p).rb_count, ITTI_MSG_NAME(msg_p));
    int send_res = itti_send_msg_to_task(TASK_RRC_ENB, 0, msg_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending RRC_RBLIST_CFG_REQ to RRC_ENB");
    }
  }

  returnState = SS_STATE_CELL_ACTIVE;
  send_sys_cnf(resType, resVal, cnfType, NULL);
  return returnState;
}

/*
 * Function : sys_handle_pdcp_count_req
 * Description: Funtion handler of SYS_PORT. Handles the PDCP
 * count command received from TTCN via the PORTMAN.
 * Invokes subroutines for GET or SET to PDCP Count.
 * In :
 * req  - PDCP Count Request received from the TTCN via PORTMAN
 * Out:
 * newState: No impact on state machine.
 *
 */
int sys_handle_pdcp_count_req(struct Pdcp_CountReq_Type *PdcpCount)
{
  int returnState = RC.ss.State;
  int send_res = -1;

  switch (PdcpCount->d)
  {
  case Pdcp_CountReq_Type_Get:
    LOG_A(RRC, "[SYS] Pdcp_CountReq_Type_Get receivied\n");
    MessageDef *get_p = itti_alloc_new_message(TASK_SYS, 0, SS_REQ_PDCP_CNT);
    SS_REQ_PDCP_CNT(get_p).rnti = ss_rnti_nr_g;
    switch (PdcpCount->v.Get.d)
    {
    case PdcpCountGetReq_Type_AllRBs:
      LOG_A(RRC, "[SYS] Pdcp_CountReq_Type_Get AllRBs receivied\n");
      SS_REQ_PDCP_CNT(get_p).rb_id = -1;
      break;
    case PdcpCountGetReq_Type_SingleRB:
      LOG_A(RRC, "[SYS] Pdcp_CountReq_Type_Get SingleRB receivied\n");
      switch (PdcpCount->v.Get.v.SingleRB.d)
      {
      case RadioBearerId_Type_Srb:
        SS_REQ_PDCP_CNT(get_p).rb_id = PdcpCount->v.Get.v.SingleRB.v.Srb;
        break;
      case RadioBearerId_Type_Drb:
        SS_REQ_PDCP_CNT(get_p).rb_id = PdcpCount->v.Get.v.SingleRB.v.Drb + 2; /** TODO Need to check how OAI maintains RBID */
        break;
      case RadioBearerId_Type_Mrb:
        break;
      case RadioBearerId_Type_ScMrb:
        break;
      case RadioBearerId_Type_UNBOUND_VALUE:
        break;
      }
      break;
    case PdcpCountGetReq_Type_UNBOUND_VALUE:
      LOG_A(RRC, "[SYS] PdcpCountGetReq_Type_UNBOUND_VALUE received\n");
      break;
    default:
      LOG_A(RRC, "[SYS] Pdcp_CountReq_Type (GET) Invalid \n");
    }
    LOG_A(RRC," [SYS] SS_REQ_PDCP_CNT(message_p).rb_id %d\n", SS_REQ_PDCP_CNT(get_p).rb_id);
    send_res = itti_send_msg_to_task(TASK_PDCP_ENB, 0, get_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending SS_REQ_PDCP_CNT to PDCP_ENB");
    }

    break;
  case Pdcp_CountReq_Type_Set:
    LOG_A(RRC, "[SYS] Pdcp_CountReq_Type_Set receivied\n");
    MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0, SS_SET_PDCP_CNT);
    for (int i = 0; i < PdcpCount->v.Set.d; i++)
    {
      switch (PdcpCount->v.Set.v[i].RadioBearerId.d)
      {
      case RadioBearerId_Type_Srb:
        SS_SET_PDCP_CNT(message_p).rb_list[i].rb_id = PdcpCount->v.Set.v[i].RadioBearerId.v.Srb;
        break;
      case RadioBearerId_Type_Drb:
        SS_SET_PDCP_CNT(message_p).rb_list[i].rb_id = PdcpCount->v.Set.v->RadioBearerId.v.Drb;
        break;
      case RadioBearerId_Type_UNBOUND_VALUE:
        break;
      default:
        LOG_A(RRC, "[SYS] Pdcp_CountReq_Type (SET) Invalid \n");
      }
      if (PdcpCount->v.Set.v[i].UL.d == true)
      {
        SS_SET_PDCP_CNT(message_p).rb_list[i].ul_format = PdcpCount->v.Set.v[i].UL.v.Format;
        SS_SET_PDCP_CNT(message_p).rb_list[i].ul_count = bin_to_int(PdcpCount->v.Set.v[i].UL.v.Value, 32);
      }
      if (PdcpCount->v.Set.v[i].DL.d == true)
      {
        SS_SET_PDCP_CNT(message_p).rb_list[i].dl_format = PdcpCount->v.Set.v[i].DL.v.Format;
        SS_SET_PDCP_CNT(message_p).rb_list[i].dl_count = bin_to_int(PdcpCount->v.Set.v[i].DL.v.Value, 32);
      }
    }

    send_res = itti_send_msg_to_task(TASK_PDCP_ENB, 0, message_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending SS_SET_PDCP_CNT to PDCP_ENB");
    }

    break;
  case Pdcp_CountReq_Type_UNBOUND_VALUE:
    LOG_A(RRC, "[SYS] Pdcp_CountReq_Type UNBOUND receivied\n");
    break;
  default:
    LOG_A(RRC, "[SYS] Pdcp_CountReq_Type INVALID Type receivied\n");
  }
  return returnState;
}
#endif
/*
 * Function : sys_send_proxy
 * Description: Sends the messages from SYS to proxy
 */
static void sys_send_proxy(void *msg, int msgLen)
{
  LOG_A(RRC, "In sys_send_proxy\n");
  uint32_t peerIpAddr;
  uint16_t peerPort = proxy_send_port;

  IPV4_STR_ADDR_TO_INT_NWBO(local_address, peerIpAddr, " BAD IP Address");

  LOG_A(RRC, "******************* Sending CELL CONFIG length\n Buffer is :%d ", msgLen);
  int8_t *temp = msg;
  for(int i =0 ; i <msgLen;i++)
  {
    LOG_A(RRC, "%x ", temp[i]);
  }

  LOG_A(RRC, "\nCell Config End of Buffer\n ");

  /** Send to proxy */
  sys_send_udp_msg((uint8_t *)msg, msgLen, 0, peerIpAddr, peerPort);
  return;
}

/*
 * Function : sys_cell_attn_update
 * Description: Sends the attenuation updates received from TTCN to proxy
 */
static void sys_cell_attn_update(uint8_t cellId, uint8_t attnVal)
{
  LOG_A(RRC, "In sys_cell_attn_update\n");
  attenuationConfigReq_t *attnConf = NULL;
  uint32_t peerIpAddr;
  uint16_t peerPort = proxy_send_port;

  attnConf = (attenuationConfigReq_t *) calloc(1, sizeof(attenuationConfigReq_t));
  attnConf->header.preamble = 0xFEEDC0DE;
  attnConf->header.msg_id = SS_ATTN_LIST;
  attnConf->header.cell_id = SS_context.cellId;
  attnConf->attnVal = attnVal;
  IPV4_STR_ADDR_TO_INT_NWBO(local_address, peerIpAddr, " BAD IP Address");

  /** Send to proxy */
  sys_send_udp_msg((uint8_t *)attnConf, sizeof(attenuationConfigReq_t), 0, peerIpAddr, peerPort);
  LOG_A(RRC, "Out sys_cell_attn_update\n");
  return;
}
#if 0
/*
 * Function : sys_handle_cell_attn_req
 * Description: Handles the attenuation updates received from TTCN
 */
static void sys_handle_cell_attn_req(struct CellAttenuationConfig_Type_CellAttenuationList_Dynamic *CellAttenuationList)
{
  /** TODO: Considering only one cell for now */
  uint8_t cellId = (uint8_t)CellAttenuationList->v->CellId;
  uint8_t attnVal = 0; // default set it Off

  switch (CellAttenuationList->v->Attenuation.d)
  {
  case Attenuation_Type_Value:
    attnVal = CellAttenuationList->v->Attenuation.v.Value;
    LOG_A(RRC, "[SYS] CellAttenuationList for Cell_id %d value %d dBm received\n",
          cellId, attnVal);
    sys_cell_attn_update(cellId, attnVal);
    break;
  case Attenuation_Type_Off:
    LOG_A(RRC, "[SYS] CellAttenuationList turn off for Cell_id %d received\n",
          cellId);
    sys_cell_attn_update(cellId, attnVal);
    break;
  case Attenuation_Type_UNBOUND_VALUE:
    LOG_A(RRC, "[SYS] CellAttenuationList Attenuation_Type_UNBOUND_VALUE received\n");
    break;
  default:
    LOG_A(RRC, "[SYS] Invalid CellAttenuationList received\n");
  }
}
/*
 * Function : sys_handle_cell_attn_req
 * Description: Handles the attenuation updates received from TTCN
 */

static void sys_handle_paging_req(struct PagingTrigger_Type *pagingRequest, ss_set_timinfo_t tinfo)
{

  LOG_A(RRC, "[SYS] Enter sys_handle_paging_req Paging_IND for processing\n");

  /** TODO: Considering only one cell for now */
  uint8_t cellId = 0; //(uint8_t)pagingRequ ->CellId;
  uint8_t cn_domain = 0;

  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_Paging;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0,SS_SS_PAGING_IND);
  switch (pagingRequest->Paging.message.d)
  {
  case PCCH_MessageType_c1:
    if (pagingRequest->Paging.message.v.c1.d)
    {
      if (pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.d)
      {
        cn_domain = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v->cn_Domain;
        size_t pgSize = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.d * sizeof(ss_paging_identity_t);
        SS_PAGING_IND(message_p).sfn =tinfo.sfn;
        SS_PAGING_IND(message_p).sf = tinfo.sf;
        SS_PAGING_IND(message_p).paging_recordList = CALLOC(1, pgSize);
        /* id-CNDomain : convert cnDomain */
        if (cn_domain == PagingRecord_cn_Domain_e_ps)
        {
          SS_PAGING_IND(message_p).paging_recordList->cn_domain = CN_DOMAIN_PS;
        }
        else if (cn_domain == PagingRecord_cn_Domain_e_cs)
        {
          SS_PAGING_IND(message_p).paging_recordList->cn_domain = CN_DOMAIN_CS;
        }

        switch (pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v->ue_Identity.d)
        {
        case PagingUE_Identity_s_TMSI:
          SS_PAGING_IND(message_p).paging_recordList->ue_paging_identity.presenceMask = UE_PAGING_IDENTITY_s_tmsi;
          int32_t stmsi_rx = bin_to_int(pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v->ue_Identity.v.s_TMSI.m_TMSI, 32);

          SS_PAGING_IND(message_p).paging_recordList->ue_paging_identity.choice.s_tmsi.m_tmsi = stmsi_rx ;
           SS_PAGING_IND(message_p).paging_recordList->ue_paging_identity.choice.s_tmsi.mme_code =
                 bin_to_int(pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v->ue_Identity.v.s_TMSI.mmec,8);
           SS_PAGING_IND(message_p).paging_recordList->ue_index_value = paging_ue_index_g;
           paging_ue_index_g = ((paging_ue_index_g +4) % MAX_MOBILES_PER_ENB) ;
          break;
        case PagingUE_Identity_imsi:
          SS_PAGING_IND(message_p).paging_recordList->ue_paging_identity.presenceMask = UE_PAGING_IDENTITY_imsi;

          memcpy(&(SS_PAGING_IND(message_p).paging_recordList->ue_paging_identity.choice.imsi),
                 &(pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v->ue_Identity.v.imsi),
                 sizeof(s1ap_imsi_t));
          break;
        case PagingUE_Identity_ng_5G_S_TMSI_r15:
        case PagingUE_Identity_fullI_RNTI_r15:
        case PagingUE_Identity_UNBOUND_VALUE:
          LOG_A(RRC, "[SYS] Error Unhandled Paging request \n");
          break;
        default :
          LOG_A(RRC, "[SYS] Invalid Pging request received\n");

        }
      }
    }

    int send_res = itti_send_msg_to_task(TASK_RRC_ENB, 0, message_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending Paging to RRC_ENB");
    }

    LOG_A(RRC, "[SYS] Paging_IND for Cell_id %d  sent to RRC\n", cellId);
    break;
  case PCCH_MessageType_messageClassExtension:
    LOG_A(RRC, "[SYS] PCCH_MessageType_messageClassExtension for Cell_id %d received\n",
          cellId);
    break;
  case PCCH_MessageType_UNBOUND_VALUE:
    LOG_A(RRC, "[SYS] Invalid Pging request received Type_UNBOUND_VALUE received\n");
    break;
  default:
    LOG_A(RRC, "[SYS] Invalid Pging request received\n");
  }
  send_sys_cnf(resType, resVal, cnfType, NULL);
  LOG_A(RRC, "[SYS] Exit sys_handle_paging_req Paging_IND processing for Cell_id %d \n", cellId);
}
#endif
/*
 * Function : sys_handle_nr_enquire_timing
 * Description: Sends the NR enquire timing update to PORTMAN
 */
static void sys_handle_nr_enquire_timing(ss_nrset_timinfo_t *tinfo)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS_5G_NR, INSTANCE_DEFAULT, SS_NRSET_TIM_INFO);
  if (message_p)
  {
    LOG_A(RRC, "[SYS] Reporting info sfn:%d\t sf:%d.\n", tinfo->sfn, tinfo->slot);
    SS_NRSET_TIM_INFO(message_p).slot = tinfo->slot;
    SS_NRSET_TIM_INFO(message_p).sfn = tinfo->sfn;

    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending to [SS-PORTMAN]");
    }
  }
}

static void sys_handle_ue_cat_info_req(struct UE_CategoryInfo_Type *UE_Cat_Info)
{
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_UE_Cat_Info;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, INSTANCE_DEFAULT, RRC_UE_CAT_INFO);
  if (message_p)
  {
    LOG_A(RRC,"[SYS] UE Category Info received \n");
    RRC_UE_CAT_INFO(message_p).ue_Category = UE_Cat_Info->ue_Category;
    if(UE_Cat_Info->ue_Category_V1020.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_Category_V1020_present = true;
      RRC_UE_CAT_INFO(message_p).ue_Category_V1020 = UE_Cat_Info->ue_Category_V1020.v;
    }
    if(UE_Cat_Info->ue_Category_v1170.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_Category_v1170_present = true;
      RRC_UE_CAT_INFO(message_p).ue_Category_v1170 = UE_Cat_Info->ue_Category_v1170.v;
    }
    if(UE_Cat_Info->ue_Category_v11a0.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_Category_v11a0_present = true;
      RRC_UE_CAT_INFO(message_p).ue_Category_v11a0 = UE_Cat_Info->ue_Category_v11a0.v;
    }
    if(UE_Cat_Info->ue_Category_v1250.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_Category_v1250_present = true;
      RRC_UE_CAT_INFO(message_p).ue_Category_v1250 = UE_Cat_Info->ue_Category_v1250.v;
    }
    if(UE_Cat_Info->ue_CategoryDL_r12.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_CategoryDL_r12_present = true;
      RRC_UE_CAT_INFO(message_p).ue_CategoryDL_r12 = UE_Cat_Info->ue_CategoryDL_r12.v;
    }
    if(UE_Cat_Info->ue_CategoryDL_v1260.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_CategoryDL_v1260_present = true;
      RRC_UE_CAT_INFO(message_p).ue_CategoryDL_v1260 = UE_Cat_Info->ue_CategoryDL_v1260.v;
    }
    if(UE_Cat_Info->ue_CategoryDL_v1310.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_CategoryDL_v1310_present = true;
      RRC_UE_CAT_INFO(message_p).ue_CategoryDL_v1310 = UE_Cat_Info->ue_CategoryDL_v1310.v;
    }
    if(UE_Cat_Info->ue_CategoryDL_v1330.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_CategoryDL_v1330_present = true;
      RRC_UE_CAT_INFO(message_p).ue_CategoryDL_v1330 = UE_Cat_Info->ue_CategoryDL_v1330.v;
    }
    if(UE_Cat_Info->ue_CategoryDL_v1350.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_CategoryDL_v1350_present = true;
      RRC_UE_CAT_INFO(message_p).ue_CategoryDL_v1350 = UE_Cat_Info->ue_CategoryDL_v1350.v;
    }
    if(UE_Cat_Info->ue_CategoryDL_v1460.d == true)
    {
      RRC_UE_CAT_INFO(message_p).is_ue_CategoryDL_v1460_present = true;
      RRC_UE_CAT_INFO(message_p).ue_CategoryDL_v1460 = UE_Cat_Info->ue_CategoryDL_v1460.v;
    }
    int send_res = itti_send_msg_to_task(TASK_RRC_ENB, INSTANCE_DEFAULT, message_p);
    if (send_res < 0)
    {
      LOG_A(RRC, "[SYS] Error sending RRC_UE_CAT_INFO to TASK_RRC_ENB");
    }
  }
  send_sys_cnf(resType, resVal, cnfType, NULL);
}

/*
 * Function : sys_handle_as_security_req
 * Description: Funtion handler of SYS_PORT. Handles the AS
 * Security command received from TTCN via the PORTMAN.
 * In :
 * req  - AS Security Request received from the TTCN via PORTMAN
 * Out:
 * newState: No impact on state machine.
 *
 */
static void sys_handle_as_security_req(struct AS_Security_Type *ASSecurity)
{
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_AS_Security;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;
  bool intKey = FALSE;

  MessageDef *msg_p = itti_alloc_new_message(TASK_SYS, 0, RRC_AS_SECURITY_CONFIG_REQ);
  if(msg_p)
  {
    LOG_A(RRC,"[SYS] AS Security Request Received\n");
    RRC_AS_SECURITY_CONFIG_REQ(msg_p).rnti = ss_rnti_nr_g;
    if(ASSecurity->d == AS_Security_Type_StartRestart)
    {
      if(ASSecurity->v.StartRestart.Integrity.d == true)
      {
        intKey = TRUE;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).isIntegrityInfroPresent = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.integrity_algorithm = ASSecurity->v.StartRestart.Integrity.v.Algorithm;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint = CALLOC(1,32);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint,0,32);
        bitStrint_to_byteArray(ASSecurity->v.StartRestart.Integrity.v.KRRCint,256,RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint,intKey);
        for(int j=0;j<32;j++) {
          LOG_A(RRC,"[SYS] KRRCint in SS: %02x \n",RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint[j]);
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
              LOG_A(RRC, "[SYS] AS Security Act time list is Invalid \n");
            }
            if (ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].UL.d == PDCP_ActTime_Type_SQN)
            {
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].UL.format = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].UL.v.SQN.Format;
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].UL.sqn = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].UL.v.SQN.Value;
            }
            if (ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].DL.d == PDCP_ActTime_Type_SQN)
            {
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].DL.format = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].DL.v.SQN.Format;
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].DL.sqn = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].DL.v.SQN.Value;
            }
          }
        }
      }
      if(ASSecurity->v.StartRestart.Ciphering.d == true)
      {
        intKey = FALSE;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).isCipheringInfoPresent = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ciphering_algorithm = ASSecurity->v.StartRestart.Ciphering.v.Algorithm;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc,0,16);
        bitStrint_to_byteArray(ASSecurity->v.StartRestart.Ciphering.v.KRRCenc,128,RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc,intKey);
        for(int i=0;i<16;i++) {
          LOG_A(RRC,"[SYS] kRRCenc in SS: %02x \n",RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc[i]);
        }
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc,0,16);
        bitStrint_to_byteArray(ASSecurity->v.StartRestart.Ciphering.v.KUPenc,128,RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc,intKey);
        for(int k=0;k<16;k++) {
          LOG_A(RRC,"[SYS] kUPenc in SS: %02x \n",RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc[k]);
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
            LOG_A(RRC, "[SYS] AS Security Act time list is Invalid \n");
          }
          if (ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].UL.d == PDCP_ActTime_Type_SQN)
          {
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].UL.format = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].UL.v.SQN.Format;
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].UL.sqn = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].UL.v.SQN.Value;
          }
          if (ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].DL.d == PDCP_ActTime_Type_SQN)
          {
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].DL.format = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].DL.v.SQN.Format;
            RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].DL.sqn = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].DL.v.SQN.Value;
          }
        }
      }
      int send_res = itti_send_msg_to_task(TASK_RRC_ENB, 0, msg_p);
      if (send_res < 0)
      {
         LOG_A(RRC, "[SYS] Error sending RRC_AS_SECURITY_CONFIG_REQ to RRC_ENB");
      }
    }
  }
  send_sys_cnf(resType, resVal, cnfType, NULL);
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
static void ss_task_sys_nr_handle_req(struct SYSTEM_CTRL_REQ *req, ss_set_timinfo_t *tinfo)
{
	int enterState = RC.ss.State;
	int exitState = RC.ss.State;
	if(req->Common.CellId)				
		/* TODO: 5G_cell_config start */
		SS_context.eutra_cellId = req->Common.CellId;
	/* 5G_cell_config end */
	LOG_A(NR_RRC, "TASK_SYS_5G_NR: Current SS_STATE %d received SystemRequest_Type %d eutra_cellId %d cnf_flag %d\n",
			RC.ss.State, req->Request.d, SS_context.eutra_cellId, req->Common.ControlInfo.CnfFlag);
	switch (RC.ss.State)
	{
		case SS_STATE_NOT_CONFIGURED:
			if (req->Request.d == SystemRequest_Type_Cell)
			{
				LOG_A(NR_RRC, "TASK_SYS_5G_NR: SystemRequest_Type_Cell received\n");
				exitState = sys_handle_nr_cell_config_req(&(req->Request.v.Cell));
				RC.ss.State = exitState;
			}
			else
			{
				LOG_E(NR_RRC, "TASK_SYS_5G_NR: Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
						RC.ss.State, req->Request.d);
			}
			break;
		default:
			LOG_E(NR_RRC, "TASK_SYS_5G_NR: Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
					RC.ss.State, req->Request.d);
			break;
	}
	LOG_A(NR_RRC, "TASK_SYS_5G_NR: SS_STATE %d New SS_STATE %d received SystemRequest_Type %d\n",
			enterState, RC.ss.State, req->Request.d);
}

/* 
 * =============================================================================================================
 * Function Name: valid_nr_sys_msg
 * Parameter    : SYSTEM_CTRL_REQ *req, is the message having ASP Defination of NR_SYSTEM_CTRL_REQ (38.523-3) 
 *                which is received on SIDL via TTCN.
 * Description  : This function validates the validity of System Control Request Type. On successfull validation,
 *                this function sends the dummy confirmation to PORTMAN which is further forwareded towards TTCN.
 * Returns      : TRUE if recevied command is supported by SYS state handler
 *                FALSE if received command is not supported by SYS handler 
 * ============================================================================================================
*/

bool valid_nr_sys_msg(struct SYSTEM_CTRL_REQ *req)
{
  bool valid = FALSE;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = TRUE;
  bool sendDummyCnf = TRUE;
  enum SystemConfirm_Type_Sel cnfType = 0;

  LOG_A(NR_RRC, "TASK_SYS_5G_NR: received req : %d for cell %d RC.ss.state %d \n",
        req->Request.d, req->Common.CellId, RC.ss.State);
  switch (req->Request.d)
  {
	  case SystemRequest_Type_Cell:
		  if (RC.ss.State >= SS_STATE_NOT_CONFIGURED)
		  {
			  valid = TRUE;
			  sendDummyCnf = FALSE;
			  reqCnfFlag_g = req->Common.ControlInfo.CnfFlag;
		  }
		  else
		  {
			  cnfType = SystemConfirm_Type_Cell;
		  }
		  break;
	  default:
		  valid = FALSE;
		  sendDummyCnf = FALSE;
  }
  if (sendDummyCnf)
  {
    send_sys_cnf(resType, resVal, cnfType, NULL);
    LOG_A(NR_RRC, "TASK_SYS_5G_NR: Sending Dummy OK Req %d cnTfype %d ResType %d ResValue %d\n",
          req->Request.d, cnfType, resType, resVal);
  }
  return valid;
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
  static ss_nrset_timinfo_t tinfo = {.sfn = 0xFFFF, .slot = 0xFF};
  /* TODO: 5G_cell_config start */
  //SS_context.sfn = tinfo.sfn;
  //SS_context.sf  = tinfo.sf;
  /* 5G_cell_config end */

  itti_receive_msg(TASK_SYS_5G_NR, &received_msg);

  /* Check if there is a packet to handle */
  if (received_msg != NULL)
  {
    switch (ITTI_MSG_ID(received_msg))
    {
    case SS_NRUPD_TIM_INFO:
    {
      tinfo.slot = SS_NRUPD_TIM_INFO(received_msg).slot;
      tinfo.sfn = SS_NRUPD_TIM_INFO(received_msg).sfn;
    }
    break;

    case SS_SYS_PORT_MSG_IND:
    {

      if (valid_nr_sys_msg(SS_SYS_PORT_MSG_IND(received_msg).req))
      {
        ss_task_sys_nr_handle_req(SS_SYS_PORT_MSG_IND(received_msg).req, &tinfo);
      }
      else
      {
        LOG_A(NR_RRC, "TASK_SYS_5G_NR: Not handled SYS_PORT message received \n");
      }
    }
    break;

    case TERMINATE_MESSAGE:
    {
      itti_exit_task();
      break;
    }
    default:
      LOG_A(NR_RRC, "TASK_SYS_5G_NR: Received unhandled message %d:%s\n",
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
	if (RC.ss.configured == 0)
	{
  		RCconfig_nr_ssparam();
		RC.ss.configured = 1;
	}
	// Set the state to NOT_CONFIGURED for Cell Config processing mode
	if (RC.ss.mode == SS_SOFTMODEM)
	{
		RC.ss.State = SS_STATE_NOT_CONFIGURED;
	}
	// Set the state to CELL_ACTIVE for SRB processing mode
	else if (RC.ss.mode == SS_SOFTMODEM_SRB)
	{
		RC.ss.State = SS_STATE_CELL_ACTIVE;
	}


	while (1)
	{
		(void) ss_gNB_sys_process_itti_msg(NULL);
	}

	return NULL;
}
