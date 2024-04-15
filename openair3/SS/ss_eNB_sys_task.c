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
#include "SidlCommon.h"
#include "ss_eNB_sys_task.h"
#include "ss_eNB_context.h"
#include "ss_eNB_vt_timer_task.h"
#include "ss_eNB_multicell_helper.h"

#include "udp_eNB_task.h"
#include "ss_eNB_proxy_iface.h"
#include "common/utils/LOG/ss-log.h"

extern pthread_cond_t sys_confirm_done_cond;
extern pthread_mutex_t sys_confirm_done_mutex;
extern int sys_confirm_done;
extern RAN_CONTEXT_t RC;
extern uint32_t from_earfcn(int eutra_bandP, uint32_t dl_earfcn);
extern pthread_mutex_t lock_cell_si_config;
extern pthread_cond_t cond_cell_si_config;

#ifndef NR_RRC_VERSION
extern pthread_cond_t cell_config_done_cond;
extern pthread_mutex_t cell_config_done_mutex;
extern int cell_config_done;
#endif

//extern uint16_t ss_rnti_g;

static void sys_send_proxy(void *msg, int msgLen, struct TimingInfo_Type* at);
int cell_config_done_indication(void);
static uint16_t paging_ue_index_g = 0;
extern SSConfigContext_t SS_context;
int cell_index;

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
 * Function : wait_cell_si_config
 * Description: Waiting for SI cond_cell_si_config signal
 * from rrc_eNB,  which indicate the cell system information
 * configuration complete. After receiving the signal the SS
 * ready to perform the next SystemRequest_Type_Cell request.
 */
static void wait_cell_si_config(int cell_index)
{
  LOG_D(ENB_SS_SYS_TASK, "Waiting the SI configuration complete for cell: %d\n", cell_index);
  pthread_mutex_lock(&lock_cell_si_config);
  while (RC.ss.CC_update_flag[cell_index] == 1)
  {
    pthread_cond_wait(&cond_cell_si_config, &lock_cell_si_config);
  }
  pthread_mutex_unlock(&lock_cell_si_config);
}

void sys_handle_pdcch_order(struct RA_PDCCH_Order_Type *pdcchOrder);


/*
 * Function : sys_confirm_done_indication
 * Description: Sends the sys_confirm_done_mutex signl to PORTMAN_TASK,
 * as in portman_taks is waiting for the SYS confirm to be
 * received . After receiving this signal only the Portman_task proceeds
 * for processing next message.
 */
int sys_confirm_done_indication()
{
  if (sys_confirm_done < 0)
  {
    LOG_I(ENB_SS_SYS_TASK,"Signal to SYS_TASK for cell config done\n");
    pthread_mutex_lock(&sys_confirm_done_mutex);
    sys_confirm_done = 0;
    pthread_cond_broadcast(&sys_confirm_done_cond);
    pthread_mutex_unlock(&sys_confirm_done_mutex);
  }

  return 0;
}

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

/*
 * Function : bitStrint_to_byteArray
 * Description: Function used for converting Bit String to Byte Array
 */
static void bitStrint_to_byteArray(unsigned char arr[], int bit_length, unsigned char *key, bool int_key)
{
  int len = 8;
  int byte_count = bit_length/len;
  int count = byte_count/2;
  if(int_key == true)
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
int cell_config_done_indication()
{
#ifndef NR_RRC_VERSION
  if (cell_config_done < 0)
  {
    printf("Signal to OAI main code about cell config\n");
    pthread_mutex_lock(&cell_config_done_mutex);
    cell_config_done = 0;
    pthread_cond_broadcast(&cell_config_done_cond);
    pthread_mutex_unlock(&cell_config_done_mutex);
  }
#endif
  return 0;
}


void set_syscnf(bool reqCnfFlag, enum ConfirmationResult_Type_Sel resType, bool resVal, enum SystemConfirm_Type_Sel cnfType)
{
  if (reqCnfFlag == true)
  {
    SS_context.sys_cnf.resType = resType;
    SS_context.sys_cnf.resVal = resVal;
    SS_context.sys_cnf.cnfType = cnfType;
    SS_context.sys_cnf.cnfFlag = 1;
    memset(SS_context.sys_cnf.msg_buffer, 0, 1000);
  }
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
    uint16_t peerPort, struct TimingInfo_Type* at)
{
  // Create and alloc new message
  MessageDef *message_p = NULL;
  udp_data_req_t *udp_data_req_p = NULL;
  message_p = itti_alloc_new_message(TASK_SYS, 0, UDP_DATA_REQ);

  if (message_p)
  {
    udp_data_req_p = &message_p->ittiMsg.udp_data_req;
    udp_data_req_p->peer_address = peerIpAddr;
    udp_data_req_p->peer_port = peerPort;
    udp_data_req_p->buffer = buffer;
    udp_data_req_p->buffer_length = buffer_len;
    udp_data_req_p->buffer_offset = buffer_offset;
    if(at == NULL || !vt_timer_push_msg(at, TASK_UDP, 0, message_p))
    {
      LOG_A(ENB_SS_SYS_TASK, "Sending UDP_DATA_REQ to TASK_UDP\n");
      return itti_send_msg_to_task(TASK_UDP, 0, message_p);
    }

    return 0;
  }


  LOG_A(ENB_SS_SYS_TASK, "Failed Sending UDP_DATA_REQ length %u offset %u \n", buffer_len, buffer_offset);
  return -1;
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
  message_p = itti_alloc_new_message(TASK_SYS, 0, UDP_INIT);
  if (message_p == NULL)
  {
    return -1;
  }
  UDP_INIT(message_p).port = req->port;
  //addr.s_addr = req->ss_ip_addr;
  UDP_INIT(message_p).address = req->address; //inet_ntoa(addr);
  LOG_A(ENB_SS_SYS_TASK, "Tx UDP_INIT IP addr %s (%x)\n", UDP_INIT(message_p).address, UDP_INIT(message_p).port);
  return itti_send_msg_to_task(TASK_UDP, 0, message_p);
}

/*
 * Function : ss_task_sys_handle_timing_info
 * Description: Send the SS_SET_TIM_INFO message to Portman task for update the
 * timing info(sfn,sf)
 */
static void ss_task_sys_handle_timing_info(ss_set_timinfo_t *tinfo)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0, SS_SET_TIM_INFO);
  if (message_p)
  {
    LOG_A(ENB_SS_SYS_TASK, "Reporting info hsfn:%d sfn:%d\t sf:%d.\n",tinfo->hsfn, tinfo->sfn, tinfo->sf);
    SS_SET_TIM_INFO(message_p).hsfn = tinfo->hsfn;
    SS_SET_TIM_INFO(message_p).sf = tinfo->sf;
    SS_SET_TIM_INFO(message_p).sfn = tinfo->sfn;
    SS_SET_TIM_INFO(message_p).cell_index = cell_index;

    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN, 0, message_p);
    if (send_res < 0)
    {
      LOG_A(ENB_SS_SYS_TASK, "Error sending to [SS-PORTMAN] \n");
    }
  }
}
/*
 * Function : sys_add_reconfig_cell
 * Description: Funtion handler of SYS_PORT. Applies Cell
 * configuration changes to the SS. Builds the RRC config
 * message with the received cell configruation and sends
 * itti message to RRC layer.
 * Sends the CNF message for the required requests to PORTMAN
 * In :
 * AddOrReconfigure  - Cell configration received from PORTMAN
 * Out:
 * newState: The next state for the SYS State machine
 * FIXME: Currently the cell-id is same as the PCI. This needs to be
 * corrected. Does not impact TC_6_1_2_2.
 *
 */
int sys_add_reconfig_cell(struct SYSTEM_CTRL_REQ *req, bool *pIsRrcConfigReqSent)
{
  CellConfigReq_t *cellConfig;
  assert(req);
  struct CellConfigRequest_Type *Cell = &(req->Request.v.Cell);
  assert(Cell);
  struct CellConfigInfo_Type *AddOrReconfigure = &(Cell->v.AddOrReconfigure);
  assert(AddOrReconfigure);
  if (AddOrReconfigure->Basic.d == false && AddOrReconfigure->Active.d == false)
     return false;

  cellConfig = (CellConfigReq_t *)malloc(sizeof(CellConfigReq_t));
  cellConfig->header.preamble = 0xFEEDC0DE;
  cellConfig->header.msg_id = SS_CELL_CONFIG;
  cellConfig->header.length = sizeof(proxy_ss_header_t);

  for (int enb_id = 0; enb_id < RC.nb_inst; enb_id++)
  {
    printf("eNB_Inst %d Number of CC configured %d\n", enb_id, RC.nb_CC[enb_id]);
    MessageDef *msg_p = itti_alloc_new_message(TASK_SYS, ENB_MODULE_ID_TO_INSTANCE(enb_id), RRC_CONFIGURATION_REQ);

    RRC_CONFIGURATION_REQ(msg_p) = RC.rrc[enb_id]->configuration;
    RRC_CONFIGURATION_REQ(msg_p).ActiveParamPresent[cell_index] = false;
    if (AddOrReconfigure->Basic.d == true)
    {
      if (AddOrReconfigure->Basic.v.StaticCellInfo.d == true)
      {
        init_cell_context(cell_index, enb_id, msg_p);

        /** Handle Static Cell Info */
        /** TDD: 1 FDD: 0 in OAI */
        switch (AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.RAT.d)
        {
          case EUTRA_RAT_Type_FDD:
            RRC_CONFIGURATION_REQ(msg_p).frame_type[cell_index] = 0; /** FDD */
            break;
          case EUTRA_RAT_Type_TDD:
            RRC_CONFIGURATION_REQ(msg_p).frame_type[cell_index] = 1; /** TDD */
            break;
          case EUTRA_RAT_Type_HalfDuplexFDD:
          case EUTRA_RAT_Type_UNBOUND_VALUE:
            /* LOG */
            return false;
        }

        int band = AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.EutraBand;
        RRC_CONFIGURATION_REQ(msg_p).eutra_band[cell_index] = band;
        RRC_CONFIGURATION_REQ(msg_p).Nid_cell[cell_index] = AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.PhysicalCellId;
        SS_context.SSCell_list[cell_index].PhysicalCellId = AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.PhysicalCellId;

        /** TODO: Not filled now */
        /** eNB Cell ID: AddOrReconfigure->Basic.v.StaticCellInfo.v.Common.eNB_CellId.v */
        /** CellTimingInfo: */

        uint32_t dl_Freq = from_earfcn(band, AddOrReconfigure->Basic.v.StaticCellInfo.v.Downlink.Earfcn);
        RRC_CONFIGURATION_REQ(msg_p).downlink_frequency[cell_index] = dl_Freq;
        if (AddOrReconfigure->Basic.v.StaticCellInfo.v.Uplink.d == true)
        {
          uint32_t ul_Freq = from_earfcn(band, AddOrReconfigure->Basic.v.StaticCellInfo.v.Uplink.v.Earfcn);
          int ul_Freq_off = ul_Freq - dl_Freq;
          RRC_CONFIGURATION_REQ(msg_p).uplink_frequency_offset[cell_index] = (unsigned int)ul_Freq_off;
          SS_context.SSCell_list[cell_index].ul_earfcn = AddOrReconfigure->Basic.v.StaticCellInfo.v.Uplink.v.Earfcn;
          SS_context.SSCell_list[cell_index].ul_freq = ul_Freq;
        }
        // Updated the SS context for the frequency related configuration
        SS_context.SSCell_list[cell_index].dl_earfcn = AddOrReconfigure->Basic.v.StaticCellInfo.v.Downlink.Earfcn;
        SS_context.SSCell_list[cell_index].dl_freq = dl_Freq;

        switch (AddOrReconfigure->Basic.v.StaticCellInfo.v.Downlink.Bandwidth)
        {
          case SQN_CarrierBandwidthEUTRA_dl_Bandwidth_e_n6:
            RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] = 6;
            break;
          case SQN_CarrierBandwidthEUTRA_dl_Bandwidth_e_n15:
            RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] = 15;
            break;
          case SQN_CarrierBandwidthEUTRA_dl_Bandwidth_e_n25:
            RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] = 25;
            break;
          case SQN_CarrierBandwidthEUTRA_dl_Bandwidth_e_n50:
            RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] = 50;
            break;
          case SQN_CarrierBandwidthEUTRA_dl_Bandwidth_e_n75:
            RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] = 75;
            break;
          case SQN_CarrierBandwidthEUTRA_dl_Bandwidth_e_n100:
            RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index] = 100;
            break;
          default:
            /** LOG */
            LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest Invalid DL Bandwidth configuration \n");
            return false;
        }
        LOG_A(ENB_SS_SYS_TASK, "DL Bandwidth for cellIndex(%d):%d\n", cell_index, RRC_CONFIGURATION_REQ(msg_p).N_RB_DL[cell_index]);
      }
#define BCCH_CONFIG AddOrReconfigure->Basic.v.BcchConfig
      if (AddOrReconfigure->Basic.v.BcchConfig.d == true)
      {
        LOG_A(ENB_SS_SYS_TASK, "BCCH Config update in Cell config \n");
        RRC_CONFIGURATION_REQ(msg_p).stopSib1Transmission[cell_index] = (AddOrReconfigure->Basic.v.BcchConfig.v.StopSib1Transmission.d) ? 1 : 0;
        LOG_A(ENB_SS_SYS_TASK, "stopSib1Transmission for cellIndex(%d) %s\n", cell_index, RRC_CONFIGURATION_REQ(msg_p).stopSib1Transmission[cell_index] ? "yes" : "no");
        if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.d == true)
        {
          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.d == true)
          {

            /** For MIB */
            LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest PHICH Duration: %d\n", BCCH_CONFIG.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Duration);
            switch (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Duration)
            {
              case SQN_PHICH_Config_phich_Duration_e_normal:
                RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_duration = LTE_PHICH_Config__phich_Duration_normal;
                break;
              case SQN_PHICH_Config_phich_Duration_e_extended:
                RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_duration = LTE_PHICH_Config__phich_Duration_extended;
                break;
              default:
                LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest Invalid PHICH Duration\n");
                return false;
            }

            LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest PHICH Resource: %d\n", BCCH_CONFIG.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Resource);
            switch (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.v.message.phich_Config.phich_Resource)
            {
              case SQN_PHICH_Config_phich_Resource_e_oneSixth:
                RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_resource = LTE_PHICH_Config__phich_Resource_oneSixth;
                break;
              case SQN_PHICH_Config_phich_Resource_e_half:
                RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_resource = LTE_PHICH_Config__phich_Resource_half;
                break;
              case SQN_PHICH_Config_phich_Resource_e_one:
                RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_resource = LTE_PHICH_Config__phich_Resource_one;
                break;
              case SQN_PHICH_Config_phich_Resource_e_two:
                RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].phich_resource = LTE_PHICH_Config__phich_Resource_two;
                break;
              default:
                LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest Invalid PHICH Resource\n");
                return false;
            }

            RRC_CONFIGURATION_REQ(msg_p).schedulingInfoSIB1_BR_r13[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.MIB.v.message.schedulingInfoSIB1_BR_r13;
          }
          /** TODO: FIXME: Possible bug if not checking boolean flag for presence */
#define SIDL_SIB1_VAL AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIB1.v.message.v
#define SIB1_CELL_ACCESS_REL_INFO SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.cellAccessRelatedInfo
#define SIB1_CELL_SEL_INFO SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.cellSelectionInfo
#define SIB1_CELL_NON_CE SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.nonCriticalExtension.v.nonCriticalExtension
#define SIB1_CELL_Q_QUALMIN SIB1_CELL_NON_CE.v.cellSelectionInfo_v920.v.q_QualMin_r9
#define SIB1_TDD_CONFIG SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.tdd_Config.v
          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIB1.d == true)
          {
            LOG_A(ENB_SS_SYS_TASK, "[SIB1] q-RxLevMin: %d \n", SIB1_CELL_SEL_INFO.q_RxLevMin);
            RRC_CONFIGURATION_REQ(msg_p).q_RxLevMin[cell_index] = SIB1_CELL_SEL_INFO.q_RxLevMin;
            if (SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.nonCriticalExtension.d)
            {
              if (SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.nonCriticalExtension.v.nonCriticalExtension.d)
              {
                if (SIB1_CELL_NON_CE.v.cellSelectionInfo_v920.d)
                {
                  LOG_A(ENB_SS_SYS_TASK, "[SIB1] q-QualMin: %d \n", SIB1_CELL_Q_QUALMIN);
                  RRC_CONFIGURATION_REQ(msg_p).q_QualMin[cell_index] = SIB1_CELL_Q_QUALMIN;
                }
              }
            }
	    if(SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.tdd_Config.d)
            {
              RRC_CONFIGURATION_REQ(msg_p).tdd_config[cell_index] = SIB1_TDD_CONFIG.subframeAssignment;
              RRC_CONFIGURATION_REQ(msg_p).tdd_config_s[cell_index] = SIB1_TDD_CONFIG.specialSubframePatterns;
            }
          }

          RRC_CONFIGURATION_REQ(msg_p).schedulingInfo_count[cell_index] = SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.d;
          RRC_CONFIGURATION_REQ(msg_p).schedulingInfo[cell_index] = CALLOC(RRC_CONFIGURATION_REQ(msg_p).schedulingInfo_count[cell_index] , sizeof(struct lte_SchedulingInfo_s));
          int count =0;
          for (int k = 0; k < SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.d; k++)
          {
            if (SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.v[k].sib_MappingInfo.d) {
            RRC_CONFIGURATION_REQ(msg_p).schedulingInfo[cell_index][count].si_Periodicity = SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.v[k].si_Periodicity;
              for (int j = 0; j < SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.v[k].sib_MappingInfo.d; j++)
              {
                RRC_CONFIGURATION_REQ(msg_p).schedulingInfo[cell_index][count].sib_MappingInfo.LTE_SIB_Type[j] = SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.v[k].sib_MappingInfo.v[j];
              }
              RRC_CONFIGURATION_REQ(msg_p).schedulingInfo[cell_index][count].sib_MappingInfo.size = SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.schedulingInfoList.v[k].sib_MappingInfo.d;
              count ++;
            }
          }
          RRC_CONFIGURATION_REQ(msg_p).schedulingInfo_count[cell_index] = count;
          RRC_CONFIGURATION_REQ(msg_p).num_plmn[cell_index] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.d;
          RRC_CONFIGURATION_REQ(msg_p).systemInfoValueTag[cell_index] = SIDL_SIB1_VAL.c1.v.systemInformationBlockType1.systemInfoValueTag;

          RRC_CONFIGURATION_REQ(msg_p).num_plmn[cell_index] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.d;
          for (int i = 0; i < RRC_CONFIGURATION_REQ(msg_p).num_plmn[cell_index]; ++i)
          {
            if (SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mcc.d == true)
            {
              RRC_CONFIGURATION_REQ(msg_p).mcc[cell_index][i] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mcc.v[0] * 100 + SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mcc.v[1] * 10 + SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mcc.v[2];
            }
            if (SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.d == 2)
            {
              RRC_CONFIGURATION_REQ(msg_p).mnc[cell_index][i] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.v[0] * 10 + SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.v[1];
            }
            else if (SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.d == 3)
            {
              RRC_CONFIGURATION_REQ(msg_p).mnc[cell_index][i] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.v[0] * 100 + SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.v[1] * 10 + SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.v[2];
            }
            RRC_CONFIGURATION_REQ(msg_p).mnc_digit_length[cell_index][i] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].plmn_Identity.mnc.d;
            RRC_CONFIGURATION_REQ(msg_p).cellReservedForOperatorUse[cell_index][i] = SIB1_CELL_ACCESS_REL_INFO.plmn_IdentityList.v[i].cellReservedForOperatorUse;
          }
          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.d == true)
          {
            LOG_A(ENB_SS_SYS_TASK, "[SIs] size=%ld", AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.d);
            for (int i = 0; i < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.d; ++i)
            {
              if (SQN_BCCH_DL_SCH_MessageType_c1 == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.d)
              {
                if (SQN_BCCH_DL_SCH_MessageType_c1_systemInformation == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.d)
                {
                  if (SQN_SystemInformation_criticalExtensions_systemInformation_r8 == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.d)
                  {
                    for (int j = 0; j < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.d; j++)
                    {
                      if (SQN_SystemInformation_r8_IEs_sib_TypeAndInfo_s_sib2 == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].d)
                      {
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_config_index = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.radioResourceConfigCommon.prach_Config.prach_ConfigInfo.prach_ConfigIndex;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_high_speed = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.radioResourceConfigCommon.prach_Config.prach_ConfigInfo.highSpeedFlag;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_zero_correlation = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.radioResourceConfigCommon.prach_Config.prach_ConfigInfo.zeroCorrelationZoneConfig;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].prach_freq_offset = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.radioResourceConfigCommon.prach_Config.prach_ConfigInfo.prach_FreqOffset;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t300 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.ue_TimersAndConstants.t300;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t301 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.ue_TimersAndConstants.t301;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t310 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.ue_TimersAndConstants.t310;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_t311 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.ue_TimersAndConstants.t311;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_n310 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.ue_TimersAndConstants.n310;
                        RRC_CONFIGURATION_REQ(msg_p).radioresourceconfig[cell_index].ue_TimersAndConstants_n311 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib2.ue_TimersAndConstants.n311;
		      }
                      /* SIB3 */
                      if (SQN_SystemInformation_r8_IEs_sib_TypeAndInfo_s_sib3 == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].d)
                      {
                        RRC_CONFIGURATION_REQ(msg_p).q_Hyst[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.cellReselectionInfoCommon.q_Hyst;
                        RRC_CONFIGURATION_REQ(msg_p).threshServingLow[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.cellReselectionServingFreqInfo.threshServingLow;
                        RRC_CONFIGURATION_REQ(msg_p).cellReselectionPriority[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.cellReselectionServingFreqInfo.cellReselectionPriority;
                        RRC_CONFIGURATION_REQ(msg_p).sib3_q_RxLevMin[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.intraFreqCellReselectionInfo.q_RxLevMin;
                        RRC_CONFIGURATION_REQ(msg_p).t_ReselectionEUTRA[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.intraFreqCellReselectionInfo.t_ReselectionEUTRA;
                        RRC_CONFIGURATION_REQ(msg_p).neighCellConfig[cell_index] = (uint8_t)bin_to_int(AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.intraFreqCellReselectionInfo.neighCellConfig, 2);
                        if (RRC_CONFIGURATION_REQ(msg_p).sib3_q_QualMin[cell_index])
                        {
                          free(RRC_CONFIGURATION_REQ(msg_p).sib3_q_QualMin[cell_index]);
                          RRC_CONFIGURATION_REQ(msg_p).sib3_q_QualMin[cell_index] = NULL;
                        }
                        if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.q_QualMin_r9.d)
                        {
                          RRC_CONFIGURATION_REQ(msg_p).sib3_q_QualMin[cell_index] = calloc(1, sizeof(long));
                          *(RRC_CONFIGURATION_REQ(msg_p).sib3_q_QualMin[cell_index]) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.q_QualMin_r9.v;
                        }
                        if (RRC_CONFIGURATION_REQ(msg_p).sib3_threshServingLowQ[cell_index])
                        {
                          free(RRC_CONFIGURATION_REQ(msg_p).sib3_threshServingLowQ[cell_index]);
                          RRC_CONFIGURATION_REQ(msg_p).sib3_threshServingLowQ[cell_index] = NULL;
                        }
                        if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.threshServingLowQ_r9.d)
                        {
                          RRC_CONFIGURATION_REQ(msg_p).sib3_threshServingLowQ[cell_index] = calloc(1, sizeof(long));
                          *(RRC_CONFIGURATION_REQ(msg_p).sib3_threshServingLowQ[cell_index]) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.threshServingLowQ_r9.v;
                        }

                        if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.s_NonIntraSearch_v920.d){
                          RRC_CONFIGURATION_REQ(msg_p).sib3_s_NonIntraSearchP[cell_index] = calloc(1, sizeof(long));
                          *(RRC_CONFIGURATION_REQ(msg_p).sib3_s_NonIntraSearchP[cell_index]) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.s_NonIntraSearch_v920.v.s_NonIntraSearchP_r9;
                          RRC_CONFIGURATION_REQ(msg_p).sib3_s_NonIntraSearchQ[cell_index] = calloc(1, sizeof(long));
                          *(RRC_CONFIGURATION_REQ(msg_p).sib3_s_NonIntraSearchQ[cell_index]) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib3.s_NonIntraSearch_v920.v.s_NonIntraSearchQ_r9;
                        }
                      }
                      /* SIB4: Received SIB4 from TTCN */
                      if (SQN_SystemInformation_r8_IEs_sib_TypeAndInfo_s_sib4 == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].d)
                      {
                        RRC_CONFIGURATION_REQ(msg_p).sib4_Present[cell_index] = true;
                        if (true == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqNeighCellList.d)
                        {
                          RRC_CONFIGURATION_REQ(msg_p).intraFreqNeighCellListCount[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqNeighCellList.v.d;
                          RRC_CONFIGURATION_REQ(msg_p).intraFreqNeighCellList[cell_index] = CALLOC(RRC_CONFIGURATION_REQ(msg_p).intraFreqNeighCellListCount[cell_index], sizeof(struct IntraFreqNeighCellInfo_s));
                          RRC_CONFIGURATION_REQ(msg_p).intraFreqNeighCellListPresent[cell_index] = true;
                          for (int k = 0; k < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqNeighCellList.v.d; k++)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).intraFreqNeighCellList[cell_index][k].physCellId = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqNeighCellList.v.v[k].physCellId;
                            RRC_CONFIGURATION_REQ(msg_p).intraFreqNeighCellList[cell_index][k].q_OffsetCell = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqNeighCellList.v.v[k].q_OffsetCell;
                          }
                        }
                        if(true == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqExcludedCellList.d) {
                             RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellListCount[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqExcludedCellList.v.d;
                             RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellList[cell_index] = CALLOC(RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellListCount[cell_index],sizeof(struct PhysCellIdRange_s));
                             RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellListPresent[cell_index] = true;
                             for(int k=0;k < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqExcludedCellList.v.d; k++) {
                               RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellList[cell_index][k].start = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqExcludedCellList.v.v[k].start;
                                if(AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqExcludedCellList.v.v[k].range.d) {
                                 RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellList[cell_index][k].range_Present = true;
                                 RRC_CONFIGURATION_REQ(msg_p).intraFreqExcludedCellList[cell_index][k].range = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib4.intraFreqExcludedCellList.v.v[k].range.v;
                                }
                             }
                         }
                      }
                      /* SIB5: Received SIB5 from TTCN */
                      if (SQN_SystemInformation_r8_IEs_sib_TypeAndInfo_s_sib5 == AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].d)
                      {
                        RRC_CONFIGURATION_REQ(msg_p).sib5_Present[cell_index] = true;
                        RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfoCount[cell_index] = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.d;
                        RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index] = CALLOC(RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfoCount[cell_index], sizeof(struct InterFreqCarrierFreqInfo_s));
                        for (int k = 0; k < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.d; k++)
                        {
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].dl_CarrierFreq = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].dl_CarrierFreq;
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_RxLevMin = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_RxLevMin;
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].p_Max.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].p_Max_Present = true;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].p_Max = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].p_Max.v;
                          }
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].t_ReselectionEUTRA = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].t_ReselectionEUTRA;
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].t_ReselectionEUTRA_SF.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].t_ReselectionEUTRA_SF_Present = true;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].t_ReselectionEUTRA_SF->sf_Medium = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].t_ReselectionEUTRA_SF.v.sf_Medium;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].t_ReselectionEUTRA_SF->sf_High = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].t_ReselectionEUTRA_SF.v.sf_High;
                          }
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].threshX_High = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].threshX_High;
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].threshX_Low = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].threshX_Low;
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].allowedMeasBandwidth = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].allowedMeasBandwidth;
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].presenceAntennaPort1 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].presenceAntennaPort1;
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].cellReselectionPriority.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].cellReselectionPriority = CALLOC(1, sizeof(long));
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].cellReselectionPriority_Present = true;
                            *(RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].cellReselectionPriority) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].cellReselectionPriority.v;
                          }
                          RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].neighCellConfig = (uint8_t)bin_to_int(AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].neighCellConfig, 2);
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_OffsetFreq.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_OffsetFreqPresent = true;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_OffsetFreq = CALLOC(1, sizeof(enum LTE_Q_OffsetRange));
                            *(RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_OffsetFreq) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_OffsetFreq.v;
                          }
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqNeighCellList.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqNeighCellList_Present = true;
                            for (int l = 0; l < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqNeighCellList.v.d; l++)
                            {
                              RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqNeighCellList->physCellId = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqNeighCellList.v.v[l].physCellId;
                              RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqNeighCellList->q_OffsetCell = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqNeighCellList.v.v[l].q_OffsetCell;
                            }
                          }
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqExcludedCellList.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqExcludedCellList_Present = true;
                            for (int m = 0; m < AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqExcludedCellList.v.d; m++)
                            {
                              RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqExcludedCellList->start = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqExcludedCellList.v.v[m].start;
                              if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqExcludedCellList.v.v[m].range.d)
                              {
                                RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqExcludedCellList->range_Present = true;
                                RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].interFreqExcludedCellList->range = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].interFreqExcludedCellList.v.v[m].range.v;
                              }
                            }
                          }
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_QualMin_r9.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_QualMin_r9_Present = true;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_QualMin_r9 = CALLOC(1, sizeof(long));
                            *(RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_QualMin_r9) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_QualMin_r9.v;
                          }
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].threshX_Q_r9.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].threshX_Q_r9_Present = true;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].threshX_Q_r9.threshX_HighQ_r9 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].threshX_Q_r9.v.threshX_HighQ_r9;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].threshX_Q_r9.threshX_LowQ_r9 = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].threshX_Q_r9.v.threshX_LowQ_r9;
                          }
                          if (AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_QualMinWB_r11.d)
                          {
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_QualMinWB_r11_Present = true;
                            RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_QualMinWB_r11 = CALLOC(1, sizeof(long));
                            *(RRC_CONFIGURATION_REQ(msg_p).InterFreqCarrierFreqInfo[cell_index][k].q_QualMinWB_r11) = AddOrReconfigure->Basic.v.BcchConfig.v.BcchInfo.v.SIs.v.v[i].message.v.c1.v.systemInformation.criticalExtensions.v.systemInformation_r8.sib_TypeAndInfo.v[j].v.sib5.interFreqCarrierFreqList.v[k].q_QualMinWB_r11.v;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          RRC_CONFIGURATION_REQ(msg_p).tac[cell_index] = 0;
          for (int i = 0; i < 16; i++)
          {
            RRC_CONFIGURATION_REQ(msg_p).tac[cell_index] += (SIB1_CELL_ACCESS_REL_INFO.trackingAreaCode[i] << (15 - i));
          }
          LOG_A(ENB_SS_SYS_TASK, "[SIB1] tac: 0x%x\n", RRC_CONFIGURATION_REQ(msg_p).tac[cell_index]);
          RRC_CONFIGURATION_REQ(msg_p).cell_identity[cell_index] = 0;
          for (int i = 0; i < 28; i++)
          {
            RRC_CONFIGURATION_REQ(msg_p).cell_identity[cell_index] += (SIB1_CELL_ACCESS_REL_INFO.cellIdentity[i] << (27 - i));
          }
          LOG_A(ENB_SS_SYS_TASK, "[SIB1] Cell Identity: 0x%x\n", RRC_CONFIGURATION_REQ(msg_p).cell_identity[cell_index]);
          RRC_CONFIGURATION_REQ(msg_p).cellBarred[cell_index] = SIB1_CELL_ACCESS_REL_INFO.cellBarred;
          RRC_CONFIGURATION_REQ(msg_p).intraFreqReselection[cell_index] = SIB1_CELL_ACCESS_REL_INFO.intraFreqReselection;
        }
      }

      /** Handle Initial Cell power, Sending to Proxy */
      if (AddOrReconfigure->Basic.v.InitialCellPower.d == true)
      {
        SS_context.SSCell_list[cell_index].maxRefPower = AddOrReconfigure->Basic.v.InitialCellPower.v.MaxReferencePower;
        switch (AddOrReconfigure->Basic.v.InitialCellPower.v.Attenuation.d)
        {
          case Attenuation_Type_Value:
            LOG_A(ENB_SS_SYS_TASK, "[InitialCellPower.v.Attenuation.v.Value] Attenuation turned on value: %d dBm \n",
                AddOrReconfigure->Basic.v.InitialCellPower.v.Attenuation.v.Value);
            cellConfig->initialAttenuation = AddOrReconfigure->Basic.v.InitialCellPower.v.Attenuation.v.Value;
            break;
          case Attenuation_Type_Off:
            LOG_A(ENB_SS_SYS_TASK, "[InitialCellPower.v.Attenuation.v.Value] Attenuation turned off \n");
            cellConfig->initialAttenuation = 80; /* attnVal hardcoded currently but Need to handle proper Attenuation_Type_Off */
            break;
          case Attenuation_Type_UNBOUND_VALUE:
          default:
            LOG_A(ENB_SS_SYS_TASK, "[InitialCellPower.v.Attenuation.v.Value] Unbound or Invalid value received\n");
        }

        cellConfig->header.cell_id = SS_context.SSCell_list[cell_index].PhysicalCellId;
        cellConfig->maxRefPower = SS_context.SSCell_list[cell_index].maxRefPower;
        cellConfig->dl_earfcn = SS_context.SSCell_list[cell_index].dl_earfcn;
        cellConfig->header.cell_index = cell_index;
        LOG_A(ENB_SS_SYS_TASK, "===Cell configuration received for cell_id: %d Initial attenuation: %d Max ref power: %d for DL_EARFCN: %d cell_index %d=== \n",
            cellConfig->header.cell_id,
            cellConfig->initialAttenuation, cellConfig->maxRefPower,
            cellConfig->dl_earfcn, cell_index);
        sys_send_proxy((void *)cellConfig, sizeof(CellConfigReq_t), &req->Common.TimingInfo);
      }
    }
    // Cell Config Active Param
    if (AddOrReconfigure->Active.d == true)
    {
      LOG_A(ENB_SS_SYS_TASK, "Cell Config Active Present\n");
      LOG_A(ENB_SS_SYS_TASK, "Active.v.C_RNTI.d=%d Active.v.RachProcedureConfig.d=%d\n", AddOrReconfigure->Active.v.C_RNTI.d, AddOrReconfigure->Active.v.RachProcedureConfig.d);
      if (AddOrReconfigure->Active.v.C_RNTI.d == true)
      {
	RRC_CONFIGURATION_REQ(msg_p).ActiveParamPresent[cell_index] = true;
        RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].b_C_RNTI_Present = true;
        RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].C_RNTI = bin_to_int(AddOrReconfigure->Active.v.C_RNTI.v, 16);
        SS_context.SSCell_list[cell_index].ss_rnti_g = RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].C_RNTI;
        LOG_A(ENB_SS_SYS_TASK, "C_RNTI present in Active Cell Config %d\n", RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].C_RNTI);
      }
      if (AddOrReconfigure->Active.v.RachProcedureConfig.d == true)
      {
        if (AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.d == true)
        {
	  RRC_CONFIGURATION_REQ(msg_p).ActiveParamPresent[cell_index] = true;
	  RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].numRar = AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.d;
          LOG_A(ENB_SS_SYS_TASK,"SS controlled RAR config count: %d\n", RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].numRar);
          for (int i = 0; i < (AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.d); i++)
          {
            if (RandomAccessResponseConfig_Type_Ctrl == AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.d)
            {
              LOG_A(ENB_SS_SYS_TASK, "RAResponse present in Active Cell Config\n");
              if (RandomAccessResponse_Type_List == AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.v.Ctrl.Rar.d)
              {
                RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].Rar[i].b_rarResponse = true; /*Indicates RA response: Allows tx of RAR/msg2*/
                LOG_A(ENB_SS_SYS_TASK, "RAResponse allowed\n");
                for (int j = 0; j < AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.v.Ctrl.Rar.v.List.d; j++)
                {
                  if (TempC_RNTI_Type_SameAsC_RNTI == AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.v.Ctrl.Rar.v.List.v[j].TempC_RNTI.d)
                  {
                    LOG_A(ENB_SS_SYS_TASK, "RAResponse present in Active Cell Config is TempC_RNTI_Type_SameAsC_RNTI\n");
                    RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].Rar[j].Temp_C_RNTI = RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].C_RNTI;
                  }
                  else if (TempC_RNTI_Type_Explicit == AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.v.Ctrl.Rar.v.List.v[j].TempC_RNTI.d)
                  {
                    LOG_A(ENB_SS_SYS_TASK, "RAResponse present in Active Cell Config is TempC_RNTI_Type_Explicit\n");
                    RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].Rar[j].Temp_C_RNTI = bin_to_int(AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.v.Ctrl.Rar.v.List.v[j].TempC_RNTI.v.Explicit, 16);
                  }
                }
              }
              else if(RandomAccessResponse_Type_None == AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].RAResponse.v.Ctrl.Rar.d)
              {
		 RRC_CONFIGURATION_REQ(msg_p).ActiveParam[cell_index].Rar[i].b_rarResponse = false; /*Indicates non RA response: Avoids tx of RAR/msg2*/
                 LOG_A(ENB_SS_SYS_TASK, "RAResponse not allowed\n");
              }
            }
            if (AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].ContentionResolutionCtrl.d == ContentionResolutionCtrl_Type_TCRNTI_Based)
            {
              if (AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].ContentionResolutionCtrl.v.TCRNTI_Based.d == TCRNTI_ContentionResolutionCtrl_Type_MacPdu)
              {
                if (AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].ContentionResolutionCtrl.v.TCRNTI_Based.v.MacPdu.ContainedRlcPdu.d == ContentionResolution_ContainedDlschSdu_Type_RlcPduCCCH)
                {
                  RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH_Present[cell_index] = true;
                  RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH_Size[cell_index] = AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].ContentionResolutionCtrl.v.TCRNTI_Based.v.MacPdu.ContainedRlcPdu.v.RlcPduCCCH.d;
                  memcpy(RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH[cell_index], AddOrReconfigure->Active.v.RachProcedureConfig.v.RachProcedureList.v.v[i].ContentionResolutionCtrl.v.TCRNTI_Based.v.MacPdu.ContainedRlcPdu.v.RlcPduCCCH.v, RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH_Size[cell_index]);
                }
                else
                {
                  // RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH_Present[cell_index] = false;
                  if (SS_context.SSCell_list[cell_index].State == SS_STATE_NOT_CONFIGURED)
                    RC.ss.CBRA_flag[cell_index] = true;
                }
              }
            }
          }
        }
      }  
    }
    else
    {
      // RRC_CONFIGURATION_REQ(msg_p).ActiveParamPresent[cell_index] = false;
    }
    LOG_A(ENB_SS_SYS_TASK, "SS: ActiveParamPresent: %d, SS: Basic Present %d RlcPduCCCH_Present: %d, RLC Container PDU size: %d \n", RRC_CONFIGURATION_REQ(msg_p).ActiveParamPresent[cell_index], AddOrReconfigure->Basic.d,RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH_Present[cell_index], RRC_CONFIGURATION_REQ(msg_p).RlcPduCCCH_Size[cell_index]);
    if ((AddOrReconfigure->Basic.d == true) || (RRC_CONFIGURATION_REQ(msg_p).ActiveParamPresent[cell_index] == true))
    {
	    // store the modified cell config back
	    memcpy(&(RC.rrc[enb_id]->configuration), &RRC_CONFIGURATION_REQ(msg_p), sizeof(RRC_CONFIGURATION_REQ(msg_p)));
	    *pIsRrcConfigReqSent = true;
	    if (!vt_timer_push_msg(&req->Common.TimingInfo, TASK_RRC_ENB,ENB_MODULE_ID_TO_INSTANCE(enb_id), msg_p))
	    {
		    LOG_A(ENB_SS_SYS_TASK, "Sending Cell configuration to RRC from SYSTEM_CTRL_REQ \n");
		    itti_send_msg_to_task(TASK_RRC_ENB, ENB_MODULE_ID_TO_INSTANCE(enb_id), msg_p);
	    }
    }
    else
    {
	    itti_free(ITTI_MSG_ORIGIN_ID(msg_p),msg_p);
    }
    LOG_E(ENB_SS_SYS_TASK, "pIsRrcConfigReqSent %d",*pIsRrcConfigReqSent);
    /* Active Config for ULGrant Params */
    bool destTaskMAC = false;
    for (int enb_id = 0; enb_id < RC.nb_inst; enb_id++)
    {
      if (AddOrReconfigure->Active.d == true)
      {
        do
        {
          if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.d == true)
          {
            msg_p = itti_alloc_new_message(TASK_ENB_APP, ENB_MODULE_ID_TO_INSTANCE(enb_id), SS_ULGRANT_INFO);
            LOG_I(ENB_SS_SYS_TASK, "Received ULGrant in Active State for cell_index:%d\n", cell_index);
            SS_ULGRANT_INFO(msg_p).cell_index = cell_index;
            if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.d == UL_GrantConfig_Type_OnSR_Reception)
            {
              RC.ss.ulgrant_info[cell_index].ulGrantType = ON_SR_RECEPTION_PRESENT;
              SS_ULGRANT_INFO(msg_p).ulGrantType = ON_SR_RECEPTION_PRESENT;
              LOG_I(ENB_SS_SYS_TASK, "Received ulGrantType ON_SR_RECEPTION\n");
              destTaskMAC = true;
            }
            else if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.d == UL_GrantConfig_Type_None)
            {
              RC.ss.ulgrant_info[cell_index].ulGrantType = NONE_PRESENT;
              SS_ULGRANT_INFO(msg_p).ulGrantType = NONE_PRESENT;
              LOG_I(ENB_SS_SYS_TASK, "Received ulGrantType UL_GrantConfig_Type_None for cell_id:%d\n", cell_index);
              destTaskMAC = true;
            }
            else if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.d == UL_GrantConfig_Type_Periodic)
            {
              SS_ULGRANT_INFO(msg_p).ulGrantType = PERIODIC_PRESENT;
              SS_ULGRANT_INFO(msg_p).periodiGrantInfo.ULGrantPeriodType  =
                AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.Period.d;
              LOG_I(ENB_SS_SYS_TASK, "ULGrantPeriodType:%d\n", SS_ULGRANT_INFO(msg_p).periodiGrantInfo.ULGrantPeriodType);
              if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.Period.d == ULGrant_Period_Type_Duration)
              {
                SS_ULGRANT_INFO(msg_p).periodiGrantInfo.period.duration =
                  AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.Period.v.Duration;
                LOG_I(ENB_SS_SYS_TASK, "Received Periodic ULGrant type ULGrant_Period_Type_Duration: %d received:%d cell_index:%d\n",
                    SS_ULGRANT_INFO(msg_p).periodiGrantInfo.period.duration,
                    AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.Period.v.Duration,
                    cell_index);

              }
              else if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.Period.d == ULGrant_Period_Type_OnlyOnce)
              {
                LOG_I(ENB_SS_SYS_TASK, "Received Periodic ULGrant type ULGrant_Period_Type_OnlyOnce\n");
                SS_ULGRANT_INFO(msg_p).periodiGrantInfo.period.onlyOnce = true;
              }

              if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.NoOfRepetitions.d == 1)
              {
                SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.Continuous = 1;
                SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.NumOfCycles = 0;
                LOG_I(ENB_SS_SYS_TASK, "line:%d Continuous:%d NumOfCycles:%d\n",
                  __LINE__,
                  SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.Continuous,
                  SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.NumOfCycles);
              }
              else if (AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.NoOfRepetitions.d == 2)
              {
                SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.NumOfCycles =
                  AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.v.Periodic.NoOfRepetitions.v.NumOfCycles;
                SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.Continuous = 0;
                LOG_I(ENB_SS_SYS_TASK, "line:%d Continuous:%d NumOfCycles:%d\n",
                  __LINE__,
                  SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.Continuous,
                  SS_ULGRANT_INFO(msg_p).periodiGrantInfo.transRepType.NumOfCycles);
              }
              else
              {
                LOG_A(ENB_SS_SYS_TASK, "Received invalid Transmission Repetation Type in Periodic ULGrant%d\n",
                    SS_ULGRANT_INFO(msg_p).periodiGrantInfo.ULGrantPeriodType);
                break;
              }
              destTaskMAC = true;
            }
            else
            {
              LOG_A(ENB_SS_SYS_TASK, "ULGrantType %d is not supported. Current supported ULGrant is OnSR_Reception & periodic only \n",
                  AddOrReconfigure->Active.v.CcchDcchDtchConfig.v.UL.v.UL_GrantConfig.v.d);
            }
          }
        }while(0);
      }

      /* Currently only queuing UL_GrantConfig from Cell Config API. On timer expiry, message will be sent to MAC */
      if (destTaskMAC == true)
      {
        if (!vt_timer_push_msg(&req->Common.TimingInfo, TASK_MAC_ENB,ENB_MODULE_ID_TO_INSTANCE(enb_id), msg_p))
        {
          itti_send_msg_to_task(TASK_MAC_ENB, ENB_MODULE_ID_TO_INSTANCE(enb_id), msg_p);
        }
      }
    }
  }
  return true;
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
                         enum SystemConfirm_Type_Sel cnfType,
                         void *msg)
{
  /* The request has send confirm flag flase so do nothing in this funciton */
  if (reqCnfFlag_g == false)
  {
    LOG_A(ENB_SS_SYS_TASK, "No confirm required\n");
    return ;
  }

  struct SYSTEM_CTRL_CNF *msgCnf = CALLOC(1, sizeof(struct SYSTEM_CTRL_CNF));
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0, SS_SYS_PORT_MSG_CNF);

  if (message_p)
  {
    LOG_A(ENB_SS_SYS_TASK, "Send SS_SYS_PORT_MSG_CNF cnf_Type %d, res_Type %d\n",cnfType,resType);
    msgCnf->Common.CellId = SS_context.SSCell_list[cell_index].eutra_cellId;
    msgCnf->Common.Result.d = resType;
    msgCnf->Common.Result.v.Success = resVal;
    msgCnf->Confirm.d = cnfType;
    switch (cnfType)
    {
      case SystemConfirm_Type_Cell:
        {
          LOG_A(ENB_SS_SYS_TASK, "Send confirm for cell configuration\n");
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
        if (msg)
          memcpy(&msgCnf->Confirm.v.PdcpCount, msg, sizeof(struct PDCP_CountCnf_Type));
        else
          SS_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
        break;

      case SystemConfirm_Type_Paging:
        msgCnf->Confirm.v.Paging = true;
        break;
      case SystemConfirm_Type_PdcchOrder:
        {
          LOG_A(ENB_SS_SYS_TASK, "Send confirm for PDCCHOrder to Port Sys \n");
          msgCnf->Confirm.v.PdcchOrder = true;
          break;
        }
      case SystemConfirm_Type_L1MacIndCtrl:
        msgCnf->Confirm.v.L1MacIndCtrl = true;
        break;
      case SystemConfirm_Type_Sps:
      case SystemConfirm_Type_RlcIndCtrl:
      case SystemConfirm_Type_PdcpHandoverControl:
        msgCnf->Confirm.v.PdcpHandoverControl = true;
        break;
      case SystemConfirm_Type_L1_TestMode:
      case SystemConfirm_Type_ActivateScell:
      case SystemConfirm_Type_MbmsConfig:
      case SystemConfirm_Type_PDCCH_MCCH_ChangeNotification:
      case SystemConfirm_Type_MSI_Config:
      case SystemConfirm_Type_OCNG_Config:
      case SystemConfirm_Type_DirectIndicationInfo:
      default:
        LOG_A(ENB_SS_SYS_TASK, "Error not handled CNF TYPE to [SS-PORTMAN] \n");
    }
    SS_SYS_PORT_MSG_CNF(message_p).cnf = msgCnf;
    int send_res = itti_send_msg_to_task(TASK_SS_PORTMAN, 0, message_p);
    if (send_res < 0)
    {
      LOG_A(ENB_SS_SYS_TASK, "Error sending to [SS-PORTMAN] \n");
    }
  }
//  sys_confirm_done_indication();
}
/*
 * Function : sys_handle_cell_config_req
 * Description: Funtion handler of SYS_PORT. Handles the Cell
 * configuration command received from TTCN via the PORTMAN.
 * Invokes the subroutinge to accept the configuration.
 * In :
 * req  - Cell Request received from the TTCN via PORTMAN
 * Out:
 * newState: The next state for the SYS State machine
 *
 */
int sys_handle_cell_config_req(struct SYSTEM_CTRL_REQ *req)
{
  bool isRrcConfigReqSent = false;
  int status = false;
  int returnState = SS_context.SSCell_list[cell_index].State;
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_Cell;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  assert(req);
  struct CellConfigRequest_Type *Cell=&(req->Request.v.Cell);
  assert(Cell);

  switch (Cell->d)
  {
  case CellConfigRequest_Type_AddOrReconfigure:

    //reset cell configuration status
    cell_config_done = -1;
    LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest_Type_AddOrReconfigure receivied\n");
    status = sys_add_reconfig_cell(req,&isRrcConfigReqSent);
    if (status)
    {
      /** TODO Signal to main thread */
      LOG_A(ENB_SS_SYS_TASK, "Signalling main thread for cell config done indication\n");
      cell_config_done_indication();
    }
    //cell configuration
    if ( SS_context.SSCell_list[cell_index].State == SS_STATE_NOT_CONFIGURED)
    {
       //The flag is used to initilize the cell in the RRC layer during init_SI funciton
        RC.ss.CC_conf_flag[cell_index] = 1;
        RC.ss.CC_update_flag[cell_index] = 1;
        returnState = SS_STATE_CELL_CONFIGURED;
      //Increment nb_cc only from 2nd cell as the initilization is done for 1 CC
      if (cell_index)
      {
        //Increment the nb_CC supported as new cell is confiured.
        if (RC.nb_CC[0] >= MAX_NUM_CCs) {
          LOG_E (ENB_SS_SYS_TASK,"[SYS] Can't add cell, MAX_NUM_CC reached (%d > %d) \n", RC.nb_CC[0], MAX_NUM_CCs);
        } else {
          RC.nb_CC[0] ++;
          //Set the number of MAC_CC to current configured CC value
          //*RC.nb_mac_CC= RC.nb_CC[0];

          LOG_I (ENB_SS_SYS_TASK,"CC-MGMT nb_cc is incremented current Configured RC.nb_CC %d current CC_index %d RC.nb_mac_CC %d\n",
                RC.nb_CC[0],cell_index,*RC.nb_mac_CC);
        }
      }
    }
    else
    {
      RC.ss.CC_update_flag[cell_index] = 1;
      LOG_I (ENB_SS_SYS_TASK,"[SYS] CC-MGMT configured RC.nb_CC %d current updated CC_index %d RC.nb_mac_CC %d\n",
                RC.nb_CC[0],cell_index,*RC.nb_mac_CC);
    }
    if (status && Cell->v.AddOrReconfigure.Basic.d)
    {
	    /**case: When RRC_CONFIGURATION_REQ is sent
	      SS shall unblock Portman and send SYS_CONFIRM after receiving RRC_CONFIGURATION_CNF**/
	    wait_cell_si_config(cell_index);
    }
    if((status == false) ||  (isRrcConfigReqSent == false))
    {
	    /**case: When RRC_CONFIGURATION_REQ is not sent
	      SS need to  unblock Portman and sent sys_confirm from here**/
            resVal = status;
	    send_sys_cnf(resType, resVal, cnfType, NULL);
	    sys_confirm_done_indication();
    }
    break;
  case CellConfigRequest_Type_Release: /**TODO: NOT IMPLEMNTED */
    LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest_Type_Release receivied\n");
    returnState = SS_STATE_NOT_CONFIGURED;
    break;
  case CellConfigRequest_Type_UNBOUND_VALUE: /** TODO: NOT IMPLEMNTED */
    LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest_Type_UNBOUND_VALUE receivied\n");
    break;
  default:
    LOG_A(ENB_SS_SYS_TASK, "CellConfigRequest INVALID Type receivied\n");
  }

  set_syscnf(reqCnfFlag_g, resType, resVal, cnfType);
  return returnState;
}

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
static int sys_handle_radiobearer_list(struct SYSTEM_CTRL_REQ *req)
{
  int returnState = SS_context.SSCell_list[cell_index].State;
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_RadioBearerList;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  assert(req);
  struct RadioBearer_Type_RadioBearerList_Type_Dynamic *BearerList = &(req->Request.v.RadioBearerList);
  assert(BearerList);
  MessageDef *msg_p = itti_alloc_new_message(TASK_SYS, 0, RRC_RBLIST_CFG_REQ);
  if (msg_p)
  {
    LOG_A(ENB_SS_SYS_TASK, "BearerList size:%lu\n", BearerList->d);
    RRC_RBLIST_CFG_REQ(msg_p).rb_count = 0;
    RRC_RBLIST_CFG_REQ(msg_p).cell_index = cell_index;
    for (int i = 0; i < BearerList->d; i++)
    {
      LOG_A(ENB_SS_SYS_TASK, "RB Index i:%d\n", i);
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
                LOG_A(ENB_SS_SYS_TASK, "PDCP Config for Bearer Id: %d is Null\n", RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbId);
              }
              else if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.d == PDCP_RBConfig_Type_Drb)
              {
                RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isPDCPConfigValid = true;
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
                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.headerCompression.d == SQN_PDCP_Config_headerCompression_rohc)
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
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_DataSplitThreshold_r13.v.d == SQN_PDCP_Config_ul_DataSplitThreshold_r13_setup)
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
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.statusFeedback_r13.v.d == SQN_PDCP_Config_statusFeedback_r13_setup)
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
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ul_LWA_Config_r14.v.d == SQN_PDCP_Config_ul_LWA_Config_r14_setup)
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
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.uplinkOnlyHeaderCompression_r14.v.d == SQN_PDCP_Config_uplinkOnlyHeaderCompression_r14_rohc_r14)
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
                    if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_DuplicationConfig_r15.v.d == SQN_PDCP_Config_pdcp_DuplicationConfig_r15_setup)
                    {
                      RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.Pdcp.ext6->pdcp_DuplicationConfig_r15->choice.setup.pdcp_Duplication_r15 = BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.pdcp_DuplicationConfig_r15.v.v.setup.pdcp_Duplication_r15;
                    }
                  }

                  if (BearerList->v[i].Config.v.AddOrReconfigure.Pdcp.v.v.Config.Rb.v.v.Drb.v.R8.ethernetHeaderCompression_r16.d)
                  {
                    LOG_A(ENB_SS_SYS_TASK, "Unsupported IE: ethernetHeaderCompression_r16 \n");
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
            RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isRLCConfigValid = true;
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
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isLogicalChannelIdValid = true;
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.LogicalChannelId = BearerList->v[i].Config.v.AddOrReconfigure.LogicalChannelId.v;
        }
        else
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.LogicalChannelId = 0;
        }

        /* Populate the MAC Configuration for the radio Bearer */
        if (BearerList->v[i].Config.v.AddOrReconfigure.Mac.d)
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isMacConfigValid = true;
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
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isMacTestModeValid = false;
	  if(BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.TestMode.d)
	  {
	    if(BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.TestMode.v.d == MAC_TestModeConfig_Type_Info)
	    {
	      if(BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.TestMode.v.v.Info.DiffLogChId.d == MAC_Test_DLLogChID_Type_LogChId)
	      {
              RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isMacTestModeValid = true;
              RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.MacTestModeLogicalChannelId = BearerList->v[i].Config.v.AddOrReconfigure.Mac.v.TestMode.v.v.Info.DiffLogChId.v.LogChId;
        }
        else
        {
          LOG_E(ENB_SS_SYS_TASK, "isMacTestModeValid is false, Info.DiffLogChId.d != MAC_Test_DLLogChID_Type_LogChId \n");
        }
	    }
	  }
        }

        if (BearerList->v[i].Config.v.AddOrReconfigure.DiscardULData.d)
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.isDiscardULDataValid = true;
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.DiscardULData = BearerList->v[i].Config.v.AddOrReconfigure.DiscardULData.v;
        }
        else
        {
          RRC_RBLIST_CFG_REQ(msg_p).rb_list[i].RbConfig.DiscardULData = false;
        }
      }
    }

    if (!vt_timer_push_msg(&req->Common.TimingInfo, TASK_RRC_ENB,0, msg_p))
    {
      itti_send_msg_to_task(TASK_RRC_ENB, 0, msg_p);
    }

  }

  set_syscnf(reqCnfFlag_g, resType, resVal, cnfType);
  returnState = reqCnfFlag_g ? SS_STATE_CELL_ACTIVE : returnState;
//  send_sys_cnf(resType, resVal, cnfType, NULL);
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
int sys_handle_pdcp_count_req(struct PDCP_CountReq_Type *PdcpCount)
{
  int returnState = SS_context.SSCell_list[cell_index].State;
  int send_res = -1;

  switch (PdcpCount->d)
  {
  case PDCP_CountReq_Type_Get:
    LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type_Get receivied\n");
    MessageDef *get_p = itti_alloc_new_message(TASK_SYS, 0, SS_REQ_PDCP_CNT);
    SS_REQ_PDCP_CNT(get_p).rnti = SS_context.SSCell_list[cell_index].ss_rnti_g;
    switch (PdcpCount->v.Get.d)
    {
    case PdcpCountGetReq_Type_AllRBs:
      LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type_Get AllRBs receivied\n");
      SS_REQ_PDCP_CNT(get_p).rb_id = -1;
      break;
    case PdcpCountGetReq_Type_SingleRB:
      LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type_Get SingleRB receivied\n");
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
      LOG_A(ENB_SS_SYS_TASK, "PdcpCountGetReq_Type_UNBOUND_VALUE received\n");
      break;
    default:
      LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type (GET) Invalid \n");
    }
    LOG_A(ENB_SS_SYS_TASK," SS_REQ_PDCP_CNT(message_p).rb_id %d\n", SS_REQ_PDCP_CNT(get_p).rb_id);
    send_res = itti_send_msg_to_task(TASK_PDCP_ENB, 0, get_p);
    if (send_res < 0)
    {
      LOG_A(ENB_SS_SYS_TASK, "Error sending SS_REQ_PDCP_CNT to PDCP_ENB \n");
    }

    break;
  case PDCP_CountReq_Type_Set:
    LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type_Set receivied\n");
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
        LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type (SET) Invalid \n");
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
      LOG_A(ENB_SS_SYS_TASK, "Error sending SS_SET_PDCP_CNT to PDCP_ENB \n");
    }

    //Sending Confirm for Set PDCPCount
    enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_PdcpCount;
    enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
    bool resVal = true;
    struct PDCP_CountCnf_Type PdcpCount;
    PdcpCount.d = PDCP_CountCnf_Type_Set;
    PdcpCount.v.Set = true;
    send_sys_cnf(resType, resVal, cnfType, (void *)&PdcpCount);
    sys_confirm_done_indication();

    break;
  case PDCP_CountReq_Type_UNBOUND_VALUE:
    LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type UNBOUND receivied\n");
    break;
  default:
    LOG_A(ENB_SS_SYS_TASK, "Pdcp_CountReq_Type INVALID Type receivied\n");
  }
  return returnState;
}

/*
 * Function : sys_send_proxy
 * Description: Sends the messages from SYS to proxy
 */
static void sys_send_proxy(void *msg, int msgLen, struct TimingInfo_Type* at)
{
  LOG_A(ENB_SS_SYS_TASK, "In sys_send_proxy\n");
  uint32_t peerIpAddr;
  uint16_t peerPort = proxy_send_port;

  IPV4_STR_ADDR_TO_INT_NWBO(local_address, peerIpAddr, " BAD IP Address");

  LOG_A(ENB_SS_SYS_TASK, "******************* Sending CELL CONFIG length\n Buffer is :%d ", msgLen);
  int8_t *temp = msg;
  for(int i =0 ; i <msgLen;i++)
  {
    LOG_A(ENB_SS_SYS_TASK, "%x ", temp[i]);
  }

  LOG_A(ENB_SS_SYS_TASK, "\nCell Config End of Buffer\n ");

  /** Send to proxy */
  sys_send_udp_msg((uint8_t *)msg, msgLen, 0, peerIpAddr, peerPort, at);
  return;
}

/*
 * Function : sys_cell_attn_update
 * Description: Sends the attenuation updates received from TTCN to proxy
 */
static void sys_cell_attn_update(uint8_t cellId, uint8_t attnVal,int CellIndex,  struct TimingInfo_Type* at)
{
  LOG_A(ENB_SS_SYS_TASK, "In sys_cell_attn_update, cellIndex:%d \n",CellIndex);
  attenuationConfigReq_t *attnConf = NULL;
  uint32_t peerIpAddr;
  uint16_t peerPort = proxy_send_port;

  attnConf = (attenuationConfigReq_t *) calloc(1, sizeof(attenuationConfigReq_t));
  attnConf->header.preamble = 0xFEEDC0DE;
  attnConf->header.msg_id = SS_ATTN_LIST;
  attnConf->header.cell_id = SS_context.SSCell_list[CellIndex].PhysicalCellId;
  attnConf->header.cell_index = CellIndex;
  attnConf->attnVal = attnVal;
  SS_context.send_atten_cnf = false;
  IPV4_STR_ADDR_TO_INT_NWBO(local_address, peerIpAddr, " BAD IP Address");

  /** Send to proxy */
  sys_send_udp_msg((uint8_t *)attnConf, sizeof(attenuationConfigReq_t), 0, peerIpAddr, peerPort, at);
  LOG_A(ENB_SS_SYS_TASK, "Out sys_cell_attn_update\n");
  return;
}
/*
 * Function : sys_handle_cell_attn_req
 * Description: Handles the attenuation updates received from TTCN
 */
static void sys_handle_cell_attn_req(struct SYSTEM_CTRL_REQ *req)
{
  assert(req);
  struct CellAttenuationConfig_Type_CellAttenuationList_Type_Dynamic *CellAttenuationList = &(req->Request.v.CellAttenuationList);
  assert(CellAttenuationList);

  for(int i=0;i<CellAttenuationList->d;i++) {
    uint8_t cellId = (uint8_t)CellAttenuationList->v[i].CellId;
    uint8_t CellIndex = get_cell_index(cellId, SS_context.SSCell_list);
    uint8_t attnVal = 0; // default set it Off

    switch (CellAttenuationList->v[i].Attenuation.d)
    {
    case Attenuation_Type_Value:
      attnVal = CellAttenuationList->v[i].Attenuation.v.Value;
      LOG_A(ENB_SS_SYS_TASK, "CellAttenuationList for Cell_id %d value %d dBm received\n",
            cellId, attnVal);
      sys_cell_attn_update(cellId, attnVal,CellIndex, &req->Common.TimingInfo);
      break;
    case Attenuation_Type_Off:
      attnVal = 80; /* TODO: attnVal hardcoded currently but Need to handle proper Attenuation_Type_Off */
      LOG_A(ENB_SS_SYS_TASK, "CellAttenuationList turn off for Cell_id %d received with attnVal : %d\n",
            cellId,attnVal);
      sys_cell_attn_update(cellId, attnVal,CellIndex, &req->Common.TimingInfo);
      break;
    case Attenuation_Type_UNBOUND_VALUE:
      LOG_A(ENB_SS_SYS_TASK, "CellAttenuationList Attenuation_Type_UNBOUND_VALUE received\n");
      break;
    default:
      LOG_A(ENB_SS_SYS_TASK, "Invalid CellAttenuationList received\n");
    }
  }
  //sys_confirm_done_indication();
}
/*
 * Function : sys_handle_paging_req
 * Description: Handles the attenuation updates received from TTCN
 */

static void sys_handle_paging_req(struct PagingTrigger_Type *pagingRequest, ss_set_timinfo_t tinfo)
{
  LOG_A(ENB_SS_SYS_TASK, "Enter sys_handle_paging_req Paging_IND for processing\n");

	/** TODO: Considering only one cell for now */
	uint8_t cellId = 0; //(uint8_t)pagingRequ ->CellId;
	uint8_t cn_domain = 0;

	enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_Paging;
	enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
	bool resVal = true;
	static uint8_t oneTimeProcessingFlag = 0;
	MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0,SS_SS_PAGING_IND);
	if (message_p == NULL)
	{
		return;
	}

	SS_PAGING_IND(message_p).cell_index = cell_index;
	SS_PAGING_IND(message_p).sfn = tinfo.sfn;
	SS_PAGING_IND(message_p).sf = tinfo.sf;
	SS_PAGING_IND(message_p).paging_recordList = NULL;
	SS_PAGING_IND(message_p).systemInfoModification = false;
	SS_PAGING_IND(message_p).bSubframeOffsetListPresent = false;

	switch (pagingRequest->Paging.message.d)
	{
		case SQN_PCCH_MessageType_c1:
			if (pagingRequest->Paging.message.v.c1.d)
			{
				if (pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.d)
				{
					struct SQN_PagingRecord *p_sdl_msg = NULL;
					p_sdl_msg = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.v;
					/* id-CNDomain : convert cnDomain */
					uint8_t numPagingRecord = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.d;
					size_t pgSize = pagingRequest->Paging.message.v.c1.v.paging.pagingRecordList.v.d * sizeof(ss_paging_identity_t);
					SS_PAGING_IND(message_p).sfn =tinfo.sfn;
					SS_PAGING_IND(message_p).sf = tinfo.sf;
					SS_PAGING_IND(message_p).paging_recordList = CALLOC(1, pgSize);
					ss_paging_identity_t *p_record_msg = SS_PAGING_IND(message_p).paging_recordList;
					SS_PAGING_IND(message_p).num_paging_record = numPagingRecord;
					for (int count = 0; count < numPagingRecord; count++)
					{
						cn_domain = p_sdl_msg->cn_Domain;
						/* id-CNDomain : convert cnDomain */
						if (cn_domain == SQN_PagingRecord_cn_Domain_e_ps)
						{
							p_record_msg->cn_domain = CN_DOMAIN_PS;
						}
						else if (cn_domain == SQN_PagingRecord_cn_Domain_e_cs)
						{
							p_record_msg->cn_domain = CN_DOMAIN_CS;
						}

						switch (p_sdl_msg->ue_Identity.d)
						{
							case SQN_PagingUE_Identity_s_TMSI:
								p_record_msg->ue_paging_identity.presenceMask = UE_PAGING_IDENTITY_s_tmsi;
								int32_t stmsi_rx = bin_to_int(p_sdl_msg->ue_Identity.v.s_TMSI.m_TMSI, 32);

								p_record_msg->ue_paging_identity.choice.s_tmsi.m_tmsi = stmsi_rx ;
								p_record_msg->ue_paging_identity.choice.s_tmsi.mme_code =
									bin_to_int(p_sdl_msg->ue_Identity.v.s_TMSI.mmec,8);
								if (oneTimeProcessingFlag == 0)
								{
									SS_PAGING_IND(message_p).ue_index_value = paging_ue_index_g;
									paging_ue_index_g = ((paging_ue_index_g +4) % MAX_MOBILES_PER_ENB) ;
									oneTimeProcessingFlag = 1;
								}
								break;
							case SQN_PagingUE_Identity_imsi:
								p_record_msg->ue_paging_identity.presenceMask = UE_PAGING_IDENTITY_imsi;
                p_record_msg->ue_paging_identity.choice.imsi.length = p_sdl_msg->ue_Identity.v.imsi.d;
                memcpy(&(p_record_msg->ue_paging_identity.choice.imsi.buffer[0]),p_sdl_msg->ue_Identity.v.imsi.v, p_sdl_msg->ue_Identity.v.imsi.d);
                break;
							case SQN_PagingUE_Identity_ng_5G_S_TMSI_r15:
							case SQN_PagingUE_Identity_fullI_RNTI_r15:
							case SQN_PagingUE_Identity_UNBOUND_VALUE:
								LOG_A(ENB_SS_SYS_TASK, "Error Unhandled Paging request \n");
								break;
							default :
								LOG_A(ENB_SS_SYS_TASK, "Invalid Pging request received\n");

						}
						p_sdl_msg++;
						p_record_msg++;
					}
				}

				if (pagingRequest->Paging.message.v.c1.v.paging.systemInfoModification.d)
				{
					LOG_A(ENB_SS_SYS_TASK, "System Info Modification received in Paging request \n");
					if (SQN_Paging_systemInfoModification_e_true == pagingRequest->Paging.message.v.c1.v.paging.systemInfoModification.v)
					{
						SS_PAGING_IND(message_p).systemInfoModification = true;
					}
				}
			}
			if(pagingRequest->SubframeOffsetList.d)
			{
				LOG_A(ENB_SS_SYS_TASK, "Subframe Offset List present in Paging request \n");
				SS_PAGING_IND(message_p).bSubframeOffsetListPresent=true;
				SS_PAGING_IND(message_p).subframeOffsetList.num = 0;
				for (int i=0; i < pagingRequest->SubframeOffsetList.v.d; i++)
				{
					SS_PAGING_IND(message_p).subframeOffsetList.subframe_offset[i] = pagingRequest->SubframeOffsetList.v.v[i];
					SS_PAGING_IND(message_p).subframeOffsetList.num++;
				}
			}

			int send_res = itti_send_msg_to_task(TASK_RRC_ENB, 0, message_p);
			if (send_res < 0)
			{
				LOG_A(ENB_SS_SYS_TASK, "Error sending Paging to RRC_ENB");
			}
			oneTimeProcessingFlag = 0;
			LOG_A(ENB_SS_SYS_TASK, "Paging_IND for Cell_id %d  sent to RRC\n", cellId);
			break;
		case SQN_PCCH_MessageType_messageClassExtension:
			LOG_A(ENB_SS_SYS_TASK, "PCCH_MessageType_messageClassExtension for Cell_id %d received\n",
					cellId);
			break;
		case SQN_PCCH_MessageType_UNBOUND_VALUE:
			LOG_A(ENB_SS_SYS_TASK, "Invalid Pging request received Type_UNBOUND_VALUE received\n");
			break;
		default:
			LOG_A(ENB_SS_SYS_TASK, "Invalid Pging request received\n");
      break;
	}
	send_sys_cnf(resType, resVal, cnfType, NULL);
  sys_confirm_done_indication();
	LOG_A(ENB_SS_SYS_TASK, "Exit sys_handle_paging_req Paging_IND processing for Cell_id %d \n", cellId);
}


static void sys_handle_l1macind_ctrl(struct SYSTEM_CTRL_REQ *req)
{
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_L1MacIndCtrl;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  struct L1Mac_IndicationControl_Type *L1MacInd_Ctrl = &(req->Request.v.L1MacIndCtrl);
  assert(L1MacInd_Ctrl);
  LOG_A(ENB_SS_SYS_TASK,"l1macind ctrl, RachPreamble=%d UL_HARQ=%d HarqError=%d\n",L1MacInd_Ctrl->RachPreamble.d, L1MacInd_Ctrl->UL_HARQ.d, L1MacInd_Ctrl->HarqError.d);
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0, SS_L1MACIND_CTRL);
  if (message_p)
  {
    SS_L1MACIND_CTRL(message_p).cell_index = cell_index;
    if(L1MacInd_Ctrl->RachPreamble.d)
    {
      LOG_A(ENB_SS_SYS_TASK, "l1macind ctrl RachPreamble type %d received from TTCN\n", L1MacInd_Ctrl->RachPreamble.v);
      if(IndicationAndControlMode_enable == L1MacInd_Ctrl->RachPreamble.v)
      {
        SS_L1MACIND_CTRL(message_p).rachpreamble_enable = true;
      } else {
        SS_L1MACIND_CTRL(message_p).rachpreamble_enable = false;
      }
      SS_L1MACIND_CTRL(message_p).bitmask |= RACH_PREAMBLE_PRESENT;
    }
    SS_L1MACIND_CTRL(message_p).UL_HARQ_Ctrl = IndCtrlMode_NOT_PRESENT;
    if(L1MacInd_Ctrl->UL_HARQ.d)
    {
      LOG_A(ENB_SS_SYS_TASK, "l1macind ctrl UL_HARQ type %d received from TTCN\n", L1MacInd_Ctrl->UL_HARQ.v);
      if (IndicationAndControlMode_enable == L1MacInd_Ctrl->UL_HARQ.v)
      {
        SS_L1MACIND_CTRL(message_p).UL_HARQ_Ctrl = IndCtrlMode_ENABLE;
      }
      else if (IndicationAndControlMode_disable == L1MacInd_Ctrl->UL_HARQ.v)
      {
        SS_L1MACIND_CTRL(message_p).UL_HARQ_Ctrl = IndCtrlMode_DISABLE;
      }
    }
    SS_L1MACIND_CTRL(message_p).HarqError_Ctrl = IndCtrlMode_NOT_PRESENT;
    if(L1MacInd_Ctrl->HarqError.d)
    {
      LOG_A(ENB_SS_SYS_TASK, "l1macind ctrl HarqError type %d received from TTCN\n", L1MacInd_Ctrl->HarqError.v);
      if (IndicationAndControlMode_enable == L1MacInd_Ctrl->HarqError.v)
      {
        SS_L1MACIND_CTRL(message_p).HarqError_Ctrl = IndCtrlMode_ENABLE;
      }
      else if (IndicationAndControlMode_disable == L1MacInd_Ctrl->HarqError.v)
      {
        SS_L1MACIND_CTRL(message_p).HarqError_Ctrl = IndCtrlMode_DISABLE;
      }
    }
    SS_L1MACIND_CTRL(message_p).SchedReq_Ctrl = IndCtrlMode_NOT_PRESENT;
    if(L1MacInd_Ctrl->SchedReq.d)
    {
      LOG_A(ENB_SS_SYS_TASK, "l1macind ctrl SchedReq type %d received from TTCN\n", L1MacInd_Ctrl->SchedReq.v);
      if (IndicationAndControlMode_enable == L1MacInd_Ctrl->SchedReq.v)
      {
        SS_L1MACIND_CTRL(message_p).SchedReq_Ctrl = IndCtrlMode_ENABLE;
      }
      else if (IndicationAndControlMode_disable == L1MacInd_Ctrl->SchedReq.v)
      {
        SS_L1MACIND_CTRL(message_p).SchedReq_Ctrl = IndCtrlMode_DISABLE;
      }
    }

    vt_add_sf(&req->Common.TimingInfo, 4);
    if (!vt_timer_push_msg(&req->Common.TimingInfo, TASK_MAC_ENB,0, message_p))
    {
      itti_send_msg_to_task(TASK_MAC_ENB, 0, message_p);
    }
  }
  send_sys_cnf(resType, resVal, cnfType, NULL);
  sys_confirm_done_indication();
}

/*
 * Function : sys_handle_ue_cat_info_req
 * Description: Funtion handler of SYS_PORT. Handles the UE
 * Category Info command received from TTCN via the PORTMAN.
 * In :
 * req  - UE Category Info Request received from the TTCN via PORTMAN
 * Out:
 * newState: No impact on state machine.
 *
 */
static void sys_handle_ue_cat_info_req(struct UE_CategoryInfo_Type *UE_Cat_Info)
{
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_UE_Cat_Info;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0, RRC_UE_CAT_INFO);
  if (message_p)
  {
    LOG_A(ENB_SS_SYS_TASK,"UE Category Info received \n");
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
    int send_res = itti_send_msg_to_task(TASK_RRC_ENB, 0, message_p);
    if (send_res < 0)
    {
      LOG_A(ENB_SS_SYS_TASK, "Error sending RRC_UE_CAT_INFO to TASK_RRC_ENB");
    }
  }
  send_sys_cnf(resType, resVal, cnfType, NULL);
  sys_confirm_done_indication();
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
static void sys_handle_as_security_req(struct SYSTEM_CTRL_REQ *req)
{
  enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_AS_Security;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  bool intKey = false;

  assert(req);
  struct AS_Security_Type *ASSecurity = &(req->Request.v.AS_Security);
  assert(ASSecurity);
  MessageDef *msg_p = itti_alloc_new_message(TASK_SYS, 0, RRC_AS_SECURITY_CONFIG_REQ);
  if(msg_p)
  {
    LOG_A(ENB_SS_SYS_TASK,"AS Security Request Received\n");
    RRC_AS_SECURITY_CONFIG_REQ(msg_p).rnti = SS_context.SSCell_list[cell_index].ss_rnti_g;
    if(ASSecurity->d == AS_Security_Type_StartRestart)
    {
      if(ASSecurity->v.StartRestart.Integrity.d == true)
      {
        intKey = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).isIntegrityInfoPresent = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.integrity_algorithm = ASSecurity->v.StartRestart.Integrity.v.Algorithm;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint = CALLOC(1,32);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint,0,32);
        bitStrint_to_byteArray(ASSecurity->v.StartRestart.Integrity.v.KRRCint,256,RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint,intKey);
        for(int j=0;j<32;j++) {
          LOG_A(ENB_SS_SYS_TASK,"KRRCint in SS: %02x \n",RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.kRRCint[j]);
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
                RRC_AS_SECURITY_CONFIG_REQ(msg_p).Integrity.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Integrity.v.ActTimeList.v.v[i].RadioBearerId.v.Drb + 3;
                break;
              case RadioBearerId_Type_UNBOUND_VALUE:
                break;
              default:
              LOG_A(ENB_SS_SYS_TASK, "AS Security Act time list is Invalid \n");
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
        intKey = false;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).isCipheringInfoPresent = true;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ciphering_algorithm = ASSecurity->v.StartRestart.Ciphering.v.Algorithm;
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc,0,16);
        bitStrint_to_byteArray(ASSecurity->v.StartRestart.Ciphering.v.KRRCenc,128,RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc,intKey);
        for(int i=0;i<16;i++) {
          LOG_A(ENB_SS_SYS_TASK,"kRRCenc in SS: %02x \n",RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kRRCenc[i]);
        }
        RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc = CALLOC(1,16);
        memset(RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc,0,16);
        bitStrint_to_byteArray(ASSecurity->v.StartRestart.Ciphering.v.KUPenc,128,RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc,intKey);
        for(int k=0;k<16;k++) {
          LOG_A(ENB_SS_SYS_TASK,"kUPenc in SS: %02x \n",RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.kUPenc[k]);
        }
        LOG_A(ENB_SS_SYS_TASK, "Ciphering ActTimeList.d = %lu\n", ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.d);
        for(int i=0;i < ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.d; i++)
        {
          switch(ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].RadioBearerId.d)
          {
            case RadioBearerId_Type_Srb:
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].RadioBearerId.v.Srb;
              break;
            case RadioBearerId_Type_Drb:
              RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].rb_id = ASSecurity->v.StartRestart.Ciphering.v.ActTimeList.v[i].RadioBearerId.v.Drb + 3;
              break;
            case RadioBearerId_Type_UNBOUND_VALUE:
              break;
            default:
            LOG_A(ENB_SS_SYS_TASK, "AS Security Act time list is Invalid \n");
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
          LOG_A(ENB_SS_SYS_TASK, "Ciphering ActTimeList i=%d rb_id=%d\n", i, RRC_AS_SECURITY_CONFIG_REQ(msg_p).Ciphering.ActTimeList.SecurityActTime[i].rb_id);
        }
      }
      if (!vt_timer_push_msg(&req->Common.TimingInfo, TASK_RRC_ENB,0, msg_p))
      {
        itti_send_msg_to_task(TASK_RRC_ENB, 0, msg_p);
      }
    }
    else if (ASSecurity->d == AS_Security_Type_Release)
    {
      LOG_A(ENB_SS_SYS_TASK, "AS_Security_Type_Release received\n");
      if(req->Common.ControlInfo.CnfFlag == true)
      {
        send_sys_cnf(resType, resVal, cnfType, NULL);
        sys_confirm_done_indication();
      } 
    }
  }
  set_syscnf(reqCnfFlag_g, resType, resVal, cnfType);
}

/*
 * Function : ss_task_sys_handle_req
 * Description: The state handler of SYS_PORT. Handles the SYS_PORT
 * configuration command received from TTCN via the PORTMAN.
 * Applies the configuration to different layers of SS.
 * Sends the CNF message for the required requests to PORTMAN
 * In :
 * req  - Request received from the TTCN via PORTMAN
 * tinfo - Timing info to be sent to PORT for enquire timing
 *
 */
static void ss_task_sys_handle_req(struct SYSTEM_CTRL_REQ *req, ss_set_timinfo_t *tinfo)
{
  if(req->Common.CellId){
    cell_index = get_cell_index(req->Common.CellId, SS_context.SSCell_list);
    SS_context.SSCell_list[cell_index].eutra_cellId = req->Common.CellId;
    LOG_A(ENB_SS_SYS_TASK,"cell_index: %d eutra_cellId: %d \n",cell_index,SS_context.SSCell_list[cell_index].eutra_cellId);
   printf("cell_index: %d eutra_cellId: %d \n",cell_index,SS_context.SSCell_list[cell_index].eutra_cellId);

  }
  int enterState = SS_context.SSCell_list[cell_index].State;
  int exitState = SS_context.SSCell_list[cell_index].State;
  LOG_A(ENB_SS_SYS_TASK, "Current SS_STATE %d received SystemRequest_Type %d eutra_cellId %d cnf_flag %d\n",
        SS_context.SSCell_list[cell_index].State, req->Request.d, SS_context.SSCell_list[cell_index].eutra_cellId, req->Common.ControlInfo.CnfFlag);

  memset(&SS_context.sys_cnf, 0, sizeof(SS_context.sys_cnf));
  switch (SS_context.SSCell_list[cell_index].State)
  {
    case SS_STATE_NOT_CONFIGURED:
      if (req->Request.d == SystemRequest_Type_Cell)
      {
        LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_Cell received\n");
        SS_context.SSCell_list[cell_index].PhysicalCellId = req->Request.v.Cell.v.AddOrReconfigure.Basic.v.StaticCellInfo.v.Common.PhysicalCellId;
        exitState = sys_handle_cell_config_req(req);
        LOG_A(ENB_SS_SYS_TASK,"SS_STATE_NOT_CONFIGURED: PhysicalCellId is %d in SS_context \n",SS_context.SSCell_list[cell_index].PhysicalCellId);
        SS_context.SSCell_list[cell_index].State = exitState;
        if(RC.ss.State <= SS_STATE_CELL_CONFIGURED)
          RC.ss.State = exitState;
      }
      else
      {
        LOG_A(ENB_SS_SYS_TASK, "Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
            SS_context.SSCell_list[cell_index].State, req->Request.d);
      }
      break;
    case SS_STATE_CELL_CONFIGURED:
      if (req->Request.d == SystemRequest_Type_RadioBearerList)
      {
        LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_RadioBearerList received\n");
        exitState = sys_handle_radiobearer_list(req);
        SS_context.SSCell_list[cell_index].State = exitState;
        if(RC.ss.State <= SS_STATE_CELL_CONFIGURED)
          RC.ss.State = exitState;
      }
      else if (req->Request.d == SystemRequest_Type_L1MacIndCtrl)
      {
        LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type L1MacIndCtrl received\n");
        sys_handle_l1macind_ctrl(req);
      }
      else
      {
        LOG_A(ENB_SS_SYS_TASK, "Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
            SS_context.SSCell_list[cell_index].State, req->Request.d);
      }
      break;
    case SS_STATE_CELL_BROADCASTING:
      break;

    case SS_STATE_CELL_ACTIVE:
      switch (req->Request.d)
      {
        case SystemRequest_Type_Cell:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_Cell received\n");
          exitState = sys_handle_cell_config_req(req);
          LOG_A(ENB_SS_SYS_TASK,"SS_STATE_CELL_ACTIVE: PhysicalCellId is %d in SS_context \n",SS_context.SSCell_list[cell_index].PhysicalCellId);
          SS_context.SSCell_list[cell_index].State = exitState;
          if(RC.ss.State <= SS_STATE_CELL_ACTIVE)
            RC.ss.State = exitState;
          break;
        case SystemRequest_Type_RadioBearerList:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_RadioBearerList received in SS_STATE_CELL_ACTIVE state\n");
          exitState = sys_handle_radiobearer_list(req);
          SS_context.SSCell_list[cell_index].State = exitState;
          if(RC.ss.State <= SS_STATE_CELL_ACTIVE)
            RC.ss.State = exitState;
          break;
        case SystemRequest_Type_CellAttenuationList:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_CellAttenuationList received\n");
          sys_handle_cell_attn_req(req);
          break;
        case SystemRequest_Type_PdcpCount:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type PDCP_Count received\n");
          sys_handle_pdcp_count_req(&(req->Request.v.PdcpCount));
          break;
        case SystemRequest_Type_PdcpHandoverControl:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type PdcpHandoverControl received\n");
          enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_PdcpHandoverControl;
          enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
          bool resVal = true;
          send_sys_cnf(resType, resVal, cnfType, NULL);
          sys_confirm_done_indication();
          break;
        case SystemRequest_Type_AS_Security:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_AS_Security received\n");
          sys_handle_as_security_req(req);
          break;

        case SystemRequest_Type_UE_Cat_Info:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type UE_Cat received\n");
          sys_handle_ue_cat_info_req(&(req->Request.v.UE_Cat_Info));
          break;

        case SystemRequest_Type_Paging:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type Paging received\n");
          ss_set_timinfo_t pg_timinfo ;
          pg_timinfo.hsfn = req->Common.TimingInfo.v.SubFrame.HSFN.v.Number;
          pg_timinfo.sfn = req->Common.TimingInfo.v.SubFrame.SFN.v.Number;
          pg_timinfo.sf = req->Common.TimingInfo.v.SubFrame.Subframe.v.Number;
          sys_handle_paging_req(&(req->Request.v.Paging), pg_timinfo);
          break;

        case SystemRequest_Type_L1MacIndCtrl:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type L1MacIndCtrl received\n");
          sys_handle_l1macind_ctrl(req);
          break;

        case SystemRequest_Type_PdcchOrder:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_PdcchOrder received\n");
          sys_handle_pdcch_order(&req->Request.v.PdcchOrder);
          break;
        case SystemRequest_Type_UNBOUND_VALUE:
          LOG_A(ENB_SS_SYS_TASK, "SystemRequest_Type_UNBOUND_VALUE received\n");
          break;

        default:
          LOG_A(ENB_SS_SYS_TASK, "Error ! Invalid SystemRequest_Type received\n");
      }
      break;

    case SS_STATE_AS_SECURITY_ACTIVE:
      if (req->Request.d == SystemRequest_Type_RadioBearerList)
      {
        LOG_E(ENB_SS_SYS_TASK, "ERROR!!! TODO SystemRequest_Type_RadioBearerList received\n");
        //sys_handle_cell_config_req(&(req->Request.v.Cell));
        //RC.ss.State = SS_STATE_AS_RBS_ACTIVE;
      }
      else
      {
        LOG_E(ENB_SS_SYS_TASK, "Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
            SS_context.SSCell_list[cell_index].State, req->Request.d);
      }
      break;

    case SS_STATE_AS_RBS_ACTIVE:
      LOG_E(ENB_SS_SYS_TASK, "Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
          SS_context.SSCell_list[cell_index].State, req->Request.d);
      break;

    default:
      LOG_E(ENB_SS_SYS_TASK, "Error ! SS_STATE %d  Invalid SystemRequest_Type %d received\n",
          SS_context.SSCell_list[cell_index].State, req->Request.d);
      break;
  }
  LOG_A(ENB_SS_SYS_TASK, "Current SS_STATE %d New SS_STATE %d received SystemRequest_Type %d\n",
        enterState, SS_context.SSCell_list[cell_index].State, req->Request.d);
}
/*
 * Function : valid_sys_msg
 * Description:  Validates the SYS_PORT configuration command received
 * if the command received is not supported and needs CNF, sends the dummy
 * confirmation to PORTMAN , forwared towards TTCN.
 * If the command received is supported then proceeds with furhter porcessing
 *
 * In :
 * req  - Request received from the TTCN via PORTMAN
 *
 * Out:
 * TRUE - If recevied command is supported by SYS State handler
 * FALSE -If received command is not supported by SYS Handler.
 *
 */
bool valid_sys_msg(struct SYSTEM_CTRL_REQ *req)
{
  bool valid = false;
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  bool sendDummyCnf = true;
  enum SystemConfirm_Type_Sel cnfType = 0;

  // if (req->Common.ControlInfo.CnfFlag == FALSE)
  // {
  //   return FALSE;
  // }

  LOG_A(ENB_SS_SYS_TASK, "received req : %d for cell %d SS_context.SSCell_list[cell_index].State %d \n",
        req->Request.d, req->Common.CellId, SS_context.SSCell_list[cell_index].State);
  switch (req->Request.d)
  {
    case SystemRequest_Type_Cell:
      valid = true;
      sendDummyCnf = false;
      break;

    case SystemRequest_Type_EnquireTiming:
      sendDummyCnf = false;
      break;
    case SystemRequest_Type_CellAttenuationList:
      if (SS_context.SSCell_list[cell_index].State == SS_STATE_CELL_ACTIVE)
      {
        valid = true;
        sendDummyCnf = false;
      }
      break;
    case SystemRequest_Type_RadioBearerList:
      cnfType = SystemConfirm_Type_RadioBearerList;
      valid = true;
      sendDummyCnf = false;
      break;
    case SystemRequest_Type_AS_Security:
      cnfType = SystemConfirm_Type_AS_Security;
      valid = true;
      sendDummyCnf = false;
      break;
    case SystemRequest_Type_PdcpCount:
      if (SS_context.SSCell_list[cell_index].State == SS_STATE_CELL_ACTIVE)
      {
        valid = true;
        sendDummyCnf = false;
      }
      //cnfType = SystemConfirm_Type_PdcpCount;
      break;

  case SystemRequest_Type_UE_Cat_Info:
    cnfType = SystemConfirm_Type_UE_Cat_Info;
    break;
   case SystemRequest_Type_Paging:
    valid = true;
    sendDummyCnf = false;
    cnfType = SystemConfirm_Type_Paging;
    break;
   case SystemRequest_Type_L1MacIndCtrl:
    valid = true;
    sendDummyCnf = false;
    cnfType = SystemConfirm_Type_L1MacIndCtrl;
    break;
   case SystemRequest_Type_PdcchOrder:
    valid = true;
    sendDummyCnf = false;
    cnfType = SystemConfirm_Type_PdcchOrder;
    break;
   case SystemRequest_Type_PdcpHandoverControl:
    valid = true;
    sendDummyCnf = false;
    cnfType = SystemConfirm_Type_PdcpHandoverControl;
    break;
  default:
    valid = false;
    sendDummyCnf = false;
  }

  reqCnfFlag_g = req->Common.ControlInfo.CnfFlag;
  if (sendDummyCnf)
  {
    send_sys_cnf(resType, resVal, cnfType, NULL);
    LOG_A(ENB_SS_SYS_TASK, "Sending Dummy OK Req %d cnTfype %d ResType %d ResValue %d\n",
          req->Request.d, cnfType, resType, resVal);
    sys_confirm_done_indication();
  }
  return valid;
}

/*
 * Function : ss_eNB_sys_process_itti_msg
 * Description: Funtion handler of SYS_PORT. Handles the ITTI
 * message received from the TTCN on SYS Port
 * In :
 * req  - ITTI message received from the TTCN via PORTMAN
 * Out:
 * newState: No impact on state machine.
 *
 */
void *ss_eNB_sys_process_itti_msg(void *notUsed)
{
  MessageDef *received_msg = NULL;
  int result;
  static ss_set_timinfo_t tinfo = {.hsfn=0xFFFF, .sfn = 0xFFFF, .sf = 0xFF};
  SS_context.hsfn = tinfo.hsfn;
  SS_context.sfn = tinfo.sfn;
  SS_context.sf  = tinfo.sf;

  itti_receive_msg(TASK_SYS, &received_msg);

  /* Check if there is a packet to handle */
  if (received_msg != NULL)
  {
    switch (ITTI_MSG_ID(received_msg))
    {
      case RRC_CONFIGURATION_CNF:
      case RRC_RBLIST_CFG_CNF:
      case RRC_AS_SECURITY_CONFIG_CNF:
        {
          LOG_I(ENB_SS_SYS_TASK, "Received msg:%s\n", ITTI_MSG_NAME(received_msg));
          if (RRC_CONFIGURATION_CNF(received_msg).status == 1 && SS_context.sys_cnf.cnfFlag == 1)
          {
            LOG_A(ENB_SS_SYS_TASK, "Signalling main thread for cell config done indication\n");
            send_sys_cnf(SS_context.sys_cnf.resType,
                SS_context.sys_cnf.resVal,
                SS_context.sys_cnf.cnfType,
                (void *)SS_context.sys_cnf.msg_buffer);
            sys_confirm_done_indication();
          }
          break;
        }
      case SS_UPD_TIM_INFO:
        {
          /*WA: calculate hsfn here */
          if(tinfo.hsfn == 0xFFFF){
            tinfo.hsfn = 0;
          } else if(tinfo.sfn == 1023 && SS_UPD_TIM_INFO(received_msg).sfn == 0){
            tinfo.hsfn++;
            if(tinfo.hsfn == 1024){
              tinfo.hsfn = 0;
            }
          }
          tinfo.sf = SS_UPD_TIM_INFO(received_msg).sf;
          tinfo.sfn = SS_UPD_TIM_INFO(received_msg).sfn;

          SS_context.sfn = tinfo.sfn;
          SS_context.sf  = tinfo.sf;
          SS_context.hsfn  = tinfo.hsfn;

          g_log->sfn = tinfo.sfn;
          g_log->sf = (uint32_t)tinfo.sf;
          if (g_log->sfn % 64 == 0 && g_log->sf == 0) {
            LOG_I(ENB_SS_SYS_TASK, "[SYS] received SS_UPD_TIM_INFO HSFN:%d SFN: %d SF: %d\n", tinfo.hsfn,tinfo.sfn, tinfo.sf);
          }
        }
        break;

      case SS_GET_TIM_INFO:
        {
          LOG_D(ENB_SS_SYS_TASK, "received GET_TIM_INFO SFN: %d SF: %d\n", tinfo.sfn, tinfo.sf);
          ss_task_sys_handle_timing_info(&tinfo);
        }
        break;

      case SS_SYS_PORT_MSG_IND:
        {

          if (valid_sys_msg(SS_SYS_PORT_MSG_IND(received_msg).req))
          {
            ss_task_sys_handle_req(SS_SYS_PORT_MSG_IND(received_msg).req, &tinfo);
          }
          else
          {
            LOG_A(ENB_SS_SYS_TASK, "Not hanled SYS_PORT message received \n");
          }

          if (SS_SYS_PORT_MSG_IND(received_msg).req->Common.ControlInfo.CnfFlag == false)
            sys_confirm_done_indication();

          if (SS_SYS_PORT_MSG_IND(received_msg).req)
            free(SS_SYS_PORT_MSG_IND(received_msg).req);

          LOG_A(ENB_SS_SYS_TASK, "Signalling main thread for cell config done indication\n");

        }
        break;

      case SS_VNG_PROXY_REQ:
      {
        LOG_A(ENB_SS_SYS_TASK, "received %s from %s \n", ITTI_MSG_NAME(received_msg), ITTI_MSG_ORIGIN_NAME(received_msg));

        VngCmdReq_t *req      = (VngCmdReq_t *)malloc(sizeof(VngCmdReq_t));
        req->header.preamble  = 0xFEEDC0DE;
        req->header.msg_id    = SS_VNG_CMD_REQ;
        req->header.length    = sizeof(proxy_ss_header_t);
        req->header.cell_id   = SS_VNG_PROXY_REQ(received_msg).cell_id;
        req->header.cell_index = get_cell_index_pci(req->header.cell_id , SS_context.SSCell_list);
        printf("VNG send to proxy cell_index %d\n",req->header.cell_index);
        req->bw               = SS_VNG_PROXY_REQ(received_msg).bw;
        req->cmd              = SS_VNG_PROXY_REQ(received_msg).cmd;
        req->NocLevel         = SS_VNG_PROXY_REQ(received_msg).Noc_level;

        LOG_A(ENB_SS_SYS_TASK,"VNG Command for cell_id: %d CMD %d ",
            req->header.cell_id, req->cmd);
        sys_send_proxy((void *)req, sizeof(VngCmdReq_t), NULL);
      }
      break;

    case SS_GET_PDCP_CNT: /** FIXME */
    {
      LOG_A(ENB_SS_SYS_TASK, "received SS_GET_PDCP_CNT Count from PDCP size %d\n", SS_GET_PDCP_CNT(received_msg).size);
      enum SystemConfirm_Type_Sel cnfType = SystemConfirm_Type_PdcpCount;
      enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
      bool resVal = true;
      struct PDCP_CountCnf_Type PdcpCount;


      PdcpCount.d = PDCP_CountCnf_Type_Get;
      PdcpCount.v.Get.d = SS_GET_PDCP_CNT(received_msg).size;
      const size_t size = sizeof(struct PdcpCountInfo_Type) * PdcpCount.v.Get.d;
      PdcpCount.v.Get.v =(struct PdcpCountInfo_Type *)acpMalloc(size);
      for (int i = 0; i < PdcpCount.v.Get.d; i++)
      {
        if (SS_GET_PDCP_CNT(received_msg).rb_info[i].is_srb == true)
        {
          PdcpCount.v.Get.v[i].RadioBearerId.d = RadioBearerId_Type_Srb;
          PdcpCount.v.Get.v[i].RadioBearerId.v.Srb = SS_GET_PDCP_CNT(received_msg).rb_info[i].rb_id;
        }
        else
        {
          PdcpCount.v.Get.v[i].RadioBearerId.d = RadioBearerId_Type_Drb;
          PdcpCount.v.Get.v[i].RadioBearerId.v.Drb = SS_GET_PDCP_CNT(received_msg).rb_info[i].rb_id - 3;
        }
        PdcpCount.v.Get.v[i].UL.d = true;
        PdcpCount.v.Get.v[i].DL.d = true;

        PdcpCount.v.Get.v[i].UL.v.Format = SS_GET_PDCP_CNT(received_msg).rb_info[i].ul_format;
        PdcpCount.v.Get.v[i].DL.v.Format = SS_GET_PDCP_CNT(received_msg).rb_info[i].dl_format;

        int_to_bin(SS_GET_PDCP_CNT(received_msg).rb_info[i].ul_count, 32, PdcpCount.v.Get.v[i].UL.v.Value);
        int_to_bin(SS_GET_PDCP_CNT(received_msg).rb_info[i].dl_count, 32, PdcpCount.v.Get.v[i].DL.v.Value);
      }

      send_sys_cnf(resType, resVal, cnfType, (void *)&PdcpCount);
      sys_confirm_done_indication();
    }
    break;

    case UDP_DATA_IND:
    {
      proxy_ss_header_t hdr;
      attenuationConfigCnf_t attnCnf;
      VngCmdResp_t VngResp;
      LOG_A(ENB_SS_SYS_TASK, "received UDP_DATA_IND \n");
      enum SystemConfirm_Type_Sel cnfType;
      enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
      bool resVal = true;

      //if (attnCnf.header.preamble != 0xF00DC0DE ) break; /** TODO Log ! */
      memcpy(&hdr, (SS_SYS_PROXY_MSG_CNF(received_msg).buffer), sizeof(proxy_ss_header_t));

      switch (hdr.msg_id)
      {
        case SS_ATTN_LIST_CNF:
          cnfType = SystemConfirm_Type_CellAttenuationList;
          memcpy(&attnCnf, (SS_SYS_PROXY_MSG_CNF(received_msg).buffer), sizeof(attenuationConfigCnf_t));
          if(false == SS_context.send_atten_cnf) {
            LOG_A(ENB_SS_SYS_TASK, "received Cell_Attenuation_Cnf from Proxy for cell : %d \n", attnCnf.header.cell_id);
            SS_context.send_atten_cnf = true;
            send_sys_cnf(resType, resVal, cnfType, NULL);
            sys_confirm_done_indication();
          }
          break;

        case SS_VNG_CMD_RESP:
          memcpy(&VngResp, (SS_SYS_PROXY_MSG_CNF(received_msg).buffer), sizeof(VngCmdResp_t));

          MessageDef *vng_resp_p = itti_alloc_new_message(TASK_SYS, 0, SS_VNG_PROXY_RESP);
          assert(vng_resp_p);

          SS_VNG_PROXY_RESP(vng_resp_p).cell_id = VngResp.header.cell_id;
          SS_VNG_PROXY_RESP(vng_resp_p).sfn_sf  = (tinfo.sfn << 4 | tinfo.sf);
          SS_VNG_PROXY_RESP(vng_resp_p).status  = VngResp.status;

          LOG_A(ENB_SS_SYS_TASK, "Sending CMD_RESP for CNF @ sfn: %d sf: %d\n", tinfo.sfn, tinfo.sf);

          int res = itti_send_msg_to_task(TASK_VNG, 0, vng_resp_p);
          if (res < 0)
          {
            LOG_A(ENB_SS_SYS_TASK, "[SS-SYS] Error in itti_send_msg_to_task\n");
          }
          else
          {
            LOG_A(ENB_SS_SYS_TASK, "[SS-SYS] Send ITTI message to %s\n", ITTI_MSG_DESTINATION_NAME(vng_resp_p));
          }
        default:
          break;
      }
      break;
    }
    case TERMINATE_MESSAGE:
    {
      itti_exit_task();
      break;
    }
    default:
      LOG_A(ENB_SS_SYS_TASK, "Received unhandled message %d:%s\n",
            ITTI_MSG_ID(received_msg), ITTI_MSG_NAME(received_msg));
      break;
    }
    result = itti_free(ITTI_MSG_ORIGIN_ID(received_msg), received_msg);
    AssertFatal(result == EXIT_SUCCESS, "Failed to free memory (%d)!\n", result);
    received_msg = NULL;
  }
  return NULL;
}

/*
 * Function : ss_eNB_sys_task
 * Description:  The SYS_TASK main function handler. Initilizes the UDP
 * socket towards the Proxy for the configuration updates. Initilizes
 * the SYS_TASK state machine Init_State. Invoke the itti message
 * handler for the SYY_PORT.
 */
void *ss_eNB_sys_task(void *arg)
{
  udpSockReq_t req;
  req.address = local_address;
  req.port = proxy_recv_port;
  sys_send_init_udp(&req);
  sleep(5);

  // Set the state to NOT_CONFIGURED for Cell Config processing mode
  if (RC.ss.mode == SS_SOFTMODEM)
  {
    init_ss_context(SS_context.SSCell_list);
    //SS_context.SSCell_list[cell_index].State = SS_STATE_NOT_CONFIGURED;
  }
  // Set the state to CELL_ACTIVE for SRB processing mode
  else if (RC.ss.mode == SS_HWTMODEM)
  {
    SS_context.SSCell_list[cell_index].State = SS_STATE_CELL_ACTIVE;
  }
  while (1)
  {
    (void)ss_eNB_sys_process_itti_msg(NULL);
  }

  return NULL;
}


void sys_handle_pdcch_order(struct RA_PDCCH_Order_Type *pdcchOrder)
{
  MessageDef *message_p = itti_alloc_new_message(TASK_SYS, 0, SS_L1MACIND_CTRL);
  enum ConfirmationResult_Type_Sel resType = ConfirmationResult_Type_Success;
  bool resVal = true;
  if (message_p)
  {
    LOG_A(ENB_SS_SYS_TASK,"pdcchOrder: preambleIndex%d prachMaskIndex:%d\n",pdcchOrder->PreambleIndex, pdcchOrder->PrachMaskIndex);
    SS_L1MACIND_CTRL(message_p).pdcchOrder.preambleIndex = pdcchOrder->PreambleIndex;
    SS_L1MACIND_CTRL(message_p).pdcchOrder.prachMaskIndex = pdcchOrder->PrachMaskIndex;
    SS_L1MACIND_CTRL(message_p).bitmask |= PDCCH_ORDER_PRESENT;
  }
  int send_res = itti_send_msg_to_task(TASK_MAC_ENB, 0, message_p);
  if (send_res < 0)
  {
    LOG_A(ENB_SS_SYS_TASK, "Error sending SS_L1MACIND_CTRL with PdcchOrder to MAC");
  }
  else
  {
    send_sys_cnf(resType, resVal, SystemConfirm_Type_PdcchOrder, NULL);
    sys_confirm_done_indication();
  }
}
