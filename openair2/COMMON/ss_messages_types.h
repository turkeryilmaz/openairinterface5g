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

#ifndef SS_MESSAGES_TYPES_H_
#define SS_MESSAGES_TYPES_H_

#define SS_SYS_PORT_MSG_IND(mSGpTR)            (mSGpTR)->ittiMsg.ss_sys_port_msg_ind
#define SS_SYS_PORT_MSG_CNF(mSGpTR)            (mSGpTR)->ittiMsg.ss_sys_port_msg_cnf

#define SS_NR_SYS_PORT_MSG_IND(mSGpTR)         (mSGpTR)->ittiMsg.ss_nr_sys_port_msg_ind
#define SS_NR_SYS_PORT_MSG_CNF(mSGpTR)         (mSGpTR)->ittiMsg.ss_nr_sys_port_msg_cnf

#define SS_GET_TIM_INFO(mSGpTR)                (mSGpTR)->ittiMsg.ss_get_timinfo
#define SS_SET_TIM_INFO(mSGpTR)                (mSGpTR)->ittiMsg.ss_set_timinfo
#define SS_NRSET_TIM_INFO(mSGpTR)              (mSGpTR)->ittiMsg.ss_nrset_timinfo
#define SS_UPD_TIM_INFO(mSGpTR)                (mSGpTR)->ittiMsg.ss_upd_timinfo
#define SS_NRUPD_TIM_INFO(mSGpTR)              (mSGpTR)->ittiMsg.ss_nrupd_timinfo
#define SS_CELL_ATTN_LIST_IND(mSGpTR)          (mSGpTR)->ittiMsg.ss_cell_attn_list_ind
#define SS_CELL_ATTN_LIST_CNF(mSGpTR)          (mSGpTR)->ittiMsg.ss_cell_attn_list_cnf

/** PDCP Count */
#define SS_REQ_PDCP_CNT(mSGpTR)                (mSGpTR)->ittiMsg.ss_req_pdcp_cnt
#define SS_GET_PDCP_CNT(mSGpTR)                (mSGpTR)->ittiMsg.ss_get_pdcp_cnt
#define SS_SET_PDCP_CNT(mSGpTR)                (mSGpTR)->ittiMsg.ss_set_pdcp_cnt

#define SS_RRC_PDU_REQ(mSGpTR)                (mSGpTR)->ittiMsg.ss_rrc_pdu_req
#define SS_RRC_PDU_IND(mSGpTR)                (mSGpTR)->ittiMsg.ss_rrc_pdu_ind
#define SS_NRRRC_PDU_REQ(mSGpTR)              (mSGpTR)->ittiMsg.ss_nrrrc_pdu_req
#define SS_NRRRC_PDU_IND(mSGpTR)              (mSGpTR)->ittiMsg.ss_nrrrc_pdu_ind
#define SS_SYS_PROXY_MSG_CNF(mSGpTR)          (mSGpTR)->ittiMsg.udp_data_ind
#define SS_PAGING_IND(mSGpTR)                 (mSGpTR)->ittiMsg.ss_paging_ind
#define SS_L1MACIND_CTRL(mSGpTR)              (mSGpTR)->ittiMsg.ss_l1macind_ctrl

/** VNG */
#define SS_VNG_PROXY_REQ(mSGpTR)              (mSGpTR)->ittiMsg.ss_vng_proxy_req
#define SS_VNG_PROXY_RESP(mSGpTR)             (mSGpTR)->ittiMsg.ss_vng_proxy_resp

/** DRB **/
#define SS_DRB_PDU_REQ(mSGpTR)                (mSGpTR)->ittiMsg.ss_drb_pdu_req
#define SS_DRB_PDU_IND(mSGpTR)                (mSGpTR)->ittiMsg.ss_drb_pdu_ind

// VTP
#define SS_VTP_PROXY_UPD(mSGpTR)              (mSGpTR)->ittiMsg.ss_vtp_proxy_upd
#define SS_VTP_PROXY_ACK(mSGpTR)              (mSGpTR)->ittiMsg.ss_vtp_proxy_ack
#define SS_VT_TIME_OUT(mSGpTR)                (mSGpTR)->ittiMsg.ss_vt_time_out
#define SDU_SIZE                           (1024)

/** SYS IND */
#define SS_SYSTEM_IND(mSGpTR)                 (mSGpTR)->ittiMsg.ss_system_ind

/** NR SRB */
#define SS_RRC_PDU_REQ(mSGpTR)                (mSGpTR)->ittiMsg.ss_rrc_pdu_req
#define SS_RRC_PDU_IND(mSGpTR)                (mSGpTR)->ittiMsg.ss_rrc_pdu_ind

/** PORTMAN */
typedef struct ss_sys_port_msg_ind {
  struct SYSTEM_CTRL_REQ* req;
  int userId;
} ss_sys_port_msg_ind_t;

typedef struct ss_sys_port_msg_cnf {
  struct SYSTEM_CTRL_CNF* cnf;
} ss_sys_port_msg_cnf_t;

typedef struct ss_nr_sys_port_msg_ind {
  struct NR_SYSTEM_CTRL_REQ* req;
  int userId;
} ss_nr_sys_port_msg_ind_t;

typedef struct ss_nr_sys_port_msg_cnf {
  struct NR_SYSTEM_CTRL_CNF* cnf;
} ss_nr_sys_port_msg_cnf_t;

/** SYS */
typedef struct ss_set_timinfo_s {
  uint16_t sfn;
  uint8_t  sf;
  int      cell_index;
  int      physCellId;
} ss_set_timinfo_t;

typedef ss_set_timinfo_t ss_upd_timinfo_t;

typedef struct ss_nrset_timinfo_s {
  uint16_t sfn;
  uint32_t  slot;
} ss_nrset_timinfo_t;

typedef ss_nrset_timinfo_t ss_nrupd_timinfo_t;

typedef struct ss_get_timinfo_s {
  uint8_t  EnquireTiming;
} ss_get_timinfo_t;

typedef struct ss_cell_attn_list_ind {
  uint16_t cell_id;
  uint8_t attn;               /*!< \brief 0xFF -> Off */
  ss_set_timinfo_t time_info; /*!< \brief Optional.*/
} ss_cell_attn_list_ind_t;

typedef struct ss_cell_attn_list_cnf {
  uint8_t status;
} ss_cell_attn_list_cnf_t;

enum PdcpCountFormat_Type_e {
        E_PdcpCount_Srb = 0,
        E_PdcpCount_DrbLongSQN = 1,
        E_PdcpCount_DrbShortSQN = 2,
        E_NrPdcpCount_Srb = 3,
        E_NrPdcpCount_DrbSQN12 = 4,
        E_NrPdcpCount_DrbSQN18 = 5,
};

typedef enum PdcpCountFormat_Type_e PdcpCountFormat_Type_e;

typedef struct pdcp_count_rb_s {
  uint8_t rb_id;
  uint8_t is_srb;
  PdcpCountFormat_Type_e ul_format;
  PdcpCountFormat_Type_e dl_format;
  uint32_t ul_count;
  uint32_t dl_count;
} pdcp_count_rb_t;

typedef struct ss_set_pdcp_cnt_s {
  pdcp_count_rb_t rb_list[MAX_RBS];
} ss_set_pdcp_cnt_t;

typedef struct ss_get_pdcp_cnt_s {
  //struct PdcpCountInfo_Type_Get_Dynamic Get;
  uint32_t size;
  pdcp_count_rb_t rb_info[MAX_RBS];
} ss_get_pdcp_cnt_t;

typedef struct ss_req_pdcp_cnt_s {
  rnti_t rnti;
  uint8_t rb_id;
} ss_req_pdcp_cnt_t;

typedef struct ss_l1macind_ctrl_s {
  bool rachpreamble_enable;
} ss_l1macind_ctrl_t;

/** LTE SRB */
typedef struct ss_rrc_pdu_req_s {
  uint8_t   srb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
} ss_rrc_pdu_req_t;

typedef struct ss_rrc_pdu_ind_s {
  uint8_t   srb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
  int       physCellId;
  frame_t     frame;         /*!< \brief  LTE frame number.*/
  sub_frame_t subframe;      /*!< \brief  LTE sub frame number.*/
} ss_rrc_pdu_ind_t;

/** NR SRB */
typedef struct ss_nrrrc_pdu_req_s {
  uint8_t   srb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
} ss_nrrrc_pdu_req_t;

typedef struct ss_nrrrc_pdu_ind_s {
  uint8_t   srb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
  frame_t     frame;         /*!< \brief  NR frame number.*/
  sub_frame_t subframe;      /*!< \brief  NR sub frame number.*/
} ss_nrrrc_pdu_ind_t;

/** VNG */

typedef struct ss_vng_proxy_resp_s {
  uint8_t     cell_id; /** Cell_id of the cell for
                           which VNG request came */
  uint32_t    sfn_sf;  /** Time at which response was
                           received from Proxy in the SYS task */
  uint8_t     status;  /** 0 Success: 1 Failure */
} ss_vng_proxy_resp_t;

/** DRB **/
typedef struct ss_drb_pdu_req_s {
  uint8_t   drb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  uint16_t  rnti;
} ss_drb_pdu_req_t;

typedef struct ss_drb_pdu_ind_s {
  uint8_t   drb_id;
  uint32_t  sdu_size;
  uint8_t   sdu[SDU_SIZE];
  frame_t     frame;         /*!< \brief  LTE frame number.*/
  sub_frame_t subframe;      /*!< \brief  LTE sub frame number.*/
  int       physCellId;
} ss_drb_pdu_ind_t;

typedef enum carrierBandwidthEUTRA_dl_Bandwidth_e {
        carrierBandwidthEUTRA_dl_Bandwidth_e_n6 = 0,
        carrierBandwidthEUTRA_dl_Bandwidth_e_n15 = 1,
        carrierBandwidthEUTRA_dl_Bandwidth_e_n25 = 2,
        carrierBandwidthEUTRA_dl_Bandwidth_e_n50 = 3,
        carrierBandwidthEUTRA_dl_Bandwidth_e_n75 = 4,
        carrierBandwidthEUTRA_dl_Bandwidth_e_n100 = 5,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare10 = 6,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare9 = 7,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare8 = 8,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare7 = 9,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare6 = 10,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare5 = 11,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare4 = 12,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare3 = 13,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare2 = 14,
        carrierBandwidthEUTRA_dl_Bandwidth_e_spare1 = 15,
        carrierBandwidthEUTRA_dl_Bandwidth_e_NONE = 16,
        carrierBandwidthEUTRA_dl_Bandwidth_e_INVALID = 0xFF
} Dl_Bw_e;

typedef enum VngCmd_e {
  INVALID = 0,
  CONFIGURE = 1,
  ACTIVATE,
  DEACTIVATE
} VngCmd;

typedef struct ss_vng_proxy_req_s {
  uint16_t    cell_id;    /** PCI of the cell for
                           which VNG request came */
  Dl_Bw_e     bw;         /** DL Bandwidth enum (ASN1) */
  int32_t     Noc_level;  /** 0 Success: 1 Failure */
  VngCmd      cmd;        /** CONF, ACTV, DEACTV */
} ss_vng_proxy_req_t;


typedef struct ss_paging_identity_s {

  /* UE paging identity */
  ue_paging_identity_t ue_paging_identity;

  /* Indicates origin of paging */
  cn_domain_t cn_domain;
}ss_paging_identity_t;

typedef struct subframe_offset_list_s {
  uint8_t num;
  sub_frame_t subframe_offset[10];
}subframe_offset_list_t;

typedef struct ss_paging_ind_s {
  uint16_t sfn;
  uint8_t  sf;

  /* UE identity index value.
   * Specified in 3GPP TS 36.304
   */
  unsigned ue_index_value:10;

  ss_paging_identity_t *paging_recordList;
  bool systemInfoModification;
  bool bSubframeOffsetListPresent;
  subframe_offset_list_t subframeOffsetList;
} ss_paging_ind_t;

typedef enum VtpCmd_e {
  VTP_DISABLE = 0,
  VTP_ENABLE = 1
  } VtpCmd;

typedef struct ss_vtp_proxy_upd_s {
  VtpCmd      cmd;
  ss_upd_timinfo_t    tinfo;
} ss_vtp_proxy_upd_t;

typedef struct ss_vtp_proxy_ack_s {
  VtpCmd      cmd;
  ss_upd_timinfo_t    tinfo;
} ss_vtp_proxy_ack_t;

typedef struct ss_vt_time_out_s {
  void *msg;
} ss_vt_time_out_t;

/** SYS IND */
typedef struct ss_system_ind_s
{
    bool           bitmask; //Flag for presence of optional parameter repetitionsPerPreambleAttempt
    frame_t        sfn;
    sub_frame_t    sf;
    int            physCellId;
    uint8_t        ra_PreambleIndex;
    bool           prtPower_Type;
    uint32_t       repetitionsPerPreambleAttempt;
} ss_system_ind_t;

#endif /* SS_MESSAGES_TYPES_H_ */
