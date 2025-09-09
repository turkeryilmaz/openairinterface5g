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

#ifndef _NR_SDAP_ENTITY_H_
#define _NR_SDAP_ENTITY_H_

#include <assertions.h>
#include <stdbool.h>
#include <stdint.h>
#include "NR_QFI.h"
#include "NR_SDAP-Config.h"
#include "common/platform_constants.h"

#define SDAP_BITMASK_DC             (0x80)
#define SDAP_BITMASK_R              (0x40)
#define SDAP_BITMASK_QFI            (0x3F)
#define SDAP_BITMASK_RQI            (0x40)
#define SDAP_HDR_UL_DATA_PDU        (1)
#define SDAP_HDR_UL_CTRL_PDU        (0)
#define SDAP_HDR_LENGTH             (1)
#define SDAP_MAX_QFI                (64)
#define SDAP_MAP_RULE_EMPTY         (0)
#define AVLBL_DRB                   (5)
#define SDAP_NO_MAPPING_RULE        (0)
#define SDAP_REFLECTIVE_MAPPING     (1)
#define SDAP_RQI_HANDLING           (1)
#define SDAP_CTRL_PDU_MAP_DEF_DRB   (0)
#define SDAP_CTRL_PDU_MAP_RULE_DRB  (1)
#define SDAP_MAX_PDU                (9000)
#define SDAP_MAX_NUM_OF_ENTITIES    (MAX_DRBS_PER_UE * MAX_MOBILES_PER_ENB)
#define SDAP_MAX_UE_ID              (65536)

/*
 * The values of QoS Flow ID (QFI) and Reflective QoS Indication,
 * are located in the PDU Session Container, which is conveyed by
 * the GTP-U Extension Header. Inside the DL PDU SESSION INFORMATION frame.
 * TS 38.415 Fig. 5.5.2.1-1
 */

extern instance_t *N3GTPUInst;

typedef struct nr_sdap_dl_hdr_s {
  uint8_t QFI:6;
  uint8_t RQI:1;
  uint8_t RDI:1;
} __attribute__((packed)) nr_sdap_dl_hdr_t;

typedef struct nr_sdap_ul_hdr_s {
  uint8_t QFI:6;
  uint8_t R:1;
  uint8_t DC:1;
} __attribute__((packed)) nr_sdap_ul_hdr_t;

typedef struct qfi2drb_s {
  rb_id_t drb_id;
  bool    has_sdap_rx;
  bool    has_sdap_tx;
} qfi2drb_t;

void nr_pdcp_submit_sdap_ctrl_pdu(ue_id_t ue_id, rb_id_t sdap_ctrl_pdu_drb, nr_sdap_ul_hdr_t ctrl_pdu);

typedef struct sdap_configuration_s {
  int pdusession_id;
  int drb_id;
  bool sdap_rx;
  bool sdap_tx;
  bool defaultDRB;
  NR_QFI_t mappedQFIs2Add[SDAP_MAX_QFI];
  uint8_t mappedQFIs2AddCount;
  NR_QFI_t mappedQFIs2Release[SDAP_MAX_QFI];
  uint8_t mappedQFIs2ReleaseCount;
} sdap_config_t;

typedef struct nr_sdap_entity_s {
  ue_id_t ue_id;
  rb_id_t default_drb;
  /// sdap_tun_read_thread needs to know if we are gNB/UE, so for noS1 mode,
  /// store which one we are
  bool is_gnb;
  int pdusession_id;
  int pdusession_sock;
  pthread_t pdusession_thread;
  bool stop_thread;
  int qfi;

  qfi2drb_t qfi2drb_table[SDAP_MAX_QFI];

  void (*qfi2drb_map_update)(struct nr_sdap_entity_s *entity, uint8_t qfi, rb_id_t drb, bool has_sdap_rx, bool has_sdap_tx);
  void (*qfi2drb_map_delete)(struct nr_sdap_entity_s *entity, uint8_t qfi);
  int (*qfi2drb_map)(struct nr_sdap_entity_s *entity, uint8_t qfi);

  nr_sdap_ul_hdr_t (*sdap_construct_ctrl_pdu)(uint8_t qfi);
  int (*sdap_map_ctrl_pdu)(struct nr_sdap_entity_s *entity, int map_type, uint8_t dl_qfi);
  void (*sdap_submit_ctrl_pdu)(ue_id_t ue_id, rb_id_t sdap_ctrl_pdu_drb, nr_sdap_ul_hdr_t ctrl_pdu);

  bool (*tx_entity)(struct nr_sdap_entity_s *entity,
                    protocol_ctxt_t *ctxt_p,
                    const srb_flag_t srb_flag,
                    const rb_id_t rb_id,
                    const mui_t mui,
                    const confirm_t confirm,
                    const sdu_size_t sdu_buffer_size,
                    unsigned char *const sdu_buffer,
                    const pdcp_transmission_mode_t pt_mode,
                    const uint32_t *sourceL2Id,
                    const uint32_t *destinationL2Id,
                    const uint8_t qfi,
                    const bool rqi);

  void (*rx_entity)(struct nr_sdap_entity_s *entity,
                    rb_id_t pdcp_entity,
                    int is_gnb,
                    bool has_sdap_rx,
                    int pdusession_id,
                    ue_id_t ue_id,
                    char *buf,
                    int size);

  /* List of entities */
  struct nr_sdap_entity_s *next_entity;
} nr_sdap_entity_t;

/* QFI to DRB Mapping Related Function */
void nr_sdap_qfi2drb_map_update(nr_sdap_entity_t *entity, uint8_t qfi, rb_id_t drb, bool has_sdap_rx, bool has_sdap_tx);

/* QFI to DRB Mapping Related Function */
void nr_sdap_qfi2drb_map_del(nr_sdap_entity_t *entity, uint8_t qfi);

/*
 * TS 37.324 5.3 QoS flow to DRB Mapping 
 * construct an end-marker control PDU, as specified in the subclause 6.2.3, for the QoS flow;
 */
nr_sdap_ul_hdr_t nr_sdap_construct_ctrl_pdu(uint8_t qfi);

/*
 * TS 37.324 5.3 QoS flow to DRB Mapping 
 * map the end-marker control PDU to the
 * 1.) default DRB or 
 * 2.) DRB according to the stored QoS flow to DRB mapping rule
 */
int nr_sdap_map_ctrl_pdu(nr_sdap_entity_t *entity, int map_type, uint8_t dl_qfi);

/*
 * TS 37.324 5.3 QoS flow to DRB Mapping 
 * Submit the end-marker control PDU to the lower layer.
 */
void nr_sdap_submit_ctrl_pdu(ue_id_t ue_id, rb_id_t sdap_ctrl_pdu_drb, nr_sdap_ul_hdr_t ctrl_pdu);

void nr_sdap_ue_qfi2drb_config(nr_sdap_entity_t *entity, const ue_id_t ue_id, const sdap_config_t sdap);

/*
 * TS 37.324 4.4 5.1.1 SDAP entity establishment
 * Establish an SDAP entity.
 */
nr_sdap_entity_t *new_nr_sdap_entity(const int is_gnb, const ue_id_t ue_id, const sdap_config_t sdap);

/* Entity Handling Related Functions */
nr_sdap_entity_t *nr_sdap_get_entity(ue_id_t ue_id, int pdusession_id);

sdap_config_t get_sdap_Config(int is_gnb, ue_id_t UEid, NR_SDAP_Config_t *sdap_Config, int drb_id);

void nr_sdap_release_drb(ue_id_t ue_id, int drb_id, int pdusession_id);

/**
 * @brief Function to delete a single SDAP Entity based on the ue_id and pdusession_id.
 * @note  1. SDAP entities may have the same ue_id.
 * @note  2. SDAP entities may have the same pdusession_id, but not with the same ue_id.
 * @note  3. The combination of ue_id and pdusession_id, is unique for each SDAP entity.
 * @param[in] ue_id         Unique identifier for the User Equipment. ID Range [0, 65536].
 * @param[in] pdusession_id Unique identifier for the Packet Data Unit Session. ID Range [0, 256].
 * @return                  True, if successfully deleted entity, false otherwise.
 */
bool nr_sdap_delete_entity(ue_id_t ue_id, int pdusession_id);

/**
 * @brief Function to delete all SDAP Entities based on the ue_id.
 * @param[in] ue_id         Unique identifier for the User Equipment. ID Range [0, 65536].
 * @return                  True, it deleted at least one entity, false otherwise.
 */
bool nr_sdap_delete_ue_entities(ue_id_t ue_id);

/**
 * @brief indicates whether it is a receiving SDAP entity
 *        i.e. for UE, header for DL data is present
 *             for gNB, header for UL data is present
 */
bool is_sdap_rx(bool is_gnb, NR_SDAP_Config_t *sdap_config);

/**
 * @brief indicates whether it is a transmitting SDAP entity
 *        i.e. for UE, header for UL data is present
 *             for gNB, header for DL data is present
 */
bool is_sdap_tx(bool is_gnb, NR_SDAP_Config_t *sdap_config);

/**
 * @brief               Run the SDAP reconfiguration for the DRB
 * @param[in] ue_id     Unique identifier for the User Equipment. ID Range [0, 65536].
 */
void nr_reconfigure_sdap_entity(NR_SDAP_Config_t *sdap_config, ue_id_t ue_id, int pdusession_id, int drb_id);

void set_qfi(uint8_t qfi, uint8_t pduid, ue_id_t ue_id);
void remove_ip_if(nr_sdap_entity_t *entity);
#endif
