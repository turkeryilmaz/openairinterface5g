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

#ifndef __MAC_DEFS_SL_H__
#define __MAC_DEFS_SL_H__

#include "sidelink_nr_ue_interface.h"
#include "NR_SL-ResourcePool-r16.h"
#include "NR_TDD-UL-DL-ConfigCommon.h"
#include "NR_MAC_COMMON/nr_mac.h"
#include "NR_UE_PHY_INTERFACE/NR_IF_Module.h"
#include "nr_ue_sci.h"
#include <pthread.h>
#include "mac_defs.h"

#define HARQ_ROUND_MAX 4
#define SL_NR_MAC_NUM_RX_RESOURCE_POOLS 1
#define SL_NR_MAC_NUM_TX_RESOURCE_POOLS 1
#define SL_NUM_BYTES_TIMERESOURCEBITMAP 20

//every 16 frames, SSB is repeated.
#define SL_NR_SSB_REPETITION_IN_FRAMES 16
#define SL_FRAME_NUMBER_CYCLE 1024

// Size of Fixed fields prio (3), sci_2ndstage(2),
// betaoffsetindicator(2), num dmrs ports (1), mcs (5bits)
#define SL_SCI_FORMAT_1A_LEN_IN_BITS_FIXED_FIELDS 13


#define sci_field_t dci_field_t

#define NR_UE_SL_SCHED_LOCK(lock)                                        \
  do {                                                             \
    int rc = pthread_mutex_lock(lock);                             \
    AssertFatal(rc == 0, "error while locking scheduler mutex\n"); \
  } while (0)

#define NR_UE_SL_SCHED_UNLOCK(lock)                                      \
  do {                                                             \
    int rc = pthread_mutex_unlock(lock);                           \
    AssertFatal(rc == 0, "error while locking scheduler mutex\n"); \
  } while (0)

#define NR_UE_SL_SCHED_ENSURE_LOCKED(lock)\
  do {\
    int rc = pthread_mutex_trylock(lock); \
    AssertFatal(rc == EBUSY, "this function should be called with the scheduler mutex locked\n");\
  } while (0)

typedef struct sidelink_sci_format_1a_fields {

  // Priority of this transmission
  uint8_t     priority; //3 bits
  //Indicates the format to be used in 2nd stage i.e SCI format 2 sent on PSSCH
  //00 - SCI FORMAT 2A, 01 - SCI FORMAT 2B, 10, 11 - Reserved
  //Spec 38.212 Table 8.3.1.1-1
  uint8_t     sci_format_2nd_stage; //2 bits
  //Num modulated symbols for stage 2 SCI - TBD:
  // Spec 38.212 Table 8.3.1.1-2
  uint8_t     beta_offset_indicator; //2 bits
  //determine the number of layers for data on PSSCH
  // Spec 38.212 Table 8.3.1.1-3
  uint8_t     num_dmrs_ports; //1 bit
  //Modulation and coding scheme to be used for data on PSSCH
  uint8_t     mcs; //5 bits

  //Identifies the frequence resource (subchannels) to be used for PSSCH/PSCCH
  //sl-MaxNumPerReserve is 2 - ceil(log2(N_subch*(N_subch+1)/2)) bits
  //sl-MaxNumPerReserve is 3 - ceil(log2(N_subch*(N_subch+1)(2*N_subch+1)/6)) bits
  sci_field_t frequency_resource_assignment; //variable
  //Identifies the Time resource (slots) to be used for PSSCH/PSCCH
  //sl-MaxNumPerReserve is 2 - 5 bits
  //sl-MaxNumPerReserve is 3 - 9 bits
  sci_field_t time_resource_assignment; //variable
  //TBD:
  //sl-MultiReserveResource is not configured - 0 bits
  //sl-MultiReserveResource is configured - ceil(log2(number of entries in sl-ResourceReservePeriodList)) bits
  sci_field_t resource_reservation_period; //variable
  //Identifies the DMRS Pattern to be used on PSSCH
  //ceil(log2(number of dmrs patterns in sl-PSSCH-DMRS-TimePatternList)) bits
  sci_field_t dmrs_pattern; //variable
  //Identifies the TABLE to be used to determine MCS on PSSCH
  //1 table configured in sl-Additional-MCS-Table - 1 bit
  //2 tables configured in sl-Additional-MCS-Table - 2 bits
  //Not configured- 0 bits
  sci_field_t additional_mcs_table_indicator; //variable
  //Identifies the number of symbols for PSFCH
  //sl-PSFCH-Period Not configured- 0 bits
  //if sl-PSFCH-Period configured and value 2 or 4 - 1 bit
  sci_field_t psfch_overhead_indication; //variable
  //number of bits determined by sl-NumReservedbits
  //Value encoded is 0
  sci_field_t reserved_bits;

} sidelink_sci_format_1a_fields_t;

typedef struct SL_ResourcePool_params {

  //This holds the structure from RRC
  NR_SL_ResourcePool_r16_t *respool;

  BIT_STRING_t phy_sl_bitmap;
  //NUM Subchannels in this resource pool
  uint16_t num_subch;

  //SCI-1A length is the same for this resource pool.
  uint16_t sci_1a_len;

  //SCI-1A configuration according to RESPOOL configured.
  sidelink_sci_format_1a_fields_t sci_1a;

  uint8_t tproc0;  // T_proc0 in slots
  uint8_t tproc1;  // T_proc1 in slots
  uint16_t t0;      // T0 - Sensing window
  uint8_t t1; // T1 - The offset in number of slots between the slot in which the resource
              // selection is triggered and the start of the selection window
  uint16_t t2;  // T2 - The configured value of T2 (end of selection window)
  uint8_t t2min; // t2min
} SL_ResourcePool_params_t;

typedef struct sl_ssb_timealloc {

  uint32_t sl_NumSSB_WithinPeriod;
  uint32_t sl_TimeOffsetSSB;
  uint32_t sl_TimeInterval;

} sl_ssb_timealloc_t;

typedef struct sl_bch_params {

  //configured from RRC
  //Parameters used to determine PSBCH slot
  sl_ssb_timealloc_t ssb_time_alloc;
  uint16_t slss_id;
  bool     status;
  uint8_t  sl_mib[4];

  //Parameters incremented by MAC PSBCH scheduler
  //after every SSB txn/reception
  uint16_t num_ssb;
  uint16_t ssb_slot;

} sl_bch_params_t;

/**
 * \brief Structure to pass parameters to trigger the selection of candidate
 * resources as per TR 38.214 Section 8.1.4
 */
typedef struct {
    uint8_t priority;    // L1 priority prio_TX
    uint16_t packet_delay_budget_ms;   // remaining packet delay budget
    uint16_t l_subch;    // L_subCH; number of subchannels to be used
    uint16_t rri;        // resource reservation interval
    uint16_t resel_counter;       // C_resel counter
    int sl_thresh_rsrp;  // A threshold in dBm used for sensing based UE autonomous resource selection
    float sl_res_ratio;  /* The percentage threshold to indicate the
                                 minimum number of candidate single-slot
                                 resources to be selected using sensing procedure.
                              */
} nr_sl_transmission_params_t;

typedef struct sl_nr_ue_mac_params {

  //Holds the RX resource pool from RRC and its related parameters
  SL_ResourcePool_params_t *sl_RxPool[SL_NR_MAC_NUM_RX_RESOURCE_POOLS];
  //Holds the TX resource pool from RRC and its related parameters
  SL_ResourcePool_params_t *sl_TxPool[SL_NR_MAC_NUM_TX_RESOURCE_POOLS];

  //Holds either the TDD config from RRC
  //or TDD config decoded from SL-MIB
  NR_TDD_UL_DL_ConfigCommon_t *sl_TDD_config;
  nr_sl_transmission_params_t  mac_tx_params;

  // CSI params configured locally
  uint8_t symb_l0;
  uint8_t csi_type;
  uint8_t power_control_offset;
  uint8_t power_control_offset_ss;
  uint8_t freq_density;
  uint8_t subcarrier_spacing;
  uint8_t cyclic_prefix;
  uint16_t start_rb;
  uint16_t nr_of_rbs;
  uint8_t row;
  uint16_t freq_domain;
  uint8_t cdm_type;
  uint16_t scramb_id;
  uint8_t measurement_bitmap;
  uint8_t sl_LatencyBoundCSI_Report;

  // configured grant harq parameters
  uint8_t sl_Num_HARQ_Processes;
  uint8_t sl_HARQ_ProcID_offset;
  uint16_t sl_Periodic_RRI;

  //Configured from RRC
  uint32_t sl_MaxNumConsecutiveDTX;
  uint32_t sl_SSB_PriorityNR;
  uint8_t sl_CSI_Acquisition;

  //MAC prepares this and sends it to PHY
  nr_sl_phy_config_t sl_phy_config;

  //Holds Broadcast params incase UE sends Sidelink SSB
  sl_bch_params_t tx_sl_bch;
  //Holds Broadcast params incase UE receives SL-SSB
  sl_bch_params_t rx_sl_bch;

  //SSB RSRP in dBm
  int16_t ssb_rsrp_dBm;

  //Bitmap indicating which slots belong to sidelink
  //Right now supports 30Khz and 15Khz
  uint32_t sl_slot_bitmap;

  //adjust timing after new timing from sync is acquired.
  uint8_t adjust_timing;

  //Sidelink slots per frame
  uint16_t N_SL_SLOTS_perframe;

  uint16_t decoded_DFN;
  uint16_t decoded_slot;
  NR_bler_options_t sl_bler;

} sl_nr_ue_mac_params_t;


#endif
