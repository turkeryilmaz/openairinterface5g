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

/*
  \brief NR UE PHY functions prototypes
  \author R. Knopp, F. Kaltenberger
  \company EURECOM
  \email knopp@eurecom.fr
*/

#ifndef __openair_SCHED_H__
#define __openair_SCHED_H__

#include "PHY/defs_nr_UE.h"
#include "openair1/PHY/nr_phy_common/inc/nr_phy_common.h"


/*enum THREAD_INDEX { OPENAIR_THREAD_INDEX = 0,
                    TOP_LEVEL_SCHEDULER_THREAD_INDEX,
                    DLC_SCHED_THREAD_INDEX,
                    openair_SCHED_NB_THREADS
                  };*/ // do not modify this line


#define OPENAIR_THREAD_PRIORITY        255


#define OPENAIR_THREAD_STACK_SIZE     PTHREAD_STACK_MIN //4096 //RTL_PTHREAD_STACK_MIN*6
//#define DLC_THREAD_STACK_SIZE        4096 //DLC stack size
//#define UE_SLOT_PARALLELISATION
//#define UE_DLSCH_PARALLELISATION

/*enum openair_SCHED_STATUS {
  openair_SCHED_STOPPED=1,
  openair_SCHED_STARTING,
  openair_SCHED_STARTED,
  openair_SCHED_STOPPING
};*/

/*enum openair_ERROR {
  // HARDWARE CAUSES
  openair_ERROR_HARDWARE_CLOCK_STOPPED= 1,

  // SCHEDULER CAUSE
  openair_ERROR_OPENAIR_RUNNING_LATE,
  openair_ERROR_OPENAIR_SCHEDULING_FAILED,

  // OTHERS
  openair_ERROR_OPENAIR_TIMING_OFFSET_OUT_OF_BOUNDS,
};*/

/*enum openair_SYNCH_STATUS {
  openair_NOT_SYNCHED=1,
  openair_SYNCHED,
  openair_SCHED_EXIT
};*/

/*enum openair_HARQ_TYPE {
  openair_harq_DL = 0,
  openair_harq_UL,
  openair_harq_RA
};*/

#define DAQ_AGC_ON 1
#define DAQ_AGC_OFF 0

/** @addtogroup _PHY_PROCEDURES_
 * @{
 */

/*! \brief Scheduling for UE TX procedures in normal subframes.
  @param ue Pointer to UE variables on which to act
  @param proc Pointer to RXn-TXnp4 proc information
@param phy_data
*/
void phy_procedures_nrUE_TX(PHY_VARS_NR_UE *ue, const UE_nr_rxtx_proc_t *proc, nr_phy_data_tx_t *phy_data);

int pbch_processing(PHY_VARS_NR_UE *UE,
                    const UE_nr_rxtx_proc_t *proc,
                    const int symbol,
                    const c16_t rxdataF[UE->frame_parms.nb_antennas_rx][ALNARS_32_8(UE->frame_parms.ofdm_symbol_size)],
                    int *ssbIndex,
                    c16_t pbch_ch_est_time[UE->frame_parms.nb_antennas_rx][ALNARS_32_8(UE->frame_parms.ofdm_symbol_size)],
                    int16_t pbch_e_rx[NR_POLAR_PBCH_E],
                    int *pbchSymbCnt);

void pdsch_processing(PHY_VARS_NR_UE *ue,
                      notifiedFIFO_elt_t *msgData,
                      int symbol,
                      c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)],
                      c16_t *rxdataF_ext,
                      c16_t *pdsch_ch_estimates);

void prs_processing(const PHY_VARS_NR_UE *ue,
                    const UE_nr_rxtx_proc_t *proc,
                    const int symbol,
                    const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)]);

void processSlotTX(void *arg);

/*! \brief UE PRACH procedures.
 */
void nr_ue_prach_procedures(PHY_VARS_NR_UE *ue, const UE_nr_rxtx_proc_t *proc);

int8_t nr_find_ue(uint16_t rnti, PHY_VARS_eNB *phy_vars_eNB);

/*! \brief UL time alignment procedures for TA application
  @param ue
  @param slot_tx
  @param frame_tx
*/
void ue_ta_procedures(PHY_VARS_NR_UE *ue, int slot_tx, int frame_tx);

void set_tx_harq_id(NR_UE_ULSCH_t *ulsch, int harq_pid, int slot_tx);
int get_tx_harq_id(NR_UE_ULSCH_t *ulsch, int slot_tx);

int is_pbch_in_slot(fapi_nr_config_request_t *config, int frame, int slot, NR_DL_FRAME_PARMS *fp);
int is_ssb_in_slot(fapi_nr_config_request_t *config, int frame, int slot, NR_DL_FRAME_PARMS *fp);
bool is_csi_rs_in_symbol(fapi_nr_dl_config_csirs_pdu_rel15_t csirs_config_pdu, int symbol);

/*! \brief This function prepares the dl indication to pass to the MAC
 */
void nr_fill_dl_indication(nr_downlink_indication_t *dl_ind,
                           fapi_nr_dci_indication_t *dci_ind,
                           fapi_nr_rx_indication_t *rx_ind,
                           const UE_nr_rxtx_proc_t *proc,
                           const PHY_VARS_NR_UE *ue,
                           void *phy_data);

/*@}*/

/*! \brief This function prepares the dl rx indication
 */
void nr_fill_rx_indication(fapi_nr_rx_indication_t *rx_ind,
                           const uint8_t pdu_type,
                           const PHY_VARS_NR_UE *ue,
                           const NR_UE_DLSCH_t *dlsch0,
                           const NR_UE_DLSCH_t *dlsch1,
                           const uint16_t n_pdus,
                           const UE_nr_rxtx_proc_t *proc,
                           const void *typeSpecific,
                           uint8_t *b);

void nr_ue_pdcch_slot_init(nr_phy_data_t *phyData, UE_nr_rxtx_proc_t *proc, PHY_VARS_NR_UE *ue, nr_ue_phy_slot_data_t *slot_data);
void nr_ue_pdsch_slot_init(nr_phy_data_t *phyData, UE_nr_rxtx_proc_t *proc, PHY_VARS_NR_UE *ue, nr_ue_phy_slot_data_t *slot_data);
void nr_ue_pbch_slot_init(nr_phy_data_t *phyData, UE_nr_rxtx_proc_t *proc, PHY_VARS_NR_UE *ue, nr_ue_phy_slot_data_t *slot_data);
void nr_ue_csi_slot_init(nr_phy_data_t *phyData, UE_nr_rxtx_proc_t *proc, PHY_VARS_NR_UE *ue, nr_ue_phy_slot_data_t *slot_data);

bool nr_ue_pdcch_procedures(PHY_VARS_NR_UE *ue,
                            const UE_nr_rxtx_proc_t *proc,
                            nr_phy_data_t *phy_data,
                            nr_ue_phy_slot_data_t *slot_data,
                            int symbol,
                            c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)]);

bool nr_ue_dlsch_procedures(PHY_VARS_NR_UE *ue,
                            const UE_nr_rxtx_proc_t *proc,
                            NR_UE_DLSCH_t dlsch[2],
                            int16_t *llr[2]);

bool nr_ue_pdsch_procedures(PHY_VARS_NR_UE *ue,
                            const UE_nr_rxtx_proc_t *proc,
                            NR_UE_DLSCH_t dlsch[2],
                            c16_t *pdsch_dl_ch_estimates,
                            c16_t *rxdataF_ext);

void nr_pdsch_slot_init(nr_phy_data_t *phyData, PHY_VARS_NR_UE *ue);

void nr_pdcch_generate_llr(const PHY_VARS_NR_UE *ue,
                           const UE_nr_rxtx_proc_t *proc,
                           const int symbol,
                           const nr_phy_data_t *phy_data,
                           const int llrSize,
                           const int max_monOcc,
                           const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)],
                           c16_t llr[phy_data->phy_pdcch_config.nb_search_space * max_monOcc * llrSize]);

void nr_pdcch_dci_indication(const UE_nr_rxtx_proc_t *proc,
                             const int llrSize,
                             const int max_monOcc,
                             PHY_VARS_NR_UE *ue,
                             nr_phy_data_t *phy_data,
                             const c16_t llr[phy_data->phy_pdcch_config.nb_search_space * max_monOcc * llrSize]);

void nr_csi_im_symbol_power_estimation(const NR_DL_FRAME_PARMS *frame_parms,
                                       const UE_nr_rxtx_proc_t *proc,
                                       const fapi_nr_dl_config_csiim_pdu_rel15_t *csiim_config_pdu,
                                       const int symbol,
                                       const c16_t rxdataF[frame_parms->nb_antennas_rx][ALNARS_32_8(frame_parms->ofdm_symbol_size)],
                                       nr_csi_symbol_res_t *csi_im_res);

void csi_processing(PHY_VARS_NR_UE *ue,
                    const UE_nr_rxtx_proc_t *proc,
                    nr_phy_data_t *phy_data,
                    int symbol,
                    nr_ue_phy_slot_data_t *slot_data,
                    const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)]);

void nr_ue_csi_im_procedures(const fapi_nr_dl_config_csiim_pdu_rel15_t *csiim_config_pdu,
                             const nr_csi_symbol_res_t *res,
                             nr_csi_info_t *csi_phy_parms);

void nr_csi_slot_init(const PHY_VARS_NR_UE *ue,
                      const UE_nr_rxtx_proc_t *proc,
                      const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
                      nr_csi_info_t *nr_csi_info,
                      csi_mapping_parms_t *mapping_parms);

void nr_ue_csi_rs_symbol_procedures(
    const PHY_VARS_NR_UE *ue,
    const UE_nr_rxtx_proc_t *proc,
    const csi_mapping_parms_t *mapping_parms,
    const int symbol,
    const fapi_nr_dl_config_csirs_pdu_rel15_t *csirs_config_pdu,
    const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)],
    c16_t csi_rs_ls_estimates[ue->frame_parms.nb_antennas_rx][mapping_parms->ports][ue->frame_parms.ofdm_symbol_size],
    nr_csi_symbol_res_t *csi_symb_res);

void nr_ue_csi_rs_procedures(
    const PHY_VARS_NR_UE *ue,
    const UE_nr_rxtx_proc_t *proc,
    const NR_UE_CSI_RS *csirs_vars,
    const csi_mapping_parms_t *mapping_parms,
    nr_csi_info_t *csi_info,
    nr_csi_symbol_res_t *res,
    c16_t csi_rs_ls_estimated_channel[ue->frame_parms.nb_antennas_rx][mapping_parms->ports][ue->frame_parms.ofdm_symbol_size]);

void sl_slot_init(UE_nr_rxtx_proc_t *proc, PHY_VARS_NR_UE *ue, nr_ue_phy_slot_data_t *slot_data);
int psbch_pscch_processing(PHY_VARS_NR_UE *ue,
                           const UE_nr_rxtx_proc_t *proc,
                           nr_phy_data_t *phy_data,
                           nr_ue_phy_slot_data_t *slot_data,
                           int symbol,
                           const c16_t rxdataF[][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)]);
void phy_procedures_nrUE_SL_TX(PHY_VARS_NR_UE *ue, const UE_nr_rxtx_proc_t *proc, nr_phy_data_tx_t *phy_data);
/*! \brief This function prepares the sl indication to pass to the MAC
 */
void nr_fill_sl_indication(nr_sidelink_indication_t *sl_ind,
                           sl_nr_rx_indication_t *rx_ind,
                           sl_nr_sci_indication_t *sci_ind,
                           const UE_nr_rxtx_proc_t *proc,
                           const PHY_VARS_NR_UE *ue,
                           void *phy_data);
void nr_fill_sl_rx_indication(sl_nr_rx_indication_t *rx_ind,
                              uint8_t pdu_type,
                              PHY_VARS_NR_UE *ue,
                              uint16_t n_pdus,
                              const UE_nr_rxtx_proc_t *proc,
                              void *typeSpecific,
                              uint16_t rx_slss_id);

int nr_pdsch_generate_channel_estimates(
    const PHY_VARS_NR_UE *ue,
    const UE_nr_rxtx_proc_t *proc,
    const int symbol,
    const NR_UE_DLSCH_t *dlsch,
    const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)],
    uint32_t *nvar,
    c16_t channel_estimates[dlsch->Nl][ue->frame_parms.nb_antennas_rx][ue->frame_parms.ofdm_symbol_size]);

void nr_generate_pdsch_extracted_rxdataF(
    const PHY_VARS_NR_UE *ue,
    const UE_nr_rxtx_proc_t *proc,
    const int symbol,
    const NR_UE_DLSCH_t *dlsch,
    const c16_t rxdataF[ue->frame_parms.nb_antennas_rx][ALNARS_32_8(ue->frame_parms.ofdm_symbol_size)],
    c16_t rxdataF_ext[ue->frame_parms.nb_antennas_rx][dlsch->dlsch_config.number_rbs * NR_NB_SC_PER_RB]);

void free_pdsch_slot_proc_buffers(nr_ue_symb_data_t *symb_data);
#endif
/** @}*/
