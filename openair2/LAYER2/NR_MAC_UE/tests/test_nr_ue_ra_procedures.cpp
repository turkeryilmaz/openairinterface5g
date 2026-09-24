/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "gtest/gtest.h"
extern "C" {
#include "common/platform_types.h"
#include "openair2/LAYER2/NR_MAC_UE/mac_proto.h"
#include "NR_SearchSpace.h"
#include "executables/softmodem-common.h"
#include "executables/agc_options.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_oai_api.h"
#include "common/utils/ocp_itti/intertask_interface.h"

static agc_options_t test_agc_options;
const agc_options_t *get_agc_options(void)
{
  return &test_agc_options;
}

static softmodem_params_t softmodem_params;
softmodem_params_t *get_softmodem_params(void)
{
  return &softmodem_params;
}
void nr_mac_rrc_ra_ind(const module_id_t mod_id, bool success)
{
  UNUSED(mod_id);
  UNUSED(success);
}
void nr_mac_rrc_msg3_ind(const module_id_t mod_id, const int rnti, bool prepare_payload)
{
  UNUSED(mod_id);
  UNUSED(rnti);
  UNUSED(prepare_payload);
}
void nr_mac_rlc_status_ind(uint16_t ue_id, frame_t frame, int n_ch, const logical_chan_id_t *ch, mac_rlc_status_resp_t *ret)
{
  UNUSED(ue_id);
  UNUSED(frame);
  UNUSED(n_ch);
  UNUSED(ch);
  UNUSED(ret);
}
void nr_mac_rrc_inactivity_timer_ind(const module_id_t mod_id)
{
  UNUSED(mod_id);
}
tbs_size_t nr_mac_rlc_data_req(const module_id_t module_idP,
                               const uint16_t ue_id,
                               const bool gnb_flagP,
                               const logical_chan_id_t channel_idP,
                               const tb_size_t tb_sizeP,
                               char *buffer_pP)
{
  UNUSED(module_idP);
  UNUSED(ue_id);
  UNUSED(gnb_flagP);
  UNUSED(channel_idP);
  UNUSED(tb_sizeP);
  UNUSED(buffer_pP);
  return 0;
}
void nr_mac_rlc_data_ind(const module_id_t module_idP,
                         const uint16_t ue_id,
                         const bool gnb_flagP,
                         const nr_rlc_data_ind_t *data,
                         int num_data)
{
  UNUSED(module_idP);
  UNUSED(ue_id);
  UNUSED(gnb_flagP);
  UNUSED(data);
  UNUSED(num_data);
}
void nr_mac_rrc_verification_failed(const module_id_t mod_id)
{
  UNUSED(mod_id);
}
bool nr_rlc_activate_srb0(int ue_id,
                          void *data,
                          void (*send_initial_ul_rrc_message)(int ue_id, const uint8_t *sdu, sdu_size_t sdu_len, void *data))
{
  UNUSED(ue_id);
  UNUSED(data);
  UNUSED(send_initial_ul_rrc_message);
  return true;
}
int nr_rlc_module_init(nr_rlc_op_mode_t mode)
{
  UNUSED(mode);
  return 0;
}
MessageDef *itti_alloc_new_message(task_id_t origin_task_id, instance_t originInstance, MessagesIds message_id)
{
  UNUSED(origin_task_id);
  UNUSED(originInstance);
  UNUSED(message_id);
  return NULL;
}
int itti_send_msg_to_task(task_id_t task_id, instance_t instance, MessageDef *message)
{
  UNUSED(task_id);
  UNUSED(instance);
  UNUSED(message);
  return 0;
}
typedef uint32_t channel_t;
int8_t nr_mac_rrc_data_ind_ue(const module_id_t module_id,
                              const int CC_id,
                              const uint8_t gNB_index,
                              const int hfn,
                              const frame_t frame,
                              const int slot,
                              const rnti_t rnti,
                              const uint32_t cellid,
                              const long arfcn,
                              const channel_t channel,
                              const uint8_t *pduP,
                              const sdu_size_t pdu_len)
{
  UNUSED(module_id);
  UNUSED(CC_id);
  UNUSED(gNB_index);
  UNUSED(hfn);
  UNUSED(frame);
  UNUSED(slot);
  UNUSED(rnti);
  UNUSED(cellid);
  UNUSED(arfcn);
  UNUSED(channel);
  UNUSED(pduP);
  UNUSED(pdu_len);
  return 0;
}
bool check_csi_report_consistency(const NR_CSI_MeasConfig_t *meas)
{
  UNUSED(meas);
  return true;
}
void nr_mac_rrc_meas_ind_ue(module_id_t module_id,
                            uint32_t gNB_index,
                            uint16_t Nid_cell,
                            bool csi_meas,
                            bool is_neighboring_cell,
                            int rsrp_dBm)
{
  UNUSED(module_id);
  UNUSED(gNB_index);
  UNUSED(Nid_cell);
  UNUSED(csi_meas);
  UNUSED(is_neighboring_cell);
  UNUSED(rsrp_dBm);
}
}
#include <climits>
#include <cstdio>
#include <cstdlib>
#include "common/utils/LOG/log.h"

TEST(test_pucch_config, common_resource_uses_default_power_procedure)
{
  NR_UE_MAC_INST_t mac = {0};
  NR_UE_UL_BWP_t current_bwp = {0};
  NR_PUCCH_ConfigCommon_t pucch_config_common = {0};
  long p0_nominal = 4;
  current_bwp.scs = 1;
  current_bwp.BWPSize = 106;
  current_bwp.channel_bandwidth = 20;
  current_bwp.pucch_ConfigCommon = &pucch_config_common;
  pucch_config_common.p0_nominal = &p0_nominal;
  mac.current_UL_BWP = &current_bwp;
  mac.p_Max = INT_MIN;
  mac.nr_band = 20;
  mac.frequency_range = FR1;
  mac.frame_structure.frame_type = TDD;
  mac.frame_structure.numb_slots_frame = 20;

  PUCCH_sched_t pucch = {0};
  pucch.n_harq = 1;
  pucch.ack_payload = 1;
  pucch.initial_pucch_id = 0;
  pucch.pucch_ResourceCommon = 0;
  fapi_nr_ul_config_pucch_pdu pucch_pdu = {0};
  ASSERT_EQ(0, nr_ue_configure_pucch(&mac, 0, 0, 0x1234, &pucch, &pucch_pdu));
  EXPECT_EQ(0, pucch_pdu.format_type);
  EXPECT_EQ(7, pucch_pdu.pucch_tx_power);

  mac.p_Max = -5;
  mac.pucch_power_control_initialized = true;
  mac.G_b_f_c = 0;
  pucch_pdu = (fapi_nr_ul_config_pucch_pdu){0};
  ASSERT_EQ(0, nr_ue_configure_pucch(&mac, 0, 0, 0x1234, &pucch, &pucch_pdu));
  EXPECT_EQ(-5, pucch_pdu.pucch_tx_power);
}

TEST(test_ssb_pathloss_context, release_preserves_reestablished_serving_cell_measurement)
{
  NR_UE_MAC_INST_t mac = {0};
  mac.mib_ssb = 0;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.ssb_measurements[mac.mib_ssb].ssb_rsrp_dBm = -86;

  ASSERT_NE(nullptr, get_dl_bwp_structure(&mac, 0, true));
  ASSERT_NE(nullptr, get_ul_bwp_structure(&mac, 0, true));
  auto *common_search_space = static_cast<NR_SearchSpace_t *>(calloc(1, sizeof(NR_SearchSpace_t)));
  ASSERT_NE(nullptr, common_search_space);
  NR_SearchSpace_t *common_search_spaces[] = {common_search_space};
  mac.config_BWP_PDCCH[0].list_common_SS.array = common_search_spaces;
  mac.config_BWP_PDCCH[0].list_common_SS.count = 1;
  mac.config_BWP_PDCCH[0].list_common_SS.size = 1;
  int16_t pathloss = INT16_MIN;
  ASSERT_TRUE(compute_nr_SSB_PL(&mac, &pathloss));
  EXPECT_EQ(61, pathloss);

  release_mac_configuration(&mac, RE_ESTABLISHMENT);
  EXPECT_EQ(-86, mac.ssb_measurements[mac.mib_ssb].ssb_rsrp_dBm);
  ASSERT_TRUE(compute_nr_SSB_PL(&mac, &pathloss));
  EXPECT_EQ(61, pathloss);

  release_mac_configuration(&mac, RRC_SETUP_REESTAB_RESUME);
  EXPECT_EQ(-86, mac.ssb_measurements[mac.mib_ssb].ssb_rsrp_dBm);
  ASSERT_TRUE(compute_nr_SSB_PL(&mac, &pathloss));
  EXPECT_EQ(61, pathloss);

  release_mac_configuration(&mac, GO_TO_IDLE_KEEP_CAMPED);
  EXPECT_EQ(-86, mac.ssb_measurements[mac.mib_ssb].ssb_rsrp_dBm);
  ASSERT_TRUE(compute_nr_SSB_PL(&mac, &pathloss));
  EXPECT_EQ(61, pathloss);

  release_mac_configuration(&mac, GO_TO_IDLE);
  EXPECT_EQ(INT_MIN, mac.ssb_measurements[mac.mib_ssb].ssb_rsrp_dBm);
  EXPECT_FALSE(compute_nr_SSB_PL(&mac, &pathloss));
}

TEST(test_ssb_pathloss_context, unavailable_does_not_become_a_power_or_phr_value)
{
  NR_UE_MAC_INST_t mac = {0};
  mac.mib_ssb = 0;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.ssb_measurements[0].ssb_rsrp_dBm = INT_MIN;
  mac.ra.prach_resources.ra_preamble_rx_target_power = -70;
  mac.ra.prach_resources.Pc_max = 23;

  int16_t pathloss = 123;
  int16_t prach_power = 321;
  EXPECT_FALSE(compute_nr_SSB_PL(&mac, &pathloss));
  EXPECT_EQ(123, pathloss);
  EXPECT_FALSE(get_prach_tx_power(&mac, &prach_power));
  EXPECT_EQ(321, prach_power);

  mac.mib_ssb = MAX_NB_SSB;
  EXPECT_FALSE(compute_nr_SSB_PL(&mac, &pathloss));

  mac.mib_ssb = 0;
  mac.ssb_measurements[0].ssb_rsrp_dBm = -90;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -100;
  EXPECT_FALSE(compute_nr_SSB_PL(&mac, &pathloss));

  mac.ssb_measurements[0].ssb_rsrp_dBm = INT16_MIN;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = INT16_MAX;
  EXPECT_FALSE(compute_nr_SSB_PL(&mac, &pathloss));

  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  ASSERT_TRUE(compute_nr_SSB_PL(&mac, &pathloss));
  EXPECT_EQ(61, pathloss);
  ASSERT_TRUE(get_prach_tx_power(&mac, &prach_power));
  EXPECT_EQ(-9, prach_power);

  nr_phr_info_t *phr_info = &mac.scheduling_info.phr_info;
  phr_info->is_configured = true;
  phr_info->PathlossLastValue = 50;
  phr_info->PathlossChange_db = 1;
  phr_info->phr_reporting = 0;
  nr_timer_setup(&phr_info->prohibitPHR_Timer, 1, 1);
  nr_timer_start(&phr_info->prohibitPHR_Timer);
  mac.ssb_measurements[0].ssb_rsrp_dBm = INT_MIN;

  update_mac_ul_timers(&mac);
  EXPECT_FALSE(nr_timer_is_active(&phr_info->prohibitPHR_Timer));
  EXPECT_EQ(50, phr_info->PathlossLastValue);
  EXPECT_EQ(0, phr_info->phr_reporting);

  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  update_mac_ul_timers(&mac);
  const int phr_after_fresh_measurement = phr_info->phr_reporting;
  EXPECT_NE(0, phr_after_fresh_measurement);
  EXPECT_EQ(50, phr_info->PathlossLastValue);

  update_mac_ul_timers(&mac);
  EXPECT_EQ(phr_after_fresh_measurement, phr_info->phr_reporting);
}

TEST(test_ssb_pathloss_context, expired_backoff_defers_then_scheduler_selects_once)
{
  NR_UE_MAC_INST_t mac = {0};
  NR_RACH_ConfigCommon_t rach_config_common = {0};
  NR_UE_UL_BWP_t current_bwp = {0};
  NR_UE_DL_BWP_t current_dl_bwp = {0};
  fapi_nr_ul_config_request_t ul_config = {0};
  long msg1_scs = 1;

  mac.current_UL_BWP = &current_bwp;
  mac.current_DL_BWP = &current_dl_bwp;
  mac.ul_config_request = &ul_config;
  mac.mib_ssb = 0;
  mac.ssb_list.nb_tx_ssb = 1;
  mac.ssb_list.nb_ssb_per_index[0] = 0;
  mac.ssb_ro_preambles = (ssb_ro_preambles_t){.ssb_per_ro = 1, .preambles_per_ssb = 4};
  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.p_Max = 23;
  mac.p_Max_alt = INT_MIN;
  mac.nr_band = 78;
  mac.frequency_range = FR1;
  mac.frame_structure.frame_type = FDD;
  mac.frame_structure.numb_slots_frame = 20;

  current_bwp.scs = msg1_scs;
  current_bwp.bwp_id = 0;
  current_bwp.channel_bandwidth = 40;
  current_bwp.rach_ConfigCommon = &rach_config_common;
  current_dl_bwp.bwp_id = 0;
  rach_config_common.msg1_SubcarrierSpacing = &msg1_scs;
  rach_config_common.rach_ConfigGeneric.prach_ConfigurationIndex = 0;
  rach_config_common.rach_ConfigGeneric.msg1_FDM = 0;
  ASSERT_EQ(0, pthread_mutex_init(&ul_config.mutex_ul_config, NULL));

  init_RA(&mac);
  mac.ra.prach_resources.preamble_tx_counter = 3;
  mac.ra.prach_resources.preamble_power_ramping_cnt = 2;
  mac.ra.ra_state = nrRA_WAIT_RAR;
  nr_timer_setup(&mac.ra.RA_backoff_timer, 1, 1);
  nr_timer_start(&mac.ra.RA_backoff_timer);
  mac.ssb_measurements[0].ssb_rsrp_dBm = INT_MIN;

  update_mac_ul_timers(&mac);
  EXPECT_FALSE(nr_timer_is_active(&mac.ra.RA_backoff_timer));
  EXPECT_EQ(nrRA_GENERATE_PREAMBLE, mac.ra.ra_state);
  EXPECT_TRUE(mac.ra.defer_preamble_for_ssb_pathloss);
  EXPECT_EQ(3, mac.ra.prach_resources.preamble_tx_counter);
  EXPECT_EQ(2, mac.ra.prach_resources.preamble_power_ramping_cnt);

  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  nr_uplink_indication_t ul_info = {.frame = 2, .slot = 0};
  nr_ue_ul_scheduler(&mac, &ul_info);
  EXPECT_FALSE(mac.ra.defer_preamble_for_ssb_pathloss);
  EXPECT_EQ(3, mac.ra.prach_resources.preamble_tx_counter);
  EXPECT_EQ(3, mac.ra.prach_resources.preamble_power_ramping_cnt);

  nr_ue_ul_scheduler(&mac, &ul_info);
  EXPECT_FALSE(mac.ra.defer_preamble_for_ssb_pathloss);
  EXPECT_EQ(3, mac.ra.prach_resources.preamble_tx_counter);
  EXPECT_EQ(3, mac.ra.prach_resources.preamble_power_ramping_cnt);
  EXPECT_EQ(0, ul_config.number_pdus);
  EXPECT_EQ(0, pthread_mutex_destroy(&ul_config.mutex_ul_config));
}

TEST(test_ssb_pathloss_context, unavailable_scheduler_retires_current_slot_pusch)
{
  NR_UE_MAC_INST_t mac = {0};
  fapi_nr_ul_config_request_t ul_config = {0};

  mac.state = UE_CONNECTED;
  mac.mib_ssb = 0;
  mac.ssb_measurements[0].ssb_rsrp_dBm = INT_MIN;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.frame_structure.frame_type = FDD;
  mac.frame_structure.numb_slots_frame = 20;
  mac.ul_config_request = &ul_config;
  ul_config.frame = 17;
  ul_config.slot = 0;
  ul_config.number_pdus = 1;
  ul_config.ul_config_list[0].pdu_type = FAPI_NR_UL_CONFIG_TYPE_PUSCH;
  ASSERT_EQ(0, pthread_mutex_init(&ul_config.mutex_ul_config, NULL));
  ul_config.ul_config_list[0].lock = &ul_config.mutex_ul_config;
  ul_config.ul_config_list[0].privateNBpdus = &ul_config.number_pdus;

  nr_uplink_indication_t ul_info = {.frame = 17, .slot = 0};
  nr_ue_ul_scheduler(&mac, &ul_info);
  EXPECT_EQ(0, ul_config.number_pdus);

  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  fapi_nr_ul_config_request_pdu_t *pdu = fapiLockIterator(&ul_config, 17, 0);
  ASSERT_NE(nullptr, pdu);
  EXPECT_EQ(FAPI_NR_END, pdu->pdu_type);
  EXPECT_EQ(0, *pdu->privateNBpdus);
  release_ul_config(pdu, false);
  EXPECT_EQ(0, pthread_mutex_destroy(&ul_config.mutex_ul_config));
}

TEST(test_ssb_pathloss_context, unavailable_scheduler_retires_only_current_feedback)
{
  NR_UE_MAC_INST_t mac = {0};
  fapi_nr_ul_config_request_t ul_config[3] = {};

  mac.state = UE_CONNECTED;
  mac.mib_ssb = 0;
  mac.ssb_measurements[0].ssb_rsrp_dBm = INT_MIN;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.frame_structure.frame_type = FDD;
  mac.frame_structure.numb_slots_frame = 20;
  mac.ul_config_request = ul_config;
  for (auto &config : ul_config)
    ASSERT_EQ(0, pthread_mutex_init(&config.mutex_ul_config, NULL));

  mac.ra.ra_pucch = static_cast<RA_PUCCH_SCHED_t *>(calloc(1, sizeof(*mac.ra.ra_pucch)));
  ASSERT_NE(nullptr, mac.ra.ra_pucch);
  mac.ra.ra_pucch->sched_frame = 17;
  mac.ra.ra_pucch->sched_slot = 0;
  mac.dl_harq_info[0][0] = {.active = true, .ack_received = true, .ul_frame = 17, .ul_slot = 0};
  mac.dl_harq_info[1][0] = {.active = true, .ack_received = true, .ul_frame = 17, .ul_slot = 2};

  nr_uplink_indication_t missed_now = {.frame = 17, .slot = 0};
  nr_ue_ul_scheduler(&mac, &missed_now);
  EXPECT_EQ(nullptr, mac.ra.ra_pucch);
  EXPECT_FALSE(mac.dl_harq_info[0][0].active);
  EXPECT_FALSE(mac.dl_harq_info[0][0].ack_received);
  EXPECT_TRUE(mac.dl_harq_info[1][0].active);
  EXPECT_TRUE(mac.dl_harq_info[1][0].ack_received);

  mac.ra.ra_pucch = static_cast<RA_PUCCH_SCHED_t *>(calloc(1, sizeof(*mac.ra.ra_pucch)));
  ASSERT_NE(nullptr, mac.ra.ra_pucch);
  mac.ra.ra_pucch->sched_frame = 17;
  mac.ra.ra_pucch->sched_slot = 2;
  nr_uplink_indication_t missed_before_future = {.frame = 17, .slot = 1};
  nr_ue_ul_scheduler(&mac, &missed_before_future);
  ASSERT_NE(nullptr, mac.ra.ra_pucch);
  EXPECT_EQ(17, mac.ra.ra_pucch->sched_frame);
  EXPECT_EQ(2, mac.ra.ra_pucch->sched_slot);

  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  nr_uplink_indication_t fresh_future = {.frame = 17, .slot = 2};
  nr_ue_ul_scheduler(&mac, &fresh_future);
  EXPECT_EQ(nullptr, mac.ra.ra_pucch);
  EXPECT_EQ(1, ul_config[2].number_pdus);
  EXPECT_EQ(FAPI_NR_UL_CONFIG_TYPE_PUCCH, ul_config[2].ul_config_list[0].pdu_type);
  // The future ordinary ACK was not retired by either skipped target or cached-RA PUCCH delivery.
  EXPECT_TRUE(mac.dl_harq_info[1][0].active);
  EXPECT_TRUE(mac.dl_harq_info[1][0].ack_received);

  for (auto &config : ul_config)
    EXPECT_EQ(0, pthread_mutex_destroy(&config.mutex_ul_config));
}

TEST(test_ssb_pathloss_context, unavailable_queued_msg3_retires_attempt_once_then_retries_fresh)
{
  NR_UE_MAC_INST_t mac = {0};
  NR_RACH_ConfigCommon_t rach_config_common = {0};
  NR_UE_UL_BWP_t current_bwp = {0};
  NR_UE_DL_BWP_t current_dl_bwp = {0};
  fapi_nr_ul_config_request_t ul_config = {0};
  long msg1_scs = 1;

  mac.current_UL_BWP = &current_bwp;
  mac.current_DL_BWP = &current_dl_bwp;
  mac.ul_config_request = &ul_config;
  mac.mib_ssb = 0;
  mac.ssb_list.nb_tx_ssb = 1;
  mac.ssb_list.nb_ssb_per_index[0] = 0;
  mac.ssb_ro_preambles = (ssb_ro_preambles_t){.ssb_per_ro = 1, .preambles_per_ssb = 4};
  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.p_Max = 23;
  mac.p_Max_alt = INT_MIN;
  mac.nr_band = 78;
  mac.frequency_range = FR1;
  mac.frame_structure.frame_type = FDD;
  mac.frame_structure.numb_slots_frame = 20;

  current_bwp.scs = msg1_scs;
  current_bwp.bwp_id = 0;
  current_bwp.channel_bandwidth = 40;
  current_bwp.rach_ConfigCommon = &rach_config_common;
  current_dl_bwp.bwp_id = 0;
  rach_config_common.msg1_SubcarrierSpacing = &msg1_scs;
  rach_config_common.rach_ConfigGeneric.prach_ConfigurationIndex = 0;
  rach_config_common.rach_ConfigGeneric.msg1_FDM = 0;
  ASSERT_EQ(0, pthread_mutex_init(&ul_config.mutex_ul_config, NULL));

  init_RA(&mac);
  mac.ra.ra_state = nrRA_WAIT_RAR;
  mac.ra.cfra = false;
  mac.ra.t_crnti = 0x1234;
  mac.ra.preambleTransMax = 4;
  mac.ra.prach_resources.preamble_tx_counter = 1;
  mac.ra.prach_resources.preamble_power_ramping_cnt = 1;
  nr_timer_setup(&mac.ra.response_window_timer, 5, 1);
  nr_timer_start(&mac.ra.response_window_timer);
  ul_config.frame = 30;
  ul_config.slot = 0;
  ul_config.number_pdus = 1;
  ul_config.ul_config_list[0].pdu_type = FAPI_NR_UL_CONFIG_TYPE_PUSCH;
  ul_config.ul_config_list[0].lock = &ul_config.mutex_ul_config;
  ul_config.ul_config_list[0].privateNBpdus = &ul_config.number_pdus;
  mac.ssb_measurements[0].ssb_rsrp_dBm = INT_MIN;

  nr_uplink_indication_t unavailable_msg3 = {.frame = 30, .slot = 0};
  nr_ue_ul_scheduler(&mac, &unavailable_msg3);
  EXPECT_EQ(0, ul_config.number_pdus);
  EXPECT_EQ(2, mac.ra.prach_resources.preamble_tx_counter);
  EXPECT_EQ(1, mac.ra.prach_resources.preamble_power_ramping_cnt);
  EXPECT_TRUE(nr_timer_is_active(&mac.ra.RA_backoff_timer));
  EXPECT_FALSE(nr_timer_is_active(&mac.ra.response_window_timer));
  EXPECT_FALSE(nr_timer_is_active(&mac.ra.contention_resolution_timer));
  EXPECT_EQ(0, mac.ra.t_crnti);

  nr_ue_ul_scheduler(&mac, &unavailable_msg3);
  EXPECT_EQ(2, mac.ra.prach_resources.preamble_tx_counter);

  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  update_mac_ul_timers(&mac);
  EXPECT_FALSE(nr_timer_is_active(&mac.ra.RA_backoff_timer));
  EXPECT_FALSE(mac.ra.defer_preamble_for_ssb_pathloss);
  EXPECT_EQ(nrRA_GENERATE_PREAMBLE, mac.ra.ra_state);
  EXPECT_EQ(2, mac.ra.prach_resources.preamble_tx_counter);
  EXPECT_EQ(2, mac.ra.prach_resources.preamble_power_ramping_cnt);
  EXPECT_EQ(0, pthread_mutex_destroy(&ul_config.mutex_ul_config));
}

TEST(test_ssb_pathloss_context, msg3_not_transmitted_ignores_cfra)
{
  NR_UE_MAC_INST_t mac = {0};
  mac.ra.cfra = true;
  mac.ra.ra_state = nrRA_WAIT_RAR;
  mac.ra.t_crnti = 0x1234;
  mac.ra.preambleTransMax = 4;
  mac.ra.prach_resources.preamble_tx_counter = 1;
  mac.ra.prach_resources.preamble_power_ramping_cnt = 1;
  nr_timer_setup(&mac.ra.response_window_timer, 5, 1);
  nr_timer_setup(&mac.ra.contention_resolution_timer, 5, 1);
  nr_timer_start(&mac.ra.response_window_timer);
  nr_timer_start(&mac.ra.contention_resolution_timer);

  nr_msg3_not_transmitted(&mac);

  EXPECT_EQ(nrRA_WAIT_RAR, mac.ra.ra_state);
  EXPECT_EQ(0x1234, mac.ra.t_crnti);
  EXPECT_EQ(1, mac.ra.prach_resources.preamble_tx_counter);
  EXPECT_EQ(1, mac.ra.prach_resources.preamble_power_ramping_cnt);
  EXPECT_TRUE(nr_timer_is_active(&mac.ra.response_window_timer));
  EXPECT_TRUE(nr_timer_is_active(&mac.ra.contention_resolution_timer));
  EXPECT_FALSE(nr_timer_is_active(&mac.ra.RA_backoff_timer));
}

TEST(test_init_ra, four_step_cbra)
{
  NR_UE_MAC_INST_t mac = {0};
  RA_config_t *ra = &mac.ra;
  NR_RACH_ConfigCommon_t nr_rach_ConfigCommon = {0};
  NR_RACH_ConfigGeneric_t rach_ConfigGeneric = {0};
  NR_RACH_ConfigDedicated_t rach_ConfigDedicated = {0};
  NR_UE_UL_BWP_t current_bwp;
  NR_UE_DL_BWP_t dl_bwp;
  mac.current_UL_BWP = &current_bwp;
  mac.current_DL_BWP = &dl_bwp;
  mac.mib_ssb = 0;
  long scs = 1;
  current_bwp.scs = scs;
  current_bwp.bwp_id = 0;
  dl_bwp.bwp_id = 0;
  current_bwp.channel_bandwidth = 40;
  nr_rach_ConfigCommon.msg1_SubcarrierSpacing = &scs;
  nr_rach_ConfigCommon.rach_ConfigGeneric = rach_ConfigGeneric;
  current_bwp.rach_ConfigCommon = &nr_rach_ConfigCommon;
  ra->rach_ConfigDedicated = &rach_ConfigDedicated;
  mac.p_Max = 23;
  mac.nr_band = 78;
  mac.frame_structure.frame_type = TDD;
  mac.frame_structure.numb_slots_frame = 20;
  mac.frequency_range = FR1;

  init_RA(&mac);

  EXPECT_EQ(mac.ra.ra_type, RA_4_STEP);
  EXPECT_EQ(mac.state, UE_PERFORMING_RA);
  EXPECT_EQ(mac.ra.RA_active, true);
  EXPECT_EQ(mac.ra.cfra, 0);
}

TEST(test_init_ra, four_step_cfra)
{
  NR_UE_MAC_INST_t mac = {0};
  RA_config_t *ra = &mac.ra;
  NR_RACH_ConfigCommon_t nr_rach_ConfigCommon = {0};
  NR_RACH_ConfigGeneric_t rach_ConfigGeneric = {0};
  NR_UE_UL_BWP_t current_bwp;
  NR_UE_DL_BWP_t dl_bwp;
  mac.current_UL_BWP = &current_bwp;
  mac.current_DL_BWP = &dl_bwp;
  mac.mib_ssb = 0;
  long scs = 1;
  current_bwp.scs = scs;
  current_bwp.bwp_id = 0;
  dl_bwp.bwp_id = 0;
  current_bwp.channel_bandwidth = 40;
  nr_rach_ConfigCommon.msg1_SubcarrierSpacing = &scs;
  nr_rach_ConfigCommon.rach_ConfigGeneric = rach_ConfigGeneric;
  current_bwp.rach_ConfigCommon = &nr_rach_ConfigCommon;
  mac.p_Max = 23;
  mac.nr_band = 78;
  mac.frame_structure.frame_type = TDD;
  mac.frame_structure.numb_slots_frame = 20;
  mac.frequency_range = FR1;

  NR_RACH_ConfigDedicated_t rach_ConfigDedicated = {0};
  NR_CFRA_t cfra;
  rach_ConfigDedicated.cfra = &cfra;
  ra->rach_ConfigDedicated = &rach_ConfigDedicated;

  init_RA(&mac);

  EXPECT_EQ(mac.ra.ra_type, RA_4_STEP);
  EXPECT_EQ(mac.state, UE_PERFORMING_RA);
  EXPECT_EQ(mac.ra.RA_active, true);
  EXPECT_EQ(mac.ra.cfra, 1);
}

static void check_accepted_rar_power_control(bool managed)
{
  NR_UE_MAC_INST_t mac = {0};
  NR_UE_UL_BWP_t ul_bwp = {0};
  NR_UE_DL_BWP_t dl_bwp = {0};
  fapi_nr_ul_config_request_t ul_config[20] = {0};
  NR_SearchSpace_t search_space = {0};
  NR_SearchSpace::NR_SearchSpace__searchSpaceType search_space_type = {};
  NR_SearchSpace_t *common_search_spaces[] = {&search_space};
  long control_resource_set_id = 0;
  uint8_t rar_buffer[sizeof(NR_RA_HEADER_RAPID) + sizeof(NR_MAC_RAR)] = {0};
  fapi_nr_rx_indication_t rx_indication = {0};
  nr_downlink_indication_t dl_info = {.frame = 10, .slot = 0, .rx_ind = &rx_indication};

  mac.current_UL_BWP = &ul_bwp;
  mac.current_DL_BWP = &dl_bwp;
  mac.ul_config_request = ul_config;
  mac.mib_ssb = 0;
  mac.ssb_measurements[0].ssb_rsrp_dBm = -86;
  mac.phy_config.config_req.ssb_config.ss_pbch_power = -25;
  mac.p_Max = 23;
  mac.p_Max_alt = INT_MIN;
  mac.nr_band = 78;
  mac.frequency_range = FR1;
  mac.frame_structure.frame_type = FDD;
  mac.frame_structure.numb_slots_frame = 20;
  mac.sc_info.initial_ul_BWPStart = 0;
  mac.sc_info.initial_ul_BWPSize = 106;
  mac.crnti = 0x4321;

  ul_bwp.scs = 1;
  ul_bwp.bwp_id = 0;
  ul_bwp.BWPSize = 106;
  ul_bwp.channel_bandwidth = 40;
  dl_bwp.bwp_id = 0;

  search_space.searchSpaceId = 1;
  search_space.controlResourceSetId = &control_resource_set_id;
  search_space.searchSpaceType = &search_space_type;
  search_space_type.present = NR_SearchSpace__searchSpaceType_PR_common;
  mac.config_BWP_PDCCH[0].ra_SS_id = 1;
  mac.config_BWP_PDCCH[0].list_common_SS.array = common_search_spaces;
  mac.config_BWP_PDCCH[0].list_common_SS.count = 1;
  mac.config_BWP_PDCCH[0].list_common_SS.size = 1;
  for (auto &config : ul_config)
    ASSERT_EQ(0, pthread_mutex_init(&config.mutex_ul_config, NULL));

  mac.ra.ra_type = RA_4_STEP;
  mac.ra.RA_active = true;
  mac.ra.ra_state = nrRA_WAIT_RAR;
  mac.ra.cfra = false;
  mac.ra.ra_PreambleIndex = 7;
  mac.ra.ra_rnti = 0x47;
  mac.ra.prach_resources.ra_preamble_rx_target_power = -70;
  mac.ra.prach_resources.preamble_power_ramping_cnt = 1;
  mac.ra.prach_resources.preamble_power_ramping_step = 2;
  mac.f_b_f_c = -17;
  mac.pusch_power_control_initialized = true;
  nr_timer_setup(&mac.ra.response_window_timer, 5, 1);
  nr_timer_start(&mac.ra.response_window_timer);

  NR_RA_HEADER_RAPID *header = reinterpret_cast<NR_RA_HEADER_RAPID *>(rar_buffer);
  header->RAPID = mac.ra.ra_PreambleIndex;
  header->T = 1;
  header->E = 0;
  NR_MAC_RAR *rar = reinterpret_cast<NR_MAC_RAR *>(rar_buffer + sizeof(*header));
  rar->UL_GRANT_4 = (managed ? 5 : 3) << 1; // RAR TPC: command 5 is +4 dB; command 3 is 0 dB.
  rar->TCRNTI_1 = 0x12;
  rar->TCRNTI_2 = 0x34;

  rx_indication.number_pdus = 1;
  rx_indication.rx_indication_body[0].pdu_type = FAPI_NR_RX_PDU_TYPE_RAR;
  rx_indication.rx_indication_body[0].pdsch_pdu.ack_nack = 1;
  rx_indication.rx_indication_body[0].pdsch_pdu.pdu = rar_buffer;
  rx_indication.rx_indication_body[0].pdsch_pdu.pdu_length = sizeof(rar_buffer);

  nr_ue_send_sdu(&mac, &dl_info, 0);

  fapi_nr_ul_config_request_pdu_t *msg3 = NULL;
  for (auto &config : ul_config) {
    if (config.number_pdus == 1 && config.ul_config_list[0].pdu_type == FAPI_NR_UL_CONFIG_TYPE_PUSCH)
      msg3 = &config.ul_config_list[0];
  }
  ASSERT_NE(nullptr, msg3);
  if (managed) {
    EXPECT_FALSE(mac.pusch_power_control_initialized);
    EXPECT_EQ(-17, mac.f_b_f_c);
    EXPECT_TRUE(msg3->pusch_config_pdu.oai_deferred_tx_power);
    EXPECT_TRUE(msg3->pusch_config_pdu.oai_deferred_is_rar_tx_retx);
    EXPECT_EQ(4, msg3->pusch_config_pdu.oai_deferred_tpc_delta);
    mac.ra.Msg3_TPC = -6; // A later RA state change cannot replace the queued command.
    ASSERT_TRUE(nr_ue_apply_deferred_pusch_tx_power(&mac, &msg3->pusch_config_pdu, mac.ul_pusch_config_generation));
    EXPECT_TRUE(mac.pusch_power_control_initialized);
    EXPECT_EQ(4, mac.f_b_f_c);
    EXPECT_EQ(4, mac.delta_msg2);
    EXPECT_FALSE(nr_ue_apply_deferred_pusch_tx_power(&mac, &msg3->pusch_config_pdu, mac.ul_pusch_config_generation));
    EXPECT_EQ(4, mac.f_b_f_c);

    // Model a queued TC-RNTI retransmission with its own decoded +1 dB DCI command.
    nfapi_nr_ue_pusch_pdu_t retransmission = msg3->pusch_config_pdu;
    retransmission.oai_deferred_tx_power = 1;
    retransmission.oai_deferred_tpc_delta = 1;
    ASSERT_TRUE(nr_ue_apply_deferred_pusch_tx_power(&mac, &retransmission, mac.ul_pusch_config_generation));
    EXPECT_EQ(5, mac.f_b_f_c);
    EXPECT_EQ(4, mac.delta_msg2);
    EXPECT_FALSE(nr_ue_apply_deferred_pusch_tx_power(&mac, &retransmission, mac.ul_pusch_config_generation));
    EXPECT_EQ(5, mac.f_b_f_c);
  } else {
    EXPECT_TRUE(mac.pusch_power_control_initialized);
    EXPECT_EQ(0, mac.f_b_f_c);
    EXPECT_EQ(0, mac.delta_msg2);
  }
  // A retransmission must retain accumulated closed-loop state rather than reapply the first-Msg3 ramp.
  mac.f_b_f_c = 5;
  const int retransmission_f = mac.f_b_f_c;
  EXPECT_NE(INT_MIN,
            get_pusch_tx_power_ue(&mac,
                                  msg3->pusch_config_pdu.rb_size,
                                  msg3->pusch_config_pdu.rb_start,
                                  msg3->pusch_config_pdu.nr_of_symbols,
                                  0,
                                  0,
                                  msg3->pusch_config_pdu.qam_mod_order,
                                  msg3->pusch_config_pdu.target_code_rate,
                                  0,
                                  msg3->pusch_config_pdu.pusch_data.tb_size << 3,
                                  0,
                                  true,
                                  false));
  EXPECT_EQ(retransmission_f, mac.f_b_f_c);

  mac.ra.ra_state = nrRA_WAIT_RAR;
  mac.ra.cfra = true;
  mac.f_b_f_c = -17;
  mac.pusch_power_control_initialized = true;
  nr_timer_start(&mac.ra.response_window_timer);
  nr_ue_send_sdu(&mac, &dl_info, 0);
  EXPECT_TRUE(mac.pusch_power_control_initialized);
  EXPECT_EQ(-17, mac.f_b_f_c);

  for (auto &config : ul_config)
    EXPECT_EQ(0, pthread_mutex_destroy(&config.mutex_ul_config));
}

TEST(test_msg3_power_control, accepted_rar_reinitializes_cbra_only)
{
  check_accepted_rar_power_control(false);
}

TEST(test_msg3_power_control, accepted_nonzero_rar_is_consumed_once)
{
  test_agc_options.tx_actuation = true;
  check_accepted_rar_power_control(true);
  // ASSERT failures return from the helper, so the accessor is restored for later tests.
  test_agc_options.tx_actuation = false;
}

int main(int argc, char **argv)
{
  logInit();
  configmodule_interface_t *uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY);
  g_log->log_component[MAC].level = OAILOG_DEBUG;
  g_log->log_component[NR_MAC].level = OAILOG_DEBUG;
  testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  end_configmodule(uniqCfg);
  return ret;
}
