#include "ran_func_kpm.h"
#include "openair2/E2AP/flexric/test/rnd/fill_rnd_data_kpm.h"
#include "openair2/E2AP/flexric/src/util/time_now_us.h"
#include "common/ran_context.h"
#include "openair2/LAYER2/NR_MAC_gNB/mac_proto.h"
#include "openair2/RRC/NR/rrc_gNB_UE_context.h"
#include "openair2/E2AP/RAN_FUNCTION/CUSTOMIZED/ran_func_rlc.h"
#include <assert.h>
#include <stdio.h>

static
const int mod_id = 0;

static 
gnb_e2sm_t fill_gnb_data(uint16_t rnti)
{
  gnb_e2sm_t gnb = {0};

  // 6.2.3.16
  // Mandatory
  // AMF UE NGAP ID
  // fill with openair3/NGAP/ngap_gNB_ue_context.h:61
  struct rrc_gNB_ue_context_s *ue_context_p = rrc_gNB_get_ue_context_by_rnti(RC.nrrrc[mod_id], rnti);
  gnb.amf_ue_ngap_id = ue_context_p->ue_context.gNB_ue_ngap_id;

  // Mandatory
  //GUAMI 6.2.3.17 
  gnb.guami.plmn_id = (e2sm_plmn_t) {
                                    .mcc = ue_context_p->ue_context.ue_guami.mcc,
                                    .mnc = ue_context_p->ue_context.ue_guami.mnc,
                                    .mnc_digit_len = ue_context_p->ue_context.ue_guami.mnc_len
                                    };
  
  gnb.guami.amf_region_id = ue_context_p->ue_context.ue_guami.amf_region_id;
  gnb.guami.amf_set_id = ue_context_p->ue_context.ue_guami.amf_set_id;
  gnb.guami.amf_ptr = ue_context_p->ue_context.ue_guami.amf_pointer;

  return gnb;
}

static
    gnb_e2sm_t fill_rnd_gnb_data(void)
{
  gnb_e2sm_t gnb = {0};
  gnb.amf_ue_ngap_id = (rand() % 2^40) + 0;
  gnb.guami.plmn_id = (e2sm_plmn_t) {.mcc = 505, .mnc = 1, .mnc_digit_len = 2};
  gnb.guami.amf_region_id = (rand() % 2^8) + 0;
  gnb.guami.amf_set_id = (rand() % 2^10) + 0;
  gnb.guami.amf_ptr = (rand() % 2^6) + 0;
  return gnb;
}

static 
ue_id_e2sm_t fill_ue_id_data(uint16_t rnti)
{
  ue_id_e2sm_t ue_id_data = {0};

  ue_id_data.type = GNB_UE_ID_E2SM;
  ue_id_data.gnb = fill_gnb_data(rnti);

  return ue_id_data;
}

static
ue_id_e2sm_t fill_rnd_ue_id_data()
{
  ue_id_e2sm_t ue_id_data = {0};
  ue_id_data.type = GNB_UE_ID_E2SM;
  ue_id_data.gnb = fill_rnd_gnb_data();
  return ue_id_data;
}

void copy_ba_to_str(const uint8_t* buffer, size_t length, char* str) {
  memcpy(str, buffer, length);
  str[length] = '\0';
}

double dl_thr_st_val[MAX_MOBILES_PER_GNB] = {0};
double dl_thr_avg_val[MAX_MOBILES_PER_GNB] = {0};
int dl_thr_count[MAX_MOBILES_PER_GNB] = {0};
void cal_dl_thr_bps(uint64_t const dl_total_bytes, uint32_t const gran_period_ms, size_t const ue_idx) {
  size_t count_max = 1000/gran_period_ms;
  // DL
  if (dl_thr_count[ue_idx] == 0)
    dl_thr_st_val[ue_idx] = dl_total_bytes;
  dl_thr_count[ue_idx] += 1;
  if (dl_thr_count[ue_idx] == count_max) {
    dl_thr_avg_val[ue_idx] = (dl_total_bytes - dl_thr_st_val[ue_idx])*8;
    dl_thr_count[ue_idx] = 0;
  }
}

double ul_thr_st_val[MAX_MOBILES_PER_GNB] = {0};
double ul_thr_avg_val[MAX_MOBILES_PER_GNB] = {0};
int ul_thr_count[MAX_MOBILES_PER_GNB] = {0};
void cal_ul_thr_bps(uint64_t const ul_total_bytes, uint32_t const gran_period_ms, size_t const ue_idx) {
  size_t count_max = 1000/gran_period_ms;
  // UL
  if (ul_thr_count[ue_idx] == 0)
    ul_thr_st_val[ue_idx] = ul_total_bytes;
  ul_thr_avg_val[ue_idx] += 1;
  if (ul_thr_count[ue_idx] == count_max) {
    ul_thr_avg_val[ue_idx] = (ul_total_bytes - ul_thr_st_val[ue_idx])*8;
    ul_thr_count[ue_idx] = 0;
  }
}


static 
kpm_ind_msg_format_1_t fill_kpm_ind_msg_frm_1(NR_UE_info_t* const UE, size_t const ue_idx, kpm_act_def_format_1_t const act_def_fr1)
{
  kpm_ind_msg_format_1_t msg_frm_1 = {0};

  // Measurement Data
  NR_UEs_t *UE_info = &RC.nrmac[mod_id]->UE_info;
  uint32_t num_drbs = num_act_rb(UE_info);
  msg_frm_1.meas_data_lst_len = num_drbs;  // (rand() % 65535) + 1;
  msg_frm_1.meas_data_lst = calloc(msg_frm_1.meas_data_lst_len, sizeof(*msg_frm_1.meas_data_lst));
  assert(msg_frm_1.meas_data_lst != NULL && "Memory exhausted" );


  size_t const rec_data_len = act_def_fr1.meas_info_lst_len; // Recoding Data Length
  size_t meas_data_idx = 0;
  for (int rb_id = 1; rb_id < 6; ++rb_id) {
    nr_rlc_statistics_t rb_rlc = {0};
    const int srb_flag = 0;
    const bool rc = nr_rlc_get_statistics(UE->rnti, srb_flag, rb_id, &rb_rlc);
    if(!rc) continue;

    // Measurement Record
    meas_data_lst_t* meas_data = &msg_frm_1.meas_data_lst[meas_data_idx];
    meas_data->meas_record_len = rec_data_len;
    meas_data->meas_record_lst = calloc(meas_data->meas_record_len, sizeof(meas_record_lst_t));
    assert(meas_data->meas_record_lst != NULL && "Memory exhausted");
    for (size_t j = 0; j < meas_data->meas_record_len; j++) {
      meas_record_lst_t* meas_record = &meas_data->meas_record_lst[j];
      meas_type_t meas_info_type = act_def_fr1.meas_info_lst[j].meas_type;
      if (meas_info_type.type == NAME_MEAS_TYPE) {
        // Get Meas Info Name from Action Definition
        size_t length = meas_info_type.name.len;
        char meas_info_name_str[length + 1];
        copy_ba_to_str(meas_info_type.name.buf, length, meas_info_name_str);

        // Get value based on Meas Info Name
        if (strcmp(meas_info_name_str, "DRB.IPThpDl.QCI") == 0 ? true : false) {
          meas_record->value = REAL_MEAS_VALUE;
          cal_dl_thr_bps(UE->mac_stats.dl.total_bytes, act_def_fr1.gran_period_ms, ue_idx);
          meas_record->real_val = dl_thr_avg_val[ue_idx];
        } else if (strcmp(meas_info_name_str, "DRB.IPThpUl.QCI") == 0 ? true : false) {
          meas_record->value = REAL_MEAS_VALUE;
          cal_ul_thr_bps(UE->mac_stats.ul.total_bytes, act_def_fr1.gran_period_ms, ue_idx);
          meas_record->real_val = ul_thr_avg_val[ue_idx];
        } else if (strcmp(meas_info_name_str, "DRB.RlcSduDelayDl") == 0 ? true : false) {
          meas_record->value = REAL_MEAS_VALUE;
          active_avg_to_tx(UE_info);
          meas_record->real_val = rb_rlc.txsdu_avg_time_to_tx;
        } else {
          printf("not implement value for measurement name %s\n", meas_info_name_str);
          meas_record->value = REAL_MEAS_VALUE;
          meas_record->real_val = 1234;
        }
      } else {
        printf("not implement value for measurement type %d\n", meas_info_type.type);
        meas_record->value = REAL_MEAS_VALUE;
        meas_record->real_val = 1234;
      }
    }
    meas_data_idx++;
  }

  // Measurement Information - OPTIONAL
  msg_frm_1.meas_info_lst_len = rec_data_len;
  msg_frm_1.meas_info_lst = calloc(msg_frm_1.meas_info_lst_len, sizeof(meas_info_format_1_lst_t));
  assert(msg_frm_1.meas_info_lst != NULL && "Memory exhausted" );

  // Get measInfo from action definition
  for (size_t i = 0; i < msg_frm_1.meas_info_lst_len; i++) {
    // Measurement Type
    msg_frm_1.meas_info_lst[i].meas_type.type = act_def_fr1.meas_info_lst[i].meas_type.type;
    // Measurement Name
    if (act_def_fr1.meas_info_lst[i].meas_type.type == NAME_MEAS_TYPE) {
      msg_frm_1.meas_info_lst[i].meas_type.name.buf = calloc(act_def_fr1.meas_info_lst[i].meas_type.name.len, sizeof(uint8_t));
      memcpy(msg_frm_1.meas_info_lst[i].meas_type.name.buf, act_def_fr1.meas_info_lst[i].meas_type.name.buf, act_def_fr1.meas_info_lst[i].meas_type.name.len);
      msg_frm_1.meas_info_lst[i].meas_type.name.len = act_def_fr1.meas_info_lst[i].meas_type.name.len;
    } else {
      msg_frm_1.meas_info_lst[i].meas_type.id = act_def_fr1.meas_info_lst[i].meas_type.id;
    }


    // Label Information
    msg_frm_1.meas_info_lst[i].label_info_lst_len = 1;
    msg_frm_1.meas_info_lst[i].label_info_lst = calloc(msg_frm_1.meas_info_lst[i].label_info_lst_len, sizeof(label_info_lst_t));
    assert(msg_frm_1.meas_info_lst[i].label_info_lst != NULL && "Memory exhausted" );

    for (size_t j = 0; j < msg_frm_1.meas_info_lst[i].label_info_lst_len; j++) {
      msg_frm_1.meas_info_lst[i].label_info_lst[j].noLabel = malloc(sizeof(enum_value_e));
      *msg_frm_1.meas_info_lst[i].label_info_lst[j].noLabel = TRUE_ENUM_VALUE;
    }
  }

  return msg_frm_1;
}

static kpm_ind_msg_format_1_t fill_rnd_kpm_ind_msg_frm_1(void)
{
  kpm_ind_msg_format_1_t msg_frm_1 = {0};

  // Measurement Data
  msg_frm_1.meas_data_lst_len = 1;  // (rand() % 65535) + 1;
  msg_frm_1.meas_data_lst = calloc(msg_frm_1.meas_data_lst_len, sizeof(*msg_frm_1.meas_data_lst));
  assert(msg_frm_1.meas_data_lst != NULL && "Memory exhausted" );

  for (size_t i = 0; i < msg_frm_1.meas_data_lst_len; i++)
  {
    // Incomplete Flag
    msg_frm_1.meas_data_lst[i].incomplete_flag = calloc(1, sizeof(enum_value_e));
    assert(msg_frm_1.meas_data_lst[i].incomplete_flag != NULL && "Memory exhausted");
    *msg_frm_1.meas_data_lst[i].incomplete_flag = TRUE_ENUM_VALUE;

    // Measurement Record
    msg_frm_1.meas_data_lst[i].meas_record_len = 1;
    msg_frm_1.meas_data_lst[i].meas_record_lst = calloc(msg_frm_1.meas_data_lst[i].meas_record_len, sizeof(meas_record_lst_t));
    assert(msg_frm_1.meas_data_lst[i].meas_record_lst != NULL && "Memory exhausted" );

    for (size_t j = 0; j < msg_frm_1.meas_data_lst[i].meas_record_len; j++)
    {
      msg_frm_1.meas_data_lst[i].meas_record_lst[j].value = REAL_MEAS_VALUE;
      msg_frm_1.meas_data_lst[i].meas_record_lst[j].real_val = time_now_us();
    }
  }

  // Measurement Information - OPTIONAL
  msg_frm_1.meas_info_lst_len = 1;
  msg_frm_1.meas_info_lst = calloc(msg_frm_1.meas_info_lst_len, sizeof(meas_info_format_1_lst_t));
  assert(msg_frm_1.meas_info_lst != NULL && "Memory exhausted" );

  for (size_t i = 0; i < msg_frm_1.meas_info_lst_len; i++)
  {
    // Measurement Type
    char* s = "timestamp";
    msg_frm_1.meas_info_lst[i].meas_type.name.len = strlen(s) + 1;
    msg_frm_1.meas_info_lst[i].meas_type.name.buf = malloc(strlen(s) + 1);
    assert(msg_frm_1.meas_info_lst[i].meas_type.name.buf != NULL && "memory exhausted");
    memcpy(msg_frm_1.meas_info_lst[i].meas_type.name.buf, s, strlen(s));
    msg_frm_1.meas_info_lst[i].meas_type.name.buf[strlen(s)] = '\0';

    // Label Information
    msg_frm_1.meas_info_lst[i].label_info_lst_len = 1;
    msg_frm_1.meas_info_lst[i].label_info_lst = calloc(msg_frm_1.meas_info_lst[i].label_info_lst_len, sizeof(label_info_lst_t));
    assert(msg_frm_1.meas_info_lst[i].label_info_lst != NULL && "Memory exhausted" );

    for (size_t j = 0; j < msg_frm_1.meas_info_lst[i].label_info_lst_len; j++)
    {
      msg_frm_1.meas_info_lst[i].label_info_lst[j].noLabel = malloc(sizeof(enum_value_e));
      *msg_frm_1.meas_info_lst[i].label_info_lst[j].noLabel = TRUE_ENUM_VALUE;
    }
  }

  return msg_frm_1;
}

static
kpm_ind_msg_format_3_t fill_kpm_ind_msg_frm_3_sta(kpm_act_def_format_1_t const act_def_fr1)
{
  kpm_ind_msg_format_3_t msg_frm_3 = {0};

  // get the number of connected UEs
  NR_UEs_t *UE_info = &RC.nrmac[mod_id]->UE_info;
  uint32_t num_ues = 0;
  UE_iterator(UE_info->list, ue) {
    if (ue)
      num_ues += 1;
  }

  msg_frm_3.ue_meas_report_lst_len = num_ues == 0 ? 1: num_ues;  // (rand() % 65535) + 1;

  msg_frm_3.meas_report_per_ue = calloc(msg_frm_3.ue_meas_report_lst_len, sizeof(meas_report_per_ue_t));
  assert(msg_frm_3.meas_report_per_ue != NULL && "Memory exhausted");

  // TODO: handle the case when the number of UE is 0, if (msg_frm_3.ue_meas_report_lst_len > 0)
  if (num_ues > 0) {
    size_t ue_idx = 0;
    UE_iterator(UE_info->list, UE)
    {
      uint16_t const rnti = UE->rnti;
      msg_frm_3.meas_report_per_ue[ue_idx].ue_meas_report_lst = fill_ue_id_data(rnti);
      msg_frm_3.meas_report_per_ue[ue_idx].ind_msg_format_1 = fill_kpm_ind_msg_frm_1(UE, ue_idx, act_def_fr1);
      ue_idx+=1;
    }
  } else {
    msg_frm_3.meas_report_per_ue[0].ue_meas_report_lst = fill_rnd_ue_id_data();
    msg_frm_3.meas_report_per_ue[0].ind_msg_format_1 = fill_rnd_kpm_ind_msg_frm_1();
  }


  return msg_frm_3;
}

static 
kpm_ric_ind_hdr_format_1_t fill_kpm_ind_hdr_frm_1(void)
{
  kpm_ric_ind_hdr_format_1_t hdr_frm_1 = {0};

  hdr_frm_1.collectStartTime = time_now_us();
  
  hdr_frm_1.fileformat_version = NULL;
  
  hdr_frm_1.sender_name = calloc(1, sizeof(byte_array_t));
  hdr_frm_1.sender_name->buf = calloc(strlen("My OAI-MONO") + 1, sizeof(char));
  memcpy(hdr_frm_1.sender_name->buf, "My OAI-MONO", strlen("My OAI-MONO"));
  hdr_frm_1.sender_name->len = strlen("My OAI-MONO");
  
  hdr_frm_1.sender_type = calloc(1, sizeof(byte_array_t));
  hdr_frm_1.sender_type->buf = calloc(strlen("MONO") + 1, sizeof(char));
  memcpy(hdr_frm_1.sender_type->buf, "MONO", strlen("MONO"));
  hdr_frm_1.sender_type->len = strlen("MONO");
  
  hdr_frm_1.vendor_name = calloc(1, sizeof(byte_array_t));
  hdr_frm_1.vendor_name->buf = calloc(strlen("OAI") + 1, sizeof(char));
  memcpy(hdr_frm_1.vendor_name->buf, "OAI", strlen("OAI"));
  hdr_frm_1.vendor_name->len = strlen("OAI");

  return hdr_frm_1;
}

static
kpm_ind_hdr_t fill_kpm_ind_hdr_sta(void)
{
  kpm_ind_hdr_t hdr = {0};

  hdr.type = FORMAT_1_INDICATION_HEADER;
  hdr.kpm_ric_ind_hdr_format_1 = fill_kpm_ind_hdr_frm_1();

  return hdr;
}

void read_kpm_sm(void* data)
{
  assert(data != NULL);
  //assert(data->type == KPM_STATS_V3_0);

  kpm_rd_ind_data_t* const kpm = (kpm_rd_ind_data_t*)data;

  assert(kpm->act_def!= NULL && "Cannot be NULL");
  if(kpm->act_def->type == FORMAT_4_ACTION_DEFINITION){

    if(kpm->act_def->frm_4.matching_cond_lst[0].test_info_lst.test_cond_type == CQI_TEST_COND_TYPE
        && *kpm->act_def->frm_4.matching_cond_lst[0].test_info_lst.test_cond == GREATERTHAN_TEST_COND){
      printf("Matching condition: UEs with CQI greater than %ld \n", *kpm->act_def->frm_4.matching_cond_lst[0].test_info_lst.int_value );
    }

    for (size_t i = 0; i < kpm->act_def->frm_4.action_def_format_1.meas_info_lst_len; i++)
      printf("Parameter to report: %s \n", kpm->act_def->frm_4.action_def_format_1.meas_info_lst[i].meas_type.name.buf);

    kpm->ind.hdr = fill_kpm_ind_hdr_sta(); 
    // 7.8 Supported RIC Styles and E2SM IE Formats
    // Format 4 corresponds to indication message 3
    kpm->ind.msg.type = FORMAT_3_INDICATION_MESSAGE;
    kpm->ind.msg.frm_3 = fill_kpm_ind_msg_frm_3_sta(kpm->act_def->frm_4.action_def_format_1);
  } else {
     kpm->ind.hdr = fill_kpm_ind_hdr(); 
     kpm->ind.msg = fill_kpm_ind_msg(); 
  }
}

void read_kpm_setup_sm(void* e2ap)
{
  assert(e2ap != NULL);
//  assert(e2ap->type == KPM_V3_0_AGENT_IF_E2_SETUP_ANS_V0);

  kpm_e2_setup_t* kpm = (kpm_e2_setup_t*)(e2ap);
  kpm->ran_func_def = fill_kpm_ran_func_def(); 
}

sm_ag_if_ans_t write_ctrl_kpm_sm(void const* src)
{
  assert(0 !=0 && "Not supported");
  (void)src;
  sm_ag_if_ans_t ans = {0};
  return ans;
}

