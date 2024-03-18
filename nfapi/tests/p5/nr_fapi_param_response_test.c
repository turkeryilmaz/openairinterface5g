#include "nr_fapi_test.h"
void test_param_response_tlv(fapi_nr_param_response_scf_t unpacked_req, fapi_nr_param_response_scf_t req)
{
  printf(".cell_param.release_capability.tl.tag: 0x%02x\n", unpacked_req.cell_param.release_capability.tl.tag);
  AssertFatal(unpacked_req.cell_param.release_capability.tl.tag == req.cell_param.release_capability.tl.tag,
              ".cell_param.release_capability.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.cell_param.release_capability.tl.tag,
              req.cell_param.release_capability.tl.tag);

  printf(".cell_param.release_capability.value: 0x%02x\n", unpacked_req.cell_param.release_capability.value);
  AssertFatal(unpacked_req.cell_param.release_capability.value == req.cell_param.release_capability.value,
              ".cell_param.release_capability.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.cell_param.release_capability.value,
              req.cell_param.release_capability.value);

  printf(".cell_param.phy_state.tl.tag: 0x%02x\n", unpacked_req.cell_param.phy_state.tl.tag);
  AssertFatal(unpacked_req.cell_param.phy_state.tl.tag == req.cell_param.phy_state.tl.tag,
              ".cell_param.phy_state.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was "
              "0x%02x\n",
              unpacked_req.cell_param.phy_state.tl.tag,
              req.cell_param.phy_state.tl.tag);

  printf(".cell_param.phy_state.value: 0x%02x\n", unpacked_req.cell_param.phy_state.value);
  AssertFatal(
      unpacked_req.cell_param.phy_state.value == req.cell_param.phy_state.value,
      ".cell_param.phy_state.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
      unpacked_req.cell_param.phy_state.value,
      req.cell_param.phy_state.value);

  printf(".cell_param.skip_blank_dl_config.tl.tag: 0x%02x\n", unpacked_req.cell_param.skip_blank_dl_config.tl.tag);
  AssertFatal(unpacked_req.cell_param.skip_blank_dl_config.tl.tag == req.cell_param.skip_blank_dl_config.tl.tag,
              ".cell_param.skip_blank_dl_config.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.cell_param.skip_blank_dl_config.tl.tag,
              req.cell_param.skip_blank_dl_config.tl.tag);

  printf(".cell_param.skip_blank_dl_config.value: 0x%02x\n", unpacked_req.cell_param.skip_blank_dl_config.value);
  AssertFatal(unpacked_req.cell_param.skip_blank_dl_config.value == req.cell_param.skip_blank_dl_config.value,
              ".cell_param.skip_blank_dl_config.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.cell_param.skip_blank_dl_config.value,
              req.cell_param.skip_blank_dl_config.value);

  printf(".cell_param.skip_blank_ul_config.tl.tag: 0x%02x\n", unpacked_req.cell_param.skip_blank_ul_config.tl.tag);
  AssertFatal(unpacked_req.cell_param.skip_blank_ul_config.tl.tag == req.cell_param.skip_blank_ul_config.tl.tag,
              ".cell_param.skip_blank_ul_config.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.cell_param.skip_blank_ul_config.tl.tag,
              req.cell_param.skip_blank_ul_config.tl.tag);

  printf(".cell_param.skip_blank_ul_config.value: 0x%02x\n", unpacked_req.cell_param.skip_blank_ul_config.value);
  AssertFatal(unpacked_req.cell_param.skip_blank_ul_config.value == req.cell_param.skip_blank_ul_config.value,
              ".cell_param.skip_blank_ul_config.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.cell_param.skip_blank_ul_config.value,
              req.cell_param.skip_blank_ul_config.value);

  printf(".cell_param.num_config_tlvs_to_report.tl.tag: 0x%02x\n", unpacked_req.cell_param.num_config_tlvs_to_report.tl.tag);
  AssertFatal(unpacked_req.cell_param.num_config_tlvs_to_report.tl.tag == req.cell_param.num_config_tlvs_to_report.tl.tag,
              ".cell_param.num_config_tlvs_to_report.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.cell_param.num_config_tlvs_to_report.tl.tag,
              req.cell_param.num_config_tlvs_to_report.tl.tag);

  printf(".cell_param.num_config_tlvs_to_report.value: 0x%02x\n", unpacked_req.cell_param.num_config_tlvs_to_report.value);
  AssertFatal(unpacked_req.cell_param.num_config_tlvs_to_report.value == req.cell_param.num_config_tlvs_to_report.value,
              ".cell_param.num_config_tlvs_to_report.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.cell_param.num_config_tlvs_to_report.value,
              req.cell_param.num_config_tlvs_to_report.value);

  printf(".carrier_param.cyclic_prefix.tl.tag: 0x%02x\n", unpacked_req.carrier_param.cyclic_prefix.tl.tag);
  AssertFatal(unpacked_req.carrier_param.cyclic_prefix.tl.tag == req.carrier_param.cyclic_prefix.tl.tag,
              ".carrier_param.cyclic_prefix.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.carrier_param.cyclic_prefix.tl.tag,
              req.carrier_param.cyclic_prefix.tl.tag);

  printf(".carrier_param.cyclic_prefix.value: 0x%02x\n", unpacked_req.carrier_param.cyclic_prefix.value);
  AssertFatal(unpacked_req.carrier_param.cyclic_prefix.value == req.carrier_param.cyclic_prefix.value,
              ".carrier_param.cyclic_prefix.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.carrier_param.cyclic_prefix.value,
              req.carrier_param.cyclic_prefix.value);

  printf(".carrier_param.supported_subcarrier_spacings_dl.tl.tag: 0x%02x\n",
         unpacked_req.carrier_param.supported_subcarrier_spacings_dl.tl.tag);
  AssertFatal(unpacked_req.carrier_param.supported_subcarrier_spacings_dl.tl.tag
                  == req.carrier_param.supported_subcarrier_spacings_dl.tl.tag,
              ".carrier_param.supported_subcarrier_spacings_dl.tl.tag was not the same as the packed value! Unpacked value was "
              "0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.carrier_param.supported_subcarrier_spacings_dl.tl.tag,
              req.carrier_param.supported_subcarrier_spacings_dl.tl.tag);

  printf(".carrier_param.supported_subcarrier_spacings_dl.value: 0x%02x\n",
         unpacked_req.carrier_param.supported_subcarrier_spacings_dl.value);
  AssertFatal(
      unpacked_req.carrier_param.supported_subcarrier_spacings_dl.value == req.carrier_param.supported_subcarrier_spacings_dl.value,
      ".carrier_param.supported_subcarrier_spacings_dl.value was not the same as the packed value! Unpacked value was 0x%02x , and "
      "Packed value was 0x%02x\n",
      unpacked_req.carrier_param.supported_subcarrier_spacings_dl.value,
      req.carrier_param.supported_subcarrier_spacings_dl.value);

  printf(".carrier_param.supported_bandwidth_dl.tl.tag: 0x%02x\n", unpacked_req.carrier_param.supported_bandwidth_dl.tl.tag);
  AssertFatal(unpacked_req.carrier_param.supported_bandwidth_dl.tl.tag == req.carrier_param.supported_bandwidth_dl.tl.tag,
              ".carrier_param.supported_bandwidth_dl.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.carrier_param.supported_bandwidth_dl.tl.tag,
              req.carrier_param.supported_bandwidth_dl.tl.tag);

  printf(".carrier_param.supported_bandwidth_dl.value: 0x%02x\n", unpacked_req.carrier_param.supported_bandwidth_dl.value);
  AssertFatal(unpacked_req.carrier_param.supported_bandwidth_dl.value == req.carrier_param.supported_bandwidth_dl.value,
              ".carrier_param.supported_bandwidth_dl.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.carrier_param.supported_bandwidth_dl.value,
              req.carrier_param.supported_bandwidth_dl.value);

  printf(".carrier_param.supported_subcarrier_spacings_ul.tl.tag: 0x%02x\n",
         unpacked_req.carrier_param.supported_subcarrier_spacings_ul.tl.tag);
  AssertFatal(unpacked_req.carrier_param.supported_subcarrier_spacings_ul.tl.tag
                  == req.carrier_param.supported_subcarrier_spacings_ul.tl.tag,
              ".carrier_param.supported_subcarrier_spacings_ul.tl.tag was not the same as the packed value! Unpacked value was "
              "0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.carrier_param.supported_subcarrier_spacings_ul.tl.tag,
              req.carrier_param.supported_subcarrier_spacings_ul.tl.tag);

  printf(".carrier_param.supported_subcarrier_spacings_ul.value: 0x%02x\n",
         unpacked_req.carrier_param.supported_subcarrier_spacings_ul.value);
  AssertFatal(
      unpacked_req.carrier_param.supported_subcarrier_spacings_ul.value == req.carrier_param.supported_subcarrier_spacings_ul.value,
      ".carrier_param.supported_subcarrier_spacings_ul.value was not the same as the packed value! Unpacked value was 0x%02x , and "
      "Packed value was 0x%02x\n",
      unpacked_req.carrier_param.supported_subcarrier_spacings_ul.value,
      req.carrier_param.supported_subcarrier_spacings_ul.value);

  printf(".carrier_param.supported_bandwidth_ul.tl.tag: 0x%02x\n", unpacked_req.carrier_param.supported_bandwidth_ul.tl.tag);
  AssertFatal(unpacked_req.carrier_param.supported_bandwidth_ul.tl.tag == req.carrier_param.supported_bandwidth_ul.tl.tag,
              ".carrier_param.supported_bandwidth_ul.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.carrier_param.supported_bandwidth_ul.tl.tag,
              req.carrier_param.supported_bandwidth_ul.tl.tag);

  printf(".carrier_param.supported_bandwidth_ul.value: 0x%02x\n", unpacked_req.carrier_param.supported_bandwidth_ul.value);
  AssertFatal(unpacked_req.carrier_param.supported_bandwidth_ul.value == req.carrier_param.supported_bandwidth_ul.value,
              ".carrier_param.supported_bandwidth_ul.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.carrier_param.supported_bandwidth_ul.value,
              req.carrier_param.supported_bandwidth_ul.value);

  printf(".pdcch_param.cce_mapping_type.tl.tag: 0x%02x\n", unpacked_req.pdcch_param.cce_mapping_type.tl.tag);
  AssertFatal(unpacked_req.pdcch_param.cce_mapping_type.tl.tag == req.pdcch_param.cce_mapping_type.tl.tag,
              ".pdcch_param.cce_mapping_type.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdcch_param.cce_mapping_type.tl.tag,
              req.pdcch_param.cce_mapping_type.tl.tag);

  printf(".pdcch_param.cce_mapping_type.value: 0x%02x\n", unpacked_req.pdcch_param.cce_mapping_type.value);
  AssertFatal(unpacked_req.pdcch_param.cce_mapping_type.value == req.pdcch_param.cce_mapping_type.value,
              ".pdcch_param.cce_mapping_type.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdcch_param.cce_mapping_type.value,
              req.pdcch_param.cce_mapping_type.value);

  printf(".pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag: 0x%02x\n",
         unpacked_req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag);
  AssertFatal(unpacked_req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag
                  == req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag,
              ".pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag was not the same as the packed value! Unpacked "
              "value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag,
              req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag);

  printf(".pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value: 0x%02x\n",
         unpacked_req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value);
  AssertFatal(unpacked_req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value
                  == req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value,
              ".pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value was not the same as the packed value! Unpacked "
              "value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value,
              req.pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value);

  printf(".pdcch_param.coreset_precoder_granularity_coreset.tl.tag: 0x%02x\n",
         unpacked_req.pdcch_param.coreset_precoder_granularity_coreset.tl.tag);
  AssertFatal(unpacked_req.pdcch_param.coreset_precoder_granularity_coreset.tl.tag
                  == req.pdcch_param.coreset_precoder_granularity_coreset.tl.tag,
              ".pdcch_param.coreset_precoder_granularity_coreset.tl.tag was not the same as the packed value! Unpacked value was "
              "0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.pdcch_param.coreset_precoder_granularity_coreset.tl.tag,
              req.pdcch_param.coreset_precoder_granularity_coreset.tl.tag);

  printf(".pdcch_param.coreset_precoder_granularity_coreset.value: 0x%02x\n",
         unpacked_req.pdcch_param.coreset_precoder_granularity_coreset.value);
  AssertFatal(unpacked_req.pdcch_param.coreset_precoder_granularity_coreset.value
                  == req.pdcch_param.coreset_precoder_granularity_coreset.value,
              ".pdcch_param.coreset_precoder_granularity_coreset.value was not the same as the packed value! Unpacked value was "
              "0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.pdcch_param.coreset_precoder_granularity_coreset.value,
              req.pdcch_param.coreset_precoder_granularity_coreset.value);

  printf(".pdcch_param.pdcch_mu_mimo.tl.tag: 0x%02x\n", unpacked_req.pdcch_param.pdcch_mu_mimo.tl.tag);
  AssertFatal(unpacked_req.pdcch_param.pdcch_mu_mimo.tl.tag == req.pdcch_param.pdcch_mu_mimo.tl.tag,
              ".pdcch_param.pdcch_mu_mimo.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdcch_param.pdcch_mu_mimo.tl.tag,
              req.pdcch_param.pdcch_mu_mimo.tl.tag);

  printf(".pdcch_param.pdcch_mu_mimo.value: 0x%02x\n", unpacked_req.pdcch_param.pdcch_mu_mimo.value);
  AssertFatal(unpacked_req.pdcch_param.pdcch_mu_mimo.value == req.pdcch_param.pdcch_mu_mimo.value,
              ".pdcch_param.pdcch_mu_mimo.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed value "
              "was 0x%02x\n",
              unpacked_req.pdcch_param.pdcch_mu_mimo.value,
              req.pdcch_param.pdcch_mu_mimo.value);

  printf(".pdcch_param.pdcch_precoder_cycling.tl.tag: 0x%02x\n", unpacked_req.pdcch_param.pdcch_precoder_cycling.tl.tag);
  AssertFatal(unpacked_req.pdcch_param.pdcch_precoder_cycling.tl.tag == req.pdcch_param.pdcch_precoder_cycling.tl.tag,
              ".pdcch_param.pdcch_precoder_cycling.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdcch_param.pdcch_precoder_cycling.tl.tag,
              req.pdcch_param.pdcch_precoder_cycling.tl.tag);

  printf(".pdcch_param.pdcch_precoder_cycling.value: 0x%02x\n", unpacked_req.pdcch_param.pdcch_precoder_cycling.value);
  AssertFatal(unpacked_req.pdcch_param.pdcch_precoder_cycling.value == req.pdcch_param.pdcch_precoder_cycling.value,
              ".pdcch_param.pdcch_precoder_cycling.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdcch_param.pdcch_precoder_cycling.value,
              req.pdcch_param.pdcch_precoder_cycling.value);

  printf(".pdcch_param.max_pdcch_per_slot.tl.tag: 0x%02x\n", unpacked_req.pdcch_param.max_pdcch_per_slot.tl.tag);
  AssertFatal(unpacked_req.pdcch_param.max_pdcch_per_slot.tl.tag == req.pdcch_param.max_pdcch_per_slot.tl.tag,
              ".pdcch_param.max_pdcch_per_slot.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdcch_param.max_pdcch_per_slot.tl.tag,
              req.pdcch_param.max_pdcch_per_slot.tl.tag);

  printf(".pdcch_param.max_pdcch_per_slot.value: 0x%02x\n", unpacked_req.pdcch_param.max_pdcch_per_slot.value);
  AssertFatal(unpacked_req.pdcch_param.max_pdcch_per_slot.value == req.pdcch_param.max_pdcch_per_slot.value,
              ".pdcch_param.max_pdcch_per_slot.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdcch_param.max_pdcch_per_slot.value,
              req.pdcch_param.max_pdcch_per_slot.value);

  printf(".pucch_param.pucch_formats.tl.tag: 0x%02x\n", unpacked_req.pucch_param.pucch_formats.tl.tag);
  AssertFatal(unpacked_req.pucch_param.pucch_formats.tl.tag == req.pucch_param.pucch_formats.tl.tag,
              ".pucch_param.pucch_formats.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pucch_param.pucch_formats.tl.tag,
              req.pucch_param.pucch_formats.tl.tag);

  printf(".pucch_param.pucch_formats.value: 0x%02x\n", unpacked_req.pucch_param.pucch_formats.value);
  AssertFatal(unpacked_req.pucch_param.pucch_formats.value == req.pucch_param.pucch_formats.value,
              ".pucch_param.pucch_formats.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed value "
              "was 0x%02x\n",
              unpacked_req.pucch_param.pucch_formats.value,
              req.pucch_param.pucch_formats.value);

  printf(".pucch_param.max_pucchs_per_slot.tl.tag: 0x%02x\n", unpacked_req.pucch_param.max_pucchs_per_slot.tl.tag);
  AssertFatal(unpacked_req.pucch_param.max_pucchs_per_slot.tl.tag == req.pucch_param.max_pucchs_per_slot.tl.tag,
              ".pucch_param.max_pucchs_per_slot.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pucch_param.max_pucchs_per_slot.tl.tag,
              req.pucch_param.max_pucchs_per_slot.tl.tag);

  printf(".pucch_param.max_pucchs_per_slot.value: 0x%02x\n", unpacked_req.pucch_param.max_pucchs_per_slot.value);
  AssertFatal(unpacked_req.pucch_param.max_pucchs_per_slot.value == req.pucch_param.max_pucchs_per_slot.value,
              ".pucch_param.max_pucchs_per_slot.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pucch_param.max_pucchs_per_slot.value,
              req.pucch_param.max_pucchs_per_slot.value);

  printf(".pdsch_param.pdsch_mapping_type.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_mapping_type.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_mapping_type.tl.tag == req.pdsch_param.pdsch_mapping_type.tl.tag,
              ".pdsch_param.pdsch_mapping_type.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_mapping_type.tl.tag,
              req.pdsch_param.pdsch_mapping_type.tl.tag);

  printf(".pdsch_param.pdsch_mapping_type.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_mapping_type.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_mapping_type.value == req.pdsch_param.pdsch_mapping_type.value,
              ".pdsch_param.pdsch_mapping_type.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_mapping_type.value,
              req.pdsch_param.pdsch_mapping_type.value);

  printf(".pdsch_param.pdsch_dmrs_additional_pos.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_dmrs_additional_pos.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_dmrs_additional_pos.tl.tag == req.pdsch_param.pdsch_dmrs_additional_pos.tl.tag,
              ".pdsch_param.pdsch_dmrs_additional_pos.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_dmrs_additional_pos.tl.tag,
              req.pdsch_param.pdsch_dmrs_additional_pos.tl.tag);

  printf(".pdsch_param.pdsch_dmrs_additional_pos.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_dmrs_additional_pos.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_dmrs_additional_pos.value == req.pdsch_param.pdsch_dmrs_additional_pos.value,
              ".pdsch_param.pdsch_dmrs_additional_pos.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_dmrs_additional_pos.value,
              req.pdsch_param.pdsch_dmrs_additional_pos.value);

  printf(".pdsch_param.pdsch_allocation_types.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_allocation_types.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_allocation_types.tl.tag == req.pdsch_param.pdsch_allocation_types.tl.tag,
              ".pdsch_param.pdsch_allocation_types.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_allocation_types.tl.tag,
              req.pdsch_param.pdsch_allocation_types.tl.tag);

  printf(".pdsch_param.pdsch_allocation_types.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_allocation_types.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_allocation_types.value == req.pdsch_param.pdsch_allocation_types.value,
              ".pdsch_param.pdsch_allocation_types.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_allocation_types.value,
              req.pdsch_param.pdsch_allocation_types.value);

  printf(".pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag == req.pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag,
              ".pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag,
              req.pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag);

  printf(".pdsch_param.pdsch_vrb_to_prb_mapping.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_vrb_to_prb_mapping.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_vrb_to_prb_mapping.value == req.pdsch_param.pdsch_vrb_to_prb_mapping.value,
              ".pdsch_param.pdsch_vrb_to_prb_mapping.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_vrb_to_prb_mapping.value,
              req.pdsch_param.pdsch_vrb_to_prb_mapping.value);

  printf(".pdsch_param.pdsch_cbg.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_cbg.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_cbg.tl.tag == req.pdsch_param.pdsch_cbg.tl.tag,
              ".pdsch_param.pdsch_cbg.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed value "
              "was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_cbg.tl.tag,
              req.pdsch_param.pdsch_cbg.tl.tag);

  printf(".pdsch_param.pdsch_cbg.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_cbg.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_cbg.value == req.pdsch_param.pdsch_cbg.value,
              ".pdsch_param.pdsch_cbg.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was "
              "0x%02x\n",
              unpacked_req.pdsch_param.pdsch_cbg.value,
              req.pdsch_param.pdsch_cbg.value);

  printf(".pdsch_param.pdsch_dmrs_config_types.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_dmrs_config_types.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_dmrs_config_types.tl.tag == req.pdsch_param.pdsch_dmrs_config_types.tl.tag,
              ".pdsch_param.pdsch_dmrs_config_types.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_dmrs_config_types.tl.tag,
              req.pdsch_param.pdsch_dmrs_config_types.tl.tag);

  printf(".pdsch_param.pdsch_dmrs_config_types.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_dmrs_config_types.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_dmrs_config_types.value == req.pdsch_param.pdsch_dmrs_config_types.value,
              ".pdsch_param.pdsch_dmrs_config_types.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_dmrs_config_types.value,
              req.pdsch_param.pdsch_dmrs_config_types.value);

  printf(".pdsch_param.max_number_mimo_layers_pdsch.tl.tag: 0x%02x\n",
         unpacked_req.pdsch_param.max_number_mimo_layers_pdsch.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.max_number_mimo_layers_pdsch.tl.tag == req.pdsch_param.max_number_mimo_layers_pdsch.tl.tag,
              ".pdsch_param.max_number_mimo_layers_pdsch.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , "
              "and Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.max_number_mimo_layers_pdsch.tl.tag,
              req.pdsch_param.max_number_mimo_layers_pdsch.tl.tag);

  printf(".pdsch_param.max_number_mimo_layers_pdsch.value: 0x%02x\n", unpacked_req.pdsch_param.max_number_mimo_layers_pdsch.value);
  AssertFatal(unpacked_req.pdsch_param.max_number_mimo_layers_pdsch.value == req.pdsch_param.max_number_mimo_layers_pdsch.value,
              ".pdsch_param.max_number_mimo_layers_pdsch.value was not the same as the packed value! Unpacked value was 0x%02x , "
              "and Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.max_number_mimo_layers_pdsch.value,
              req.pdsch_param.max_number_mimo_layers_pdsch.value);

  printf(".pdsch_param.max_mu_mimo_users_dl.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.max_mu_mimo_users_dl.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.max_mu_mimo_users_dl.tl.tag == req.pdsch_param.max_mu_mimo_users_dl.tl.tag,
              ".pdsch_param.max_mu_mimo_users_dl.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.max_mu_mimo_users_dl.tl.tag,
              req.pdsch_param.max_mu_mimo_users_dl.tl.tag);

  printf(".pdsch_param.max_mu_mimo_users_dl.value: 0x%02x\n", unpacked_req.pdsch_param.max_mu_mimo_users_dl.value);
  AssertFatal(unpacked_req.pdsch_param.max_mu_mimo_users_dl.value == req.pdsch_param.max_mu_mimo_users_dl.value,
              ".pdsch_param.max_mu_mimo_users_dl.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.max_mu_mimo_users_dl.value,
              req.pdsch_param.max_mu_mimo_users_dl.value);

  printf(".pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag == req.pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag,
              ".pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , "
              "and Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag,
              req.pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag);

  printf(".pdsch_param.pdsch_data_in_dmrs_symbols.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_data_in_dmrs_symbols.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_data_in_dmrs_symbols.value == req.pdsch_param.pdsch_data_in_dmrs_symbols.value,
              ".pdsch_param.pdsch_data_in_dmrs_symbols.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_data_in_dmrs_symbols.value,
              req.pdsch_param.pdsch_data_in_dmrs_symbols.value);

  printf(".pdsch_param.premption_support.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.premption_support.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.premption_support.tl.tag == req.pdsch_param.premption_support.tl.tag,
              ".pdsch_param.premption_support.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdsch_param.premption_support.tl.tag,
              req.pdsch_param.premption_support.tl.tag);

  printf(".pdsch_param.premption_support.value: 0x%02x\n", unpacked_req.pdsch_param.premption_support.value);
  AssertFatal(unpacked_req.pdsch_param.premption_support.value == req.pdsch_param.premption_support.value,
              ".pdsch_param.premption_support.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pdsch_param.premption_support.value,
              req.pdsch_param.premption_support.value);

  printf(".pdsch_param.pdsch_non_slot_support.tl.tag: 0x%02x\n", unpacked_req.pdsch_param.pdsch_non_slot_support.tl.tag);
  AssertFatal(unpacked_req.pdsch_param.pdsch_non_slot_support.tl.tag == req.pdsch_param.pdsch_non_slot_support.tl.tag,
              ".pdsch_param.pdsch_non_slot_support.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_non_slot_support.tl.tag,
              req.pdsch_param.pdsch_non_slot_support.tl.tag);

  printf(".pdsch_param.pdsch_non_slot_support.value: 0x%02x\n", unpacked_req.pdsch_param.pdsch_non_slot_support.value);
  AssertFatal(unpacked_req.pdsch_param.pdsch_non_slot_support.value == req.pdsch_param.pdsch_non_slot_support.value,
              ".pdsch_param.pdsch_non_slot_support.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pdsch_param.pdsch_non_slot_support.value,
              req.pdsch_param.pdsch_non_slot_support.value);

  printf(".pusch_param.uci_mux_ulsch_in_pusch.tl.tag: 0x%02x\n", unpacked_req.pusch_param.uci_mux_ulsch_in_pusch.tl.tag);
  AssertFatal(unpacked_req.pusch_param.uci_mux_ulsch_in_pusch.tl.tag == req.pusch_param.uci_mux_ulsch_in_pusch.tl.tag,
              ".pusch_param.uci_mux_ulsch_in_pusch.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.uci_mux_ulsch_in_pusch.tl.tag,
              req.pusch_param.uci_mux_ulsch_in_pusch.tl.tag);

  printf(".pusch_param.uci_mux_ulsch_in_pusch.value: 0x%02x\n", unpacked_req.pusch_param.uci_mux_ulsch_in_pusch.value);
  AssertFatal(unpacked_req.pusch_param.uci_mux_ulsch_in_pusch.value == req.pusch_param.uci_mux_ulsch_in_pusch.value,
              ".pusch_param.uci_mux_ulsch_in_pusch.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.uci_mux_ulsch_in_pusch.value,
              req.pusch_param.uci_mux_ulsch_in_pusch.value);

  printf(".pusch_param.uci_only_pusch.tl.tag: 0x%02x\n", unpacked_req.pusch_param.uci_only_pusch.tl.tag);
  AssertFatal(unpacked_req.pusch_param.uci_only_pusch.tl.tag == req.pusch_param.uci_only_pusch.tl.tag,
              ".pusch_param.uci_only_pusch.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.uci_only_pusch.tl.tag,
              req.pusch_param.uci_only_pusch.tl.tag);

  printf(".pusch_param.uci_only_pusch.value: 0x%02x\n", unpacked_req.pusch_param.uci_only_pusch.value);
  AssertFatal(unpacked_req.pusch_param.uci_only_pusch.value == req.pusch_param.uci_only_pusch.value,
              ".pusch_param.uci_only_pusch.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.uci_only_pusch.value,
              req.pusch_param.uci_only_pusch.value);

  printf(".pusch_param.pusch_frequency_hopping.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_frequency_hopping.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_frequency_hopping.tl.tag == req.pusch_param.pusch_frequency_hopping.tl.tag,
              ".pusch_param.pusch_frequency_hopping.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_frequency_hopping.tl.tag,
              req.pusch_param.pusch_frequency_hopping.tl.tag);

  printf(".pusch_param.pusch_frequency_hopping.value: 0x%02x\n", unpacked_req.pusch_param.pusch_frequency_hopping.value);
  AssertFatal(unpacked_req.pusch_param.pusch_frequency_hopping.value == req.pusch_param.pusch_frequency_hopping.value,
              ".pusch_param.pusch_frequency_hopping.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_frequency_hopping.value,
              req.pusch_param.pusch_frequency_hopping.value);

  printf(".pusch_param.pusch_dmrs_config_types.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_dmrs_config_types.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_dmrs_config_types.tl.tag == req.pusch_param.pusch_dmrs_config_types.tl.tag,
              ".pusch_param.pusch_dmrs_config_types.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_dmrs_config_types.tl.tag,
              req.pusch_param.pusch_dmrs_config_types.tl.tag);

  printf(".pusch_param.pusch_dmrs_config_types.value: 0x%02x\n", unpacked_req.pusch_param.pusch_dmrs_config_types.value);
  AssertFatal(unpacked_req.pusch_param.pusch_dmrs_config_types.value == req.pusch_param.pusch_dmrs_config_types.value,
              ".pusch_param.pusch_dmrs_config_types.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_dmrs_config_types.value,
              req.pusch_param.pusch_dmrs_config_types.value);

  printf(".pusch_param.pusch_dmrs_max_len.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_dmrs_max_len.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_dmrs_max_len.tl.tag == req.pusch_param.pusch_dmrs_max_len.tl.tag,
              ".pusch_param.pusch_dmrs_max_len.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_dmrs_max_len.tl.tag,
              req.pusch_param.pusch_dmrs_max_len.tl.tag);

  printf(".pusch_param.pusch_dmrs_max_len.value: 0x%02x\n", unpacked_req.pusch_param.pusch_dmrs_max_len.value);
  AssertFatal(unpacked_req.pusch_param.pusch_dmrs_max_len.value == req.pusch_param.pusch_dmrs_max_len.value,
              ".pusch_param.pusch_dmrs_max_len.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_dmrs_max_len.value,
              req.pusch_param.pusch_dmrs_max_len.value);

  printf(".pusch_param.pusch_dmrs_additional_pos.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_dmrs_additional_pos.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_dmrs_additional_pos.tl.tag == req.pusch_param.pusch_dmrs_additional_pos.tl.tag,
              ".pusch_param.pusch_dmrs_additional_pos.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_dmrs_additional_pos.tl.tag,
              req.pusch_param.pusch_dmrs_additional_pos.tl.tag);

  printf(".pusch_param.pusch_dmrs_additional_pos.value: 0x%02x\n", unpacked_req.pusch_param.pusch_dmrs_additional_pos.value);
  AssertFatal(unpacked_req.pusch_param.pusch_dmrs_additional_pos.value == req.pusch_param.pusch_dmrs_additional_pos.value,
              ".pusch_param.pusch_dmrs_additional_pos.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_dmrs_additional_pos.value,
              req.pusch_param.pusch_dmrs_additional_pos.value);

  printf(".pusch_param.pusch_cbg.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_cbg.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_cbg.tl.tag == req.pusch_param.pusch_cbg.tl.tag,
              ".pusch_param.pusch_cbg.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed value "
              "was 0x%02x\n",
              unpacked_req.pusch_param.pusch_cbg.tl.tag,
              req.pusch_param.pusch_cbg.tl.tag);

  printf(".pusch_param.pusch_cbg.value: 0x%02x\n", unpacked_req.pusch_param.pusch_cbg.value);
  AssertFatal(unpacked_req.pusch_param.pusch_cbg.value == req.pusch_param.pusch_cbg.value,
              ".pusch_param.pusch_cbg.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was "
              "0x%02x\n",
              unpacked_req.pusch_param.pusch_cbg.value,
              req.pusch_param.pusch_cbg.value);

  printf(".pusch_param.pusch_mapping_type.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_mapping_type.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_mapping_type.tl.tag == req.pusch_param.pusch_mapping_type.tl.tag,
              ".pusch_param.pusch_mapping_type.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_mapping_type.tl.tag,
              req.pusch_param.pusch_mapping_type.tl.tag);

  printf(".pusch_param.pusch_mapping_type.value: 0x%02x\n", unpacked_req.pusch_param.pusch_mapping_type.value);
  AssertFatal(unpacked_req.pusch_param.pusch_mapping_type.value == req.pusch_param.pusch_mapping_type.value,
              ".pusch_param.pusch_mapping_type.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_mapping_type.value,
              req.pusch_param.pusch_mapping_type.value);

  printf(".pusch_param.pusch_allocation_types.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_allocation_types.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_allocation_types.tl.tag == req.pusch_param.pusch_allocation_types.tl.tag,
              ".pusch_param.pusch_allocation_types.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_allocation_types.tl.tag,
              req.pusch_param.pusch_allocation_types.tl.tag);

  printf(".pusch_param.pusch_allocation_types.value: 0x%02x\n", unpacked_req.pusch_param.pusch_allocation_types.value);
  AssertFatal(unpacked_req.pusch_param.pusch_allocation_types.value == req.pusch_param.pusch_allocation_types.value,
              ".pusch_param.pusch_allocation_types.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_allocation_types.value,
              req.pusch_param.pusch_allocation_types.value);

  printf(".pusch_param.pusch_vrb_to_prb_mapping.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_vrb_to_prb_mapping.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_vrb_to_prb_mapping.tl.tag == req.pusch_param.pusch_vrb_to_prb_mapping.tl.tag,
              ".pusch_param.pusch_vrb_to_prb_mapping.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_vrb_to_prb_mapping.tl.tag,
              req.pusch_param.pusch_vrb_to_prb_mapping.tl.tag);

  printf(".pusch_param.pusch_vrb_to_prb_mapping.value: 0x%02x\n", unpacked_req.pusch_param.pusch_vrb_to_prb_mapping.value);
  AssertFatal(unpacked_req.pusch_param.pusch_vrb_to_prb_mapping.value == req.pusch_param.pusch_vrb_to_prb_mapping.value,
              ".pusch_param.pusch_vrb_to_prb_mapping.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_vrb_to_prb_mapping.value,
              req.pusch_param.pusch_vrb_to_prb_mapping.value);

  printf(".pusch_param.pusch_max_ptrs_ports.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_max_ptrs_ports.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_max_ptrs_ports.tl.tag == req.pusch_param.pusch_max_ptrs_ports.tl.tag,
              ".pusch_param.pusch_max_ptrs_ports.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_max_ptrs_ports.tl.tag,
              req.pusch_param.pusch_max_ptrs_ports.tl.tag);

  printf(".pusch_param.pusch_max_ptrs_ports.value: 0x%02x\n", unpacked_req.pusch_param.pusch_max_ptrs_ports.value);
  AssertFatal(unpacked_req.pusch_param.pusch_max_ptrs_ports.value == req.pusch_param.pusch_max_ptrs_ports.value,
              ".pusch_param.pusch_max_ptrs_ports.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_max_ptrs_ports.value,
              req.pusch_param.pusch_max_ptrs_ports.value);

  printf(".pusch_param.max_pduschs_tbs_per_slot.tl.tag: 0x%02x\n", unpacked_req.pusch_param.max_pduschs_tbs_per_slot.tl.tag);
  AssertFatal(unpacked_req.pusch_param.max_pduschs_tbs_per_slot.tl.tag == req.pusch_param.max_pduschs_tbs_per_slot.tl.tag,
              ".pusch_param.max_pduschs_tbs_per_slot.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.max_pduschs_tbs_per_slot.tl.tag,
              req.pusch_param.max_pduschs_tbs_per_slot.tl.tag);

  printf(".pusch_param.max_pduschs_tbs_per_slot.value: 0x%02x\n", unpacked_req.pusch_param.max_pduschs_tbs_per_slot.value);
  AssertFatal(unpacked_req.pusch_param.max_pduschs_tbs_per_slot.value == req.pusch_param.max_pduschs_tbs_per_slot.value,
              ".pusch_param.max_pduschs_tbs_per_slot.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.max_pduschs_tbs_per_slot.value,
              req.pusch_param.max_pduschs_tbs_per_slot.value);

  printf(".pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag: 0x%02x\n",
         unpacked_req.pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag);
  AssertFatal(unpacked_req.pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag
                  == req.pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag,
              ".pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag was not the same as the packed value! Unpacked value was "
              "0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag,
              req.pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag);

  printf(".pusch_param.max_number_mimo_layers_non_cb_pusch.value: 0x%02x\n",
         unpacked_req.pusch_param.max_number_mimo_layers_non_cb_pusch.value);
  AssertFatal(unpacked_req.pusch_param.max_number_mimo_layers_non_cb_pusch.value
                  == req.pusch_param.max_number_mimo_layers_non_cb_pusch.value,
              ".pusch_param.max_number_mimo_layers_non_cb_pusch.value was not the same as the packed value! Unpacked value was "
              "0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.pusch_param.max_number_mimo_layers_non_cb_pusch.value,
              req.pusch_param.max_number_mimo_layers_non_cb_pusch.value);

  printf(".pusch_param.supported_modulation_order_ul.tl.tag: 0x%02x\n",
         unpacked_req.pusch_param.supported_modulation_order_ul.tl.tag);
  AssertFatal(unpacked_req.pusch_param.supported_modulation_order_ul.tl.tag == req.pusch_param.supported_modulation_order_ul.tl.tag,
              ".pusch_param.supported_modulation_order_ul.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , "
              "and Packed value was 0x%02x\n",
              unpacked_req.pusch_param.supported_modulation_order_ul.tl.tag,
              req.pusch_param.supported_modulation_order_ul.tl.tag);

  printf(".pusch_param.supported_modulation_order_ul.value: 0x%02x\n",
         unpacked_req.pusch_param.supported_modulation_order_ul.value);
  AssertFatal(unpacked_req.pusch_param.supported_modulation_order_ul.value == req.pusch_param.supported_modulation_order_ul.value,
              ".pusch_param.supported_modulation_order_ul.value was not the same as the packed value! Unpacked value was 0x%02x , "
              "and Packed value was 0x%02x\n",
              unpacked_req.pusch_param.supported_modulation_order_ul.value,
              req.pusch_param.supported_modulation_order_ul.value);

  printf(".pusch_param.max_mu_mimo_users_ul.tl.tag: 0x%02x\n", unpacked_req.pusch_param.max_mu_mimo_users_ul.tl.tag);
  AssertFatal(unpacked_req.pusch_param.max_mu_mimo_users_ul.tl.tag == req.pusch_param.max_mu_mimo_users_ul.tl.tag,
              ".pusch_param.max_mu_mimo_users_ul.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.max_mu_mimo_users_ul.tl.tag,
              req.pusch_param.max_mu_mimo_users_ul.tl.tag);

  printf(".pusch_param.max_mu_mimo_users_ul.value: 0x%02x\n", unpacked_req.pusch_param.max_mu_mimo_users_ul.value);
  AssertFatal(unpacked_req.pusch_param.max_mu_mimo_users_ul.value == req.pusch_param.max_mu_mimo_users_ul.value,
              ".pusch_param.max_mu_mimo_users_ul.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.max_mu_mimo_users_ul.value,
              req.pusch_param.max_mu_mimo_users_ul.value);

  printf(".pusch_param.dfts_ofdm_support.tl.tag: 0x%02x\n", unpacked_req.pusch_param.dfts_ofdm_support.tl.tag);
  AssertFatal(unpacked_req.pusch_param.dfts_ofdm_support.tl.tag == req.pusch_param.dfts_ofdm_support.tl.tag,
              ".pusch_param.dfts_ofdm_support.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.dfts_ofdm_support.tl.tag,
              req.pusch_param.dfts_ofdm_support.tl.tag);

  printf(".pusch_param.dfts_ofdm_support.value: 0x%02x\n", unpacked_req.pusch_param.dfts_ofdm_support.value);
  AssertFatal(unpacked_req.pusch_param.dfts_ofdm_support.value == req.pusch_param.dfts_ofdm_support.value,
              ".pusch_param.dfts_ofdm_support.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.pusch_param.dfts_ofdm_support.value,
              req.pusch_param.dfts_ofdm_support.value);

  printf(".pusch_param.pusch_aggregation_factor.tl.tag: 0x%02x\n", unpacked_req.pusch_param.pusch_aggregation_factor.tl.tag);
  AssertFatal(unpacked_req.pusch_param.pusch_aggregation_factor.tl.tag == req.pusch_param.pusch_aggregation_factor.tl.tag,
              ".pusch_param.pusch_aggregation_factor.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_aggregation_factor.tl.tag,
              req.pusch_param.pusch_aggregation_factor.tl.tag);

  printf(".pusch_param.pusch_aggregation_factor.value: 0x%02x\n", unpacked_req.pusch_param.pusch_aggregation_factor.value);
  AssertFatal(unpacked_req.pusch_param.pusch_aggregation_factor.value == req.pusch_param.pusch_aggregation_factor.value,
              ".pusch_param.pusch_aggregation_factor.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.pusch_param.pusch_aggregation_factor.value,
              req.pusch_param.pusch_aggregation_factor.value);

  printf(".prach_param.prach_long_formats.tl.tag: 0x%02x\n", unpacked_req.prach_param.prach_long_formats.tl.tag);
  AssertFatal(unpacked_req.prach_param.prach_long_formats.tl.tag == req.prach_param.prach_long_formats.tl.tag,
              ".prach_param.prach_long_formats.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.prach_param.prach_long_formats.tl.tag,
              req.prach_param.prach_long_formats.tl.tag);

  printf(".prach_param.prach_long_formats.value: 0x%02x\n", unpacked_req.prach_param.prach_long_formats.value);
  AssertFatal(unpacked_req.prach_param.prach_long_formats.value == req.prach_param.prach_long_formats.value,
              ".prach_param.prach_long_formats.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.prach_param.prach_long_formats.value,
              req.prach_param.prach_long_formats.value);

  printf(".prach_param.prach_short_formats.tl.tag: 0x%02x\n", unpacked_req.prach_param.prach_short_formats.tl.tag);
  AssertFatal(unpacked_req.prach_param.prach_short_formats.tl.tag == req.prach_param.prach_short_formats.tl.tag,
              ".prach_param.prach_short_formats.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.prach_param.prach_short_formats.tl.tag,
              req.prach_param.prach_short_formats.tl.tag);

  printf(".prach_param.prach_short_formats.value: 0x%02x\n", unpacked_req.prach_param.prach_short_formats.value);
  AssertFatal(unpacked_req.prach_param.prach_short_formats.value == req.prach_param.prach_short_formats.value,
              ".prach_param.prach_short_formats.value was not the same as the packed value! Unpacked value was 0x%02x , and Packed "
              "value was 0x%02x\n",
              unpacked_req.prach_param.prach_short_formats.value,
              req.prach_param.prach_short_formats.value);

  printf(".prach_param.prach_restricted_sets.tl.tag: 0x%02x\n", unpacked_req.prach_param.prach_restricted_sets.tl.tag);
  AssertFatal(unpacked_req.prach_param.prach_restricted_sets.tl.tag == req.prach_param.prach_restricted_sets.tl.tag,
              ".prach_param.prach_restricted_sets.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.prach_param.prach_restricted_sets.tl.tag,
              req.prach_param.prach_restricted_sets.tl.tag);

  printf(".prach_param.prach_restricted_sets.value: 0x%02x\n", unpacked_req.prach_param.prach_restricted_sets.value);
  AssertFatal(unpacked_req.prach_param.prach_restricted_sets.value == req.prach_param.prach_restricted_sets.value,
              ".prach_param.prach_restricted_sets.value was not the same as the packed value! Unpacked value was 0x%02x , and "
              "Packed value was 0x%02x\n",
              unpacked_req.prach_param.prach_restricted_sets.value,
              req.prach_param.prach_restricted_sets.value);

  printf(".prach_param.max_prach_fd_occasions_in_a_slot.tl.tag: 0x%02x\n",
         unpacked_req.prach_param.max_prach_fd_occasions_in_a_slot.tl.tag);
  AssertFatal(
      unpacked_req.prach_param.max_prach_fd_occasions_in_a_slot.tl.tag == req.prach_param.max_prach_fd_occasions_in_a_slot.tl.tag,
      ".prach_param.max_prach_fd_occasions_in_a_slot.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
      "Packed value was 0x%02x\n",
      unpacked_req.prach_param.max_prach_fd_occasions_in_a_slot.tl.tag,
      req.prach_param.max_prach_fd_occasions_in_a_slot.tl.tag);

  printf(".prach_param.max_prach_fd_occasions_in_a_slot.value: 0x%02x\n",
         unpacked_req.prach_param.max_prach_fd_occasions_in_a_slot.value);
  AssertFatal(
      unpacked_req.prach_param.max_prach_fd_occasions_in_a_slot.value == req.prach_param.max_prach_fd_occasions_in_a_slot.value,
      ".prach_param.max_prach_fd_occasions_in_a_slot.value was not the same as the packed value! Unpacked value was 0x%02x , and "
      "Packed value was 0x%02x\n",
      unpacked_req.prach_param.max_prach_fd_occasions_in_a_slot.value,
      req.prach_param.max_prach_fd_occasions_in_a_slot.value);

  printf(".measurement_param.rssi_measurement_support.tl.tag: 0x%02x\n",
         unpacked_req.measurement_param.rssi_measurement_support.tl.tag);
  AssertFatal(
      unpacked_req.measurement_param.rssi_measurement_support.tl.tag == req.measurement_param.rssi_measurement_support.tl.tag,
      ".measurement_param.rssi_measurement_support.tl.tag was not the same as the packed value! Unpacked value was 0x%02x , and "
      "Packed value was 0x%02x\n",
      unpacked_req.measurement_param.rssi_measurement_support.tl.tag,
      req.measurement_param.rssi_measurement_support.tl.tag);

  printf(".measurement_param.rssi_measurement_support.value: 0x%02x\n",
         unpacked_req.measurement_param.rssi_measurement_support.value);
  AssertFatal(unpacked_req.measurement_param.rssi_measurement_support.value == req.measurement_param.rssi_measurement_support.value,
              ".measurement_param.rssi_measurement_support.value was not the same as the packed value! Unpacked value was 0x%02x , "
              "and Packed value was 0x%02x\n",
              unpacked_req.measurement_param.rssi_measurement_support.value,
              req.measurement_param.rssi_measurement_support.value);
}

void fill_param_response_tlv(fapi_nr_param_response_scf_t *nfapi_resp)
{
  nfapi_resp->cell_param.release_capability.tl.tag = NFAPI_NR_PARAM_TLV_RELEASE_CAPABILITY_TAG;
  nfapi_resp->cell_param.release_capability.value = rand16();
  nfapi_resp->num_tlv++;

  nfapi_resp->cell_param.phy_state.tl.tag = NFAPI_NR_PARAM_TLV_PHY_STATE_TAG;
  nfapi_resp->cell_param.phy_state.value = rand16();
  nfapi_resp->num_tlv++;

  nfapi_resp->cell_param.skip_blank_dl_config.tl.tag = NFAPI_NR_PARAM_TLV_SKIP_BLANK_DL_CONFIG_TAG;
  nfapi_resp->cell_param.skip_blank_dl_config.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->cell_param.skip_blank_ul_config.tl.tag = NFAPI_NR_PARAM_TLV_SKIP_BLANK_UL_CONFIG_TAG;
  nfapi_resp->cell_param.skip_blank_ul_config.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->cell_param.num_config_tlvs_to_report.tl.tag = NFAPI_NR_PARAM_TLV_NUM_CONFIG_TLVS_TO_REPORT_TAG;
  nfapi_resp->cell_param.num_config_tlvs_to_report.value = rand16();
  nfapi_resp->num_tlv++;

  nfapi_resp->carrier_param.cyclic_prefix.tl.tag = NFAPI_NR_PARAM_TLV_CYCLIC_PREFIX_TAG;
  nfapi_resp->carrier_param.cyclic_prefix.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->carrier_param.supported_subcarrier_spacings_dl.tl.tag = NFAPI_NR_PARAM_TLV_SUPPORTED_SUBCARRIER_SPACINGS_DL_TAG;
  nfapi_resp->carrier_param.supported_subcarrier_spacings_dl.value = rand16();
  nfapi_resp->num_tlv++;

  nfapi_resp->carrier_param.supported_bandwidth_dl.tl.tag = NFAPI_NR_PARAM_TLV_SUPPORTED_BANDWIDTH_DL_TAG;
  nfapi_resp->carrier_param.supported_bandwidth_dl.value = rand16();
  nfapi_resp->num_tlv++;

  nfapi_resp->carrier_param.supported_subcarrier_spacings_ul.tl.tag = NFAPI_NR_PARAM_TLV_SUPPORTED_SUBCARRIER_SPACINGS_UL_TAG;
  nfapi_resp->carrier_param.supported_subcarrier_spacings_ul.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->carrier_param.supported_bandwidth_ul.tl.tag = NFAPI_NR_PARAM_TLV_SUPPORTED_BANDWIDTH_UL_TAG;
  nfapi_resp->carrier_param.supported_bandwidth_ul.value = rand16();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdcch_param.cce_mapping_type.tl.tag = NFAPI_NR_PARAM_TLV_CCE_MAPPING_TYPE_TAG;
  nfapi_resp->pdcch_param.cce_mapping_type.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.tl.tag =
      NFAPI_NR_PARAM_TLV_CORESET_OUTSIDE_FIRST_3_OFDM_SYMS_OF_SLOT_TAG;
  nfapi_resp->pdcch_param.coreset_outside_first_3_of_ofdm_syms_of_slot.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdcch_param.coreset_precoder_granularity_coreset.tl.tag = NFAPI_NR_PARAM_TLV_PRECODER_GRANULARITY_CORESET_TAG;
  nfapi_resp->pdcch_param.coreset_precoder_granularity_coreset.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdcch_param.pdcch_mu_mimo.tl.tag = NFAPI_NR_PARAM_TLV_PDCCH_MU_MIMO_TAG;
  nfapi_resp->pdcch_param.pdcch_mu_mimo.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdcch_param.pdcch_precoder_cycling.tl.tag = NFAPI_NR_PARAM_TLV_PDCCH_PRECODER_CYCLING_TAG;
  nfapi_resp->pdcch_param.pdcch_precoder_cycling.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdcch_param.max_pdcch_per_slot.tl.tag = NFAPI_NR_PARAM_TLV_MAX_PDCCHS_PER_SLOT_TAG;
  nfapi_resp->pdcch_param.max_pdcch_per_slot.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pucch_param.pucch_formats.tl.tag = NFAPI_NR_PARAM_TLV_PUCCH_FORMATS_TAG;
  nfapi_resp->pucch_param.pucch_formats.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pucch_param.max_pucchs_per_slot.tl.tag = NFAPI_NR_PARAM_TLV_MAX_PUCCHS_PER_SLOT_TAG;
  nfapi_resp->pucch_param.max_pucchs_per_slot.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_mapping_type.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_MAPPING_TYPE_TAG;
  nfapi_resp->pdsch_param.pdsch_mapping_type.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_dmrs_additional_pos.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_DMRS_ADDITIONAL_POS_TAG;
  nfapi_resp->pdsch_param.pdsch_dmrs_additional_pos.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_allocation_types.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_ALLOCATION_TYPES_TAG;
  nfapi_resp->pdsch_param.pdsch_allocation_types.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_vrb_to_prb_mapping.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_VRB_TO_PRB_MAPPING_TAG;
  nfapi_resp->pdsch_param.pdsch_vrb_to_prb_mapping.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_cbg.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_CBG_TAG;
  nfapi_resp->pdsch_param.pdsch_cbg.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_dmrs_config_types.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_DMRS_CONFIG_TYPES_TAG;
  nfapi_resp->pdsch_param.pdsch_dmrs_config_types.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.max_number_mimo_layers_pdsch.tl.tag = NFAPI_NR_PARAM_TLV_MAX_NUMBER_MIMO_LAYERS_PDSCH_TAG;
  nfapi_resp->pdsch_param.max_number_mimo_layers_pdsch.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.max_mu_mimo_users_dl.tl.tag = NFAPI_NR_PARAM_TLV_MAX_MU_MIMO_USERS_DL_TAG;
  nfapi_resp->pdsch_param.max_mu_mimo_users_dl.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_data_in_dmrs_symbols.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_DATA_IN_DMRS_SYMBOLS_TAG;
  nfapi_resp->pdsch_param.pdsch_data_in_dmrs_symbols.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.premption_support.tl.tag = NFAPI_NR_PARAM_TLV_PREMPTION_SUPPORT_TAG;
  nfapi_resp->pdsch_param.premption_support.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pdsch_param.pdsch_non_slot_support.tl.tag = NFAPI_NR_PARAM_TLV_PDSCH_NON_SLOT_SUPPORT_TAG;
  nfapi_resp->pdsch_param.pdsch_non_slot_support.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.uci_mux_ulsch_in_pusch.tl.tag = NFAPI_NR_PARAM_TLV_UCI_MUX_ULSCH_IN_PUSCH_TAG;
  nfapi_resp->pusch_param.uci_mux_ulsch_in_pusch.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.uci_only_pusch.tl.tag = NFAPI_NR_PARAM_TLV_UCI_ONLY_PUSCH_TAG;
  nfapi_resp->pusch_param.uci_only_pusch.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_frequency_hopping.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_FREQUENCY_HOPPING_TAG;
  nfapi_resp->pusch_param.pusch_frequency_hopping.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_dmrs_config_types.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_DMRS_CONFIG_TYPES_TAG;
  nfapi_resp->pusch_param.pusch_dmrs_config_types.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_dmrs_max_len.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_DMRS_MAX_LEN_TAG;
  nfapi_resp->pusch_param.pusch_dmrs_max_len.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_dmrs_additional_pos.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_DMRS_ADDITIONAL_POS_TAG;
  nfapi_resp->pusch_param.pusch_dmrs_additional_pos.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_cbg.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_CBG_TAG;
  nfapi_resp->pusch_param.pusch_cbg.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_mapping_type.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_MAPPING_TYPE_TAG;
  nfapi_resp->pusch_param.pusch_mapping_type.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_allocation_types.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_ALLOCATION_TYPES_TAG;
  nfapi_resp->pusch_param.pusch_allocation_types.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_vrb_to_prb_mapping.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_VRB_TO_PRB_MAPPING_TAG;
  nfapi_resp->pusch_param.pusch_vrb_to_prb_mapping.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_max_ptrs_ports.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_MAX_PTRS_PORTS_TAG;
  nfapi_resp->pusch_param.pusch_max_ptrs_ports.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.max_pduschs_tbs_per_slot.tl.tag = NFAPI_NR_PARAM_TLV_MAX_PDUSCHS_TBS_PER_SLOT_TAG;
  nfapi_resp->pusch_param.max_pduschs_tbs_per_slot.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.max_number_mimo_layers_non_cb_pusch.tl.tag = NFAPI_NR_PARAM_TLV_MAX_NUMBER_MIMO_LAYERS_NON_CB_PUSCH_TAG;
  nfapi_resp->pusch_param.max_number_mimo_layers_non_cb_pusch.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.supported_modulation_order_ul.tl.tag = NFAPI_NR_PARAM_TLV_SUPPORTED_MODULATION_ORDER_UL_TAG;
  nfapi_resp->pusch_param.supported_modulation_order_ul.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.max_mu_mimo_users_ul.tl.tag = NFAPI_NR_PARAM_TLV_MAX_MU_MIMO_USERS_UL_TAG;
  nfapi_resp->pusch_param.max_mu_mimo_users_ul.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.dfts_ofdm_support.tl.tag = NFAPI_NR_PARAM_TLV_DFTS_OFDM_SUPPORT_TAG;
  nfapi_resp->pusch_param.dfts_ofdm_support.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->pusch_param.pusch_aggregation_factor.tl.tag = NFAPI_NR_PARAM_TLV_PUSCH_AGGREGATION_FACTOR_TAG;
  nfapi_resp->pusch_param.pusch_aggregation_factor.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->prach_param.prach_long_formats.tl.tag = NFAPI_NR_PARAM_TLV_PRACH_LONG_FORMATS_TAG;
  nfapi_resp->prach_param.prach_long_formats.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->prach_param.prach_short_formats.tl.tag = NFAPI_NR_PARAM_TLV_PRACH_SHORT_FORMATS_TAG;
  nfapi_resp->prach_param.prach_short_formats.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->prach_param.prach_restricted_sets.tl.tag = NFAPI_NR_PARAM_TLV_PRACH_RESTRICTED_SETS_TAG;
  nfapi_resp->prach_param.prach_restricted_sets.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->prach_param.max_prach_fd_occasions_in_a_slot.tl.tag = NFAPI_NR_PARAM_TLV_MAX_PRACH_FD_OCCASIONS_IN_A_SLOT_TAG;
  nfapi_resp->prach_param.max_prach_fd_occasions_in_a_slot.value = rand8();
  nfapi_resp->num_tlv++;

  nfapi_resp->measurement_param.rssi_measurement_support.tl.tag = NFAPI_NR_PARAM_TLV_RSSI_MEASUREMENT_SUPPORT_TAG;
  nfapi_resp->measurement_param.rssi_measurement_support.value = rand8();
  nfapi_resp->num_tlv++;
}

int main(int n, char *v[])
{
  srand(time(NULL));
#ifndef _STANDALONE_TESTING_
  logInit();
  set_glog(OAILOG_DISABLE);
#endif

  fapi_nr_param_response_scf_t req;
  memset(&req, 0, sizeof(req));
  req.header.message_id = NFAPI_NR_PHY_MSG_TYPE_PARAM_RESPONSE;
  req.header.num_msg = rand16();
  req.header.opaque_handle = rand16();
  uint8_t msg_buf[8192];
  uint16_t msg_len = sizeof(req);

  // Fill Param response TVLs
  fill_param_response_tlv(&req);

  // first test the packing procedure
  printf("Test the packing procedure by checking the return value\n");
  int pack_result = fapi_nr_p5_message_pack(&req, msg_len, msg_buf, sizeof(msg_buf), NULL);
  // PARAM.response message body length is AT LEAST 10 (NFAPI_HEADER_LENGTH + 1 byte error_code + 1 byte num_tlv)
  AssertFatal(pack_result >= NFAPI_HEADER_LENGTH + 1 + 1,
              "fapi_p5_message_pack packed_length not AT LEAST equal to NFAPI_HEADER_LENGTH + 1 byte error_code + 1 byte num_tlv "
              "(10)! Reported value was %d\n",
              pack_result);
  printf("fapi_p5_message_pack packed_length 0x%02x\n", pack_result);
  // update req message_length value with value calculated in message_pack procedure
  req.header.message_length = pack_result - NFAPI_HEADER_LENGTH;
  // test the unpacking of the header
  // copy first NFAPI_HEADER_LENGTH bytes into a new buffer, to simulate SCTP PEEK
  fapi_message_header_t header;
  uint32_t header_buffer_size = NFAPI_HEADER_LENGTH;
  uint8_t header_buffer[header_buffer_size];
  for (int idx = 0; idx < header_buffer_size; idx++) {
    header_buffer[idx] = msg_buf[idx];
  }
  uint8_t *pReadPackedMessage = header_buffer;
  printf("Test the header unpacking and compare with initial message\n");
  int unpack_header_result = fapi_nr_p5_message_header_unpack(&pReadPackedMessage, NFAPI_HEADER_LENGTH, &header, sizeof(header), 0);
  AssertFatal(unpack_header_result >= 0, "nfapi_p5_message_header_unpack failed with return %d\n", unpack_header_result);
  printf("num_msg: 0x%02x\n", header.num_msg);
  AssertFatal(header.num_msg == req.header.num_msg,
              "num_msg was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.num_msg,
              req.header.num_msg);
  printf("opaque_handle: 0x%02x\n", header.opaque_handle);
  AssertFatal(header.opaque_handle == req.header.opaque_handle,
              "Spare was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.opaque_handle,
              req.header.opaque_handle);
  printf("Message ID : 0x%02x\n", header.message_id);
  AssertFatal(header.message_id == req.header.message_id,
              "Message ID was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.message_id,
              req.header.message_id);
  printf("Message length : 0x%02x\n", header.message_length);
  AssertFatal(header.message_length == req.header.message_length,
              "Message length was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.message_length,
              req.header.message_length);

  printf("Test the unpacking and compare with initial message\n");
  // test the unpaking and compare with initial message
  fapi_nr_param_response_scf_t unpacked_req;
  memset(&unpacked_req, 0, sizeof(unpacked_req));
  int unpack_result =
      fapi_nr_p5_message_unpack(msg_buf, header.message_length + NFAPI_HEADER_LENGTH, &unpacked_req, sizeof(unpacked_req), NULL);
  AssertFatal(unpack_result >= 0, "fapi_nr_p5_message_unpack failed with return %d\n", unpack_result);
  printf("num_msg: 0x%02x\n", unpacked_req.header.num_msg);
  AssertFatal(unpacked_req.header.num_msg == req.header.num_msg,
              "num_msg was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.header.num_msg,
              req.header.num_msg);
  printf("opaque_handle: 0x%02x\n", unpacked_req.header.opaque_handle);
  AssertFatal(unpacked_req.header.opaque_handle == req.header.opaque_handle,
              "opaque_handle was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.header.opaque_handle,
              req.header.opaque_handle);
  printf("Message id : 0x%02x\n", unpacked_req.header.message_id);
  AssertFatal(unpacked_req.header.message_id == req.header.message_id,
              "Message id was not 0x%02x, was 0x%02x\n",
              req.header.message_id,
              unpacked_req.header.message_id);
  printf("Message length : 0x%02x\n", unpacked_req.header.message_length);
  AssertFatal(unpacked_req.header.message_length == req.header.message_length,
              "Message length was not the same as the value previously packed, was 0x%02x\n",
              unpacked_req.header.message_length);
  // Test the message body
  printf("Error Code: 0x%02x\n", unpacked_req.error_code);
  AssertFatal(unpacked_req.error_code == req.error_code,
              "Error Code was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.error_code,
              req.error_code);

  printf("Num TLVs: 0x%02x\n", unpacked_req.num_tlv);
  AssertFatal(unpacked_req.num_tlv == req.num_tlv,
              "Error Code was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.num_tlv,
              req.num_tlv);

  test_param_response_tlv(unpacked_req, req);

  // All tests successful!
  return 0;
}
