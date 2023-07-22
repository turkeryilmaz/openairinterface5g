uint32_t nr_sci_size(const NR_SL_ResourcePool_r16_t *sl_res_pool,
	             nr_sci_pdu_t *sci_pdu,
	             nr_sci_format_t format) {
			     
  int size=0;

  switch(format) {
    case NR_SL_SCI_FORMAT_1A:
	    // priority
	    size+=3;
	    // frequency resource assignment
	    long Nsc = sl_res_pool->sl_NumSubchannel_r16;
            if (sl_res_pool->sl_UE_SelectedConfigRP_r16.sl_MaxNumPerReserve_r16 == NR_SL_UE_SelectedConfigRP_r16__sl_MaxNumPerReserve_r16_n2)
	      sci_pdu->frequency_resource_assignment.nbits =  (uint8_t)ceil(log2((Nsc * (Nsc + 1)) >>1));  
	    else
	      sci_pdu->frequency_resource_assignment.nbits =  (uint8_t)ceil(log2((Nsc * (Nsc + 1) * (2*Nsc + 1)) /6));  
            size += sci_pdu->frequency_resource_assignment.nbits;
	    // time-domain-assignment
            if (sl_res_pool->sl_UE_SelectedConfigRP_r16.sl_MaxNumPerReserve_r16 == NR_SL_UE_SelectedConfigRP_r16__sl_MaxNumPerReserve_r16_n2)
		sci_pdu->time_resource_assignment.nbits = 5;
	    else
		sci_pdu->time_resource_assignment.nbits = 9;
            size += sci_pdu->time_resource_assignment.nbits;
	    
	    // resource reservation period
	    
	    if (0 /*!sl_res_pool->sl_MultiReserveResource*/) // not defined in 17.4 RRC
	       sci_pdu->resource_reservation_period.nbits = 0;
            size += sci_pdu->resource_reservation_period.nbits;

	    // DMRS pattern
	    int dmrs_pattern_num = sl_res_pool->sl_PSSCH_Config_r16->choice.setup->sl_PSSCH_DMRS_TimePatternList_r16->list.count;
	    sci_pdu->dmrs_pattern.nbits = (uint8_t)ceil(log2(dmrs_pattern_num));
	    size += sci_pdu->dmrs_pattern.nbits;

            // second_stage_sci_format // 2 bits - Table 8.3.1.1-1
	    size += 2;
	    // beta_offset_indicator // 2 bits - depending sl-BetaOffsets2ndSCI and Table 8.3.1.1-2
	    size += 2
	    // number_of_dmrs_port // 1 bit - Table 8.3.1.1-3
	    size += 1;
            // mcs // 5 bits
	    size += 5;
           
	    // additional_mcs; // depending on sl-Additional-MCS-Table
	    if (sl_res_pool->sl_Additional_MCS_Table_r16) 
	       sci_pdu->additional_mcs.nbits = (*sl_res_pool->sl_Additional_MCS_Table_r16 < 2) ? 1 : 2;
	    else sci_pdu->additional_mcs.nbits=0;
	    size += sci_pdu->additional_mcs.nbits;

	    // psfch_overhead; // depending on sl-PSFCH-Period
	    if (sl_res_pool->PSFCH_config && sl_res_pool->PSFCH_config->sl_PSFCH_Period_r16 && *sl_res_pool->PSFCH_config->sl_PSFCH_Period_r16>1)
		sci_pdu->psfch_overhead.nbits=1;
	    else sci_pdu->psfch_overhead.nbits=0;
	    size += sci_pdu->psfch_overhead.nbits;

   	    // reserved; // depending on N_reserved (sl-NumReservedBits) and sl-IndicationUE-B
	    // note R17 dependence no sl_IndicationUE-B needs to be added here
	    AssertFatal(sl_res_pool->sl_PSCCH_Config_r16!=NULL,"sl_res_pool->sl_PSCCH_Config_r16 is null\n");
	    AssertFatal(sl_res_pool->sl_PSCCH_Config_r16->choice.setup!=NULL,"sl_res_pool->sl_PSCCH_Config_r16->choice.setup is null\n");
	    AssertFatal(sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_NumReservedBits_r16!=NULL, "sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_NumReservedBits_r16 is null\n");
            sci_pdu->reserved.nbits = sl_res_pool->sl_PSCCH_Config_r16->choice.setup->sl_NumReservedBits_r16;

            // conflict_information_receiver; // depending on sl-IndicationUE-B 
	    // note: R17 field not included here
	    sci_pdu->conflict_information_receiver.nbits=0;
	    break;
    case NR_SL_SCI_FORMAT_2A:
    case NR_SL_SCI_FORMAT_2B:
    case NR_SL_SCI_FORMAT_2C:
	    // common components
	    //harq_pid; // 4 bits
	    //ndi; // 1 bit
	    //rv_index; // 2 bits
	    //source_id; // 8 bits
	    //dest_id; // 16 bits
	    //harq_feedback; //1 bit
	    size += (4+1+2+8+16+1);
	    if (format==NR_SL_SCI_FORMAT_2A)
	      //cast_type // 2 bits formac 2A
	      size += 2;
	    if (format==NR_SL_SCI_FORMAT_2C || format==NR_SL_SCI_FORMAT_2A)
	      // csi_req // 1 bit format 2A, format 2C
              size +=1;
	    if (format==NR_SL_SCI_FORMAT_2B) { 
	      // zone_id // 12 bits format 2B
	      size +=12;

	      // communication_range; // 4 bits depending on sl-ZoneConfigMCR-Index, format 2B
	      // note fill in for R17
	      if (0) size +=4;
	    }
	    else if (format==NR_SL_SCI_FORMAT_2C) {

       	     // providing_req_ind; // 1 bit, format 2C
	     size += 1;
             // resource_combinations; // depending on n_subChannel^SL (sl-NumSubchennel), N_rsv_period (sl-ResourceReservePeriodList) and sl-MultiReservedResource, format 2C
	     // first_resource_location; // 8 bits, format 2C
	     size += 8;
	     // reference_slot_location; // depending on mu, format 2C
             // resource_set_type; // 1 bit, format 2C
	     size += 1;
	     //	lowest_subchannel_indices; // depending on n_subChannel^SL, format 2C
	     //
	    }
	    break;
  }
  return(size);
}


void fill_sci_pdu(sl_nr_sci_indication_pdu_t *nr_sl_sci_pdu, 
		  nr_sci_pdu_t *sci_pdu, 
		  nr_sci_format_t format,
		  int sci_size) {
  int pos=0,fsize;
  uint64_t *sci_payload =  (uint64_t *)sci_payloadBits;
  switch(format) {
    case NR_SL_SCI_FORMAT_1A:
	    // priority 3 bits
	    fsize=3;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->priority >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    
	    // frequency resource assignment
	    fsize = sci_pdu->frequency_resource_assignment.nbits;  
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->frequency_domain_assignment.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // time-domain-assignment
	    fsize = sci_pdu->time_resource_assignment.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->time_resource_assignment.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // resource reservation period
            fsize = sci_pdu->resource_reservation_period.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->resource_reservation_period.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // DMRS pattern
	    fsize = sci_pdu->dmrs_pattern.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->dmrs_pattern.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
            // second_stage_sci_format // 2 bits - Table 8.3.1.1-1
	    fsize = 2;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->second_stage_sci_format >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // beta_offset_indicator // 2 bits - depending sl-BetaOffsets2ndSCI and Table 8.3.1.1-2
	    fsize = 2;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->beta_offset_indicator >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // number_of_dmrs_port // 1 bit - Table 8.3.1.1-3
	    fsize = 1;
	    *sci_payload |= (((uint64_t)sci_pdu->number_of_dmrs_port&1)) << (sci_size - pos++)
            // mcs // 5 bits
	    fsize = 5;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->mcs >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // additional_mcs; // depending on sl-Additional-MCS-Table
	    fsize = sci_pdu->additional_mcs.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->additional_mcs.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    // psfch_overhead; // depending on sl-PSFCH-Period
   	    fsize = sci_pdu->psfch_overhead.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->psfch_overhead.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);

   	    // reserved; // depending on N_reserved (sl-NumReservedBits) and sl-IndicationUE-B
            fsize = sci_pdu->reserved.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->reserved.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
            // conflict_information_receiver; // depending on sl-IndicationUE-B 
	    fsize = sci_pdu->conflict_information_receiver.nbits;
	    for (int i = 0; i < fsize; i++)
		   *sci_payload |= (((uint64_t)sci_pdu->conflict_information_receiver.val >> (fsize - i - 1)) & 1) << (sci_size - pos++);
	    break;
    case NR_SL_SCI_FORMAT_2A:
    case NR_SL_SCI_FORMAT_2B:
    case NR_SL_SCI_FORMAT_2C:


	    break;
