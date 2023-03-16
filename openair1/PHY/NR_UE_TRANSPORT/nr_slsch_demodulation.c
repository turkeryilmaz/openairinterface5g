#include "PHY/defs_gNB.h"
#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "nr_transport_proto.h"
#include "PHY/impl_defs_top.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/NR_ESTIMATION/nr_ul_estimation.h"
#include "PHY/defs_nr_common.h"
#include "common/utils/nr/nr_common.h"

//#define DEBUG_CH_COMP
//#define DEBUG_RB_EXT
//#define DEBUG_CH_MAG

#define INVALID_VALUE 255

void nr_slsch_extract_rbs(int32_t **rxdataF,
                          NR_UE_PSSCH *pssch_vars,
                          int slot,
                          unsigned char symbol,
                          uint8_t is_dmrs_symbol,
                          nfapi_nr_pssch_pdu_t *pssch_pdu,
                          NR_DL_FRAME_PARMS *frame_parms) {

  unsigned short start_re, re, nb_re_pssch;
  unsigned char aarx, aatx;
  uint32_t rxF_ext_index = 0;
  uint32_t sl_ch0_ext_index = 0;
  uint32_t sl_ch0_index = 0;
  int16_t *rxF, *rxF_ext;
  int *sl_ch0, *sl_ch0_ext;
  uint16_t nb_re_sci1 = 0;
  uint16_t n = 0;
  uint8_t k_prime = 0;

  if (1 <= symbol && symbol <= 3) {
    nb_re_sci1 = NR_NB_SC_PER_RB * NB_RB_SCI1;
  }
  int soffset = (slot & 3) * frame_parms->symbols_per_slot * frame_parms->ofdm_symbol_size;

#ifdef DEBUG_RB_EXT
  printf("--------------------symbol = %d-----------------------\n", symbol);
  printf("--------------------ch_ext_index = %d-----------------------\n", symbol * NR_NB_SC_PER_RB * pusch_pdu->rb_size);
#endif

  uint8_t is_data_re;
  start_re = (frame_parms->first_carrier_offset + (pssch_pdu->rb_start + pssch_pdu->bwp_start) * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
  if (1 <= symbol && symbol <= 3) {
    start_re += nb_re_sci1;
  }
  nb_re_pssch = NR_NB_SC_PER_RB * pssch_pdu->rb_size;

#ifdef __AVX2__
  int nb_re_pssch2 = nb_re_pssch + (nb_re_pssch & 7);
#else
  int nb_re_pssch2 = nb_re_pssch;
#endif
  uint16_t G_SCI2_bits = 32; // TODO: provide value as arg.
  for (aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    uint16_t m0 = 0;
    uint16_t m = G_SCI2_bits;
    rxF = (int16_t *)&rxdataF[aarx][soffset + (symbol * frame_parms->ofdm_symbol_size)];
    rxF_ext = (int16_t *)&pssch_vars->rxdataF_ext[aarx][symbol * nb_re_pssch2]; // [hna] rxdataF_ext isn't contiguous in order to solve an alignment problem ib llr computation in case of mod_order = 4, 6

    if (is_dmrs_symbol == 0) {
      uint16_t nb_re = 0;
      nb_re = start_re + (nb_re_pssch - nb_re_sci1);
      if (nb_re <= frame_parms->ofdm_symbol_size) {
          memcpy1((void*)&rxF_ext[m << 1], (void*)&rxF[start_re * 2], (nb_re_pssch - nb_re_sci1) * sizeof(int32_t));
      } else {
        int neg_length = frame_parms->ofdm_symbol_size - start_re;
        int pos_length = nb_re_pssch - neg_length;
        memcpy1((void*)&rxF_ext[m << 1], (void*)&rxF[start_re * 2], neg_length * sizeof(int32_t));
        memcpy1((void*)&rxF_ext[(m << 1) + (2 * neg_length)], (void*)rxF, pos_length * sizeof(int32_t));
      }

      for (aatx = 0; aatx < pssch_pdu->nrOfLayers; aatx++) {
        sl_ch0 = &pssch_vars->sl_ch_estimates[aatx * frame_parms->nb_antennas_rx + aarx][pssch_vars->dmrs_symbol * frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
        sl_ch0_ext = &pssch_vars->sl_ch_estimates_ext[aatx * frame_parms->nb_antennas_rx + aarx][symbol * nb_re_pssch2];
        memcpy1((void*)sl_ch0_ext, (void*)sl_ch0, nb_re_pssch * sizeof(int32_t));
      }

    } else { // DMRS case
      for (aatx = 0; aatx < pssch_pdu->nrOfLayers; aatx++) {
        sl_ch0 = &pssch_vars->sl_ch_estimates[aatx * frame_parms->nb_antennas_rx + aarx][pssch_vars->dmrs_symbol * frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
        sl_ch0_ext = &pssch_vars->sl_ch_estimates_ext[aatx * frame_parms->nb_antennas_rx + aarx][symbol * nb_re_pssch2];

        rxF_ext_index = 0;
        sl_ch0_ext_index = 0;
        sl_ch0_index = 0;
        for (re = 0; re < nb_re_pssch; re++) {
          uint16_t k = start_re + re;
          is_data_re = allowed_xlsch_re_in_dmrs_symbol(k, start_re, frame_parms->ofdm_symbol_size, pssch_pdu->num_dmrs_cdm_grps_no_data, pssch_pdu->dmrs_config_type);
          if (++k >= frame_parms->ofdm_symbol_size) {
            k -= frame_parms->ofdm_symbol_size;
          }

          #ifdef DEBUG_RB_EXT
          printf("re = %d, is_dmrs_symbol = %d, symbol = %d\n", re, is_dmrs_symbol, symbol);
          #endif

          // save only data and respective channel estimates
          if (is_data_re == 1) {
            if (m0 < G_SCI2_bits) {
              rxF_ext[m0 << 1]     = (rxF[ ((start_re + re) * 2)      % (frame_parms->ofdm_symbol_size * 2)]);
              rxF_ext[(m0 << 1) + 1] = (rxF[(((start_re + re) * 2) + 1) % (frame_parms->ofdm_symbol_size * 2)]);
              m0++;
            } else {
              rxF_ext[m << 1]     = (rxF[ ((start_re + re) * 2)      % (frame_parms->ofdm_symbol_size * 2)]);
              rxF_ext[(m << 1) + 1] = (rxF[(((start_re + re) * 2) + 1) % (frame_parms->ofdm_symbol_size * 2)]);
              m++;
            }

            sl_ch0_ext[sl_ch0_ext_index] = sl_ch0[sl_ch0_index];
            sl_ch0_ext_index++;

            #ifdef DEBUG_RB_EXT
            printf("dmrs symb %d: rxF_ext[%d] = (%d,%d), ul_ch0_ext[%d] = (%d,%d)\n",
                 is_dmrs_symbol, rxF_ext_index >> 1, rxF_ext[rxF_ext_index], rxF_ext[rxF_ext_index + 1],
                 sl_ch0_ext_index,  ((int16_t*)&sl_ch0_ext[sl_ch0_ext_index])[0],  ((int16_t*)&sl_ch0_ext[sl_ch0_ext_index])[1]);
            #endif
          }
          sl_ch0_index++;
        }
      }
    }
  }
}

//==============================================================================================

/* Main Function */
void nr_rx_pssch(PHY_VARS_NR_UE *nrUE,
                 uint8_t dlsch_id,
                 uint32_t frame,
                 uint8_t slot,
                 unsigned char harq_pid)
{

  uint8_t aarx, aatx;
  uint32_t nb_re_pusch, bwp_start_subcarrier;
  int avgs = 0;

  NR_DL_FRAME_PARMS *frame_parms = &nrUE->frame_parms;
  nfapi_nr_pssch_pdu_t *rel16_sl = &nrUE->slsch_rx[dlsch_id][0][0]->harq_processes[harq_pid]->pssch_pdu;
  int avg[frame_parms->nb_antennas_rx*rel16_sl->nrOfLayers];

  nrUE->pssch_vars[dlsch_id]->dmrs_symbol = INVALID_VALUE;
  nrUE->pssch_vars[dlsch_id]->cl_done = 0;

  bwp_start_subcarrier = ((rel16_sl->rb_start + rel16_sl->bwp_start)*NR_NB_SC_PER_RB + frame_parms->first_carrier_offset) % frame_parms->ofdm_symbol_size;
  LOG_D(NR_PHY, "pssch %d.%d : bwp_start_subcarrier %d, rb_start %d, first_carrier_offset %d\n", frame, slot, bwp_start_subcarrier, rel16_sl->rb_start, frame_parms->first_carrier_offset);
  LOG_D(NR_PHY, "pssch %d.%d : sl_dmrs_symb_pos %x\n", frame, slot, rel16_sl->sl_dmrs_symb_pos);
  LOG_D(NR_PHY, "slsch RX %x : start_rb %d nb_rb %d mcs %d Nl %d Tpmi %d bwp_start %d start_sc %d start_symbol %d num_symbols %d cdmgrpsnodata %d num_dmrs %d dmrs_ports %d\n",
          rel16_sl->rnti,rel16_sl->rb_start,rel16_sl->rb_size,rel16_sl->mcs_index,
          rel16_sl->nrOfLayers, 0, rel16_sl->bwp_start,0,rel16_sl->start_symbol_index,rel16_sl->nr_of_symbols,
          rel16_sl->num_dmrs_cdm_grps_no_data,rel16_sl->sl_dmrs_symb_pos,rel16_sl->dmrs_ports);
  //----------------------------------------------------------
  //--------------------- Channel estimation ---------------------
  //----------------------------------------------------------
  start_meas(&nrUE->slsch_channel_estimation_stats);
  for(uint8_t symbol = rel16_sl->start_symbol_index; symbol < (rel16_sl->start_symbol_index + rel16_sl->nr_of_symbols); symbol++) {
    uint8_t dmrs_symbol_flag = (rel16_sl->sl_dmrs_symb_pos >> symbol) & 0x01;
    LOG_D(PHY, "symbol %d, dmrs_symbol_flag :%d\n", symbol, dmrs_symbol_flag);
    
#if 0
    if (dmrs_symbol_flag == 1) {
      if (nrUE->pssch_vars[dlsch_id]->dmrs_symbol == INVALID_VALUE)
        nrUE->pssch_vars[dlsch_id]->dmrs_symbol = symbol;

      for (int nl=0; nl<rel16_sl->nrOfLayers; nl++) {
        
        nr_pusch_channel_estimation(gNB,
                                    slot,
                                    get_dmrs_port(nl,rel16_sl->dmrs_ports),
                                    symbol,
                                    dlsch_id,
                                    bwp_start_subcarrier,
                                    rel16_sl);
      }

      nr_gnb_measurements(gNB, dlsch_id, harq_pid, symbol,rel16_sl->nrOfLayers);

      for (aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
        if (symbol == rel16_sl->start_symbol_index) {
          nrUE->pssch_vars[dlsch_id]->ulsch_power[aarx] = 0;
          nrUE->pssch_vars[dlsch_id]->ulsch_noise_power[aarx] = 0;
        }
        for (aatx = 0; aatx < rel16_sl->nrOfLayers; aatx++) {
          nrUE->pssch_vars[dlsch_id]->ulsch_power[aarx] += signal_energy_nodc(
            &nrUE->pssch_vars[dlsch_id]->ul_ch_estimates[aatx*nrUE->frame_parms.nb_antennas_rx+aarx][symbol * frame_parms->ofdm_symbol_size],
            rel16_sl->rb_size * 12);
        }
        for (int rb = 0; rb < rel16_sl->rb_size; rb++) {
          nrUE->pssch_vars[dlsch_id]->ulsch_noise_power[aarx] +=
              nrUE->measurements.n0_subband_power[aarx][rel16_sl->bwp_start + rel16_sl->rb_start + rb] /
              rel16_sl->rb_size;
        }
        LOG_D(PHY,"aa %d, bwp_start%d, rb_start %d, rb_size %d: ulsch_power %d, ulsch_noise_power %d\n",aarx,
	      rel16_sl->bwp_start,rel16_sl->rb_start,rel16_sl->rb_size,
              nrUE->pssch_vars[dlsch_id]->ulsch_power[aarx],
              nrUE->pssch_vars[dlsch_id]->ulsch_noise_power[aarx]);
      }
    }
#endif
  }

  if (nrUE->chest_time == 1) { // averaging time domain channel estimates
    nr_chest_time_domain_avg(frame_parms,
                             nrUE->pssch_vars[dlsch_id]->sl_ch_estimates,
                             rel16_sl->nr_of_symbols,
                             rel16_sl->start_symbol_index,
                             rel16_sl->sl_dmrs_symb_pos,
                             rel16_sl->rb_size);

    nrUE->pssch_vars[dlsch_id]->dmrs_symbol = get_next_dmrs_symbol_in_slot(rel16_sl->sl_dmrs_symb_pos, rel16_sl->start_symbol_index, rel16_sl->nr_of_symbols);
  }
  stop_meas(&nrUE->slsch_channel_estimation_stats);

#ifdef __AVX2__
  int off = ((rel16_sl->rb_size&1) == 1)? 4:0;
#else
  int off = 0;
#endif
  uint32_t rxdataF_ext_offset = 0;

  for(uint8_t symbol = rel16_sl->start_symbol_index; symbol < (rel16_sl->start_symbol_index + rel16_sl->nr_of_symbols); symbol++) {
    uint8_t dmrs_symbol_flag = (rel16_sl->sl_dmrs_symb_pos >> symbol) & 0x01;
    if (dmrs_symbol_flag == 1) {
      if ((rel16_sl->sl_dmrs_symb_pos >> ((symbol + 1) % frame_parms->symbols_per_slot)) & 0x01)
        AssertFatal(1==0,"Double DMRS configuration is not yet supported\n");

      if (nrUE->chest_time == 0) // Non averaging time domain channel estimates
        nrUE->pssch_vars[dlsch_id]->dmrs_symbol = symbol;

      if (rel16_sl->dmrs_config_type == 0) {
        // if no data in dmrs cdm group is 1 only even REs have no data
        // if no data in dmrs cdm group is 2 both odd and even REs have no data
        nb_re_pusch = rel16_sl->rb_size *(12 - (rel16_sl->num_dmrs_cdm_grps_no_data*6));
      }
      else {
        nb_re_pusch = rel16_sl->rb_size *(12 - (rel16_sl->num_dmrs_cdm_grps_no_data*4));
      }
    } 
    else {
      nb_re_pusch = rel16_sl->rb_size * NR_NB_SC_PER_RB;
    }

    nrUE->pssch_vars[dlsch_id]->sl_valid_re_per_slot[symbol] = nb_re_pusch;
    LOG_D(PHY,"symbol %d: nb_re_pusch %d, DMRS symbl used for Chest :%d \n", symbol, nb_re_pusch, nrUE->pssch_vars[dlsch_id]->dmrs_symbol);

    //----------------------------------------------------------
    //--------------------- RBs extraction ---------------------
    //----------------------------------------------------------
    if (nb_re_pusch > 0) {
      start_meas(&nrUE->slsch_rbs_extraction_stats);
      int32_t **rxdataF = nrUE->common_vars.common_vars_rx_data_per_thread[proc->thread_id].rxdataF;

      nr_slsch_extract_rbs(rxdataF,
                           nrUE->pssch_vars[dlsch_id],
                           slot,
                           symbol,
                           dmrs_symbol_flag,
                           rel16_sl,
                           frame_parms);
      stop_meas(&nrUE->slsch_rbs_extraction_stats);

      //----------------------------------------------------------
      //--------------------- Channel Scaling --------------------
      //----------------------------------------------------------
      nr_ulsch_scale_channel(nrUE->pssch_vars[dlsch_id]->ul_ch_estimates_ext,
                            frame_parms,
                            nrUE->ulsch[dlsch_id],
                            symbol,
                            dmrs_symbol_flag,
                            nb_re_pusch,
                            rel16_sl->nrOfLayers,
                            rel16_sl->rb_size);

      if (nrUE->pssch_vars[dlsch_id]->cl_done==0) {
        nr_ulsch_channel_level(nrUE->pssch_vars[dlsch_id]->ul_ch_estimates_ext,
                              frame_parms,
                              avg,
                              symbol,
                              nb_re_pusch,
                              rel16_sl->nrOfLayers,
                              rel16_sl->rb_size);

        avgs = 0;

        for (aatx=0;aatx<rel16_sl->nrOfLayers;aatx++)
          for (aarx=0;aarx<frame_parms->nb_antennas_rx;aarx++)
            avgs = cmax(avgs,avg[aatx*frame_parms->nb_antennas_rx+aarx]);

        nrUE->pssch_vars[dlsch_id]->log2_maxh = (log2_approx(avgs)/2)+2;
        nrUE->pssch_vars[dlsch_id]->cl_done = 1;
      }

      //----------------------------------------------------------
      //--------------------- Channel Compensation ---------------
      //----------------------------------------------------------
      start_meas(&nrUE->ulsch_channel_compensation_stats);
      LOG_D(PHY,"Doing channel compensations log2_maxh %d, avgs %d (%d,%d)\n",nrUE->pssch_vars[dlsch_id]->log2_maxh,avgs,avg[0],avg[1]);
      nr_ulsch_channel_compensation(nrUE->pssch_vars[dlsch_id]->rxdataF_ext,
                                    nrUE->pssch_vars[dlsch_id]->ul_ch_estimates_ext,
                                    nrUE->pssch_vars[dlsch_id]->ul_ch_mag0,
                                    nrUE->pssch_vars[dlsch_id]->ul_ch_magb0,
                                    nrUE->pssch_vars[dlsch_id]->rxdataF_comp,
                                    (rel16_sl->nrOfLayers>1) ? nrUE->pssch_vars[dlsch_id]->rho : NULL,
                                    frame_parms,
                                    symbol,
                                    nb_re_pusch,
                                    dmrs_symbol_flag,
                                    rel16_sl->qam_mod_order,
                                    rel16_sl->nrOfLayers,
                                    rel16_sl->rb_size,
                                    nrUE->pssch_vars[dlsch_id]->log2_maxh);
      stop_meas(&nrUE->ulsch_channel_compensation_stats);

      start_meas(&nrUE->ulsch_mrc_stats);
      nr_ulsch_detection_mrc(frame_parms,
                             nrUE->pssch_vars[dlsch_id]->rxdataF_comp,
                             nrUE->pssch_vars[dlsch_id]->ul_ch_mag0,
                             nrUE->pssch_vars[dlsch_id]->ul_ch_magb0,
                             (rel16_sl->nrOfLayers>1) ? nrUE->pssch_vars[dlsch_id]->rho : NULL,
                             rel16_sl->nrOfLayers,
                             symbol,
                             rel16_sl->rb_size,
                             nb_re_pusch);
                 
      if (rel16_sl->nrOfLayers == 2)//Apply zero forcing for 2 Tx layers
        nr_ulsch_zero_forcing_rx_2layers(nrUE->pssch_vars[dlsch_id]->rxdataF_comp,
                                   nrUE->pssch_vars[dlsch_id]->ul_ch_mag0,
                                   nrUE->pssch_vars[dlsch_id]->ul_ch_magb0,                                   
                                   nrUE->pssch_vars[dlsch_id]->ul_ch_estimates_ext,
                                   rel16_sl->rb_size,
                                   frame_parms->nb_antennas_rx,
                                   rel16_sl->qam_mod_order,
                                   nrUE->pssch_vars[dlsch_id]->log2_maxh,
                                   symbol,
                                   nb_re_pusch);
      stop_meas(&nrUE->ulsch_mrc_stats);

      if (rel16_sl->transform_precoding == transformPrecoder_enabled) {
         #ifdef __AVX2__
        // For odd number of resource blocks need byte alignment to multiple of 8
        int nb_re_pusch2 = nb_re_pusch + (nb_re_pusch&7);
        #else
        int nb_re_pusch2 = nb_re_pusch;
        #endif

        // perform IDFT operation on the compensated rxdata if transform precoding is enabled
        nr_idft(&nrUE->pssch_vars[dlsch_id]->rxdataF_comp[0][symbol * nb_re_pusch2], nb_re_pusch);
        LOG_D(PHY,"Transform precoding being done on data- symbol: %d, nb_re_pusch: %d\n", symbol, nb_re_pusch);
      }

      //----------------------------------------------------------
      //--------------------- PTRS Processing --------------------
      //----------------------------------------------------------
      /* In case PTRS is enabled then LLR will be calculated after PTRS symbols are processed *
      * otherwise LLR are calculated for each symbol based upon DMRS channel estimates only. */
      if (rel16_sl->pdu_bit_map & PUSCH_PDU_BITMAP_PUSCH_PTRS) {
        start_meas(&nrUE->ulsch_ptrs_processing_stats);
        nr_pusch_ptrs_processing(gNB,
                                 frame_parms,
                                 rel16_sl,
                                 dlsch_id,
                                 slot,
                                 symbol,
                                 nb_re_pusch);
        stop_meas(&nrUE->ulsch_ptrs_processing_stats);

        /*  Subtract total PTRS RE's in the symbol from PUSCH RE's */
        nrUE->pssch_vars[dlsch_id]->ul_valid_re_per_slot[symbol] -= nrUE->pssch_vars[dlsch_id]->ptrs_re_per_slot;
      }

      /*---------------------------------------------------------------------------------------------------- */
      /*--------------------  LLRs computation  -------------------------------------------------------------*/
      /*-----------------------------------------------------------------------------------------------------*/
      start_meas(&nrUE->ulsch_llr_stats);
      for (aatx=0; aatx < rel16_sl->nrOfLayers; aatx++) {
        nr_ulsch_compute_llr(&nrUE->pssch_vars[dlsch_id]->rxdataF_comp[aatx*frame_parms->nb_antennas_rx][symbol * (off + rel16_sl->rb_size * NR_NB_SC_PER_RB)],
                             nrUE->pssch_vars[dlsch_id]->ul_ch_mag0[aatx*frame_parms->nb_antennas_rx],
                             nrUE->pssch_vars[dlsch_id]->ul_ch_magb0[aatx*frame_parms->nb_antennas_rx],
                             &nrUE->pssch_vars[dlsch_id]->llr_layers[aatx][rxdataF_ext_offset * rel16_sl->qam_mod_order],
                             rel16_sl->rb_size,
                             nrUE->pssch_vars[dlsch_id]->ul_valid_re_per_slot[symbol],
                             symbol,
                             rel16_sl->qam_mod_order);
      }
      stop_meas(&nrUE->ulsch_llr_stats);
      rxdataF_ext_offset += nrUE->pssch_vars[dlsch_id]->ul_valid_re_per_slot[symbol];
    }
  } // symbol loop
}
