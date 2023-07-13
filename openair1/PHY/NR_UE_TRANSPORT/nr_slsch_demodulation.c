#include "PHY/defs_nr_UE.h"
#include "PHY/phy_extern.h"
#include "nr_transport_proto_ue.h"
#include "PHY/impl_defs_top.h"
#include "PHY/NR_TRANSPORT/nr_sch_dmrs.h"
#include "PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/NR_REFSIG/ptrs_nr.h"
#include "PHY/defs_nr_common.h"
#include "common/utils/nr/nr_common.h"
#include "openair1/PHY/NR_TRANSPORT/nr_transport_common_proto.h"

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
                          NR_DL_FRAME_PARMS *frame_parms,
                          NR_DL_UE_HARQ_t *harq,
                          int chest_time_type) {

  unsigned short start_re, re, nb_re_pssch;
  unsigned char aarx, aatx;
  uint32_t sl_ch0_ext_index = 0;
  uint32_t sl_ch0_index = 0;
  int16_t *rxF, *rxF_ext;
  int *sl_ch0, *sl_ch0_ext;
  uint16_t nb_re_sci1 = 0;

  int8_t validDmrsEst;
  if (chest_time_type == 0)
    validDmrsEst = get_valid_dmrs_idx_for_channel_est(harq->dlDmrsSymbPos, symbol);
  else
    validDmrsEst = get_next_dmrs_symbol_in_slot(harq->dlDmrsSymbPos, harq->start_symbol, harq->nb_symbols); // get first dmrs symbol index

  if (1 <= symbol && symbol <= 3) {
    nb_re_sci1 = NR_NB_SC_PER_RB * NB_RB_SCI1;
  }
  int soffset = (slot & 3) * frame_parms->symbols_per_slot * frame_parms->ofdm_symbol_size;

#ifdef DEBUG_RB_EXT
  printf("--------------------symbol = %d-----------------------\n", symbol);
  printf("--------------------ch_ext_index = %d-----------------------\n", symbol * NR_NB_SC_PER_RB * pssch_pdu->rb_size);
#endif

  uint8_t is_data_re;
  start_re = (frame_parms->first_carrier_offset + (pssch_pdu->rb_start + pssch_pdu->bwp_start) * NR_NB_SC_PER_RB) % frame_parms->ofdm_symbol_size;
  if (1 <= symbol && symbol <= 3) {
    start_re += nb_re_sci1;
  }
  nb_re_pssch = NR_NB_SC_PER_RB * harq->nb_rb;

#ifdef __AVX2__
  int nb_re_pssch2 = nb_re_pssch + (nb_re_pssch & 7);
#else
  int nb_re_pssch2 = nb_re_pssch;
#endif
  for (aarx = 0; aarx < frame_parms->nb_antennas_rx; aarx++) {
    uint16_t m = 0;
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
        sl_ch0 = &pssch_vars->sl_ch_estimates[aatx * frame_parms->nb_antennas_rx + aarx][validDmrsEst * frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
        sl_ch0_ext = &pssch_vars->sl_ch_estimates_ext[aatx * frame_parms->nb_antennas_rx + aarx][symbol * nb_re_pssch2];
        memcpy1((void*)sl_ch0_ext, (void*)sl_ch0, (nb_re_pssch - nb_re_sci1) * sizeof(int32_t));
      }

    } else { // DMRS case
      for (aatx = 0; aatx < pssch_pdu->nrOfLayers; aatx++) {
        sl_ch0 = &pssch_vars->sl_ch_estimates[aatx * frame_parms->nb_antennas_rx + aarx][validDmrsEst * frame_parms->ofdm_symbol_size]; // update channel estimates if new dmrs symbol are available
        sl_ch0_ext = &pssch_vars->sl_ch_estimates_ext[aatx * frame_parms->nb_antennas_rx + aarx][symbol * nb_re_pssch2];

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
            if (aatx == 0) {
              rxF_ext[m << 1]     = (rxF[ ((start_re + re) * 2)      % (frame_parms->ofdm_symbol_size * 2)]);
              rxF_ext[(m << 1) + 1] = (rxF[(((start_re + re) * 2) + 1) % (frame_parms->ofdm_symbol_size * 2)]);
              m++;
            }

            sl_ch0_ext[sl_ch0_ext_index] = sl_ch0[sl_ch0_index];
            sl_ch0_ext_index++;

            #ifdef DEBUG_RB_EXT
            uint32_t rxF_ext_index = 0;
            printf("dmrs symb %d: rxF_ext[%d] = (%d,%d), sl_ch0_ext[%d] = (%d,%d)\n",
                 is_dmrs_symbol, rxF_ext_index>>1, rxF_ext[rxF_ext_index], rxF_ext[rxF_ext_index+1],
                 sl_ch0_ext_index,  ((int16_t*)&sl_ch0_ext[sl_ch0_ext_index])[0],  ((int16_t*)&sl_ch0_ext[sl_ch0_ext_index])[1]);
            #endif
          }
          sl_ch0_index++;
        }
      }
    }
  }
}
