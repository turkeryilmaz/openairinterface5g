#include <armral.h>
#include <rte_branch_prediction.h>

#include "armral_bfp_compression.h"
#include "common/utils/LOG/log.h"

/**
 * The function operates on a fixed block size of one Physical Resource Block
 * (PRB). Each block consists of 12 16-bit complex resource elements. Each block
 * taken as input is compressed into 24 samples with a common unsigned scaling factor.
 *
 * @param[in]     iq_width  Width in bits of compressed samples.
 * @param[in]     n_prb     The number of input resource blocks.
 * @param[in]     src       Points to the input complex samples sequence.
 * @param[out]    dst       Points to the output compressed data.
 */
void armral_bfp_compression(uint32_t iq_width, uint32_t n_prb, int16_t *src, int8_t *dst) {

  switch(iq_width) {
  
    case 9:
      armral_status ret = armral_block_float_compr_9bit(n_prb, (armral_cmplx_int16_t *)src, (armral_compressed_data_9bit *)dst, NULL);
      if(unlikely(ret != ARMRAL_SUCCESS)) {
        LOG_E(PHY, "armral_block_scaling_compr_9bit returned an error: %d\n", (int)ret);
      }
      break;

    default:
      LOG_E(PHY, "Unsupported IQ width in BFP compression: %d\n", iq_width);
  
  }

}

/**
 * The function operates on a fixed block size of one Physical Resource Block
 * (PRB). Each block consists of 12 compressed complex resource elements.
 * Each block taken as input is expanded into 12 16-bit complex samples.
 *
 * @param[in]     iq_width  Width in bits of compressed samples.
 * @param[in]     n_prb     The number of input resource blocks.
 * @param[in]     src       Points to the input compressed data.
 * @param[out]    dst       Points to the output complex samples sequence.
 */
void armral_bfp_decompression(uint32_t iq_width, uint32_t n_prb, int8_t *src, int16_t *dst) {

  switch(iq_width) {
  
    case 9:
      armral_status ret = armral_block_float_decompr_9bit(n_prb, (armral_compressed_data_9bit *)src, (armral_cmplx_int16_t *)dst, NULL);
        if(unlikely(ret != ARMRAL_SUCCESS)) {
        LOG_E(PHY, "armral_block_scaling_decompr_9bit returned an error: %d\n", (int)ret);
      }
      break;

    default:
      LOG_E(PHY, "Unsupported IQ width in BFP decompression: %d\n", iq_width);
  
  }

}

