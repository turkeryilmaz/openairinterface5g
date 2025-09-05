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

#ifndef __OAI_CUDA_H__
#define __OAI_CUDA_H__

#include <stdint.h>


#ifdef __NVCC__
    typedef struct complex16 {
      int16_t r;
      int16_t i;
    } c16_t;
#else
    #include "PHY/TOOLS/tools_defs.h"
#endif

#ifdef __NVCC__
    #include <curand_kernel.h>

    __device__ float2 complex_mul(float2 a, float2 b);

    __global__ void multipath_channel_kernel(
        const float2* __restrict__ d_channel_coeffs,
        // const float2* __restrict__ tx_sig,
        const float* __restrict__ tx_sig,
        float2* __restrict__ rx_sig,
        int num_samples,
        int channel_length,
        int nb_tx,
        int nb_rx);

    __global__ void add_noise_and_phase_noise_kernel(
        const float2* __restrict__ r_sig,
        short2* __restrict__ output_sig,
        curandState_t* states,
        int num_samples,
        float sigma,
        float pn_std_dev,
        uint16_t pdu_bit_map,
        uint16_t ptrs_bit_map
    );

#endif // __NVCC__


#ifdef __cplusplus
extern "C" {
#endif

void run_channel_pipeline_cuda(
    c16_t **output_signal,
    int nb_tx, int nb_rx, int channel_length, uint32_t num_samples, // Note: This is the number of IQ pairs
    float *h_channel_coeffs,
    float sigma2, double ts,
    uint16_t pdu_bit_map, uint16_t ptrs_bit_map, 
    int slot_offset, int delay,                 
    void *d_tx_sig, void *d_intermediate_sig, void* d_final_output,
    void *d_curand_states, void* h_tx_sig_pinned, void* h_final_output_pinned,
    void *d_channel_coeffs
);


void run_channel_pipeline_cuda_batched(
    // todo: implement interleaved version
    int num_channels,
    int nb_tx, int nb_rx, int channel_length, uint32_t num_samples,
    void *d_channel_coeffs_batch,
    float sigma2, double ts,
    uint16_t pdu_bit_map, uint16_t ptrs_bit_map,
    void *d_tx_sig_batch, void *d_intermediate_sig_batch, void *d_final_output_batch,
    void *d_curand_states
);

void run_channel_pipeline_cuda_streamed(
    int nb_tx, int nb_rx, int channel_length, uint32_t num_samples,
    float *h_channel_coeffs,
    float sigma2, double ts,
    uint16_t pdu_bit_map, uint16_t ptrs_bit_map,
    void *d_tx_sig_void, void *d_intermediate_sig_void, void *d_final_output_void,
    void *d_curand_states_void, void* h_tx_sig_pinned_void,
    void *d_channel_coeffs_void,
    void* stream_void
);

void multipath_channel_cuda(
    float **rx_sig_re, float **rx_sig_im,
    int nb_tx, int nb_rx, int channel_length,
    uint32_t length, uint64_t channel_offset,
    float *h_channel_coeffs,
    void *d_tx_sig, void *d_rx_sig,
    void *d_channel_coeffs,
    void *h_tx_sig_pinned
);

void add_noise_cuda(
    const float **r_re,
    const float **r_im,
    c16_t **output_signal,
    int num_samples,
    int nb_rx,
    float sigma2,
    double ts,
    int slot_offset,
    int delay,
    uint16_t pdu_bit_map,
    uint16_t ptrs_bit_map,
    void *d_r_sig,
    void *d_output_sig,
    void *d_curand_states,
    void *h_r_sig_pinned,
    void *h_output_sig_pinned
);

void sum_channel_outputs_cuda(
    void **d_individual_outputs,
    void *d_final_output,
    int num_channels,
    int nb_rx,
    int num_samples
);

void interleave_channel_output_cuda(float **rx_sig_re,
                                    float **rx_sig_im,
                                    void **output_interleaved,
                                    int nb_rx,
                                    int num_samples);

// Note: output_interleaved should point to arrays that can hold float2 data
// Each output_interleaved[i] should be allocated as: malloc(num_samples * sizeof(float2))
// The caller can safely cast to (float2**) after the function returns

void* create_and_init_curand_states_cuda(int num_elements, unsigned long long seed);
void destroy_curand_states_cuda(void* d_curand_states);


#ifdef __cplusplus
}
#endif

#endif // __OAI_CUDA_H__
