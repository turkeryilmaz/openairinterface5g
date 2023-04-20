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

/*!\file nrLDPC_intrinsics.h
 * \brief Defines the new intrinsics for efficient processing
 * \author Sebastian Wagner (EURECOM) Email: <mailto:sebastian.wagner@eurecom.fr>
 * \date 21-02-2023
 * \version 1.0
 * \note
 * \warning
 */

#ifndef __NR_LDPC_INTRINSICS__H__
#define __NR_LDPC_INTRINSICS__H__
#include "PHY/sse_intrin.h"

// Shift mask, first 16 entries shift left, second 16 entries shift right
static const int8_t shiftmask256_epi8[33][32] __attribute__ ((aligned(32))) = {
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,         0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,         -1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14},
    {-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,         -1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13},
    {-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,         -1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12},
    {-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,         -1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11},
    {-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,         -1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10},
    {-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,         -1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,9},
    {-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8,        -1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,8},
    {-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7,       -1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,7},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6,      -1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,6},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5,     -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,5},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4,    -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,4},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3,   -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,3},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2,  -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1,2},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0},
    
    {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,           0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15},
    {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1,          1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1},
    {2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,         2,3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-1},
    {3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,        3,4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1},
    {4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,       4,5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1},
    {5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,      5,6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1},
    {6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,     6,7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1},
    {7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,    7,8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1},
    {8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,   8,9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1},
    {9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,  9,10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 10,11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 11,12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 12,13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 13,14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 14,15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, 15,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1},
    {-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}};


// variable left left and right shift
static inline __m256i simde_mm256_sliv_si256(__m256i a, uint8_t shift)
{
    return simde_mm256_shuffle_epi8(a,*(__m256i*)&shiftmask256_epi8[shift]);
}

// right shift of concatinated vectors a and b, shift = 0,...,31
static inline __m256i _mm256_srliv_si2x256(__m256i a, __m256i b, uint8_t shift)
{
    if (shift == 0)
    {
        return a;
    }
    else if(shift > 16)
    {
        // Shift is > 16
        __m256i a0 = simde_mm256_sliv_si256(a,shift);
        __m256i b0 = simde_mm256_sliv_si256(b,shift);
        __m256i b1 = simde_mm256_sliv_si256(b,32-shift);
        __m256i c0 = simde_mm256_permute2x128_si256(a0,b0,0x21);
        return simde_mm256_adds_epi8(c0,b1);
    }
    else
    {
        __m256i a0 = simde_mm256_sliv_si256(a,shift+16);
        __m256i b0 = simde_mm256_sliv_si256(b,16-shift);
        __m256i a1 = simde_mm256_sliv_si256(a,16-shift);
        __m256i c0 = simde_mm256_permute2x128_si256(a1,b0,0x21);
        return simde_mm256_adds_epi8(c0,a0);
    }
}

static inline __m256i _mm256_srli_si2x256_loadu(int8_t* a, uint8_t shift)
{
    return simde_mm256_loadu_si256((__m256i*)&(a[shift]));
}

static inline __m256i _mm256_srli_0_si2x256(__m256i a, __m256i b)
{
    // return a;
    return _mm256_permute2x128_si256(a,b,0x10);
}

static inline __m256i _mm256_srli_1_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,1);
    __m256i b0 = _mm256_slli_si256(b,16-1);
    __m256i a1 = _mm256_slli_si256(a,16-1);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_2_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,2);
    __m256i b0 = _mm256_slli_si256(b,16-2);
    __m256i a1 = _mm256_slli_si256(a,16-2);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_3_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,3);
    __m256i b0 = _mm256_slli_si256(b,16-3);
    __m256i a1 = _mm256_slli_si256(a,16-3);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_4_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,4);
    __m256i b0 = _mm256_slli_si256(b,16-4);
    __m256i a1 = _mm256_slli_si256(a,16-4);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_5_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,5);
    __m256i b0 = _mm256_slli_si256(b,16-5);
    __m256i a1 = _mm256_slli_si256(a,16-5);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_6_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,6);
    __m256i b0 = _mm256_slli_si256(b,16-6);
    __m256i a1 = _mm256_slli_si256(a,16-6);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_7_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,7);
    __m256i b0 = _mm256_slli_si256(b,16-7);
    __m256i a1 = _mm256_slli_si256(a,16-7);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_8_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,8);
    __m256i b0 = _mm256_slli_si256(b,16-8);
    __m256i a1 = _mm256_slli_si256(a,16-8);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
/**
   \brief Performs right shift of concatinated vectors a and b by 9 bytes
   \param a first input vector
   \param Z second input vector
*/
static inline __m256i _mm256_srli_9_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,9);
    __m256i b0 = _mm256_slli_si256(b,16-9);
    __m256i a1 = _mm256_slli_si256(a,16-9);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_10_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,10);
    __m256i b0 = _mm256_slli_si256(b,16-10);
    __m256i a1 = _mm256_slli_si256(a,16-10);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_11_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,11);
    __m256i b0 = _mm256_slli_si256(b,16-11);
    __m256i a1 = _mm256_slli_si256(a,16-11);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_12_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,12);
    __m256i b0 = _mm256_slli_si256(b,16-12);
    __m256i a1 = _mm256_slli_si256(a,16-12);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_13_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,13);
    __m256i b0 = _mm256_slli_si256(b,16-13);
    __m256i a1 = _mm256_slli_si256(a,16-13);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_14_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,14);
    __m256i b0 = _mm256_slli_si256(b,16-14);
    __m256i a1 = _mm256_slli_si256(a,16-14);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_15_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,15);
    __m256i b0 = _mm256_slli_si256(b,16-15);
    __m256i a1 = _mm256_slli_si256(a,16-15);
    __m256i c0 = _mm256_permute2x128_si256(a1,b0,0x21);
    return _mm256_adds_epi8(c0,a0);
}
static inline __m256i _mm256_srli_16_si2x256(__m256i a, __m256i b)
{
    return _mm256_permute2x128_si256(a,b,0x21);
}

// Shifts larger than 16 bytes
static inline __m256i _mm256_srli_17_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,17-16);
    __m256i b0 = _mm256_srli_si256(b,17-16);
    __m256i b1 = _mm256_slli_si256(b,32-17);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_18_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,18-16);
    __m256i b0 = _mm256_srli_si256(b,18-16);
    __m256i b1 = _mm256_slli_si256(b,32-18);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_19_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,19-16);
    __m256i b0 = _mm256_srli_si256(b,19-16);
    __m256i b1 = _mm256_slli_si256(b,32-19);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
/**
   \brief Performs right shift of concatinated vectors a and b by 20 bytes
   \param a first input vector
   \param Z second input vector
*/
static inline __m256i _mm256_srli_20_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,20-16);
    __m256i b0 = _mm256_srli_si256(b,20-16);
    __m256i b1 = _mm256_slli_si256(b,32-20);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_21_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,21-16);
    __m256i b0 = _mm256_srli_si256(b,21-16);
    __m256i b1 = _mm256_slli_si256(b,32-21);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_22_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,22-16);
    __m256i b0 = _mm256_srli_si256(b,22-16);
    __m256i b1 = _mm256_slli_si256(b,32-22);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_23_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,23-16);
    __m256i b0 = _mm256_srli_si256(b,23-16);
    __m256i b1 = _mm256_slli_si256(b,32-23);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_24_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,24-16);
    __m256i b0 = _mm256_srli_si256(b,24-16);
    __m256i b1 = _mm256_slli_si256(b,32-24);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_25_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,25-16);
    __m256i b0 = _mm256_srli_si256(b,25-16);
    __m256i b1 = _mm256_slli_si256(b,32-25);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_26_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,26-16);
    __m256i b0 = _mm256_srli_si256(b,26-16);
    __m256i b1 = _mm256_slli_si256(b,32-26);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_27_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,27-16);
    __m256i b0 = _mm256_srli_si256(b,27-16);
    __m256i b1 = _mm256_slli_si256(b,32-27);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_28_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,28-16);
    __m256i b0 = _mm256_srli_si256(b,28-16);
    __m256i b1 = _mm256_slli_si256(b,32-28);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_29_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,29-16);
    __m256i b0 = _mm256_srli_si256(b,29-16);
    __m256i b1 = _mm256_slli_si256(b,32-29);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_30_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,30-16);
    __m256i b0 = _mm256_srli_si256(b,30-16);
    __m256i b1 = _mm256_slli_si256(b,32-30);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}
static inline __m256i _mm256_srli_31_si2x256(__m256i a, __m256i b)
{
    __m256i a0 = _mm256_srli_si256(a,31-16);
    __m256i b0 = _mm256_srli_si256(b,31-16);
    __m256i b1 = _mm256_slli_si256(b,32-31);
    __m256i c0 = _mm256_permute2x128_si256(a0,b0,0x21);
    return _mm256_adds_epi8(c0,b1);
}

typedef __m256i (*t_nrLDPC_mm256_srli_si2x256)(__m256i, __m256i);

static const t_nrLDPC_mm256_srli_si2x256 _mm256_srli_si2x256[32] = 
{
    _mm256_srli_0_si2x256,
    _mm256_srli_1_si2x256,
    _mm256_srli_2_si2x256,
    _mm256_srli_3_si2x256,
    _mm256_srli_4_si2x256,
    _mm256_srli_5_si2x256,
    _mm256_srli_6_si2x256,
    _mm256_srli_7_si2x256,
    _mm256_srli_8_si2x256,
    _mm256_srli_9_si2x256,
    _mm256_srli_10_si2x256,
    _mm256_srli_11_si2x256,
    _mm256_srli_12_si2x256,
    _mm256_srli_13_si2x256,
    _mm256_srli_14_si2x256,
    _mm256_srli_15_si2x256,
    _mm256_srli_16_si2x256,
    _mm256_srli_17_si2x256,
    _mm256_srli_18_si2x256,
    _mm256_srli_19_si2x256,
    _mm256_srli_20_si2x256,
    _mm256_srli_21_si2x256,
    _mm256_srli_22_si2x256,
    _mm256_srli_23_si2x256,
    _mm256_srli_24_si2x256,
    _mm256_srli_25_si2x256,
    _mm256_srli_26_si2x256,
    _mm256_srli_27_si2x256,
    _mm256_srli_28_si2x256,
    _mm256_srli_29_si2x256,
    _mm256_srli_30_si2x256,
    _mm256_srli_31_si2x256
};

#endif