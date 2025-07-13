/* SPDX-License-Identifier: BSD-3-Clause
 * Copyright(c) 2017 Intel Corporation
 */

/*! \file PHY/CODING/nrLDPC_coding/nrLDPC_coding_aal/nrLDPC_coding_al.c
 * \note: based on testbbdev test_bbdev_perf.c functions. Harq buffer offset added.
 * \mbuf and mempool allocated at the init step, LDPC parameters updated from OAI.
 */

#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_interface.h"
#include "PHY/CODING/nrLDPC_coding/nrLDPC_coding_aal/nrLDPC_coding_aal.h"
#include "PHY/sse_intrin.h"
#include <common/utils/LOG/log.h>
#define NR_LDPC_ENABLE_PARITY_CHECK

#include <stdint.h>
#include <getopt.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include <math.h>

#include <rte_eal.h>
#include <rte_common.h>
#include <rte_string_fns.h>
#include <rte_cycles.h>
#include <rte_lcore.h>
#include <rte_pdump.h>

#include <rte_dev.h>
#include <rte_launch.h>
#include <rte_bbdev.h>
#include <rte_malloc.h>
#include <rte_random.h>
#include <rte_hexdump.h>
#include <rte_interrupts.h>

// this socket is the NUMA socket, so the hardware CPU id (numa is complex)
#define GET_SOCKET(socket_id) (((socket_id) == SOCKET_ID_ANY) ? 0 : (socket_id))
#define MAX_QUEUES RTE_MAX_LCORE
#define OPS_CACHE_SIZE 256U
#define OPS_POOL_SIZE_MIN 511U /* 0.5K per queue */
#define SYNC_WAIT 0
#define SYNC_START 1
#define TIME_OUT_POLL 1e8
/* Increment for next code block in external HARQ memory */
#define HARQ_INCR 32768
/* Headroom for filler LLRs insertion in HARQ buffer */
#define FILLER_HEADROOM 2048
/* Number of segments that could be stored in HARQ combined buffers */
#define HARQ_CODEBLOCK_ID_MAX (16 << 5)

pthread_mutex_t encode_mutex;
pthread_mutex_t decode_mutex;
struct test_op_params *enc_op_params = NULL;
struct test_op_params *dec_op_params = NULL;

const char *typeStr[] = {
    "RTE_BBDEV_OP_NONE", /**< Dummy operation that does nothing */
    "RTE_BBDEV_OP_TURBO_DEC", /**< Turbo decode */
    "RTE_BBDEV_OP_TURBO_ENC", /**< Turbo encode */
    "RTE_BBDEV_OP_LDPC_DEC", /**< LDPC decode */
    "RTE_BBDEV_OP_LDPC_ENC", /**< LDPC encode */
    "RTE_BBDEV_OP_FFT", /**< Count of different op types */
};

struct active_device {
  const char *driver_name;
  uint8_t dev_id;

  // device capabilites
  struct rte_bbdev_info info;

  // queue resources
  // encoder queues: [0, ..., nb_queues - 1]
  // decoder queues: [nb_queues, ...]
  uint16_t queue_ids[MAX_QUEUES];
  uint16_t nb_queues;

  // ldpc encoder mempool resources
  struct rte_mempool *mp_enc_ops;
  struct rte_mempool *mp_enc_inputs;
  struct rte_mempool *mp_enc_hard_outputs;

  // ldpc decoder mempool resources
  struct rte_mempool *mp_dec_ops;
  struct rte_mempool *mp_dec_inputs;
  struct rte_mempool *mp_dec_hard_outputs;
  // NOTE: OAI does not support HARQ combining for now.
  // Thus, the following mempools are not used.
  struct rte_mempool *mp_dec_soft_outputs;
  struct rte_mempool *mp_dec_harq_inputs;
  struct rte_mempool *mp_dec_harq_outputs;
} active_dev; // we assume one active BBdevice only

/* Data buffers used by BBDEV ops */
struct data_buffers {
  struct rte_bbdev_op_data *inputs;
  struct rte_bbdev_op_data *hard_outputs;
  struct rte_bbdev_op_data *soft_outputs;
  struct rte_bbdev_op_data *harq_inputs;
  struct rte_bbdev_op_data *harq_outputs;
};

/* Operation parameters specific for given test case */
struct test_op_params {
  struct rte_mempool *mp_dec;
  struct rte_mempool *mp_enc;
  struct rte_bbdev_dec_op *ref_dec_op;
  struct rte_bbdev_enc_op *ref_enc_op;
  uint16_t burst_sz;
  uint16_t num_to_process;
  uint16_t num_lcores;
  int vector_mask;
  rte_atomic16_t sync;
  struct data_buffers q_bufs[RTE_MAX_NUMA_NODES][MAX_QUEUES];
};

/* Contains per lcore params */
struct thread_params {
  uint8_t dev_id;
  uint16_t queue_id;
  uint32_t lcore_id;
  nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters;
  nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters;
  uint8_t iter_count;
  rte_atomic16_t nb_dequeued;
  rte_atomic16_t processing_status;
  rte_atomic16_t burst_sz;
  struct test_op_params *op_params;
  struct rte_bbdev_dec_op *dec_ops[MAX_BURST];
  struct rte_bbdev_enc_op *enc_ops[MAX_BURST];
};

// static lookup table to store precomputed bit values for all possible bytes (0-255)
static uint8_t bit_unpack_table[256][8];
// flag to ensure the table is initialized only once
static int table_initialized = 0;

// function to initialize the lookup table
static void init_bit_unpack_table(void)
{
  for (int b = 0; b < 256; ++b) {
    for (int i = 0; i < 8; ++i) {
      // Extract bits from MSB (7) to LSB (0)
      bit_unpack_table[b][i] = (b >> (7 - i)) & 1;
    }
  }
  table_initialized = 1;
}

static uint16_t nb_segments_decoding(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  uint16_t nb_segments = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
    nb_segments += nrLDPC_slot_decoding_parameters->TBs[h].C;
  }
  return nb_segments;
}

static uint16_t nb_segments_encoding(nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  uint16_t nb_segments = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_encoding_parameters->nb_TBs; ++h) {
    nb_segments += nrLDPC_slot_encoding_parameters->TBs[h].C;
  }
  return nb_segments;
}

// DPDK BBDEV copy
static inline void mbuf_reset(struct rte_mbuf *m)
{
  m->pkt_len = 0;

  do {
    m->data_len = 0;
    m = m->next;
  } while (m != NULL);
}

/* Read flag value 0/1 from bitmap */
// DPDK BBDEV copy
static inline bool check_bit(uint32_t bitmap, uint32_t bitmask)
{
  return bitmap & bitmask;
}

/* calculates optimal mempool size not smaller than the val */
// DPDK BBDEV copy
static unsigned int optimal_mempool_size(unsigned int val)
{
  return rte_align32pow2(val + 1) - 1;
}

// based on DPDK BBDEV create_mempools
static int create_mempools(struct active_device *ad, int socket_id, uint16_t num_ops, int out_buff_sz, int in_max_sz)
{
  unsigned int ops_pool_size, mbuf_pool_size, data_room_size = 0;
  uint8_t nb_segments = 1;

  // allocate enc/dec ops mempool
  ops_pool_size = optimal_mempool_size(RTE_MAX(
      /* Ops used plus 1 reference op */
      RTE_MAX((unsigned int)(ad->nb_queues * num_ops + 1),
              /* Minimal cache size plus 1 reference op */
              (unsigned int)(1.5 * rte_lcore_count() * OPS_CACHE_SIZE + 1)),
      OPS_POOL_SIZE_MIN));

  // encoder ops pool
  ad->mp_enc_ops = rte_bbdev_op_pool_create("mp_enc_ops", RTE_BBDEV_OP_LDPC_ENC, ops_pool_size, OPS_CACHE_SIZE, socket_id);
  AssertFatal(ad->mp_enc_ops != NULL,
              "ERROR Failed to create %u items ops pool for dev %u on socket %d.",
              ops_pool_size,
              ad->dev_id,
              socket_id);

  // decoder ops pool
  ad->mp_dec_ops = rte_bbdev_op_pool_create("mp_dec_ops", RTE_BBDEV_OP_LDPC_DEC, ops_pool_size, OPS_CACHE_SIZE, socket_id);
  AssertFatal(ad->mp_dec_ops != NULL,
              "ERROR Failed to create %u items ops pool for dev %u on socket %d.",
              ops_pool_size,
              ad->dev_id,
              socket_id);

  // encoder resources
  if (ad->mp_enc_ops) {
    mbuf_pool_size = optimal_mempool_size(ops_pool_size * nb_segments);
    // inputs
    data_room_size = RTE_MAX(in_max_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    ad->mp_enc_inputs = rte_pktmbuf_pool_create("mp_enc_inputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_enc_inputs != NULL,
                "ERROR Failed to create %u items input pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);

    // hard outputs
    data_room_size = RTE_MAX(out_buff_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    ad->mp_enc_hard_outputs = rte_pktmbuf_pool_create("mp_enc_hard_outputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_enc_hard_outputs != NULL,
                "ERROR Failed to create %u items hard output pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);
  }

  // decoder resources
  if (ad->mp_dec_ops) {
    mbuf_pool_size = optimal_mempool_size(ops_pool_size * nb_segments);

    // inputs
    data_room_size = RTE_MAX(in_max_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    ad->mp_dec_inputs = rte_pktmbuf_pool_create("mp_dec_inputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_dec_inputs != NULL,
                "ERROR Failed to create %u items input pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);

    // hard outputs
    data_room_size = RTE_MAX(out_buff_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    ad->mp_dec_hard_outputs = rte_pktmbuf_pool_create("mp_dec_hard_outputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_dec_hard_outputs != NULL,
                "ERROR Failed to create %u items hard output pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);

    // soft outputs
    data_room_size = RTE_MAX(out_buff_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    ad->mp_dec_soft_outputs = rte_pktmbuf_pool_create("mp_dec_soft_outputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_dec_soft_outputs != NULL,
                "ERROR Failed to create %u items soft output pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);

    // HARQ inputs
    data_room_size = RTE_MAX(in_max_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    ad->mp_dec_harq_inputs = rte_pktmbuf_pool_create("mp_dec_harq_inputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_dec_harq_inputs != NULL,
                "ERROR Failed to create %u items HARQ input pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);

    // HARQ outputs
    data_room_size = RTE_MAX(out_buff_sz + RTE_PKTMBUF_HEADROOM + FILLER_HEADROOM, (unsigned int)RTE_MBUF_DEFAULT_BUF_SIZE);
    data_room_size = 2.5 * data_room_size; // TODO: not sure why we need 2.5? some calculation is off.
    ad->mp_dec_harq_outputs = rte_pktmbuf_pool_create("mp_dec_harq_outputs", mbuf_pool_size, 0, 0, data_room_size, socket_id);
    AssertFatal(ad->mp_dec_harq_outputs != NULL,
                "ERROR Failed to create %u items HARQ output pktmbuf pool for dev %u on socket %d.",
                mbuf_pool_size,
                ad->dev_id,
                socket_id);
  }
  return 0;
}

const char *ldpcenc_flag_bitmask[] = {
    /** Set for bit-level interleaver bypass on output stream. */
    "RTE_BBDEV_LDPC_INTERLEAVER_BYPASS",
    /** If rate matching is to be performed */
    "RTE_BBDEV_LDPC_RATE_MATCH",
    /** Set for transport block CRC-24A attach */
    "RTE_BBDEV_LDPC_CRC_24A_ATTACH",
    /** Set for code block CRC-24B attach */
    "RTE_BBDEV_LDPC_CRC_24B_ATTACH",
    /** Set for code block CRC-16 attach */
    "RTE_BBDEV_LDPC_CRC_16_ATTACH",
    /** Set if a device supports encoder dequeue interrupts. */
    "RTE_BBDEV_LDPC_ENC_INTERRUPTS",
    /** Set if a device supports scatter-gather functionality. */
    "RTE_BBDEV_LDPC_ENC_SCATTER_GATHER",
    /** Set if a device supports concatenation of non byte aligned output */
    "RTE_BBDEV_LDPC_ENC_CONCATENATION",
};

const char *ldpcdec_flag_bitmask[] = {
    /** Set for transport block CRC-24A checking */
    "RTE_BBDEV_LDPC_CRC_TYPE_24A_CHECK",
    /** Set for code block CRC-24B checking */
    "RTE_BBDEV_LDPC_CRC_TYPE_24B_CHECK",
    /** Set to drop the last CRC bits decoding output */
    "RTE_BBDEV_LDPC_CRC_TYPE_24B_DROP"
    /** Set for bit-level de-interleaver bypass on Rx stream. */
    "RTE_BBDEV_LDPC_DEINTERLEAVER_BYPASS",
    /** Set for HARQ combined input stream enable. */
    "RTE_BBDEV_LDPC_HQ_COMBINE_IN_ENABLE",
    /** Set for HARQ combined output stream enable. */
    "RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE",
    /** Set for LDPC decoder bypass.
     *  RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE must be set.
     */
    "RTE_BBDEV_LDPC_DECODE_BYPASS",
    /** Set for soft-output stream enable */
    "RTE_BBDEV_LDPC_SOFT_OUT_ENABLE",
    /** Set for Rate-Matching bypass on soft-out stream. */
    "RTE_BBDEV_LDPC_SOFT_OUT_RM_BYPASS",
    /** Set for bit-level de-interleaver bypass on soft-output stream. */
    "RTE_BBDEV_LDPC_SOFT_OUT_DEINTERLEAVER_BYPASS",
    /** Set for iteration stopping on successful decode condition
     *  i.e. a successful syndrome check.
     */
    "RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE",
    /** Set if a device supports decoder dequeue interrupts. */
    "RTE_BBDEV_LDPC_DEC_INTERRUPTS",
    /** Set if a device supports scatter-gather functionality. */
    "RTE_BBDEV_LDPC_DEC_SCATTER_GATHER",
    /** Set if a device supports input/output HARQ compression. */
    "RTE_BBDEV_LDPC_HARQ_6BIT_COMPRESSION",
    /** Set if a device supports input LLR compression. */
    "RTE_BBDEV_LDPC_LLR_COMPRESSION",
    /** Set if a device supports HARQ input from
     *  device's internal memory.
     */
    "RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_IN_ENABLE",
    /** Set if a device supports HARQ output to
     *  device's internal memory.
     */
    "RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_OUT_ENABLE",
    /** Set if a device supports loop-back access to
     *  HARQ internal memory. Intended for troubleshooting.
     */
    "RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_LOOPBACK",
    /** Set if a device includes LLR filler bits in the circular buffer
     *  for HARQ memory. If not set, it is assumed the filler bits are not
     *  in HARQ memory and handled directly by the LDPC decoder.
     */
    "RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_FILLERS",
};

void debug_dev_capabilities(uint8_t dev_id, struct rte_bbdev_info *info)
{
  /* Display for debug the capabilities of the card */
  for (int i = 0; info->drv.capabilities[i].type != RTE_BBDEV_OP_NONE; i++) {
    printf("device: %d, capability[%d]=%s\n", dev_id, i, typeStr[info->drv.capabilities[i].type]);
    if (info->drv.capabilities[i].type == RTE_BBDEV_OP_LDPC_ENC) {
      const struct rte_bbdev_op_cap_ldpc_enc cap = info->drv.capabilities[i].cap.ldpc_enc;
      printf("    buffers: src = %d, dst = %d\n   capabilites: ", cap.num_buffers_src, cap.num_buffers_dst);
      for (int j = 0; j < sizeof(cap.capability_flags) * 8; j++)
        if (cap.capability_flags & (1ULL << j))
          printf("%s ", ldpcenc_flag_bitmask[j]);
      printf("\n");
    }
    if (info->drv.capabilities[i].type == RTE_BBDEV_OP_LDPC_DEC) {
      const struct rte_bbdev_op_cap_ldpc_dec cap = info->drv.capabilities[i].cap.ldpc_dec;
      printf("    buffers: src = %d, hard out = %d, soft_out %d, llr size %d, llr decimals %d \n   capabilities: ",
             cap.num_buffers_src,
             cap.num_buffers_hard_out,
             cap.num_buffers_soft_out,
             cap.llr_size,
             cap.llr_decimals);
      for (int j = 0; j < sizeof(cap.capability_flags) * 8; j++)
        if (cap.capability_flags & (1ULL << j))
          printf("%s ", ldpcdec_flag_bitmask[j]);
      printf("\n");
    }
  }
}

void check_required_dev_capabilities(struct rte_bbdev_info *info)
{
  // check ldpc enc/ dec support
  bool ldpc_enc = false;
  bool ldpc_dec = false;
  for (int i = 0; info->drv.capabilities[i].type != RTE_BBDEV_OP_NONE; i++) {
    if (info->drv.capabilities[i].type == RTE_BBDEV_OP_LDPC_ENC) {
      ldpc_enc = true;
    }
    if (info->drv.capabilities[i].type == RTE_BBDEV_OP_LDPC_DEC) {
      ldpc_dec = true;
    }
  }
  AssertFatal(ldpc_enc, "ERROR: bbdev device does not support LDPC encoding\n");
  AssertFatal(ldpc_dec, "ERROR: bbdev device does not support LDPC decoding\n");

  // check encoding capabilities
  bool rate_match = check_bit(info->drv.capabilities[RTE_BBDEV_OP_LDPC_ENC].cap.ldpc_enc.capability_flags, RTE_BBDEV_LDPC_RATE_MATCH);
  AssertFatal(rate_match, "ERROR: bbdev device does not support LDPC encoding with rate matching\n");

  // check decoding capabilities
  bool iter_stop = check_bit(info->drv.capabilities[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.capability_flags, RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE);
  AssertFatal(iter_stop, "ERROR: bbdev device does not support LDPC decoding with iteration stop\n");

  bool crc_24b_drop = check_bit(info->drv.capabilities[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.capability_flags, RTE_BBDEV_LDPC_CRC_TYPE_24B_DROP);
  AssertFatal(crc_24b_drop, "ERROR: bbdev device does not support LDPC decoding with CRC-24B drop\n");

  bool crc_24b_check = check_bit(info->drv.capabilities[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.capability_flags, RTE_BBDEV_LDPC_CRC_TYPE_24B_CHECK);
  AssertFatal(crc_24b_check, "ERROR: bbdev device does not support LDPC decoding with CRC-24B check\n");
}

static bool check_non24b_crc(struct rte_bbdev_info *info)
{
  bool crc_24a_check = check_bit(info->drv.capabilities[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.capability_flags, RTE_BBDEV_LDPC_CRC_TYPE_24A_CHECK);
  bool crc_16_check = check_bit(info->drv.capabilities[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.capability_flags, RTE_BBDEV_LDPC_CRC_TYPE_16_CHECK);
  bool non_24b_crc = crc_24a_check & crc_16_check;
  return non_24b_crc;
}

// based on DPDK BBDEV add_bbdev_dev
static int add_dev(uint8_t dev_id)
{
  int ret;
  unsigned int queue_id;
  unsigned int nb_queues;
  struct rte_bbdev_queue_conf qconf;

  // retrieve device capabilities
  rte_bbdev_info_get(dev_id, &active_dev.info);
  printf("using bbdev %d: %s\n", dev_id, active_dev.info.dev_name);

  active_dev.driver_name = active_dev.info.drv.driver_name;
  active_dev.dev_id = dev_id;

  nb_queues = RTE_MIN(rte_lcore_count(), active_dev.info.drv.max_num_queues);
  nb_queues = RTE_MIN(nb_queues, (unsigned int)MAX_QUEUES);

  // debug device capabilities
  debug_dev_capabilities(dev_id, &active_dev.info);

  // check required device capabilities
  check_required_dev_capabilities(&active_dev.info);

  // device setup
  ret = rte_bbdev_setup_queues(dev_id, nb_queues, active_dev.info.socket_id);
  AssertFatal(ret == 0, "rte_bbdev_setup_queues(%u, %u, %d) ret %i\n", dev_id, nb_queues, active_dev.info.socket_id, ret);

  // queue setup
  qconf.socket = active_dev.info.socket_id;
  qconf.queue_size = active_dev.info.drv.default_queue_conf.queue_size;
  qconf.priority = 0;
  qconf.deferred_start = 0;

  // here, we allocate queues equally among encoding and decoding
  AssertFatal(nb_queues % 2 == 0, "ERROR: odd number of queues %u", nb_queues);
  printf("total %u queues available\n", nb_queues);
  printf("to allocate %u queues for encoding and %u queues for decoding\n", nb_queues / 2, nb_queues / 2);

  for (queue_id = 0; queue_id < nb_queues / 2; queue_id++) {
    qconf.op_type = RTE_BBDEV_OP_LDPC_ENC;
    ret = rte_bbdev_queue_configure(dev_id, queue_id, &qconf);
    if (ret != 0) {
      break;
    } else {
      printf("allocated LDPC encoding queue (id=%u) at prio%u on dev%u\n", queue_id, qconf.priority, dev_id);
    }
    active_dev.queue_ids[queue_id] = queue_id;
  }
  AssertFatal(queue_id == (nb_queues / 2), "ERROR Failed to configure encoding queues on dev %u", dev_id);

  for (queue_id = nb_queues / 2; queue_id < nb_queues; queue_id++) {
    qconf.op_type = RTE_BBDEV_OP_LDPC_DEC;
    ret = rte_bbdev_queue_configure(dev_id, queue_id, &qconf);
    if (ret != 0) {
      break;
    } else {
      printf("allocated LDPC decoding queue (id=%u) at prio%u on dev%u\n", queue_id, qconf.priority, dev_id);
    }
    active_dev.queue_ids[queue_id] = queue_id;
  }
  AssertFatal(queue_id == nb_queues, "ERROR Failed to configure decoding queues on dev %u", dev_id);

  active_dev.nb_queues = nb_queues / 2;
  return 0;
}

// based on DPDK BBDEV init_op_data_objs
static int init_op_data_objs_dec(struct rte_bbdev_op_data *bufs,
                                 int8_t *input,
                                 nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters,
                                 struct rte_mempool *mbuf_pool,
                                 enum op_data_type op_type,
                                 uint16_t min_alignment)
{
  bool large_input = false;
  uint16_t j = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
    for (uint16_t i = 0; i < nrLDPC_slot_decoding_parameters->TBs[h].C; ++i) {
      uint32_t data_len = nrLDPC_slot_decoding_parameters->TBs[h].segments[i].E * sizeof(int8_t);
      char *data;
      struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
      AssertFatal(m_head != NULL,
                  "Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
                  op_type,
                  nb_segments_decoding(nrLDPC_slot_decoding_parameters),
                  mbuf_pool->size);

      if (data_len > RTE_BBDEV_LDPC_E_MAX_MBUF) {
        printf("Warning: Larger input size than DPDK mbuf %u\n", data_len);
        large_input = true;
      }
      bufs[j].data = m_head;
      bufs[j].offset = 0;
      bufs[j].length = 0;

      if ((op_type == DATA_INPUT) || (op_type == DATA_HARQ_INPUT)) {
        if ((op_type == DATA_INPUT) && large_input) {
          /* Allocate a fake overused mbuf */
          data = rte_malloc(NULL, data_len, 0);
          AssertFatal(data != NULL, "rte malloc failed with %u bytes", data_len);
          memcpy(data, &input[j * LDPC_MAX_CB_SIZE], data_len);
          m_head->buf_addr = data;
          m_head->buf_iova = rte_malloc_virt2iova(data);
          m_head->data_off = 0;
          m_head->data_len = data_len;
        } else {
          rte_pktmbuf_reset(m_head);
          data = rte_pktmbuf_append(m_head, data_len);
          AssertFatal(data != NULL, "Couldn't append %u bytes to mbuf from %d data type mbuf pool", data_len, op_type);
          AssertFatal(data == RTE_PTR_ALIGN(data, min_alignment),
                      "Data addr in mbuf (%p) is not aligned to device min alignment (%u)",
                      data,
                      min_alignment);
          rte_memcpy(data, &input[j * LDPC_MAX_CB_SIZE], data_len);
        }
        bufs[j].length += data_len;
      }
      ++j;
    }
  }
  return 0;
}

// based on DPDK BBDEV init_op_data_objs
static int init_op_data_objs_enc(struct rte_bbdev_op_data *bufs,
                                 nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters,
                                 struct rte_mempool *mbuf_pool,
                                 enum op_data_type op_type,
                                 uint16_t min_alignment)
{
  bool large_input = false;
  uint16_t j = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_encoding_parameters->nb_TBs; ++h) {
    for (int i = 0; i < nrLDPC_slot_encoding_parameters->TBs[h].C; ++i) {
      uint32_t data_len = (nrLDPC_slot_encoding_parameters->TBs[h].K - nrLDPC_slot_encoding_parameters->TBs[h].F + 7) / 8;
      char *data;
      struct rte_mbuf *m_head = rte_pktmbuf_alloc(mbuf_pool);
      AssertFatal(m_head != NULL,
                  "Not enough mbufs in %d data type mbuf pool (needed %u, available %u)",
                  op_type,
                  nb_segments_encoding(nrLDPC_slot_encoding_parameters),
                  mbuf_pool->size);

      if (data_len > RTE_BBDEV_LDPC_E_MAX_MBUF) {
        printf("Warning: Larger input size than DPDK mbuf %u\n", data_len);
        large_input = true;
      }
      bufs[j].data = m_head;
      bufs[j].offset = 0;
      bufs[j].length = 0;

      if ((op_type == DATA_INPUT) || (op_type == DATA_HARQ_INPUT)) {
        if ((op_type == DATA_INPUT) && large_input) {
          /* Allocate a fake overused mbuf */
          data = rte_malloc(NULL, data_len, 0);
          AssertFatal(data != NULL, "rte malloc failed with %u bytes", data_len);
          memcpy(data, nrLDPC_slot_encoding_parameters->TBs[h].segments[i].c, data_len);
          m_head->buf_addr = data;
          m_head->buf_iova = rte_malloc_virt2iova(data);
          m_head->data_off = 0;
          m_head->data_len = data_len;
        } else {
          rte_pktmbuf_reset(m_head);
          data = rte_pktmbuf_append(m_head, data_len);
          AssertFatal(data != NULL, "Couldn't append %u bytes to mbuf from %d data type mbuf pool", data_len, op_type);
          AssertFatal(data == RTE_PTR_ALIGN(data, min_alignment),
                      "Data addr in mbuf (%p) is not aligned to device min alignment (%u)",
                      data,
                      min_alignment);
          rte_memcpy(data, nrLDPC_slot_encoding_parameters->TBs[h].segments[i].c, data_len);
        }
        bufs[j].length += data_len;
      }
      ++j;
    }
  }
  return 0;
}

// DPDK BBEV copy
static int allocate_buffers_on_socket(struct rte_bbdev_op_data **buffers, const int len, const int socket)
{
  int i;

  *buffers = rte_zmalloc_socket(NULL, len, 0, socket);
  if (*buffers == NULL) {
    printf("WARNING: Failed to allocate op_data on socket %d\n", socket);
    /* try to allocate memory on other detected sockets */
    for (i = 0; i < socket; i++) {
      *buffers = rte_zmalloc_socket(NULL, len, 0, i);
      if (*buffers != NULL)
        break;
    }
  }

  return (*buffers == NULL) ? -1 : 0;
}

// DPDK BBDEV copy
static void free_buffers(struct active_device *ad, struct test_op_params *op_params)
{
  // encoder resources
  rte_mempool_free(ad->mp_enc_ops);
  rte_mempool_free(ad->mp_enc_inputs);
  rte_mempool_free(ad->mp_enc_hard_outputs);

  // decoder resources
  rte_mempool_free(ad->mp_dec_ops);
  rte_mempool_free(ad->mp_dec_inputs);
  rte_mempool_free(ad->mp_dec_hard_outputs);
  rte_mempool_free(ad->mp_dec_soft_outputs);
  rte_mempool_free(ad->mp_dec_harq_inputs);
  rte_mempool_free(ad->mp_dec_harq_outputs);

  for (int i = 2; i < rte_lcore_count(); ++i) {
    for (int j = 0; j < RTE_MAX_NUMA_NODES; ++j) {
      rte_free(op_params->q_bufs[j][i].inputs);
      rte_free(op_params->q_bufs[j][i].hard_outputs);
      rte_free(op_params->q_bufs[j][i].soft_outputs);
      rte_free(op_params->q_bufs[j][i].harq_inputs);
      rte_free(op_params->q_bufs[j][i].harq_outputs);
    }
  }
}

// based on DPDK BBDEV copy_reference_ldpc_dec_op
static void set_ldpc_dec_op(struct rte_bbdev_dec_op **ops,
                            unsigned int start_idx,
                            struct data_buffers *bufs,
                            nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  unsigned int h;
  unsigned int i;
  unsigned int j = 0;
  for (h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
    for (i = 0; i < nrLDPC_slot_decoding_parameters->TBs[h].C; ++i) {
      ops[j]->ldpc_dec.basegraph = nrLDPC_slot_decoding_parameters->TBs[h].BG;
      ops[j]->ldpc_dec.z_c = nrLDPC_slot_decoding_parameters->TBs[h].Z;
      ops[j]->ldpc_dec.q_m = nrLDPC_slot_decoding_parameters->TBs[h].Qm;
      ops[j]->ldpc_dec.n_filler = nrLDPC_slot_decoding_parameters->TBs[h].F;
      ops[j]->ldpc_dec.n_cb = (nrLDPC_slot_decoding_parameters->TBs[h].BG == 1) ? (66 * nrLDPC_slot_decoding_parameters->TBs[h].Z)
                                                                                : (50 * nrLDPC_slot_decoding_parameters->TBs[h].Z);
      ops[j]->ldpc_dec.iter_max = nrLDPC_slot_decoding_parameters->TBs[h].max_ldpc_iterations;
      ops[j]->ldpc_dec.rv_index = nrLDPC_slot_decoding_parameters->TBs[h].rv_index;
      ops[j]->ldpc_dec.op_flags = RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE;
      // NOTE: OAI does not support HARQ combining for now.
      // ops[j]->ldpc_dec.op_flags = RTE_BBDEV_LDPC_ITERATION_STOP_ENABLE |
      //                             RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_IN_ENABLE |
      //                             RTE_BBDEV_LDPC_INTERNAL_HARQ_MEMORY_OUT_ENABLE |
      //                             RTE_BBDEV_LDPC_HQ_COMBINE_OUT_ENABLE;
      if (*nrLDPC_slot_decoding_parameters->TBs[h].segments[i].d_to_be_cleared) {
        *nrLDPC_slot_decoding_parameters->TBs[h].segments[i].d_to_be_cleared = false;
        *nrLDPC_slot_decoding_parameters->TBs[h].processedSegments = 0;
      } else {
        // NOTE: OAI does not support HARQ combining for now.
        // ops[j]->ldpc_dec.op_flags |= RTE_BBDEV_LDPC_HQ_COMBINE_IN_ENABLE;
      }
      if (nrLDPC_slot_decoding_parameters->TBs[h].C > 1) {
        ops[j]->ldpc_dec.code_block_mode = 1;
        ops[j]->ldpc_dec.cb_params.e = nrLDPC_slot_decoding_parameters->TBs[h].segments[i].E;
        ops[j]->ldpc_dec.op_flags |= RTE_BBDEV_LDPC_CRC_TYPE_24B_DROP;
        ops[j]->ldpc_dec.op_flags |= RTE_BBDEV_LDPC_CRC_TYPE_24B_CHECK;
      } else {
        /**
         * This is a special case when #TB = 1 and #CB = 1
         * In this case, we must use TB mode
         * Quoted from: https://doc.dpdk.org/guides-23.11/prog_guide/bbdev.html#bbdev-ldpc-decode-operation
         * The case when one CB belongs to TB and is being enqueued individually to BBDEV, this case is considered as a
         * special case of partial TB where its number of CBs is 1. Therefore, it requires to get processed in TB-mode.
         */
        ops[j]->ldpc_dec.code_block_mode = 0;
        ops[j]->ldpc_dec.tb_params.c = 1;
        ops[j]->ldpc_dec.tb_params.r = 0;
        ops[j]->ldpc_dec.tb_params.cab = 1;
        ops[j]->ldpc_dec.tb_params.ea = nrLDPC_slot_decoding_parameters->TBs[h].segments[i].E;
        ops[j]->ldpc_dec.tb_params.eb = nrLDPC_slot_decoding_parameters->TBs[h].segments[i].E;
        bool non24b_crc = check_non24b_crc(&active_dev.info);
        if(non24b_crc) {
          uint8_t crc_type = crcType(nrLDPC_slot_decoding_parameters->TBs[h].C, nrLDPC_slot_decoding_parameters->TBs[h].A);
          if(crc_type == 0)     // CRC_24A
            ops[j]->ldpc_dec.op_flags |= RTE_BBDEV_LDPC_CRC_TYPE_24A_CHECK;
          else if(crc_type == 2) // CRC_16
            ops[j]->ldpc_dec.op_flags |= RTE_BBDEV_LDPC_CRC_TYPE_16_CHECK;
          else
            AssertFatal(0, "ERROR: Unsupported CRC type %d\n", crc_type);
        }
      }

      // Calculate offset in the HARQ combined buffers
      // Unique segment offset
      // uint32_t segment_offset = (nrLDPC_slot_decoding_parameters->TBs[h].harq_unique_pid * NR_LDPC_MAX_NUM_CB) + i;
      // Prune to avoid shooting above maximum id
      // uint32_t pruned_segment_offset = segment_offset % HARQ_CODEBLOCK_ID_MAX;
      // Segment offset to byte offset
      // uint32_t harq_combined_offset = pruned_segment_offset * LDPC_MAX_CB_SIZE;

      // ops[j]->ldpc_dec.harq_combined_input.offset = harq_combined_offset;
      // ops[j]->ldpc_dec.harq_combined_output.offset = harq_combined_offset;

      if (bufs->hard_outputs != NULL)
        ops[j]->ldpc_dec.hard_output = bufs->hard_outputs[start_idx + j];
      if (bufs->inputs != NULL)
        ops[j]->ldpc_dec.input = bufs->inputs[start_idx + j];
      // if (bufs->soft_outputs != NULL)
      //   ops[j]->ldpc_dec.soft_output = bufs->soft_outputs[start_idx + j];
      // if (bufs->harq_inputs != NULL)
      //   ops[j]->ldpc_dec.harq_combined_input = bufs->harq_inputs[start_idx + j];
      // if (bufs->harq_outputs != NULL)
      //   ops[j]->ldpc_dec.harq_combined_output = bufs->harq_outputs[start_idx + j];
      ++j;
    }
  }
}

// based on DPDK BBDEV copy_reference_ldpc_enc_op
static void set_ldpc_enc_op(struct rte_bbdev_enc_op **ops,
                            unsigned int start_idx,
                            struct rte_bbdev_op_data *inputs,
                            struct rte_bbdev_op_data *outputs,
                            nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  unsigned int h;
  unsigned int i;
  unsigned int j = 0;
  for (h = 0; h < nrLDPC_slot_encoding_parameters->nb_TBs; ++h) {
    for (i = 0; i < nrLDPC_slot_encoding_parameters->TBs[h].C; ++i) {
      ops[j]->ldpc_enc.basegraph = nrLDPC_slot_encoding_parameters->TBs[h].BG;
      ops[j]->ldpc_enc.z_c = nrLDPC_slot_encoding_parameters->TBs[h].Z;
      ops[j]->ldpc_enc.q_m = nrLDPC_slot_encoding_parameters->TBs[h].Qm;
      ops[j]->ldpc_enc.n_filler = nrLDPC_slot_encoding_parameters->TBs[h].F;
      ops[j]->ldpc_enc.n_cb = (nrLDPC_slot_encoding_parameters->TBs[h].BG == 1) ? (66 * nrLDPC_slot_encoding_parameters->TBs[h].Z)
      : (50 * nrLDPC_slot_encoding_parameters->TBs[h].Z);
      if (nrLDPC_slot_encoding_parameters->TBs[h].tbslbrm != 0) {
        uint32_t Nref = 3 * nrLDPC_slot_encoding_parameters->TBs[h].tbslbrm / (2 * nrLDPC_slot_encoding_parameters->TBs[h].C);
        ops[j]->ldpc_enc.n_cb = min(ops[j]->ldpc_enc.n_cb, Nref);
      }
      ops[j]->ldpc_enc.rv_index = nrLDPC_slot_encoding_parameters->TBs[h].rv_index;
      ops[j]->ldpc_enc.op_flags = RTE_BBDEV_LDPC_RATE_MATCH;
      if(nrLDPC_slot_encoding_parameters->TBs[h].C > 1) {
        ops[j]->ldpc_enc.code_block_mode = 1;
        ops[j]->ldpc_enc.cb_params.e = nrLDPC_slot_encoding_parameters->TBs[h].segments[i].E;
      } else {
        /**
         * This is a special case when #TB = 1 and #CB = 1
         * In this case, we must use TB mode
         * Quoted from: https://doc.dpdk.org/guides-23.11/prog_guide/bbdev.html#bbdev-ldpc-decode-operation
         * The case when one CB belongs to TB and is being enqueued individually to BBDEV, this case is considered as a
         * special case of partial TB where its number of CBs is 1. Therefore, it requires to get processed in TB-mode.
         */
        ops[j]->ldpc_enc.code_block_mode = 0;
        ops[j]->ldpc_enc.tb_params.c = 1;
        ops[j]->ldpc_enc.tb_params.r = 0;
        ops[j]->ldpc_enc.tb_params.cab = 1;
        ops[j]->ldpc_enc.tb_params.ea = nrLDPC_slot_encoding_parameters->TBs[h].segments[i].E;
        ops[j]->ldpc_enc.tb_params.eb = nrLDPC_slot_encoding_parameters->TBs[h].segments[i].E;
      }

      ops[j]->ldpc_enc.output = outputs[start_idx + j];
      ops[j]->ldpc_enc.input = inputs[start_idx + j];
      ++j;
    }
  }
}

static int retrieve_ldpc_dec_op(struct rte_bbdev_dec_op **ops,
                                const int vector_mask,
                                nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  struct rte_bbdev_op_data *hard_output;
  uint16_t data_len = 0;
  struct rte_mbuf *m;
  char *data;
  unsigned int h;
  unsigned int i;
  unsigned int j = 0;
  bool decode_success = false;
  for (h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
    for (i = 0; i < nrLDPC_slot_decoding_parameters->TBs[h].C; ++i) {
      hard_output = &ops[j]->ldpc_dec.hard_output;
      m = hard_output->data;
      data_len = rte_pktmbuf_data_len(m) - hard_output->offset;
      data = m->buf_addr;
      memcpy(nrLDPC_slot_decoding_parameters->TBs[h].segments[i].c, data + m->data_off, data_len);
      rte_pktmbuf_free(ops[j]->ldpc_dec.hard_output.data);
      rte_pktmbuf_free(ops[j]->ldpc_dec.input.data);
      rte_pktmbuf_free(ops[j]->ldpc_dec.soft_output.data);
      rte_pktmbuf_free(ops[j]->ldpc_dec.harq_combined_input.data);
      rte_pktmbuf_free(ops[j]->ldpc_dec.harq_combined_output.data);
      ++j;
    }
  }
  return 0;
}

static int retrieve_ldpc_enc_op(struct rte_bbdev_enc_op **ops, nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  if (!table_initialized) {
    init_bit_unpack_table();
  }

  unsigned int j = 0;
  for (unsigned int h = 0; h < nrLDPC_slot_encoding_parameters->nb_TBs; ++h) {
    for (unsigned int i = 0; i < nrLDPC_slot_encoding_parameters->TBs[h].C; ++i) {
      struct rte_bbdev_op_data *output = &ops[j]->ldpc_enc.output;
      struct rte_mbuf *m = output->data;
      uint16_t data_len = rte_pktmbuf_data_len(m) - output->offset;
      uint8_t *out = nrLDPC_slot_encoding_parameters->TBs[h].segments[i].output;
      const char *data = m->buf_addr + m->data_off;
      const char *end = data + data_len;

      while (data < end) {
        uint8_t byte = *data++;
        memcpy(out, bit_unpack_table[byte], 8);
        out += 8;
      }
      rte_pktmbuf_free(ops[j]->ldpc_enc.output.data);
      rte_pktmbuf_free(ops[j]->ldpc_enc.input.data);
      ++j;
    }
  }
  return 0;
}

// based on DPDK BBDEV init_test_op_params
static int init_test_op_params(struct test_op_params *op_params,
                               enum rte_bbdev_op_type op_type,
                               struct rte_mempool *ops_mp,
                               uint16_t burst_sz,
                               uint16_t num_to_process,
                               uint16_t num_lcores)
{
  int ret = 0;
  if (op_type == RTE_BBDEV_OP_LDPC_DEC) {
    ret = rte_bbdev_dec_op_alloc_bulk(ops_mp, &op_params->ref_dec_op, num_to_process);
    op_params->mp_dec = ops_mp;
  } else {
    ret = rte_bbdev_enc_op_alloc_bulk(ops_mp, &op_params->ref_enc_op, 1);
    op_params->mp_enc = ops_mp;
  }
  AssertFatal(ret == 0, "rte_bbdev_op_alloc_bulk() failed");

  op_params->burst_sz = burst_sz;
  op_params->num_to_process = num_to_process;
  op_params->num_lcores = num_lcores;
  return 0;
}

// based on DPDK BBDEV throughput_pmd_lcore_ldpc_dec
static int pmd_lcore_ldpc_dec(void *arg)
{
  struct thread_params *tp = arg;
  nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters = tp->nrLDPC_slot_decoding_parameters;
  uint16_t enq, deq;
  int time_out = 0;
  const uint16_t queue_id = tp->queue_id;
  const uint16_t num_segments = nb_segments_decoding(nrLDPC_slot_decoding_parameters);
  struct rte_bbdev_dec_op **ops_enq;
  struct rte_bbdev_dec_op **ops_deq;
  ops_enq = (struct rte_bbdev_dec_op **)
      rte_calloc("struct rte_bbdev_dec_op **ops_enq", num_segments, sizeof(struct rte_bbdev_dec_op *), RTE_CACHE_LINE_SIZE);
  ops_deq = (struct rte_bbdev_dec_op **)
      rte_calloc("struct rte_bbdev_dec_op **ops_dec", num_segments, sizeof(struct rte_bbdev_dec_op *), RTE_CACHE_LINE_SIZE);
  struct data_buffers *bufs = NULL;
  uint16_t h, i, j;
  int ret;
  uint16_t num_to_enq;


  AssertFatal((num_segments < MAX_BURST), "BURST_SIZE should be <= %u", MAX_BURST);

  bufs = &tp->op_params->q_bufs[active_dev.info.socket_id][queue_id];
  ret = rte_bbdev_dec_op_alloc_bulk(tp->op_params->mp_dec, ops_enq, num_segments);
  AssertFatal(ret == 0, "Allocation failed for %d ops", num_segments);
  set_ldpc_dec_op(ops_enq, 0, bufs, nrLDPC_slot_decoding_parameters);

  for (enq = 0, deq = 0; enq < num_segments;) {
    num_to_enq = num_segments;
    if (unlikely(num_segments - enq < num_to_enq))
      num_to_enq = num_segments - enq;

    enq += rte_bbdev_enqueue_ldpc_dec_ops(tp->dev_id, queue_id, &ops_enq[enq], num_to_enq);
    deq += rte_bbdev_dequeue_ldpc_dec_ops(tp->dev_id, queue_id, &ops_deq[deq], enq - deq);
  }

  /* dequeue the remaining */
  while (deq < enq) {
    deq += rte_bbdev_dequeue_ldpc_dec_ops(tp->dev_id, queue_id, &ops_deq[deq], enq - deq);
    time_out++;
    DevAssert(time_out <= TIME_OUT_POLL);
  }
  if (deq == enq) {
    ret = retrieve_ldpc_dec_op(ops_deq, tp->op_params->vector_mask, nrLDPC_slot_decoding_parameters);
    AssertFatal(ret == 0, "LDPC offload decoder failed!");
    tp->iter_count = 0;
    /* get the max of iter_count for all dequeued ops */
    j = 0;
    for (h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
      for (i = 0; i < nrLDPC_slot_decoding_parameters->TBs[h].C; ++i) {
        bool *status = &nrLDPC_slot_decoding_parameters->TBs[h].segments[i].decodeSuccess;
        tp->iter_count = RTE_MAX(ops_enq[j]->ldpc_dec.iter_count, tp->iter_count);

        // Check if CRC is available otherwise rely on ops_enq[j]->status to detect decoding success
        if (nrLDPC_slot_decoding_parameters->TBs[h].C > 1) {
          *status = (ops_enq[j]->status == 0);
        } else {
          bool crc_non_24b = check_non24b_crc(&active_dev.info);
          if(crc_non_24b) {
            *status = (ops_enq[j]->status == 0);
          } else {
            uint8_t *decoded_bytes = nrLDPC_slot_decoding_parameters->TBs[h].segments[i].c;
            uint8_t crc_type = crcType(nrLDPC_slot_decoding_parameters->TBs[h].C, nrLDPC_slot_decoding_parameters->TBs[h].A);
            uint32_t len_with_crc = lenWithCrc(nrLDPC_slot_decoding_parameters->TBs[h].C, nrLDPC_slot_decoding_parameters->TBs[h].A);
            int crc_status = check_crc(decoded_bytes, len_with_crc, crc_type);
            *status= (crc_status == 1);
          }
        }
        if (*status) {
          *nrLDPC_slot_decoding_parameters->TBs[h].processedSegments =
              *nrLDPC_slot_decoding_parameters->TBs[h].processedSegments + 1;
        }
        ++j;
      }
    }
  }

  rte_bbdev_dec_op_free_bulk(ops_enq, num_segments);
  rte_free(ops_enq);
  rte_free(ops_deq);
  // Return the worst decoding number of iterations for all segments
  return tp->iter_count;
}

// based on DPDK BBDEV throughput_pmd_lcore_ldpc_enc
static int pmd_lcore_ldpc_enc(void *arg)
{
  struct thread_params *tp = arg;
  nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters = tp->nrLDPC_slot_encoding_parameters;
  uint16_t enq, deq;
  int time_out = 0;
  const uint16_t queue_id = tp->queue_id;
  const uint16_t num_segments = nb_segments_encoding(nrLDPC_slot_encoding_parameters);
  struct rte_bbdev_enc_op **ops_enq;
  struct rte_bbdev_enc_op **ops_deq;
  ops_enq = (struct rte_bbdev_enc_op **)
      rte_calloc("struct rte_bbdev_dec_op **ops_enq", num_segments, sizeof(struct rte_bbdev_enc_op *), RTE_CACHE_LINE_SIZE);
  ops_deq = (struct rte_bbdev_enc_op **)
      rte_calloc("struct rte_bbdev_dec_op **ops_dec", num_segments, sizeof(struct rte_bbdev_enc_op *), RTE_CACHE_LINE_SIZE);
  int ret;
  struct data_buffers *bufs = NULL;
  uint16_t num_to_enq;

  AssertFatal((num_segments < MAX_BURST), "BURST_SIZE should be <= %u", MAX_BURST);

  bufs = &tp->op_params->q_bufs[active_dev.info.socket_id][queue_id];
  ret = rte_bbdev_enc_op_alloc_bulk(tp->op_params->mp_enc, ops_enq, num_segments);
  AssertFatal(ret == 0, "Allocation failed for %d ops", num_segments);
  set_ldpc_enc_op(ops_enq, 0, bufs->inputs, bufs->hard_outputs, nrLDPC_slot_encoding_parameters);

  for (enq = 0, deq = 0; enq < num_segments;) {
    num_to_enq = num_segments;
    if (unlikely(num_segments - enq < num_to_enq))
      num_to_enq = num_segments - enq;
    enq += rte_bbdev_enqueue_ldpc_enc_ops(tp->dev_id, queue_id, &ops_enq[enq], num_to_enq);
    deq += rte_bbdev_dequeue_ldpc_enc_ops(tp->dev_id, queue_id, &ops_deq[deq], enq - deq);
  }
  /* dequeue the remaining */
  while (deq < enq) {
    deq += rte_bbdev_dequeue_ldpc_enc_ops(tp->dev_id, queue_id, &ops_deq[deq], enq - deq);
    time_out++;
    DevAssert(time_out <= TIME_OUT_POLL);
  }
  ret = retrieve_ldpc_enc_op(ops_deq, nrLDPC_slot_encoding_parameters);
  AssertFatal(ret == 0, "Failed to retrieve LDPC encoding op!");

  rte_bbdev_enc_op_free_bulk(ops_enq, num_segments);
  rte_free(ops_enq);
  rte_free(ops_deq);
  return ret;
}

// based on DPDK BBDEV throughput_pmd_lcore_dec
int start_pmd_dec(struct active_device *ad,
                  struct test_op_params *op_params,
                  nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  int ret;
  unsigned int lcore_id, used_cores = 0;
  uint16_t num_lcores;
  /* Set number of lcores */
  num_lcores = (ad->nb_queues < (op_params->num_lcores)) ? ad->nb_queues : op_params->num_lcores;
  /* Allocate memory for thread parameters structure */
  struct thread_params *t_params = rte_zmalloc(NULL, num_lcores * sizeof(struct thread_params), RTE_CACHE_LINE_SIZE);
  AssertFatal(t_params != 0,
              "Failed to alloc %zuB for t_params",
              RTE_ALIGN(sizeof(struct thread_params) * num_lcores, RTE_CACHE_LINE_SIZE));
  // rte_atomic16_set(&op_params->sync, SYNC_WAIT);
  /* Master core is set at first entry */
  t_params[0].dev_id = ad->dev_id;
  t_params[0].lcore_id = rte_lcore_id();
  t_params[0].op_params = op_params;
  t_params[0].queue_id = ad->queue_ids[ad->nb_queues]; // TODO: revise this lookup process
  t_params[0].iter_count = 0;
  t_params[0].nrLDPC_slot_decoding_parameters = nrLDPC_slot_decoding_parameters;
  used_cores++;
  // For now, we never enter here, we don't use the DPDK thread pool
  // RTE_LCORE_FOREACH_WORKER(lcore_id) {
  //   if (used_cores >= num_lcores)
  //     break;
  //   t_params[used_cores].dev_id = ad->dev_id;
  //   t_params[used_cores].lcore_id = lcore_id;
  //   t_params[used_cores].op_params = op_params;
  //   t_params[used_cores].queue_id = ad->queue_ids[used_cores];
  //   t_params[used_cores].iter_count = 0;
  //   t_params[used_cores].nrLDPC_slot_decoding_parameters = nrLDPC_slot_decoding_parameters;
  //   rte_eal_remote_launch(pmd_lcore_ldpc_dec, &t_params[used_cores++], lcore_id);
  // }
  // rte_atomic16_set(&op_params->sync, SYNC_START);
  ret = pmd_lcore_ldpc_dec(&t_params[0]);
  /* Master core is always used */
  // for (used_cores = 1; used_cores < num_lcores; used_cores++)
  //	ret |= rte_eal_wait_lcore(t_params[used_cores].lcore_id);
  rte_free(t_params);
  return ret;
}

// based on DPDK BBDEV throughput_pmd_lcore_enc
int32_t start_pmd_enc(struct active_device *ad,
                      struct test_op_params *op_params,
                      nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  unsigned int lcore_id, used_cores = 0;
  uint16_t num_lcores;
  int ret;
  num_lcores = (ad->nb_queues < (op_params->num_lcores)) ? ad->nb_queues : op_params->num_lcores;
  struct thread_params *t_params = rte_zmalloc(NULL, num_lcores * sizeof(struct thread_params), RTE_CACHE_LINE_SIZE);
  // rte_atomic16_set(&op_params->sync, SYNC_WAIT);
  t_params[0].dev_id = ad->dev_id;
  t_params[0].lcore_id = rte_lcore_id() + 1;
  t_params[0].op_params = op_params;
  t_params[0].queue_id = ad->queue_ids[0]; // TODO: revise this lookup process
  t_params[0].iter_count = 0;
  t_params[0].nrLDPC_slot_encoding_parameters = nrLDPC_slot_encoding_parameters;
  used_cores++;
  // For now, we never enter here, we don't use the DPDK thread pool
  // RTE_LCORE_FOREACH_WORKER(lcore_id) {
  //   if (used_cores >= num_lcores)
  //     break;
  //   t_params[used_cores].dev_id = ad->dev_id;
  //   t_params[used_cores].lcore_id = lcore_id;
  //   t_params[used_cores].op_params = op_params;
  //   t_params[used_cores].queue_id = ad->queue_ids[1];
  //   t_params[used_cores].iter_count = 0;
  //   t_params[used_cores].nrLDPC_slot_encoding_parameters = nrLDPC_slot_encoding_parameters;
  //   rte_eal_remote_launch(pmd_lcore_ldpc_enc, &t_params[used_cores++], lcore_id);
  // }
  // rte_atomic16_set(&op_params->sync, SYNC_START);
  ret = pmd_lcore_ldpc_enc(&t_params[0]);
  rte_free(t_params);
  return ret;
}

// OAI CODE
int32_t nrLDPC_coding_init()
{
  pthread_mutex_init(&encode_mutex, NULL);
  pthread_mutex_init(&decode_mutex, NULL);

  int ret;
  int dev_id = -1;

  char *dpdk_dev = NULL;            // PCI address of the card
  char *dpdk_core_list = NULL;      // cores used by DPDK for the accelerator card
  char *dpdk_file_prefix = NULL;
  char *vfio_vf_token = NULL;       // vfio token for the card
  paramdef_t LoaderParams[] = {
    {"dpdk_dev", NULL, 0, .strptr = &dpdk_dev, .defstrval = NULL, TYPE_STRING, 0, NULL},
    {"dpdk_core_list", NULL, 0, .strptr = &dpdk_core_list, .defstrval = NULL, TYPE_STRING, 0, NULL},
    {"dpdk_file_prefix", NULL, 0, .strptr = &dpdk_file_prefix, .defstrval = "b6", TYPE_STRING, 0, NULL},
    {"vfio_vf_token", NULL, 0, .strptr = &vfio_vf_token, .defstrval = NULL, TYPE_STRING, 0, NULL}
  };
  config_get(config_get_if(), LoaderParams, sizeofArray(LoaderParams), "nrLDPC_coding_aal");
  AssertFatal(dpdk_dev != NULL, "nrLDPC_coding_aal.dpdk_dev was not provided");
  AssertFatal(dpdk_core_list != NULL, "nrLDPC_coding_aal.dpdk_core_list was not provided");

  if(vfio_vf_token != NULL) {
    char *argv[] = {"bbdev", "-l", dpdk_core_list, "-a", dpdk_dev, "--vfio-vf-token", vfio_vf_token, "--"};
    ret = rte_eal_init(sizeofArray(argv), argv);
  } else {
    char *argv[] = {"bbdev", "-l", dpdk_core_list, "-a", dpdk_dev, "--"};
    ret = rte_eal_init(sizeofArray(argv), argv);
  }
  if (ret < 0) {
    printf("EAL initialization failed, probing DPDK device %s\n", dpdk_dev);
    if (rte_dev_probe(dpdk_dev) != 0) {
      LOG_E(PHY, "bbdev %s not found\n", dpdk_dev);
      return (-1);
    }
  }

  // we assume only one device
  uint16_t nb_bbdevs = rte_bbdev_count();
  AssertFatal(nb_bbdevs > 0, "no bbdev devices found");

  // find bbdev that matches the dpdk_dev specified in the config file
  struct rte_bbdev_info info;
  printf("detected %u bbdev devices.\n", nb_bbdevs);
  for (uint16_t device_id = 0; device_id < nb_bbdevs; device_id++) {
    rte_bbdev_info_get(device_id, &info);
    // check if info matches the dpdk dev we looking for
    if (strcmp(info.dev_name, dpdk_dev) == 0) {
      printf("bbdev %s found.\n", info.dev_name);
      dev_id = device_id;
      break;
    }
  }
  AssertFatal(dev_id != -1, "bbdev %s not found.", dpdk_dev);

  // set number of queues based on number of initialized cores (-l option) and driver capabilities
  AssertFatal(add_dev(dev_id) == 0, "failed to setup bbdev");
  AssertFatal(rte_bbdev_stats_reset(dev_id) == 0, "failed to reset stats of bbdev %u", dev_id);
  AssertFatal(rte_bbdev_start(dev_id) == 0, "failed to start bbdev %u", dev_id);

  // the previous calls have populated this global variable (beurk)
  // One more global to remove, not thread safe global op_params
  enc_op_params = rte_zmalloc(NULL, sizeof(struct test_op_params), RTE_CACHE_LINE_SIZE);
  AssertFatal(enc_op_params != NULL,
              "Failed to alloc %zuB for enc_op_params",
              RTE_ALIGN(sizeof(struct test_op_params), RTE_CACHE_LINE_SIZE));
  dec_op_params = rte_zmalloc(NULL, sizeof(struct test_op_params), RTE_CACHE_LINE_SIZE);
  AssertFatal(dec_op_params != NULL,
              "Failed to alloc %zuB for dec_op_params",
              RTE_ALIGN(sizeof(struct test_op_params), RTE_CACHE_LINE_SIZE));

  int socket_id = active_dev.info.socket_id;
  int out_max_sz = 8448; // max code block size (for BG1), 22 * 384
  int in_max_sz = LDPC_MAX_CB_SIZE; // max number of encoded bits (for BG2 and MCS0)
  int num_queues = active_dev.nb_queues;

  int f_ret = create_mempools(&active_dev, socket_id, num_queues, out_max_sz, in_max_sz);
  AssertFatal(f_ret == 0, "Couldn't create mempools");

  f_ret = init_test_op_params(dec_op_params, RTE_BBDEV_OP_LDPC_DEC, active_dev.mp_dec_ops, num_queues, num_queues, 1);
  AssertFatal(f_ret == 0, "Couldn't init test op params");

  f_ret = init_test_op_params(enc_op_params, RTE_BBDEV_OP_LDPC_ENC, active_dev.mp_enc_ops, num_queues, num_queues, 1);
  AssertFatal(f_ret == 0, "Couldn't init test op params");

  // optimization: for encoder output processing using lookup table
  init_bit_unpack_table();

  return 0;
}

int32_t nrLDPC_coding_shutdown()
{
  int dev_id = 0;
  struct rte_bbdev_stats stats;
  free_buffers(&active_dev, enc_op_params);
  free_buffers(&active_dev, dec_op_params);

  rte_free(enc_op_params);
  rte_free(dec_op_params);

  rte_bbdev_stats_get(dev_id, &stats);
  rte_bbdev_stop(dev_id);
  rte_bbdev_close(dev_id);
  memset(&active_dev, 0, sizeof(active_dev));
  return 0;
}

/**
 * @brief Scale LLR values to fit into the fixed-point representation.
 * @note 
 * This functions scales the LLR values to fit into the fixed-point representation.
 * The implementation includes some hacks to improve its robustness at low SNRs.
 * @todo 
 * This LLR scaling method is far from optimal. More testing is needed to understand its performance.
 * Can consider introducing SINR information to improve scaling.
 * A better approach would be to do LLR scaling/quantization right after demodulation, since both the software LDPC decoder also uses an 8-bit input.
 */
void llr_scaling(int16_t *llr, int llr_len, int8_t *llr_scaled, int8_t llr_size, int8_t llr_decimal, int8_t nb_layers, int8_t Qm) {
  const int16_t llr_max = (1 << (llr_size - 1)) - 1;
  const int16_t llr_min = -llr_max;

  // Step 1: Find the max absolute LLR
  int16_t max_abs = 1; // prevent divide-by-zero
  // SCALAR IMPLEMENTATION
  // for (int i = 0; i < llr_len; i++) {
  //     int16_t abs_val = abs(llr[i]);
  //     if (abs_val > max_abs) max_abs = abs_val;
  // }
  // VECTORIZED IMPLEMENTATION
  simde__m128i max_vec = simde_mm_set1_epi16(1);
  for (int i = 0; i < llr_len; i += 8) {
      simde__m128i llr_vec = simde_mm_loadu_si128((simde__m128i*)&llr[i]);
      simde__m128i abs_vec = simde_mm_abs_epi16(llr_vec);
      max_vec = simde_mm_max_epi16(max_vec, abs_vec);
  }
  // reduce max_vec to single max_abs
  int16_t temp[8];
  simde_mm_storeu_si128((simde__m128i*)temp, max_vec);
  for (int i = 0; i < 8; i++) {
      if (temp[i] > max_abs) max_abs = temp[i];
  }

  // Step 2: Compute dynamic scale factor
  float fixed_point_range = (float)llr_max / (1 << llr_decimal);
  float scale = fixed_point_range / (float)max_abs;
  if(max_abs < fixed_point_range) {
    scale = 1.0f;
  }

  // Step 3: Scale and saturate
  // SCALAR IMPLEMENTATION
  // for (int i = 0; i < llr_len; i++) {
  //     float scaled = (float)llr[i] * scale; // map into fixed-point domain
  //     if(Qm == 2) { /* this is a hack! */
  //       scaled = (int8_t)roundf(scaled) >> llr_decimal;
  //     } else {
  //       scaled = (int8_t)roundf(scaled * (1 << llr_decimal));
  //     }
  //     // Clamp to [-128, 127]
  //     if (scaled > llr_max) llr_scaled[i] = llr_max;
  //     else if (scaled < llr_min) llr_scaled[i] = llr_min;
  // }
  // VECTORIZED IMPLEMENTATION
  simde__m128 scale_vec = simde_mm_set1_ps(scale);
  simde__m128i llr_max_vec = simde_mm_set1_epi16(llr_max);
  simde__m128i llr_min_vec = simde_mm_set1_epi16(llr_min);
  simde__m128i decimal_shift = simde_mm_set1_epi16(1 << llr_decimal);
  for (int i = 0; i < llr_len; i += 8) {
    // load LLR values
    simde__m128i llr_vec = simde_mm_loadu_si128((simde__m128i*)&llr[i]);

    // convert to float for scaling
    simde__m128i llr_lo = simde_mm_cvtepi16_epi32(llr_vec);
    simde__m128i llr_hi = simde_mm_cvtepi16_epi32(simde_mm_srli_si128(llr_vec, 8));
    simde__m128 float_lo = simde_mm_cvtepi32_ps(llr_lo);
    simde__m128 float_hi = simde_mm_cvtepi32_ps(llr_hi);

    // scale
    float_lo = simde_mm_mul_ps(float_lo, scale_vec);
    float_hi = simde_mm_mul_ps(float_hi, scale_vec);

    // convert back to int16 with saturation
    llr_lo = simde_mm_cvtps_epi32(float_lo);
    llr_hi = simde_mm_cvtps_epi32(float_hi);
    simde__m128i scaled_vec = simde_mm_packs_epi32(llr_lo, llr_hi);
    
    if (Qm == 2) {  /* this is a hack! */
      scaled_vec = simde_mm_srai_epi16(scaled_vec, 1);
    } else {
      scaled_vec = simde_mm_mullo_epi16(scaled_vec, decimal_shift);
    }

    // clamp to [llr_min, llr_max]
    scaled_vec = simde_mm_min_epi16(scaled_vec, llr_max_vec);
    scaled_vec = simde_mm_max_epi16(scaled_vec, llr_min_vec);

    // Pack to int8 and store
    simde__m128i result = simde_mm_packs_epi16(scaled_vec, simde_mm_setzero_si128());
    simde_mm_storeu_si128((simde__m128i*)&llr_scaled[i], result);
  }
}

/**
 * @brief nrLDPC_coding_decoder
 * @param nrLDPC_slot_decoding_parameters
 * @return
 *   -1 on error
 *    0 on success
 * @note
 *   In OAI, nr_ulsch_decoding.c assumes:
 *   - Input: LLR soft bits after demodulation/unscrambling.
 *   - Output: Individual CBs without CRC24B.
 *   The LDPC decoder must perform: (1) rate dematching, (2) LDPC decoding, and (3) CRC checking/dropping for CBs.
 *   nr_ulsch_decoding.c performs CB concatenation and CRC checking for the entire TB later.
 * @todo
 *   In the future, we should check the underlying LDPC AAL HW capabilities in nr_ulsch_decoding.c first.
 *   This will require nrLDPC_coding_interface.h to be updated, as we will need to keep track of the HW capabilities globally after nrLPDC_coding_init.
 *   Example of other operations that can be offloaded/enabled in the decoding process include:
 *   - CB segmentation (i.e., bypassnr_segmentation()) if the HW supports TB mode.
 *   - CRC24A/CRC16 checking for the entire TB (RTE_BBDEV_LDPC_CRC_TYPE_24A_CHECK/RTE_BBDEV_LDPC_CRC_TYPE_16_CHECK)
 *   - Input LLR compression (RTE_BBDEV_LDPC_LLR_COMPRESSION)
 *   Other features like HARQ combining is not supported for now.
 */
int32_t nrLDPC_coding_decoder(nrLDPC_slot_decoding_parameters_t *nrLDPC_slot_decoding_parameters)
{
  pthread_mutex_lock(&decode_mutex);

  int ret;
  int socket_id = active_dev.info.socket_id;

  uint16_t num_blocks = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
    num_blocks += nrLDPC_slot_decoding_parameters->TBs[h].C;
  }

  /* It is not unlikely that l_ol becomes big enough to overflow the stack
   * If you observe this behavior then move it to the heap
   * Then you would better do a persistent allocation to limit the overhead
   */
  int8_t l_ol[num_blocks * LDPC_MAX_CB_SIZE] __attribute__((aligned(16)));

  // fill_queue_buffers -> init_op_data_objs
  struct rte_mempool *in_mp = active_dev.mp_dec_inputs;
  struct rte_mempool *hard_out_mp = active_dev.mp_dec_hard_outputs;
  struct rte_mempool *soft_out_mp = active_dev.mp_dec_soft_outputs;
  struct rte_mempool *harq_in_mp = active_dev.mp_dec_harq_inputs;
  struct rte_mempool *harq_out_mp = active_dev.mp_dec_harq_outputs;
  struct rte_mempool *mbuf_pools[DATA_NUM_TYPES] = {in_mp, soft_out_mp, hard_out_mp, harq_in_mp, harq_out_mp};

  uint16_t queue_id = active_dev.queue_ids[active_dev.nb_queues]; // TODO: to revise this lookup process

  struct rte_bbdev_op_data **queue_ops[DATA_NUM_TYPES] = {&dec_op_params->q_bufs[socket_id][queue_id].inputs,
                                                          &dec_op_params->q_bufs[socket_id][queue_id].soft_outputs,
                                                          &dec_op_params->q_bufs[socket_id][queue_id].hard_outputs,
                                                          &dec_op_params->q_bufs[socket_id][queue_id].harq_inputs,
                                                          &dec_op_params->q_bufs[socket_id][queue_id].harq_outputs};
  int8_t llr_size = (active_dev.info.drv.capabilities)[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.llr_size;
  int8_t llr_decimal = (active_dev.info.drv.capabilities)[RTE_BBDEV_OP_LDPC_DEC].cap.ldpc_dec.llr_decimals;
  int offset = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_decoding_parameters->nb_TBs; ++h) {
    for (int r = 0; r < nrLDPC_slot_decoding_parameters->TBs[h].C; r++) {
      llr_scaling(nrLDPC_slot_decoding_parameters->TBs[h].segments[r].llr,
                  nrLDPC_slot_decoding_parameters->TBs[h].segments[r].E,
                  &l_ol[offset],
                  llr_size,
                  llr_decimal,
                  nrLDPC_slot_decoding_parameters->TBs[h].nb_layers,
                  nrLDPC_slot_decoding_parameters->TBs[h].Qm);
      offset += LDPC_MAX_CB_SIZE;
    }
  }

  // NOTE: OAI does not support HARQ combining for now; thus we don't use the HARQ buffers
  // for (enum op_data_type type = DATA_INPUT; type < DATA_NUM_TYPES; type++) {
  for (enum op_data_type type = DATA_INPUT; type < 3; type += 2) {
    ret = allocate_buffers_on_socket(queue_ops[type], num_blocks * sizeof(struct rte_bbdev_op_data), socket_id);
    AssertFatal(ret == 0, "Couldn't allocate memory for rte_bbdev_op_data structs");
    ret = init_op_data_objs_dec(*queue_ops[type],
                                l_ol,
                                nrLDPC_slot_decoding_parameters,
                                mbuf_pools[type],
                                type,
                                active_dev.info.drv.min_alignment);
    AssertFatal(ret == 0, "Couldn't init rte_bbdev_op_data structs");
  }

  ret = start_pmd_dec(&active_dev, dec_op_params, nrLDPC_slot_decoding_parameters);
  if (ret < 0) {
    LOG_E(PHY, "Couldn't start pmd dec\n");
  }

  pthread_mutex_unlock(&decode_mutex);

  return 0;
}

/**
 * @brief nrLDPC_coding_encoder
 * @param nrLDPC_slot_encoding_parameters
 * @return
 *   -1 on error
 *    0 on success
 * @note
 *   In OAI, nr_dlsch_coding.c assumes:
 *   - Input: TB with ALL the CRC bits -- CRC24A/CRC16 for the TB, and CRC24B for the CBs.
 *   - Output: Individual encoded CBs (TODO: To update after !3168 is finalized, and the new output will be concatenated CBs).
 *   The LDPC encoder must perform: (1) LDPC encoding, (2) rate matching and interleaving.
 * @todo:
 *  In the future, we should check the underlying LDPC AAL HW capabilities in nr_dlsch_coding.c first.
 *  This will require nrLDPC_coding_interface.h to be updated, as we will need to keep track of the HW capabilities globally after nrLPDC_coding_init.
 *  Example of other operations that can be offloaded in the encoding process include:
 *  - CB segmentation (i.e., bypass nr_segmentation) if the HW supports TB mode.
 *  - CRC24A/CRC16 computation for each TB (RTE_BBDEV_LDPC_CRC_24A_ATTACH/RTE_BBDEV_LDPC_CRC_16_ATTACH).
 *  - CRC24B computation for each CBs (RTE_BBDEV_LDPC_CRC_24B_ATTACH).
 */
int32_t nrLDPC_coding_encoder(nrLDPC_slot_encoding_parameters_t *nrLDPC_slot_encoding_parameters)
{
  pthread_mutex_lock(&encode_mutex);

  int ret;
  int socket_id = active_dev.info.socket_id;

  struct rte_mempool *in_mp = active_dev.mp_enc_inputs;
  struct rte_mempool *hard_out_mp = active_dev.mp_enc_hard_outputs;
  struct rte_mempool *mbuf_pools[2] = {in_mp, hard_out_mp};

  // TODO: to revise this lookup process
  uint16_t queue_id = active_dev.queue_ids[0];

  struct rte_bbdev_op_data **queue_ops[2] = {&enc_op_params->q_bufs[socket_id][queue_id].inputs,
                                             &enc_op_params->q_bufs[socket_id][queue_id].hard_outputs};

  uint16_t num_blocks = 0;
  for (uint16_t h = 0; h < nrLDPC_slot_encoding_parameters->nb_TBs; ++h) {
    num_blocks += nrLDPC_slot_encoding_parameters->TBs[h].C;
  }

  for (uint16_t type = 0; type < 2; type++) {
    ret = allocate_buffers_on_socket(queue_ops[type], num_blocks * sizeof(struct rte_bbdev_op_data), socket_id);
    AssertFatal(ret == 0, "Couldn't allocate memory for rte_bbdev_op_data structs");
    ret = init_op_data_objs_enc(*queue_ops[type],
                                nrLDPC_slot_encoding_parameters,
                                mbuf_pools[type],
                                type,
                                active_dev.info.drv.min_alignment);
    AssertFatal(ret == 0, "Couldn't init rte_bbdev_op_data structs");
  }
  ret = start_pmd_enc(&active_dev, enc_op_params, nrLDPC_slot_encoding_parameters);
  pthread_mutex_unlock(&encode_mutex);
  return ret;
}
