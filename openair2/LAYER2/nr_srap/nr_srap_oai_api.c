/*
Author: Ejaz Ahmed
Email ID: ejaz.ahmed@applied.co
*/

#define _GNU_SOURCE

#include "nr_srap_oai_api.h"
#include "openair2/LAYER2/nr_pdcp/nr_pdcp_oai_api.h"
#include <executables/nr-uesoftmodem.h>
#include "nr_srap_header.h"
#include "nr_srap_manager.h"

extern nr_srap_manager_t *nr_srap_manager;

static srap_data_req_queue tx_srap_uu_q;
static srap_data_req_queue tx_srap_pc5_q;
static srap_data_ind_queue rx_srap_uu_q;
static srap_data_ind_queue rx_srap_pc5_q;

static void init_nr_srap_data_ind_queue(bool gNB_flag);

static void *tx_srap_rlc_pc5_data_req_thread(void *_)
{
  int i;

  LOG_D(NR_SRAP, "tx_srap_rlc_pc5_data_req_thread created on core %d\n", sched_getcpu());
  pthread_setname_np(pthread_self(), "Tx SRAP RLC PC5 queue");
  while (1) {
    if (pthread_mutex_lock(&tx_srap_pc5_q.m) != 0) abort();
    while (tx_srap_pc5_q.length == 0)
      if (pthread_cond_wait(&tx_srap_pc5_q.c, &tx_srap_pc5_q.m) != 0) abort();
    i = tx_srap_pc5_q.start;
    if (pthread_mutex_unlock(&tx_srap_pc5_q.m) != 0) abort();
    LOG_D(NR_SRAP, "Pass data from SRAP tx_srap_pc5_q to RLC%s %d\n\n", __FUNCTION__, __LINE__);
    rlc_data_req(&tx_srap_pc5_q.q[i].ctxt_pP,
                 tx_srap_pc5_q.q[i].srb_flagP,
                 tx_srap_pc5_q.q[i].MBMS_flagP,
                 tx_srap_pc5_q.q[i].rb_idP,
                 tx_srap_pc5_q.q[i].muiP,
                 tx_srap_pc5_q.q[i].confirmP,
                 tx_srap_pc5_q.q[i].sdu_sizeP,
                 tx_srap_pc5_q.q[i].sdu_pP,
                 NULL,
                 NULL);

    if (pthread_mutex_lock(&tx_srap_pc5_q.m) != 0) abort();

    tx_srap_pc5_q.length--;
    tx_srap_pc5_q.start = (tx_srap_pc5_q.start + 1) % SRAP_DATA_REQ_QUEUE_SIZE;

    if (pthread_cond_signal(&tx_srap_pc5_q.c) != 0) abort();
    if (pthread_mutex_unlock(&tx_srap_pc5_q.m) != 0) abort();
  }
}

static void *tx_srap_rlc_uu_data_req_thread(void *_)
{
  int i;

  LOG_D(NR_SRAP,"tx_srap_rlc_uu_data_req_thread created on core %d\n", sched_getcpu());
  pthread_setname_np(pthread_self(), "Tx SRAP RLC Uu queue");
  while (1) {
    if (pthread_mutex_lock(&tx_srap_uu_q.m) != 0) abort();
    while (tx_srap_uu_q.length == 0)
      if (pthread_cond_wait(&tx_srap_uu_q.c, &tx_srap_uu_q.m) != 0) abort();
    i = tx_srap_uu_q.start;
    if (pthread_mutex_unlock(&tx_srap_uu_q.m) != 0) abort();

    rlc_data_req(&tx_srap_uu_q.q[i].ctxt_pP,
                 tx_srap_uu_q.q[i].srb_flagP,
                 tx_srap_uu_q.q[i].MBMS_flagP,
                 tx_srap_uu_q.q[i].rb_idP,
                 tx_srap_uu_q.q[i].muiP,
                 tx_srap_uu_q.q[i].confirmP,
                 tx_srap_uu_q.q[i].sdu_sizeP,
                 tx_srap_uu_q.q[i].sdu_pP,
                 NULL,
                 NULL);

    if (pthread_mutex_lock(&tx_srap_uu_q.m) != 0) abort();

    tx_srap_uu_q.length--;
    tx_srap_uu_q.start = (tx_srap_uu_q.start + 1) % SRAP_DATA_REQ_QUEUE_SIZE;

    if (pthread_cond_signal(&tx_srap_uu_q.c) != 0) abort();
    if (pthread_mutex_unlock(&tx_srap_uu_q.m) != 0) abort();
  }
}

static void init_nr_srap_rlc_data_req_queue(bool gNB_flag)
{
  pthread_t t1, t2;

  if (!gNB_flag) {
    pthread_mutex_init(&tx_srap_pc5_q.m, NULL);
    pthread_cond_init(&tx_srap_pc5_q.c, NULL);
    threadCreate(&t1, tx_srap_rlc_pc5_data_req_thread, NULL, "tx_srap_rlc_pc5_data_req_thread", -1, OAI_PRIORITY_RT_MAX-1);
  }

  if (gNB_flag || (!gNB_flag && get_softmodem_params()->is_relay_ue && (get_softmodem_params()->relay_type == U2N))) {
    pthread_mutex_init(&tx_srap_uu_q.m, NULL);
    pthread_cond_init(&tx_srap_uu_q.c, NULL);
    threadCreate(&t2, tx_srap_rlc_uu_data_req_thread, NULL, "tx_srap_rlc_uu_data_req_thread", -1, OAI_PRIORITY_RT_MAX-1);
  }
}

void enqueue_srap_pc5_data_req(const protocol_ctxt_t *const ctxt_pP,
                               const srb_flag_t   srb_flagP,
                               const MBMS_flag_t  MBMS_flagP,
                               const rb_id_t      rb_idP,
                               const mui_t        muiP,
                               confirm_t    confirmP,
                               sdu_size_t   sdu_sizeP,
                               mem_block_t *sdu_pP)
{
  int i;
  int logged = 0;
  if (pthread_mutex_lock(&tx_srap_pc5_q.m) != 0) abort();
  while (tx_srap_pc5_q.length == SRAP_DATA_REQ_QUEUE_SIZE) {
    if (!logged) {
      logged = 1;
      LOG_W(NR_SRAP, "%s: tx_srap_pc5_q data queue is full\n", __FUNCTION__);
    }
    if (pthread_cond_wait(&tx_srap_pc5_q.c, &tx_srap_pc5_q.m) != 0) abort();
  }

  i = (tx_srap_pc5_q.start + tx_srap_pc5_q.length) % SRAP_DATA_REQ_QUEUE_SIZE;
  tx_srap_pc5_q.length++;

  tx_srap_pc5_q.q[i].ctxt_pP    = *ctxt_pP;
  tx_srap_pc5_q.q[i].srb_flagP  = srb_flagP;
  tx_srap_pc5_q.q[i].MBMS_flagP = MBMS_flagP;
  tx_srap_pc5_q.q[i].rb_idP     = rb_idP;
  tx_srap_pc5_q.q[i].muiP       = muiP;
  tx_srap_pc5_q.q[i].confirmP   = confirmP;
  tx_srap_pc5_q.q[i].sdu_sizeP  = sdu_sizeP;
  tx_srap_pc5_q.q[i].sdu_pP     = sdu_pP;

  if (pthread_cond_signal(&tx_srap_pc5_q.c) != 0) abort();
  if (pthread_mutex_unlock(&tx_srap_pc5_q.m) != 0) abort();
}

void srap_deliver_sdu_srb(void *_ue, nr_srap_entity_t *entity,
                          char *buf, int size)
{
  // TODO:
}

void srap_forward_sdu_drb(void *_ue, nr_srap_entity_t *entity,
                          char *buf, int size)
{
  // TODO:
}

void srap_deliver_pdu_drb(protocol_ctxt_t *ctxt, int rb_id,
                          char *buf, int size, int sdu_id) {
  mem_block_t *memblock = get_free_mem_block(size, __FUNCTION__);
  memcpy(memblock->data, buf, size);
  enqueue_srap_pc5_data_req(ctxt, 0, MBMS_FLAG_NO, rb_id, sdu_id, 0, size, memblock);
}

void srap_deliver_pdu_srb(void *_ue, nr_srap_entity_t *entity,
                          char *buf, int size)
{
  //TODO:
}

int srap_module_init(bool gNB_flag)
{
  static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
  static int inited = 0;
  static int inited_ue = 0;

  if (pthread_mutex_lock(&lock)) abort();

  if (gNB_flag == true && inited) {
    LOG_E(NR_SRAP, "%s:%d:%s: fatal, inited already 1\n", __FILE__, __LINE__, __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if (gNB_flag == false && inited_ue) {
    LOG_E(NR_SRAP, "%s:%d:%s: fatal, inited_ue already 1\n", __FILE__, __LINE__, __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  if (gNB_flag == true) inited = 1;
  if (gNB_flag == false) inited_ue = 1;

  LOG_D(NR_SRAP, "Created NR SRAP UE Mananger!!!");

  if (pthread_mutex_unlock(&lock)) abort();

  return 0;
}

void nr_srap_layer_init(bool gNB_flag)
{
  /* hack: be sure to initialize only once */
  static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
  static int initialized = 0;
  if (pthread_mutex_lock(&m) != 0) abort();
  if (initialized) {
    if (pthread_mutex_unlock(&m) != 0) abort();
    return;
  }
  initialized = 1;
  if (pthread_mutex_unlock(&m) != 0) abort();

  nr_srap_manager = new_nr_srap_manager(gNB_flag);

  init_nr_srap_data_ind_queue(gNB_flag);
  init_nr_srap_rlc_data_req_queue(gNB_flag);

}

static void do_srap_data_ind(const protocol_ctxt_t *const  ctxt_pP,
                             const srb_flag_t srb_flagP,
                             const MBMS_flag_t MBMS_flagP,
                             const rb_id_t rb_id,
                             const sdu_size_t sdu_buffer_size,
                             mem_block_t *sdu_buffer)
{
  if (ctxt_pP->module_id != 0 ||
      ctxt_pP->instance != 0  ||
      ctxt_pP->eNB_index != 0 ||
      ctxt_pP->brOption != 0) {
    LOG_E(NR_SRAP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
    exit(EXIT_FAILURE);
  }
  ngran_node_t node_type = get_node_type();
  if (node_type == 0) {
    LOG_W(NR_SRAP, "Currently, not supporting gNB!!!\n");
    return;
  }

  nr_srap_manager_lock(nr_srap_manager);
  nr_srap_manager_t  *m = get_nr_srap_manager();

  nr_srap_entity_t* srap_entity = nr_srap_get_entity(m, NR_SRAP_PC5);
  if ((srap_entity != NULL)) {
    uint8_t relay_type = get_softmodem_params()->relay_type;
    uint8_t header_size;
    if (relay_type == U2N) {
      U2NHeader_t u2n_header;
      header_size = sizeof(u2n_header);
      decode_srap_header(&u2n_header, sdu_buffer->data);
      LOG_D(NR_SRAP, "Rx - bearer id: %x, ue id: %x\n", u2n_header.octet1 & 0x1F, u2n_header.octet2);
    } else if (relay_type == U2U) {
      U2UHeader_t u2u_header;
      header_size = sizeof(u2u_header);
      decode_srap_header(&u2u_header, sdu_buffer->data);
      LOG_D(NR_SRAP, "Rx - bearer id: %x, source ue id: %x, destination ue id: %x\n",
            u2u_header.octet1 & 0x1F, u2u_header.octet2, u2u_header.octet3);
    } else
      return;
    srap_entity->recv_pdu(ctxt_pP, srap_entity, (char *)(sdu_buffer->data + header_size), sdu_buffer_size - header_size, srb_flagP, MBMS_flagP, rb_id);
  } else {
    LOG_E(NR_SRAP, "%s:%d:%s: no SRAP entity found (rb_id %ld, srb_flag %d)\n", __FILE__, __LINE__, __FUNCTION__, rb_id, srb_flagP);
  }
  nr_srap_manager_unlock(nr_srap_manager);

  free_mem_block(sdu_buffer, __FUNCTION__);
}

static void *srap_pc5_data_ind_thread(void *_)
{
  int i;

  pthread_setname_np(pthread_self(), "SRAP PC5 data ind");
  while (1) {
    if (pthread_mutex_lock(&rx_srap_pc5_q.m) != 0) {
      abort();
    }
    while (rx_srap_pc5_q.length == 0)
      if (pthread_cond_wait(&rx_srap_pc5_q.c, &rx_srap_pc5_q.m) != 0) {
        abort();
      }
    i = rx_srap_pc5_q.start;
    if (pthread_mutex_unlock(&rx_srap_pc5_q.m) != 0) {
      abort();
    }
    LOG_D(NR_SRAP, "PC5 Sending indication to above layer by passing rx_srap_pc5_q in %s rntiMaybeYUEid %ld\n",
          __FUNCTION__,
          rx_srap_pc5_q.q[i].ctxt_pP.rntiMaybeUEid);
    do_srap_data_ind(&rx_srap_pc5_q.q[i].ctxt_pP,
                     rx_srap_pc5_q.q[i].srb_flagP,
                     rx_srap_pc5_q.q[i].MBMS_flagP,
                     rx_srap_pc5_q.q[i].rb_id,
                     rx_srap_pc5_q.q[i].sdu_buffer_size,
                     rx_srap_pc5_q.q[i].sdu_buffer);

    if (pthread_mutex_lock(&rx_srap_pc5_q.m) != 0) {
      abort();
    }

    rx_srap_pc5_q.length--;
    rx_srap_pc5_q.start = (rx_srap_pc5_q.start + 1) % SRAP_DATA_IND_QUEUE_SIZE;

    if (pthread_cond_signal(&rx_srap_pc5_q.c) != 0) {
      abort();
    }
    if (pthread_mutex_unlock(&rx_srap_pc5_q.m) != 0) {
      abort();
    }
  }
}

static void *srap_uu_data_ind_thread(void *_)
{
  int i;

  pthread_setname_np(pthread_self(), "SRAP Uu data ind");
  while (1) {
    if (pthread_mutex_lock(&rx_srap_uu_q.m) != 0) abort();
    while (rx_srap_uu_q.length == 0)
      if (pthread_cond_wait(&rx_srap_uu_q.c, &rx_srap_uu_q.m) != 0) abort();
    i = rx_srap_uu_q.start;
    if (pthread_mutex_unlock(&rx_srap_uu_q.m) != 0) abort();

    do_srap_data_ind(&rx_srap_uu_q.q[i].ctxt_pP,
                     rx_srap_uu_q.q[i].srb_flagP,
                     rx_srap_uu_q.q[i].MBMS_flagP,
                     rx_srap_uu_q.q[i].rb_id,
                     rx_srap_uu_q.q[i].sdu_buffer_size,
                     rx_srap_uu_q.q[i].sdu_buffer);

    if (pthread_mutex_lock(&rx_srap_uu_q.m) != 0) abort();

    rx_srap_uu_q.length--;
    rx_srap_uu_q.start = (rx_srap_uu_q.start + 1) % SRAP_DATA_IND_QUEUE_SIZE;

    if (pthread_cond_signal(&rx_srap_uu_q.c) != 0) abort();
    if (pthread_mutex_unlock(&rx_srap_uu_q.m) != 0) abort();
  }
}

static void init_nr_srap_data_ind_queue(bool gNB_flag)
{
  pthread_t t1, t2;

  if (!gNB_flag) {
    pthread_mutex_init(&rx_srap_pc5_q.m, NULL);
    pthread_cond_init(&rx_srap_pc5_q.c, NULL);
    if (pthread_create(&t1, NULL, srap_pc5_data_ind_thread, NULL) != 0) {
      LOG_E(NR_SRAP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
      exit(EXIT_FAILURE);
    }
  }
  if (gNB_flag || (!gNB_flag && get_softmodem_params()->is_relay_ue && (get_softmodem_params()->relay_type == U2N))) {
    pthread_mutex_init(&rx_srap_uu_q.m, NULL);
    pthread_cond_init(&rx_srap_uu_q.c, NULL);
    if (pthread_create(&t2, NULL, srap_uu_data_ind_thread, NULL) != 0) {
      LOG_E(NR_SRAP, "%s:%d:%s: fatal\n", __FILE__, __LINE__, __FUNCTION__);
      exit(EXIT_FAILURE);
    }
  }
}

static void enqueue_srap_pc5_data_ind(const protocol_ctxt_t *const  ctxt_pP,
                                      const srb_flag_t srb_flagP,
                                      const MBMS_flag_t MBMS_flagP,
                                      const rb_id_t rb_id,
                                      const sdu_size_t sdu_buffer_size,
                                      mem_block_t *const sdu_buffer)
{
  int i;
  int logged = 0;

  if (pthread_mutex_lock(&rx_srap_pc5_q.m) != 0) abort();
  while (rx_srap_pc5_q.length == SRAP_DATA_IND_QUEUE_SIZE) {
    if (!logged) {
      logged = 1;
      LOG_W(NR_SRAP, "%s: pdcp_data_ind queue is full\n", __FUNCTION__);
    }
    if (pthread_cond_wait(&rx_srap_pc5_q.c, &rx_srap_pc5_q.m) != 0) abort();
  }
  LOG_D(NR_SRAP, "Enqueuing data in rx_srap_pc5_q %s %d\n", __FUNCTION__, __LINE__);
  i = (rx_srap_pc5_q.start + rx_srap_pc5_q.length) % SRAP_DATA_IND_QUEUE_SIZE;
  rx_srap_pc5_q.length++;

  rx_srap_pc5_q.q[i].ctxt_pP         = *ctxt_pP;
  rx_srap_pc5_q.q[i].srb_flagP       = srb_flagP;
  rx_srap_pc5_q.q[i].MBMS_flagP      = MBMS_flagP;
  rx_srap_pc5_q.q[i].rb_id           = rb_id;
  rx_srap_pc5_q.q[i].sdu_buffer_size = sdu_buffer_size;
  rx_srap_pc5_q.q[i].sdu_buffer      = sdu_buffer;

  if (pthread_cond_signal(&rx_srap_pc5_q.c) != 0) abort();
  if (pthread_mutex_unlock(&rx_srap_pc5_q.m) != 0) abort();
}

static void enqueue_srap_uu_data_ind(const protocol_ctxt_t *const  ctxt_pP,
                                     const srb_flag_t srb_flagP,
                                     const MBMS_flag_t MBMS_flagP,
                                     const rb_id_t rb_id,
                                     const sdu_size_t sdu_buffer_size,
                                     mem_block_t *const sdu_buffer)
{
  int i;
  int logged = 0;

  if (pthread_mutex_lock(&rx_srap_uu_q.m) != 0) abort();
  while (rx_srap_uu_q.length == SRAP_DATA_IND_QUEUE_SIZE) {
    if (!logged) {
      logged = 1;
      LOG_W(NR_SRAP, "%s: pdcp_data_ind queue is full\n", __FUNCTION__);
    }
    if (pthread_cond_wait(&rx_srap_uu_q.c, &rx_srap_uu_q.m) != 0) abort();
  }
  LOG_D(NR_SRAP, "Calling SRAP layer from RLC in %s\n", __FUNCTION__);
  i = (rx_srap_uu_q.start + rx_srap_uu_q.length) % SRAP_DATA_IND_QUEUE_SIZE;
  rx_srap_uu_q.length++;

  rx_srap_uu_q.q[i].ctxt_pP         = *ctxt_pP;
  rx_srap_uu_q.q[i].srb_flagP       = srb_flagP;
  rx_srap_uu_q.q[i].MBMS_flagP      = MBMS_flagP;
  rx_srap_uu_q.q[i].rb_id           = rb_id;
  rx_srap_uu_q.q[i].sdu_buffer_size = sdu_buffer_size;
  rx_srap_uu_q.q[i].sdu_buffer      = sdu_buffer;

  if (pthread_cond_signal(&rx_srap_uu_q.c) != 0) abort();
  if (pthread_mutex_unlock(&rx_srap_uu_q.m) != 0) abort();
}

bool srap_data_ind(const protocol_ctxt_t *const ctxt_pP,
                   const srb_flag_t srb_flagP,
                   const MBMS_flag_t MBMS_flagP,
                   const rb_id_t rb_id,
                   const sdu_size_t sdu_buffer_size,
                   mem_block_t *const sdu_buffer,
                   const uint32_t *const srcID,
                   const uint32_t *const dstID,
                   nr_intf_type_t intf_type)
{

  if (intf_type == PC5) {
    enqueue_srap_pc5_data_ind(ctxt_pP,
                              srb_flagP,
                              MBMS_flagP,
                              rb_id,
                              sdu_buffer_size,
                              sdu_buffer);
  } else if (intf_type == UU) {
    enqueue_srap_uu_data_ind(ctxt_pP,
                             srb_flagP,
                             MBMS_flagP,
                             rb_id,
                             sdu_buffer_size,
                             sdu_buffer);
  }
  return true;
}

void srap_deliver_sdu_drb(const protocol_ctxt_t *const  ctxt_pP,
                          void *_ue, nr_srap_entity_t *entity,
                          char *buf, int size,
                          const srb_flag_t srb_flagP,
                          const MBMS_flag_t MBMS_flagP,
                          const rb_id_t rb_id) {

  mem_block_t *memblock = get_free_mem_block(size, __func__);

  if (memblock == NULL) {
    LOG_E(NR_SRAP, "%s:%d:%s: ERROR: malloc16 failed\n", __FILE__, __LINE__, __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  memcpy(memblock->data, buf, size);

  // Sending data indication to PDCP at the destination
  if (!pdcp_data_ind(ctxt_pP, srb_flagP, MBMS_flagP, rb_id, size, memblock, NULL, NULL)) {
    LOG_E(NR_SRAP, "%s:%d:%s: ERROR: pdcp_data_ind failed\n", __FILE__, __LINE__, __FUNCTION__);
    /* what to do in case of failure? for the moment: nothing */
  }
}

bool nr_srap_data_req_drb(protocol_ctxt_t *ctxt,
                          const rb_id_t rb_id,
                          const mui_t sdu_id,
                          const sdu_size_t sdu_buffer_size,
                          char *sdu_buffer) {

    uint8_t relay_type = get_softmodem_params()->relay_type;
    nr_srap_manager_internal_t *m = nr_srap_manager;

    if (m == NULL) {
        LOG_E(NR_SRAP, "SRAP manager is not initialized!!!");
        return false;
    }

    if (m && (m->srap_entity[0] == NULL)) {
        LOG_E(NR_SRAP, "SRAP entity is not initialized!!!");
        return false;
    }

    U2NHeader_t u2n_header;
    U2UHeader_t u2u_header;
    int srap_pdu_size = sdu_buffer_size + ((relay_type == U2N) ? sizeof(u2n_header) : sizeof(u2u_header));
    char pdu_buf[srap_pdu_size];
    m->srap_entity[0]->process_sdu(sdu_buffer, sdu_buffer_size, relay_type, rb_id, pdu_buf,
                                  (relay_type == U2N) ? sizeof(u2n_header) : sizeof(u2u_header),
                                  (relay_type == U2N) ? (void*)&u2n_header : (void*)&u2u_header);
    srap_deliver_pdu deliver_pdu_cb = m->srap_entity[0]->deliver_pdu;
    deliver_pdu_cb(ctxt, rb_id, pdu_buf, srap_pdu_size, sdu_id);
  return true;
}
