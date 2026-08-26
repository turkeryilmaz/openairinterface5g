/*
 * SPDX-License-Identifier: LicenseRef-CSSL-1.0
 */

#include "nr_sdap.h"
#include "assertions.h"
#include "utils.h"
#include <errno.h>
#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "nr_sdap_entity.h"
#include "common/utils/LOG/log.h"
#include "rlc.h"
#include "tuntap_if.h"
#include "system.h"

bool sdap_data_req(protocol_ctxt_t *ctxt_p,
                   const ue_id_t ue_id,
                   const srb_flag_t srb_flag,
                   const mui_t mui,
                   const confirm_t confirm,
                   const sdu_size_t sdu_buffer_size,
                   unsigned char *const sdu_buffer,
                   const pdcp_transmission_mode_t pt_mode,
                   const uint32_t *sourceL2Id,
                   const uint32_t *destinationL2Id,
                   const uint8_t qfi,
                   const bool rqi,
                   const int pdusession_id) {
  nr_sdap_entity_t *sdap_entity;
  sdap_entity = nr_sdap_get_entity(ue_id, pdusession_id);

  if(sdap_entity == NULL) {
    LOG_E(SDAP, "%s:%d:%s: Entity not found with ue: 0x%"PRIx64" and pdusession id: %d\n", __FILE__, __LINE__, __FUNCTION__, ue_id, pdusession_id);
    return 0;
  }

  return sdap_entity->tx_entity(sdap_entity,
                                ctxt_p,
                                srb_flag,
                                mui,
                                confirm,
                                sdu_buffer_size,
                                sdu_buffer,
                                pt_mode,
                                sourceL2Id,
                                destinationL2Id,
                                qfi,
                                rqi);
}

void sdap_data_ind(int drb_id, int is_gnb, int pdusession_id, ue_id_t ue_id, char *buf, int size)
{
  nr_sdap_entity_t *sdap_entity;
  sdap_entity = nr_sdap_get_entity(ue_id, pdusession_id);

  if (sdap_entity == NULL) {
    LOG_E(SDAP, "Entity not found for ue rnti/ue_id: %lx and pdusession id: %d\n", ue_id, pdusession_id);
    return;
  }

  sdap_entity->rx_entity(sdap_entity, drb_id, is_gnb, pdusession_id, ue_id, buf, size);
}

static void *sdap_tun_read_thread(void *arg)
{
  nr_sdap_entity_t *entity = arg;
  DevAssert(entity != NULL);
  DevAssert(entity->tun.sock >= 0);

  char rx_buf[NL_MAX_PAYLOAD];
  tuntap_reblock(entity->tun.sock);

  while (1) {
    int len = read(entity->tun.sock, rx_buf, NL_MAX_PAYLOAD);
    if (len == -1) {
      if (errno == EINTR)
        continue; // interrupted system call

      if (errno == EBADF || errno == EINVAL) {
        LOG_I(SDAP,
              "Socket closed, exiting TUN read thread for UE %ld, PDU session %d\n",
              entity->tun.ue_id,
              entity->tun.pdusession_id);
        break;
      }

      LOG_E(SDAP, "read() failed: errno %d (%s)\n", errno, strerror(errno));
      break;
    }

    if (len == 0) {
      LOG_W(SDAP, "TUN socket returned EOF - exiting thread\n");
      break;
    }

    LOG_D(SDAP, "read data of size %d\n", len);

    protocol_ctxt_t ctxt = {.enb_flag = entity->tun.is_gnb, .rntiMaybeUEid = entity->tun.ue_id};

    bool dc = entity->tun.is_gnb ? false : SDAP_HDR_UL_DATA_PDU;

    entity->tx_entity(entity,
                      &ctxt,
                      SRB_FLAG_NO,
                      RLC_MUI_UNDEFINED,
                      RLC_SDU_CONFIRM_NO,
                      len,
                      (unsigned char *)rx_buf,
                      PDCP_TRANSMISSION_MODE_DATA,
                      NULL,
                      NULL,
                      entity->qfi,
                      dc);
  }

  return NULL;
}

/** @brief Bind NAS TUN sock and QFI on an SDAP entity for connected UP
 * Borrow the fd/QFI without taking ownership (NAS owns the TUN, UE entity must not close it)
 * @param[in] entity SDAP entity for this PDU session
 * @param[in] sock TUN fd to borrow
 * @param[in] qfi QFI for the UL SDU */
void nr_sdap_tun_bind(nr_sdap_entity_t *entity, int sock, int qfi)
{
  DevAssert(entity != NULL);
  if (sock < 0)
    return;
  DevAssert(qfi >= 0 && qfi < SDAP_MAX_QFI);
  entity->tun.sock = sock;
  entity->qfi = qfi;
}

/** @brief Start the connected TUN UL reader on an SDAP entity
 * @param[in] entity SDAP entity for this PDU session
 * @param[in,out] thread Reader pthread handle to create into
 * @param[in] name Thread name for threadCreate */
void nr_sdap_tun_start_reader(nr_sdap_entity_t *entity, pthread_t *thread, char *name)
{
  DevAssert(entity);
  DevAssert(entity->tun.sock >= 0);
  DevAssert(thread);
  DevAssert(*thread == 0);
  DevAssert(name);

  threadCreate(thread, sdap_tun_read_thread, entity, name, -1, OAI_PRIORITY_RT_LOW);
  LOG_I(SDAP, "UE %ld PDU session %d: started TUN reader '%s'\n", entity->tun.ue_id, entity->tun.pdusession_id, name);
}

/** @brief Stop a TUN reader thread and clear the handle
 * @note Does not close the TUN socket */
void nr_sdap_tun_stop_reader(pthread_t *thread)
{
  if (thread == NULL || *thread == 0)
    return; // nothing to do
  /* ESRCH: thread already exited (e.g. read returned after close on some OS) */
  int cancel_ret = pthread_cancel(*thread);
  AssertFatal(cancel_ret == 0 || cancel_ret == ESRCH, "pthread_cancel() failed: %d (%s)\n", cancel_ret, strerror(cancel_ret));
  int ret = pthread_join(*thread, NULL);
  AssertFatal(ret == 0, "pthread_join() failed: %d (%s)\n", ret, strerror(ret));
  *thread = 0;
  LOG_I(SDAP, "TUN reader stopped\n");
}

/** @brief Fill ifname[IFNAMSIZ] for the gNB default PDU-session TUN */
void nr_sdap_generate_gnb_tun_ifname(char *ifname, ue_id_t ue_id)
{
  DevAssert(ifname);
  const char *ifprefix = get_softmodem_params()->nsa ? "oaitun_gnb" : "oaitun_enb";
  tun_generate_ifname(ifname, ifprefix, ue_id - 1);
}

void start_sdap_tun_gnb_first_ue_default_pdu_session(ue_id_t ue_id, int pdu_session_id)
{
  nr_sdap_entity_t *entity = nr_sdap_get_entity(ue_id, pdu_session_id);
  DevAssert(entity != NULL);
  DevAssert(entity->tun.is_gnb);

  char ifname[IFNAMSIZ];
  nr_sdap_generate_gnb_tun_ifname(ifname, ue_id);
  entity->tun.sock = tuntap_alloc(IFF_TUN, ifname);
  tun_config(ifname, "10.0.1.1", NULL);
  nr_sdap_tun_start_reader(entity, &entity->pdusession_thread, "gnb_tun_read_thread");
}
