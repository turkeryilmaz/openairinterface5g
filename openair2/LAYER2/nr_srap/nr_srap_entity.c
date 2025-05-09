/*
Author: Ejaz Ahmed
Email ID: ejaz.ahmed@applied.co
*/

#include "nr_srap_header.h"
#include "nr_srap_entity.h"
#include "nr_srap_oai_api.h"
#include "executables/softmodem-common.h"

// Rx side function to receive SRAP pdu containing SRAP headers and PDCP PDU
void nr_srap_entity_recv_pdu(const protocol_ctxt_t *const  ctxt_pP,
                             nr_srap_entity_t *entity,
                             char *_buffer, int size,
                             const srb_flag_t srb_flagP,
                             const MBMS_flag_t MBMS_flagP,
                             const rb_id_t rb_id) {
  unsigned char    *buffer = (unsigned char *)_buffer;
  nr_srap_sdu_t    *sdu;

  if (size < 1) {
    LOG_E(NR_SRAP, "bad PDU received (size = %d)\n", size);
    return;
  }

  entity->stats.rxpdu_pkts++;
  entity->stats.rxpdu_bytes += size;

  sdu = nr_srap_new_sdu((char *) buffer, size);

  if (get_softmodem_params()->is_relay_ue) {
    if (entity->type == NR_SRAP_PC5) {
      // TODO: Send to NR_SRAP_UU entity
      LOG_E(NR_SRAP, "Doing nothing after receiving SRAP PDU at %s\n", __FUNCTION__);
    } else if (entity->type == NR_SRAP_UU) {
      LOG_E(NR_SRAP, "Doing nothing after receiving SRAP PDU at %s\n", __FUNCTION__);
      // TODO: Send to NR_SRAP_PC5 entity
    }
  } else {
    entity->deliver_sdu(ctxt_pP, entity->deliver_sdu_data, entity, sdu->buffer, sdu->size, srb_flagP, MBMS_flagP, rb_id);
  }
  entity->stats.txsdu_pkts++;
  entity->stats.txsdu_bytes += sdu->size;
  nr_srap_free_sdu(sdu);
}

void nr_srap_entity_delete(nr_srap_entity_t *entity)
{
  nr_srap_sdu_t *cur = entity->rx_list;
  while (cur != NULL) {
    nr_srap_sdu_t *next = cur->next;
    nr_srap_free_sdu(cur);
    cur = next;
  }
  free(entity);
}

static void nr_srap_entity_get_stats(nr_srap_entity_t *entity,
                                     nr_srap_statistics_t *out)
{
  *out = entity->stats;
}

static void nr_srap_entity_process_sdu(char *buffer,
                                       int size,
                                       uint8_t relay_type,
                                       int rb_id,
                                       char *pdu_buffer,
                                       uint8_t header_size,
                                       void *header) {
  char *pdu_buf = pdu_buffer;
  if (relay_type == U2N) {
    uint8_t dc_bit = 1; // control : 0, data : 1
    uint8_t remote_ue_id = get_softmodem_params()->remote_ue_id;
    create_header(dc_bit, relay_type, rb_id, -1, remote_ue_id, header); // In U2N relay case, we do not have a ue_src_id field in the header, the spec. 38351, 6.3.2 defines only U2N remote ue id.
  } else if (relay_type == U2U) {
    // TODO: Call the following functions to enable U2U relay support
    // create_header(relay_type, rb_id, src_ue_id, dest_ue_id, header);
  }
  encode_srap_header(header, (uint8_t*)pdu_buf);
  memcpy(pdu_buf + header_size, buffer, size);
  if (relay_type == U2N) {
    LOG_D(NR_SRAP, "Tx UE %s(): (drb %d) Adding SRAP headers to PDCP sdu: size %d bearer id %x, ue id: %x\n",
          __func__, rb_id, size, (((U2NHeader_t*)header)->octet1 & 0x1F), ((U2NHeader_t*)header)->octet2);
  } else if (relay_type == U2U) {
    LOG_D(NR_SRAP, "Tx UE %s(): (drb %d) Adding SRAP headers to PDCP sdu: size %d bearer id %x, src ue id: %x dest ue id: %x\n",
          __func__, rb_id, size, (((U2UHeader_t*)header)->octet1 & 0x1F), ((U2UHeader_t*)header)->octet2, ((U2UHeader_t*)header)->octet3);
  }
}

nr_srap_entity_t *new_nr_srap_entity(nr_srap_entity_type_t type,
                                     void (*deliver_sdu)(const protocol_ctxt_t *const  ctxt_pP, void *deliver_sdu_data,
                                                         nr_srap_entity_t *entity, char *buf, int size,
                                                         const srb_flag_t srb_flagP, const MBMS_flag_t MBMS_flagP,
                                                         const rb_id_t rb_id),
                                     void *deliver_sdu_data,
                                     void (*deliver_pdu)(protocol_ctxt_t *ctxt, int rb_id,
                                                         char *buf, int size, int sdu_id),
                                     void *deliver_pdu_data)
{
  nr_srap_entity_t *ret;
  ret = calloc(1, sizeof(nr_srap_entity_t));
  if (ret == NULL) {
    LOG_E(NR_SRAP, "%s:%d:%s: out of memory\n", __FILE__, __LINE__, __FUNCTION__);
    exit(EXIT_FAILURE);
  }

  ret->type = type;

  ret->recv_pdu     = nr_srap_entity_recv_pdu;
  ret->process_sdu  = nr_srap_entity_process_sdu;
  ret->delete_entity = nr_srap_entity_delete;

  ret->get_stats = nr_srap_entity_get_stats;
  ret->deliver_sdu = deliver_sdu;
  ret->deliver_sdu_data = deliver_sdu_data;

  ret->deliver_pdu = deliver_pdu;
  ret->deliver_pdu_data = deliver_pdu_data;
  return ret;
}
