#ifndef XNAP_COMMON_H_
#define XNAP_COMMON_H_
#include "common/openairinterface5g_limits.h"
#include "oai_asn1.h"
#include "XNAP_ProtocolIE-Field.h"
#include "XNAP_InitiatingMessage.h"
#include "XNAP_SuccessfulOutcome.h"
#include "XNAP_UnsuccessfulOutcome.h"
#include "XNAP_XnAP-PDU.h"
#include "intertask_interface.h"
#include "XNAP_ProtocolIE-FieldPair.h"
#include "XNAP_ProtocolIE-ContainerPair.h"
#include "XNAP_ProtocolExtensionField.h"
#include "XNAP_ProtocolExtensionContainer.h"
#include "XNAP_asn_constant.h"

#ifndef XNAP_PORT
# define XNAP_PORT 38423
#endif

extern int asn1_xer_print;

#if defined(ENB_MODE)
# include "common/utils/LOG/log.h"
# define XNAP_INFO(x, args...) LOG_I(XNAP, x, ##args)
# define XNAP_ERROR(x, args...) LOG_E(XNAP, x, ##args)
# define XNAP_WARN(x, args...)  LOG_W(XNAP, x, ##args)
# define XNAP_DEBUG(x, args...) LOG_D(XNAP, x, ##args)
#else
# define XNAP_INFO(x, args...) do { fprintf(stdout, "[XNAP][I]"x, ##args); } while(0)
# define XNAP_ERROR(x, args...) do { fprintf(stdout, "[XNAP][E]"x, ##args); } while(0)
# define XNAP_WARN(x, args...)  do { fprintf(stdout, "[XNAP][W]"x, ##args); } while(0)
# define XNAP_DEBUG(x, args...) do { fprintf(stdout, "[XNAP][D]"x, ##args); } while(0)
#endif

#define XNAP_FIND_PROTOCOLIE_BY_ID(IE_TYPE, ie, container, IE_ID, mandatory) \
  do {\
    IE_TYPE **ptr; \
    ie = NULL; \
    for (ptr = container->protocolIEs.list.array; \
         ptr < &container->protocolIEs.list.array[container->protocolIEs.list.count]; \
         ptr++) { \
      if((*ptr)->id == IE_ID) { \
        ie = *ptr; \
        break; \
      } \
    } \
    if (mandatory) DevAssert(ie != NULL); \
  } while(0)

typedef int (*xnap_message_decoded_callback)(
  instance_t instance,
  uint32_t assocId,
  uint32_t stream,
  XNAP_XnAP_PDU_t *pdu);
  
/** \brief Encode a successfull outcome message
 \param buffer pointer to buffer in which data will be encoded
 \param length pointer to the length of buffer
 \param procedureCode Procedure code for the message
 \param criticality Criticality of the message
 \param td ASN1C type descriptor of the sptr
 \param sptr Deferenced pointer to the structure to encode
 @returns size in bytes encded on success or 0 on failure
 **/
ssize_t xnap_generate_successfull_outcome(
  uint8_t               **buffer,
  uint32_t               *length,
  XNAP_ProcedureCode_t         procedureCode,
  XNAP_Criticality_t           criticality,
  asn_TYPE_descriptor_t  *td,
  void                   *sptr);

/** \brief Encode an initiating message
 \param buffer pointer to buffer in which data will be encoded
 \param length pointer to the length of buffer
 \param procedureCode Procedure code for the message
 \param criticality Criticality of the message
 \param td ASN1C type descriptor of the sptr
 \param sptr Deferenced pointer to the structure to encode
 @returns size in bytes encded on success or 0 on failure
 **/
ssize_t xnap_generate_initiating_message(
  uint8_t               **buffer,
  uint32_t               *length,
  XNAP_ProcedureCode_t    procedureCode,
  XNAP_Criticality_t      criticality,
  asn_TYPE_descriptor_t  *td,
  void                   *sptr);

/** \brief Encode an unsuccessfull outcome message
 \param buffer pointer to buffer in which data will be encoded
 \param length pointer to the length of buffer
 \param procedureCode Procedure code for the message
 \param criticality Criticality of the message
 \param td ASN1C type descriptor of the sptr
 \param sptr Deferenced pointer to the structure to encode
 @returns size in bytes encded on success or 0 on failure
 **/
ssize_t xnap_generate_unsuccessfull_outcome(
  uint8_t               **buffer,
  uint32_t               *length,
  XNAP_ProcedureCode_t         procedureCode,
  XNAP_Criticality_t           criticality,
  asn_TYPE_descriptor_t  *td,
  void                   *sptr);

/** \brief Handle criticality
 \param criticality Criticality of the IE
 @returns void
 **/
void xnap_handle_criticality(XNAP_Criticality_t criticality);

#endif /* XNAP_COMMON_H_ */

