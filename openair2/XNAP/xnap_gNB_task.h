
#include <stdio.h>
#include <stdint.h>

/** @defgroup _x2ap_impl_ X2AP Layer Reference Implementation
 * @ingroup _ref_implementation_
 * @{
 */

#ifndef XNAP_H_
#define XNAP_H_

#define XNAP_SCTP_PPID   (61)    ///< XNAP SCTP Payload Protocol Identifier (PPID)
#include "xnap_gNB_defs.h"

int xnap_gNB_init_sctp (xnap_gNB_instance_t *instance_p,
                        net_ip_address_t    *local_ip_addr,
                        uint32_t gnb_port_for_XNC);

void *xnap_task(void *arg);

int is_xnap_enabled(void);
void xnap_trigger(void);

#endif /* XNAP_H_ */

/**
 * @}
 */
