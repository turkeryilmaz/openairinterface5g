#include <stdint.h>
#include "openair2/COMMON/platform_types.h"

#ifndef _SRAP_HEADER_H_
#define _SRAP_HEADER_H_

typedef enum {
    NO_RELAY = 0,
    U2N = 1,
    U2U = 2
} relay_type_t;

// 38.351 6.2.2 SRAP data pdu: 2 bytes hdr for U2N, 3 bytes hdr for U2U
// U2N Header (2 octets)
typedef struct U2NHeader {
    uint8_t octet1;
    uint8_t octet2;
} U2NHeader_t;

// U2U Header (3 octets)
typedef struct U2UHeader {
    uint8_t octet1;
    uint8_t octet2;
    uint8_t octet3;
} U2UHeader_t;

// Function to encode SRAP Header
void encode_srap_header(void* header, uint8_t* buffer);

// Function to decode SRAP Header
void decode_srap_header(void* header, uint8_t* buffer);

// Function to create SRAP headers
void create_header(uint8_t dc_bit, relay_type_t relay_type, uint8_t bearer_id, int8_t src_ue_id, int8_t dest_ue_id, void* header);

#endif /* _SRAP_HEADER_H_ */