#include <stdint.h>
#include "openair2/COMMON/platform_types.h"

#ifndef _SRAP_HEADER_H_
#define _SRAP_HEADER_H_

typedef enum {
    U2N = 0,
    U2U = 1
} relay_type_t;

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
void create_header(relay_type_t relay_type, uint8_t bearer_id, uint8_t src_ue_id, uint8_t dest_ue_id, void* header);

#endif /* _SRAP_HEADER_H_ */