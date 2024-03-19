#include "nr_fapi_test.h"

int main(int n, char *v[])
{
  srand(time(NULL));
  logInit();
  set_glog(OAILOG_DISABLE);

  nfapi_nr_param_request_scf_t req;
  memset(&req, 0, sizeof(req));
  req.header.message_id = NFAPI_NR_PHY_MSG_TYPE_PARAM_REQUEST;
  // FAPI doesn't have these 2 following parameters, dont use
  req.header.phy_id = 0;
  req.header.spare = 0;
  uint8_t msg_buf[8192];
  uint16_t msg_len = sizeof(req);

  // first test the packing procedure
  int pack_result = fapi_nr_p5_message_pack(&req, msg_len, msg_buf, sizeof(msg_buf), NULL);
  // PARAM.request message body length is 0
  DevAssert(pack_result == 0 + NFAPI_HEADER_LENGTH);
  for (int i = 0; i < pack_result; i++) {
    printf("0x%02x ", msg_buf[i]);
  }
  printf("\n");
  // update req message_length value with value calculated in message_pack procedure
  req.header.message_length = pack_result - NFAPI_HEADER_LENGTH;
  // test the unpacking of the header
  // copy first NFAPI_HEADER_LENGTH bytes into a new buffer, to simulate SCTP PEEK
  fapi_message_header_t header;
  uint32_t header_buffer_size = NFAPI_HEADER_LENGTH;
  uint8_t header_buffer[header_buffer_size];
  for (int idx = 0; idx < header_buffer_size; idx++) {
    header_buffer[idx] = msg_buf[idx];
  }
  uint8_t *pReadPackedMessage = header_buffer;
  printf("Test the header unpacking and compare with initial message\n");
  int unpack_header_result = fapi_nr_p5_message_header_unpack(&pReadPackedMessage, NFAPI_HEADER_LENGTH, &header, sizeof(header), 0);
  AssertFatal(unpack_header_result >= 0, "nfapi_p5_message_header_unpack failed with return %d\n", unpack_header_result);
  DevAssert(header.message_id == req.header.message_id);
  DevAssert(header.message_length == req.header.message_length);
  // test the unpacking and compare with initial message
  nfapi_nr_param_request_scf_t unpacked_req;
  memset(&unpacked_req, 0, sizeof(unpacked_req));
  int unpack_result =
      fapi_nr_p5_message_unpack(msg_buf, header.message_length + NFAPI_HEADER_LENGTH, &unpacked_req, sizeof(unpacked_req), NULL);
  DevAssert(unpack_result >= 0);
  DevAssert(unpacked_req.header.message_id == req.header.message_id);
  DevAssert(unpacked_req.header.message_length == req.header.message_length);
  // All tests successful!
  return 0;
}
