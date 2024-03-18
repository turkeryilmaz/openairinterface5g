#include "nr_fapi_test.h"

int main(int n, char *v[])
{
  srand(time(NULL));
#ifndef _STANDALONE_TESTING_
  logInit();
  set_glog(OAILOG_DISABLE);
#endif

  fapi_nr_param_request_scf_t req;
  memset(&req, 0, sizeof(req));
  req.header.message_id = NFAPI_NR_PHY_MSG_TYPE_PARAM_REQUEST;
  req.header.num_msg = rand16();
  req.header.opaque_handle = rand16();
  uint8_t msg_buf[8192];
  uint16_t msg_len = sizeof(req);

  // first test the packing procedure
  printf("Test the packing procedure by checking the return value\n");
  int pack_result = fapi_nr_p5_message_pack(&req, msg_len, msg_buf, sizeof(msg_buf), NULL);

  // PARAM.request message body length is 0
  AssertFatal(pack_result == 0 + NFAPI_HEADER_LENGTH,
              "fapi_p5_message_pack packed_length not equal to NFAPI_HEADER_LENGTH + body length (8+0)! Reported value was %d\n",
              pack_result);
  printf("fapi_p5_message_pack packed_length 0x%02x\n", pack_result);
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
  printf("num_msg: 0x%02x\n", header.num_msg);
  AssertFatal(header.num_msg == req.header.num_msg,
              "num_msg was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.num_msg,
              req.header.num_msg);
  printf("opaque_handle: 0x%02x\n", header.opaque_handle);
  AssertFatal(header.opaque_handle == req.header.opaque_handle,
              "Spare was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.opaque_handle,
              req.header.opaque_handle);
  printf("Message ID : 0x%02x\n", header.message_id);
  AssertFatal(header.message_id == req.header.message_id,
              "Message ID was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.message_id,
              req.header.message_id);
  printf("Message length : 0x%02x\n", header.message_length);
  AssertFatal(header.message_length == req.header.message_length,
              "Message length was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              header.message_length,
              req.header.message_length);

  printf("Test the unpacking and compare with initial message\n");
  // test the unpacking and compare with initial message
  fapi_nr_param_request_scf_t unpacked_req;
  memset(&unpacked_req, 0, sizeof(unpacked_req));
  int unpack_result =
      fapi_nr_p5_message_unpack(msg_buf, header.message_length + NFAPI_HEADER_LENGTH, &unpacked_req, sizeof(unpacked_req), NULL);
  AssertFatal(unpack_result >= 0, "fapi_nr_p5_message_unpack failed with return %d\n", unpack_result);
  printf("num_msg: 0x%02x\n", unpacked_req.header.num_msg);
  AssertFatal(unpacked_req.header.num_msg == req.header.num_msg,
              "num_msg was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.header.num_msg,
              req.header.num_msg);
  printf("opaque_handle: 0x%02x\n", unpacked_req.header.opaque_handle);
  AssertFatal(unpacked_req.header.opaque_handle == req.header.opaque_handle,
              "opaque_handle was not the same as the packed value! Unpacked value was 0x%02x , and Packed value was 0x%02x\n",
              unpacked_req.header.opaque_handle,
              req.header.opaque_handle);
  printf("Message id : 0x%02x\n", unpacked_req.header.message_id);
  AssertFatal(unpacked_req.header.message_id == req.header.message_id,
              "Message id was not 0x%02x, was 0x%02x\n",
              req.header.message_id,
              unpacked_req.header.message_id);
  printf("Message length : 0x%02x\n", unpacked_req.header.message_length);
  AssertFatal(unpacked_req.header.message_length == req.header.message_length,
              "Message length was not the same as the value previously packed, was 0x%02x\n",
              unpacked_req.header.message_length);

  // All tests successful!
  return 0;
}
