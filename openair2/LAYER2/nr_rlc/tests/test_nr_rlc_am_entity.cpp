/*
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.1  (the "License"); you may not use this file
 * except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.openairinterface.org/?page_id=698
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *-------------------------------------------------------------------------------
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#include "gtest/gtest.h"
extern "C" {
#include "openair2/LAYER2/nr_rlc/nr_rlc_entity_am.h"
#include "openair2/LAYER2/nr_rlc/nr_rlc_entity.h"
#include "common/utils/LOG/log.h"
#include "common/config/config_load_configmodule.h"
extern configmodule_interface_t *uniqCfg;
}
#include <iostream>

int sdu_delivered_count = 0;
int sdu_acked_count = 0;
void deliver_sdu(void *deliver_sdu_data, struct nr_rlc_entity_t *entity, char *buf, int size)
{
  sdu_delivered_count++;
  char payload[300];
  ASSERT_LE(size, sizeof(payload));
  snprintf(payload, size, "%s", buf);
  std::cout << "Delivered sdu: " << payload << std::endl;
}

void sdu_successful_delivery(void *sdu_successful_delivery_data, struct nr_rlc_entity_t *entity, int sdu_id)
{
  sdu_acked_count++;
  std::cout << "SDU " << sdu_id << " acked" << std::endl;
}

void max_retx_reached(void *max_retx_reached_data, struct nr_rlc_entity_t *entity)
{
}

TEST(nr_rlc_am_entity, test_init)
{
  nr_rlc_entity_t *entity = new_nr_rlc_entity_am(100,
                                                 100,
                                                 deliver_sdu,
                                                 NULL,
                                                 sdu_successful_delivery,
                                                 NULL,
                                                 max_retx_reached,
                                                 NULL,
                                                 45,
                                                 35,
                                                 0,
                                                 -1,
                                                 -1,
                                                 8,
                                                 12);
  char buf[30];
  EXPECT_EQ(entity->generate_pdu(entity, buf, sizeof(buf)), 0) << "No PDCP SDU provided to RLC, expected no RLC PDUS";
}

TEST(nr_rlc_am_entity, test_segmentation_reassembly)
{
  nr_rlc_entity_t *tx_entity = new_nr_rlc_entity_am(100,
                                                    100,
                                                    deliver_sdu,
                                                    NULL,
                                                    sdu_successful_delivery,
                                                    NULL,
                                                    max_retx_reached,
                                                    NULL,
                                                    45,
                                                    35,
                                                    0,
                                                    -1,
                                                    -1,
                                                    8,
                                                    12);

  nr_rlc_entity_t *rx_entity = new_nr_rlc_entity_am(100,
                                                    100,
                                                    deliver_sdu,
                                                    NULL,
                                                    sdu_successful_delivery,
                                                    NULL,
                                                    max_retx_reached,
                                                    NULL,
                                                    45,
                                                    35,
                                                    0,
                                                    -1,
                                                    -1,
                                                    8,
                                                    12);
  char buf[30] = {0};
  snprintf(buf, sizeof(buf), "%s", "Message");
  EXPECT_EQ(tx_entity->generate_pdu(tx_entity, buf, sizeof(buf)), 0) << "No PDCP SDU provided to RLC, expected no RLC PDUS";

  // Higher layer IF -> deliver PDCP SDU
  tx_entity->recv_sdu(tx_entity, buf, sizeof(buf), 0);
  sdu_delivered_count = 0;
  for (auto i = 0; i < 10; i++) {
    if (sdu_delivered_count == 1) {
      break;
    }
    int size = tx_entity->generate_pdu(tx_entity, buf, 10);
    EXPECT_GT(size, 0);
    rx_entity->recv_pdu(rx_entity, buf, size);
  }
  EXPECT_EQ(sdu_delivered_count, 1);
  sdu_acked_count = 0;
  int size = rx_entity->generate_pdu(rx_entity, buf, 30);
  tx_entity->recv_pdu(tx_entity, buf, size);
  EXPECT_EQ(sdu_acked_count, 1);
}

TEST(nr_rlc_am_entity, test_ack_out_of_order)
{
  nr_rlc_entity_t *tx_entity = new_nr_rlc_entity_am(100,
                                                    100,
                                                    deliver_sdu,
                                                    NULL,
                                                    sdu_successful_delivery,
                                                    NULL,
                                                    max_retx_reached,
                                                    NULL,
                                                    45,
                                                    35,
                                                    0,
                                                    4,
                                                    -1,
                                                    8,
                                                    12);

  nr_rlc_entity_t *rx_entity = new_nr_rlc_entity_am(100,
                                                    100,
                                                    deliver_sdu,
                                                    NULL,
                                                    sdu_successful_delivery,
                                                    NULL,
                                                    max_retx_reached,
                                                    NULL,
                                                    45,
                                                    35,
                                                    0,
                                                    4,
                                                    -1,
                                                    8,
                                                    12);

  char buf[30] = {0};
  EXPECT_EQ(tx_entity->generate_pdu(tx_entity, buf, sizeof(buf)), 0) << "No PDCP SDU provided to RLC, expected no RLC PDUS";
  // Higher layer IF -> deliver PDCP SDU
  // Generate 4 PDCP SDUS
  snprintf(buf, sizeof(buf), "%s", "Message 1");
  tx_entity->recv_sdu(tx_entity, buf, sizeof(buf), 0);
  snprintf(buf, sizeof(buf), "%s", "Message 2");
  tx_entity->recv_sdu(tx_entity, buf, sizeof(buf), 1);
  snprintf(buf, sizeof(buf), "%s", "Message 3");
  tx_entity->recv_sdu(tx_entity, buf, sizeof(buf), 2);

  sdu_delivered_count = 0;
  char pdu_buf[40];
  for (auto i = 0; i < 3; i++) {
    int size = tx_entity->generate_pdu(tx_entity, pdu_buf, sizeof(pdu_buf));
    EXPECT_GT(size, 0);
    // Do not deliver PDU 2 to RX entity
    if (i != 1) {
      rx_entity->recv_pdu(rx_entity, pdu_buf, size);
    }
  }
  EXPECT_EQ(sdu_delivered_count, 2) << "Expected 2 out of 3 SDUs delivered";

  sdu_acked_count = 0;

  int size = rx_entity->generate_pdu(rx_entity, pdu_buf, sizeof(pdu_buf));
  EXPECT_GT(size, 0);
  tx_entity->recv_pdu(tx_entity, pdu_buf, size);

  EXPECT_EQ(sdu_acked_count, 1);
}

static void print_list_size(const char *name, nr_rlc_sdu_segment_t *list)
{
  int list_size = 0;
  while (list != NULL) {
    list = list->next;
    list_size++;
  }
  std::cout << name << " list size: " << list_size << std::endl;
}

TEST(nr_rlc_am_entity, performance)
{
  nr_rlc_entity_t *tx_entity = new_nr_rlc_entity_am(10000,
                                                    10000,
                                                    deliver_sdu,
                                                    NULL,
                                                    sdu_successful_delivery,
                                                    NULL,
                                                    max_retx_reached,
                                                    NULL,
                                                    45,
                                                    35,
                                                    0,
                                                    4,
                                                    -1,
                                                    8,
                                                    12);

  nr_rlc_entity_t *rx_entity = new_nr_rlc_entity_am(10000,
                                                    10000,
                                                    deliver_sdu,
                                                    NULL,
                                                    sdu_successful_delivery,
                                                    NULL,
                                                    max_retx_reached,
                                                    NULL,
                                                    45,
                                                    35,
                                                    0,
                                                    4,
                                                    -1,
                                                    8,
                                                    12);
  char buf[30] = {0};
  char pdu_buf[40];
  for (int i = 0; i < 10000; i++) {
    snprintf(buf, sizeof(buf), "%s%d", "Message", i);
    tx_entity->recv_sdu(tx_entity, buf, sizeof(buf), i);
    int size = tx_entity->generate_pdu(tx_entity, pdu_buf, sizeof(pdu_buf));
    if (size != 0) {
      // Losing one out of 10 packets
      if (i % 10 != 0)
        rx_entity->recv_pdu(rx_entity, pdu_buf, size);
    }
    if (i % 5 == 0) {
      size = rx_entity->generate_pdu(rx_entity, pdu_buf, sizeof(pdu_buf));
      if (size != 0) {
        // No packet lost in the other direction
        tx_entity->recv_pdu(tx_entity, pdu_buf, size);
      }
    }
    tx_entity->set_time(tx_entity, i + 1);
    rx_entity->set_time(tx_entity, i + 1);
  }

  nr_rlc_entity_am_t *tx_entity_am = (nr_rlc_entity_am_t *)tx_entity;
  print_list_size("wait", tx_entity_am->wait_list);
  print_list_size("restransmit", tx_entity_am->retransmit_list);
}

int main(int argc, char **argv)
{
  logInit();
  uniqCfg = load_configmodule(argc, argv, CONFIG_ENABLECMDLINEONLY);
  // g_log->log_component[RLC].level = OAILOG_TRACE;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
