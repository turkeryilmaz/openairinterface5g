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

#include <crow.h>
#include "nr_ue_phy_rest_api_module.h"
#define PORT 11101

void api_run(void);

extern "C" {
void nr_ue_rest_api_thread(void* arg)
{
  api_run();
}
}

void api_run(void)
{
  crow::SimpleApp ue_rest_api;

  CROW_ROUTE(ue_rest_api, "/healthcheck")([]() { return "OK"; });

  crow::Blueprint bp = nr_ue::rest_api::phy::register_routes("phy");
  ue_rest_api.register_blueprint(bp);

  ue_rest_api.port(PORT).run();
}
