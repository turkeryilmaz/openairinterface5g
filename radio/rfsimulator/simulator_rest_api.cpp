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
#include <vector>
#include <nlohmann/json.hpp>

extern "C" {
#include "rfsimulator.h"
extern rfsimulator_state_t *rfsimulator_state;
}
namespace rfsimulator::rest_api {
nlohmann::json generate_channel_json(channel_desc_t *channel_model, int conn_sock)
{
  return nlohmann::json({{"name", std::string(channel_model->model_name)},
                         {"conn_sock", conn_sock},
                         {"path_loss_dB", channel_model->path_loss_dB},
                         {"noise_power_dB", channel_model->noise_power_dB},
                         {"nb_tx", channel_model->nb_tx},
                         {"nb_rx", channel_model->nb_rx}});
}

extern "C" void register_routes(crow::Blueprint *bp)
{
  bp->new_rule_dynamic("/healthcheck/")([]() { return crow::response(crow::status::OK, std::string("OK")); });

  bp->new_rule_dynamic("/channel/")([]() {
    nlohmann::json response_data = nlohmann::json::array();
    for (auto buffer_index = 0; buffer_index < MAX_FD_RFSIMU; buffer_index++) {
      auto buf = &rfsimulator_state->buf[buffer_index];
      if (buf->conn_sock != -1 && buf->channel_model != nullptr) {
        response_data.push_back(generate_channel_json(buf->channel_model, buffer_index));
      }
    }
    return crow::response(crow::status::OK, response_data.dump());
  });

  bp->new_rule_dynamic("/channel/<string>/")
      .methods(crow::HTTPMethod::POST, crow::HTTPMethod::GET)([](const crow::request &req, std::string channel_name) {
        nlohmann::json req_json;
        try {
          req_json = nlohmann::json::parse(req.body);
        } catch (const nlohmann::json::exception &e) {
          std::cerr << "Exception: " << e.what() << std::endl;
          return crow::response(crow::status::BAD_REQUEST, std::string(e.what()));
        }

        for (auto buffer_index = 0; buffer_index < MAX_FD_RFSIMU; buffer_index++) {
          auto buf = &rfsimulator_state->buf[buffer_index];
          if (buf->conn_sock != -1 && buf->channel_model != nullptr) {
            if (channel_name.compare(buf->channel_model->model_name) == 0) {
              if (req.method == crow::HTTPMethod::GET) {
                return crow::response(generate_channel_json(buf->channel_model, buf->conn_sock).dump());
              } else {
                const char path_loss_db[] = "path_loss_dB";
                if (req_json.contains(path_loss_db)) {
                  buf->channel_model->path_loss_dB = req_json[path_loss_db];
                }
                return crow::response(crow::status::OK, std::string("OK"));
              }
            }
          }
        }
        return crow::response(crow::status::BAD_REQUEST, std::string("Channel not found"));
      });
}

} // namespace rfsimulator::rest_api
