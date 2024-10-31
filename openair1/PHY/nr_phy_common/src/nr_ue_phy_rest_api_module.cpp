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
#include "nr_ue_phy_rest_api_module.h"
#include "nr_ue_phy_meas.h"
#include "PHY/defs_nr_UE.h"
#include "time_meas.h"


extern "C" {
extern PHY_VARS_NR_UE*** PHY_vars_UE_g;
}
namespace nr_ue::rest_api::phy {
crow::json::wvalue generate_meas_json(time_stats_t *stat)
{
  return crow::json::wvalue({{"name", std::string(stat->meas_name)},
                             {"count", std::uint64_t(stat->trials)},
                             {"uS", (float)(stat->diff / (cpu_freq_GHz * 1000))}});
}

void register_routes(crow::Blueprint *bp)
{
  bp->new_rule_dynamic("/healthcheck")
  ([](){
    return crow::response(std::string("OK"));
  });
  bp->new_rule_dynamic("/cpumeas/<int>")
  ([](int ue_index){
    if (ue_index != 0) {
      return crow::response(crow::status::BAD_REQUEST);
    }
    std::vector<crow::json::wvalue> response_data;
    PHY_VARS_NR_UE *phy_vars_nr_ue = PHY_vars_UE_g[ue_index][0];
    nr_ue_phy_cpu_stat_t* phy_cpu_stats = &phy_vars_nr_ue->phy_cpu_stats;
    for (auto meas_index = 0; meas_index < MAX_CPU_STAT_TYPE; meas_index++) {
      response_data.push_back(generate_meas_json(&phy_cpu_stats->cpu_time_stats[meas_index]));
    }
    return crow::response(crow::json::wvalue(response_data));
  });

  bp->new_rule_dynamic("/cpumeas/<int>/<int>")
  ([](int ue_index, int meas_index) {
    if (ue_index != 0) {
      return crow::response(crow::status::BAD_REQUEST);
    }
    if (meas_index < 0 || meas_index >= MAX_CPU_STAT_TYPE) {
        return crow::response(crow::status::BAD_REQUEST);
    }
    PHY_VARS_NR_UE *phy_vars_nr_ue = PHY_vars_UE_g[ue_index][0];
    nr_ue_phy_cpu_stat_t* phy_cpu_stats = &phy_vars_nr_ue->phy_cpu_stats;
    return crow::response(generate_meas_json(&phy_cpu_stats->cpu_time_stats[meas_index]));
  });

  bp->new_rule_dynamic("/cpumeas/enable")
  ([]() {
    cpumeas(CPUMEAS_ENABLE);
    return crow::response(crow::status::OK, std::string("OK"));
  });

  bp->new_rule_dynamic("/cpumeas/disable")
  ([]() {
    cpumeas(CPUMEAS_DISABLE);
    return crow::response(crow::status::OK, std::string("OK"));
  });
}

} // namespace nr_ue::rest_api::phy
