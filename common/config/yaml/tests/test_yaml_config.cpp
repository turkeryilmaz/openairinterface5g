#include "gtest/gtest.h"
extern "C" {
uint64_t get_softmodem_optmask(void)
{
  return 0;
}
#include "common/config/config_paramdesc.h"
configmodule_interface_t *uniqCfg;
void exit_function(const char *file, const char *function, const int line, const char *s, const int assert)
{
  if (assert) {
    abort();
  } else {
    exit(EXIT_SUCCESS);
  }
}

#include "common/utils/LOG/log.h"
#include "common/config/config_load_configmodule.h"
#include "common/config/config_userapi.h"
}
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <algorithm>

int config_yaml_init(configmodule_interface_t *cfg);
void config_yaml_end(configmodule_interface_t *cfg);
int config_yaml_get(configmodule_interface_t *cfg, paramdef_t *cfgoptions, int numoptions, char *prefix);
int config_yaml_getlist(configmodule_interface_t *cfg, paramlist_def_t *ParamList, paramdef_t *params, int numparams, char *prefix);
void config_yaml_write_parsedcfg(configmodule_interface_t *cfg);
int config_yaml_set(configmodule_interface_t *cfg, paramdef_t *cfgoptions, int numoptions, char *prefix);

TEST(yaml_config, yaml_basic) {
  configmodule_interface_t cfg;
  cfg.cfgP[0] = strdup("test1.yaml");
  EXPECT_EQ(config_yaml_init(&cfg), 0);
  config_yaml_end(&cfg);
}


TEST(yaml_config, yaml_get_existing_values) {
  configmodule_interface_t cfg;
  cfg.cfgP[0] = strdup("test1.yaml");
  EXPECT_EQ(config_yaml_init(&cfg), 0);

  // Testing paremters present in the test node
  paramdef_t p = {0};
  for (auto i = 1; i <= 3; i++) {
    sprintf(p.optname, "%s%d", "value", i);
    uint16_t value;
    p.type = TYPE_UINT16;
    p.u16ptr = &value;
    char prefix[] = "test";
    config_yaml_get(&cfg, &p, 1, prefix);
    EXPECT_EQ(value, i);
  }


  config_yaml_end(&cfg);
}

TEST(yaml_config, yaml_get_non_existing_values) {
  configmodule_interface_t cfg;
  cfg.cfgP[0] = strdup("test1.yaml");
  EXPECT_EQ(config_yaml_init(&cfg), 0);

  // Testing paremters present in the test node
  paramdef_t p = {0};
  for (auto i = 4; i <= 5; i++) {
    sprintf(p.optname, "%s%d", "value", i);
    uint16_t value;
    p.type = TYPE_UINT16;
    p.u16ptr = &value;
    p.defuintval = i;
    char prefix[] = "test";
    config_yaml_get(&cfg, &p, 1, prefix);
    EXPECT_EQ(value, i);
  }

  config_yaml_end(&cfg);
}

TEST(yaml_config, test_high_recusion) {
  configmodule_interface_t cfg;
  cfg.cfgP[0] = strdup("test_recursion.yaml");
  EXPECT_EQ(config_yaml_init(&cfg), 0);

  // Testing paremters present in the test node
  paramdef_t p = {0};
  for (auto i = 1; i <= 3; i++) {
    sprintf(p.optname, "%s%d", "value", i);
    uint16_t value;
    p.type = TYPE_UINT16;
    p.u16ptr = &value;
    char prefix[] = "test.test1.test2.test3.test4";
    EXPECT_EQ(config_yaml_get(&cfg, &p, 1, prefix), 0);
    EXPECT_EQ(value, i);
  }

  config_yaml_end(&cfg);
}

TEST(yaml_config, test_list) {
  configmodule_interface_t cfg;
  cfg.cfgP[0] = strdup("test_list.yaml");
  EXPECT_EQ(config_yaml_init(&cfg), 0);
//int config_yaml_getlist(configmodule_interface_t *cfg, paramlist_def_t *ParamList, paramdef_t *params, int numparams, char *prefix)

  paramlist_def_t param_list = {0};
  sprintf(param_list.listname, "%s", "test");
  paramdef_t params[3] = {0};
  uint16_t value[3];
  for (auto i = 0; i < 3; i++) {
    sprintf(params[i].optname, "%s%d", "value", i+1);
    params[i].type = TYPE_UINT16;
    params[i].u16ptr = &value[i];
  }


  config_yaml_getlist(&cfg, &param_list, params, 3, nullptr);

  for (auto i = 0; i < 3; i++) {
    for (auto j = 0; j < 3; j++) {
      EXPECT_EQ(*param_list.paramarray[i][j].u16ptr, i * 3 + j + 1);
    }
  }
  config_yaml_end(&cfg);
}

int main(int argc, char** argv)
{
  logInit();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

