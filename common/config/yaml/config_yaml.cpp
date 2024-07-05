#include "yaml-cpp/yaml.h"
extern "C" {
#include "common/config/config_load_configmodule.h"
#include "common/config/config_userapi.h"
void *config_allocate_new(configmodule_interface_t *cfg, int sz, bool autoFree);
void config_check_valptr(configmodule_interface_t *cfg, paramdef_t *cfgoptions, int elt_sz, int nb_elt);
}
#include <cstring>
#include <string>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace config_yaml {
class YamlConfig {
 public:
  YAML::Node config;
  YamlConfig(std::string filename)
  {
    config = YAML::LoadFile(filename);
  }
};

void SetDefault(paramdef_t *param)
{
  switch (param->type) {
    case TYPE_INT:
      *param->iptr = param->defintval;
      break;
    case TYPE_UINT:
      *param->uptr = param->defuintval;
      break;
    case TYPE_STRING:
      *param->strptr = strdup(param->defstrval);
      break;
    case TYPE_UINT8:
      *param->i8ptr = param->defintval;
      break;
    case TYPE_INT16:
      *param->i16ptr = param->defintval;
      break;
    case TYPE_UINT16:
      *param->u16ptr = param->defuintval;
      break;
    case TYPE_INT64:
      *param->i64ptr = param->defint64val;
      break;
    case TYPE_UINT64:
      *param->u64ptr = param->defint64val;
      break;
    case TYPE_DOUBLE:
      *param->dblptr = param->defdblval;
      break;
    default:
      AssertFatal(false, "Unhandled type");
  }
}

void SetNonDefault(const YAML::Node &node, paramdef_t *param)
{
  auto optname = std::string(param->optname);
  switch (param->type) {
    case TYPE_INT:
      *param->iptr = node[optname].as<int>();
      break;
    case TYPE_UINT:
      *param->uptr = node[optname].as<uint>();
      break;
    case TYPE_STRING:
      *param->strptr = strdup(node[optname].as<std::string>().c_str());
      break;
    case TYPE_UINT8:
      *param->i8ptr = node[optname].as<uint8_t>();
      break;
    case TYPE_INT16:
      *param->i16ptr = node[optname].as<int16_t>();
      break;
    case TYPE_UINT16:
      *param->u16ptr = node[optname].as<uint16_t>();
      break;
    case TYPE_INT64:
      *param->i64ptr = node[optname].as<int64_t>();
      break;
    case TYPE_UINT64:
      *param->u64ptr = node[optname].as<uint64_t>();
      break;
    case TYPE_DOUBLE:
      *param->dblptr = node[optname].as<double>();
      break;
  }
}

void GetParams(const YAML::Node &node, paramdef_t *params, int num_params)
{
  for (auto i = 0; i < num_params; i++) {
    if (node && node[std::string(params[i].optname)]) {
      SetNonDefault(node, &params[i]);
    } else {
      SetDefault(&params[i]);
    }
  }
}

static YamlConfig *config;
static YAML::Node invalid_node;
} // namespace config_yaml

int config_yaml_init(configmodule_interface_t *cfg)
{
  char **cfgP = cfg->cfgP;
  cfg->numptrs = 0;
  pthread_mutex_init(&cfg->memBlocks_mutex, NULL);
  memset(cfg->oneBlock, 0, sizeof(cfg->oneBlock));

  config_yaml::config = new config_yaml::YamlConfig(std::string(cfgP[0]));
  return 0;
}

void config_yaml_end(configmodule_interface_t *cfg)
{
  delete config_yaml::config;
}

const YAML::Node& find_node_recursive(const YAML::Node &node, std::stringstream& prefix) {
  std::string word;
  if (prefix >> word) {
    if (node[word]) {
      return find_node_recursive(node[word], prefix);
    }
    else {
      return config_yaml::invalid_node;
    }
  }
  return node;
}

const YAML::Node &find_node(char *prefix)
{
  std::string s = std::string(prefix);
  std::replace(s.begin(), s.end(), '.', ' ');
  std::stringstream ss(s);
  return find_node_recursive(config_yaml::config->config, ss);
}

int config_yaml_get(configmodule_interface_t *cfg, paramdef_t *cfgoptions, int numoptions, char *prefix)
{
  const YAML::Node& node = find_node(prefix);
  if (node == config_yaml::invalid_node) {
    return -1;
  }
  for (auto i = 0; i < numoptions; i++) {
    if (cfgoptions[i].type != TYPE_STRING && cfgoptions[i].voidptr == nullptr) {
      config_check_valptr(cfg, &cfgoptions[i], sizeof(void*), 1);
    }
  }
  config_yaml::GetParams(node, cfgoptions, numoptions);
  return 0;
}

int config_yaml_getlist(configmodule_interface_t *cfg, paramlist_def_t *ParamList, paramdef_t *params, int numparams, char *prefix)
{
  char path[512];
  if (prefix != nullptr) {
    sprintf(path, "%s.%s", prefix, ParamList->listname);
  } else {
    sprintf(path, "%s", ParamList->listname);
  }
  ParamList->numelt = 0;
  auto node = find_node(path);
  if (node == config_yaml::invalid_node) {
    return -1;
  }

  if (!node.IsSequence()) {
    return -1;
  }
  ParamList->numelt = numparams;

  if (ParamList->numelt > 0 && params != NULL) {
    ParamList->paramarray = static_cast<paramdef_t **>(config_allocate_new(cfg, ParamList->numelt * sizeof(paramdef_t *), true));

    for (int i = 0; i < ParamList->numelt; i++) {
      ParamList->paramarray[i] = static_cast<paramdef *>(config_allocate_new(cfg, numparams * sizeof(paramdef_t), true));
      memcpy(ParamList->paramarray[i], params, sizeof(paramdef_t) * numparams);

      for (int j = 0; j < numparams; j++) {
        ParamList->paramarray[i][j].strptr = NULL;
        if (ParamList->paramarray[i][j].type != TYPE_STRING) {
          config_check_valptr(cfg, &ParamList->paramarray[i][j], sizeof(void*), 1);
        }
      }

      config_yaml::GetParams(node[i], ParamList->paramarray[i], numparams);
    }
  }

  return 0;
}

void config_yaml_write_parsedcfg(configmodule_interface_t *cfg)
{
  (void)cfg;
}

int config_yaml_set(configmodule_interface_t *cfg, paramdef_t *cfgoptions, int numoptions, char *prefix)
{
  (void)cfg;
  (void)cfgoptions;
  (void)numoptions;
  (void)prefix;
  return 0;
}
