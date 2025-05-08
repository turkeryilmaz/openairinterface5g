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
* Author and copyright: Laurent Thomas, open-cells.com
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


/*
 * Open issues and limitations
 * The read and write should be called in the same thread, that is not new USRP UHD design
 * When the opposite side switch from passive reading to active R+Write, the synchro is not fully deterministic
 */

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <sys/epoll.h>
#include <string.h>
#include <zmq.h>
#include <common/utils/assertions.h>
#include <common/utils/LOG/log.h>
#include <common/utils/load_module_shlib.h>
#include <common/utils/telnetsrv/telnetsrv.h>
#include <common/config/config_userapi.h>
#include "common_lib.h"
#include <openair1/PHY/defs_eNB.h>
#include "openair1/PHY/defs_UE.h"
#define CHANNELMOD_DYNAMICLOAD
#include <openair1/SIMULATION/TOOLS/sim.h>
#include "rfsimulator.h"
#include <sys/time.h>
#define XSUBPORT 5555 // default ports for this simulator in pubsub
#define XPUBPORT 5556

#define PORT 4043
#define CirSize 48880000 
// #define CirSize 6144000 // 100ms is enough
#define sampleToByte(a,b) ((a)*(b)*sizeof(sample_t))
#define byteToSample(a,b) ((a)/(sizeof(sample_t)*(b)))

#define MAX_SIMULATION_CONNECTED_NODES 5
#define GENERATE_CHANNEL 10 //each frame in DL
#define SEND_BUFF_SIZE 100000000 // Socket buffer size

//
// typedef enum { SIMU_ROLE_SERVER = 1, SIMU_ROLE_CLIENT } simuRole;
#define RFSIMU_SECTION    "rfsimulator"
#define RFSIMU_OPTIONS_PARAMNAME "options"


#define RFSIM_CONFIG_HELP_OPTIONS     " list of comma separated options to enable rf simulator functionalities. Available options: \n"\
  "        chanmod:   enable channel modelisation\n"\
  "        saviq:     enable saving written iqs to a file\n"
/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
/*                                            configuration parameters for the rfsimulator device                                                                              */
/*   optname                     helpstr                     paramflags           XXXptr                               defXXXval                          type         numelt  */
/*-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------*/
#define simOpt PARAMFLAG_NOFREE|PARAMFLAG_CMDLINE_NOPREFIXENABLED
#define RFSIMULATOR_PARAMS_DESC {					\
    {"brokerip",             "<broker ip address to connect to>\n",        simOpt,  .strptr=&rfsimulator->brokerip,               .defstrval="0.0.0.0",           TYPE_STRING,    0 },\
    {"serveraddr",             "<ip address to connect to>\n",        simOpt,  .strptr=&rfsimulator->ip,               .defstrval="127.0.0.1",           TYPE_STRING,    0 },\
    {"xsubport",             "<port to connect to xsubsocket>\n",        simOpt,  .u16ptr=&rfsimulator->xsubport,               .defuintval=XSUBPORT,           TYPE_UINT16,    0 },\
    {"xpubport",             "<port to connect to xpubsocket>\n",        simOpt,  .u16ptr=&rfsimulator->xpubport,               .defuintval=XPUBPORT,           TYPE_UINT16,    0 },\
    {"device_id",             "<device id>\n",              simOpt,  .strptr=&rfsimulator->device_id,           .defstrval="0",                 TYPE_STRING,    0 },\
    {RFSIMU_OPTIONS_PARAMNAME, RFSIM_CONFIG_HELP_OPTIONS,             0,       .strlistptr=NULL,                       .defstrlistval=NULL,              TYPE_STRINGLIST,0 },\
    {"IQfile",                 "<file path to use when saving IQs>\n",simOpt,  .strptr=&saveF,                         .defstrval="/tmp/rfsimulator.iqs",TYPE_STRING,    0 },\
    {"modelname",              "<channel model name>\n",              simOpt,  .strptr=&modelname,                     .defstrval="AWGN",                TYPE_STRING,    0 },\
    {"ploss",                  "<channel path loss in dB>\n",         simOpt,  .dblptr=&(rfsimulator->chan_pathloss),  .defdblval=0,                     TYPE_DOUBLE,    0 },\
    {"forgetfact",             "<channel forget factor ((0 to 1)>\n", simOpt,  .dblptr=&(rfsimulator->chan_forgetfact),.defdblval=0,                     TYPE_DOUBLE,    0 },\
    {"offset",                 "<channel offset in samps>\n",         simOpt,  .iptr=&(rfsimulator->chan_offset),      .defintval=0,                     TYPE_INT,       0 },\
    {"wait_timeout",           "<wait timeout if no UE connected>\n", simOpt,  .iptr=&(rfsimulator->wait_timeout),     .defintval=1,                     TYPE_INT,       0 },\
  };

static void getset_currentchannels_type(char *buf, int debug, webdatadef_t *tdata, telnet_printfunc_t prnt);
extern int get_currentchannels_type(char *buf, int debug, webdatadef_t *tdata, telnet_printfunc_t prnt); // in random_channel.c
static int rfsimu_setchanmod_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg);
static int rfsimu_setdistance_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg);
static int rfsimu_getdistance_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg);
static int rfsimu_vtime_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg);
static telnetshell_cmddef_t rfsimu_cmdarray[] = {
    {"show models", "", (cmdfunc_t)rfsimu_setchanmod_cmd, {(webfunc_t)getset_currentchannels_type}, TELNETSRV_CMDFLAG_WEBSRVONLY | TELNETSRV_CMDFLAG_GETWEBTBLDATA, NULL},
    {"setmodel", "<model name> <model type>", (cmdfunc_t)rfsimu_setchanmod_cmd, {NULL}, TELNETSRV_CMDFLAG_PUSHINTPOOLQ | TELNETSRV_CMDFLAG_TELNETONLY, NULL},
    {"setdistance", "<model name> <distance>", (cmdfunc_t)rfsimu_setdistance_cmd, {NULL}, TELNETSRV_CMDFLAG_PUSHINTPOOLQ | TELNETSRV_CMDFLAG_NEEDPARAM },
    {"getdistance", "<model name>", (cmdfunc_t)rfsimu_getdistance_cmd, {NULL}, TELNETSRV_CMDFLAG_PUSHINTPOOLQ},
    {"vtime", "", (cmdfunc_t)rfsimu_vtime_cmd, {NULL}, TELNETSRV_CMDFLAG_PUSHINTPOOLQ | TELNETSRV_CMDFLAG_AUTOUPDATE},
    {"", "", NULL},
};
static telnetshell_cmddef_t *setmodel_cmddef = &(rfsimu_cmdarray[1]);

static telnetshell_vardef_t rfsimu_vardef[] = {{"", 0, 0, NULL}};
pthread_mutex_t Sockmutex;

unsigned int nb_ue = 0;

typedef c16_t sample_t; // 2*16 bits complex number

typedef struct buffer_s {
  int fd_pub_sock;
  int fd_sub_sock;
  int conn_device_id;
  openair0_timestamp lastReceivedTS;
  bool headerMode;
  bool trashingPacket;
  samplesBlockHeader_t th;
  char *transferPtr;
  uint64_t remainToTransfer;
  char *circularBufEnd;
  sample_t *circularBuf;
  channel_desc_t *channel_model;
} buffer_t;

typedef struct {
  openair0_timestamp nextRxTstamp;
  openair0_timestamp lastWroteTS;
  uint64_t typeStamp;
  char *device_id;
  void *context;
  void *pub_sock;
  void *sub_sock;
  int fd_pub_sock;
  int fd_sub_sock;
  char connected_devices[16]; // to keep track of connected devices
  char *brokerip;
  uint16_t xsubport;
  uint16_t xpubport;
  char *ip;
  uint16_t port;
  int saveIQfile;
  buffer_t buf[FD_SETSIZE];
  int rx_num_channels;
  int tx_num_channels;
  double sample_rate;
  double tx_bw;
  int channelmod;
  double chan_pathloss;
  double chan_forgetfact;
  int    chan_offset;
  float  noise_power_dB;
  void *telnetcmd_qid;
  poll_telnetcmdq_func_t poll_telnetcmdq;
  int wait_timeout;
  char *transferPtrTr;
} rfsimulator_state_t;


static void allocCirBuf(rfsimulator_state_t *bridge, int id) {
  buffer_t *ptr=&bridge->buf[id];
  AssertFatal ( (ptr->circularBuf=(sample_t *) malloc(sampleToByte(CirSize,1))) != NULL, "");
  ptr->circularBufEnd=((char *)ptr->circularBuf)+sampleToByte(CirSize,1);
  ptr->conn_device_id = id;
  ptr->fd_pub_sock = bridge->fd_pub_sock;
  ptr->fd_sub_sock = bridge->fd_sub_sock;
  ptr->lastReceivedTS=0;
  ptr->headerMode=true;
  ptr->trashingPacket=false;
  ptr->transferPtr=(char *)&ptr->th;
  ptr->remainToTransfer=sizeof(samplesBlockHeader_t);
  int sendbuff=1000*1000*100;;
  size_t optlen = sizeof(sendbuff);
  if (zmq_setsockopt(bridge->pub_sock, ZMQ_SNDBUF, &sendbuff, optlen) != 0) {
    LOG_E(HW, "zmq_setsockopt(SO_SNDBUF) failed\n");
  }
  
  if (bridge->typeStamp == ENB_MAGICDL) {
    int rcvhwm = 0;
    if (zmq_setsockopt(bridge->sub_sock, ZMQ_RCVHWM, &rcvhwm, sizeof(int)) != 0) {
      LOG_E(HW, "zmq_setsockopt(ZMQ_RCVHWM) failed\n");
    };
  }

  if ( bridge->channelmod > 0) {
    // create channel simulation model for this mode reception
    // snr_dB is pure global, coming from configuration paramter "-s"
    // Fixme: referenceSignalPower should come from the right place
    // but the datamodel is inconsistant
    // legacy: RC.ru[ru_id]->frame_parms.pdsch_config_common.referenceSignalPower
    // (must not come from ru[]->frame_parms as it doesn't belong to ru !!!)
    // Legacy sets it as:
    // ptr->channel_model->path_loss_dB = -132.24 + snr_dB - RC.ru[0]->frame_parms->pdsch_config_common.referenceSignalPower;
    // we use directly the paramter passed on the command line ("-s")
    // the value channel_model->path_loss_dB seems only a storage place (new_channel_desc_scm() only copy the passed value)
    // Legacy changes directlty the variable channel_model->path_loss_dB place to place
    // while calling new_channel_desc_scm() with path losses = 0
    static bool init_done=false;

    if (!init_done) {
      uint64_t rand;
      FILE *h=fopen("/dev/random","r");

      if ( 1 != fread(&rand,sizeof(rand),1,h) )
        LOG_W(HW, "Simulator can't read /dev/random\n");

      fclose(h);
      randominit(rand);
      tableNor(rand);
      init_done=true;
    }
    char *modelname = (bridge->typeStamp == ENB_MAGICDL) ? "rfsimu_channel_ue0":"rfsimu_channel_enB0";
    ptr->channel_model=find_channel_desc_fromname(modelname); // path_loss in dB
    AssertFatal((ptr->channel_model!= NULL),"Channel model %s not found, check config file\n",modelname);
    set_channeldesc_owner(ptr->channel_model, RFSIMU_MODULEID);
    random_channel(ptr->channel_model,false);
  }
}

static void removeCirBuf(rfsimulator_state_t *bridge, int id) {
  free(bridge->buf[id].circularBuf);
  // Fixme: no free_channel_desc_scm(bridge->buf[sock].channel_model) implemented
  // a lot of mem leaks
  //free(bridge->buf[sock].channel_model);
  memset(&bridge->buf[id], 0, sizeof(buffer_t));
  bridge->buf[id].fd_pub_sock=-1;
  bridge->buf[id].fd_sub_sock=-1;
  nb_ue--;
}


enum  blocking_t {
  notBlocking,
  blocking
};


static bool flushInput(rfsimulator_state_t *t, int timeout, int nsamps);
static int rfsimulator_write_internal(rfsimulator_state_t *t,
                                      openair0_timestamp timestamp,
                                      void **samplesVoid,
                                      int nsamps,
                                      int nbAnt,
                                      int flags,
                                      bool alreadyLocked,
                                      int firstMessage);

static void fullwrite(int *pub_sock, void *_buf, ssize_t count, rfsimulator_state_t *t, int firstMessage) {
  if (t->saveIQfile != -1) {
    if (write(t->saveIQfile, _buf, count) != count )
      LOG_E(HW,"write in save iq file failed (%s)\n",strerror(errno));
  }

  char *buf = _buf;
  ssize_t l;
  // Sending topic
  while (count) {
    if (t->typeStamp == ENB_MAGICDL){
      if (firstMessage) { // to avoid race conditions when having multiple UEs
        char topic[] = "first";
        LOG_D(HW,"sending first message\n");
        // zmq_send(pub_sock, topic, strlen(topic), ZMQ_SNDMORE | ZMQ_DONTWAIT);
        zmq_send(pub_sock, topic, strlen(topic), ZMQ_SNDMORE);
      } else {
        char topic[] = "sync";
        LOG_D(HW,"sending data on sync topic\n");
        // zmq_send(pub_sock, topic, strlen(topic), ZMQ_SNDMORE | ZMQ_DONTWAIT);
        zmq_send(pub_sock, topic, strlen(topic), ZMQ_SNDMORE);
      }
    } else {
      LOG_D(HW,"sending data on ue topic\n");
      char topic[] = "ue";
      char formatted_topic[256];

      // Format the topic with the device ID
      sprintf(formatted_topic, "%s %s", topic, t->device_id);
      // zmq_send(pub_sock, formatted_topic, strlen(formatted_topic), ZMQ_SNDMORE | ZMQ_DONTWAIT);
      zmq_send(pub_sock, formatted_topic, strlen(formatted_topic), ZMQ_SNDMORE);
      LOG_D(HW,"sending formatted topic: %s\n",formatted_topic);
    }

    // l = zmq_send(pub_sock, buf, count, ZMQ_DONTWAIT);
    l = zmq_send(pub_sock, buf, count, 0);

    if (l == 0) {
      LOG_E(HW, "write() failed, returned 0\n");
      return;
    }
    if (l < 0) {
      if (errno == EINTR)
        continue;

      if (errno == EAGAIN) {
        LOG_D(HW, "write() failed, errno(%d)\n", errno);
        usleep(250);
        continue;
      } else {
        LOG_E(HW, "write() failed, errno(%d)\n", errno);
        return;
      }
    }
    LOG_D(HW, "Successfully sent %zd bytes.\n", l);

    count -= l;
    buf += l;
  }
}

static void rfsimulator_readconfig(rfsimulator_state_t *rfsimulator) {
  char *saveF=NULL;
  char *modelname=NULL;
  paramdef_t rfsimu_params[] = RFSIMULATOR_PARAMS_DESC;
  int p = config_paramidx_fromname(rfsimu_params,sizeof(rfsimu_params)/sizeof(paramdef_t), RFSIMU_OPTIONS_PARAMNAME) ;
  int ret = config_get( rfsimu_params,sizeof(rfsimu_params)/sizeof(paramdef_t),RFSIMU_SECTION);
  AssertFatal(ret >= 0, "configuration couldn't be performed");
  rfsimulator->saveIQfile = -1;

  for(int i=0; i<rfsimu_params[p].numelt ; i++) {
    if (strcmp(rfsimu_params[p].strlistptr[i],"saviq") == 0) {
      rfsimulator->saveIQfile=open(saveF,O_APPEND| O_CREAT|O_TRUNC | O_WRONLY, 0666);

      if ( rfsimulator->saveIQfile != -1 )
        LOG_I(HW,"rfsimulator: will save written IQ samples  in %s\n", saveF);
      else
        LOG_E(HW, "can't open %s for IQ saving (%s)\n", saveF, strerror(errno));

      break;
    } else if (strcmp(rfsimu_params[p].strlistptr[i],"chanmod") == 0) {
      init_channelmod();
      load_channellist(rfsimulator->tx_num_channels, rfsimulator->rx_num_channels, rfsimulator->sample_rate, rfsimulator->tx_bw);
      rfsimulator->channelmod=true;
    } else {
      fprintf(stderr,"Unknown rfsimulator option: %s\n",rfsimu_params[p].strlistptr[i]);
      exit(-1);
    }
  }

  /* for compatibility keep environment variable usage */
  if ( getenv("RFSIMULATOR") != NULL ) {
    rfsimulator->ip=getenv("RFSIMULATOR");
    LOG_W(HW, "The RFSIMULATOR environment variable is deprecated and support will be removed in the future. Instead, add parameter --rfsimulator.serveraddr %s to set the server address. Note: the default is \"server\"; for the gNB/eNB, you don't have to set any configuration.\n", rfsimulator->ip);
    LOG_I(HW, "Remove RFSIMULATOR environment variable to get rid of this message and the sleep.\n");
    sleep(10);
  }

  if ( strncasecmp(rfsimulator->ip,"enb",3) == 0 ||
       strncasecmp(rfsimulator->ip,"server",3) == 0 )
    rfsimulator->typeStamp = ENB_MAGICDL;
  else
    rfsimulator->typeStamp = UE_MAGICDL;
}

static int rfsimu_setchanmod_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg) {
  char *modelname=NULL;
  char *modeltype=NULL;
  rfsimulator_state_t *t = (rfsimulator_state_t *)arg;
  if (t->channelmod == false) {
    prnt("ERROR channel modelisation disabled...\n");
    return 0;
  }
  if (buff == NULL) {
    prnt("ERROR wrong rfsimu setchannelmod command...\n");
    return 0;
  }
  if (debug)
  	  prnt("rfsimu_setchanmod_cmd buffer \"%s\"\n",buff);
  int s = sscanf(buff,"%m[^ ] %ms\n",&modelname, &modeltype);

  if (s == 2) {
    int channelmod=modelid_fromstrtype(modeltype);

    if (channelmod<0)
      prnt("ERROR: model type %s unknown\n",modeltype);
    else {
      rfsimulator_state_t *t = (rfsimulator_state_t *)arg;
      int found=0;
      for (int i=0; i<FD_SETSIZE; i++) {
        buffer_t *b=&t->buf[i];
        if ( b->channel_model==NULL)
          continue;
        if (b->channel_model->model_name==NULL)
          continue;
        if (b->fd_pub_sock >= 0 && (strcmp(b->channel_model->model_name,modelname)==0)) {
          channel_desc_t *newmodel = new_channel_desc_scm(t->tx_num_channels,
                                                          t->rx_num_channels,
                                                          channelmod,
                                                          t->sample_rate,
                                                          0,
                                                          t->tx_bw,
                                                          30e-9, // TDL delay-spread parameter
                                                          0.0,
                                                          CORR_LEVEL_LOW,
                                                          t->chan_forgetfact, // forgetting_factor
                                                          t->chan_offset, // maybe used for TA
                                                          t->chan_pathloss,
                                                          t->noise_power_dB); // path_loss in dB
          set_channeldesc_owner(newmodel, RFSIMU_MODULEID);
          set_channeldesc_name(newmodel,modelname);
          random_channel(newmodel,false);
          channel_desc_t *oldmodel=b->channel_model;
          b->channel_model=newmodel;
          free_channel_desc_scm(oldmodel);
          prnt("New model type %s applied to channel %s connected to sock %d\n",modeltype,modelname,i);
          found=1;
          break;
        }
      } /* for */
      if (found==0)
      	prnt("Channel %s not found or not currently used\n",modelname); 
    }
  } else {
    prnt("ERROR: 2 parameters required: model name and model type (%i found)\n",s);
  }

  free(modelname);
  free(modeltype);
  return CMDSTATUS_FOUND;
}

static void getset_currentchannels_type(char *buf, int debug, webdatadef_t *tdata, telnet_printfunc_t prnt)
{
  if (strncmp(buf, "set", 3) == 0) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "setmodel %s %s", tdata->lines[0].val[1], tdata->lines[0].val[3]);
    push_telnetcmd_func_t push_telnetcmd = (push_telnetcmd_func_t)get_shlibmodule_fptr("telnetsrv", TELNET_PUSHCMD_FNAME);
    push_telnetcmd(setmodel_cmddef, cmd, prnt);
  } else {
    get_currentchannels_type("modify type", debug, tdata, prnt);
  }
}
/*getset_currentchannels_type */

//static void print_cirBuf(struct complex16 *circularBuf,
//                         uint64_t firstSample,
//                         uint32_t cirSize,
//                         int neg,
//                         int pos,
//                         int nbTx)
//{
//  for (int i = -neg; i < pos ; ++i) {
//    for (int txAnt = 0; txAnt < nbTx; txAnt++) {
//      const int idx = ((firstSample + i) * nbTx + txAnt + cirSize) % cirSize;
//      if (i == 0)
//        printf("->");
//      printf("%08x%08x\n", circularBuf[idx].r, circularBuf[idx].i);
//    }
//  }
//  printf("\n");
//}

static void rfsimu_offset_change_cirBuf(struct complex16 *circularBuf,
                                        uint64_t firstSample,
                                        uint32_t cirSize,
                                        int old_offset,
                                        int new_offset,
                                        int nbTx)
{
  //int start = max(new_offset, old_offset) + 10;
  //int end = 10;
  //printf("new_offset %d old_offset %d start %d end %d\n", new_offset, old_offset, start, end);
  //printf("ringbuffer before:\n");
  //print_cirBuf(circularBuf, firstSample, cirSize, start, end, nbTx);

  int doffset = new_offset - old_offset;
  if (doffset > 0) {
    /* Moving away, creating a gap. We need to insert "zero" samples between
     * the previous (end of the) slot and the new slot (at the ringbuffer
     * index) to prevent that the receiving side detects things that are not
     * in the channel (e.g., samples that have already been delivered). */
    for (int i = new_offset; i > 0; --i) {
      for (int txAnt = 0; txAnt < nbTx; txAnt++) {
        const int newidx = ((firstSample - i) * nbTx + txAnt + cirSize) % cirSize;
        if (i > doffset) {
          // shift samples not read yet
          const int oldidx = (newidx + doffset) % cirSize;
          circularBuf[newidx] = circularBuf[oldidx];
        } else {
          // create zero samples between slots
          const struct complex16 nullsample = {0, 0};
          circularBuf[newidx] = nullsample;
        }
      }
    }
  } else {
    /* Moving closer, creating overlap between samples. For simplicity, we
     * simply drop `doffset` samples at the end of the previous slot
     * (this is, in a sense, arbitrary). In a real channel, there would be
     * some overlap between samples, e.g., for `doffset == 1` we could add
     * two samples. I think that we cannot do that for multiple samples,
     * though, and so we just drop some */
    // drop the last -doffset samples of the previous slot
    for (int i = old_offset; i > -doffset; --i) {
      for (int txAnt = 0; txAnt < nbTx; txAnt++) {
        const int oldidx = ((firstSample - i) * nbTx + txAnt + cirSize) % cirSize;
        const int newidx = (oldidx - doffset) % cirSize;
        circularBuf[newidx] = circularBuf[oldidx];
      }
    }
  }

  //printf("ringbuffer after:\n");
  //print_cirBuf(circularBuf, firstSample, cirSize, start, end, nbTx);
}

static int rfsimu_setdistance_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg)
{
  if (debug)
    prnt("%s() buffer \"%s\"\n", __func__, buff);

  char *modelname;
  int distance;
  int s = sscanf(buff,"%m[^ ] %d\n", &modelname, &distance);
  if (s != 2) {
    prnt("require exact two parameters\n");
    return CMDSTATUS_VARNOTFOUND;
  }

  rfsimulator_state_t *t = (rfsimulator_state_t *)arg;
  const double sample_rate = t->sample_rate;
  const double c = 299792458; /* 3e8 */

  const int new_offset = (double) distance * sample_rate / c;
  const double new_distance = (double) new_offset * c / sample_rate;

  prnt("\nnew_offset %d new (exact) distance %.3f m\n", new_offset, new_distance);

  /* Set distance in rfsim and channel model, update channel and ringbuffer */
  for (int i=0; i<FD_SETSIZE; i++) {
    buffer_t *b=&t->buf[i];
    if (b->fd_pub_sock <= 0 || b->channel_model == NULL || b->channel_model->model_name == NULL || strcmp(b->channel_model->model_name, modelname) != 0) {
      if (b->channel_model != NULL && b->channel_model->model_name != NULL)
        prnt("  model %s unmodified\n", b->channel_model->model_name);
      continue;
    }

    channel_desc_t *cd = b->channel_model;
    const int old_offset = cd->channel_offset;
    cd->channel_offset = new_offset;

    const int nbTx = cd->nb_tx;
    prnt("  Modifying model %s...\n", modelname);
    rfsimu_offset_change_cirBuf(b->circularBuf, t->nextRxTstamp, CirSize, old_offset, new_offset, nbTx);
  }

  free(modelname);

  return CMDSTATUS_FOUND;
}

static int rfsimu_getdistance_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg)
{
  if (debug)
    prnt("%s() buffer \"%s\"\n", __func__, (buff != NULL) ? buff : "NULL");

  rfsimulator_state_t *t = (rfsimulator_state_t *)arg;
  const double sample_rate = t->sample_rate;
  const double c = 299792458; /* 3e8 */

  for (int i=0; i<FD_SETSIZE; i++) {
    buffer_t *b=&t->buf[i];
    if (b->fd_pub_sock <= 0 || b->channel_model == NULL || b->channel_model->model_name == NULL)
      continue;

    channel_desc_t *cd = b->channel_model;
    const int offset = cd->channel_offset;
    const double distance = (double) offset * c / sample_rate;
    prnt("\%s offset %d distance %.3f m\n", cd->model_name, offset, distance);
  }

  return CMDSTATUS_FOUND;
}

static int rfsimu_vtime_cmd(char *buff, int debug, telnet_printfunc_t prnt, void *arg)
{
  rfsimulator_state_t *t = (rfsimulator_state_t *)arg;
  const openair0_timestamp ts = t->nextRxTstamp;
  const double sample_rate = t->sample_rate;
  prnt("vtime measurement: TS %llu sample_rate %.3f\n", ts, sample_rate);
  return CMDSTATUS_FOUND;
}

static int startServer(openair0_device *device) {
  rfsimulator_state_t *t = (rfsimulator_state_t *) device->priv;
  t->typeStamp=ENB_MAGICDL;
  t->context = zmq_ctx_new();
  AssertFatal(t->context != NULL, "Failed to create ZeroMQ context");
  int io_threads = 4;
  zmq_ctx_set(t->context, ZMQ_IO_THREADS, io_threads);
  // Create the publisher socket
  t->pub_sock = zmq_socket(t->context, ZMQ_PUB);
  AssertFatal(t->pub_sock != NULL, "Failed to create publisher socket");

  // Set up monitoring for publisher socket to detect connection to the broker
  int rc = zmq_socket_monitor(t->pub_sock, "inproc://monitor.pub", ZMQ_EVENT_ALL);
  AssertFatal(rc == 0, "Failed to set up socket monitoring for publisher");

  void *pub_monitor = zmq_socket(t->context, ZMQ_PAIR);
  AssertFatal(pub_monitor != NULL, "Failed to create publisher monitor socket");
  rc = zmq_connect(pub_monitor, "inproc://monitor.pub");
  AssertFatal(rc == 0, "Failed to connect publisher monitor socket");

  // Create the subscriber socket
  t->sub_sock = zmq_socket(t->context, ZMQ_SUB);
  AssertFatal(t->sub_sock != NULL, "Failed to create subscriber socket");

  // Set up monitoring for subscriber socket to detect connection to the broker
  rc = zmq_socket_monitor(t->sub_sock, "inproc://monitor.sub", ZMQ_EVENT_ALL);
  AssertFatal(rc == 0, "Failed to set up socket monitoring for subscriber");

  void *sub_monitor = zmq_socket(t->context, ZMQ_PAIR);
  AssertFatal(sub_monitor != NULL, "Failed to create subscriber monitor socket");
  rc = zmq_connect(sub_monitor, "inproc://monitor.sub");
  AssertFatal(rc == 0, "Failed to connect subscriber monitor socket");

  // Connect the sockets to the broker
  char pub_endpoint[256];
  snprintf(pub_endpoint, sizeof(pub_endpoint), "tcp://%s:%d", t->brokerip, t->xsubport);
  rc = zmq_connect(t->pub_sock, pub_endpoint);
  AssertFatal(rc == 0, "Failed to connect publisher socket");

  char sub_endpoint[256];
  snprintf(sub_endpoint, sizeof(sub_endpoint), "tcp://%s:%d", t->brokerip, t->xpubport);
  rc = zmq_connect(t->sub_sock, sub_endpoint);
  AssertFatal(rc == 0, "Failed to connect subscriber socket");
  const char *topic = "ue"; // receive data from nearby UEs
  rc = zmq_setsockopt(t->sub_sock, ZMQ_SUBSCRIBE, topic, strlen(topic));
  AssertFatal(rc == 0, "Failed to subscribe to topic");
  const char *topic2 = "join"; // detect UE connections
  rc = zmq_setsockopt(t->sub_sock, ZMQ_SUBSCRIBE, topic2, strlen(topic2));
  AssertFatal(rc == 0, "Failed to subscribe to topic");
  size_t fd_size = sizeof(t->fd_pub_sock);
  rc = zmq_getsockopt(t->pub_sock, ZMQ_FD, &t->fd_pub_sock, &fd_size);
  AssertFatal(rc == 0, "Cannot get fd for pub_sock");
  rc = zmq_getsockopt(t->sub_sock, ZMQ_FD, &t->fd_sub_sock, &fd_size);
  AssertFatal(rc == 0, "Cannot get fd for sub_sock");

  // Monitor the sockets to detect when they are connected
  bool pub_connected = false;
  bool sub_connected = false;
  struct timeval start, now;
  gettimeofday(&start, NULL);

  while (!pub_connected || !sub_connected) {
    // Process events for publisher
    printf("trying to connect to the broker\n");
    zmq_msg_t event_msg;
    zmq_msg_init(&event_msg);
    rc = zmq_msg_recv(&event_msg, pub_monitor, ZMQ_DONTWAIT);
    if (rc != -1) {
      uint16_t event = *(uint16_t *)zmq_msg_data(&event_msg);
      zmq_msg_close(&event_msg);

      if (event == ZMQ_EVENT_CONNECTED) {
        LOG_D(HW, "Publisher socket connected \n");
        pub_connected = true;
      }
    } else {
      zmq_msg_close(&event_msg);
    }

    // Process events for subscriber
    zmq_msg_init(&event_msg);
    rc = zmq_msg_recv(&event_msg, sub_monitor, ZMQ_DONTWAIT);
    if (rc != -1) {
      uint16_t event = *(uint16_t *)zmq_msg_data(&event_msg);
      zmq_msg_close(&event_msg);

      if (event == ZMQ_EVENT_CONNECTED) {
        LOG_D(HW, "Subscriber socket connected \n");
        sub_connected = true;
      }
    } else {
      zmq_msg_close(&event_msg);
    }
    // Avoid infinite loop
    gettimeofday(&now, NULL);
    double elapsed = (now.tv_sec - start.tv_sec) + (now.tv_usec - start.tv_usec) / 1000000.0;
    if (elapsed > 7.0) {
      LOG_W(HW, "Waited more than 7 seconds for connection to the broker, exiting loop\n");
      break;
    }
    usleep(10000);
  }

  LOG_I(HW, "Connection to the broker established\n");
  zmq_close(pub_monitor);
  zmq_close(sub_monitor);

  return 0;
}

static int startClient(openair0_device *device) {
  rfsimulator_state_t *t = device->priv;
  t->typeStamp=UE_MAGICDL;
  t->context = zmq_ctx_new();
  // Create the publisher socket
  t->pub_sock = zmq_socket(t->context, ZMQ_PUB);
  AssertFatal(t->pub_sock != NULL, "Failed to create publisher socket");
  // Set up monitoring for publisher socket to detect connection to the broker
  int rc = zmq_socket_monitor(t->pub_sock, "inproc://monitor.pub", ZMQ_EVENT_ALL);
  AssertFatal(rc == 0, "Failed to set up socket monitoring for publisher");
  bool connected=false;
  void *pub_monitor = zmq_socket(t->context, ZMQ_PAIR);
  AssertFatal(pub_monitor != NULL, "Failed to create publisher monitor socket");
  rc = zmq_connect(pub_monitor, "inproc://monitor.pub");
  AssertFatal(rc == 0, "Failed to connect publisher monitor socket");
  // Create the subscriber socket
  t->sub_sock = zmq_socket(t->context, ZMQ_SUB);
  AssertFatal(t->sub_sock != NULL, "Failed to create subscriber socket");

  // Set up monitoring for subscriber socket to detect connection to the broker
  rc = zmq_socket_monitor(t->sub_sock, "inproc://monitor.sub", ZMQ_EVENT_ALL);
  AssertFatal(rc == 0, "Failed to set up socket monitoring for subscriber");

  void *sub_monitor = zmq_socket(t->context, ZMQ_PAIR);
  AssertFatal(sub_monitor != NULL, "Failed to create subscriber monitor socket");
  rc = zmq_connect(sub_monitor, "inproc://monitor.sub");
  AssertFatal(rc == 0, "Failed to connect subscriber monitor socket");
  // Connect the sockets to the broker
  char pub_endpoint[256];
  snprintf(pub_endpoint, sizeof(pub_endpoint), "tcp://%s:%d", t->brokerip, t->xsubport);
  rc = zmq_connect(t->pub_sock, pub_endpoint);
  AssertFatal(rc == 0, "Failed to connect publisher socket");
  printf("connecting to endpoint: %s\n", pub_endpoint);

  char sub_endpoint[256];
  snprintf(sub_endpoint, sizeof(sub_endpoint), "tcp://%s:%d", t->brokerip, t->xpubport);
  rc = zmq_connect(t->sub_sock, sub_endpoint);
  AssertFatal(rc == 0, "Failed to connect subscriber socket");
  const char *topic = "first";
  rc = zmq_setsockopt(t->sub_sock, ZMQ_SUBSCRIBE, topic, strlen(topic));
  AssertFatal(rc == 0, "Failed to subscribe to topic");
  size_t fd_size = sizeof(t->fd_pub_sock);
  rc = zmq_getsockopt(t->pub_sock, ZMQ_FD, &t->fd_pub_sock, &fd_size);
  AssertFatal(rc == 0, "Cannot get fd for pub_sock");
  rc = zmq_getsockopt(t->sub_sock, ZMQ_FD, &t->fd_sub_sock, &fd_size);
  AssertFatal(rc == 0, "Cannot get fd for sub_sock");

  // Monitor the sockets to detect when they are connected
  bool pub_connected = false;
  bool sub_connected = false;
  while (!pub_connected || !sub_connected) {
    // Process events for publisher
    printf("trying to connect to the broker\n");
    zmq_msg_t event_msg;
    zmq_msg_init(&event_msg);
    rc = zmq_msg_recv(&event_msg, pub_monitor, ZMQ_DONTWAIT);
    if (rc != -1) {
      uint16_t event = *(uint16_t *)zmq_msg_data(&event_msg);
      zmq_msg_close(&event_msg);

      if (event == ZMQ_EVENT_CONNECTED) {
        LOG_D(HW, "Publisher socket connected\n");
        pub_connected = true;
      }
    } else {
      zmq_msg_close(&event_msg);
    }

    // Process events for subscriber
    zmq_msg_init(&event_msg);
    rc = zmq_msg_recv(&event_msg, sub_monitor, ZMQ_DONTWAIT);
    if (rc != -1) {
      uint16_t event = *(uint16_t *)zmq_msg_data(&event_msg);
      zmq_msg_close(&event_msg);

      if (event == ZMQ_EVENT_CONNECTED) {
        LOG_D(HW, "Subscriber socket connected\n");
        sub_connected = true;
      }
    } else {
      zmq_msg_close(&event_msg);
    }

    usleep(10000);
  }

  LOG_I(HW, "Connection to the broker established\n");
  zmq_close(pub_monitor);
  zmq_close(sub_monitor);
  // Subscribe to the sync topic
  const char *topic1 = "sync"; // recieve data from synchref
  rc = zmq_setsockopt(t->sub_sock, ZMQ_SUBSCRIBE, topic1, strlen(topic1));
  AssertFatal(rc == 0, "Failed to subscribe to topic");
  char jointopic[] = "join"; // send a join message
  zmq_send(t->pub_sock, jointopic, strlen(jointopic), ZMQ_SNDMORE);
  zmq_send(t->pub_sock, t->device_id, strlen(t->device_id), 0);
  allocCirBuf(t, atoi(t->device_id));
  return 0;
}

static int rfsimulator_write_internal(rfsimulator_state_t *t,
                                      openair0_timestamp timestamp,
                                      void **samplesVoid,
                                      int nsamps,
                                      int nbAnt,
                                      int flags,
                                      bool alreadyLocked,
                                      int firstMessage) {
  if (!alreadyLocked)
    pthread_mutex_lock(&Sockmutex);

  LOG_D(HW,"sending %d samples at time: %ld, nbAnt %d\n", nsamps, timestamp, nbAnt);
  // all connected UEs need to have a buffer to broadcast the data
  int count = 0;
  for (int i = 0; i < FD_SETSIZE; i++) {
    buffer_t *b = &t->buf[i];
    if (b->fd_pub_sock >= 0)
      count++;
      LOG_D(HW,"there are %d connect UE",count);
  }
  if (((count != 0))) { // changed here
    if (t->fd_pub_sock >= 0) {
      samplesBlockHeader_t header= {t->typeStamp, nsamps, nbAnt, timestamp};
      fullwrite(t->pub_sock,&header, sizeof(header), t,0);
      sample_t tmpSamples[nsamps][nbAnt];

      for(int a=0; a<nbAnt; a++) {
        sample_t *in=(sample_t *)samplesVoid[a];

        for(int s=0; s<nsamps; s++)
          tmpSamples[s][a]=in[s];
      }
      if (t->fd_pub_sock >= 0 ) {
        fullwrite(t->pub_sock, (void *)tmpSamples, sampleToByte(nsamps,nbAnt), t,0);
      }
    }
  }

  if ( t->lastWroteTS != 0 && fabs((double)t->lastWroteTS-timestamp) > (double)CirSize)
    LOG_E(HW,"Discontinuous TX gap too large Tx:%lu, %lu\n", t->lastWroteTS, timestamp);

  if (t->lastWroteTS > timestamp+nsamps)
    LOG_E(HW,"Not supported to send Tx out of order (same in USRP) %lu, %lu\n",
          t->lastWroteTS, timestamp);

  t->lastWroteTS=timestamp+nsamps;

  if (!alreadyLocked)
    pthread_mutex_unlock(&Sockmutex);

  LOG_D(HW,"sent %d samples at time: %ld->%ld, energy in first antenna: %d\n",
      nsamps, timestamp, timestamp+nsamps, signal_energy(samplesVoid[0], nsamps) );
  return nsamps;
}

static int rfsimulator_write(openair0_device *device, openair0_timestamp timestamp, void **samplesVoid, int nsamps, int nbAnt, int flags) {
  return rfsimulator_write_internal(device->priv, timestamp, samplesVoid, nsamps, nbAnt, flags, false,0);
}

static bool flushInput(rfsimulator_state_t *t, int timeout, int nsamps_for_initial) {
  // Process all incoming events on sockets
  // store the data in lists
  zmq_pollitem_t items[] = {
      {t->sub_sock, 0, ZMQ_POLLIN, 0} // maybe this should be moved to another function
  };
  int rc = zmq_poll(items, 1, timeout);
  if (rc < 0) {
    if (errno == EINTR) {
      return false;
    }
    LOG_W(HW, "zmq_poll() failed, errno(%d)\n", errno);
    return false;
  }
  if (rc == 0) {
    return false;
  }

  if (items[0].revents & ZMQ_POLLIN) {
    // receiving topic
    char topic[256];
    int cap = sizeof(topic);
    // int tsize= zmq_recv(t->sub_sock, topic,cap-1 , ZMQ_DONTWAIT);
    int tsize = zmq_recv(t->sub_sock, topic, cap - 1, 0);
    if (tsize < 0) {
      if (errno != EAGAIN) {
        LOG_E(HW, "zmq_recv() failed, errno(%d)\n", errno);
        AssertFatal(false, "Failed in reading the topic\n");
        // abort();
      }
    }
    topic[tsize < cap ? tsize : cap - 1] = '\0';
    LOG_D(HW, "received topic %s\n", topic);
    // handle first message
    // if (strncasecmp(topic, "first", 2) == 0 && t->nextRxTstamp == 0) {
    //   rc = zmq_setsockopt(t->sub_sock, ZMQ_SUBSCRIBE, topic1, strlen(topic1));
    //   AssertFatal(rc == 0, "Failed to subscribe to topic");
    //   // Subscribe to the ue topic
    //   // const char *topic2 = "ue";
    //   // rc = zmq_setsockopt(t->sub_sock, ZMQ_SUBSCRIBE, topic2, strlen(topic2));
    //   // AssertFatal(rc == 0, "Failed to subscribe to topic");
    // nb_ue++;
    // }
    if ((strncasecmp(topic, "join", 2) == 0)) {
      char deviceid[256];
      int cap = sizeof(deviceid);
      // int idsize= zmq_recv(t->sub_sock, deviceid,cap-1 , ZMQ_DONTWAIT);
      int idsize = zmq_recv(t->sub_sock, deviceid, cap - 1, 0);
      if (idsize < 0) {
        if (errno != EAGAIN) {
          LOG_E(HW, "zmq_recv() failed, errno(%d)\n", errno);
          AssertFatal(false, "Failed in reading the device id\n");
        }
      }
      deviceid[idsize < cap ? idsize : cap - 1] = '\0';
      LOG_D(HW, "received device_id %s\n", deviceid);
      int device_id = atoi(deviceid);
      allocCirBuf(t, device_id);  
      LOG_I(HW,"A client connected, sending the current time\n");
      c16_t v= {0};
      // nb_ue++;
      void *samplesVoid[t->tx_num_channels];

      for ( int i=0; i < t->tx_num_channels; i++)
        samplesVoid[i]=(void *)&v;

      rfsimulator_write_internal(t, t->lastWroteTS > 1 ? t->lastWroteTS-1 : 0,
                                samplesVoid, 1,
                                t->tx_num_channels, 1, false,0);
      return rc > 0;
      }
      buffer_t *b = NULL;
      // Receiving formatted topic = topic + device_id
      if ((t->typeStamp == ENB_MAGICDL) && (strncasecmp(topic, "ue", 2) == 0)) {// recv by server (sync ref)
        char deviceid[256];
        sscanf(topic, "ue %255s", deviceid);
        int id = atoi(deviceid);
        LOG_D(HW,"received data from device:%d\n",id);
        b = &t->buf[id];
        } 
      // if ((t->typeStamp == UE_MAGICDL) && (strncasecmp(topic, "ue", 2) == 0)){
      //   char deviceid[256];
      //   sscanf(topic, "ue %255s", deviceid);
      //   int id = atoi(deviceid);
      //   if (id != atoi(t->device_id)){
      //     b = &t->buf[id];
      //     LOG_D(HW,"recieved data from another UE\n");
      //   } else {
      //     // Empty recv socket, we don't need our message
      //     zmq_msg_t msg;
      //     int rc = zmq_msg_init (&msg);
      //     zmq_msg_recv(&msg,t->sub_sock, ZMQ_DONTWAIT);
      //     LOG_D(HW,"recieved from myself\n");
      //     zmq_msg_close (&msg);
      //     return false;
      //     }
      // }
      if ((strncasecmp(topic, "sync", 2) == 0)) {
        b = &t->buf[atoi(t->device_id)];// buffer of itself means the comm buf with the syncref
      }
      if (!b)
        return false;
      if ( b->circularBuf == NULL ) {
        LOG_E(HW, "received data on not connected socket\n");
        return rc > 0;
      }

      ssize_t blockSz;

      if ( b->headerMode)
        blockSz=b->remainToTransfer;
      else
        blockSz= b->transferPtr + b->remainToTransfer <= b->circularBufEnd ?
                 b->remainToTransfer :
                 b->circularBufEnd - b->transferPtr ;

      // receiving data ( iq samples ) or header
      ssize_t sz = zmq_recv(t->sub_sock, b->transferPtr, blockSz, ZMQ_DONTWAIT);
      if ( sz < 0 ) {
        if ( errno != EAGAIN ) {
          LOG_E(HW, "zmq_recv() failed, errno(%d)\n", errno);
          //abort();
        }
      } else if ( sz == 0 )
        return rc > 0;
      if (sz > 0) {
        LOG_D(HW, "Received on topic %s, %zd bytes\n", topic, sz);
        AssertFatal((b->remainToTransfer-=sz) >= 0, "");
        b->transferPtr+=sz;

        if (b->transferPtr==b->circularBufEnd )
          b->transferPtr=(char *)b->circularBuf;

        // check the header and start block transfer
        if ( b->headerMode==true && b->remainToTransfer==0) {
          AssertFatal( (t->typeStamp == UE_MAGICDL  && b->th.magic==ENB_MAGICDL) ||
                      (t->typeStamp == ENB_MAGICDL && b->th.magic==UE_MAGICDL), "Socket Error in protocol");
          b->headerMode=false;

        if ( t->nextRxTstamp == 0 ) { // First block in UE, resync with the eNB current TS
          t->nextRxTstamp=b->th.timestamp> nsamps_for_initial ?
                           b->th.timestamp -  nsamps_for_initial :
                           0;
          b->lastReceivedTS=b->th.timestamp> nsamps_for_initial ?
                            b->th.timestamp :
                            nsamps_for_initial;
          LOG_W(HW,"UE got first timestamp: starting at %lu\n",  t->nextRxTstamp);
          b->trashingPacket=true;
        } else if ( b->lastReceivedTS < b->th.timestamp) {
          int nbAnt= b->th.nbAnt;

          if ( b->th.timestamp-b->lastReceivedTS < CirSize ) {
            for (uint64_t index=b->lastReceivedTS; index < b->th.timestamp; index++ ) {
              for (int a=0; a < nbAnt; a++) {
                b->circularBuf[(index*nbAnt+a)%CirSize].r = 0;
                b->circularBuf[(index*nbAnt+a)%CirSize].i = 0;
              }
            }
          } else {
            memset(b->circularBuf, 0, sampleToByte(CirSize,1));
          }

          if (b->lastReceivedTS != 0 && b->th.timestamp-b->lastReceivedTS < 1000)
            LOG_W(HW,"gap of: %ld in reception\n", b->th.timestamp-b->lastReceivedTS );

          b->lastReceivedTS=b->th.timestamp;
        } else if ( b->lastReceivedTS > b->th.timestamp && b->th.size == 1 ) {
          LOG_W(HW,"Received Rx/Tx synchro out of order\n");
          b->trashingPacket=true;
        } else if ( b->lastReceivedTS == b->th.timestamp ) {
          // normal case
        } else {
          LOG_E(HW, "received data in past: current is %lu, new reception: %lu!\n", b->lastReceivedTS, b->th.timestamp);
          b->trashingPacket=true;
        }

        pthread_mutex_lock(&Sockmutex);

        if (t->lastWroteTS != 0 && (fabs((double)t->lastWroteTS-b->lastReceivedTS) > (double)CirSize))
          LOG_W(HW, "UEsock(sub_sock) Tx/Rx shift too large Tx:%lu, Rx:%lu\n", t->lastWroteTS, b->lastReceivedTS);

        pthread_mutex_unlock(&Sockmutex);
        b->transferPtr=(char *)&b->circularBuf[(b->lastReceivedTS*b->th.nbAnt)%CirSize];
        b->remainToTransfer=sampleToByte(b->th.size, b->th.nbAnt);
      }

      if ( b->headerMode==false ) {
        if ( ! b->trashingPacket ) {
          b->lastReceivedTS=b->th.timestamp+b->th.size-byteToSample(b->remainToTransfer,b->th.nbAnt);
          LOG_D(HW, "UEsock: sub_sock Set b->lastReceivedTS %ld\n", b->lastReceivedTS);
        }

        if ( b->remainToTransfer==0) {
          LOG_D(HW, "UEsock: sub_sock Completed block reception: %ld\n", b->lastReceivedTS);
          b->headerMode=true;
          b->transferPtr=(char *)&b->th;
          b->remainToTransfer=sizeof(samplesBlockHeader_t);
          b->th.magic=-1;
          b->trashingPacket=false;
        }
      }
      }
    }
  

  return rc>0;
}

static int rfsimulator_read(openair0_device *device, openair0_timestamp *ptimestamp, void **samplesVoid, int nsamps, int nbAnt) {
  if (nbAnt > 4) {
    LOG_W(HW, "rfsimulator: only 4 antenna tested\n");
  }

  rfsimulator_state_t *t = device->priv;
  LOG_D(HW, "Enter rfsimulator_read, expect %d samples, will release at TS: %ld, nbAnt %d\n", nsamps, t->nextRxTstamp+nsamps, nbAnt);
  // deliver data from received data
  // check if a UE is connected
  int first_sock;

  for (first_sock=0; first_sock<FD_SETSIZE; first_sock++)
    if (t->buf[first_sock].circularBuf != NULL )
      break;

  if ( first_sock ==  FD_SETSIZE ) {
    // no connected device (we are eNB, no UE is connected)
    if ( t->nextRxTstamp == 0)
      LOG_W(HW,"No connected device, generating void samples...\n");

    if (!flushInput(t, t->wait_timeout,  nsamps)) {
      for (int x=0; x < nbAnt; x++)
        memset(samplesVoid[x],0,sampleToByte(nsamps,1));

      t->nextRxTstamp+=nsamps;

      if ( ((t->nextRxTstamp/nsamps)%100) == 0)
        LOG_D(HW,"No UE, Generated void samples for Rx: %ld\n", t->nextRxTstamp);

      *ptimestamp = t->nextRxTstamp-nsamps;
      return nsamps;
    }
  } else {

    bool have_to_wait;

    do {
      have_to_wait=false;
      buffer_t *b = NULL;
      for ( int sock=0; sock<FD_SETSIZE; sock++) {
        b=&t->buf[sock];

        if ( b->circularBuf )
          if ( t->nextRxTstamp+nsamps > b->lastReceivedTS ) {
            have_to_wait=true;
            break;
          }
      }

      if (have_to_wait){

        // LOG_D(HW,"Waiting, current last ts: %ld, expected at least : %ld\n",b->lastReceivedTS,t->nextRxTstamp + nsamps);
       flushInput(t, 3, nsamps);
      }
    } while (have_to_wait);
  }

  // Clear the output buffer
  for (int a=0; a<nbAnt; a++)
    memset(samplesVoid[a],0,sampleToByte(nsamps,1));

  // Add all input nodes signal in the output buffer
  for (int sock=0; sock<FD_SETSIZE; sock++) {
    buffer_t *ptr=&t->buf[sock];

    if ( ptr->circularBuf ) {
      bool reGenerateChannel=false;

      //fixme: when do we regenerate
      // it seems legacy behavior is: never in UL, each frame in DL
      if (reGenerateChannel)
        random_channel(ptr->channel_model,0);

      if (t->poll_telnetcmdq)
        t->poll_telnetcmdq(t->telnetcmd_qid,t);

      for (int a=0; a<nbAnt; a++) {//loop over number of Rx antennas
        if ( ptr->channel_model != NULL ) { // apply a channel model
          rxAddInput(ptr->circularBuf, (c16_t *) samplesVoid[a],
                     a,
                     ptr->channel_model,
                     nsamps,
                     t->nextRxTstamp,
                     CirSize);
        }
        else { // no channel modeling
          
          double H_awgn_mimo[4][4] ={{1.0, 0.2, 0.1, 0.05}, //rx 0
                                      {0.2, 1.0, 0.2, 0.1}, //rx 1
                                     {0.1, 0.2, 1.0, 0.2}, //rx 2
                                     {0.05, 0.1, 0.2, 1.0}};//rx 3

          sample_t *out=(sample_t *)samplesVoid[a];
          int nbAnt_tx = ptr->th.nbAnt;//number of Tx antennas

          //LOG_I(HW, "nbAnt_tx %d\n",nbAnt_tx);
          for (int i=0; i < nsamps; i++) {//loop over nsamps
            for (int a_tx=0; a_tx<nbAnt_tx; a_tx++) { //sum up signals from nbAnt_tx antennas
              out[i].r += (short)(ptr->circularBuf[((t->nextRxTstamp+i)*nbAnt_tx+a_tx)%CirSize].r*H_awgn_mimo[a][a_tx]);
              out[i].i += (short)(ptr->circularBuf[((t->nextRxTstamp+i)*nbAnt_tx+a_tx)%CirSize].i*H_awgn_mimo[a][a_tx]);
            } // end for a_tx
          } // end for i (number of samps)
        } // end of no channel modeling
      } // end for a (number of rx antennas)
    }
  }

  *ptimestamp = t->nextRxTstamp; // return the time of the first sample
  t->nextRxTstamp+=nsamps;
  LOG_D(HW,"Rx to upper layer: %d from %ld to %ld, energy in first antenna %d\n",
        nsamps,
        *ptimestamp, t->nextRxTstamp,
        signal_energy(samplesVoid[0], nsamps));
  return nsamps;
}

static int rfsimulator_get_stats(openair0_device *device) {
  return 0;
}
static int rfsimulator_reset_stats(openair0_device *device) {
  return 0;
}
static void rfsimulator_end(openair0_device *device) {
  rfsimulator_state_t* s = device->priv;
  zmq_close(s->sub_sock);
  zmq_close(s->pub_sock);
  zmq_ctx_destroy(s->context);
  for (int i = 0; i < FD_SETSIZE; i++) {
    buffer_t *b = &s->buf[i];
    if (b->fd_pub_sock >= 0 )
      removeCirBuf(s, b->conn_device_id);
  }
  s->fd_pub_sock = -1;
  s->fd_sub_sock = -1;
}
static int rfsimulator_stop(openair0_device *device) {
  return 0;
}
static int rfsimulator_set_freq(openair0_device *device, openair0_config_t *openair0_cfg) {
  return 0;
}
static int rfsimulator_set_gains(openair0_device *device, openair0_config_t *openair0_cfg) {
  return 0;
}
static int rfsimulator_write_init(openair0_device *device) {
  return 0;
}
__attribute__((__visibility__("default")))
int device_init(openair0_device *device, openair0_config_t *openair0_cfg) {
  // to change the log level, use this on command line
  // --log_config.hw_log_level debug
  rfsimulator_state_t *rfsimulator = (rfsimulator_state_t *)calloc(sizeof(rfsimulator_state_t),1);
  // initialize channel simulation
  rfsimulator->tx_num_channels=openair0_cfg->tx_num_channels;
  rfsimulator->rx_num_channels=openair0_cfg->rx_num_channels;
  rfsimulator->sample_rate=openair0_cfg->sample_rate;
  rfsimulator->tx_bw=openair0_cfg->tx_bw;  
  rfsimulator_readconfig(rfsimulator);
  LOG_W(HW, "rfsim: sample_rate %f\n", rfsimulator->sample_rate);
  pthread_mutex_init(&Sockmutex, NULL);
  LOG_I(HW,"rfsimulator: running as %s\n", rfsimulator-> typeStamp == ENB_MAGICDL ? "server waiting opposite rfsimulators to connect" : "client: will connect to a rfsimulator server side");
  device->trx_start_func       = rfsimulator->typeStamp == ENB_MAGICDL ?
                                 startServer :
                                 startClient;
  device->trx_get_stats_func   = rfsimulator_get_stats;
  device->trx_reset_stats_func = rfsimulator_reset_stats;
  device->trx_end_func         = rfsimulator_end;
  device->trx_stop_func        = rfsimulator_stop;
  device->trx_set_freq_func    = rfsimulator_set_freq;
  device->trx_set_gains_func   = rfsimulator_set_gains;
  device->trx_write_func       = rfsimulator_write;
  device->trx_read_func      = rfsimulator_read;
  /* let's pretend to be a b2x0 */
  device->type = RFSIMULATOR;
  openair0_cfg[0].rx_gain[0] = 0;
  device->openair0_cfg=&openair0_cfg[0];
  device->priv = rfsimulator;
  device->trx_write_init = rfsimulator_write_init;

  for (int i = 0; i < FD_SETSIZE; i++) {
    rfsimulator->buf[i].fd_sub_sock = -1;
    rfsimulator->buf[i].fd_pub_sock = -1;
  }
  rfsimulator->fd_pub_sock = -1;
  rfsimulator->fd_sub_sock = -1;
  
  // we need to call randominit() for telnet server (use gaussdouble=>uniformrand)
  randominit(0);
  set_taus_seed(0);
  /* look for telnet server, if it is loaded, add the channel modeling commands to it */
  add_telnetcmd_func_t addcmd = (add_telnetcmd_func_t)get_shlibmodule_fptr("telnetsrv", TELNET_ADDCMD_FNAME);

  if (addcmd != NULL) {
    rfsimulator->poll_telnetcmdq =  (poll_telnetcmdq_func_t)get_shlibmodule_fptr("telnetsrv", TELNET_POLLCMDQ_FNAME);
    addcmd("rfsimu",rfsimu_vardef,rfsimu_cmdarray);

    for(int i=0; rfsimu_cmdarray[i].cmdfunc != NULL; i++) {
      if (  rfsimu_cmdarray[i].qptr != NULL) {
        rfsimulator->telnetcmd_qid = rfsimu_cmdarray[i].qptr;
        break;
      }
    }
  }

  return 0;
}
