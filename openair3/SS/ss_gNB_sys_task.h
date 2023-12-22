#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#ifndef SS_GNB_TASK_H_
#define SS_GNB_TASK_H_

// void  ss_eNB_init(void);
void *ss_gNB_sys_process_itti_msg(void *);
void *ss_gNB_sys_task(void *arg);
enum Proxy_Msg_Id
{
    Cell_Attenuation_Req = 1,
    Cell_Attenuation_Cnf = 2,
    Cell_Config_Req = 3,
    Cell_Config_Cnf = 4,
    Max_Msg_Id = 5
};

typedef struct udpSockReq_s
{
  uint32_t  port;
  char     *address;
} udpSockReq_t;

typedef enum _sys_Type
{
	SYS_TYPE_LTE,
	SYS_TYPE_NR,
	SYS_TYPE_ENDC
}sys_Type;

#endif /* SS_ENB_TASK_H_ */
