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

#ifndef INTERTASK_INTERFACE_H_
#define INTERTASK_INTERFACE_H_
#include <stdint.h>
#include <sys/epoll.h>
#include "string.h"
#include <signal.h>

#include <mem_block.h>
#include <assertions.h>


typedef enum timer_type_s {
  TIMER_PERIODIC,
  TIMER_ONE_SHOT,
  TIMER_TYPE_MAX,
} timer_type_t;

typedef struct {
  void *arg;
  long  timer_id;
} timer_has_expired_t;

typedef struct {
  uint32_t      interval_sec;
  uint32_t      interval_us;
  long          task_id;
  instance_t       instance;
  timer_type_t  type;
  void         *timer_arg;
  long          timer_id;
} timer_create_t;

typedef struct {
  long          task_id;
  long          timer_id;
} timer_delete_t;


typedef struct itti_lte_time_s {
  uint32_t frame;
  uint8_t slot;
} itti_lte_time_t;


typedef struct IttiMsgEmpty_s {
  // This dummy element is to avoid CLANG warning: empty struct has size 0 in C, size 1 in C++
  // To be removed if the structure is filled
  uint32_t dummy;
} IttiMsgEmpty;

typedef struct IttiMsgText_s {
  uint32_t  size;
  char      text[];
} IttiMsgText;

void *rrc_enb_process_itti_msg(void *);

typedef uint32_t MessageHeaderSize;
typedef uint32_t itti_message_types_t;
typedef unsigned long message_number_t;
#define MESSAGE_NUMBER_SIZE (sizeof(unsigned long))

typedef enum task_priorities_e {
  TASK_PRIORITY_MAX       = 100,
  TASK_PRIORITY_MAX_LEAST = 85,
  TASK_PRIORITY_MED_PLUS  = 70,
  TASK_PRIORITY_MED       = 55,
  TASK_PRIORITY_MED_LEAST = 40,
  TASK_PRIORITY_MIN_PLUS  = 25,
  TASK_PRIORITY_MIN       = 10,
} task_priorities_t;

typedef struct {
  task_priorities_t priority;
  unsigned int queue_size;
  /* Printable name */
  char name[256];
  void *(*func)(void *) ;
  void *(*threadFunc)(void *) ;
} task_info_t;
//
//TASK_DEF(TASK_RRC_ENB,  TASK_PRIORITY_MED,  200, NULL,NULL)
//TASK_DEF(TASK_RRC_ENB,  TASK_PRIORITY_MED,  200, NULL, NULL)
//TASK_DEF(TASK_GTPV1_U,  TASK_PRIORITY_MED,  1000,NULL, NULL)
//TASK_DEF(TASK_UDP,      TASK_PRIORITY_MED,  1000, NULL, NULL)
void *rrc_enb_process_msg(void *);
#define FOREACH_TASK(TASK_DEF)                                       \
  TASK_DEF(TASK_UNKNOWN, TASK_PRIORITY_MED, 50, NULL, NULL)          \
  TASK_DEF(TASK_TIMER, TASK_PRIORITY_MED, 10, NULL, NULL)            \
  TASK_DEF(TASK_L2L1, TASK_PRIORITY_MAX, 200, NULL, NULL)            \
  TASK_DEF(TASK_BM, TASK_PRIORITY_MED, 200, NULL, NULL)              \
  TASK_DEF(TASK_PHY_ENB, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_MAC_GNB, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_RLC_ENB, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_RRC_ENB_NB_IoT, TASK_PRIORITY_MED, 200, NULL, NULL)  \
  TASK_DEF(TASK_PDCP_ENB, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_PDCP_GNB, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_DATA_FORWARDING, TASK_PRIORITY_MED, 200, NULL, NULL) \
  TASK_DEF(TASK_END_MARKER, TASK_PRIORITY_MED, 200, NULL, NULL)      \
  TASK_DEF(TASK_RRC_ENB, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_RRC_GNB, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_RAL_ENB, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_S1AP, TASK_PRIORITY_MED, 200, NULL, NULL)            \
  TASK_DEF(TASK_NGAP, TASK_PRIORITY_MED, 200, NULL, NULL)            \
  TASK_DEF(TASK_X2AP, TASK_PRIORITY_MED, 200, NULL, NULL)            \
  TASK_DEF(TASK_M2AP_ENB, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_M2AP_MCE, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_M3AP, TASK_PRIORITY_MED, 200, NULL, NULL)            \
  TASK_DEF(TASK_M3AP_MME, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_M3AP_MCE, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_SCTP, TASK_PRIORITY_MED, 200, NULL, NULL)            \
  TASK_DEF(TASK_ENB_APP, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_GNB_APP, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_MCE_APP, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_MME_APP, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_PHY_UE, TASK_PRIORITY_MED, 200, NULL, NULL)          \
  TASK_DEF(TASK_MAC_UE, TASK_PRIORITY_MED, 200, NULL, NULL)          \
  TASK_DEF(TASK_RLC_UE, TASK_PRIORITY_MED, 200, NULL, NULL)          \
  TASK_DEF(TASK_PDCP_UE, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_RRC_UE, TASK_PRIORITY_MED, 200, NULL, NULL)          \
  TASK_DEF(TASK_RRC_NRUE, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_NAS_UE, TASK_PRIORITY_MED, 200, NULL, NULL)          \
  TASK_DEF(TASK_RAL_UE, TASK_PRIORITY_MED, 200, NULL, NULL)          \
  TASK_DEF(TASK_GTPV1_U, TASK_PRIORITY_MED, 1000, NULL, NULL)        \
  TASK_DEF(TASK_CU_F1, TASK_PRIORITY_MED, 200, NULL, NULL)           \
  TASK_DEF(TASK_DU_F1, TASK_PRIORITY_MED, 200, NULL, NULL)           \
  TASK_DEF(TASK_CUCP_E1, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_CUUP_E1, TASK_PRIORITY_MED, 200, NULL, NULL)         \
  TASK_DEF(TASK_RRC_UE_SIM, TASK_PRIORITY_MED, 200, NULL, NULL)      \
  TASK_DEF(TASK_RRC_GNB_SIM, TASK_PRIORITY_MED, 200, NULL, NULL)     \
  TASK_DEF(TASK_RRC_NSA_UE, TASK_PRIORITY_MED, 200, NULL, NULL)      \
  TASK_DEF(TASK_RRC_NSA_NRUE, TASK_PRIORITY_MED, 200, NULL, NULL)    \
  TASK_DEF(TASK_NAS_NRUE, TASK_PRIORITY_MED, 200, NULL, NULL)        \
  TASK_DEF(TASK_MAX, TASK_PRIORITY_MED, 200, NULL, NULL)

#define TASK_DEF(TaskID, pRIO, qUEUEsIZE, FuNc, ThreadFunc)          { pRIO, qUEUEsIZE, #TaskID, FuNc, ThreadFunc },

/* Map task id to printable name. */
static const task_info_t tasks_info[] = {
  FOREACH_TASK(TASK_DEF)
};

#define TASK_ENUM(TaskID, pRIO, qUEUEsIZE, FuNc,ThreadFunc ) TaskID,
//! Tasks id of each task
typedef enum {
  FOREACH_TASK(TASK_ENUM)
} task_id_t;

typedef task_id_t thread_id_t;

typedef enum message_priorities_e {
  MESSAGE_PRIORITY_MAX       = 100,
  MESSAGE_PRIORITY_MAX_LEAST = 85,
  MESSAGE_PRIORITY_MED_PLUS  = 70,
  MESSAGE_PRIORITY_MED       = 55,
  MESSAGE_PRIORITY_MED_LEAST = 40,
  MESSAGE_PRIORITY_MIN_PLUS  = 25,
  MESSAGE_PRIORITY_MIN       = 10,
} message_priorities_t;

#define FOREACH_MSG(INTERNAL_MSG)                                            \
  INTERNAL_MSG(TIMER_HAS_EXPIRED, MESSAGE_PRIORITY_MED, timer_has_expired_t) \
  INTERNAL_MSG(INITIALIZE_MESSAGE, MESSAGE_PRIORITY_MED, IttiMsgEmpty)       \
  INTERNAL_MSG(ACTIVATE_MESSAGE, MESSAGE_PRIORITY_MED, IttiMsgEmpty)         \
  INTERNAL_MSG(DEACTIVATE_MESSAGE, MESSAGE_PRIORITY_MED, IttiMsgEmpty)       \
  INTERNAL_MSG(TERMINATE_MESSAGE, MESSAGE_PRIORITY_MAX, IttiMsgEmpty)        \
  INTERNAL_MSG(MESSAGE_TEST, MESSAGE_PRIORITY_MED, IttiMsgEmpty)

/* This enum defines messages ids. Each one is unique. */
typedef enum {
#define MESSAGE_DEF(iD, pRIO, sTRUCT) iD,
  FOREACH_MSG(MESSAGE_DEF)
#include <all_msg.h>
#undef MESSAGE_DEF
      MESSAGES_ID_MAX,
} MessagesIds;

typedef struct MessageHeader_s {
  MessagesIds messageId;          /**< Unique message id as referenced in enum MessagesIds */
  task_id_t  originTaskId;        /**< ID of the sender task */
  task_id_t  destinationTaskId;   /**< ID of the destination task */
  instance_t originInstance;
  instance_t destinationInstance;
  itti_lte_time_t lte_time;
  MessageHeaderSize ittiMsgSize;         /**< Message size (not including header size) */
} MessageHeader;

typedef struct message_info_s {
  MessagesIds id;
  message_priorities_t priority;
  /* Printable name */
  const char name[256];
} message_info_t;

typedef struct MessageDef_s {
  MessageHeader ittiMsgHeader; /**< Message header */
  void*         ittiMsg;
} MessageDef;


/** \brief Alloc and memset(0) a new itti message.
   \param origin_task_id Task ID of the sending task
   \param message_id Message ID
   \param size size of the payload to send
   @returns NULL in case of failure or newly allocated mesage ref
 **/
MessageDef *itti_alloc_sized(task_id_t origin_task_id, instance_t originInstance, MessagesIds message_id, MessageHeaderSize size);

#define MESSAGE_DEF(iD, pRIO, sTRUCT)              \
  static inline sTRUCT *iD##_data(MessageDef *msg) \
  {                                                \
    return (sTRUCT *)msg->ittiMsg;                 \
  }
FOREACH_MSG(MESSAGE_DEF)
#undef MESSAGE_DEF
#define MESSAGE_DEF(iD, pRIO, sTRUCT)                                                 \
  static inline MessageDef *iD##_alloc(task_id_t origintaskID, instance_t originINST) \
  {                                                                                   \
    return itti_alloc_sized(origintaskID, originINST, iD, sizeof(sTRUCT));            \
  }
  FOREACH_MSG(MESSAGE_DEF)
#undef MESSAGE_DEF

/* Map message id to message information */
  static const message_info_t messages_info[] = {
#define MESSAGE_DEF(iD, pRIO, sTRUCT) {iD, pRIO, #iD},
      FOREACH_MSG(MESSAGE_DEF)
#include <all_msg.h>
#undef MESSAGE_DEF
  };

/* Extract the instance from a message */
#define ITTI_MSG_ID(mSGpTR)                 ((mSGpTR)->ittiMsgHeader.messageId)
#define ITTI_MSG_ORIGIN_ID(mSGpTR)          ((mSGpTR)->ittiMsgHeader.originTaskId)
#define ITTI_MSG_DESTINATION_ID(mSGpTR)     ((mSGpTR)->ittiMsgHeader.destinationTaskId)
#define ITTI_MSG_ORIGIN_INSTANCE(mSGpTR)           ((mSGpTR)->ittiMsgHeader.originInstance)
#define ITTI_MSG_DESTINATION_INSTANCE(mSGpTR)           ((mSGpTR)->ittiMsgHeader.destinationInstance)
#define ITTI_MSG_NAME(mSGpTR)               itti_get_message_name(ITTI_MSG_ID(mSGpTR))
#define ITTI_MSG_ORIGIN_NAME(mSGpTR)        itti_get_task_name(ITTI_MSG_ORIGIN_ID(mSGpTR))
#define ITTI_MSG_DESTINATION_NAME(mSGpTR)   itti_get_task_name(ITTI_MSG_DESTINATION_ID(mSGpTR))

#define INSTANCE_DEFAULT    (UINT16_MAX - 1)

#ifdef __cplusplus
extern "C" {
#endif

/** \brief Send a message to a task (could be itself)
  \param task_id Task ID
  \param instance Instance of the task used for virtualization
  \param message Pointer to the message to send
  @returns -1 on failure, 0 otherwise
 **/
int itti_send_msg_to_task(task_id_t task_id, instance_t instance, MessageDef *message);

/** \brief Add a new fd to monitor.
   NOTE: it is up to the user to read data associated with the fd
    \param task_id Task ID of the receiving task
    \param fd The file descriptor to monitor
 **/
void itti_subscribe_event_fd(task_id_t task_id, int fd);

/** \brief Remove a fd from the list of fd to monitor
    \param task_id Task ID of the task
    \param fd The file descriptor to remove
 **/
void itti_unsubscribe_event_fd(task_id_t task_id, int fd);

/** \brief Return the list of events excluding the fd associated with itti
    \the fd associated with itti can return, but it is marked events[i].events &= ~EPOLLIN
    \as it is not EPOLLIN, the reader should ignore this fd
    \or it can manage the list of fd's in his interest, so ignore the other ones
    \param task_id Task ID of the task
    \param events events list
    @returns number of events to handle
 **/
  int itti_get_events(task_id_t task_id, struct epoll_event *events, int nb_max_evts);

/** \brief Retrieves a message in the queue associated to task_id.
   If the queue is empty, the thread is blocked till a new message arrives.
  \param task_id Task ID of the receiving task
  \param received_msg Pointer to the allocated message
 **/
void itti_receive_msg(task_id_t task_id, MessageDef **received_msg);

/** \brief Try to retrieves a message in the queue associated to task_id.
  \param task_id Task ID of the receiving task
  \param received_msg Pointer to the allocated message
 **/
void itti_poll_msg(task_id_t task_id, MessageDef **received_msg);

/** \brief Start thread associated to the task
   \param task_id task to start
   \param start_routine entry point for the task
   \param args_p Optional argument to pass to the start routine
   @returns -1 on failure, 0 otherwise
 **/
int itti_create_task(task_id_t task_id,
                     void *(*start_routine) (void *),
                     void *args_p);

int itti_create_queue(const task_info_t *task_info);

/** \brief Exit the current task.
 **/
void itti_exit_task(void);

/** \brief Initiate termination of all tasks.
   \param task_id task that is completed
 **/
void itti_terminate_tasks(task_id_t task_id);

// Void for legacy compatibility
void itti_wait_ready(int wait_tasks);
void itti_mark_task_ready(task_id_t task_id);

/** \brief Return the printable string associated with the message
   \param message_id Id of the message
 **/
const char *itti_get_message_name(MessagesIds message_id);

/** \brief Return the printable string associated with a task id
   \param thread_id Id of the task
 **/
const char *itti_get_task_name(task_id_t task_id);

/** \brief Alloc and memset(0) a new itti message.
   \param origin_task_id Task ID of the sending task
   \param message_id Message ID
   \param size size of the payload to send
   @returns NULL in case of failure or newly allocated mesage ref
 **/
MessageDef *itti_alloc_new_message_sized(
  task_id_t         origin_task_id,
  instance_t originInstance,
  MessagesIds       message_id,
  MessageHeaderSize size);

/** \brief Wait for SIGINT/SIGTERM signals to unblock ITTI.
   This function should be called from the main thread after having created all ITTI tasks. If handler is NULL, a default handler is installed.
    \param handler a custom signal handler. To unblock, it should call itti_wait_tasks_unblock()
 **/
void itti_wait_tasks_end(void (*handler)(int));
/** \brif unblocks ITTI waiting in itti_wait_tasks_end(). **/
void itti_wait_tasks_unblock(void);
void itti_set_task_real_time(task_id_t task_id);

/** \brief Send a termination message to all tasks.
   \param task_id task that is broadcasting the message.
 **/
void itti_send_terminate_message(task_id_t task_id);

void *itti_malloc(task_id_t origin_task_id, task_id_t destination_task_id, ssize_t size);
int memory_read(const char *datafile, void *data, size_t size);
int itti_free(task_id_t task_id, void *ptr);

int itti_init(task_id_t task_max, const task_info_t *tasks_info);
int timer_setup(
  uint32_t      interval_sec,
  uint32_t      interval_us,
  task_id_t     task_id,
  instance_t       instance,
  timer_type_t  type,
  void         *timer_arg,
  long         *timer_id);


int timer_remove(long timer_id);
#define timer_stop timer_remove
int signal_handle(int *end);
int signal_mask(void);

void log_scheduler(const char *label);
#ifdef __cplusplus
}
#endif
#endif /* INTERTASK_INTERFACE_H_ */
