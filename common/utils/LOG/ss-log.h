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

/*! \file ss-log.h
* \brief System Simulator Log interface
*/

#ifndef __SS_LOG_H__
#define __SS_LOG_H__
#include "ss-tp.h"

#define LOG_SYS(evId, ...)	do { if (1) { char buf[1024]; \
						sprintf(buf, __VA_ARGS__); \
						tracepoint(SSeNB, SS_SYS, "SS-eNB-SYS", \
						evId, g_log->sfn, g_log->sf, \
						__func__, __LINE__, buf); \
				} } while (0)
#define LOG_SRB(evId, ...)	do { if (1) { char buf[1024]; \
						sprintf(buf, __VA_ARGS__); \
						tracepoint(SSeNB, SS_SRB, "SS-eNB-SRB", \
						evId, g_log->sfn, g_log->sf, \
						__func__, __LINE__, buf); \
				} } while (0)
#define LOG_DRB(evId, ...)	do { if (1) { char buf[1024]; \
						sprintf(buf, __VA_ARGS__); \
						tracepoint(SSeNB, SS_DRB, "SS-eNB-DRB", \
						evId, g_log->sfn, g_log->sf, \
						__func__, __LINE__, buf); \
				} } while (0)
#define LOG_VNG(evId, ...)	do { if (1) { char buf[1024]; \
						sprintf(buf, __VA_ARGS__); \
						tracepoint(SSeNB, SS_VNG, "SS-eNB-VNG", \
						evId, g_log->sfn, g_log->sf, \
						__func__, __LINE__, buf); \
				} } while (0)
#define LOG_SS(component, log)			tracepoint(SSeNB, SS_LOG, \
						component, -1,  g_log->sfn, \
						g_log->sf, __func__, __LINE__,\
					       	log)
#define LOG_P(component, _string, buf, len)     tracepoint(SSeNB, SS_PKT, \
						"SS-PDU", -1,  g_log->sfn, \
						g_log->sf, _string, buf, len)
#endif /** __SS_LOG_H__ */
