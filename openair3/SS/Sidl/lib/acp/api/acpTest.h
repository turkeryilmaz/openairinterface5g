/*
 * Copyright 2022 Sequans Communications.
 *
 * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The OpenAirInterface Software Alliance licenses this file to You under
 * the OAI Public License, Version 1.0  (the "License"); you may not use this file
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
 * For more information about the OpenAirInterface (OAI) Software Alliance:
 *      contact@openairinterface.org
 */

#pragma once

#include "SIDL_Test.h"
#include "acp.h"

SIDL_BEGIN_C_INTERFACE

void acpTestHelloFromSSInitClt(acpCtx_t _ctx, char** StrArray, size_t StrQty);

int acpTestHelloFromSSEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, size_t StrQty, const char* StrArray);

int acpTestHelloFromSSDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, size_t* StrQty, char** StrArray);

void acpTestHelloFromSSFree0Srv(char* StrArray);

void acpTestHelloFromSSFreeSrv(char* StrArray);

void acpTestHelloFromSSFree0CltSrv(char* StrArray);

void acpTestHelloFromSSFreeCltSrv(char* StrArray);

void acpTestHelloToSSInitSrv(acpCtx_t _ctx, char** StrArray, size_t StrQty);

int acpTestHelloToSSEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, size_t StrQty, const char* StrArray);

int acpTestHelloToSSDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, size_t* StrQty, char** StrArray);

void acpTestHelloToSSFree0Clt(char* StrArray);

void acpTestHelloToSSFreeClt(char* StrArray);

void acpTestHelloToSSFree0SrvClt(char* StrArray);

void acpTestHelloToSSFreeSrvClt(char* StrArray);

int acpTestPingEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, uint32_t FromSS);

int acpTestPingDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, uint32_t* FromSS);

int acpTestPingEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, uint32_t ToSS);

int acpTestPingDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, uint32_t* ToSS);

void acpTestEchoInitClt(acpCtx_t _ctx, struct EchoData** FromSS);

int acpTestEchoEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct EchoData* FromSS);

int acpTestEchoDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct EchoData** FromSS);

void acpTestEchoFree0Srv(struct EchoData* FromSS);

void acpTestEchoFreeSrv(struct EchoData* FromSS);

void acpTestEchoFree0CltSrv(struct EchoData* FromSS);

void acpTestEchoFreeCltSrv(struct EchoData* FromSS);

void acpTestEchoInitSrv(acpCtx_t _ctx, struct EchoData** ToSS);

int acpTestEchoEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct EchoData* ToSS);

int acpTestEchoDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct EchoData** ToSS);

void acpTestEchoFree0Clt(struct EchoData* ToSS);

void acpTestEchoFreeClt(struct EchoData* ToSS);

void acpTestEchoFree0SrvClt(struct EchoData* ToSS);

void acpTestEchoFreeSrvClt(struct EchoData* ToSS);

void acpTestTest1InitClt(acpCtx_t _ctx, struct Output** out);

int acpTestTest1EncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct Output* out);

int acpTestTest1DecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct Output** out);

void acpTestTest1Free0Srv(struct Output* out);

void acpTestTest1FreeSrv(struct Output* out);

void acpTestTest1Free0CltSrv(struct Output* out);

void acpTestTest1FreeCltSrv(struct Output* out);

void acpTestTest2InitSrv(acpCtx_t _ctx, struct Output** out);

int acpTestTest2EncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct Output* out);

int acpTestTest2DecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct Output** out);

void acpTestTest2Free0Clt(struct Output* out);

void acpTestTest2FreeClt(struct Output* out);

void acpTestTest2Free0SrvClt(struct Output* out);

void acpTestTest2FreeSrvClt(struct Output* out);

void acpTestOtherInitClt(acpCtx_t _ctx, struct Empty** in1, char** in3Array, size_t in3Qty, char** in4, struct Empty** in9Array, size_t in9Qty, struct Empty2** in10, struct New** in11);

int acpTestOtherEncClt(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct Empty* in1, uint32_t in2, size_t in3Qty, const char* in3Array, const char* in4, bool in5, int in6, float in7, SomeEnum in8, size_t in9Qty, const struct Empty* in9Array, const struct Empty2* in10, const struct New* in11);

int acpTestOtherDecSrv(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct Empty** in1, uint32_t* in2, size_t* in3Qty, char** in3Array, char** in4, bool* in5, int* in6, float* in7, SomeEnum* in8, size_t* in9Qty, struct Empty** in9Array, struct Empty2** in10, struct New** in11);

void acpTestOtherFree0Srv(struct Empty* in1, char* in3Array, char* in4, struct Empty* in9Array, size_t in9Qty, struct Empty2* in10, struct New* in11);

void acpTestOtherFreeSrv(struct Empty* in1, char* in3Array, char* in4, struct Empty* in9Array, size_t in9Qty, struct Empty2* in10, struct New* in11);

void acpTestOtherFree0CltSrv(struct Empty* in1, char* in3Array, char* in4, struct Empty* in9Array, size_t in9Qty, struct Empty2* in10, struct New* in11);

void acpTestOtherFreeCltSrv(struct Empty* in1, char* in3Array, char* in4, struct Empty* in9Array, size_t in9Qty, struct Empty2* in10, struct New* in11);

void acpTestOtherInitSrv(acpCtx_t _ctx, struct Empty** out1, char** out3Array, size_t out3Qty, char** out4, struct Empty** out9Array, size_t out9Qty, struct Empty2** out10, struct New** out11);

int acpTestOtherEncSrv(acpCtx_t _ctx, unsigned char* _buffer, size_t* _size, const struct Empty* out1, uint32_t out2, size_t out3Qty, const char* out3Array, const char* out4, bool out5, int out6, float out7, SomeEnum out8, size_t out9Qty, const struct Empty* out9Array, const struct Empty2* out10, const struct New* out11);

int acpTestOtherDecClt(acpCtx_t _ctx, const unsigned char* _buffer, size_t _size, struct Empty** out1, uint32_t* out2, size_t* out3Qty, char** out3Array, char** out4, bool* out5, int* out6, float* out7, SomeEnum* out8, size_t* out9Qty, struct Empty** out9Array, struct Empty2** out10, struct New** out11);

void acpTestOtherFree0Clt(struct Empty* out1, char* out3Array, char* out4, struct Empty* out9Array, size_t out9Qty, struct Empty2* out10, struct New* out11);

void acpTestOtherFreeClt(struct Empty* out1, char* out3Array, char* out4, struct Empty* out9Array, size_t out9Qty, struct Empty2* out10, struct New* out11);

void acpTestOtherFree0SrvClt(struct Empty* out1, char* out3Array, char* out4, struct Empty* out9Array, size_t out9Qty, struct Empty2* out10, struct New* out11);

void acpTestOtherFreeSrvClt(struct Empty* out1, char* out3Array, char* out4, struct Empty* out9Array, size_t out9Qty, struct Empty2* out10, struct New* out11);

SIDL_END_C_INTERFACE
