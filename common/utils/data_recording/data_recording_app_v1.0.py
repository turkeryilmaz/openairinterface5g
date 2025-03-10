#/*
# * Licensed to the OpenAirInterface (OAI) Software Alliance under one or more
# * contributor license agreements.  See the NOTICE file distributed with
# * this work for additional information regarding copyright ownership.
# * The OpenAirInterface Software Alliance licenses this file to You under
# * the OAI Public License, Version 1.1  (the "License"); you may not use this file
# * except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.openairinterface.org/?page_id=698
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *-------------------------------------------------------------------------------
# * For more information about the OpenAirInterface (OAI) Software Alliance:
# *      contact@openairinterface.org
# */
#---------------------------------------------------------------------
# file common/utils/data_recording/data_recording_app_v1.0.py
# brief main application of synchronized real-time data recording
# author Abdo Gaber
# date 2024
# version 1.0
# company Emerson, NI Test and Measurement
# email:
# note
# warning

import os
import sysv_ipc as ipc
import struct
import time
from datetime import datetime
import argparse
from termcolor import colored, cprint
import numpy as np
import json
import concurrent.futures
# from concurrent import futures
import threading
# import related functions
from lib import sigmf_interface
from bitarray import bitarray

# globally applicable metadata
global_info = {
    "author": "Abdo Gaber",
    "description": "Synchronized Real-Time Data Recording",
    "timestamp": 0,
    "collection_file_prefix": "data-collection",  # collection file name prefix "deap-rx + str(...)"
    "collection_file": "",  # Reserved to be created in the code: “data-collection_rec-0_TIME-STAMP”
    "datetime_offset": "",  # datetime offset between current location and UTC/Zulu timezone
    # Example: "+01:00" for Berlin, Germany
    "save_config_data_recording_app_json": True,
    "waveform_generator": "5gnr_oai",
    "extensions": {},
}

# Supported OAI Trace messages
# UL receiver messages
# gNB IQ Msgs: "GNB_PHY_UL_FD_PUSCH_IQ", "GNB_PHY_UL_FD_DMRS", "GNB_PHY_UL_FD_CHAN_EST_DMRS_POS"
# gNB BITS Msgs: "GNB_PHY_UL_PAYLOAD_RX_BITS"
# UE BITS Msgs: "UE_PHY_UL_SCRAMBLED_TX_BITS", "UE_PHY_UL_PAYLOAD_TX_BITS"

supported_oai_tracer_messages = {
    # gNB messages
    "GNB_PHY_UL_FD_PUSCH_IQ": {
        "file_name_prefix": "rx-fd-data",
        "scope": "gNB",
        "description": "Frequency-domain RX data",
        "serialization_scheme": ["subcarriers", "ofdm_symbols"],
    },
    "GNB_PHY_UL_FD_DMRS": {
        "file_name_prefix": "tx-pilots-fd-data",
        "scope": "gNB",
        "description": "Frequency-domain TX PUSCH DMRS data",
        "serialization_scheme": ["subcarriers", "ofdm_symbols"],
    },
    "GNB_PHY_UL_FD_CHAN_EST_DMRS_POS": {
        "file_name_prefix": "raw-ce-fd-data",
        "scope": "gNB",
        "description": "Frequency-domain raw channel estimates (at DMRS positions)",
        "serialization_scheme": ["subcarriers", "ofdm_symbols"],
    },
    "GNB_PHY_UL_FD_CHAN_EST_DMRS_INTERPL": {
        "file_name_prefix": "raw-inter-ce-fd-data",
        "scope": "gNB",
        "description": "Interpolcated Frequency-domain raw channel estimates",
        "serialization_scheme": ["subcarriers", "ofdm_symbols"],
    },
    "GNB_PHY_UL_PAYLOAD_RX_BITS": {
        "file_name_prefix": "rx-payload-bits",
        "scope": "gNB",
        "description": "Received PUSCH payload bits",
        "serialization_scheme": ["bits", "subcarriers", "ofdm_symbols"],
    },
    # UE messages
    "UE_PHY_UL_SCRAMBLED_TX_BITS": {
        "file_name_prefix": "tx-scrambled-bits",
        "scope": "UE",
        "description": "Transmitted scrambled PUSCH bits",
        "serialization_scheme": ["bits", "subcarriers", "ofdm_symbols"],
    },
    "UE_PHY_UL_PAYLOAD_TX_BITS": {
        "file_name_prefix": "tx-payload-bits",
        "scope": "UE",
        "description": "Transmitted PUSCH payload bits",
        "serialization_scheme": ["bits", "subcarriers", "ofdm_symbols"],
    },
}
# -------------------------------------------
# System configuration: gNB
project_id_gnb = 2335
read_shm_path_gnb = "/tmp/gnb_app1"
write_shm_path_gnb = "/tmp/gnb_app2"

# System configuration: UE
project_id_ue = 2336
read_shm_path_ue = "/tmp/ue_app1"
write_shm_path_ue = "/tmp/ue_app2"

# initialize shared memory
def attach_shm(shm_path, project_id):
    key = ipc.ftok(shm_path, project_id)
    shm = ipc.SharedMemory(key, 0, 0)
    # I found if we do not attach ourselves
    # it will attach as ReadOnly.
    shm.attach(0, 0)
    return shm

def detach_shm(shm):
    try:
        shm.detach()
        print("Shared memory detached successfully.")
    except ipc.ExistentialError:
        print("Shared memory segment does not exist.")

def remove_shm(shm):
    try:
        shm.remove()
        print("Shared memory removed successfully.")
    except ipc.ExistentialError:
        print("Shared memory segment does not exist.")

# Function to parse the OAI T_messages file and get the index of a given string
def parse_message_file(file_path):
    with open(file_path, "r") as file:
        content = file.readlines()

    # Extract lines that start with 'ID' and remove the 'ID = ' prefix
    tracer_msgs_identities = [
        line.strip().replace("ID = ", "")
        for line in content
        if line.strip().startswith("ID")
    ]

    return tracer_msgs_identities

# Function to get the index of a given string in the list of ID lines
def get_index_of_id(tracer_msgs_identities, message_id):
    try:
        return tracer_msgs_identities.index(message_id)
    except ValueError:
        return -1  # Return -1 if the string is not found

def real_to_complex(real_vector):
    # Ensure the length of the real vector is even
    if len(real_vector) % 2 != 0:
        raise ValueError("The length of the real vector must be even.")

    # Split the real vector into real and imaginary parts
    real_part = real_vector[::2]
    imag_part = real_vector[1::2]

    # Combine the real and imaginary parts to form a complex vector
    complex_vector = real_part + 1j * imag_part

    return complex_vector

# Data Collection Trace Messages - General message structure - number of bytes
def get_general_msg_header_list():
    """
    shared memory layout written from the app:
    =================================
    msg_id                  (uint8)  message type ID
    frame                   (uint16)
    slot                    (uint8)
    datetime_yyyymmdd       (uint32)
    datetime_hhmmssmmm      (uint32)
    frame_type              (uint8)
    freq_range              (uint8)
    subcarrier_spacing      (uint8)
    cyclic_prefix           (uint8)
    symbols_per_slot        (uint8)
    Nid_cell                (uint16)
    rnti                    (uint16)
    rb_size                 (uint16)
    rb_start                (uint16)
    start_symbol_index      (uint8)
    nr_of_symbols           (uint8)
    qam_mod_order           (uint8)
    mcs_index               (uint8)
    mcs_table               (uint8)
    nrOfLayers              (uint8)
    transform_precoding     (uint8)
    dmrs_config_type        (uint8)
    ul_dmrs_symb_pos        (uint16)
    number_dmrs_symbols     (uint8)
    dmrs_port               (uint16)
    dmrs_scid               (uint16)
    nb_antennas             (uint8)
    number_of_bits          (uint32)
    length_bytes            (uint32)
    For IQ Data: IQ samples: I0, Q0, I1, Q1, ... I_x, Q_x (int16)
    For bit data: bits: b0, b1, b2, ... b_x (uint8)
    """
    # Data Collection Trace Messages - General message structure - number of bytes
    general_msg_header_list = {
        "msg_id": 2,
        "frame": 2,
        "slot": 1,
        "datetime_yyyymmdd": 4,
        "datetime_hhmmssmmm": 4,
        "frame_type": 1,
        "freq_range": 1,
        "subcarrier_spacing": 1,
        "cyclic_prefix": 1,
        "symbols_per_slot": 1,
        "Nid_cell": 2,
        "rnti": 2,
        "rb_size": 2,
        "rb_start": 2,
        "start_symbol_index": 1,
        "nr_of_symbols": 1,
        "qam_mod_order": 1,
        "mcs_index": 1,
        "mcs_table": 1,
        "nrOfLayers": 1,
        "transform_precoding": 1,
        "dmrs_config_type": 1,
        "ul_dmrs_symb_pos": 2,
        "number_dmrs_symbols": 1,
        "dmrs_port": 2,
        "dmrs_scid": 2,
        "nb_antennas": 1,  # for gNB or nb_antennas_tx for UE
        "number_of_bits": 4,
        "length_bytes": 4,
    }
    # initial number of bytes to read to get data
    general_message_header_length = 0
    for key, value in general_msg_header_list.items():
        general_message_header_length = general_message_header_length + value
    return general_msg_header_list, general_message_header_length

# check data if avalible in the shared memory
def is_data_available_in_memory(shm, bufIdx, general_message_header_length, timeout=20):
    start_time = time.time()
    while True:
        buf = shm.read(bufIdx + general_message_header_length)
        n_bytes = sum(buf)
        print("Data Recording App: Waiting for Measurements!")
        if n_bytes > 0:
            print("There is data in memory, n_bytes: ", n_bytes)
            return True
        if (time.time() - start_time) > timeout:
            break
        time.sleep(1)
    return False

# check if first frame ahead:
def is_frame_ahead(frame1, frame2, max_frame=1023):
    """
    Check if frame1 is ahead of frame2, considering wrap-around from max_frame to 0.
    Args:
        frame1 (int): The first frame number.
        frame2 (int): The second frame number.
        max_frame (int): The maximum frame number before wrap-around. Default is 1023.
    Returns:
        bool: True if frame1 is ahead of frame2, False otherwise.
    """
    # Calculate the difference considering wrap-around
    diff = (frame1 - frame2 + (max_frame + 1)) % (max_frame + 1)
    # If the difference is less than half the range, frame1 is ahead
    return diff < (max_frame + 1) // 2

# Sync data between gNB and UE
def sync_gnb_ue_captured_data(shm_reading_gnb, shm_reading_ue):
    """
    Function to get the sync (frame, slot) data between gNB and UE
    Args:
    shm_reading_gnb: Shared memory for gNB
    shm_reading_ue: Shared memory for UE
    Returns:
    sync_info: Dictionary containing the sync information between gNB and UE
        sync_info["frame_start"] = frame_start
        sync_info["slot_start"]  = slot_start
        sync_info["gnb_frame_ahead"] = gnb_frame_ahead
        sync_info["frame_diff"] = frame_diff
    """
    # get general message header list
    general_msg_header_list, general_message_header_length = (
        get_general_msg_header_list()
    )

    # Read data from gNB T-tracer Application
    def get_frame_slot_start(
        shm_reading, bufIdx, general_msg_header_list, general_message_header_length
    ):
        buf = shm_reading.read(bufIdx + general_message_header_length)
        msg_id = struct.unpack(
            "<H", buf[bufIdx : bufIdx + general_msg_header_list.get("msg_id")]
        )[0]
        bufIdx += general_msg_header_list.get("msg_id")
        frame = struct.unpack(
            "<H", buf[bufIdx : bufIdx + general_msg_header_list.get("frame")]
        )[0]
        bufIdx += general_msg_header_list.get("frame")
        slot = struct.unpack(
            "B", buf[bufIdx : bufIdx + general_msg_header_list.get("slot")]
        )[0]
        return frame, slot

    # Read data from gNB T-tracer Application
    bufIdx = 0
    frame_gnb, slot_gnb = get_frame_slot_start(
        shm_reading_gnb, bufIdx, general_msg_header_list, general_message_header_length
    )
    # Read data from UE T-tracer Application
    bufIdx = 0
    frame_ue, slot_ue = get_frame_slot_start(
        shm_reading_ue, bufIdx, general_msg_header_list, general_message_header_length
    )

    # Sync data between gNB and UE
    # We noticed that the maximum difference between the frame number of gNB and UE is 3 frames
    # Calculate the frame difference considering the wrap-around from 1023 to 0
    sync_info = {}
    if frame_ue == frame_gnb:
        frame_start = frame_gnb
        slot_start = max(slot_gnb, slot_ue)
        gnb_frame_ahead = True
        frame_diff = 0
    elif is_frame_ahead(frame_gnb, frame_ue):
        frame_start = frame_gnb
        slot_start = slot_gnb
        gnb_frame_ahead = True
        frame_diff = (frame_gnb - frame_ue + 1024) % 1024
    elif is_frame_ahead(frame_ue, frame_gnb):
        frame_start = frame_ue
        slot_start = slot_ue
        gnb_frame_ahead = False
        frame_diff = (frame_ue - frame_gnb + 1024) % 1024
    # Determine the starting frame and slot for data sync
    sync_info["frame_gNB"] = frame_gnb
    sync_info["slot_gNB"] = slot_gnb
    sync_info["frame_UE"] = frame_ue
    sync_info["slot_UE"] = slot_ue
    sync_info["frame_start"] = frame_start
    sync_info["slot_start"] = slot_start
    sync_info["gnb_frame_ahead"] = gnb_frame_ahead
    sync_info["frame_diff"] = frame_diff
    return sync_info

# Read data from Shared memory based Data Conversion Service message structure
def read_data_from_shm(shm, bufIdx, tracer_msgs_identities):
    # get general message header list
    general_msg_header_list, general_message_header_length = get_general_msg_header_list()
    buf = shm.read(bufIdx + general_message_header_length)
    n_bytes = sum(buf)
    if n_bytes == 0:
        raise Exception('ERROR: No data available in memory')
    msg_id = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("msg_id")])[0]
    bufIdx += general_msg_header_list.get("msg_id")
    frame = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("frame")])[0]
    bufIdx += general_msg_header_list.get("frame")
    slot  = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("slot")])[0]
    bufIdx += general_msg_header_list.get("slot")
    # get time stamp:  yyyy mm dd hh mm ss msec
    nr_trace_time_stamp_yyymmdd = struct.unpack('<i', buf[bufIdx:bufIdx+general_msg_header_list.get("datetime_yyyymmdd")])[0]
    bufIdx += general_msg_header_list.get("datetime_yyyymmdd")
    nr_trace_time_stamp_hhmmssmmm = struct.unpack('<i', buf[bufIdx:bufIdx+general_msg_header_list.get("datetime_hhmmssmmm")])[0]
    bufIdx += general_msg_header_list.get("datetime_hhmmssmmm")
    time_stamp_milli_sec = str(nr_trace_time_stamp_yyymmdd)+"_"+str(nr_trace_time_stamp_hhmmssmmm)
    # get frame type
    frame_type = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("frame_type")])[0]
    bufIdx += general_msg_header_list.get("frame_type")
    # get frequency range
    freq_range = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("freq_range")])[0]
    bufIdx += general_msg_header_list.get("freq_range")
    # get subcarrier spacing
    subcarrier_spacing = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("subcarrier_spacing")])[0]
    bufIdx += general_msg_header_list.get("subcarrier_spacing")
    # get cyclic prefix
    cyclic_prefix = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("cyclic_prefix")])[0]
    bufIdx += general_msg_header_list.get("cyclic_prefix")
    # get symbols per slot
    symbols_per_slot = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("symbols_per_slot")])[0]
    bufIdx += general_msg_header_list.get("symbols_per_slot")
    # get Nid cell
    Nid_cell = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("Nid_cell")])[0]
    bufIdx += general_msg_header_list.get("Nid_cell")
    # get rnti
    rnti = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("rnti")])[0]
    bufIdx += general_msg_header_list.get("rnti")
    # get rb size
    rb_size = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("rb_size")])[0]
    bufIdx += general_msg_header_list.get("rb_size")
    # get rb start
    rb_start = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("rb_start")])[0]
    bufIdx += general_msg_header_list.get("rb_start")
    # get start symbol index    
    start_symbol_index = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("start_symbol_index")])[0]
    bufIdx += general_msg_header_list.get("start_symbol_index")
    # get number of symbols
    nr_of_symbols = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("nr_of_symbols")])[0]
    bufIdx += general_msg_header_list.get("nr_of_symbols")
    # get qam modulation order
    qam_mod_order = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("qam_mod_order")])[0]
    bufIdx += general_msg_header_list.get("qam_mod_order")
    # get mcs index
    mcs_index = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("mcs_index")])[0]
    bufIdx += general_msg_header_list.get("mcs_index")
    # get mcs table
    mcs_table = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("mcs_table")])[0]
    bufIdx += general_msg_header_list.get("mcs_table")
    # get number of layers
    nrOfLayers = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("nrOfLayers")])[0]
    bufIdx += general_msg_header_list.get("nrOfLayers")
    # get transform precoding
    transform_precoding = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("transform_precoding")])[0]
    bufIdx += general_msg_header_list.get("transform_precoding")
    # get dmrs config type
    dmrs_config_type = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("dmrs_config_type")])[0]
    bufIdx += general_msg_header_list.get("dmrs_config_type")
    # get ul dmrs symb pos
    ul_dmrs_symb_pos = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("ul_dmrs_symb_pos")])[0]
    bufIdx += general_msg_header_list.get("ul_dmrs_symb_pos")
    # get number dmrs symbols
    number_dmrs_symbols = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("number_dmrs_symbols")])[0]
    bufIdx += general_msg_header_list.get("number_dmrs_symbols")
    # get dmrs port
    dmrs_port = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("dmrs_port")])[0]
    bufIdx += general_msg_header_list.get("dmrs_port")
    # get dmrs scid
    dmrs_scid = struct.unpack('<H', buf[bufIdx:bufIdx+general_msg_header_list.get("dmrs_scid")])[0]
    bufIdx += general_msg_header_list.get("dmrs_scid")
    # get nb antennas rx for gNB or nb antennas tx for UE
    nb_antennas = struct.unpack('B', buf[bufIdx:bufIdx+general_msg_header_list.get("nb_antennas")])[0]
    bufIdx += general_msg_header_list.get("nb_antennas")
    # get number of bits
    number_of_bits = struct.unpack('<I', buf[bufIdx:bufIdx+general_msg_header_list.get("number_of_bits")])[0]
    bufIdx += general_msg_header_list.get("number_of_bits")
    # get length of bytes
    length_bytes = struct.unpack('<I', buf[bufIdx:bufIdx+general_msg_header_list.get("length_bytes")])[0]
    bufIdx += general_msg_header_list.get("length_bytes")

   
    # print all captured data
    print(" ")
    print(f"Time stamp: {time_stamp_milli_sec}")
    print(f"MSG ID: {msg_id:<5} MSG Name: {tracer_msgs_identities[msg_id]}")
    print(f"Frame: {frame:<5} Slot: {slot:<5}")
    print(f"Frame Type: {frame_type:<5} Frequency Range: {freq_range:<5} Subcarrier Spacing: {subcarrier_spacing:<5} Cyclic Prefix: {cyclic_prefix:<5} Symbols per Slot: {symbols_per_slot:<5}")
    print(f"Nid Cell: {Nid_cell:<5} RNTI: {rnti:<5}")
    print(f"RB Size: {rb_size:<5} RB Start: {rb_start:<5} Start Symbol Index: {start_symbol_index:<5} Number of Symbols: {nr_of_symbols:<5}")
    print(f"QAM Modulation Order: {qam_mod_order:<5} MCS Index: {mcs_index:<5} MCS Table: {mcs_table:<5}")
    print(f"Number of Layers: {nrOfLayers:<5} Transform Precoding: {transform_precoding:<5}")
    print(f"DMRS Config Type: {dmrs_config_type:<5} UL DMRS Symbol Position: {ul_dmrs_symb_pos:<5} Number of DMRS Symbols: {number_dmrs_symbols:<5}")
    print(f"DMRS Port: {dmrs_port:<5} DMRS SCID: {dmrs_scid:<5} Number of Antennas: {nb_antennas:<5}")
    print(f"Number of bits: {number_of_bits:<5} Length of bytes: {length_bytes:<5}")

    # get recorded data
    buf = shm.read(bufIdx + length_bytes)
    #bit_msg_index = get_index_of_id(tracer_msgs_identities, "GNB_PHY_UL_PAYLOAD_RX_BITS")
    captured_data = {}
    # If message is bit message, store data in bytes
    # then the field number_of_bits should be not zero
    
    if "_BITS" in  tracer_msgs_identities[msg_id]:
        #recorded_data = buf[bufIdx:bufIdx + length_bytes]
        recorded_data = struct.unpack("<"+ int(length_bytes) *'B', buf[bufIdx:bufIdx + length_bytes])
        bufIdx += length_bytes
        # convert data in bytes to bits
        bits_vector = []
        for byte in recorded_data:
            bits_vector.extend([int(bit) for bit in format(int(byte), '08b')])
        captured_data["sigmf_data_type"] = "ri8_le"
        captured_data["recorded_data"] = np.asarray(bits_vector).astype(np.uint8) # convert to uint8
        #recorded_data_formated = recorded_data.astype(np.complex64) # convert to complex64
    else:
        recorded_data = struct.unpack("<"+ int(length_bytes/2) *'h', buf[bufIdx:bufIdx + length_bytes])
        bufIdx += length_bytes

        #print("IQ data I/Q: ", recorded_data)
    
        # Convert real data to complext data
        # converting list to array
        recorded_data = np.asarray(recorded_data)
        #recorded_data_complex = recorded_data
        recorded_data_complex = real_to_complex(recorded_data)
        captured_data["sigmf_data_type"] = "cf32_le"
        captured_data["recorded_data"] = recorded_data_complex.astype(np.complex64) # convert to complex64
    #print("Recorded Data: ", captured_data["recorded_data"])
    # store data in dictonary
    captured_data["message_id"] = msg_id
    captured_data["message_type"] = tracer_msgs_identities[msg_id]
    captured_data["frame"] = frame
    captured_data["slot"] = slot
    captured_data["time_stamp"] = time_stamp_milli_sec
    captured_data["frame_type"] = frame_type
    captured_data["freq_range"] = freq_range
    captured_data["subcarrier_spacing"] = subcarrier_spacing
    captured_data["cyclic_prefix"] = cyclic_prefix
    #captured_data["symbols_per_slot"] = symbols_per_slot  ... not used
    captured_data["Nid_cell"] = Nid_cell
    captured_data["rnti"] = rnti
    captured_data["rb_size"] = rb_size
    captured_data["rb_start"] = rb_start
    captured_data["start_symbol_index"] = start_symbol_index
    captured_data["nr_of_symbols"] = nr_of_symbols
    captured_data["qam_mod_order"] = qam_mod_order
    captured_data["mcs_index"] = mcs_index
    captured_data["mcs_table"] = mcs_table
    captured_data["nrOfLayers"] = nrOfLayers
    captured_data["transform_precoding"] = transform_precoding
    captured_data["dmrs_config_type"] = dmrs_config_type
    captured_data["ul_dmrs_symb_pos"] = ul_dmrs_symb_pos
    captured_data["number_dmrs_symbols"] = number_dmrs_symbols
    captured_data["dmrs_port"] = dmrs_port
    captured_data["dmrs_scid"] = dmrs_scid
    captured_data["nb_antennas"] = nb_antennas
    captured_data["number_of_bits"] = number_of_bits
    
    return captured_data, bufIdx

def sync_data_conversion_service(
    shm_reading_gnb, shm_reading_ue, sync_info, config_meta_data
):
    # Initialize variables
    record_idx = 0
    prev_frame = -1
    prev_slot = -1
    ue_bufIdx = 0
    gnb_bufIdx = 0

    gnb_args.num_requested_tracer_msgs = len(
        config_meta_data["data_recording_config"]["base_station"][
            "requested_tracer_messages"
        ]
    )
    ue_args.num_requested_tracer_msgs = len(
        config_meta_data["data_recording_config"]["user_equipment"][
            "requested_tracer_messages"
        ]
    )
    tracer_msgs_identities = config_meta_data["data_recording_config"][
        "tracer_msgs_identities"
    ]

    global_info = config_meta_data["data_recording_config"]["global_info"]

    # Sync data between gNB and UE: Get buffer index where the sync data starts
    # Get the buffer index where the sync data starts for UE
    bufIdx = 0
    timeout_sync = time.time() + 5  # 5 seconds if no sync data found, stop the process
    while True:
        # wait for the next record
        # To do: check if we need to add exta waiting times between different events in case of 
        # data streaming via network such as on UE side or gNB side
        time.sleep(0.0035)  # Note: 2.3 ms = latency of T tracer to capture data from the RAN
        ue_bufIdx = bufIdx
        captured_data, bufIdx = read_data_from_shm(
            shm_reading_ue, bufIdx, tracer_msgs_identities
        )
        if (
            captured_data["frame"] == sync_info["frame_start"]
            and captured_data["slot"] == sync_info["slot_start"]
        ):
            break
        if time.time() > timeout_sync:
            raise Exception(
                "ERROR: Data Recording NO Sync Found, check Tracer Services if they are connected!"
            )

    # Get the buffer index where the sync data starts for gNB
    bufIdx = 0
    while True:
        time.sleep(0.0035)
        gnb_bufIdx = bufIdx
        captured_data, bufIdx = read_data_from_shm(
            shm_reading_gnb, bufIdx, tracer_msgs_identities
        )
        if (
            captured_data["frame"] == sync_info["frame_start"]
            and captured_data["slot"] == sync_info["slot_start"]
        ):
            break

    # Read Synchronized data between gNB and UE
    while True:  # read all records
        print(" ")
        print("Record number: ", record_idx)
        print(f"Buffer Index gNB: {gnb_bufIdx}, Buffer Index UE: {ue_bufIdx}")
        # wait for the next record
        # To do: check if we need to add exta waiting times between different events in case of 
        # data streaming via network such as on UE side or gNB side
        time.sleep(0.0035)  # 2.3 ms = latency of T tracer to capture data from the RAN

        collected_metafiles = []
        # Read data from gNB T-tracer Application
        for idx in range(gnb_args.num_requested_tracer_msgs):
            time.sleep(0.0015)
            print("Reading gNB data", idx)
            captured_data, gnb_bufIdx = read_data_from_shm(
                shm_reading_gnb, gnb_bufIdx, tracer_msgs_identities
            )

            # drive the collection file name from the first message per record
            if idx == 0:
                # Get time stamp
                time_stamp_ms, time_stamp_ms_file_name = (
                    sigmf_interface.time_stamp_formating(
                        captured_data["time_stamp"], global_info["datetime_offset"]
                    )
                )
                global_info["collection_file"] = (
                    global_info["collection_file_prefix"]
                    + "-rec-"
                    + str(record_idx)
                    + "-"
                    + str(time_stamp_ms_file_name)
                )
                global_info["timestamp"] = time_stamp_ms

            # Write data into files with the given format
            collected_metafiles.append(
                sigmf_interface.write_recorded_data_to_sigmf(
                    captured_data, config_meta_data, global_info, record_idx
                )
            )

        # Read data from UE T-tracer Application
        for idx in range(ue_args.num_requested_tracer_msgs):
            time.sleep(0.0015)
            captured_data, ue_bufIdx = read_data_from_shm(
                shm_reading_ue, ue_bufIdx, tracer_msgs_identities
            )

            # Write data into files with the given format
            collected_metafiles.append(
                sigmf_interface.write_recorded_data_to_sigmf(
                    captured_data, config_meta_data, global_info, record_idx
                )
            )

        # generate SigMF collection file
        data_storage_path = config_meta_data["data_recording_config"][
            "data_storage_path"
        ]
        description = global_info["description"]
        sigmf_interface.save_sigmf_collection(
            collected_metafiles, global_info, description, data_storage_path
        )

        frame = captured_data["frame"]
        slot = captured_data["slot"]

        # Check for changes in frame or slot
        if frame != prev_frame or slot != prev_slot:
            record_idx += 1
            # We have reached the end of the data. Break the loop
            if record_idx >= config_meta_data["data_recording_config"]["num_records"]:
                break
        # Update previous frame and slot
        prev_frame = frame
        prev_slot = slot

# data conversion service
def data_conversion_service(shm_reading, config_meta_data):
    # Initialize variables
    record_idx = 0
    prev_frame = -1
    prev_slot = -1
    bufIdx = 0
    # Read data from T-tracer Application
    gnb_args.num_requested_tracer_msgs = len(
        config_meta_data["data_recording_config"]["base_station"][
            "requested_tracer_messages"
        ]
    )
    ue_args.num_requested_tracer_msgs = len(
        config_meta_data["data_recording_config"]["user_equipment"][
            "requested_tracer_messages"
        ]
    )

    if gnb_args.num_requested_tracer_msgs > 0:
        num_requested_tracer_msgs = gnb_args.num_requested_tracer_msgs
    elif ue_args.num_requested_tracer_msgs > 0:
        num_requested_tracer_msgs = ue_args.num_requested_tracer_msgs
    else:
        raise Exception("ERROR: No requested tracer messages found!")

    tracer_msgs_identities = config_meta_data["data_recording_config"][
        "tracer_msgs_identities"
    ]
    global_info = config_meta_data["data_recording_config"]["global_info"]

    # Read data from T-tracer Application
    while True:  # read all records
        print(" ")
        print("Record number: ", record_idx)
        print(f"Buffer Index: {bufIdx}")
        # wait for the next record
        # To do: check if we need to add exta waiting times between different events in case of 
        # data streaming via network such as on UE side or gNB side
        time.sleep(0.0035)  # 2.3 ms = latency of T tracer to capture data from the RAN

        collected_metafiles = []
        for idx in range(num_requested_tracer_msgs):
            time.sleep(0.0015)
            captured_data, bufIdx = read_data_from_shm(
                shm_reading, bufIdx, tracer_msgs_identities
            )
            # derive the collection file name from the first message per record
            if idx == 0:
                # Get time stamp
                time_stamp_ms, time_stamp_ms_file_name = (
                    sigmf_interface.time_stamp_formating(
                        captured_data["time_stamp"], global_info["datetime_offset"]
                    )
                )
                global_info["collection_file"] = (
                    global_info["collection_file_prefix"]
                    + "-rec-"
                    + str(record_idx)
                    + "-"
                    + str(time_stamp_ms_file_name)
                )
                global_info["timestamp"] = time_stamp_ms
            # Write data into files with the given format
            collected_metafiles.append(
                sigmf_interface.write_recorded_data_to_sigmf(
                    captured_data, config_meta_data, global_info, record_idx
                )
            )

        frame = captured_data["frame"]
        slot = captured_data["slot"]

        # generate SigMF collection file
        data_storage_path = config_meta_data["data_recording_config"][
            "data_storage_path"
        ]
        description = global_info["description"]
        sigmf_interface.save_sigmf_collection(
            collected_metafiles, global_info, description, data_storage_path
        )

        # Check for changes in frame or slot
        if frame != prev_frame or slot != prev_slot:
            record_idx += 1
            # We have reached the end of the data. Break the loop
            if record_idx >= config_meta_data["data_recording_config"]["num_records"]:
                break
        # Update previous frame and slot
        prev_frame = frame
        prev_slot = slot

# Write Tracer Control Message
def write_shm(shm, args):
    if args.action == "record":
        # Note: Big Endian >, Little Endian <
        # Determine the length of the IP address
        ip_length = len(args.bytes_IPaddress) + 1  # String terminator

        # Construct the format string dynamically
        format_string = f"{ip_length}s"

        shm.write(
            struct.pack("<B", ip_length)
            + struct.pack(format_string, args.bytes_IPaddress)
            +
            # message +
            struct.pack("<h", int(args.port))
            + struct.pack("<B", args.num_requested_tracer_msgs)
            + struct.pack(
                "<{}h".format(len(args.req_tracer_msgs_indices)),
                *args.req_tracer_msgs_indices,
            )
            + struct.pack("<I", args.num_records)
            + struct.pack("<h", args.start_frame_number)
        )
        print(
            "Record: N Messags: ",
            args.num_requested_tracer_msgs,
            ", Msg IDs: ",
            args.req_tracer_msgs_indices,
            ", Num records: ",
            args.num_records,
            ", Start Frame: ",
            args.start_frame_number,
        )
    elif args.action == "quit":
        print("Quit:", bytes(args.action, "utf-8"))
        shm.write(bytes(args.action, "utf-8"))
    else:
        print("Unknown action for data recording system!")


def get_requested_tracer_msgs_indices(
    requested_tracer_messages, tracer_msgs_identities
):
    # get requested tracer messages indices
    req_tracer_msgs_indices = []
    for idx, value in enumerate(requested_tracer_messages):
        msg_index = get_index_of_id(tracer_msgs_identities, value)
        req_tracer_msgs_indices.append(msg_index)
    print("Requested Traces IDs: ", req_tracer_msgs_indices)
    return req_tracer_msgs_indices


def write_config_data_recording_app_json(config_meta_data):
    if config_meta_data["data_recording_config"]["global_info"][
        "save_config_data_recording_app_json"
    ]:
        try:
            json.dumps(config_meta_data)
            is_json_serializable = True
        except (TypeError, ValueError) as e:
            is_json_serializable = False
            print(f"data_recording_config_meta_json is not JSON serializable: {e}")

        # Specify the file name
        output_file = (
            config_meta_data["data_recording_config"]["data_storage_path"]
            + "config_data_recording_app.json"
        )
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Write the JSON data to the file
        with open(output_file, "w") as file:
            try:
                json.dump(config_meta_data, file, indent=4)
                print(f"JSON file created successfully at {output_file}")
            except Exception as e:
                print(f"Failed to create JSON file: {e}")


if __name__ == "__main__":
    # -------------------------------------------
    # ------------- Configuration --------------
    # ------------------------------------------
    # Data recording app Configuration
    data_recording_config_file = "config/config_data_recording.json"

    # ----------- Execution ------------------------
    parser = argparse.ArgumentParser(description="request messages IDs")
    ue_args = parser.parse_args()
    gnb_args = parser.parse_args()

    # -------------------------------------------
    # get the configuration from the JSON file
    # -------------------------------------------
    # collect args to write them to shared memory
    # get base station and user equipment configurations
    # Read and parse the JSON file
    with open(data_recording_config_file, "r") as file:
        config_meta_data = json.load(file)
  
    # get Time Stamp
    start_time = time.time()

    # get Tracer Messages IDs from the T-Tracer Messages file
    #   ... common/utils/T/T_messages.txt
    # Parse the T_messages file
    tracer_msgs_identities = parse_message_file(
        config_meta_data["data_recording_config"]["t_tracer_message_definition_file"]
    )
    config_meta_data["data_recording_config"][
        "tracer_msgs_identities"
    ] = tracer_msgs_identities

    # get requested tracer messages indices for gNB
    gnb_args.requested_tracer_messages = config_meta_data["data_recording_config"][
        "base_station"
    ]["requested_tracer_messages"]

    # get requested tracer messages indices for UE
    ue_args.requested_tracer_messages = config_meta_data["data_recording_config"][
        "user_equipment"
    ]["requested_tracer_messages"]

    # check if both lists of requested tracer messages IDs are not empty
    if not gnb_args.requested_tracer_messages and not ue_args.requested_tracer_messages:
        raise Exception("ERROR: No requested tracer messages IDs are provided")

    # check if gnb_requested_tracer_messages is not empty
    if gnb_args.requested_tracer_messages:
        config_meta_data["data_recording_config"]["base_station"][
            "req_tracer_msgs_indices"
        ] = get_requested_tracer_msgs_indices(
            gnb_args.requested_tracer_messages, tracer_msgs_identities
        )
        # attach to the shared memory
        shm_writing_gnb = attach_shm(write_shm_path_gnb, project_id_gnb)
        shm_reading_gnb = attach_shm(read_shm_path_gnb, project_id_gnb)

        # get gNB Trace Messages
        gnb_args.num_records = config_meta_data["data_recording_config"]["num_records"]
        gnb_args.start_frame_number = config_meta_data["data_recording_config"][
            "start_frame_number"
        ]
        gnb_args.req_tracer_msgs_indices = config_meta_data["data_recording_config"][
            "base_station"
        ]["req_tracer_msgs_indices"]
        gnb_args.num_requested_tracer_msgs = len(
            config_meta_data["data_recording_config"]["base_station"][
                "req_tracer_msgs_indices"
            ]
        )
        # Split the string into IP and port
        gnb_args.IPaddress, gnb_args.port = config_meta_data["data_recording_config"][
            "tracer_service_baseStation_address"
        ].split(":")
        gnb_args.bytes_IPaddress = bytes(gnb_args.IPaddress, "utf-8")
        gnb_args.action = "record"

    # check if ue_requested_tracer_messages is not empty
    if ue_args.requested_tracer_messages:
        config_meta_data["data_recording_config"]["user_equipment"][
            "req_tracer_msgs_indices"
        ] = get_requested_tracer_msgs_indices(
            ue_args.requested_tracer_messages, tracer_msgs_identities
        )

        # attach to the shared memory
        shm_writing_ue = attach_shm(write_shm_path_ue, project_id_ue)
        shm_reading_ue = attach_shm(read_shm_path_ue, project_id_ue)

        # get UE Trace Messages
        ue_args.num_records = config_meta_data["data_recording_config"]["num_records"]
        ue_args.start_frame_number = config_meta_data["data_recording_config"][
            "start_frame_number"
        ]
        ue_args.req_tracer_msgs_indices = config_meta_data["data_recording_config"][
            "user_equipment"
        ]["req_tracer_msgs_indices"]
        ue_args.num_requested_tracer_msgs = len(
            config_meta_data["data_recording_config"]["user_equipment"][
                "req_tracer_msgs_indices"
            ]
        )
        ue_args.IPaddress, ue_args.port = config_meta_data["data_recording_config"][
            "tracer_service_userEquipment_address"
        ].split(":")
        ue_args.bytes_IPaddress = bytes(ue_args.IPaddress, "utf-8")
        ue_args.action = "record"

    # -------------------------------------------
    # send Tracer Control Message request to UE T-Tracer Application
    # -------------------------------------------
    #   It consists of the following fields:
    #       IP address
    #       Port Number
    #       Number of records in slots
    #       Start SFN: Frame Index to start data collection from it, useful for future data sync between gNB and UE  --> not yet used
    #       Number of traces
    #       Trace Type ID 1, …, ID N

    # Get the current time
    time_stamp_micro_sec = datetime.now().strftime("%Y%m%d-%H%M%S%f")
    print("Send data logging request us:", time_stamp_micro_sec)

    if gnb_args.requested_tracer_messages and ue_args.requested_tracer_messages:
        # Create a barrier to synchronize the threads
        barrier = threading.Barrier(2)

        def write_shm_ue(ue_args):
            # Wait for both threads to be ready
            barrier.wait()
            # send request to UE T-Tracer Application
            write_shm(shm_writing_ue, ue_args)

        # send request to gNB T-Tracer Application
        def write_shm_gnb(gnb_args):
            # Wait for both threads to be ready
            barrier.wait()
            # send request to gNB T-Tracer Application
            write_shm(shm_writing_gnb, gnb_args)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("ue_args: ", ue_args)
            print("gnb_args: ", gnb_args)
            tracer_ue = executor.submit(write_shm_ue, ue_args)
            tracer_gnb = executor.submit(write_shm_gnb, gnb_args)

            # Wait for both functions to complete
            concurrent.futures.wait([tracer_ue, tracer_gnb])

    elif gnb_args.requested_tracer_messages:
        print("gnb_args: ", gnb_args)
        write_shm(shm_writing_gnb, gnb_args)
    elif ue_args.requested_tracer_messages:
        print("ue_args: ", ue_args)
        write_shm(shm_writing_ue, ue_args)
    else:
        raise Exception("ERROR: No requested tracer messages IDs are provided")

    # -------------------------------------------
    # Read data from UE T-tracer Application
    # -------------------------------------------
    # Check if data is available in memory
    # get general message header list
    general_msg_header_list, general_message_header_length = (
        get_general_msg_header_list()
    )
    # Initialize variables
    bufIdx = 0
    timeout = 10  # 10 seconds from now

    # If gNB MSGs are requested
    if gnb_args.requested_tracer_messages:
        if not is_data_available_in_memory(
            shm_reading_gnb, bufIdx, general_message_header_length, timeout
        ):
            time.sleep(1)  # Wait for the client to read the status
            raise Exception(
                "ERROR: Time out, check if gNB T-Tracer APP connected to stack"
            )

    # If UE MSGs are requested
    if ue_args.requested_tracer_messages:
        # Check if data is available in UE memory
        if not is_data_available_in_memory(
            shm_reading_ue, bufIdx, general_message_header_length, timeout
        ):
            raise Exception(
                "ERROR: Time out, check if UE T-Tracer APP connected to stack"
            )

    # -------------------------------------------
    # Sync data between gNB and UE
    # -------------------------------------------
    # Add supported OAI Tracer Messages
    config_meta_data["data_recording_config"][
        "supported_oai_tracer_messages"
    ] = supported_oai_tracer_messages
    # Add global info
    config_meta_data["data_recording_config"]["global_info"] = global_info

    # Create JSON file
    write_config_data_recording_app_json(config_meta_data)

    if gnb_args.requested_tracer_messages and ue_args.requested_tracer_messages:
        # Sync data between gNB and UE
        sync_info = sync_gnb_ue_captured_data(shm_reading_gnb, shm_reading_ue)
        print("\nSync data between gNB and UE: ", sync_info)
        
        # Read data from gNB and UE T-tracer Applications
        sync_data_conversion_service(
            shm_reading_gnb, shm_reading_ue, sync_info, config_meta_data
        )
    elif gnb_args.requested_tracer_messages:
        # Read data from gNB T-tracer Application
        data_conversion_service(shm_reading_gnb, config_meta_data)
    elif ue_args.requested_tracer_messages:
        # Read data from UE T-tracer Application
        data_conversion_service(shm_reading_ue, config_meta_data)
    else:
        raise Exception("ERROR: No requested tracer messages IDs are provided")

    # Stop T-Tracer Application function
    # For future work, we can add a function to stop the T-Tracer application
    # currenlty, the t-tracer stops if the number of records is reached
    if gnb_args.requested_tracer_messages:
        gnb_args.action = "quit"
        write_shm(shm_writing_gnb, gnb_args)
        # Clean shared memory
        detach_shm(shm_reading_gnb)
        detach_shm(shm_writing_gnb)
        remove_shm(shm_reading_gnb)
        remove_shm(shm_writing_gnb)
    if ue_args.requested_tracer_messages:
        ue_args.action = "quit"
        write_shm(shm_writing_ue, ue_args)
        # Clean shared memory
        detach_shm(shm_reading_ue)
        detach_shm(shm_writing_ue)
        remove_shm(shm_reading_ue)
        remove_shm(shm_writing_ue)

    # measure Elapsed time
    end_time = time.time()
    time_elapsed = end_time - start_time
    time_elapsed_ms = int(time_elapsed * 1000)
    print(
        "Elapsed time of getting Requested Messages and writing data and meta data files:",
        colored(time_elapsed_ms, "yellow"),
        "ms",
    )
    print(" ")

    print("End of the RF Data Recording API")
    pass
