#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import pandas as pd


# In[98]:


dataPath = './'
csv_filename = 'iq_643_6_ueTimeDomainSamples_mcs23' + '.csv'
df = pd.read_csv(dataPath + csv_filename, sep=';')
dataR = df['real'].to_numpy().astype(np.int16)
dataI = df['imag'].to_numpy().astype(np.int16)
dataInter = np.zeros((2*dataR.size),dtype=np.int16)
dataInter[0::2] = dataR
dataInter[1::2] = dataI


# In[99]:


cell_config = {
    "dl_freq" : 1815000000,
    "frame" : 643,
    "slot" : 6,
    "cellid" : 0,
    "rnti" : 32768,
}


# In[101]:


# Create bin file with sib1, rrcsetup config and samples
sib_file = 'keysight_gnb_sib1.bin'
rrc_file = 'keysight_gnb_rrcsetup.bin'
write_file = 'dlsim_input_data.bin'
with open(dataPath + write_file, 'wb') as wf:
    for key,value in cell_config.items():
        wf.write(np.uint64(value).tobytes())
    with open(dataPath + sib_file, 'rb') as f:
        data = f.read()
        wf.write(np.uint32(len(data)).tobytes())
        wf.write(data)
    with open(dataPath + rrc_file, 'rb') as f:
        data = f.read()
        wf.write(np.uint32(len(data)).tobytes())
        wf.write(data)
    wf.write(np.uint64(dataInter.size))
    dataInter.tofile(wf)
