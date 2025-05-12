#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import matplotlib.pyplot as plt
import struct
import pandas as pd


# In[109]:


dataPath = '/home/sakthi/work/dlsim_perf/'
#csv_filename = 'iq_14_6_ueTimeDomainSamples_mcs26' + '.csv'
csv_filename = 'iq_197_6_ueTimeDomainSamples_mcs26_powerDrop1dB' + '.csv'
df = pd.read_csv(dataPath + csv_filename, sep=';')
dataR = df['real'].to_numpy().astype(np.int16)
dataI = df['imag'].to_numpy().astype(np.int16)
#shift = 2200
#dataR = np.roll(dataR, shift)
#dataI = np.roll(dataI, shift)
plt.plot(dataR)
print(dataI)
dataInter = np.zeros((2*dataR.size),dtype=np.int16)
dataInter[0::2] = dataR
dataInter[1::2] = dataI
print(dataInter)


# In[78]:


slotLen = 30720
numSlots = 10
slot = 6
dataFrame = np.zeros((numSlots * slotLen,2), dtype=np.int16)
dataFrame[slotLen*6:slotLen*6+slotLen,0] = dataR
dataFrame[slotLen*6:slotLen*6+slotLen,1] = dataI
write_file = 'wavejudge_input.txt'
np.savetxt(dataPath + write_file,dataFrame,delimiter=',', fmt='%d')


# In[80]:


# PDSCH config
import yaml

config_file = 'PDCCH_703-6_MCS22'
config_ext = '.txt'
with open(dataPath + config_file + config_ext, "r") as f:
    content = f.read().replace('\t', '  ')  # two spaces

with open(dataPath + config_file + 'space' + config_ext, "w") as f:
    f.write(content)

print(content)


# In[107]:


cell_config = {
    "dl_freq" : 1815000000,
    "frame" : 643,
    "slot" : 6,
    "cellid" : 0,
    "rnti" : 32768,
}


# In[100]:


write_file = 'dlsim_input_data.bin'
with open(dataPath + write_file, 'wb') as f:
    for key,value in cell_config.items():
        f.write(np.uint64(value).tobytes()) # write pdsch config
    dataInter.tofile(f)            # Write time domain samples


# In[110]:


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


# In[50]:


get_ipython().system('jupyter nbconvert --to script dlsim_ext_data.ipynb')


# In[ ]:




