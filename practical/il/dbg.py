#%%
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('.')
import time
import os
print(os.getcwd())
#%%
import pickle
#%%
with open('data/baxter.pkl' , 'rb') as f:
    data = pickle.load(f)
#%%

def cleanFromGripperShit(data):
    data_out = []
    index = [15, 16, 32, 33]
    for i, traj in enumerate(data):
        traj_out = []
        for j, q in enumerate(traj):
           q_done = np.delete(q, index)
           traj_out.append(q_done)
        data_out.append(np.array(traj_out))
    return data_out
data = cleanFromGripperShit(data)
#%%fasdf
import pbdlib as pbd
#%%
model = pbd.HMM(nb_states=7, nb_dim=30)
#%%
model.init_hmm_kbins(data)
#%%
model.em(data, obs_fixed=True)
#%%
q =  data[0][0][0:17]
#%%
q
#%%
model.sigma[0][0:2, 0:2]
#%%
model.sigma[0]
#%%
for i in range(10):
    msg = model.predict(data[0][i][0:15], i)
    print("pred: ",msg)
    print("gt: ", data[0][i][15:])

#%%
