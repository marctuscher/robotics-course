#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
import openmesh
import glob

#%%
path = '/home/ralfi/git/robotics-course/mesh/'
#%%
i= 1
for filename in glob.iglob(path + '**/*.obj', recursive=True):
     if(i% 1000 == 0):
          print('objects precessed: ', i)
     mesh_ = openmesh.read_polymesh(filename)
     openmesh.write_mesh(filename[:-3] + 'stl', mesh_, binary=True)
     i += 1


#%%
