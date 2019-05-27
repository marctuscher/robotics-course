#%%
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline
#%%
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('.')
import time
import os
print(os.getcwd())
#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage, getGraspPosition, maskDepth
from practical import utils
import libry as ry

#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')

#%%
#robot.C.addObject(name="ball3", shape=ry.ST.sphere, size=[.05], pos=[0.8,-0.2,1], color=[0.,0.,1.])

#%%
#ball = robot.C.frame('ball3')
#%%
#ball.setPosition([0.4, 0.8, 1.3])

#%%
robot.C.makeObjectsConvex()
#%%
def gatherDataSet(steps=10, pos = [0.2, 0.8, 1]):
    data = []
    for _ in range(steps):
        robot.goHome(hard=False, randomHome=True)
        q_data, q_dot_data = robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True, collectData=True)
        tmp = []
        for i in range(len(q_data)):
            tmp.append(np.concatenate([q_dot_data[i][0], q_dot_data[i][1]]))
        data += [tmp]
    return data


#%%
data = gatherDataSet()
#%%
len(data[0][1])
#%%
