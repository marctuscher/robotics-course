#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
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
def gatherDataSet(steps=10, pos = [0.2, 1, 1]):
    data = []
    for _ in range(steps):
        robot.goHome(hard=True, randomHome=True)
        q_data, q_dot_data = robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True, collectData=True)
        data.append(q_data + q_dot_data)
    return data



#%%
robot.openBaxterR()
#%%
robot.goHome(True)

#%%
robot.C.addObject(name="ball3", shape=ry.ST.sphere, size=[.05], pos=[0.8,-0.2,1], color=[0.,0.,1.])

#%%
ball = robot.C.frame('ball3')
#%%
ball.setPosition([0.2, 0.7, 1.1])
#%%
pos = [0.2, 1, 1]
robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True)
#%%
img, d = robot.imgAndDepth('cam')
print(d)
plt.imshow(d)



#%%
boxMax = np.array([1.5, 1.5, 1.5])
boxMin = -boxMax
img, depth = robot.imgAndDepth('cam')
res =  findBallPosition(img, depth)
if res:
    pc, x, y = res
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img,(x,y),2,(255,0,0),3)
    plt.imshow(img)
    pos = robot.computeCartesianPos(pc, 'pcl')
    if utils.isPointInsideBox(boxMin, boxMax, pos):
        robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True)

#%%
