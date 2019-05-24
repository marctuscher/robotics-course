
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
robot =  RaiRobot('marc2node', 'rai-robotModels/baxter/baxter_new.g')

#%%
robot.sendToReal(True)
robot.goHome()
#%%
img, d = robot.imgAndDepth('cam')
print(d)
plt.imshow(d)
#%%
boxMax = np.array([1.5, 1.5, 1.5])
boxMin = -boxMax
lastPos = np.array([0, 0, 0])
time.sleep(0.2)
img, depth = robot.imgAndDepth('cam')
res =  findBallPosition(img, depth)
robot.addPointCloud()
if res:
    pc, x, y = res
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img,(x,y),2,(255,0,0),3)
    plt.imshow(img)
    pos = robot.computeCartesianPos(pc, 'pcl')
    if utils.isPointInsideBox(boxMin, boxMax, pos):
        robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True)

#%%
