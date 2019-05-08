#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')


#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage
from practical.utils import *
import libry as ry

#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')



#%%
ball = robot.C.frame('ball2')
#%%

while True:
    img, depth = robot.imgAndDepth()
    pc = findBallPosition(img, depth)
    pos = robot.computeCartesianPos(pc, 'pcl')
    ball.setPosition(pos)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#%%
p = robot.getPose('base')
print(p)
h_p = pose7d2homTF(p)
print(h_p)

p_7d = homTF2pose7d(h_p)
print(p_7d)

h = P[3:7]
cdg = quat2



#%%
l = np.array([1,0,0,0])
q = quaternion.as_quat_array(l)
print(q)