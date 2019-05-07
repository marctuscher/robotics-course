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
import libry as ry

#%%
def reset(robot, model):
    """
    reset robot model
    only works when not connected to rosnode
    """
    robot.C = 0
    robot.D = 0
    robot.B = 0
    return RaiRobot('', model)

#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')

#%%
# for resetting when running without rosnode
robot = reset(robot, 'rai-robotModels/baxter/baxter_new.g')

#%%
ball = robot.C.frame('ball2')
#%%
img, d = robot.virtImgAndDepth()

#%%
plt.imshow(img)

#%%
ball.setPosition([5, 0, 1])





#%%
