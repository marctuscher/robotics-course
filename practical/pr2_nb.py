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
from practical import utils
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
from practical.objectives import moveToPose, qItself
#%%
p_act = robot.getPose('baxterL')
print('p_act: ', p_act)

p_target = p_act + np.array([0, 0, -.1, 0, 0, 0, 0])
print('p_tar: ', p_target)

q = robot.C.getJointState()
for i in range(0, 100):
    p_act = robot.getPose('baxterL')
    t = robot.computeCartesianTwist(p_act, p_target, gain=0.01)
    print('twist: ', t)

    target = p_act + t
    #target = p_target


    q = robot.inverseKinematics(
        [
        moveToPose(target, 'baxterL'),
        qItself(q, 1.)
        ]
    )
    robot.move([q])


#%%


#%%


#%%


#%%


#%%


#%%
