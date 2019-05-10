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
import time


#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage
from practical import utils
import libry as ry

#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')


#%%
boxMax = np.array([2., 2., 2.])
boxMin = -boxMax
ball_target = robot.C.frame('ball2')
ball_reg = robot.C.frame('ball')
i = 0
r = 0.2
c = [0.8, 0, 1]
while True:
    ball_target.setPosition(utils.calcBallPos(i, r, c))
    img, depth = robot.imgAndDepth()
    res =  findBallPosition(img, depth)
    if res:
        pc, x, y = res
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.circle(img,(x,y),2,(255,0,0),3)
        cv2.imshow('rgb', img)
        pos = robot.computeCartesianPos(pc, 'pcl')
        if utils.isPointInsideBox(boxMin, boxMax, pos):
            robot.trackAndGraspTarget(pos, 'ball', 'baxterL')
    time.sleep(0.5)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#%%
from practical.objectives import moveToPose, qItself
#%%
targetFrame = 'baxterL'
p_act = robot.getPose(targetFrame)
print('p_act: ', p_act)

p_target = p_act + np.array([0, 0, .3, 0, 0, 0, 0])
print('p_tar: ', p_target)

boxMax = np.array([2., 2., 2.])
boxMin = -boxMax
ball_target = robot.C.frame('ball2')
ball_reg = robot.C.frame('ball')
i = 0
r = 0.2
c = [0.8, 0, 1]
while True:
    ball_target.setPosition(utils.calcBallPos(i, r, c))
    img, depth = robot.imgAndDepth()
    res =  findBallPosition(img, depth)
    if res:
        pc, x, y = res
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.circle(img,(x,y),2,(255,0,0),3)
        cv2.imshow('rgb', img)
        pos = robot.computeCartesianPos(pc, 'pcl')
        if utils.isPointInsideBox(boxMin, boxMax, pos):
            #robot.trackAndGraspTarget(pos, 'ball', 'baxterL')
        



    time.sleep(0.5)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

q = robot.C.getJointState()
while(True)
    p_act = robot.getPose(targetFrame)
    t = robot.computeCartesianTwist(p_act, p_target, gain=0.01)
    print('twist: ', t)

    targetPos = p_act + t
    #target = p_target
    target = robot.C.frame(targetFrame)

    target.setPosition(targetPos[0:3])
    q = robot.inverseKinematics(
        [
            gazeAt([gripperFrame, targetFrame]), 
            scalarProductXZ([gripperFrame, targetFrame], 0), 
            scalarProductZZ([gripperFrame, targetFrame], 0), 
            distance([gripperFrame, targetFrame], -0.1),
            qItself(q, 1.)
        ]
    )
    self.move([q])
#%%