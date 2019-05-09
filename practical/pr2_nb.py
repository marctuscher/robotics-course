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
robot =  RaiRobot('awesome', 'rai-robotModels/baxter/baxter_new.g')


#%%
_, d = robot.imgAndDepth()
#%%
t_arr = np.zeros((3,3))

def timeFilter(t_arr):
    x = np.mean(t_arr[:][0])
    y = np.mean(t_arr[:][1])
    z = np.mean(t_arr[:][2])
    return np.array([x, y, z])

def addNewPoint(pos, t_arr):
    t_arr[0] = t_arr[1]
    t_arr[1] = t_arr[2]
    t_arr[2] = pos
    return t_arr
#%%
ball = robot.C.frame('ball2')
f = 1./np.tan(0.5*60.8*np.pi/180.)
f = f * 320.
boxMax = np.array([2., 2., 2.])
boxMin = -boxMax
while True:
    img, depth = robot.imgAndDepth()
    pc, x, y = findBallPosition(img, depth)
    if pc:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.circle(img,(x,y),2,(255,0,0),3)
        cv2.imshow('rgb', img)
        pos = robot.computeCartesianPos2(pc, 'pcl')
        t_arr = addNewPoint(pos, t_arr)
        p = timeFilter(t_arr)
        if utils.isPointInsideBox(boxMin, boxMax, pos):
            ball.setPosition(pos)
    robot.addPointCloud()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



#%%
from practical.objectives import moveToPose, qItself, accumulatedCollisions


#%%
frame = robot.C.frame('baxterL')
#%%
p_act = frame.getPose()
print('p_act: ', p_act)

p_target = p_act + np.array([0.1, 0, 0, 0, 0, 0, 0])
print('p_tar: ', p_target)

q = robot.C.getJointState()
for i in range(0, 100):
    p_act = frame.getPose()
    #t = robot.computeCartesianTwist(p_act, p_target, gain=0.01)
    #print('twist: ', t)


    target = (p_act - p_target) * 0.01
    #target = p_target


    q = robot.inverseKinematics(
        [
        moveToPose(target, 'baxterL'),
        #qItself(q, 0.1)
        ]
    )
    robot.move([q])


#%%


#%%


#%%


#%%


#%%


#%%
