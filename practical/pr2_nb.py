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
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt

#%%
def reset(robot, model):
    robot.C = 0
    robot.D = 0
    robot.B = 0
    return RaiRobot('', model)

#%%
robot =  RaiRobot('', 'rai-robotModels/pr2/pr2.g')

#%%
robot = reset(robot, 'rai-robotModels/pr2/pr2.g')

#%%
def calcBallPos(i, r, c):
    i = i % 360
    x = c[0]
    y = c[1] + r * np.sin(i)
    z = c[2] + r * np.cos(i)
    return [x,y,z]

#%%
c = [0.7, 0, 1.2]
gripperFrame = 'pr2L'
gripper = robot.C.frame('pr2L')
i = 0
r = 0.1
pos = calcBallPos(i, r, c)
while True: #np.linalg.norm(pos - gripper.getPosition()) > 0.05:
    robot.trackAndGraspTarget(pos, 'ball2', 'pr2L', -3, 1)
    i += 1
    pos = calcBallPos(i, r, c)



#%%
robot.getFrameNames()

#%%
robot.C.addFrame(name='endeffKinect', parent='endeffHead', args='Q:<t(0 0 -0.01) d(-19 1 0 0)>')
C = robot.C.view(frame='endeffKinect')
cameraView = robot.C.cameraView()
cameraView.addSensor(name='kinect', frameAttached='endeffKinect',  width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')
cameraView.selectSensor('kinect')

#%%
ball = robot.C.frame('ball2')

#%%
ball.setPosition([2, 0, 0.7])

#%%
img, d = cameraView.computeImageAndDepth()



#%%
cv2.imshow("img", img)

#%%
plt.imshow(img)

#%%
from practical.vision import findBallPosition
#%%
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
#%%
pc = findBallPosition(img, d, {'fx': 1, 'fy': 1, 'px': 1, 'py': 1})
pw = robot.computeCartesianPos(pc, 'endeffKinect')
#%%
