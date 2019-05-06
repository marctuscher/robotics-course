#%%
import sys
import numpy as np
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
while np.linalg.norm(pos - gripper.getPosition()) > 0.05:
    print(i)
    pos = calcBallPos(i, r, c)
    robot.trackAndGraspTarget(pos, 'ball2', 'pr2L', -3, 1)
    i += 1

