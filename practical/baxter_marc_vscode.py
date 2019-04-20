#%%
import sys
sys.path.append('../')


#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, align, distance
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
robot.inverseKinematics([
    *align(['l_gripper_frame', 'ball']), 
    distance(['l_gripper_frame', 'ball'], 0)
    ])

#%%
robot.getFrameNames()




#%%
