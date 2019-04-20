#%%
import sys
sys.path.append('../')


#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
from practical.raiRobot import RaiRobot
#%%
def reset(robot, model):
    robot.C = 0
    robot.D = 0
    robot.B = 0
    return RaiRobot('', model)


#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter.g')


#%%
robot = reset(robot, 'rai-robotModels/baxter/baxter.g')


#%%
robot.moveToPosition([0, 4, 1], 'pr2L')


#%%
robot.IK.getReport()










#%%
