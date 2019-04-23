#%%
import sys
sys.path.append('../')


#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, align, distance, gazeAt
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
robot.grasp_ball('baxterL', 'ball', -1)

#%%
robot.getFrameNames()

#%%
robot.setGripper(0.04, -1)



#%%
camView = robot.addCamera(name='camera', parent='head_tilt_link', args='Q:<d(-90 1 0 0) d(180 0 0 1)> focalLength:.3', view=True, width=80, height=80)

#%%
camView.computeImageAndDepth()
#%%
robot.deleteFrame('camera')

#%%
cameraView = robot.C.cameraView()


#%%
robot.setGripper(0, -3)