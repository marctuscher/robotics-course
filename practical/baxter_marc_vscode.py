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
robot =  RaiRobot('', 'rai-robotModels/pr2/pr2.g')

#%%
robot = reset(robot, 'rai-robotModels/pr2/pr2.g')

#%%
robot.inverseKinematics([
    gazeAt(['l_wrist_roll_link_1', 'ball']), 
    distance(['l_wrist_roll_link_1', 'ball'], 1)
    ])

#%%
robot.getFrameNames()


#%%
cameraView = robot.C.cameraView()
cameraView.addSensor(name='kinect', frameAttached='endeffKinect',  width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')

#%%
import matplotlib.pyplot as plt
cameraView.selectSensor('kinect')
I = cameraView.computeImageAndDepth()
depth = I[1]
rgb = I[0]
print('rgb', rgb)
print('d', depth)
plt.imshow(rgb)
plt.show()

#%%
robot.deleteFrame('camera')

#%%
cameraView = robot.C.cameraView()

#%%
robot.setGripper(0, -4)