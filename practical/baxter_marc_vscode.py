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

#%% do some simple computer vision
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!

import cv2

cameraView.selectSensor('kinect')
I = cameraView.computeImageAndDepth()
depth = I[1]
rgb = I[0]

print('rgb', rgb)
print('d', depth)
plt.imshow(rgb)
plt.show()

#%% 
print('gray scale image')
plt.figure()
graying = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
plt.imshow(graying, cmap='gray', vmin=0, vmax=255)

#%%
print('gaussian blur')
blurred = cv2.GaussianBlur(graying,(3,3), 0)
plt.imshow(blurred, cmap='gray', vmin=0, vmax=255)

#%%
blurred = cv2.GaussianBlur(graying,(3,3), 0)
circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
#print('circles: ', circles)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(blurred,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(blurred,(i[0],i[1]),2,(0,0,255),3)

print('show detected circles')
plt.imshow(blurred, cmap='gray', vmin=0, vmax=255)

#%%
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

# as we found te center pixel of the ball in image space we
# want to compute cartesian coordinates using depth infromation 
# from kinect

u = circles[0,0,0]
v = circles[0,0,1]

#cv2.circle(depth,(i[0],i[1]),2,(0,0,255),3)
#plt.imshow(depth)

# now use depht pixel and also do a spatial filtering to account the 
# neighborhood 

range_x = [-1,0,1]
range_y = [-1,0,1]
cumulated_depth = 0
for x in range_x:
    for y in range_y:
        cumulated_depth += depth[u + x][v + y]

mean_depth = cumulated_depth / (len(range_x) * len(range_y))

print('standard_depth: ', depth[u][y])
print('mean_depth: ', mean_depth)

# as we got the depth of the circle in reference to the kinect frame
# we transform the point in cartesian space


#%%
robot.deleteFrame('camera')

#%%
cameraView = robot.C.cameraView()

#%%
robot.setGripper(0, -4)

