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
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter.g')

#%%
robot = reset(robot, 'rai-robotModels/baxter/baxter.g')

#%%
robot.grasp('baxterL', 'ball2', -1)
robot.goHome()
#%%
robot.getFrameNames()

#%%
robot.setGripper(0.04, -1)

#%%
robot.C.addFrame(name='endeffKinect', parent='endeffHead', args='Q:<t(0 0 -0.01) d(-19 1 0 0)>')
#C = K.view(frame='camera')
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
from PIL import Image
im = plt.imread('ball.jpg', format='jpeg')
#rgb = im
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
med = np.median(blurred)
lower = int(max(0,(1.0 - 0.33) * med))
upper = int(max(0,(1.0 + 0.33) * med))
print('lower: ', lower, 'upper: ', upper)
circles = cv2.HoughCircles(blurred,cv2.HOUGH_GRADIENT,1,45, param1=100,param2=10,minRadius=0,maxRadius=0)
#print('circles: ', circles)
print('circles: ', circles.shape)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    print('f: ', i)
    # draw the outer circle
    cv2.circle(rgb,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(rgb,(i[0],i[1]),2,(0,0,255),3)

print('show detected circles')
plt.imshow(rgb, cmap='gray', vmin=0, vmax=255)

#%%
circles
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

