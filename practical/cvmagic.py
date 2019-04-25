#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from practical.vision import computeHoughsTransform

#%%
os.getcwd()

#%%
robot.C.addFrame(name='endeffKinect', parent='endeffHead', args='Q:<t(0 0 -0.01) d(-19 1 0 0)>')
#C = K.view(frame='camera')
cameraView = robot.C.cameraView()
S =cameraView.addSensor(name='kinect', frameAttached='endeffKinect',  width=640, height=480, focalLength=580./480., orthoAbsHeight=-1., zRange=[.1, 50.], backgroundImageFile='')


#%%
cameraView.selectSensor('kinect')
I = cameraView.computeImageAndDepth()
depth = I[1]
rgb = I[0]
PC = cameraView.computePointCloud(depth, globalCoordinates=True)
#cameraView.watch_PCL(PC)
print(PC)

#%%
img = cv2.imread('practical/ball3.jpg')

#%%
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#%%
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#%%
mask1 = cv2.inRange(img_hsv, (0, 70, 70), (10, 255, 255))
mask2 = cv2.inRange(img_hsv, (170, 70, 70), (180, 255, 255))

mask = mask1 | mask2

#%%
plt.imshow(mask, cmap='gray', vmin=0, vmax=1)


#%%
circles = circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,45, param1=150,param2=13,minRadius=3,maxRadius=50)
#(mask, None)

#%%
circles = np.uint16(np.around(circles))
img_ = img
'''
for i in [circles[0,:]]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
'''
u = circles[0,0,0]
v = circles[0,0,1]
r = circles[0,0,2]
# draw the outer circle
cv2.circle(img_,(u,v),r,(0,255,0),2)
# draw the center of the circle
cv2.circle(img_,(u,v),2,(0,0,255),3)

print('show detected circles')
plt.imshow(img_, cmap='gray', vmin=0, vmax=255)



#%%

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

#%%
# as we got the depth of the circle in reference to the kinect frame
# we transform the point in cartesian space
### some parameters for kinect depth camera from the internet
fx_d = 5.9421434211923247e+02
fy_d = 5.9104053696870778e+02
cx_d = 3.3930780975300314e+02
cy_d = 2.4273913761751615e+02
k1_d = -2.6386489753128833e-01
k2_d = 9.9966832163729757e-01
p1_d = -7.6275862143610667e-04
p2_d = 5.0350940090814270e-03
k3_d = -1.3053628089976321e+00

ds = 1/1000 # Assumption MM_per;_M

#camvars = struct('fx',fx,'fy',fy,'cx',cx,'cy',cy,'ds',ds)


# Straighten the XYZ coordinates using the camera parameters
# do we have to flip th image???
I, J = np.meshgrid(np.arange(0,depth.shape[0]), np.arange(0,depth.shape[1]))
Z = -np.double(depth) * ds
X = (I - cx_d) @ Z / fx_d
Y = (J - cy_d) @ Z / fy_d

#XY = np.concatenate((X,Y,Z.T), axis=1)
#TODO make on mxnx3 matrix from 3 mxn matrices

#invalidIndex = find(depth(:)==0)
#szImg = numel(depthImage)
#xyz(invalidIndex) = NaN
#xyz(invalidIndex+szImg) = NaN
#xyz(invalidIndex+szImg*2) = NaN

#xyz = fliplr(xyz)

#xyz(:,:,1) = -xyz(:,:,1)
#xyz(:,:,2) = -xyz(:,:,2)
#%xyz(:,:,3) = -xyz(:,:,3)

#%%
-np.double(depth) * ds
#%%
#print(depth)
#I = np.meshgrid(1:len(depth)
print('x: ',len(depth[1]))
print('y: ',len(depth[0]))
print(depth.shape[1])