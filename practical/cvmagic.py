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
from practical.vision import houghsTransformFromBGR, plotCircles
import ry.libry as ry
#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
cam = ry.Camera()
#%%
while(True):
    # Capture frame-by-frame
    frame = cam.getRgb()
    depth = cam.getDepth()

    # Display the resulting frame
    circles, mask = houghsTransformFromBGR(frame, minRadius=20, maxRadius=100)
    if circles is not None:
        frame = plotCircles(frame, circles)
    cv2.imshow("frame", frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
#%%
circles = houghsTransformFromBGR(img)


#%%
plotCircles(img, circles)
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