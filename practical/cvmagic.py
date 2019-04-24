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
img = cv2.imread('practical/cvtest.jpg')

#%%
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#%%
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#%%
mask1 = cv2.inRange(img_hsv, (0, 70, 70), (10, 255, 255))
mask2 = cv2.inRange(img_hsv, (170, 70, 70), (180, 255, 255))

mask = mask1 | mask2

#%%
plt.imshow(mask)


#%%
circles = computeHoughsTransform(mask, None)

#%%
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

print('show detected circles')
plt.imshow(mask, cmap='gray', vmin=0, vmax=255)



#%%
