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

#%%
os.getcwd()
#%%
img = cv2.imread('practical/cvtest.jpg')

#%%
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

#%%
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#%%
mask1 = cv2.inRange(img_hsv, (0, 70, 50), (10, 255, 255))
mask2 = cv2.inRange(img_hsv, (170, 70, 50), (180, 255, 255))

mask = mask1 | mask2

#%%
plt.imshow(mask)



#%%
