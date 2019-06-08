#%%
import sys
sys.path.append('rai/rai/ry')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from practical.vision import redMask
import libry as ry
import tensorflow as tf
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
from practical.tf_utils import loadRCNN, loadFrozenGraph
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage
from practical import utils
from practical.rcnn import RetinaNet
#%%
os.environ["ROS_MASTER_URI"] = "http://thecount.local:11311/"
os.environ["ROS_IP"] = "129.69.216.204"
#%%
os.getcwd()

model = RetinaNet('models/retinanet/retinanet_resnet152_level_1_v1.2_converted.h5')

#%%
cam = ry.Camera("test", "/camera/rgb/image_raw/", "/camera/depth/image_rect/")
#%%
img = cam.getRgb()
#%%
plt.imshow(img)
#%%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#%%
plt.imshow(img)

#%%
img2 = img.copy()
#%%
model.predict(img, True)




#%%
