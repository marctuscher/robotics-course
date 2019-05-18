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
os.getcwd()

model = RetinaNet('models/retinanet/retinanet_resnet152_level_1_v1.2_converted.h5')

#%%
cam = ry.Camera("test", "/camera/color/image_raw/", "/camera/depth/image_rect_raw/")
#%%
img = cam.getRgb()
model.predict(img, True)
#%%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#%%
d = cam.getDepth()
plt.imshow(d)


