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
#%%
os.getcwd()

#%%
tf.reset_default_graph()

#graph = loadFrozenGraph('models/rcnn/rfcn_resnet101_coco_2018_01_28/')

#%%
l = [n.name for n in graph.as_graph_def().node]
#%%
l[-30:]
#%%
sess, x, classes, scores, boxes = loadRCNN('models/rcnn/rfcn_resnet101_coco_2018_01_28/')

#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')

#%%
img, d = robot.imgAndDepth('cam')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#%%
plt.imshow(img)

#%%
pred_class, pred_score, pred_box = sess.run([classes, scores, boxes], {x : [img]})


#%%
output


#%%
