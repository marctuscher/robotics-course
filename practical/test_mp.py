#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
sys.path.append('.')
import time
import os
print(os.getcwd())
#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage, getGraspPosition, maskDepth
from practical import utils
from practical.dexnet import utils as dex_utils
import libry as ry
#%%
from practical.webserver import sampleClient

#%%
os.environ["ROS_MASTER_URI"] = "http://thecount.local:11311/"
os.environ["ROS_IP"] = "129.69.216.204"
gc.collect()
#%%
robot =  RaiRobot('test', 'rai-robotModels/baxter/baxter_new.g')
#%%
robot.sendToReal(True)
#%%
robot.goHome()
#%%
robot.move(robot.q_zero)
#%%
robot.graspPath(np.array([0.8, 0, 1]), 1.3,'ball2', 'baxterR', sendQ=True)

#%%
robot.path.getReport()
#%%
img, d = robot.imgAndDepth('cam')
#%%
grasp = sampleClient.predictMask(d, 'http://localhost:5000')
#%%
grasp
#%%
from practical import vision
#%%
grasp
#%%
vision.plotCircleAroundCenter(img, grasp['x'], grasp['y'])
#%%
robot.approachGrasp(grasp, d)
#%%
robot.goHome()
#%%
