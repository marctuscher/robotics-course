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
import libry as ry
#%%
from practical.webserver.sampleClient import predictGQCNN, predictFCGQCNN

#%%
gc.collect()
#%%
robot =  RaiRobot('', 'rai-robotModels/baxter/baxter_new.g')
#%%
robot.goHome()
#%%
robot.graspPath(np.array([0.8, 0, 1]), 1.3,'ball2', 'baxterR', sendQ=True)


#%%
