#%%
import sys
import numpy as np
sys.path.append('../')

from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, distance, gazeAt


robot =  RaiRobot('awesomeNode', 'rai-robotModels/baxter/baxter.g')
robot.sendToReal(True)
robot.grasp_ball('baxterL', 'ball2', -1)
robot.goHome()

