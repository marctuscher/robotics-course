#%%
import sys
import numpy as np

from practical.raiRobot import RaiRobot


robot =  RaiRobot('awesomeNode', 'rai-robotModels/baxter/baxter.g')
robot.sendToReal(True)
robot.grasp_ball('baxterL', 'ball2', -1)
robot.goHome()

