from practical.raiRobot import RaiRobot
import numpy as np


def reset(robot, model):
    """
    reset robot model
    only works when not connected to rosnode
    """
    robot.C = 0
    robot.D = 0
    robot.B = 0
    return RaiRobot('', model)


def calcBallPos(i, r, c):
    """
    calculate ball position in circle in from of robot
    """
    i = i % 360
    x = c[0]
    y = c[1] + r * np.sin(i)
    z = c[2] + r * np.cos(i)
    return [x,y,z]