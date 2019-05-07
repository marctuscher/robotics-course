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



##################################
# Maths stuff                    #
##################################

def quatMultiply(quat0, quat1):
    w0, x0, y0, z0 = quat0
    w1, x1, y1, z1 = quat1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0, 
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def quatConj(quat):
    w0, x0, y0, z0 = quat
    return np.array([w0, -x0, -y0, -z0], dtype=np.float64)