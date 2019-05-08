import sys
sys.path.append('../')
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R




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

def arr2quat(q):
    return quaternion.as_quat_array(q)

def quat2arr(q):
    return quaternion.as_float_array(q)

def quatMultiply(quat0, quat1):
    # deprecated: will be removed soon
    w0, x0, y0, z0 = quat0
    w1, x1, y1, z1 = quat1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                    x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0, 
                    -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                    x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

def quatConj(quat0):
    # deprecated: will be removed soon
    w0, x0, y0, z0 = quat0
    return np.array([w0, -x0, -y0, -z0], dtype=np.float64)

def pose7d2homTF(pose7d):
    homTF = np.eye(4)
    
    pos = pose7d[0:3]
    q = arr2quat(pose7d[3:7])
    R = quaternion.as_rotation_matrix(q)

    homTF[0:3, 0:3] = R
    homTF[0:3, 3] = np.transpose(pos)
    return homTF

def homTF2pose7d(homTF):
    
    pose7d = np.zeros(7)
    pose7d[0:3] = np.transpose(homTF[0:3, 3])

    R = homTF[0:3, 0:3]
    q_ = quaternion.from_rotation_matrix(R)
    q = quaternion.as_float_array(q_)

    pose7d[3:7] = q
    return pose7d

def quat2rotm(q):

    q_ = arr2quat(q)
    return quaternion.as_rotation_matrix(q_)

def rotm2quat(rotm):

    q = quaternion.from_rotation_matrix(rotm)
    return quaternion.as_float_array(q)

def rotm2eulZYX(rotm):

    r = R.from_dcm(rotm)
    zyx = r.as_euler('zyx', degrees=False)
    return zyx
