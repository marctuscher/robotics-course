import sys
sys.path.append('../')
import numpy as np
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

def pose7d2homTF(pose7d):
    homTF = np.eye(4)
    
    pos = pose7d[0:3]
    rot = pose7d[3:7]

    # re-arrange quaternion to fit scipys noatation
    rot = np.array([rot[3], rot[1], rot[2], rot[0]])
   
    r_obj = R.from_quat(rot)
    homTF[0:3, 0:3] = r_obj.as_dcm()
    homTF[0:3, 3] = np.transpose(pos)
    return homTF

def homTF2pose7d(homTF):
    
    pose7d = np.zeros(7)
    pose7d[0:3] = np.transpose(homTF[0:3, 3])
    r_obj = R.from_dcm(homTF[0:3, 0:3])
    rot = r_obj.as_quat()
    
    # re-arrange quaternion to fit scipys noatation
    rot = np.array([rot[3], rot[1], rot[2], rot[0]])

    pose7d[3:7] = rot
    return pose7d


#%%


#%%
