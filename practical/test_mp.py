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
robot =  RaiRobot('marc2', 'rai-robotModels/baxter/baxter_new.g')
#%%
robot.sendToReal(True)
#%%
robot.goHome()
#%%
robot.move(robot.q_zero)
robot.sync()
#%%
img, d = robot.imgAndDepth('cam')
m = maskDepth(d, 0.6, 1.4)
#%%
plt.imshow(d)
#%%
grasp = predictGQCNN(img, d,'http://ralfi.nat.selfnet.de:5000')#segmask=m)
#%%
grasp = predictFCGQCNN(img, d,m,'http://ralfi.nat.selfnet.de:5000')#segmask=m)

#%%
grasp
#%%
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.circle(img,(int(grasp['x']),int(grasp['y'])),2,(255,0,0),3)
plt.imshow(img)

#%%
robot.sendToReal(True)
#%%
robot.sendToReal(False)
#%%
res =  getGraspPosition(d, grasp['x'], grasp['y'])
if res:
    pc, x, y = res
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img2,(int(x),int(y)),2,(255,0,0),3)
    plt.imshow(img2)
    pos = robot.computeCartesianPos(pc, 'pcl')
    robot.graspPath(np.array([pos[0], pos[1], pos[2]]), grasp['angle'],'ball2', 'baxterR', sendQ=True)
#%%
robot.C.addObject(name="ball3", shape=ry.ST.sphere, size=[.05], pos=[0.8,-0.2,1], color=[0.,0.,1.])

#%%
ball = robot.C.frame('ball3')
#%%
ball.setPosition([0.4, 0.8, 1.3])

#%%
robot.C.makeObjectsConvex()
#%%
def gatherDataSet(steps=10, pos = [0.2, 0.8, 1]):
    data = []
    for _ in range(steps):
        robot.goHome(hard=False, randomHome=True)
        q_data, q_dot_data = robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True, collectData=True)
        tmp = []
        for i in range(len(q_data)):
            tmp.append(np.concatenate([q_dot_data[i][0], q_dot_data[i][1]]))
        data += [tmp]
    return data


#%%
robot.closeBaxterR()
#%%
data = gatherDataSet()
#%%
len(data[0][1])
#%%
