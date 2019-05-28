#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

#%%
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')
import time
#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage, getGraspPosition, maskDepth
from practical import utils
import libry as ry

#%%
from autolab_core import YamlConfig
from gqcnn.grasping import Grasp2D, SuctionPoint2D, CrossEntropyRobustGraspingPolicy, RgbdImageState
from gqcnn.utils import GripperMode, NoValidGraspsException
from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
from visualization import Visualizer2D as vis
from practical.vision import baxterCamIntrinsics as intr
#%%
robot =  RaiRobot('marc2node', 'rai-robotModels/baxter/baxter_new.g')

#%%
robot.sendToReal(True)
robot.goHome()

#%%
robot.sync()
#%%
robot.move(robot.q_zero)

#%%
robot.goHome()
#%%
cfg = YamlConfig('practical/cfg/gqcnn_pj.yaml')
#%%
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])
#%%
gc.collect()
#%%
img, d = robot.imgAndDepth('cam')

#%%
plt.imshow(d)
#%%
d = maskDepth(d, 0.8, 1.4)
#%%
plt.imshow(d)
#%%
cam_intr = CameraIntrinsics(frame='pcl', fx=intr['fx'], fy=intr['fy'], cx=intr['cx'], cy=intr['cy'], height=intr['height'], width=intr['width'])
color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame='pcl')
depth_im = DepthImage(d.astype(np.float32), frame='pcl')
color_im = color_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im = depth_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
rgbd_state = RgbdImageState(rgbd_im, cam_intr)

#%%
grasp = grasp_policy(rgbd_state)
#%%
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.circle(img2,(int(grasp.grasp.center.x),int(grasp.grasp.center.y)),2,(255,0,0),3)
plt.imshow(img2)
print(d[int(grasp.grasp.center.x)][int(grasp.grasp.center.y)])

#%%
boxMax = np.array([1.5, 1.5, 1.5])
boxMin = -boxMax
lastPos = np.array([0, 0, 0])
time.sleep(0.2)
img, depth = robot.imgAndDepth('cam')
res =  getGraspPosition(d, grasp.grasp.center.x, grasp.grasp.center.y)
if res:
    pc, x, y = res
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img2,(int(x),int(y)),2,(255,0,0),3)
    plt.imshow(img2)
    pos = robot.computeCartesianPos(pc, 'pcl')
    if utils.isPointInsideBox(boxMin, boxMax, pos):
        robot.trackAndGraspTarget(np.array([pos[0], pos[1], pos[2]+0.05]), 'ball2', 'baxterR', sendQ=True)
  
#%%

robot.trackAndGraspTarget(np.array([pos[0], pos[1], pos[2]]), 'ball2', 'baxterR', sendQ=True)
#%%
robot.openBaxterR()
#%%
robot.closeBaxterR()
#%%
robot.setGripper(0.07, -2)
#%%
robot.goHome()
#%%
boxMax = np.array([1.5, 1.5, 1.5])
boxMin = -boxMax
lastPos = np.array([0, 0, 0])
time.sleep(0.2)
img, depth = robot.imgAndDepth('cam')
res =  findBallPosition(img, depth)
robot.addPointCloud()
if res:
    pc, x, y = res
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img,(x,y),2,(255,0,0),3)
    plt.imshow(img)
    pos = robot.computeCartesianPos(pc, 'pcl')
    if utils.isPointInsideBox(boxMin, boxMax, pos):
        robot.trackPath(pos, 'ball2', 'baxterR', sendQ=True)


#%%
