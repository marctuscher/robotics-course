#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline
#%%
import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')


#%%
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, findBallInImage, virtCamIntrinsics as intr
from practical import utils
import libry as ry


#%%
from autolab_core import YamlConfig
from gqcnn.grasping import Grasp2D, SuctionPoint2D, CrossEntropyRobustGraspingPolicy, RgbdImageState
from gqcnn.utils import GripperMode, NoValidGraspsException
from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
from visualization import Visualizer2D as vis

#%%
robot =  RaiRobot('awesome', 'rai-robotModels/baxter/baxter_new.g')


#%%
img, d = robot.imgAndDepth('cam')
#%%
plt.imshow(img)
#%%
cfg = YamlConfig('practical/cfg/gqcnn_pj.yaml')
#%%
cam_intr = CameraIntrinsics(frame='pcl', fx=intr['fx'], fy=intr['fy'], cx=intr['px'], cy=intr['py'], height=intr['height'], width=intr['width'])

#%%
color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame='pcl')
depth_im = DepthImage(d.astype(np.float32), frame='pcl')

#%%
plt.imshow(color_im._data)
#%%
color_im = color_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im = depth_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])

#%%
plt.imshow(color_im._data)

#%%
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
#%%
plt.imshow(rgbd_im.color._data)

#%%
rgbd_state = RgbdImageState(rgbd_im, cam_intr)

#%%
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])


#%%
grasp = grasp_policy(rgbd_state)
#%%

img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.circle(img2,(int(grasp.grasp.center.x),int(grasp.grasp.center.y)),2,(255,0,0),3)
plt.imshow(img2)
#%%
grasp.grasp.pose()

#%%
grasp.grasp.center.x

#%%
grasp.grasp.center.x
