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
from autolab_core import YamlConfig
from gqcnn.grasping import Grasp2D, SuctionPoint2D, CrossEntropyRobustGraspingPolicy, RgbdImageState
from gqcnn.utils import GripperMode, NoValidGraspsException
from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
from visualization import Visualizer2D as vis

#%%
import rospy
from cv_bridge import CvBridge, CvBridgeError
from practical.rosComm import RosComm
import sensor_msgs

#%%
from practical.webserver import sampleClient
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, calcDepth,findBallInImage, virtCamIntrinsics as intr
from practical import utils
import libry as ry

#%%
rosco = RosComm()

#%% 
rospy.init_node('z')
#%%
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')

#%%
intr = rosco.get_camera_intrinsics('/camera/color/camera_info')
#%%
intr

#%%
cfg = YamlConfig('practical/cfg/gqcnn_pj_dbg.yaml')
#%%
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])

#%%
img = rosco.rgb
d = rosco.depth
#%%
plt.imshow(img)
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


#%%
grasp.grasp.Grasp2D

#%%
#%%
cam = ry.Camera('lolz', '/camera/color/image_raw/', '/camera/depth/image_rect_raw/', True)
#%%
img_ry = cam.getRgb()
d_ry = cam.getDepth()
