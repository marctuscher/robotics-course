#%%
remoteCalc = False
#%%
%reload_ext autoreload
%autoreload 2
%matplotlib inline

import sys
import numpy as np
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
sys.path.append('../')
from autolab_core import YamlConfig

if not remoteCalc:
    
    from gqcnn.grasping import Grasp2D, SuctionPoint2D, CrossEntropyRobustGraspingPolicy, RgbdImageState
    from gqcnn.utils import GripperMode, NoValidGraspsException
    from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
    from visualization import Visualizer2D as vis
    #from visualization import Visualizer as vis2D

import rospy
from cv_bridge import CvBridge, CvBridgeError
from practical.rosComm import RosComm
import sensor_msgs

from practical.webserver import sampleClient
from practical.raiRobot import RaiRobot
from practical.objectives import moveToPosition, gazeAt, scalarProductXZ, scalarProductZZ, distance
from practical.vision import findBallPosition, calcDepth,findBallInImage, virtCamIntrinsics as intr
from practical import utils
from practical import vision
#%%
import libry as ry
#%%
rosco = RosComm()
rospy.init_node('z')
intr_rs = rosco.get_camera_intrinsics('/camera/color/camera_info')
cfg = YamlConfig('practical/cfg/gqcnn_pj_dbg.yaml')
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')

#%%
cam = ry.Camera('h', "/camera/color/image_raw", "/camera/depth/image_rect_raw", True)
#%%
img = cam.getRgb()
d = cam.getDepth()
import time
time.sleep(.1)
img_ro = rosco.rgb
d_ro = rosco.depth
#%%
d[255][255]
#%%
d_ro[255][255]
#%%
plt.imshow(img)
#%%
plt.imshow(d, cmap='gray')
#%%
cam_intr = CameraIntrinsics(frame='pcl', fx=intr_rs['fx'], fy=intr_rs['fy'], cx=intr_rs['cx'], cy=intr_rs['cy'], height=intr_rs['height'], width=intr_rs['width'])
color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame='pcl')
depth_im = DepthImage(d.astype(np.float32), frame='pcl')
color_im = color_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
depth_im = depth_im.inpaint(rescale_factor=cfg['inpaint_rescale_factor'])
rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
rgbd_state = RgbdImageState(rgbd_im, cam_intr)

#%%
grasp = grasp_policy(rgbd_state)

#%%
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_ = cv2.circle(img,(int(grasp.grasp.center.x),int(grasp.grasp.center.y)),2,(255,0,0),3)
#%%
plt.imshow(img)
#%%
from autolab_core import Point
vis.figure()
vis.imshow(img)
grasp= Grasp2D(img, 0, grasp.grasp.center, 0.05)
vis.grasp(grasp)
vis.show()


#%%
rosco = RosComm()
rospy.init_node('z')
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')
intr_rs = rosco.get_camera_intrinsics('/camera/color/camera_info')
cfg = YamlConfig('practical/cfg/gqcnn_pj_dbg.yaml')
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])

