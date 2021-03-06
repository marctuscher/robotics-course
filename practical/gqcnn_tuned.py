#%%
remoteCalc = True
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
import libry as ry

#%%
rosco = RosComm()
#%% 
rospy.init_node('z')
#%%
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')

#%%
intr_rs = rosco.get_camera_intrinsics('/camera/color/camera_info')
#%%
cfg = YamlConfig('practical/cfg/gqcnn_pj_tuned.yaml')
#%%
grasp_policy = CrossEntropyRobustGraspingPolicy(cfg['policy'])
#%%
img = rosco.rgb
d = rosco.temp_filtered_depth(20, blur ='bilateral', mode='median')
#%%
plt.imshow(img)
#%%
plt.imshow(d, cmap='gray')
#%% x1,y1,x2,y2 = bbox
b_w = 120
b_h = 90
img = vision.imcrop(img, [b_w, b_h, np.shape(img)[1] - b_w, np.shape(img)[0] -b_h])
d = vision.imcrop(d, [b_w, b_h, np.shape(d)[1] - b_w, np.shape(d)[0] - b_h])

#%%
int(np.floor(np.min(d)))

#%%
grasp = sampleClient.predictGQCNN(img, d, host='http://ralfi.nat.selfnet.de:5000',height=intr_rs['height'], width=intr_rs['width'], fx=intr_rs['fx'], fy=intr_rs['fy'], cx=intr_rs['cx'], cy=intr_rs['cy'])
#%%
mask = vision.maskDepth(d, 0.8,1.0)
plt.imshow(mask, cmap='gray')
#%%
grasp = sampleClient.predictFCGQCNN(img ,d ,mask, host='http://ralfi.nat.selfnet.de:5000',height=intr_rs['height'], width=intr_rs['width'], fx=intr_rs['fx'], fy=intr_rs['fy'], cx=intr_rs['cx'], cy=intr_rs['cy'])
#%%
#%%
img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.circle(img_,(int(grasp['x']),int(grasp['y'])),2,(255,0,0),3)
plt.imshow(img_)
#%%
plt.imshow(mask)


#%%
plt.imshow(img)
#%%
plt.imshow(d)
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
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.circle(img,(int(grasp.grasp.center.x),int(grasp.grasp.center.y)),2,(255,0,0),3)
#%%
plt.imshow(img)
#%%
grasp.grasp.Grasp2D
