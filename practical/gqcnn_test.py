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
rosco = RosComm()

#%% 
rospy.init_node('z')
#%%
rosco.subscribe_synced_rgbd('/camera/color/image_raw/', '/camera/depth/image_rect_raw/')
#%%
bridge = CvBridge()

#%%


#%%

rosco.subscribe('/camera/color/camera_info', sensor_msgs.msg.CameraInfo)
l = rosco.output_registry.get('/camera/color/camera_info')
rosco.stop_subscription('/camera/color/camera_info')

#%%
msg = rosco.get_camera_intrinsics('/camera/color/camera_info')

#%%
print(msg)
#%%
intr = {}
intr['frame_id'] = msg.header.frame_id
intr['K'] = msg.K
intr['P'] = msg.P
intr['height'] = msg.height
intr['width'] = msg.width
intr['fx'] = msg.K[0]
intr['fy'] = msg.K[4]
intr['cx'] = msg.K[2]
intr['cy'] = msg.K[5]

#%%
msg.K[0]

#%%
rosco.output_registry

#%% 
type(rosco.output_registry)
#%%
del rosco.subscriber_registry['/camera/color/camera_info']


#%%
rosco.subscriber_registry

#%%
msg = rosco.getLatestMessage('/camera/color/camera_info')
#%%
print(msg)
#%%
img = bridge.imgmsg_to_cv2(msg, "bgr8")
plt.imshow(img)
#%%
cfg = YamlConfig('practical/cfg/gqcnn_pj.yaml')
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
cam_intr = CameraIntrinsics(frame='pcl', fx=intr['fx'], fy=intr['fy'], cx=intr['px'], cy=intr['py'], height=intr['height'], width=intr['width'])
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
