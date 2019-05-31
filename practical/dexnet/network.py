from autolab_core import YamlConfig
from gqcnn.grasping import Grasp2D, SuctionPoint2D, CrossEntropyRobustGraspingPolicy, RgbdImageState
from gqcnn.utils import GripperMode, NoValidGraspsException
from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
from visualization import Visualizer2D as vis
from practical.vision import baxterCamIntrinsics
import numpy as np
from gqcnn.grasping import FullyConvolutionalGraspingPolicyParallelJaw



class GQCNNLoader():


    def __init__(self, cfgFile="practical/cfg/gqcnn_pj_serv.yaml"):
        self.cfg = YamlConfig(cfgFile)
        self.graspPolicy = CrossEntropyRobustGraspingPolicy(self.cfg['policy'])

    def rgbd2state(self, img, d, segmask=None, frame='pcl',rgbEncoding='bgr8', intr=baxterCamIntrinsics):
        cam_intr = CameraIntrinsics(frame=frame, fx=intr['fx'], fy=intr['fy'], cx=intr['cx'], cy=intr['cy'], height=intr['height'], width=intr['width'])
        color_im = ColorImage(img.astype(np.uint8), encoding=rgbEncoding, frame=frame)
        depth_im = DepthImage(d.astype(np.float32), frame=frame)
        color_im = color_im.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        depth_im = depth_im.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        if segmask is not None:
            mask_img = BinaryImage(segmask.astype(np.uint8), frame='pcl')
            rgbd_state = RgbdImageState(rgbd_im, cam_intr, segmask=mask_img)
        else:
            rgbd_state = RgbdImageState(rgbd_im, cam_intr)
        return rgbd_state
    
    def predict(self, rgbdState):
        res = self.graspPolicy(rgbdState)
        pred = {
            "x": res.grasp.center.x,
            "y": res.grasp.center.y,
            "angle": res.grasp.angle,
            "q": res.q_value,
            "approachAxis": [int(res.grasp.approach_axis[0]), int(res.grasp.approach_axis[1]), int(res.grasp.approach_axis[2])],
            "axis": [res.grasp.axis[0], res.grasp.axis[0]],
            "width": res.grasp.width,
            "depth": res.grasp.depth,
            "approachAngle": res.grasp.approach_angle
        }
        return pred



class FCGQCNNLoader():

    def __init__(self, cfgFile="practical/cfg/fcgqcnn_pj.yaml"):
        self.cfg = YamlConfig(cfgFile)
        self.graspPolicy = FullyConvolutionalGraspingPolicyParallelJaw(self.cfg['policy'])

    def rgbd2state(self, img, d, segmask,frame='pcl',rgbEncoding='bgr8', intr=baxterCamIntrinsics):
        cam_intr = CameraIntrinsics(frame=frame, fx=intr['fx'], fy=intr['fy'], cx=intr['cx'], cy=intr['cy'], height=intr['height'], width=intr['width'])
        color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame=frame)
        depth_im = DepthImage(d.astype(np.float32), frame=frame)
        color_im = color_im.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        depth_im = depth_im.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        mask_img = BinaryImage(segmask.astype(np.uint8), frame='pcl')
        rgbd_state = RgbdImageState(rgbd_im, cam_intr, segmask=mask_img)
        return rgbd_state
    
    def predict(self, rgbdState):
        res = self.graspPolicy(rgbdState)
        pred = {
            "x": res.grasp.center.x,
            "y": res.grasp.center.y,
            "angle": res.grasp.angle,
            "q": res.q_value,
            "approachAxis": [int(res.grasp.approach_axis[0]), int(res.grasp.approach_axis[1]), int(res.grasp.approach_axis[2])],
            "axis": [res.grasp.axis[0], res.grasp.axis[0]],
            "width": res.grasp.width,
            "depth": res.grasp.depth,
            "approachAngle": res.grasp.approach_angle
        }
        return pred

