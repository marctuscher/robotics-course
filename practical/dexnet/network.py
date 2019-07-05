from autolab_core import YamlConfig
from gqcnn.grasping import Grasp2D, SuctionPoint2D, CrossEntropyRobustGraspingPolicy, RgbdImageState
from gqcnn.utils import GripperMode, NoValidGraspsException
from perception import CameraIntrinsics, ColorImage, DepthImage, BinaryImage, RgbdImage
from visualization import Visualizer2D as vis
from practical.vision import baxterCamIntrinsics
import numpy as np
from gqcnn.grasping import FullyConvolutionalGraspingPolicyParallelJaw, FullyConvolutionalGraspingPolicySuction

class DexnetLoader():
    def __init__(self, cfgFile):
        self.cfg = YamlConfig(cfgFile)
        self.metricType = self.cfg['policy']['metric']['type']
        if self.metricType == "fcgqcnn":
            self.type = self.cfg['policy']['type']
            if self.type == "fully_conv_pj":
                self.graspPolicy = FullyConvolutionalGraspingPolicyParallelJaw(self.cfg['policy'])
            else:
                self.graspPolicy = FullyConvolutionalGraspingPolicySuction(self.cfg['policy'])
        else: 
            self.graspPolicy = CrossEntropyRobustGraspingPolicy(self.cfg['policy'])

    def rgbd2state(self, img, d, segmask=None, frame='pcl', rgbEncoding='bgr8', intr=baxterCamIntrinsics, obj_mask=False):
        cam_intr = CameraIntrinsics(frame=frame, fx=intr['fx'], fy=intr['fy'], cx=intr['cx'], cy=intr['cy'], height=intr['height'], width=intr['width'])
        color_im = ColorImage(img.astype(np.uint8), encoding=rgbEncoding, frame=frame)
        depth_im = DepthImage(d.astype(np.float32), frame=frame)
        mask_img = None
        if segmask is not None:
            mask_img = BinaryImage(segmask.astype(np.uint8) * 255, frame='pcl')
            valid_pxls = depth_im.invalid_pixel_mask().inverse()
            mask_img = mask_img.mask_binary(valid_pxls)
        depth_im = depth_im.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        color_im = color_im.inpaint(rescale_factor=self.cfg['inpaint_rescale_factor'])
        rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        rgbd_state = RgbdImageState(rgbd_im, cam_intr, segmask=mask_img)
        return rgbd_state

    def predict(self, rgbdState):
        res = self.graspPolicy(rgbdState)
        if isinstance(res.grasp, SuctionPoint2D):
            pred = {
                "x": float(res.grasp.center.x),
                "y": float(res.grasp.center.y),
                "angle": float(res.grasp.angle),
                "q": float(res.q_value),
                "approachAxis": [float(res.grasp.approach_axis[0]), float(res.grasp.approach_axis[1]), float(res.grasp.approach_axis[2])],
                "axis": [res.grasp.axis[0], res.grasp.axis[0]],
                "depth": float(res.grasp.depth),
                "approachAngle": float(res.grasp.approach_angle),
                "type": "suction"
            }
        else:
            pred = {
                "x": res.grasp.center.x,
                "y": res.grasp.center.y,
                "angle": res.grasp.angle,
                "q": res.q_value,
                "approachAxis": [float(res.grasp.approach_axis[0]), float(res.grasp.approach_axis[1]), float(res.grasp.approach_axis[2])],
                "axis": [res.grasp.axis[0], res.grasp.axis[0]],
                "width": res.grasp.width,
                "depth": res.grasp.depth,
                "approachAngle": res.grasp.approach_angle,
                "type": "pj"
            }
        return pred

