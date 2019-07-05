#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'practical/clean'))
	print(os.getcwd())
except:
	pass

#%%
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')
get_ipython().run_line_magic('matplotlib', 'inline')
  
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import colors
from visualization import Visualizer2D as vis
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


#%%
# the following cells will use the rai framework for robot control, simulation and motion planning
# https://github.com/MarcToussaint/rai
import sys
import os 
print(os.getcwd())
# add the folder where libry.so is located to the path. Otherwise the import will crash.
sys.path.append('../../ry/')
sys.path.append('../../')
sys.path.append('../')
import libry as ry
import time
import gc
import os
from collections import defaultdict
import utils

from webserver import sampleClient
import vision
import cv2

# add simulation. Note: if the string argument is not an empty string, a ROS node is started
# and the joint state topics of the real baxter are subscribed. This won't work if you can't connect to Baxter.
# In order to connect to Baxter, uncomment the next 2 lines and set the correct IP address:
os.environ["ROS_MASTER_URI"] = "http://thecount.local:11311/"
os.environ["ROS_IP"] = "129.69.216.153"
"""
total ERROR = 0.00241405
total ERROR after radius correction = 0.00221046
*** total Pinv:
[0.00180045, 5.51994e-06, -0.569533, -0.0330757,
 -1.82321e-06, -0.00133149, 1.00136, 0.125005,
 5.08217e-05, -0.00117336, -0.439092, 1.55487]
*** camera intrinsics K:
[555.197, -8.21031, -334.467,
 0, -563.526, -271.392,
 0, 0, -1.02162]
*** camera world pos: [-0.0330757, 0.125005, 1.55487]
*** camera world rot: [0.935411, 0.35328, -0.0133783, 0.00451155]
"""
cam_world_pos= [-0.0330757, 0.125005, 1.55487]
cam_world_rot= [0.935411, 0.35328, -0.0133783, 0.00451155]
chestCamIntrinsics = {'fx': 555.197 , 
                       'fy':  -563.526, 
                       'cx': -334.467, 
                       'cy': -271.392, 'width': 640, 'height':480}
cam_rot = [0.935411, 0.35328, -0.0133783, 0.00451155]
K = np.array([
[555.197, -8.21031, -334.467],
 [0, -563.526, -271.392],
 [0, 0, -1.02162]
])
pinv_chest = np.array([[0.00180045, 5.51994e-06, -0.569533, -0.0330757],
  [-1.82321e-06, -0.00133149, 1.00136, 0.125005],
  [5.08217e-05, -0.00117336, -0.439092, 1.55487]])


#%%
# clear views, config and operate by setting shared pointers to 0. Otherwise the notebook has to be restarted,
# which is pretty annoying.
C = 0
v = 0
B = 0
gc.collect()
    
# initialize config
C = ry.Config()
v = C.view()
C.clear()
C.addFile('../../rai-robotModels/baxter/baxter_new.g')
cam = C.addObject(name="cam", parent="base_footprint", shape=ry.ST.sphere, size=[0.01], color=[0,1,0], pos=cam_world_pos, quat=cam_world_rot)
nodeName = "ralf"

q_home = C.getJointState()
q_zero = q_home.copy() * 0.
B = C.operate(nodeName)
B.sync(C)
C.makeObjectsConvex()
B.sendToReal(False)
cam = ry.Camera(nodeName, "/camera/rgb/image_rect_color", "/camera/depth/image_rect_raw", True)


#%%
def check_target(targetFrame):
    if not targetFrame in C.getFrameNames():
        frame = C.addObject(name=targetFrame, parent="base_footprint" ,shape=ry.ST.sphere, size=[.01], pos=[0,0,0], color=[0.,0.,1.])
    return C.frame(targetFrame)

def close_pj(close=True):
    B.sync(C)
    q = C.getJointState()
    if close:
        q[-2] = 0.04
    else:
        q[-2] = 0
    B.moveHard(q)
    
def close_suction(close=True):
    B.sync(C)
    q = C.getJointState()
    if close:
        q[-1] = 0.1
    else:
        q[-2] = 0
    B.moveHard(q)

def plan_path_pj(targetPos, angle, targetFrame, gripperFrame, steps, time):
    intermediatePos = [targetPos[0], targetPos[1], targetPos[2] + 0.1]
    intermediate = check_target("intermediate")
    target = check_target(targetFrame)
    axis = pinv_chest @ np.array([0, 0, 1, 1])
    r = R.from_quat([axis[0]*np.sin(-angle/2), axis[1]*np.sin(-angle/2), axis[2]*np.sin(-angle/2), np.cos(angle/2)])
    angle_z = (r.as_euler('zyx'))[0]
    rotM = utils.rotz(angle_z)
    print("anglez:", angle_z)
    quat = utils.rotm2quat(rotM)
    B.sync(C)
    target.setPosition(targetPos)
    target.setQuaternion(quat)
    intermediate.setPosition(intermediatePos)
    pp = C.komo_path(1, steps, time, False)
    pp.setConfigurations(C)
    pp.clearObjectives()
    pp.addObjective(type= ry.OT.eq, feature= ry.FS.scalarProductZZ, frames=[gripperFrame, targetFrame], target= [1], time= [.5, 1])
    pp.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[gripperFrame, targetFrame], target=[0], time=[.5, 1])
    pp.addObjective(type= ry.OT.eq, feature= ry.FS.distance, frames= [gripperFrame, 'intermediate'], target= [0], time= [.5])
    pp.addObjective(type= ry.OT.eq, feature= ry.FS.positionDiff, frames= [gripperFrame, targetFrame], target=[0,0,0],time= [1.])
    pp.addObjective(type= ry.OT.sos, feature= ry.FS.qItself,order=1, frames= [], target=[],time= [])
    pp.optimize(False)
    t = pp.getT()
    path = []
    for i in range(t):
        frames = pp.getConfiguration(i)
        C.setFrameState(frames)
        q = C.getJointState()
        q[-2] = 0
        q[-1] = 0
        path += [q]
    return path, pp

def plan_path_suction(targetPos, targetFrame, gripperFrame, steps, time):
    f = check_target(targetFrame)
    f.setPosition(targetPos)
    B.sync(C)
    pp = C.komo_path(1, steps, time, False)
    pp.setConfigurations(C)
    pp.clearObjectives()
    pp.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=[targetFrame, gripperFrame], target=[0, 0, -0.1], scale=[1], time=[.5])
    pp.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductZZ, frames=[targetFrame, gripperFrame], target=[1], scale=[1], time=[.5, 1.])
    pp.addObjective(type=ry.OT.eq, feature=ry.FS.positionDiff, frames=[targetFrame, gripperFrame], target=[0, 0, 0.02],scale=[1], time=[ 1.])
    pp.addObjective(type= ry.OT.sos, feature= ry.FS.qItself,order=1, frames= [], target=[],time= [], scale=[1])
    pp.optimize(False)
    t = pp.getT()
    path = []
    for i in range(t):
        frames = pp.getConfiguration(i)
        C.setFrameState(frames)
        q = C.getJointState()
        q[-2] = 0
        q[-1] = 0
        path += [q]
    return path, pp


#%%
class GraspPlanner():
    
    def getImageAndMask(self):
        self.img = cam.getRgb()
        self.d = cam.getDepth()
        dm, m = vision.maskDepth(self.d, 0.7,1.4 )
        self.m = cv2.medianBlur(m.astype(np.uint8), 5)
        self.segmask = sampleClient.predictMask(self.d)
        print("sampled mask")
        image = self.img.copy()
        rcolors = np.random.randint(0, len(STANDARD_COLORS), size=len(self.segmask["masks"]))
        for i, mask in enumerate(self.segmask["masks"]):
            c = colors.to_rgb(STANDARD_COLORS[rcolors[i]])
            mask = np.bitwise_and(mask.astype(np.bool), self.m.astype(np.bool))
            colored = np.ones((*mask.shape, 3)) * c
            colored[~mask.astype(np.bool)] = 0
            image = cv2.addWeighted(image.astype(np.uint8),1, (colored * 255).astype(np.uint8), 0.9, 0)
        self.image=image
    
    def getClick(self):
        self.ind = 0
        self.clicked = False
        def mouse_callback(event, x, y, flags, params):
            if event == 1:
                print("event")
                for i, mask in enumerate(self.segmask["masks"]):
                    if mask[y, x]:
                        self.ind = i
                        print(self.ind)
                        self.clicked = True
                        break

        #set mouse callback function for window
        scale_width = 640 / self.image.shape[1]
        scale_height = 480 / self.image.shape[0]
        scale = min(scale_width, scale_height)
        window_width = int(self.image.shape[1] * scale)
        window_height = int(self.image.shape[0] * scale)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', window_width, window_height)
        cv2.setMouseCallback('image', mouse_callback)
        print("click")
        while not self.clicked:
            cv2.imshow('image',self.image)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
            elif k == ord('a'):
                print (gx,gy)
                
    def getGrasp(self,grasp_type=None, steps=10, times=10):
        self.steps = steps
        self.times = times
        B.moveHard(q_home)
        if grasp_type is None: 
            self.pj = sampleClient.predictFCGQCNN_pj(self.img, self.d, self.segmask["masks"][self.ind],host="http://multitask.ddnss.de:5000", **chestCamIntrinsics)
            self.suction = sampleClient.predictFCGQCNN_suction(self.img,self.d,host="http://multitask.ddnss.de:5000", segmask=self.segmask["masks"][self.ind], **chestCamIntrinsics)
            print("sample grasp")
            print("suction: {}, pj: {}".format(self.suction["q"], self.pj["q"]))
            if self.suction["q"] > self.pj["q"]:
                self.grasp_type="suction"
                self.grasp = self.suction
            else:
                self.grasp_type = "pj"
                self.grasp = self.pj
        elif grasp_type == "suction":
            self.suction = sampleClient.predictFCGQCNN_suction(self.img, self.d,host="http://multitask.ddnss.de:5000", segmask=self.segmask["masks"][self.ind], **chestCamIntrinsics)
            self.grasp = self.suction
        else:
            self.pj = sampleClient.predictFCGQCNN_pj(self.img, self.d,self.segmask["masks"][self.ind], host="http://multitask.ddnss.de:5000",**chestCamIntrinsics)
            self.grasp = self.pj
        grasp_p, x, y = vision.getGraspPosition_noIntr(self.d,self.grasp["x"], self.grasp["y"], depthVal=self.grasp["depth"])
        x = pinv_chest @ np.array(list(grasp_p) + [1])
        if self.grasp_type == "suction":
            self.p, pp = plan_path_suction(x, "suction", "baxterL", steps, times)
        else:
            print("pj")
            self.p, pp = plan_path_pj(x, self.grasp["angle"], "pj", "baxterR", steps, times)
        print("done")
        
    def exectuteGrasp(self, bin_pos=q_home.copy()):
        B.move(self.p, [self.times/self.steps * i for i in range(len(self.p))], False)
        B.wait()
        if self.grasp["type"] == "suction":
            close_suction(True)
            bin_pos[-1] = 0.1
            B.move([bin_pos], [5], False)
            B.wait()
            close_suction(False)
        else:
            close_pj(True)
            B.move([bin_pos], [5], False)
            B.wait()
            close_pj(False)
            


#%%
planner = GraspPlanner()


#%%
planner.getImageAndMask()
planner.getClick()


#%%
planner.getGrasp()


#%%
B.sendToReal(True)
planner.exectuteGrasp()


#%%
close_suction(True)


#%%
B.moveHard(q_home)


#%%
q_copy = C.getJointState()


#%%
x = grasp["x"]; y=grasp["y"]
data = np.array([x, y])
plt.scatter(x,y)
axis=np.array([np.cos(grasp["angle"]), np.cos(grasp["angle"])])
width_px = 10
g1= data - (width_px/2) * axis
g2= data + (width_px/2) * axis
g1p = g1 - 2* 5
g2p = g2 + 2* 5
plt.plot([[g1[0], g2[0]], [g1[1], g2[1]]],linewidth=10)
# direction of jaw line
jaw_dir = 5 * np.array([axis[1], -axis[0]])

# length of arrow
alpha = 2*(5 - 2)

# plot first jaw
g1_line = np.c_[g1p, g1 - 2*2*axis].T
plt.arrow(g1p[0], g1p[1], alpha*axis[0], alpha*axis[1])
jaw_line1 = np.c_[g1 + jaw_dir, g1 - jaw_dir].T

plt.plot(jaw_line1[:,0], jaw_line1[:,1]) 

# plot second jaw
g2_line = np.c_[g2p, g2 + 10*axis].T
plt.arrow(g2p[0], g2p[1], -alpha*axis[0], -alpha*axis[1])
jaw_line2 = np.c_[g2 + jaw_dir, g2 - jaw_dir].T
plt.plot(jaw_line2[:,0], jaw_line2[:,1])
plt.imshow(img)


#%%



