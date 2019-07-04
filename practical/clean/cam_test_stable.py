#%%
# Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
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
import matplotlib.pyplot as plt
from matplotlib import colors
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

# add simulation. Note: if the string argument is not an empty string, a ROS node is started
# and the joint state topics of the real baxter are subscribed. This won't work if you can't connect to Baxter.
# In order to connect to Baxter, uncomment the next 2 lines and set the correct IP address:
os.environ["ROS_MASTER_URI"] = "http://thecount.local:11311/"
os.environ["ROS_IP"] = "129.69.216.153"


#%%
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
#%%
cam = C.addObject(name="cam", parent="base_footprint", shape=ry.ST.sphere, size=[0.01], color=[0,1,0], pos=cam_world_pos, quat=cam_world_rot)
nodeName = "ralf"
#%%
q_home = C.getJointState()
q_zero = q_home.copy() * 0.
B = C.operate(nodeName)
B.sync(C)
C.makeObjectsConvex()
B.sendToReal(False)


#%%
q_home


#%%
def check_target(targetFrame):
    if not targetFrame in C.getFrameNames():
        frame = C.addObject(name=targetFrame, parent="base_footprint" ,shape=ry.ST.sphere, size=[.01], pos=[0,0,0], color=[0.,0.,1.])
    return C.frame(targetFrame)

def close_gripper(close=True):
    B.sync(C)
    q = C.getJointState()
    if close:
        q[-2] = 0.04
    else:
        q[-2] = 0
    B.moveHard(q)

def plan_path(targetPos, angle, targetFrame, gripperFrame, steps, time):
    intermediatePos = cam_world_pos + 0.9 * (targetPos - cam_world_pos)
    intermediate = check_target("intermediate")
    target = check_target(targetFrame)
    rotA = utils.rotz(angle)
    rotC = utils.quat2rotm(cam_rot)
    rotM = rotA @ rotC
    B.sync(C)
    quat = utils.rotm2quat(rotM)
    target.setPosition(targetPos)
    intermediate.setPosition(intermediatePos)
    target.setQuaternion(quat)
    pp = C.komo_path(1, 30, 10, False)
    pp.setConfigurations(C)
    pp.clearObjectives()
    pp.addObjective(type=ry.OT.eq, feature=ry.FS.scalarProductYZ, frames=[gripperFrame, targetFrame], target=[0], time=[.5, 1])
    pp.addObjective(type= ry.OT.eq, feature= ry.FS.scalarProductZZ, frames= [gripperFrame, "cam"], target= [1], time= [.5, 1])
    #pp.addObjective(type= ry.OT.sos,feature= ry.FS.qItself, frames= [], target= q_home, time=[1.])
    pp.addObjective(type= ry.OT.eq, feature= ry.FS.distance, frames= [gripperFrame, 'intermediate'], target= [0], time= [.5])
    pp.addObjective(type= ry.OT.eq, feature= ry.FS.positionDiff, frames= [gripperFrame, targetFrame], target=[0,0,-0.04],time= [1.])
    pp.addObjective(type= ry.OT.sos, feature= ry.FS.qItself, frames= [], order= 1,  time= [0, 1])
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
B.sendToReal(True)
B.moveHard(q_home)
close_gripper(False)


#%%
B.sendToReal(True)


#%%
B.moveHard(q_zero)


#%%
from webserver import sampleClient
import vision
import cv2
from skimage import measure, morphology


#%%
cam = ry.Camera("ralf", "/camera/rgb/image_rect_color", "/camera/depth/image_rect_raw", True)


#%%
img = cam.getRgb()
d = cam.getDepth()


#%%
plt.imshow(d)


#%%
dm, m = vision.maskDepth(d, 0.7,1.4 )
m = cv2.medianBlur(m.astype(np.uint8), 5)
plt.imshow(m)


#%%
segmask = sampleClient.predictMask(d)


#%%
segmask


#%%
image = img.copy()
rcolors = np.random.randint(0, len(STANDARD_COLORS), size=len(segmask["masks"]))
for i, mask in enumerate(segmask["masks"]):
    plt.figure()
    plt.imshow(mask)
    c = colors.to_rgb(STANDARD_COLORS[rcolors[i]])
    #mask = np.bitwise_and(mask.astype(np.bool), m.astype(np.bool))
    colored = np.ones((*mask.shape, 3)) * c
    colored[~mask.astype(np.bool)] = 0
    image = cv2.addWeighted(image.astype(np.uint8),1, (colored * 255).astype(np.uint8), 0.9, 0)


#%%
gx = 0
gy = 0
ind = 0
clicked = False
def mouse_callback(event, x, y, flags, params):

    #right-click event value is 2
    if event == 1:
        global gx, gy, ind, clicked

        #store the coordinates of the right-click event

        gx = x
        gy = y
        #this just verifies that the mouse data is being collected
        #you probably want to remove this later
        for i, mask in enumerate(segmask["masks"]):
            if mask[y, x]:
                ind = i
                clicked = True
                break

#set mouse callback function for window
scale_width = 640 / image.shape[1]
scale_height = 480 / image.shape[0]
scale = min(scale_width, scale_height)
window_width = int(image.shape[1] * scale)
window_height = int(image.shape[0] * scale)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', window_width, window_height)
cv2.setMouseCallback('image', mouse_callback)
while not clicked:
    cv2.imshow('image',image)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print (gx,gy)


#%%
ind


#%%
grasp = sampleClient.predictGQCNN_pj(img, d, host=""http://ralfi.nat.selfnet.de:5000", segmask=m, **chestCamIntrinsics)
vision.plotCircleAroundCenter(img, grasp["x"], grasp["y"])


#%%
grasp = sampleClient.predictFCGQCNN_pj(img, d, segmask["masks"][ind],host=""http://ralfi.nat.selfnet.de:5000", **chestCamIntrinsics)
vision.plotCircleAroundCenter(img, grasp["x"], grasp["y"])


#%%
grasp_p, x, y = vision.getGraspPosition_noIntr(d,grasp["x"], grasp["y"])
x = pinv_chest @ np.array(list(grasp_p) + [1])
steps = 30; time = 10
B.sendToReal(False)
B.moveHard(q_home)
p, pp = plan_path(x, grasp["angle"], "ball", "baxterR", steps, time)


#%%
B.sendToReal(True)
close_gripper(False)
B.move(p, [time/steps * i for i in range(len(p))], False)


#%%
close_gripper(True)


#%%
B.moveHard(q_home)


#%%
B.sendToReal(False)


#%%
B.sendToReal(True)


#%%



