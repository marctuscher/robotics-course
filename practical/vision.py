import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
from visualization import Visualizer2D as vis
from gqcnn.grasping.grasp import Grasp2D
from autolab_core.points import Point
import numpy as np
import matplotlib.pyplot as plt
import imutils
from perception import ColorImage

intrinsics = [538.273, 544.277, 307.502, 249.954]
f = 320./np.tan(0.5 * 60.8 * np.pi/180.)
fVirt = 1./np.tan(0.5 * 90 * np.pi/180.)
#baxterCamIntrinsics = {'fx': f, 'fy': f, 'cx': 320, 'cy': 240, 'width': 640, 'height':480}
baxterCamIntrinsics = {'fx': intrinsics[0], 'fy': intrinsics[1], 'cx': intrinsics[2], 'cy': intrinsics[3], 'width': 640, 'height':480}
virtCamIntrinsics = {'fx': 640 * fVirt, 'fy': 480 * fVirt, 'px': 320, 'py': 240, 'height': 480, 'width': 640}



def maskDepth(d, lower, upper):
    out_d = d.copy()
    mask = np.logical_and(out_d >= lower, out_d <= upper)
    out_d[~mask] = np.nan
    return out_d, mask

    
def findContoursInMask(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = imutils.grab_contours(cnts)
        return cnts

def findBallPosition(img_bgr, d, intrinsics=baxterCamIntrinsics):
    p = findBallInImage(img_bgr)
    if p:
        x, y = p
        dp = calcDepth(d, int(y), int(x))
        print(dp)
        xc = dp * (x - intrinsics['cx'])/intrinsics['fx']
        yc = -dp * (y-intrinsics['cy'])/intrinsics['fy']
        zc = -dp
        return [xc, yc, zc], int(x), int(y)

def getGraspPosition(d, x, y, intrinsics=baxterCamIntrinsics):
    dp = calcDepth(d, int(y), int(x))
    xc = dp * (x - intrinsics['cx'])/intrinsics['fx']
    yc = -dp * (y-intrinsics['cy'])/intrinsics['fy']
    zc = -dp
    return [xc, yc, zc], int(x), int(y)

def getGraspPosition_noIntr(d, x, y, depthVal=None):
    if depthVal is None:
        dp = calcDepth(d, int(y), int(x))
    else:
        dp = depthVal
    xc = dp * x
    yc = dp * y
    zc = dp
    return [xc, yc, zc], int(x), int(y)


def calcDepth(d, u, v):
    #range_x = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    #range_y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    range_x = np.arange(-3, 4, 1)
    range_y = np.arange(-3, 4, 1)
    sum_ranges = len(range_x) * len(range_y)
    cumulated_depth = 0
    for x in range_x:
        for y in range_y:
            if d.shape[0] > u+x and u+x > 0 and v+y > 0 and v+y < d.shape[1]:
                val = d[u + x][v + y]
                if np.isnan(val):
                    sum_ranges -= 1
                else:
                    cumulated_depth += val
            else:
                sum_ranges -= 1
    return cumulated_depth / (sum_ranges)

def findBallInImage(img_bgr):
    mask = greenMask(img_bgr)
    cnts = findContoursInMask(mask)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), r) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        return (x,y)

def plotCircles(img_rgb, circles):
    circles = np.uint16(np.around(circles))
    for i in range(len(circles)):
        u = circles[i,0,0]
        v = circles[i,0,1]
        r = circles[i,0,2]
        # draw the outer circle
        cv2.circle(img_rgb,(u,v),r,(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img_rgb,(u,v),2,(0,0,255),3)
    return img_rgb

def redMask(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0, 100, 100), (5, 255, 255))
    mask2 = cv2.inRange(img_hsv, (160, 100, 100), (179, 255, 255))
    return mask1 | mask2

def greenMask(img_bgr):
    greenLower = (40, 86, 6)
    greenUpper = (64, 255, 255)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask

def imcrop(img, bbox): 
    x1,y1,x2,y2 = bbox
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    if np.shape(img) == (480,640):
        return img[y1:y2, x1:x2]
    else:
        return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    print('Image padding is necessary for the cropping operation')
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2

def temp_filtered_depth(self, cam, numImages=10, blur = 'bilateral', mode= 'median'):
    self.arr = np.zeros([numImages,480,640])
    #blur =  [gaussian', 'bilateral', 'median']
    for i in range(numImages):
        #time.sleep(0.001 * 33.4) # sleep until new photo arrives
        if blur == 'bilateral':
            self.arr[i,:,:] = cv2.bilateralFilter(self.cam.getDepth(), 9,75,75)
        elif blur == 'gaussian':
            self.arr[i,:,:] = cv2.GaussianBlur(self.cam.getDepth(), (3,3), 0)
        elif blur == 'median':
            self.arr[i,:,:] = cv2.medianBlur(self.cam.getDepth(), 5)
        else:
            self.arr[i,:,:] = self.cam.getDepth()

    if mode == 'mean':
        self.fil_depth = np.nanmean(arr, axis=0, keepdims=True)
    elif mode == 'median':
        self.fil_depth = np.nanmedian(arr, axis=0, keepdims=True)
    else:
        self.fil_depth = np.nanmean(arr, axis=0, keepdims=True)
    return self.fil_depth[0,:,:]


def plotCircleAroundCenter(img, x, y, color=(255, 0, 0)):
    img2 = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    cv2.circle(img2,(int(x),int(y)),2,color,3)
    plt.imshow(img2)

class Center():

    def __init__(self, x, y):
        self.x = x
        self.y = y

def plotPjGrasp_older(img, grasp):
    #center = Center(grasp["x"], grasp["y"])
    center = Point(np.array([grasp["x"], grasp["x"]]))
    color_im = ColorImage(img.astype(np.uint8), encoding="bgr8", frame="pcl")
    g = Grasp2D(center, angle=grasp["angle"])
    vis.figure()
    vis.imshow(color_im)
    vis.grasp(g, scale=4.0,jaw_width=2.0, show_center=True, show_axis=True, color=plt.cm.RdYlBu(.1))
    vis.show()

def plotPjGrasp(img, grasp):
    xs = []; ys=[]
    x1 = grasp["x"]
    y1= grasp["y"]
    xs.append(grasp["x"])
    ys.append(grasp["y"])
    x2 = x1 + 50
    x3 = x1 - 50
    m = np.tan(grasp["angle"])
    y2 = m * 50 + y1
    y3 = m * -50 + y1
    xs.append(x2)
    ys.append(y2)
    xs.append(x3)
    ys.append(y3)
    plt.imshow(img)
    plt.scatter([x1], [y1])
    plt.plot(xs, ys) 

def plotPjGrasp_old(img, grasp):
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