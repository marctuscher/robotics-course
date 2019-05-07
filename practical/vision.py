import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils


f = 1./np.tan(0.5 * 60.8 * np.pi/180.)
f = f * 320.
intrinsics= {'fx': f, 'fy': f, 'px': 320, 'py': 240}
baxterCamIntrinsics = {'fx': f, 'fy': f, 'px': 320, 'py': 240}

def findContoursInMask(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cnts = imutils.grab_contours(cnts)
        return cnts

def findBallPosition(img_bgr, d, intrinsics=baxterCamIntrinsics):
    x, y = findBallInImage(img_bgr)
    dp = d[int(x)][int(y)]
    xc = dp * (x - intrinsics['px'])/intrinsics['fx']
    yc = -dp * (y-intrinsics['py'])/intrinsics['fy']
    zc = -dp
    return [xc, yc, zc]

def findBallInImage(img_bgr):
    mask = greenMask(img_bgr)
    cnts = findContoursInMask(mask)
    center = None
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