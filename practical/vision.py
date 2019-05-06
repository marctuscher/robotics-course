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


def findContoursInMask(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts


def greenMask(img_bgr):
    greenLower = (29, 86, 6)
    greenUpper = (64, 255, 255)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def findBallPosition(img_bgr, d, intr):
    x, y = findBallInImage(img_bgr)
    dp = d[int(x)][int(y)]
    xc = dp * (x - intr['px'])/intr['fx']
    yc = -dp * (y-intr['py'])/intr['fy']
    zc = -dp
    return [xc, yc, zc]


def findBallInImage(img_bgr):
    mask = greenMask(img_bgr)
    plt.imshow(mask)
    cnts = findContoursInMask(mask)
    center = None
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), r) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center=(int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"]))
        cv2.circle(img_bgr, (int(x), int(y)), int(r),(0, 255, 255), 2)
        cv2.circle(img_bgr, center, 5, (0, 0, 255), -1)
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
