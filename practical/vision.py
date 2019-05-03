import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np
import matplotlib.pyplot as plt



def findContoursInMask(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    return cnts

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
