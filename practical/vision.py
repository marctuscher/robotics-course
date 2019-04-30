import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np
import matplotlib.pyplot as plt



def houghsTransformFromBGR(img_bgr, dp=1, minDist=45, param1=150, param2=13, minRadius=3, maxRadius=50):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(img_hsv, (0, 70, 70), (10, 255, 255))
    mask2 = cv2.inRange(img_hsv, (170, 70, 70), (180, 255, 255))
    mask = mask1 | mask2
    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,dp,minDist, param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    return circles


def plotCircles(img_rgb, circles):
    circles = np.uint16(np.around(circles))
    u = circles[0,0,0]
    v = circles[0,0,1]
    r = circles[0,0,2]
    # draw the outer circle
    cv2.circle(img_rgb,(u,v),r,(0,255,0),2)
    # draw the center of the circle
    cv2.circle(img_rgb,(u,v),2,(0,0,255),3)

    plt.imshow(img_rgb, cmap='gray', vmin=0, vmax=255)