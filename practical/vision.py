import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np


def computeHoughsTransform(rgb, depth):
    graying = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(graying,(3,3), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=0)
    print(circles)
    return circles