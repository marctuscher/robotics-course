import sys
sys.path.append('../')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import numpy as np


def computeHoughsTransform(rgb, depth):
    blurred = cv2.GaussianBlur(rgb,(3,3), 0)
    circles = cv2.HoughCircles(rgb, cv2.HOUGH_GRADIENT, 1, rgb.shape[0]/8, param1=10, param2=5, minRadius=0, maxRadius=0)
    return circles