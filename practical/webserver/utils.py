import numpy as np
import base64
import sys
sys.path.append('.')
import cv2
import matplotlib.pyplot as plt


def rgbdFromRequest(r):
    img_rgb = rgbFromRequest(r)
    depth_str = r['d']
    intr = r['intr']
    d_buff = base64.b64decode(depth_str)
    nparr_depth = np.frombuffer(d_buff, np.float32)
    nparr_depth = np.reshape(nparr_depth, (intr["height"], intr["width"]))
    return img_rgb, nparr_depth


def rgbdSegmaskFromRequest(r):
    img, d = rgbdFromRequest(r)
    intr = r['intr']
    s_str = r['segmask']
    s_buff = base64.b64decode(s_str)
    nparr_s = np.frombuffer(s_buff, np.uint8)
    nparr_s = np.reshape(nparr_s, (intr['height'], intr['width']))
    return img, d, nparr_s


def rgbFromRequest(r):
    rgb_str = r['rgb']
    encoded = r['encoded']
    intr = r['intr']
    rgb_buff = base64.b64decode(rgb_str)
    nparr_rgb = np.frombuffer(rgb_buff, np.uint8)
    if encoded:
        img_rgb = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)
    else:
        img_rgb = np.reshape(nparr_rgb, (intr["height"], intr["width"], 3))
    return img_rgb



def plotImage(img, filename="image.png"):
    plt.figure()
    plt.imshow(img)
    plt.savefig(filename)