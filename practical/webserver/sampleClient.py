import requests
import json
import numpy as np
import sys
sys.path.append('.')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
import base64
from practical.vision import baxterCamIntrinsics as intr

def predictGQCNN(img, d,host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy'],
    segmask=None
    ):
    url = host + '/gqcnn'
    headers = {'content-type': 'application/json'}
    if encoded:
        _ ,img_dec = cv2.imencode('.png', img)
    else:
        img_dec = memoryview(img)
    d_dec = memoryview(d)
    if segmask is not None:
        s_dec = memoryview(segmask.astype(np.uint8))
        req_dict = {'rgb':base64.b64encode(img_dec), 'd': base64.b64encode(d_dec), 'segmask': base64.b64encode(s_dec) ,"encoded": encoded, "intr": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": height, "width": width,}}
    else:
        req_dict = {'rgb':base64.b64encode(img_dec), 'd': base64.b64encode(d_dec),  "encoded": encoded, "intr": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": height, "width": width,}}
    response = requests.post(url, json=req_dict, headers=headers)
    return response.json()



def predictFCGQCNN(img, d, segmask, host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy']
    ):
    url = host + '/fcgqcnn'
    headers = {'content-type': 'application/json'}
    if encoded:
        _ ,img_dec = cv2.imencode('.png', img)
    else:
        img_dec = memoryview(img)
    d_dec = memoryview(d)
    s_dec = memoryview(segmask.astype(np.uint8))
    req_dict = {'rgb':base64.b64encode(img_dec), 'd': base64.b64encode(d_dec), 'segmask': base64.b64encode(s_dec) ,"encoded": encoded, "intr": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": height, "width": width,}}
    response = requests.post(url, json=req_dict, headers=headers)
    return response.json()