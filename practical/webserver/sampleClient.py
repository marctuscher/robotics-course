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

def predictGQCNN_pj(img, d,host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy'],
    segmask=None
    ):
    url = host + '/gqcnnpj'
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



def predictFCGQCNN_pj(img, d, segmask, host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy']
    ):
    url = host + '/fcgqcnnpj'
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

def predictGQCNN_suction(img, d,host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy'],
    segmask=None
    ):
    url = host + '/gqcnnsuction'
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


def predictFCGQCNN_suction(img, d, segmask, host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy']
    ):
    url = host + '/fcgqcnnsuction'
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

def predictMask(d, host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy']
    ):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    url = host + '/mask'
    headers = {'content-type': 'application/json'}
    d_dec = memoryview(d.astype(np.float32))
    req_dict = {'d': base64.b64encode(d_dec), "intr": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": height, "width": width,}}
    response = requests.post(url, json=req_dict, headers=headers)
    res_json = response.json()
    for key in res_json.keys():
        if isinstance(res_json[key], list):
            res_json[key] = np.array(res_json[key], dtype=np.float32)
    return res_json

def predictRgb(img, host='http://multitask.ddnss.de:5000', width=640, height=480, encoded=False,
    fx=intr['fx'],
    fy=intr['fy'],
    cx=intr['cx'],
    cy=intr['cy']
    ):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    url = host + '/maskRgb'
    headers = {'content-type': 'application/json'}
    if encoded:
        _ ,img_dec = cv2.imencode('.png', img)
    else:
        img_dec = memoryview(img)
    req_dict = {'rgb': base64.b64encode(img_dec), "encoded": encoded,"intr": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "height": height, "width": width,}}
    response = requests.post(url, json=req_dict, headers=headers)
    res_json = response.json()
    for key in res_json.keys():
        if isinstance(res_json[key], list):
            res_json[key] = np.array(res_json[key], dtype=np.float32)
            if key == "masks":
                res_json[key] = np.transpose(res_json[key], axes=[2, 0, 1])
    return res_json

