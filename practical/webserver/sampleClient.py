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
from ry import libry as ry
import base64


def predictGQCNN(img, d, host='http://multitask.ddnss.de:5000'):
    url = host + '/gqcnn'
    headers = {'content-type': 'application/json'}
    _ ,img_dec = cv2.imencode('.jpg', img)
    d_dec = memoryview(d)
    response = requests.post(url, json={'rgb':base64.b64encode(img_dec), 'd': base64.b64encode(d_dec)}, headers=headers)
    return json.loads(response)
