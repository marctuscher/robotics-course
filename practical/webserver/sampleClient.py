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



cam = ry.Camera("test","/camera/rgb/image_rect_color", "/camera/depth_registered/image_raw")

addr = 'http://localhost:5000'
test_url = addr + '/gqcnn'

# prepare headers for http request
content_type = 'application/json'
headers = {'content-type': content_type}
img = cam.getRgb()
d = cam.getDepth()

_ ,img_dec = cv2.imencode('.jpg', img)
_, d_dec = cv2.imencode('.jpg', d)

response = requests.post(test_url, json={'rgb':base64.b64encode(img_dec), 'd': base64.b64encode(d_dec), 'width': 640, 'height':480}, headers=headers)
# decode response
print (json.loads(response.text))