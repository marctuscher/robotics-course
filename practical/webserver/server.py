from flask import Flask, request, Response
app = Flask(__name__)
import jsonpickle
import numpy as np
import json
import base64
import sys
import matplotlib.pyplot as plt
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2

@app.route('/')
def hello_world():
    return 'Hello, World!'

    
@app.route('/gqcnn', methods=['POST'])
def gqcnn():
    r = request.get_json()
    nparr_rgb = base64.decodebytes(r['rgb'].encode('ascii'))
    nparr_depth = base64.decodebytes(r['d'].encode('ascii'))
    img_rgb = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)
    img_d = cv2.imdecode(nparr_d, cv2.IMREAD_COLOR)
    response = {'message': 'image received. size={}x{}'.format(img_rgb.shape[1], img_d.shape[0])}
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run()