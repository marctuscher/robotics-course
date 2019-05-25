from flask import Flask, request, Response, jsonify
app = Flask(__name__)
import numpy as np
import json
import base64
import sys
sys.path.append('.')
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except ValueError:
    pass  # do nothing!
import cv2
from practical.dexnet.network import GQCNNLoader


gqcnn_net = GQCNNLoader()

@app.route('/', methods=['GET'])
def helloWorld():
    return "Hello World!"


@app.route('/gqcnn', methods=['POST'])
def gqcnn():
    img, d = rgbdFromRequest(request)
    state = gqcnn_net.rgbd2state(img, d)
    res = gqcnn_net.predict(state)
    return jsonify(res)

def rgbdFromRequest(request):
    r = request.get_json()
    rgb_str = r['rgb']
    depth_str = r['d']
    rgb_buff = base64.b64decode(rgb_str)
    d_buff = base64.b64decode(depth_str)
    nparr_rgb = np.frombuffer(rgb_buff, np.uint8)
    nparr_depth = np.frombuffer(d_buff, np.float32)
    img_rgb = cv2.imdecode(nparr_rgb, cv2.IMREAD_COLOR)
    #img_d = cv2.imdecode(nparr_depth, cv2.IMREAD_ANYDEPTH)
    return img_rgb, img_d


if __name__ == '__main__':
    app.run(host="0.0.0.0")