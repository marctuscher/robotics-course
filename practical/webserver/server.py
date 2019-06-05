from flask import Flask, request, Response, jsonify
app = Flask(__name__)
import numpy as np
import json
import base64
import sys
sys.path.append('.')
from practical.webserver.utils import rgbdFromRequest, rgbdSegmaskFromRequest, plotImage, rgbFromRequest, dFromRequest
from practical.dexnet.network import GQCNNLoader, FCGQCNNLoader
from practical.dexnet.maskNet import MaskLoader


gqcnn_net = GQCNNLoader()
fcgqcnn_net = FCGQCNNLoader()
mask_net = MaskLoader()

@app.route('/', methods=['GET'])
def helloWorld():
    return "Hello World!"


@app.route('/gqcnn', methods=['POST'])
def gqcnn():
    r = request.get_json()
    if 'segmask' in r:
        img, d, s = rgbdSegmaskFromRequest(r)
        state = gqcnn_net.rgbd2state(img, d, segmask=s, intr=r['intr'])
        res = gqcnn_net.predict(state)
    else:
        img, d = rgbdFromRequest(r)
        #plotImage(d)
        state = gqcnn_net.rgbd2state(img, d, intr=r['intr'])
        res = gqcnn_net.predict(state)
    return jsonify(res)

@app.route('/fcgqcnn', methods=['POST'])
def fcgqcnn():
    r = request.get_json()
    img, d, s = rgbdSegmaskFromRequest(r)
    state = fcgqcnn_net.rgbd2state(img, d, segmask=s, intr=r['intr'])
    res = fcgqcnn_net.predict(state)
    return jsonify(res)

@app.route('/mask', methods=['POST'])
def mask():
    r = request.get_json()
    d = dFromRequest(r)
    res = mask_net.predict(d)
    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0")