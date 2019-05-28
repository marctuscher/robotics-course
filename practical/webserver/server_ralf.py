from flask import Flask, request, Response, jsonify
app = Flask(__name__)
import numpy as np
import json
import base64
import sys
sys.path.append('.')
from practical.webserver.utils import rgbdFromRequest, rgbdSegmaskFromRequest
from practical.dexnet.network import GQCNNLoader, FCGQCNNLoader

gqcnn_net = GQCNNLoader(cfgFile="practical/cfg/gqcnn_pj_tuned.yaml")

fcgqcnn_net = FCGQCNNLoader()

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

if __name__ == '__main__':
    app.run(host="0.0.0.0")