from flask import Flask, request, Response, jsonify
app = Flask(__name__)
import numpy as np
import json
import base64
import sys
sys.path.append('.')
from practical.webserver.utils import rgbdFromRequest, rgbdSegmaskFromRequest, plotImage, rgbFromRequest, dFromRequest
from practical.dexnet.network import DexnetLoader
from practical.dexnet.maskNet import MaskLoader


mask_net = MaskLoader()
gqcnnpj_net = DexnetLoader('practical/cfg/gqcnn_pj_tuned.yaml')
gqcnnsuction_net = DexnetLoader('practical/cfg/gqcnn_suction.yaml')
fcgqcnnpj_net = DexnetLoader('practical/cfg/fcgqcnn_pj.yaml')
fcgqcnnsuction_net = DexnetLoader('practical/cfg/fc_gqcnn_suction.yaml')

@app.route('/', methods=['GET'])
def helloWorld():
    return "Hello World!"


@app.route('/gqcnnpj', methods=['POST'])
def gqcnnpj():
    r = request.get_json()
    if 'segmask' in r:
        img, d, s = rgbdSegmaskFromRequest(r)
        state = gqcnnpj_net.rgbd2state(img, d, segmask=s, intr=r['intr'])
        res = gqcnnpj_net.predict(state)
    else:
        img, d = rgbdFromRequest(r)
        state = gqcnnpj_net.rgbd2state(img, d, intr=r['intr'])
        res = gqcnnpj_net.predict(state)
    return jsonify(res)

@app.route('/gqcnnsuction', methods=['POST'])
def gqcnnsuction():
    r = request.get_json()
    if 'segmask' in r:
        img, d, s = rgbdSegmaskFromRequest(r)
        state = gqcnnsuction_net.rgbd2state(img, d, segmask=s, intr=r['intr'])
        res = gqcnnsuction_net.predict(state)
    else:
        img, d = rgbdFromRequest(r)
        state = gqcnnsuction_net.rgbd2state(img, d, intr=r['intr'])
        res = gqcnnsuction_net.predict(state)
    return jsonify(res)

@app.route('/fcgqcnnpj', methods=['POST'])
def fcgqcnnpj():
    r = request.get_json()
    img, d, s = rgbdSegmaskFromRequest(r)
    state = fcgqcnnpj_net.rgbd2state(img, d, segmask=s, intr=r['intr'])
    res = fcgqcnnpj_net.predict(state)
    return jsonify(res)

@app.route('/fcgqcnnsuction', methods=['POST'])
def fcgqcnnsuction():
    r = request.get_json()
    img, d, s = rgbdSegmaskFromRequest(r)
    state = fcgqcnnsuction_net.rgbd2state(img, d, segmask=s, intr=r['intr'])
    res = fcgqcnnsuction_net.predict(state)
    return jsonify(res)

@app.route('/mask', methods=['POST'])
def mask():
    r = request.get_json()
    d = dFromRequest(r)
    plotImage(d)
    res = mask_net.predict(d)
    return jsonify(res)

@app.route('/maskRgb', methods=['POST'])
def maskRgb():
    r = request.get_json()
    img = rgbFromRequest(r)
    res = mask_net.predictRgb(img)
    return jsonify(res)

if __name__ == '__main__':
    app.run(host="0.0.0.0")