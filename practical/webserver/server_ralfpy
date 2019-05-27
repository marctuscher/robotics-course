from flask import Flask, request, Response, jsonify
app = Flask(__name__)
import numpy as np
import json
import base64
import sys
sys.path.append('.')
from practical.webserver.utils import rgbdFromRequest
from practical.dexnet.network import GQCNNLoader


gqcnn_net = GQCNNLoader(cfgFile="practical/cfg/gqcnn_pj_tuned.yaml")

@app.route('/', methods=['GET'])
def helloWorld():
    return "Hello World!"


@app.route('/gqcnn', methods=['POST'])
def gqcnn():
    r = request.get_json()
    img, d = rgbdFromRequest(r)
    state = gqcnn_net.rgbd2state(img, d, intr=r['intr'])
    res = gqcnn_net.predict(state)
    return jsonify(res)


if __name__ == '__main__':
    app.run(host="0.0.0.0")