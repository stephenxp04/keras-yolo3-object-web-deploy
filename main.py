from flask import Flask, render_template, request, Response, send_from_directory, redirect, url_for, jsonify
from flask_sslify import SSLify
import os
from PIL import Image
import json
import base64
import cv2
import numpy as np
#from src import classifier
import time
import subprocess as sp
import requests
from OpenSSL import SSL
from yolo import YOLO

app = Flask(__name__)
sslify = SSLify(app)
yolo_class = YOLO()

#for CORS
#@app.before_request
#def before_request():
#    if not request.url.startswith('http://'):
#	url = request.url.replace('http://', 'https://', 1)
#	code = 301
#	return redirect(url, code=code)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST')
    return response


@app.route('/')
def index():
    return Response(os.getcwd())


@app.route('/object')
def remote():
    return Response(open('/home/m360/MachineLearning/keras-yolo3/recognition.html').read(), mimetype="text/html")


@app.route('/detection', methods=['POST'])
def face_recognition():
    try:
        global yolo_class

        if yolo_class is None:
            yolo_class = YOLO()

        if request.method == 'POST':
            print('POST /detection success!')

        # web_face_recognition.debug()
        image_file = json.loads(request.data)
        img = base64.b64decode(image_file['data'])
        #imgdata = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        boxes = yolo_class.detect_image(img)
        return boxes

    except Exception as e:
        print('Detection failed : %s' % e)
        return e

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    #context = SSL.Context(SSL.SSLv23_METHOD)
    #context.use_privatekey_file('/home/m360/ssl.key')
    #context.use_certificate_file('/home/m360/ssl.cert')
    global yolo_class

    #if yolo_class is None:
    #    yolo_class = YOLO()
    app.run(host='0.0.0.0', port='8080', threaded=True)

