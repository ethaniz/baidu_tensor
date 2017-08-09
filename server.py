# -*- coding:utf-8 -*-
import time
import json
import sys
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

from flask import Flask, request, abort, jsonify
from flask_restful import Resource, Api

from tensor_train import text2vec, vec2text, convert2gray, inference, captcha_size, captcha_len, width, height

app = Flask(__name__)
api = Api(app)

class TensorServer(Resource):
    def post(self):
        start = time.time()

        if not request.json or not 'image' in request.json:
            abort(400)

        image = base64.b64decode(request.json['image'])
        #print(image_path)

        image = Image.open(BytesIO(image))
        img_2 = image.point(lambda x: 255 if x > 160 else 0).convert('RGB')
        img_x = np.array(img_2)
        image = convert2gray(img_x)
        input_x = np.zeros([1, width*height])
        input_x[0, :] = image.flatten()/255

        text_list = persistent_sess.run(max_idx_p, feed_dict={X: input_x, keep_prob: 1.0})

        text = text_list[0].tolist()
        vector = np.zeros(10*11)
        i = 0
        for n in text:
            vector[i*captcha_len + n] = 1
            i += 1
        result = vec2text(vector).strip()

        #json_data = json.dumps({'y': result})
        print("Time spent handling the request: %f" % (time.time() - start))

        return {'y': result}

api.add_resource(TensorServer, "/api/predict")

if __name__ == "__main__":
    print("Loading the model")

    X = tf.placeholder(tf.float32, [None, width*height])
    keep_prob = tf.placeholder(tf.float32)
    out = inference(X, keep_prob)

    predict = tf.reshape(out, [-1, 10, 11])
    max_idx_p = tf.argmax(predict, 2)

    saver = tf.train.Saver()

    persistent_sess = tf.Session()

    if os.path.exists('/Users/ethan/MyCodes/baidu_tensor/models/checkpoint'):
        print("Found last check point, restoring...")
        saver.restore(persistent_sess, '/Users/ethan/MyCodes/baidu_tensor/models/model.ckpt')
    else:
        print("No check point found!")
        sys.exit(1)

    print("Starting the API")
    app.run(debug=True)
