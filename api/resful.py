#!/usr/bin/env python
# -*- coding: utf-8 -*-


import config as cfg
#from common import polygons_to_mask
import os
import sys
import tensorflow as tf
import numpy as np
import json
import base64
import cv2

from flask import Flask,request
from flask_restful import Resource, Api


os.environ["CUDA_VISIBLE_DEVICES"]='0'
ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)

app = Flask(__name__)
api = Api(app)

# def preprocess(image, points, size=cfg.image_size):
#     """
#     Preprocess for test.
#     Args:
#         image: test image
#         points: text polygon
#         size: test image size
#     """
#     height, width = image.shape[:2]
#     mask = polygons_to_mask([np.asarray(points, np.float32)], height, width)
#     x, y, w, h = cv2.boundingRect(mask)
#     mask = np.expand_dims(np.float32(mask), axis=-1)
#     image = image * mask
#     image = image[y:y+h, x:x+w,:]
#
#     new_height, new_width = (size, int(w*size/h)) if h>w else (int(h*size/w), size)
#     image = cv2.resize(image, (new_width, new_height))
#
#     if new_height > new_width:
#         padding_top, padding_down = 0, 0
#         padding_left = (size - new_width)//2
#         padding_right = size - padding_left - new_width
#     else:
#         padding_left, padding_right = 0, 0
#         padding_top = (size - new_height)//2
#         padding_down = size - padding_top - new_height
#
#     image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
#
#     image = image/255.
#     return image

def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]
    ew, eh = expected_size
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    new_img = np.zeros((ew, eh, 3), dtype=np.uint8)

    top = (eh - nh) // 2
    bottom = nh + top
    left = (ew - nw) // 2
    right = nw + left
    new_img[top:bottom, left:right, :] = image
    new_img = new_img / 255.
    #new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img

class TextRecognition(object):
    """
    AttentionOCR with tensorflow pb model.
    """

    def __init__(self, pb_file, seq_len):
        self.pb_file = pb_file
        self.seq_len = seq_len
        self.init_model()

    def init_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.gfile.FastGFile(self.pb_file, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                _ = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)

        self.img_ph = self.sess.graph.get_tensor_by_name('image:0')
        #self.shape = self.sess.graph.get_tensor_by_name("InceptionV4/Shape:0") #第二维必须是1024 (?,1024)
        self.label_ph = self.sess.graph.get_tensor_by_name('label:0')
        self.is_training = self.sess.graph.get_tensor_by_name('is_training:0')
        #self.dropout = self.sess.graph.get_tensor_by_name('dropout:0')
        self.dropout = self.sess.graph.get_tensor_by_name('dropout_keep_prob:0')
        self.preds = self.sess.graph.get_tensor_by_name('sequence_preds:0')
        self.probs = self.sess.graph.get_tensor_by_name('sequence_probs:0')

    def predict(self, image, label_dict, EOS='EOS'):
        results = []
        probabilities = []
        num = len(image)
        # pred_sentences, pred_probs = self.sess.run([self.preds, self.probs],
        #                                            feed_dict={self.is_training: False, self.dropout: 1.0,
        #                                                       self.img_ph: image, self.shape: (num, 1024)})
        labels = np.zeros((num, 33), dtype=np.int32)
        pred_sentences, pred_probs = self.sess.run([self.preds, self.probs],
                                                   feed_dict={self.is_training: False, self.dropout: 1.0,
                                                              self.img_ph: image, self.label_ph: labels})
        num_string = len(pred_sentences)
        for i in range(num_string):
            result = []
            for char in pred_sentences[i]:
                if label_dict[char] == EOS:
                    break
                result.append(label_dict[char])
            if len(result) > 0:
                result = "".join(result)
                results.append(result)
                probabilitie = pred_probs[i][:min(len(result), self.seq_len)]
                probabilities.append(probabilitie.mean())
        return results, probabilities

class charrecognition(Resource):
    def post(self):
        # imagestr = request.form['image']
        temp = request.get_data(as_text=True)
        data = json.loads(temp)
        images = data['image']
        imagebuf = []
        for imagestr in images:
            imagedata_base64 = base64.b64decode(imagestr)
            nparr = np.fromstring(imagedata_base64, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img = cv2_letterbox_image(img,(cfg.image_size,cfg.image_size))
            imagebuf.append(img)
        imagebuf= np.array(imagebuf)
        words, words_acc  = model.predict(imagebuf, cfg.label_dict)
        print(words)
        print(words_acc)
        words_res = []
        nlen = len(words)
        for i in range(nlen):
            temp = {
                "words": words[i],
                "probability": {"average": words_acc[i].__float__()}
            }
            words_res.append(temp)

        result = {"words_result_num": nlen,
                  "words_result": words_res
                  }
        return app.response_class(json.dumps(result), mimetype='application/json')


api.add_resource(charrecognition, '/charrecog')
model = TextRecognition('./pretrain/text_recognition_my.pb', cfg.seq_len+1)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


