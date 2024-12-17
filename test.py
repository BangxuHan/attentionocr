#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt
from google.protobuf import text_format

import config as cfg
import tensorflow as tf
from common import polygons_to_mask

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
                #tf.train.write_graph(graph_def, './tttt', 'protobuf4.pbtxt', as_text=True)

                # tf.import_graph_def(
                #     graph_def,
                #     input_map=None,
                #     return_elements=None,
                #     name="",
                #     op_dict=None,
                #     producer_op_list=None
                # )
                # sess = tf.Session(graph=self.graph)
                # saver = tf.train.Saver()
                # save_path = saver.save(sess, "tttt/model.ckpt")

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.1

        self.sess = tf.Session(graph=self.graph, config=config)
        # saver = tf.train.Saver()
        # save_path = saver.save(self.sess, "model.ckpt")
        
        self.img_ph = self.sess.graph.get_tensor_by_name('image:0')
        #self.Shape = self.sess.graph.get_tensor_by_name("InceptionV4/Shape:0")
        self.Shape = self.sess.graph.get_tensor_by_name("InceptionV4/Shape:0")

        #self.img_ph
        #self.img_ph = tf.placeholder(tf.float32, [1, 299, 299, 3],name='image:0')
        #self.label_ph = self.sess.graph.get_tensor_by_name('label:0')
        self.is_training = self.sess.graph.get_tensor_by_name('is_training:0')
        #self.dropout = self.sess.graph.get_tensor_by_name('dropout_keep_prob:0')
        self.dropout = self.sess.graph.get_tensor_by_name('dropout:0')
        self.preds = self.sess.graph.get_tensor_by_name('sequence_preds:0')
        self.probs = self.sess.graph.get_tensor_by_name('sequence_probs:0')
        
    def predict(self, image, label_dict, EOS='EOS'):
        results = []
        probabilities = []
        nlen = len(image)
        pred_sentences, pred_probs = self.sess.run([self.preds, self.probs], \
                    feed_dict={self.is_training: False, self.dropout: 1.0, self.img_ph: image,self.Shape:(nlen,1024)})
        # labels = np.zeros((nlen,33),dtype=np.int32)
        # pred_sentences, pred_probs = self.sess.run([self.preds, self.probs], \
        #                                            feed_dict={self.is_training: False,self.dropout: 1.0,
        #                                                       self.img_ph: image, self.label_ph: labels})

        # with self.graph.as_default():
        #     with self.sess.as_default():
        #         saver = tf.train.Saver()
        #         saver.store( self.sess, "./tttt",global_step=0)

        for char in pred_sentences[0]:
            if label_dict[char] == EOS:
                break
            results.append(label_dict[char])
        probabilities = pred_probs[0][:min(len(results)+1,self.seq_len)]
        
        return results, probabilities

def preprocess(image, points, size=cfg.image_size):
    """
    Preprocess for test.
    Args:
        image: test image
        points: text polygon
        size: test image size
    """
    height, width = image.shape[:2]
    mask = polygons_to_mask([np.asarray(points, np.float32)], height, width)
    x, y, w, h = cv2.boundingRect(mask)
    mask = np.expand_dims(np.float32(mask), axis=-1)
    image = image * mask
    image = image[y:y+h, x:x+w,:]

    new_height, new_width = (size, int(w*size/h)) if h>w else (int(h*size/w), size)
    image = cv2.resize(image, (new_width, new_height))

    if new_height > new_width:
        padding_top, padding_down = 0, 0
        padding_left = (size - new_width)//2
        padding_right = size - padding_left - new_width
    else:
        padding_left, padding_right = 0, 0
        padding_top = (size - new_height)//2
        padding_down = size - padding_top - new_height

    image = cv2.copyMakeBorder(image, padding_top, padding_down, padding_left, padding_right, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])

    image = image/255.
    return image

def test(args):
    model = TextRecognition(args.pb_path, cfg.seq_len+1)
    for dir_one in os.listdir(args.img_folder):
        img_p = os.path.join(args.img_folder, dir_one)
    
        for filename in os.listdir(img_p):
            img_path = os.path.join(img_p, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            points = [[0,0], [width-1,0], [width-1,height-1], [0,height-1]]
            # images=[]
            image = preprocess(image, points, cfg.image_size)
            # images.append(image)
            # images.append(image)
            # image = np.array(images)
            image = np.expand_dims(image, 0)


            before = time.time()
            preds, probs = model.predict(image, cfg.label_dict)
            after = time.time()

            print(filename, " ----", preds, np.average(probs[:-1]))

            # plt.imshow(image[0,:,:,:])
            # plt.show()


if __name__ == '__main__':
    # with tf.gfile.FastGFile("/home/yanjun/develop/opensource/AttentionOCR-master/tttt/protobuf.pbtxt", 'r') as f:
    #     graph_def = tf.GraphDef()
    #
    #     file_content = f.read()
    #
    #     # Merges the human-readable string in `file_content` into `graph_def`.
    #     text_format.Merge(file_content, graph_def)
    #     tf.train.write_graph(graph_def, './tttt', 'text_recognition_5435_bath.pb', as_text=False)

    parser = argparse.ArgumentParser(description='OCR')

    #parser.add_argument('--pb_path', type=str, help='path to tensorflow pb model', default='/home/yanjun/all/ocr/checkpoint/text_recognition_5435.pb')
    parser.add_argument('--pb_path', type=str, help='path to tensorflow pb model',
                        default='/home/kls/AttentionOCR-master/pretrain/text_recognition_5435.pb')
    parser.add_argument('--img_folder', type=str, help='path to image folder',
                        default='/home/kls/numtrain/test')
    
    args = parser.parse_args()
    test(args)
    
