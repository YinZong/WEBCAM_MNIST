#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import PIL.Image as Image
from skimage import io, transform

def recognize(img_path, pb_file_path):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            print(input_x)
            keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
            print(keep_prob)
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            print(out_softmax)
            out_label = sess.graph.get_tensor_by_name("output:0")
            print(out_label)

            img = Image.open(img_path).convert('L')
            img = img.resize((28, 28))
            arr = []
            pixelmin = float(img.getpixel((0, 0)))
            pixelmax = float(img.getpixel((0, 0)))
            for i in range(28):
                for j in range(28):
                    # mnist 裡的顏色是0代表背景，1.0代表字
                    if pixelmin > float(img.getpixel((j, i))):
                        pixelmin = float(img.getpixel((j, i)))
                    if pixelmax < float(img.getpixel((j, i))):
                        pixelmax = float(img.getpixel((j, i)))
            #print(pixelmin, pixelmax)
            for i in range(28):
                for j in range(28):
                    # mnist 裡的顏色是0代表背景，1.0代表字
                    pixel = (float(img.getpixel((j, i))) - pixelmin)/(pixelmax - pixelmin)
                    arr.append(pixel)

            #print(arr)
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(arr, [-1, 784]), keep_prob:1.0})

            print("img_out_softmax:", img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            print("label:", prediction_labels)

recognize("3.jpg", "3.pb")
