#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import PIL.Image as Image
from skimage import io, transform
import cv2
import time

cap = cv2.VideoCapture(0)


def run(model_pb_path):
    #讀取model.pb
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(model_pb_path, 'rb') as fid:
            output_graph_def.ParseFromString(fid.read())
            _ = tf.import_graph_def(output_graph_def, name = "")

    #Session your model
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            out_label = sess.graph.get_tensor_by_name("output:0")
            
            while(True):
                ret, frame = cap.read()
                #讀取影像並處理格式
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #otsu濾波
                ret2 ,th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite('345.jpg', th1)
                
                time.sleep(0.25)

                img = Image.open('345.jpg').convert('L')
                if img.size[0] != 28 or img.size[1] != 28:
                    img = img.resize((28,28))
                #暫存影像之矩陣
                arr = []
                for i in range(28):
                    for j in range(28):
                        pixcel = 1.0 - float(img.getpixel((j,i))) / 255.0
                        arr.append(pixcel)

                arr1 = np.array(arr).reshape((1,28,28,1))
                #print(arr1)

                img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(arr1, [-1, 784]),keep_prob:1.0})
        
                #print("img_out_softmax:", img_out_softmax)
                prediction_result = np.argmax(img_out_softmax, axis=1)
                print("result: ",prediction_result)
                time.sleep(0.25)
                cv2.imshow('object detection', gray)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

if __name__ == '__main__':
    run('3.pb')
