# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
import csv
import json
from sklearn.metrics import accuracy_score
import tensorflow as tf
from keras import backend as K


def activation_mut(model_path):
    model=load_model(model_path)
    model.save_weights('my_model_weight.h5')
    json_string=model.to_json()
    json_string=json_string.replace('relu','sigmoid')
    #model.layers[2].activation='sigmoid'
    #model.layers[4].activation='sigmoid'
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    return model_change



class activation_mut_single(object):
    def __init__(self,model_path):
        self.model_path=model_path

    @staticmethod
    def _sigmoid(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def _relu(x):
        return np.where(x<=0,0,x)

    @staticmethod
    def _sigmoid_single(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def _relu_single(x):
        return 0 if x<0 else x

    def extract_weight(self):
        self.sess=tf.Session()
        K.set_session(self.sess)
        with self.sess.as_default():
            model=load_model(self.model_path)
            print(model.layers)
            #784*128
            layer1=model.layers[1]
            #128*64
            layer3=model.layers[3]

            #64*10
            layer5=model.layers[5]

            self.weight1=layer1.weights[0].eval()
            self.bias1=layer1.weights[1].eval()

            self.weight2=layer3.weights[0].eval()
            self.bias2=layer3.weights[1].eval()

            self.weight3=layer5.weights[0].eval()
            self.bias3=layer5.weights[1].eval()

    def forward(self,x):
        # batch*784
        temp=np.dot(x,self.weight1)+self.bias1
        temp=self._relu(temp)

        temp=np.dot(temp,self.weight2)+self.bias2
        temp=self._relu(temp)

        temp=np.dot(temp,self.weight3)+self.bias3
        temp=self._relu(temp)

        return np.argmax(temp,axis=1)

    #这个函数是变异后的前向传播，根据自己的需求改写，这个是模版
    def forward_mut(self,x,size=10):
        # batch*784
        temp=np.dot(x,self.weight1)+self.bias1

        for mut_index in np.random.choice(temp.shape[1],size=size,replace=False):
            mut_number=self._sigmoid_single(temp[:,mut_index])
            temp=self._relu(temp)
            temp[:,mut_index]=mut_number

        temp=np.dot(temp,self.weight2)+self.bias2
        temp=self._relu(temp)

        temp=np.dot(temp,self.weight3)+self.bias3
        temp=self._relu(temp)

        return np.argmax(temp,axis=1)

    def acc_score(self,data,label):
        pred=self.forward_mut(data)
        print(accuracy_score(label,pred))


if __name__=='__main__':
    mnist=input_data.read_data_sets("MNIST_data/")
    data=mnist.test.images
    label=mnist.test.labels
    #mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model_path='./model_bp/model_raw.hdf5'
    #activation_mut(model_path)
    test=activation_mut_single(model_path)
    test.extract_weight()
    #test.forward(data)
    test.acc_score(data,label)
