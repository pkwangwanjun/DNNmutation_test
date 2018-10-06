#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Model,Input,load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Activation
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../')
from model_change_standard import accuracy_mnist


mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)


def model_dnn_3():
    #train:0.8596
    #test:0.9308
    input_data=Input((28*28,))
    temp_data=Dense(128)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(48)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(10)(temp_data)
    output_data=Activation('softmax')(temp_data)
    model=Model(inputs=[input_data],outputs=[output_data])
    modelcheck=ModelCheckpoint('model/model.hdf5',monitor='loss',verbose=1,save_best_only=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit([mnist.train.images],[mnist.train.labels],batch_size=256,epochs=1,callbacks=[modelcheck],validation_data=(mnist.test.images,mnist.test.labels))
    print('acc:{}'.format(accuracy_mnist(model,mnist)))

if __name__=='__main__':
    #model_dnn_3()
    model_path='./model/model.hdf5'
    model=load_model(model_path)
    print('acc:{}'.format(accuracy_mnist(model,mnist)))
