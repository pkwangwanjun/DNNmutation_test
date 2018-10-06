#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense,Activation
from tensorflow.examples.tutorials.mnist import input_data
from model_change_standard import accuracy_mnist


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def model_dnn_1():
    #0.9692
    input_data=Input((28*28,))
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(10)(temp_data)
    output_data=Activation('softmax')(temp_data)
    model=Model(inputs=[input_data],outputs=[output_data])
    modelcheck=ModelCheckpoint('model_dnn/model_dnn_1/model.hdf5',monitor='loss',verbose=1,save_best_only=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit([mnist.train.images],[mnist.train.labels],batch_size=256,epochs=10,callbacks=[modelcheck],validation_data=(mnist.test.images,mnist.test.labels))

    print('acc:{}'.format(accuracy_mnist(model,mnist)))


def model_dnn_3():
    #0.9724
    input_data=Input((28*28,))
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(10)(temp_data)
    output_data=Activation('softmax')(temp_data)
    model=Model(inputs=[input_data],outputs=[output_data])
    modelcheck=ModelCheckpoint('model_dnn/model_dnn_3/model.hdf5',monitor='loss',verbose=1,save_best_only=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit([mnist.train.images],[mnist.train.labels],batch_size=256,epochs=10,callbacks=[modelcheck],validation_data=(mnist.test.images,mnist.test.labels))

    print('acc:{}'.format(accuracy_mnist(model,mnist)))

def model_dnn_5():
    #0.9712
    input_data=Input((28*28,))
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(10)(temp_data)
    output_data=Activation('softmax')(temp_data)
    model=Model(inputs=[input_data],outputs=[output_data])
    modelcheck=ModelCheckpoint('model_dnn/model_dnn_5/model.hdf5',monitor='loss',verbose=1,save_best_only=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit([mnist.train.images],[mnist.train.labels],batch_size=256,epochs=10,callbacks=[modelcheck],validation_data=(mnist.test.images,mnist.test.labels))

    print('acc:{}'.format(accuracy_mnist(model,mnist)))

def model_dnn_10():
    #0.9683
    input_data=Input((28*28,))
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(temp_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(64)(input_data)
    temp_data=Activation('relu')(temp_data)
    temp_data=Dense(10)(temp_data)
    output_data=Activation('softmax')(temp_data)
    model=Model(inputs=[input_data],outputs=[output_data])
    modelcheck=ModelCheckpoint('model_dnn/model_dnn_10/model.hdf5',monitor='loss',verbose=1,save_best_only=True)
    model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    model.fit([mnist.train.images],[mnist.train.labels],batch_size=256,epochs=10,callbacks=[modelcheck],validation_data=(mnist.test.images,mnist.test.labels))

    print('acc:{}'.format(accuracy_mnist(model,mnist)))

if __name__=='__main__':
    model_dnn_10()
