#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras import Model,Input
import sys
import time
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.models import Model,load_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2


#导入数据集
from keras.datasets import mnist

def model_mnist():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28

    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

    nb_classes=10

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')

    input_tensor=Input((28,28,1))
    #28*28
    temp=Conv2D(filters=32,kernel_size=(3,3),padding='valid',use_bias=False)(input_tensor)
    temp=Activation('relu')(temp)
    #26*26
    temp=MaxPooling2D(pool_size=(2, 2))(temp)
    #13*13
    temp=Conv2D(filters=64,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #11*11
    temp=MaxPooling2D(pool_size=(2, 2))(temp)
    #5*5
    temp=Conv2D(filters=128,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #3*3
    temp=Flatten()(temp)

    temp=Dense(nb_classes)(temp)
    output=Activation('softmax')(temp)

    model=Model(input=input_tensor,outputs=output)

    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1)

    batch_size=32
    model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train)//batch_size, epochs=5,validation_data=(X_test, Y_test))

    #model.fit(X_train, Y_train, batch_size=64, nb_epoch=5,validation_data=(X_test, Y_test))
    #Y_pred = model.predict(X_test, verbose=0)
    model.save('./model/model.hdf5')
    result=[]
    for i in range(10):
        index= np.argmax(Y_test,axis=1)==i
        score_org=model.evaluate(X_test[index], Y_test[index], verbose=0)

        score_aug=model.evaluate_generator(datagen.flow(X_test[index],Y_test[index],batch_size=32))

        result.extend([score_org[1],score_aug[1]])

    score_org=model.evaluate(X_test, Y_test, verbose=0)

    score_aug=model.evaluate_generator(datagen.flow(X_test,Y_test,batch_size=32))
    result.extend([score_org[1],score_aug[1]])
    return pd.DataFrame(result).T



def model_mnist_gen(i):
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28

    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    print('Train:{},Test:{}'.format(len(X_train),len(X_test)))

    nb_classes=10

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)
    print('data success')

    input_tensor=Input((28,28,1))
    #28*28
    temp=Conv2D(filters=32,kernel_size=(3,3),padding='valid',use_bias=False)(input_tensor)
    temp=Activation('relu')(temp)
    #26*26
    temp=MaxPooling2D(pool_size=(2, 2))(temp)
    #13*13
    temp=Conv2D(filters=64,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #11*11
    temp=MaxPooling2D(pool_size=(2, 2))(temp)
    #5*5
    temp=Conv2D(filters=128,kernel_size=(3,3),padding='valid',use_bias=False)(temp)
    temp=Activation('relu')(temp)
    #3*3
    temp=Flatten()(temp)

    temp=Dense(nb_classes)(temp)
    output=Activation('softmax')(temp)

    model=Model(input=input_tensor,outputs=output)

    model.summary()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    #datagen = ImageDataGenerator(rotation_range=20,width_shift_range=0.1,height_shift_range=0.1)

    batch_size=32
    #model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
    #                    steps_per_epoch=len(X_train)//batch_size, epochs=5,validation_data=(X_test, Y_test))

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=5,validation_data=(X_test, Y_test))
    #Y_pred = model.predict(X_test, verbose=0)
    model.save('./model/model_{}/model.hdf5'.format(i))


def GrayScale(img):
    return img*0.3


def mr_eval(i,gray=False,rotation=False,width=False,height=False,shear=False,zoom=False,thin=False,fat=False):
    model=load_model('./model/model_{}/model.hdf5'.format(i))

    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()  # 28*28

    X_test = X_test.astype('float32').reshape(-1,28,28,1)

    X_test /= 255

    nb_classes=10
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    #rotation_range=0.
    #width_shift_range=0
    #height_shift_range=0.
    #shear_range=0.
    #zoom_range=0.
    #preprocessing_function  自定义灰度
    if gray:
        datagen = ImageDataGenerator(preprocessing_function=GrayScale)
    elif rotation:
        datagen = ImageDataGenerator(rotation_range=20)
    elif width:
        datagen = ImageDataGenerator(width_shift_range=0.1)
    elif height:
        datagen = ImageDataGenerator(height_shift_range=0.1)
    elif shear:
        datagen = ImageDataGenerator(shear_range=0.2)
    elif zoom:
        datagen = ImageDataGenerator(zoom_range=0.2)

    elif thin:
        datagen = ImageDataGenerator(preprocessing_function=thin_down)

    elif fat:
        datagen = ImageDataGenerator(preprocessing_function=get_fat)

    result=[]
    for i in range(10):
        index= np.argmax(Y_test,axis=1)==i
        score_org=model.evaluate(X_test[index], Y_test[index], verbose=0)

        score_aug=model.evaluate_generator(datagen.flow(X_test[index],Y_test[index],batch_size=32))

        result.extend([score_org[1],score_aug[1]])

    return pd.DataFrame(result).T


def thin_down(image):
    kernel = np.ones((2,2),np.uint8)
    erosion = cv2.erode(image,kernel,iterations=1)
    return erosion.reshape(28,28,1)

def get_fat(image):
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(image,kernel,iterations=1)
    return dilation.reshape(28,28,1)





if __name__=='__main__':
    #result=pd.DataFrame()
    #for i in range(10):
    pass
    '''
    i=0
    re=mr_eval(i,thin=True)
    re.to_csv('thin.csv')

    re=mr_eval(i,fat=True)
    re.to_csv('fat.csv')
    '''
