# -*- coding: utf-8 -*-
from keras import Model
from keras.models import load_model
import h5py
import os
import numpy as np
from sklearn.metrics import accuracy_score
from keras import backend as K
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm


def HDF5_structure(data):
    root = list(map(lambda x: x[0], data.items()))
    print(root)
    final_path = []
    data_path = []
    while True:
        if len(root) == 0:
            break
        else:
            for item in root:
                if isinstance(data[item], h5py._hl.dataset.Dataset) or len(data[item].items()) == 0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item], h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item, sub_item[0]))
                    root.remove(item)
    return data_path


def accuracy(model, images, labels):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    pred = model.predict(images)
    # print(mnist.test.images.shape) 10000*784
    # print(mnist.test.images.dtype) float32
    # print(list(mnist.test.images[0])) 784 vector
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), labels))
    return accuracy_score(test_label, pred)


def getActivationLayers(model):
    intermediate_layer_model_1 = Model(inputs=model.input, outputs=model.get_layer("activation_7").output)
    intermediate_layer_model_2 = Model(inputs=model.input, outputs=model.get_layer("activation_8").output)
    #intermediate_layer_model_3 = Model(inputs=model.input, outputs=model.get_layer("activation_3").output)
    return intermediate_layer_model_1, intermediate_layer_model_2


def statis_act_all():
    model_path = '../model_bp/model_raw.hdf5'
    model = load_model(model_path)
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    images=mnist.test.images
    labels=np.argmax(mnist.test.labels,axis=1)
    activation1, activation2 = getActivationLayers(model)
    temp1=activation1.predict(images)
    temp2=activation2.predict(images)
    print((np.sum(temp1,axis=0)>0).sum())
    print((np.sum(temp2,axis=0)>0).sum())




if __name__ == '__main__':
    #128+64个神经元 192个神经元
    model_path = '../model_bp/model_raw.hdf5'
    model = load_model(model_path)
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    images=mnist.test.images
    labels=np.argmax(mnist.test.labels,axis=1)
    activation1, activation2 = getActivationLayers(model)
    max=0
    max_index=0
    lst=range(len(images))
    choice_lst=[]
    while True:
        max=0
        max_index=0
        if len(choice_lst)!=0:
            a1=activation1.predict(images[choice_lst])
            a2=activation2.predict(images[choice_lst])
            a1=np.sum(a1,axis=0)>0
            a2=np.sum(a2,axis=0)>0
        for i in tqdm(lst):
            temp1=activation1.predict(images[[i]])
            temp2=activation2.predict(images[[i]])
            if len(choice_lst)==0:
                temp=(temp1>0).sum()+(temp2>0).sum()
            else:
                temp1=np.bitwise_or(a1,temp1>0)
                temp2=np.bitwise_or(a2,temp2>0)
                temp=temp1.sum()+temp2.sum()
            if temp>max:
                max=temp
                max_index=i
        print(max)
        if max==192:
            break
        lst.remove(max_index)
        choice_lst.append(max_index)
