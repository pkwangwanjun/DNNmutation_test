# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../')

import numpy as np
from model_change_standard import model_mutation_del_neuron
from model_change_standard import accuracy_mnist
from model_change_standard import random_del_neuron
import pandas as pd
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

def choice_test(model_path):
    model=load_model(model_path)
    pred=np.argmax(model.predict(mnist.test.images),axis=1)
    label=np.argmax(mnist.test.labels,axis=1)
    test=mnist.test.images[(pred==label)]
    label=label[(pred==label)]
    return test,label

def mutation_count(model,test_image,test_label):
    pred=np.argmax(model.predict(test_image),axis=1)
    num=(pred!=test_label).sum()
    return num,len(pred)

def mutation(model_path):
    model=load_model(model_path)
    del_neuron=model_mutation_del_neuron(model)
    neuron_num,layer_num=del_neuron.get_neuron()
    test_image,test_label=choice_test(model_path)
    kill=[]
    all=[]
    for layer,index in enumerate(neuron_num):
        for i in range(index):
            model_change=del_neuron.del_neuron((layer,i))
            kill_num,all_num=mutation_count(model_change,test_image,test_label)
            kill.append(kill_num)
            all.append(all_num)
            print(kill_num,all_num)
    return kill,all


if __name__=='__main__':
    for i in [1,3,5,10]:
        model_path='../model_dnn/model_dnn_{}/model.hdf5'.format(i)
        kill,all=mutation(model_path)
        kill=pd.DataFrame(kill)
        kill.to_csv('mut{}.csv'.format(i))
