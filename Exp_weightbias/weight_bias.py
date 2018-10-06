# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../')

from model_change_standard import model_mutation_single_neuron
from model_change_standard import model_mutation_single_neuron_cnn
from model_change_standard import accuracy_mnist
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

model_bp='../model_bp/model_raw.hdf5'
model_cnn='../model_cnn_mnist/model_raw.hdf5'

def bp_kernel(model_path):
    model=load_model(model_path)
    extent_lst=[0.01,0.1,0.5,1.5,2,3,5,10]
    statistic={i:[] for i in extent_lst}
    ratio_lst=[0.01,0.03,0.05,0.1,0.2]

    for extent in extent_lst:
        for ratio in ratio_lst:
            lst=[]
            for i in range(10):
                _,_,_,_,model_change=model_mutation_single_neuron(model,cls='kernel',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist)
                lst.append(acc)
                print(i,acc)
        statistic[extent].append(lst)
    return statistic


def bp_bias(model_path):
    model=load_model(model_path)
    extent_lst=[0.01,0.1,0.5,1.5,2,3,5,10]
    statistic={i:[] for i in extent_lst}
    ratio_lst=[0.01,0.03,0.05,0.1,0.2]

    for extent in extent_lst:
        for ratio in ratio_lst:
            lst=[]
            for i in range(10):
                print(i)
                _,_,_,_,model_change=model_mutation_single_neuron(model,cls='bias',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist)
                lst.append(acc)
        statistic[extent].append(lst)
    return statistic


def cnn_kernel(model_path):
    model=load_model(model_path)
    extent_lst=[0.01,0.1,0.5,1.5,2,3,5,10]
    statistic={i:[] for i in extent_lst}
    ratio_lst=[0.01,0.03,0.05,0.1,0.2]

    for extent in extent_lst:
        for ratio in ratio_lst:
            lst=[]
            for i in range(10):
                print(i)
                _,_,_,_,model_change=model_mutation_single_neuron_cnn(model,cls='kernel',layers='conv',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist,cnn=True)
                lst.append(acc)
                print(i,acc)
        statistic[extent].append(lst)
    return statistic


def cnn_bias(model_path):
    model=load_model(model_path)
    extent_lst=[0.01,0.1,0.5,1.5,2,3,5,10]
    statistic={i:[] for i in extent_lst}
    ratio_lst=[0.01,0.03,0.05,0.1,0.2]

    for extent in extent_lst:
        for ratio in ratio_lst:
            lst=[]
            for i in range(10):
                _,_,_,_,model_change=model_mutation_single_neuron_cnn(model,cls='bias',layers='conv',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist,cnn=True)
                lst.append(acc)
        statistic[extent].append(lst)
    return statistic


if __name__=='__main__':
    statistic=bp_bias(model_bp)
    statistic=pd.DataFrame(statistic)
    statistic.to_csv('1.csv')
    #cnn_bias(model_cnn)
