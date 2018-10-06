# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../')

from model_change_standard import model_mutation_single_neuron
from model_change_standard import accuracy_mnist
from model_change_standard import random_del_neuron
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"



mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
def test_kernel(model_path):
    model=load_model(model_path)
    dic={}
    dic[3]=[]
    dic[5]=[]
    dic[10]=[]
    dic[0.1]=[]
    dic[0.01]=[]
    for extent in [3,5,10,0.1,0.01]:
        ratio=0.0
        step=0.1
        for i in range(5):
            ratio_temp=0.0
            change=False
            acc=0.98
            while change | (acc>0.9):
                ratio=ratio+step
                if ratio==ratio_temp:
                    break
                change=False
                _,_,_,_,model_change=model_mutation_single_neuron(model,cls='kernel',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist)
                print(acc)
                if acc<=0.90:
                    if step!=0.001:
                        ratio_temp=ratio
                        step=step/10.0
                        ratio=ratio-step*10.0
                        change=True
                        print(acc,ratio)
            dic[extent].append((ratio,acc))
            ratio=0.0
            step=0.1
    return dic


def test_bias(model_path):
    model=load_model(model_path)
    dic={}
    dic[5]=[]
    dic[10]=[]
    dic[20]=[]
    dic[30]=[]
    dic[50]=[]
    for extent in [5,10,20,30,50]:
        ratio = 0.0
        step=0.1
        for i in range(5):
            ratio_temp=0.0
            change=False
            acc=0.98
            while change | (acc>0.9):
                ratio = ratio+step
                if ratio==ratio_temp:
                    break
                _,_,_,_,model_change=model_mutation_single_neuron(model,cls='bias',random_ratio=ratio,extent=extent)
                acc=accuracy_mnist(model_change,mnist)
                print(acc)
                if acc<=0.90:
                    if step!=0.001:
                        ratio_temp=ratio
                        step=step/10.0
                        ratio=ratio-step*10.0
                        change=True
                        print(step,ratio)
            dic[extent].append((ratio,acc))
            ratio = 0.0
            step=0.1
    return dic


def test_del(model_path):
    model=load_model(model_path)
    ratio_lst=[0.05,0.1,0.15,0.2,0.3]
    dic={}
    dic[0.05]=[]
    dic[0.1]=[]
    dic[0.15]=[]
    dic[0.2]=[]
    dic[0.3]=[]
    for ratio in ratio_lst:
        for i in range(5):
            model_change=random_del_neuron(model,ratio=ratio)
            acc=accuracy_mnist(model_change,mnist)
            print(acc,ratio)
            dic[ratio].append(acc)
    return dic

if __name__=='__main__':
    model_path='./model_dnn_1/model.hdf5'
    dic1=test_kernel(model_path)
    df_kernel=pd.DataFrame(dic1)
    df_kernel.to_csv('kernel_1.csv')
    dic2=test_bias(model_path)
    df_bias=pd.DataFrame(dic2)
    df_bias.to_csv('bias_1.csv')
    dic3=test_del(model_path)
    df_del=pd.DataFrame(dic3)
    df_del.to_csv('del_1.csv')
