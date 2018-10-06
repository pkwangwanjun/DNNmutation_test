# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
import h5py
import os
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
import model_change_standard
import sys
import copy
from scipy import sparse
import glob
from matplotlib import pyplot as plt


def Gen_mutaion(model_path,size=100):
    model=load_model(model_path)
    for index in range(size):
        #这个地方可以使用for语句进行多次变异
        #这个地方自己选择变异的类型权重和比例
        extent = (index/10)*0.5+3
        random_ratio = ((index+1) % 10)*0.05
        _,_,_,_,model_mut=model_change_standard.model_mutation_single_neuron(model,'kernel',random_ratio,extent)
        sys.stdout.write('index:{},extent:{},random_ratio:{}\r'.format(index,extent,random_ratio))
        sys.stdout.flush()
        model_mut.save('./model_bp/mutation/model_mutation_{}.hdf5'.format(index))

def data_kill_mutation(mutation_path,img_data,label):
    #用稀疏矩阵来存储，行数是变异体数量，列数是样本数量
    path_lst=glob.glob(mutation_path+'/*.hdf5')
    print len(path_lst)
    count=[]
    row=np.array([])
    col=np.array([])
    data=np.array([])
    for index,path in enumerate(path_lst):
        sys.stdout.write("index:{} \r".format(index))
        sys.stdout.flush()
        temp=load_model(path)
        pred=temp.predict([img_data])
        pred=np.array(map(lambda x:np.argmax(x),pred))
        real=np.array(map(lambda x:np.argmax(x),label))
        bool_arr=(pred!=real)
        num=int(os.path.basename(path).split('_')[-1][:-5])
        print bool_arr.sum()
        row_add=num*np.ones(bool_arr.sum())
        row=np.append(row,row_add)
        col=np.append(col,np.where(bool_arr)[0])
        data=np.append(data,np.ones_like(row_add))
    sparse_matrix=sparse.coo_matrix((data,(row,col)),shape=(len(path_lst),len(label)))
    return sparse_matrix


if __name__=='__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model_path='./model_bp/model_raw.hdf5'
    #Gen_mutaion(model_path)
    img_data=mnist.test.images
    label=mnist.test.labels
    mutation_path='./model_bp/mutation'
    matrix=data_kill_mutation(mutation_path,img_data,label)
    #每个变异体能杀死的样本比例
    matrix1=copy.deepcopy(matrix)
    matrix2=copy.deepcopy(matrix)
    ratio1=np.array((matrix1.sum(axis=1)).tolist()).reshape(-1)
    #每个样本能杀死的变异体比例
    ratio2=np.array((matrix2.sum(axis=0)).tolist()).reshape(-1)
    fig,axes=plt.subplots(1,2)
    axes[0].plot(ratio1)
    axes[1].hist(ratio2)
