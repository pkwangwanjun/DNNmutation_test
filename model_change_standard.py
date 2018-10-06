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


def HDF5_structure(data):
    root=data.keys()
    final_path=[]
    data_path=[]
    while True:
        if len(root)==0:
            break
        else:
            for item in root:
                if isinstance(data[item],h5py._hl.dataset.Dataset) or len(data[item].items())==0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item],h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item,sub_item[0]))
                    root.remove(item)
    return data_path


def model_mutation_single_neuron(model,cls='kernel',random_ratio=0.001,extent=1):
    '''
    model:keras DNN_model
    cls: 'kernel' or 'bias'
    extent:ratio
    '''
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    data=h5py.File('my_model_weight.h5','r+')
    data_path=HDF5_structure(data)
    lst=[]

    for path in data_path:
        if os.path.basename(path.split(':')[0])!=cls:
            continue
        if len(data[path].shape)==2:
            row,col=data[path].shape
            lst.extend([(path,i,j) for i in range(row) for j in range(col)])
        else:
            row=data[path].shape[0]
            lst.extend([(path,i) for i in range(row)])
    random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
    lst_random=np.array(lst)[[random_choice]]

    for path in lst_random:
        try:
            arr=data[path[0]][int(path[1])].copy()
            arr[int(path[2])]*=extent
            data[path[0]][int(path[1])]=arr
        except:
            arr=data[path[0]][int(path[1])]
            arr*=extent
            data[path[0]][int(path[1])]=arr
    data.close()
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    #print('parameter:{}'.format(model.count_params()))
    #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
    #print('extend :{}'.format(extent))
    return len(lst),data_path,model.count_params(),int(random_ratio*model.count_params()),model_change

def model_mutation_single_neuron_cnn(model,cls='kernel',layers='dense',random_ratio=0.001,extent=1):
    '''
    model:keras DNN_model or CNN_model
    cls: 'kernel' or 'bias'
    layers: 'dense' or 'conv'
    extent:ratio
    '''
    json_string=model.to_json()
    model.save_weights('my_model_weight.h5')
    data=h5py.File('my_model_weight.h5','r+')
    data_path=HDF5_structure(data)
    lst=[]

    for path in data_path:
        if os.path.basename(path.split(':')[0])!=cls:
            continue
        if layers not in path.split('/')[0]:
            continue

        if len(data[path].shape)==4:
            a,b,c,d=data[path].shape
            lst.extend([(path,a_index,b_index,c_index,d_index) for a_index in range(a) for b_index in range(b) for c_index in range(c) for d_index in range(d)])
        if len(data[path].shape)==2:
            row,col=data[path].shape
            lst.extend([(path,i,j) for i in range(row) for j in range(col)])
        else:
            row=data[path].shape[0]
            lst.extend([(path,i) for i in range(row)])
    random_choice=np.random.choice(range(len(lst)),replace=False,size=int(random_ratio*len(lst)))
    lst_random=np.array(lst)[[random_choice]]

    for path in lst_random:
        if len(path)==3:
            arr=data[path[0]][int(path[1])].copy()
            arr[int(path[2])]*=extent
            data[path[0]][int(path[1])]=arr
        elif len(path)==2:
            arr=data[path[0]][int(path[1])]
            arr*=extent
            data[path[0]][int(path[1])]=arr
        elif len(path)==5:
            arr=data[path[0]][int(path[1])][int(path[2])][int(path[3])].copy()
            arr[int(path[4])]*=extent
            data[path[0]][int(path[1])][int(path[2])][int(path[3])]=arr
    data.close()
    model_change = model_from_json(json_string)
    model_change.load_weights('my_model_weight.h5')
    #print('parameter:{}'.format(model.count_params()))
    #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
    #print('extend :{}'.format(extent))
    return len(lst),data_path,model.count_params(),int(random_ratio*model.count_params()),model_change



class model_mutation_del_neuron(object):
    '''
    1、初始化是模型
    2、首先看有几个全链接层，以及全连接层每层有多少个神经元
    3、del_neuron的输入是第几层神经元和第几个神经元
    4、可反复变异
    '''

    def __init__(self,model):
        self.model=model

    def get_neuron(self):
        neuron_num=0
        kernel_num=[]
        bias_num=[]
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        data_path=HDF5_structure(data)
        #print data_path
        self.data_path=[]
        self.bias_path=[]
        for path in data_path:
            if os.path.basename(path.split(':')[0])!='kernel':
                self.bias_path.append(path)
                continue
            self.data_path.append(path)


        self.data_path.sort()
        self.bias_path.sort()

        for kernel,bias in zip(self.data_path,self.bias_path):
            kernel_num.append(data[kernel].shape[0])
            bias_num.append(data[bias].shape[0])
        #print self.data_path
        #print self.data_path[2]
        #print data[self.data_path[2]].shape
        #print data[self.data_path[2]][783].shape
        data.close()
        return kernel_num,bias_num

    def del_neuron(self,neuron_index):
        '''
        neuron_index:(layer_num,index)
        '''
        layer_num,index=neuron_index
        json_string=self.model.to_json()
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        path=self.data_path[layer_num]
        if layer_num-1>=0:
            path_bias=self.bias_path[layer_num-1]
            arr_bias=data[path_bias][index].copy()
            arr_bias*=0
            data[path_bias][index]=arr_bias
        arr=data[path][index].copy()
        arr*=0
        data[path][index]=arr

        data.close()
        model_change = model_from_json(json_string)
        model_change.load_weights('my_model_weight.h5')
        #print('parameter:{}'.format(model.count_params()))
        #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
        #print('extend :{}'.format(extent))
        return model_change

    def mask_input(self,ndim,index):
        '''
        ndim:总维数
        index:需要删除的维
        '''
        json_string=self.model.to_json()
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        for i in range(len(self.data_path)):
            if data[self.data_path[i]].shape[0]==ndim:
                for j in range(data[self.data_path[i]].shape[1]):
                    arr = data[self.data_path[i]][index].copy()
                    arr[j]*=0
                    data[self.data_path[i]][index]=arr
        data.close()
        model_change = model_from_json(json_string)
        model_change.load_weights('my_model_weight.h5')
        return model_change


def accuracy_mnist(model,mnist,cnn=False):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    if not cnn:
        pred=model.predict(mnist.test.images)
    else:
        pred=model.predict(mnist.test.images.reshape(-1,28,28,1))
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),mnist.test.labels))
    return accuracy_score(test_label,pred)
'''
def accuracy_cifar(model):
    #model: CNN_model
    #return : acc of cifar
    (_, _), (X_test, Y_test) = cifar10.load_data()
    X_test=X_test.astype('float32')
    X_test/=255
    pred=model.predict(X_test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),pd.get_dummies(Y_test.reshape(-1)).values))
    return accuracy_score(test_label,pred)
'''

def random_del_neuron(model,ratio=0.01):
    del_neuron=model_mutation_del_neuron(model)
    neuron_num,layer_num=del_neuron.get_neuron()
    del_num=[int(i*ratio) for i in layer_num]

    for layer in range(len(layer_num)):
        random_chioce=np.random.choice(layer_num[layer],size=del_num[layer],replace=False)
        for index in random_chioce:
            model_change=del_neuron.del_neuron((layer,index))
    return model_change



if __name__=='__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model_path='./model_bp/model_raw.hdf5'
    #model_path='model.hdf5'
    #model_path='./model_cnn/model_raw.hdf5'
    model=load_model(model_path)
    del_neuron=model_mutation_del_neuron(model)
    kernel_num,bias_num=del_neuron.get_neuron()
    ndim =784
    model_change=del_neuron.del_neuron((2,63))

    '''
    acc=[[]for i in range(28)]
    for i in range(ndim):
        model_change = del_neuron.mask_input(ndim,i)
        acc[i/28].append(accuracy_mnist(model_change,mnist))
        #print accuracy_mnist(model_change,mnist)
    with open("mask_input.csv","w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(acc)
    '''
    #neuron_index=(0,2)
    #model_del=del_neuron.del_neuron(neuron_index)

    #print('accuracy before mutaion:{}'.format(accuracy_cifar(model)))
    #_,_,_,_,model_mut=model_mutation_single_neuron(model,cls='kernel',extent=10,random_ratio=0.001)
    #_,_,_,_,model_mut=model_mutation_single_neuron_cnn(model,cls='kernel',layers='conv',random_ratio=0.001,extent=1)
    #print('accuracy after mutation:{}'.format(accuracy_cifar(model_mut)))
