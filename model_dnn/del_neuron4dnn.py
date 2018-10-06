# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
import matplotlib.pyplot as plt
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
        layer_num=[]
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        data_path=HDF5_structure(data)
        #print data_path
        self.data_path=[]
        for path in data_path:
            if os.path.basename(path.split(':')[0])!='kernel':
                continue
            self.data_path.append(path)
            neuron_num+=data[path].shape[0]
            layer_num.append(data[path].shape[0])
        #print self.data_path
        #print self.data_path[2]
        #print data[self.data_path[2]].shape
        #print type(data[self.data_path[2]])
        #print data[self.data_path[2]][783].shape
        #print data[self.data_path[2]][456][67]
        print "neuron_num:",neuron_num-784
        data.close()
        return neuron_num,layer_num

    def del_neuron(self,data,neuron_index):
        '''
        neuron_index:(layer_num,index)
        '''
        layer_num,index=neuron_index
        #print layer_num,index
        json_string=self.model.to_json()
        path=self.data_path[layer_num]
        #print data[path].shape
        data_change=data
        arr=data[path][index].copy()
        arr*=0
        data_change[path][index]=arr
        
        #print('parameter:{}'.format(model.count_params()))
        #print('mutation param:{}'.format(int(random_ratio*model.count_params())))
        #print('extend :{}'.format(extent))
        return

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
    
    def del_neuron_random(self,ndim,num,loopnum):
    #num:每次删除的神经元个数
    #loopnum:循环的次数
        print "num:",num
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.model.save_weights('my_model_weight.h5')
        data=h5py.File('my_model_weight.h5','r+')
        lst=[]
        statis=[]
        for i in range(len(self.data_path)):
            if data[self.data_path[i]].shape[0]==ndim:#记录一下是不是输入层神经元
                continue
            for index in range(data[self.data_path[i]].shape[0]):
                temp =(i,index)
                lst.append(temp)
        
        print len(lst)
        json_string=self.model.to_json()
        model_temp=model_from_json(json_string)
        for loop in range(loopnum):
            random_choice=np.random.choice(len(lst),num,replace=False)
            print random_choice
            for j in range(num):
                #print 'num',num
                #print lst[random_choice[j]] 
                self.del_neuron(data,lst[random_choice[j]])
            model_temp.load_weights('my_model_weight.h5')
            
            statis.append(accuracy_mnist(model_temp,mnist))
            print 'accuracy of origin:',accuracy_mnist(self.model,mnist)
            print 'accuracy of change:',accuracy_mnist(model_temp,mnist)
            data.close()
            self.model.save_weights('my_model_weight.h5')
            data=h5py.File('my_model_weight.h5','r+')
        data.close()
        mean_acc = sum(statis)/len(statis)
        return mean_acc


def accuracy_mnist(model,mnist):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    pred=model.predict(mnist.test.images)
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


if __name__=='__main__':
    my_file = './my_model_weight.h5'
    if os.path.exists(my_file):
        os.remove(my_file)
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    model_path1='./model_dnn_1/model.hdf5'
    model_path3='./model_dnn_3/model.hdf5'
    model_path5='./model_dnn_5/model.hdf5'
    model_path10='./model_dnn_10/model.hdf5'
    model_path =[model_path1,model_path3,model_path5,model_path10]
    ndim =784
    ratio_lst=[0.05,0.1,0.15,0.2,0.3]
    #ratio_lst=[0.001,0.005]
    acclst1=[0 for i in range(len(ratio_lst))]
    acclst3=[0 for i in range(len(ratio_lst))]
    acclst5=[0 for i in range(len(ratio_lst))]
    acclst10=[0 for i in range(len(ratio_lst))]
    acclst=[acclst1,acclst3,acclst5,acclst10]
    
    for i in range(4):
        print i
        model=load_model(model_path[i]) 
        del_=model_mutation_del_neuron(model)
        neuron_num,layer_num = del_.get_neuron()
        #num_lst=[1,2,5]
        for k in range(len(ratio_lst)):
            ratio=ratio_lst[k]
            num=int((neuron_num-784)*ratio)
            mean_acc = del_.del_neuron_random(ndim,num,10)
            acclst[i][k]=mean_acc
    plt.title('Results of deleting neurons randomly')
    plt.xlabel("Deleting ratio of neurons(0.05,0.1,0.15,0.2,0.3)")
    plt.ylabel("Accuracy of the mutated model")
    plt.plot(ratio_lst,acclst[0], color='green', label='dnn_1')
    plt.plot(ratio_lst,acclst[1], color='red', label='dnn_3')
    plt.plot(ratio_lst,acclst[2], color='black', label='dnn_5')
    #plt.plot(ratio_lst,acclst[3], color='blue', label='dnn_10')
    plt.savefig("del_neurons4dnn.png")
    
    
    #print statislst
    
    #model_change.save_weights('my_model_weight.h5')
    ##print model_change.load_weights
    #print HDF5_structure(data)
    #print accuracy_mnist(model_change,mnist)
'''
    acc=[[]for i in range(28)]
    for i in range(ndim):
        model_change = del_.mask_input(ndim,i)
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
