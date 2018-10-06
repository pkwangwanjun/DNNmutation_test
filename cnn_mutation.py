# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.models import model_from_json
from keras.optimizers import SGD,Adam
from keras.models import Model
import h5py
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from keras.datasets import cifar10
from tensorflow.examples.tutorials.mnist import input_data
import csv
from keras.datasets import mnist
from keras.utils import np_utils


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
    return model_change


if __name__=='__main__':
    my_file = './my_model_weight.h5'
    if os.path.exists(my_file):
        os.remove(my_file)
    my_file = './cnn_mutation_result.txt'
    if os.path.exists(my_file):
        os.remove(my_file) 
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data() 
    X_train = X_train.astype('float32').reshape(-1,28,28,1)
    X_test = X_test.astype('float32').reshape(-1,28,28,1)
    X_train /= 255
    X_test /= 255
    Y_test = np_utils.to_categorical(Y_test, 10)
    model_path='./model_cnn_mnist/model_raw.hdf5'
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #model_path='model.hdf5'
    #model_path='./model_cnn/model_raw.hdf5'
    model=load_model(model_path)
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('accuracy before mutaion:{}'.format(score[1]))    
    model_mut = model
    model_mut.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    ratio = 0.0
    step=0.1
    extent_value_list=[2,3,5,10,0.9,0.5,0.1,0.05,0.01]
    for extent_value in extent_value_list:
        with open("cnn_mutation_result.txt","a") as f:
            string="extent_value:"+str(extent_value)+"\n"
            f.write(string)
        print "extent_value:",extent_value
        for j in range(10):
            change=False
            acc=0.98
            while change|(acc>0.9):
                ratio = ratio+step
                change=False
                print "ratio:",ratio,"\n"
                if ratio >1:
                    break
                model_mut=model_mutation_single_neuron_cnn(model,cls='kernel',layers='conv',random_ratio=ratio,extent=extent_value)
                model_mut.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
                acc=model_mut.evaluate(X_test, Y_test, verbose=0)[1]
                print "accuracy after mutation:",acc,"\n"
                if acc<0.9:
                    if step!=0.001:
                        step=step/10.0
                        ratio=ratio-step*10.0
                        change=True
                        print step,ratio,'\n'
                #print('accuracy after every mutation:{}\n'.format(model_mut.evaluate(X_test, Y_test, verbose=0)[1]))
            with open("cnn_mutation_result.txt","a") as f:
                string="random_ratio:"+str(ratio)+"\n"+"accuracy after mutation:"
                string=string+str(model_mut.evaluate(X_test, Y_test, verbose=0)[1])+"\n"
                f.write(string)
            ratio = 0.0
            step=0.1
            model_mut = model
            model_mut.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    with open("cnnresult.txt","a") as f:
        f.write("haha\n")