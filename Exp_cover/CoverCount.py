#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Model,Input,load_model
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from tqdm import tqdm
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

test=mnist.test.images

def cover(model_path,test,threshold=5,choice_index=[]):
    '''
    model_path:模型路径
    test：测试集合
    threhold：选择的阀值
    example：随机选择训练集合的数量
    '''
    lst=[1,3,5]
    model=load_model(model_path)
    #print model.layers
    layer_num=0
    for index in lst:
        layer_num+=int(model.layers[index].output.shape[-1])

    model_layer=Model(inputs=model.input,outputs=[model.layers[i].output for i in lst])
    image=test[choice_index]
    act_layers=model_layer.predict_on_batch(image)

    act_num=0
    for act in act_layers:
        act_num+=(np.sum(act>threshold,axis=0)>0).sum()

    ratio=act_num/float(layer_num)
    return ratio

def accuracy(model_path,test,choice_index):
    model=load_model(model_path)
    pred=model.predict(test)
    pred=list(map(lambda x:np.argmax(x),pred))
    test_label=list(map(lambda x:np.argmax(x),mnist.test.labels))
    acc=0
    for i in choice_index:
        if pred[i]==test_label[i]:
            acc+=1
    return 1.0*acc/len(choice_index)


class Exp2(object):
    def __init__(self,model_path,mnist,sample_size=1000,epoch=100,lst=[1,3,5]):
        '''
        model_path:模型的路径
        mnist:mnist数据集合
        sample_size：采样的个数
        epoch：采样的轮数
        lst：指定哪几层

        ps:抽样使用的分层抽样 每次保证正确和错误的比例相同
        '''
        self.model_path=model_path
        self.mnist=mnist
        self.sample_size=sample_size
        self.epoch=epoch
        self.lst=lst

        self.model=load_model(self.model_path)

        self.model_layer=Model(inputs=self.model.input,outputs=[self.model.layers[i].output for i in lst])

        self.lst=lst

        self.layer_num=0
        for index in lst:
            self.layer_num+=int(self.model.layers[index].output.shape[-1])


    def _split_test(self):

        label=np.argmax(self.mnist.test.labels,axis=1)
        pred=np.argmax(self.model.predict(self.mnist.test.images),axis=1)
        self.badcase=self.mnist.test.images[pred!=label]
        self.goodcase=self.mnist.test.images[pred==label]

        return len(self.badcase),len(self.goodcase)

    def _count(self,image,threshold):
        act_layers=self.model_layer.predict_on_batch(image)

        act_num=0
        act_lst=[]
        for act in act_layers:
            act_lst.append(np.sum(act>threshold,axis=0)>0)
            act_num+=(np.sum(act>threshold,axis=0)>0).sum()

        ratio=act_num/float(self.layer_num)
        return act_lst,ratio

    def exp(self):
        n1,n2=self._split_test()
        ratio1=n1/float(n1+n2)
        ratio2=n2/float(n1+n2)

        result=[]

        for epoch in range(self.epoch):
            good_index=np.random.choice(range(n2),size=int(self.sample_size*ratio2),replace=False)
            bad_index=np.random.choice(range(n1),size=int(self.sample_size*ratio1),replace=False)

            act_good,ratio_good=self._count(self.goodcase[good_index],threshold=0.1)



            act_bad,ratio_bad=self._count(self.badcase[bad_index],threshold=0.1)

            xor_ratio=sum([np.bitwise_xor(act_good[i],act_bad[i]).sum() for i in range(len(self.lst))])/float(self.layer_num)


            result.append([ratio_good,ratio_bad,xor_ratio])

        return pd.DataFrame(result,columns=['good','bad','xor'])



if __name__=='__main__':
    #ratio=cover('model/model.hdf5',test)
    #print ratio
    model_path='./model/model.hdf5'

    obj=Exp2(model_path,mnist)
    result=obj.exp()
    '''
    activation_threshold=[0,0.5,1,3,5]
    dic_cover={}
    dic_acc={}
    for i in range(5):
        dic_cover[activation_threshold[i]]=[]
        dic_acc[activation_threshold[i]]=[]
        for j in range(100):
            print j
            choice_index=np.random.choice(np.arange(len(test)),size=1000,replace=False)
            cover_ratio = cover(model_path,test,activation_threshold[i],choice_index)
            acc = accuracy(model_path,test,choice_index)
            dic_cover[activation_threshold[i]].append(cover_ratio)
            dic_acc[activation_threshold[i]].append(acc)
    df_kernel=pd.DataFrame(dic_cover)
    df_kernel.to_csv('cover_ratio.csv')
    df_kernel=pd.DataFrame(dic_acc)
    df_kernel.to_csv('accuracy.csv')
    '''
