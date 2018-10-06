#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from keras.models import Model,Input,load_model
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
from tqdm import tqdm
import foolbox


mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

test=mnist.test.images

class Exp3(object):
    def __init__(self,model_path,mnist,sample_size=100,epoch=100,lst=[1,3,5]):
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


        self.foolmodel=foolbox.models.KerasModel(self.model,bounds=(0,1),preprocessing=(0,1))

        self.attack=foolbox.attacks.IterativeGradientAttack(self.foolmodel)


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

    def _adv(self,image):
        adv_lst=[]
        org_lst=[]
        for img in image:
            label=np.argmax(self.model.predict(np.expand_dims(img,axis=0)))
            adv=self.attack(img,label,epsilons=[0.01,0.1,1],steps=100)
            if isinstance(adv,np.ndarray):
                adv_lst.append(adv)
                org_lst.append(img)
        return np.array(adv_lst),np.array(org_lst)


    def exp(self):
        _,good_num=self._split_test()

        result=[]

        for epoch in tqdm(range(self.epoch)):
            good_index=np.random.choice(range(good_num),size=self.sample_size,replace=False)

            adv_lst,org_lst=self._adv(self.goodcase[good_index])

            print(adv_lst.shape,org_lst.shape)

            act_adv,ratio_adv=self._count(adv_lst,threshold=0.1)

            act_org,ratio_org=self._count(org_lst,threshold=0.1)

            result.append([ratio_adv,ratio_org])
        return pd.DataFrame(result,columns=['adv','org'])

if __name__=='__main__':
    model_path='./model/model.hdf5'
    obj=Exp3(model_path,mnist,sample_size=100,epoch=10,lst=[1,3,5])
    result=obj.exp()
