# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.metrics import accuracy_score
from keras.models import load_model
from tqdm import tqdm


mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

labels=np.argmax(mnist.test.labels,axis=1)

images_dict={}

for i in range(10):
    images_dict[i]=mnist.test.images[labels==i]


model_path='/Users/wanjun/Desktop/DNNmutation/model_bp/model_raw.hdf5'

model=load_model(model_path)


#raw acc
raw_acc={}
for i in range(10):
    pred=model.predict(images_dict[i])
    pred=np.argmax(pred,axis=1)
    raw_acc[i]=accuracy_score(i*np.ones_like(pred),pred)

#mask
mask_acc={i:[] for i in range(10)}
for i in tqdm(range(10)):
    for index in tqdm(range(784)):
        temp=images_dict[i].copy()
        temp[:,index]=0
        pred=model.predict(temp)
        pred=np.argmax(pred,axis=1)
        mask_acc[i].append(accuracy_score(i*np.ones_like(pred),pred))

mask_acc=pd.DataFrame(mask_acc)
raw_acc=pd.Series(raw_acc)

mask_acc.to_csv('mask_acc.csv')
raw_acc.to_csv('raw_acc.csv')

if __name__=='__main__':
    pass
