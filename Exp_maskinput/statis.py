# -*- coding: utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from matplotlib import pyplot as plt
import csv
import pandas as pd

def mean_test():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)
    images=mnist.test.images
    labels=np.argmax(mnist.test.labels,axis=1)
    result=[]
    for i in range(10):
        temp=images[labels==i]
        temp=temp.reshape(-1,28,28)
        temp=np.mean(temp,axis=0)
        temp=temp>0.4
        temp=temp.astype(np.float)
        result.append(temp)
        #plt.imshow(temp,cmap='gray')
        plt.imsave('./jpg/{}'.format(i),temp.reshape(28,28),format="jpg",cmap='gray')
        #plt.savefig('./jpg/{}.jpg'.format(i))
        #plt.clf()
    return result


def mean_exp():
    mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    labels=np.argmax(mnist.test.labels,axis=1)

    images_dict={}

    for i in range(10):
        images_dict[i]=mnist.test.images[labels==i]
        print len(images_dict[i])


    sumbylabel=[0 for i in range(10)]
    for i in range(10):
        sumbylabel[i]=len(images_dict[i])

    with open("raw_acc.csv","r") as csvfile:
        read = csv.reader(csvfile)
        #print len(read)
        acc_std=[0 for i in range(10)]
        j=0
        for item in read:
            acc_std[j]=item[1]
            j+=1
        print acc_std


    data=pd.read_csv('mask_acc.csv',index_col=0)
    col=data.columns
    result=[]
    for c in col:
        print c
        print acc_std[int(c)]
        j = int(c)
        print sumbylabel[j]*float(acc_std[j])
        print int(sumbylabel[j]*data[c].values[0]+0.5)
        item=list(map(lambda x:int(int(x*sumbylabel[j]+0.5)<int(float(acc_std[j])*sumbylabel[j]+0.5)),data[c].values))
        #print data[c].values[3]
        imagename="./maskbylabel{}.jpg".format(c)
        item=np.array(item)
        #准确率不下降是黑色0，下降是白色1
        #plt.imsave(imagename,item.reshape(28,28),format="jpg",cmap='gray')
        result.append(item)

    return result



if __name__=='__main__':
    result1=mean_exp()
    result2=mean_test()
    lst=[]
    lst_test=[]
    for (x,y) in zip(result1,result2):
        y=y.reshape(-1)
        temp=x*y
        lst.append((temp==1).sum())
        lst_test.append((y==1).sum())

    np.array(lst)/np.array(lst_test).astype(np.float)
