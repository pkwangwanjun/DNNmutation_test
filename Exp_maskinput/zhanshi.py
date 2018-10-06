#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 12:16:39 2018

@author: qq
"""
'''
bias
1
0  0.5352 0.184 0.1196  0.0394

3
0  0.333 0.1074 0.0914 0.0296

5
0 0.5908 0.1896 0.0982 0.0236

10
0 0.6426 0.1982 0.14 0.0538



kernel

1
0.208 0.2402 0.1684 0.0282 0.0042
3
0.1526 0.19040000000000007 0.0606 0.014599999999999998  0.0038000000000000004
5
0.1262  0.14540000000000003 0.042800000000000005 0.011600000000000003 0.003 
10
0.19880000000000003  0.23220000000000005 0.1292 0.0182 0.0036000000000000003
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import array
from tensorflow.examples.tutorials.mnist import input_data
'''
biasxlist =[5,10,20,30,50]#倍数
kernelxlist =[0.01,0.1,3,5,10]
ratiobias1=[0,0.5352,0.184,0.1196,0.0394]
ratiobias3=[0,0.333,0.1074,0.0914,0.0296]
ratiobias5=[0,0.5908,0.1896,0.0982,0.0236]
ratiobias10=[0.4214,0.1472,0.0596,0.03,0.0056]

ratiokernel1 =[0.208,0.2402,0.1684,0.0282,0.0042]
ratiokernel3=[0.1526,0.1904,0.0606,0.0146,0.0038]
ratiokernel5=[0.1262,0.1454,0.0428,0.0116,0.003]
ratiokernel10=[0.1006,0.1096,0.0356,0.0088,0.0026]
'''
'''
plt.title('Results of modifying weights to disturb DNN model')
plt.xlabel("Modify weights by extent(0.01,0.1,3,5,10)")
plt.ylabel("Ratio of weights needed to modify")
plt.plot(kernelxlist,ratiokernel1, color='green', label='dnn_1')
plt.plot(kernelxlist,ratiokernel3, color='red', label='dnn_3')
plt.plot(kernelxlist,ratiokernel5, color='black', label='dnn_5')
plt.plot(kernelxlist,ratiokernel10, color='blue', label='dnn_10')
plt.legend()
plt.show()
'''
'''
plt.title('Results of modifying biases to disturb DNN model')
plt.xlabel("Modify biases by extent(5,10,20,30,50)")
plt.ylabel("Ratio of biases needed to modify")
plt.plot(biasxlist,ratiobias1, color='green', label='dnn_1')
plt.plot(biasxlist,ratiobias3, color='red', label='dnn_3')
plt.plot(biasxlist,ratiobias5, color='black', label='dnn_5')
plt.plot(biasxlist,ratiobias10, color='blue', label='dnn_10')
plt.legend()
plt.show()
'''
'''
avgkillinput=[[0 for j in range(10)] for i in range(12)]
axis =[i for i in range(1,11)]
for index in [1,3,5,10]:
    data=pd.read_csv('./mut{}.csv'.format(index),index_col=0)
    col=data.columns
    #print data[col[0]].values[63:67]
    #print len(data[col[0]].values)
    sumf=0
    for d in data[col[0]].values[0:784]: 
        sumf+=d
    print 1.0*sumf/784
    
    for i in range(index):
        sum =0
        for d in data[col[0]].values[784+i*64:784+i*64+64]: 
            #print d
            sum+=d
        print 1.0*sum/64
        avgkillinput[index][i]=1.0*sum/64
plt.title('Results of deleting one neuron in hidden layer(DHN)')
plt.xlabel("Index of hidden layer in 4 dnns")
plt.ylabel("Average number of test inputs killing DHN mutants")
plt.plot(axis,avgkillinput[1], color='green', label='dnn_1')
plt.plot(axis,avgkillinput[3], color='red', label='dnn_3')
plt.plot(axis,avgkillinput[5], color='black', label='dnn_5')
plt.plot(axis,avgkillinput[10], color='blue', label='dnn_10')
plt.legend()
plt.show()
'''
'''
avgkillinput=[[] for i in range(12)]
axis =[i for i in range(1,11)]
for index in [1,3,5,10]:
    data=pd.read_csv('./mut{}.csv'.format(index),index_col=0)
    col=data.columns
    #print data[col[0]].values[63:67]
    #print len(data[col[0]].values)
    sumf=0
    for d in data[col[0]].values[0:784]: 
        sumf+=d
    print 1.0*sumf/784
    
    for i in range(index):
        sum =0
        for d in data[col[0]].values[784+i*64:784+i*64+64]: 
            #print d
            sum+=d
        print 1.0*sum/64
        avgkillinput[index].append(1.0*sum/64)
plt.title('Results of deleting one neuron in hidden layer(DHN)')
plt.xlabel("Index of hidden layer in 4 dnns")
plt.ylabel("Average number of test inputs killing DHN mutants")
#plt.plot([1],avgkillinput[1], color='green', label='dnn_1')
plt.plot([1,2,3],avgkillinput[3], color='red', label='dnn_3')
plt.plot([1,2,3,4,5],avgkillinput[5], color='black', label='dnn_5')
plt.plot(axis,avgkillinput[10], color='blue', label='dnn_10')
plt.legend()
plt.show()
'''

#for i in [1,3,5,10]
'''
index =10
data=pd.read_csv('./mut{}.csv'.format(index),index_col=0)
col=data.columns
    #print data[col[0]].values[63:67]
    #print len(data[col[0]].values)
        
def draw_hist(killinput):
    plt.hist(killinput,100)
    plt.xlabel('Number of test inputs that can kill certain DHN mutant')
    #plt.xlim(0.0,30)
    plt.ylim(0,150)
    plt.ylabel('Frequency')
    plt.title('Results of DHN mutation(dnn_{})'.format(index))
    plt.show()

draw_hist(data[col[0]].values[784:784+64*index])
'''
'''
for index in [1,3,5,10]:
    data=pd.read_csv('./kernel_{}.csv'.format(index),index_col=0)

    col=data.columns

    for c in col:
        #print c
        lst=list(map(lambda x:float(x[1:-1].split(',')[0].strip()),data[c].values))
        mean=sum(lst)/len(lst)
        stdv = np.std(lst)
        cv=stdv/mean
        print('layer:{},extent:{},mean(cv):{}({})'.format(index,c,mean,cv))
        #print c,"&",mean,'(',stdv,')','&',mean,'(',stdv,')'
'''

mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

labels=np.argmax(mnist.test.labels,axis=1)

images_dict={}

for i in range(10):
    images_dict[i]=mnist.test.images[labels==i]
    print len(images_dict[i])


sumbylabel=[0 for i in range(10)]
for i in range(10):
    sumbylabel[i]=len(images_dict[i])
print  sumbylabel 
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
    plt.imsave(imagename,item.reshape(28,28),format="jpg",cmap='gray')

