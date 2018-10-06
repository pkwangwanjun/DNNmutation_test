#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


sns.set_style('whitegrid')

def plt1(path):

    data=pd.read_csv(path,index_col=0)
    data_org=data[[i for i in range(0,20,2)]]
    data_mr=data[[i for i in range(1,20,2)]]

    width = 0.5
    idx = np.arange(data_org.shape[-1])

    data1=data_org.values.reshape(-1)
    data2=data_mr.values.reshape(-1)

    p1=plt.bar(idx,data1,width,label='orignal')
    p2=plt.bar(idx,data2,width,label='augment')
    #plt.ylim((0.95, 1.0))
    plt.xlabel('mnist categore')
    plt.ylabel('accuracy')

    for a,b in zip(idx,data1):
        plt.text(a, b, '{:.4f}'.format(b), ha='center', va= 'bottom',fontsize=12)
    for a,b,c in zip(idx,data2,data1):
        plt.text(a, b, '{:.4f}'.format(b), ha='center', va= 'bottom',fontsize=12)

    plt.ylim((0.95, 1.0))
    plt.xticks(idx, ('digit:{}'.format(i) for i in range(10)))
    plt.legend((p1[0], p2[0]), ('orignal', 'augment'))
    plt.title(os.path.basename(path).split('.')[0])


    plt.savefig('{}.jpg'.format(os.path.basename(path).split('.')[0]))
    plt.clf()

def plt2():
    noaug=pd.read_csv('noaug.csv',index_col=0)
    width = 0.2
    bar_width = 0.2
    idx = np.arange(len(data_org))

    data1=data['all_org'].values
    data2=noaug.values.reshape(-1)

    p1=plt.bar(idx,data1,width,label='orignal')
    p2=plt.bar(idx+bar_width,data2,width,label='augment')

    #for a,b in zip(idx,data1):
    #    plt.text(a, b+0.0025, '{:.4f}'.format(b), ha='center', va= 'bottom',fontsize=12)
    #for a,b,c in zip(idx,data2,data1):
    #    plt.text(a+bar_width, b, '{:.4f}'.format(b), ha='center', va= 'bottom',fontsize=12)

    #p1=plt.plot(data['all_org'].values)
    #p2=plt.plot(noaug.values.reshape(-1))
    plt.xticks(idx, ('index:{}'.format(i) for i in range(1,11)))
    plt.legend((p1[0], p2[0]), ('orignal', 'augment'))
    plt.ylim((0.95, 1.0))

if __name__=='__main__':
    for path in ['fat.csv','thin.csv']:
        plt1(path)
