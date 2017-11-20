# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 14:09:06 2017

@author: Administrator
"""

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
n_samples = 5000
x,y = make_blobs(n_samples=n_samples,random_state=26,cluster_std=0.7)
plt.subplot(211)
plt.scatter(x[:,0],x[:,1],c=y)
#初始化中心点位置
cent = [[np.random.uniform(x[:,0].min(),x[:,0].max()), np.random.uniform(x[:,1].min(),x[:,1].max())] for i in range(0,3)]
cent1 = np.array(cent)
plt.scatter(cent1[:,0],cent1[:,1])
#根据中心点位置分类
def countlab(cent):
    xdis = np.zeros((n_samples,3))
    for i in range(0,3):
        for j in range(0,n_samples):
            xdis[j][i] = np.linalg.norm(cent[i]-x[j])
    lab = np.argmin(xdis,axis=1)
    return lab
#根据分类 确定新的中心点
def countcent(lab):
    newcent = np.zeros([3,2])
    xx = np.column_stack((x,lab))
    for i in range(0,3):
        x1 = np.where(xx[:,2]==i)
        x1 = xx[x1,0:2]
        newcent[i] = np.mean(x1,axis=1)
    return newcent

#迭代
num_pas = 100
cent = cent1
for i in range(0,num_pas):
    lab = countlab(cent)
    cent = countcent(lab)
plt.subplot(212)
plt.scatter(x[:,0],x[:,1],c=lab)
plt.scatter(cent[:,0],cent[:,1])
plt.show()

    
