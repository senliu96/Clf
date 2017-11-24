# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:36:03 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

x,y = make_moons(n_samples=200,noise=0.2)
plt.figure(figsize=(12,12))
plt.subplot(311)
plt.scatter(x[:,0],x[:,1],c=y)
plt.title("input")

def logit(x):
    return 1/(1+np.exp(-x))

x = np.column_stack((x,np.ones((x.shape[0],
y1=y1),dtype='float32')))
y = y.reshape(-1,1)
alfa = 0.01
m,n = x.shape
W = np.zeros((n,1))
num_pas = 1000
J=pd.Series(np.arange(num_pas),dtype='float32')
for i in range(0,num_pas):
    h = logit(x.dot(W))
    J[i] = -(1/200)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
    error = h - y
    grad = (x.T).dot(error)
    W += -grad*alfa
plt.subplot(312)
J.plot()
def pred_fun(x):
    x = np.column_stack((x,np.ones((x.shape[0],1),dtype='float32')))
    h = logit(x.dot(W))
    return h
x1_min,x1_max = x[:,0].min()-0.5,x[:,0].max()+0.5
x2_min,x2_max = x[:,1].min()-0.5,x[:,1].max()+0.5
h=0.01
xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
Z = pred_fun(np.c_[xx1.ravel(),xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.subplot(313)
plt.co1ntourf(xx1,xx2,Z)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()

