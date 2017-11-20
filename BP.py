# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 10:49:48 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

#生成数据
x,y = make_moons(n_samples=200,noise=0.1)
#plt.scatter(x[:,0],x[:,1],c=y)
#plt.show()
#BP实现
nn_input_dim = 2
nn_hidlay_dim = 5
nn_output_dim = 1
num_pas = 10000
epclo = 0.01
#激活函数
def activation_layer(x,dev=False):
    if (dev==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#初始化,正太分布
W1 = np.random.randn(nn_input_dim,nn_hidlay_dim)/np.sqrt(nn_input_dim)
b1 = np.zeros((1,nn_hidlay_dim))
W2 = np.random.randn(nn_hidlay_dim,nn_output_dim)/np.sqrt(nn_hidlay_dim)
b2 = np.zeros((1,nn_output_dim))
y = y.reshape(-1,1)
for i in range(0,num_pas):
    #前向传播
    Z1 = x.dot(W1)+b1
    A1 = activation_layer(Z1)
    Z2 = A1.dot(W2)+b2
    A2 = activation_layer(Z2)
    #返向传播
    dZ2 = A2 - y
    dW2 = (A1.T).dot(dZ2)
    db2 = np.sum(dZ2, axis=0,keepdims=True)
    dZ1 = dZ2.dot(W2.T)*activation_layer(A1,True)
    dW1 = (x.T).dot(dZ1)
    db1 = np.sum(dZ1,axis=0,keepdims=True)
    #权值更新
    W1 += -dW1*epclo
    b1 += -db1*epclo
    W2 += -dW2*epclo
    b2 += -db2*epclo
    
def pred_fun(x):
    Z1 = x.dot(W1)+b1
    A1 = activation_layer(Z1)
    Z2 = A1.dot(W2)+b2
    A2 = activation_layer(Z2)
    return A2

x1_min,x1_max = x[:,0].min()-0.5,x[:,0].max()+0.5
x2_min,x2_max = x[:,1].min()-0.5,x[:,1].max()+0.5
h = 0.01
xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,h),np.arange(x2_min,x2_max,h))
Z = pred_fun(np.c_[xx1.ravel(),xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1,xx2,Z,cmap=plt.cm.Spectral)
plt.scatter(x[:,0],x[:,1],c=y,cmap=plt.cm.Spectral)
plt.show()

y_hat=pred_fun(x)
mse=np.sqrt(np.mean((y_hat-y)**2))
print(mse)
