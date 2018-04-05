# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 18:04:59 2018

@author: Administrator
"""



#gradient descent is basically machine learning 101
#gradient refers to the derivative of a matrix
#the basics of gradient descent is similar to OLS
#they are using the same approach to minimize the mean square error
#the only difference is that OLS tries to get the global optimized result
#the parameter beta for a linear regression y=x*beta+error in OLS is (x'x)^(-1)xy in matrix form
#for gradient descent, it uses iterations to get the local optimized result
#the parameter beta for the same equation y=x*beta+error is calculated via iterations
#each round we readjust the parameter beta by deducting the previous error times a learning rate times its x
#we use readjusted parameters to minus y, then we get a new error
#we use this the new error to get new parameter etc
#when the difference between error of this lag and error from the previous lag is smaller than the stop threshold
#we reach the local optimization
#the control of stop threshold and learning rate is crucial
#if the learning rate is too large, we may not be able to converge at all, we would oscillate around the optmization
#if the learning rate is too small, it takes forever to get there
#the stop threshold determines how optmized our results would be
#the smaller, the better
#however, it also takes longer, maybe even run out of datasets
#more details can be found from machine learning course by Andrew Ng
#for this case, i am using stochastic gradient descent
#it is a bootstrapping version of gradient descent
#it doesnt follow the ascending order of datasets to converge
#it is stochastic, ofc
#thats why it works better on discrete samples instead of time series
#any autocorrelation would cause a bias
import datetime as dt
import random as rd
from statsmodels.api import add_constant as ad
from statsmodels.api import OLS
import numpy as np
import pandas as pd

#reading datasets from files
import os
os.chdir('d:/')
os.getcwd()
a=pd.read_csv('1.csv')
b=pd.concat([a['x1'],a['x2']],axis=1)

#the first part is OLS
#codes are so simple and powerful
#i also intend to see which method is faster
#so i use datetime module to see how fast both OLS and SGD print outputs
now=dt.datetime.now()
#adding a constant vector to x
x=ad(b)
y=a['y']
m=OLS(y,x).fit()
time=dt.datetime.now()
print(m.summary())
#


#firstly, i initialize all the variables
#all pandas dataframes should be converted to matrices
X=np.matrix(x.values)
Y=np.matrix(y.values)
#stop threshold should always be smaller learning rate
stop=0.00001
#i also need a variable to stop the loop when iterations reach at certain limit
#in this case i just dont wanna run outta samples
#i cap the maxloop at the size of x
#but for stochastic gradient descent, its unnecessary
#you will see later
maxloop=len(b.index)
#learning rate usually takes 0.05,0.03,0.01,0.005,0.003,0.001 etc
#we should try different parameters to see which one is a better balance between time and optimization
learn=0.001
#theta is the coefficient for each variable
#in linear regression its called beta
#its just a greek name, doesnt matter which one
#note that i am using two x, so i have three thetas
#two for x, one for the constant
theta=np.mat(np.zeros((1,3)))
#we need to compare two errors throughout the iterations
#thats why i keep two in the matrix
ero=np.mat(np.zeros((2,1)))
#i wanna make sure every element in matrix is float
ero.value=theta.value=float
#this is the counter to avoid indefinite loop
count=0

#


now1=dt.datetime.now()

#this is a bootstrapping process
#i take random numbers from 0 to the maximum sample size
#i set the random number list at 3000
#cuz i believe it will reach before 3000
#for stochastic gradient descent, i am using this random number list to control the loop
#when samples run out, its done
#for normal gradient descent, we should implement while counter<=maxloop to stop the loop
j=[]
for i in range(3000):
    k=rd.randint(0,maxloop)
    j.append(k)


for i in j:
#the error calculation is very straight forward
#for y=x*beta+error, its error=y-x*beta
    ero[0]=np.dot(X[i,:],theta.T)-Y[:,i]
    #for each coefficient, we use error and learning rate adjusted x to obtain the iterated coefficient
    theta[0,0]-=ero[0]*learn*X[i,0]
    theta[0,1]-=ero[0]*learn*X[i,1]
    theta[0,2]-=ero[0]*learn*X[i,2]
        
#when the difference between two errors is smaller than stop threshold
#we break the loop and declare the local optimization       
    if np.abs((ero[0]**2)*0.5-(ero[1]**2)*0.5)<=stop:
        
        break
        
    ero[1]=ero[0]
    count+=1
   
      

time1=dt.datetime.now()

#at last, we calculate the time for OLS and SGD
t1=float((time-now).microseconds)
t2=float((time1-now1).microseconds)
#


#from my experiments
#SGD is always faster in a large sample size
#as the inverse of the matrix becomes more difficult to get for OLS
#but the optimization is not so optimistic
#OLS identifies a better data causal relationship
#if the sample size is smaller than 1 mil
#its better to use OLS
print('\nmachine learning:\n')
print('iteration:',count,'\n')
print('parameters1:%f,parameters2:%f,parameters3:%f'%(theta[0,0],theta[0,1],theta[0,2]))
print('\ntime used:',t2,'\n')
print('\nlinear regression:\n')
print('parameters1:%f,parameters2:%f,parameters3:%f'%(m.params[0],m.params[1],m.params[2]))
print('\ntime used:',t1,'\n')
print('\ndifference:\n')
print('time difference in microseconds:',(t2-t1))
print('parameters1:%f,parameters2:%f,parameters3:%f'%(m.params[0]-theta[0,0],m.params[1]-theta[0,1],m.params[2]-theta[0,2]))