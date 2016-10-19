# encoding: utf-8
'''
Created on 13 oct. 2016

@author: Miao1
'''

#Copy from http://python.jobbole.com/82758/

import numpy as np
from scipy._lib.six import xrange

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
 
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
 
# output dataset            
z = np.array([[0,0,1,1]]).T
 
# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
 
# initialize weights randomly with mean 0
poid = 2*np.random.random((3,1)) - 1
 
for iter in xrange(10000):
    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,poid))
 
    # how much did we miss?
    l1_error = z - l1
 
    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)
 
    # update weights
    poid += np.dot(l0.T,l1_delta)
print("Output After Training:")
print(l1)