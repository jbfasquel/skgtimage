#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import numpy as np

image=np.ones((10,10),np.uint8)
for i in range(0,10): image[:,i]=10*i
#Region A
A=np.ones((10,10),np.uint8)
#Region B
B=np.zeros((10,10),np.uint8);B[1:9,1:5]=1
#Region C
C=np.zeros((10,10),np.uint8);C[2:4,2:4]=1
#Region D
D=np.zeros((10,10),np.uint8);D[6:8,2:4]=1
#Region E
E=np.zeros((10,10),np.uint8);E[1:5,6:9]=1

