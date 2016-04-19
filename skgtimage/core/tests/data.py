#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import numpy as np
import skgtimage as skgti


#Region A
A=np.ones((10,10),np.uint8)
#Region B
B=np.zeros((10,10),np.uint8);B[1:9,1:6]=1
#Region C
C=np.zeros((10,10),np.uint8);C[2:4,2:4]=1
#Region D
D=np.zeros((10,10),np.uint8);D[5:8,2:5]=1
#Region E
E=np.zeros((10,10),np.uint8);E[6,3]=1
#Region F
F=np.zeros((10,10),np.uint8);F[1:5,7:9]=1

#IMAGE
image=np.where(A,1,0)
image=np.where(B,2,image)
image=np.where(C,3,image)
image=np.where(D,4,image)
image=np.where(E,5,image)
image=np.where(F,2,image)

#Graph
'''
t_graph=skgti.core.IrDiGraph()
t_graph.add_nodes_from(['A','B','C','D','E','F'])
t_graph.add_edge('B','A');t_graph.add_edge('C','B');t_graph.add_edge('D','B')
t_graph.add_edge('E','D');
t_graph.add_edge('F','A')
'''
'''
t_graph.set_image(image)
print(skgti.core.roi_for_target(t_graph,'E').dtype)

print(np.ones(image.shape).dtype)
'''
#print(image)


