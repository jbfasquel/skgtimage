#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
import scipy as sp; from scipy import ndimage
import networkx as nx
from skgtimage.core.graph import IrDiGraph,transitive_reduction,mat2graph,graph2mat,transitive_reduction_matrix

def fill_regions(regions):
    filled_regions=[fill_region(r) for r in regions]
    return filled_regions

def fill_region(r):
    import scipy as sp; from scipy import ndimage
    se=sp.ndimage.morphology.generate_binary_structure(len(r.shape),1) #connectivity (4 in 2D or 6 in 3D: diagonal elements are not neighbors)
    filled=sp.ndimage.morphology.binary_fill_holes(r,se).astype(np.uint8)
    return filled

def residues_from_labels(labelled_image):
    tmp_label=labelled_image+1 #to avoid confusion with 0s from masked area (roi)
    min_label=np.min(tmp_label)
    max_label=np.max(tmp_label)
    #residues=[np.where(tmp_label==i,1,0) for i in range(min_label,max_label+1)]
    residues=[]
    for i in range(min_label,max_label+1):
        tmp=np.where(tmp_label==i,1,0)
        if np.sum(tmp)>0: residues+=[tmp]
    return residues

def topological_graph_from_labels(labelled_image):
    '''
    tmp_label=labelled_image+1 #to avoid confusion with 0s from masked area (roi)
    min_label=np.min(tmp_label)
    max_label=np.max(tmp_label)
    residues=[np.where(tmp_label==i,1,0) for i in range(min_label,max_label+1)]
    for r in residues:
        print("res: ", np.min(r),np.max(r))
    '''
    residues=residues_from_labels(labelled_image)
    return topological_graph_from_residues(residues,copy=False)


def inclusions_from_residues(residues,filled_residues):
    ########################
    #Discovering inclusions and intersections relationships (adj matrices) between residues and filled residues
    # M[i,j]=1 <-> ri included in rj <-> edge i->j
    #     r_0 r_1 r_2
    # r_0  0   x   x
    # r_1  x   0   x
    # r_2  x   x   0
    ###################
    n=len(filled_residues)
    adj_included=np.zeros((n,n),dtype=np.uint8) #why not 'adj_included=np.eye(n,dtype=np.uint8)'
    adj_intersection=np.zeros((n,n),dtype=np.uint8) #why not 'adj_intersection=np.eye(n,dtype=np.uint8)'

    for i in range(0,n):
        for j in range(0,n):
            if i != j:
                #Intersection
                intersection=np.logical_and(residues[i],filled_residues[j])
                does_intersect=int((np.sum(intersection) != 0))
                adj_intersection[i,j]=does_intersect
                #Inclusion
                is_included=int(np.array_equal(intersection,residues[i].astype(np.bool)))
                adj_included[i,j]=is_included
    ###################
    #Transitive reduction of the inclusion graph
    ###################
    adj_included=transitive_reduction_matrix(adj_included)
    ###################
    #Management of intersections without inclusion
    ###################
    split_matrix=adj_intersection-adj_included
    recurse=False
    if np.max(split_matrix) > 0: #if at least one element is not null
        (nzeros_i,nzeros_j)=np.nonzero(split_matrix)
        for n in range(0,len(nzeros_i)):
            i_residue=nzeros_i[n]
            j_region=nzeros_j[n]
            intersection=np.logical_and(residues[i_residue],filled_residues[j_region]).astype(np.uint8)
            #New region
            new_residue=residues[i_residue]-intersection
            if (np.max(new_residue)>0) and (np.max(intersection)>0): #if residue is not empty
                recurse=True
                residues[i_residue]=new_residue
                residues+=[intersection.astype(np.uint8)]
                filled_residues[i_residue]=fill_region(new_residue)
                filled_residues+=[fill_region(intersection)]

    ###################
    #Recursive invocation if the management of intersections has led to new residues/filled_residues
    ###################
    if recurse:
        adj_included,residues=inclusions_from_residues(residues,filled_residues)

    return adj_included,residues


def topological_graph_from_residues(residues,copy=True):
    import scipy as sp; from scipy import ndimage
    working_residues=residues
    if copy:
        working_residues=[np.copy(r) for r in residues]
    filled_residues=fill_regions(working_residues)
    adj_included,working_residues=inclusions_from_residues(working_residues,filled_residues)

    ###################
    #Build graph
    adj_included=transitive_reduction_matrix(adj_included)
    g=IrDiGraph()
    #Nodes
    for i in range(0,adj_included.shape[0]): g.add_node(i)
    #Edges
    for i in range(0,adj_included.shape[0]):
        for j in range(0,adj_included.shape[1]):
            if (i != j) and (adj_included[i,j]!=0):
                g.add_edge(i,j)
    #g=transitive_reduction(g)
    nodes=g.nodes()
    for i in range(0,len(nodes)):
        filled_r=fill_region(working_residues[i])
        g.set_region(nodes[i],filled_r)

    return g,working_residues



