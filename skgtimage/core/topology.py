#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
from skgtimage.core.graph import IrDiGraph,transitive_reduction_adjacency_matrix
from skgtimage.core.search_base import search_head_nodes


def fill_region(r):
    import scipy as sp; from scipy import ndimage
    se=sp.ndimage.morphology.generate_binary_structure(len(r.shape),1) #connectivity (4 in 2D or 6 in 3D: diagonal elements are not neighbors)
    filled=sp.ndimage.morphology.binary_fill_holes(r,se).astype(np.uint8)
    return filled

def is_included_in(a,b): #True if a included in b
    filled_b=fill_region(b)
    normed_a=a/np.max(a)
    intersection = np.logical_and(normed_a, filled_b)
    is_included=int(np.array_equal(intersection,normed_a.astype(np.bool)))
    return is_included

def inclusions_from_regions(regions, filled_regions):
    ########################
    #Discovering inclusions and intersections relationships (adj matrices) between residues and filled residues
    # M[i,j]=1 <-> ri included in rj <-> edge i->j
    #     r_0 r_1 r_2
    # r_0  0   x   x
    # r_1  x   0   x
    # r_2  x   x   0
    ###################
    n=len(filled_regions)
    adj_included=np.zeros((n,n),dtype=np.uint8) #why not 'adj_included=np.eye(n,dtype=np.uint8)'
    adj_intersection=np.zeros((n,n),dtype=np.uint8) #why not 'adj_intersection=np.eye(n,dtype=np.uint8)'

    for i in range(0,n): regions[i]= regions[i] / np.max(regions[i])
    for i in range(0,n):
        for j in range(0,n):
            if i != j:
                #Intersection
                intersection=np.logical_and(regions[i], filled_regions[j])
                does_intersect=int((np.sum(intersection) != 0))
                adj_intersection[i,j]=does_intersect
                #Inclusion
                is_included=int(np.array_equal(intersection, regions[i].astype(np.bool)))
                adj_included[i,j]=is_included
    ###################
    #Transitive reduction of the inclusion graph
    ###################
    adj_included=transitive_reduction_adjacency_matrix(adj_included)
    #adj_intersection=transitive_reduction_matrix(adj_intersection)
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
            intersection=np.logical_and(regions[i_residue], filled_regions[j_region]).astype(np.uint8)
            #New region
            new_residue= regions[i_residue] - intersection
            if (np.max(new_residue)>0) and (np.max(intersection)>0): #if residue is not empty
                recurse=True
                regions[i_residue]=new_residue
                regions+=[intersection.astype(np.uint8)]
                filled_regions[i_residue]=fill_region(new_residue)
                filled_regions+=[fill_region(intersection)]

    ###################
    #Recursive invocation if the management of intersections has led to new residues/filled_residues
    ###################
    if recurse:
        adj_included, regions=inclusions_from_regions(regions, filled_regions)

    return adj_included, regions

def topological_graph_from_regions(regions, copy=True):
    working_residues=regions
    if copy:
        working_residues=[np.copy(r) for r in regions]
    filled_residues=[fill_region(r) for r in working_residues]
    adj_included,working_residues=inclusions_from_regions(working_residues, filled_residues)

    ###################
    #Build graph
    adj_included=transitive_reduction_adjacency_matrix(adj_included)
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
        g.set_region(nodes[i],working_residues[i])

    return g,working_residues

def topological_merging_candidates(graph,source):
    #Source must be child of target, or target child of source, or both must be brother (i.e. sharing common father)
    #Note that brothers include heads of a topological graph
    #TO DO: Pour image03_region_top_ok_meanshift_5classes_vs4classes_refactorying: permettra d'avoir les "trous" de "ville fleurie" correctement identified
    candidates=set(graph.predecessors(source))
    if len(graph.successors(source)) !=0:
        father=graph.successors(source)[0]
        candidates|=set([father])
        candidates|=set(graph.predecessors(father)) #brothers
        #candidates|=set(recursive_predecessors(graph,father))

    #If source is a head, we include other heads
    heads=search_head_nodes(graph)
    if source in heads:
        candidates|=heads

    candidates-=set([source])
    return candidates

def merge_nodes_topology(graph,source,target):
    ##############
    #Check that merge is relevant
    #Source must be child of target, or target child of source, or both must be brother (i.e. sharing common father)
    condition=(target in topological_merging_candidates(graph,source))
    if not(condition): raise Exception("Impossible merge")
    ##############
    #Update region
    new_target_region=np.logical_or(graph.get_region(source),graph.get_region(target))
    graph.set_region(target,new_target_region)

    ##############
    #Update edges
    edges_to_add=[]
    edges_to_remove=[]
    #Case 1 : remove (source,target) if it is an existing edge -> -= (source,target)
    if (source,target) in graph.edges():
        edges_to_remove+=[(source,target)]
    #Case 2 : if target is a predecessor of source and if source has a successor (source father) -> += (target,source_father)
    if (target in graph.predecessors(source)) and (len(graph.successors(source))!=0):
        edges_to_add+=[(target,graph.successors(source)[0])]
    #Case 3 : all predecessors differing from target (otherwise self edge) +=(child,target) AND -= (child,source)
    for pred in graph.predecessors(source):
        if pred != target: #if target is a predecessor
            edges_to_add+=[(pred,target)]
        edges_to_remove+=[(pred,source)]
    #Add new edges
    for e in edges_to_add:
        graph.add_edge(e[0],e[1])
    #Remove old edges
    for e in edges_to_remove:
        graph.remove_edge(e[0],e[1])
    #Remove node
    graph.remove_node(source)

    ##############
    # Manage particular case of heads that, after merge, depict on inclusion relationship
    # For each head of the topological graph, we test if this head (i.e. the related region) is included
    # within another head (i.e. the related region): in such a case, we add the appropriate inclusion relationship (i.e. graph edge)
    heads=search_head_nodes(graph)
    l_heads=list(heads)
    for i in range(0,len(l_heads)):
        for j in range(0,len(l_heads)):
            if i != j:
                r_i=graph.get_region(l_heads[i])
                r_j=graph.get_region(l_heads[j])
                if is_included_in(r_i,r_j):
                    graph.add_edge(l_heads[i],l_heads[j])
