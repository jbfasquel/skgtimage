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
    #Once more: why ???
    #filled_residues=fill_regions(residues)
    #adj_included,residues=compute_mat_from_residues(residues,filled_residues)

    #adj_included,residues=compute_mat_from_residues(residues,filled_residues)
    ###################
    #Build graph
    adj_included=transitive_reduction_matrix(adj_included)
    #np.save("closure.npy",adj_included)
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


def manage_boundaries(image,roi):
    eroded_roi=sp.ndimage.morphology.binary_erosion(roi,iterations=2).astype(np.uint8)
    inner_boundary=roi/np.max(roi)-eroded_roi
    inner_boundary_values=np.ma.MaskedArray(image,mask=np.logical_not(inner_boundary)).compressed()
    bins=np.arange(np.min(inner_boundary_values),np.max(inner_boundary_values)+2)
    h,b=np.histogram(inner_boundary_values,bins)
    dominant_value=b[np.argmax(h)]
    #print("dominant_value:",dominant_value)
    modified_image=np.ma.MaskedArray(image,mask=inner_boundary).filled(dominant_value)
    if type(image)==np.ma.MaskedArray:
        modified_image=np.ma.MaskedArray(modified_image,mask=np.logical_not(roi))
    return modified_image



'''
def topological_graph_from_residues(residues):
    import scipy as sp; from scipy import ndimage
    filled_residues=fill_regions(residues)
    #Nodes: one node per residue
    g=IrDiGraph()
    for i in range(0,len(filled_residues)): g.add_node(i)
    ###############################################################################################
    #Build edges according to inclusion (checked using the fill_holes and intersection algorithms)
    #For each i and j, with evaluate whether residues(i) in included in filled_residues(j)
    #Inclusion is true if: intersection(residues(i),filled_residues(j)) == residues(i)
    ###############################################################################################
    for i in range(0,len(residues)):
        for j in range(0,len(filled_residues)):
            if i != j:
                #Intersection
                intersection=np.logical_and(residues[i],filled_residues[j])
                #Equality -> inclusion
                if np.array_equal(intersection,residues[i].astype(np.bool)): g.add_edge(i,j)
    ###################
    #To avoid transitive "inclusion", we perform a transitive reduction
    #For example: if C in B and C in A, then the previous algorithm detects that C in A
    # C -> B -> A
    # |         ^       => C -> B -> A
    # ----------|
    ###################
    g=transitive_reduction(g)
    ###################
    # Now, we check if residues intersect some filled_residues without being included
    # If intersection(residues(i),filled_residues(j)) not empty -> with have to split residues !!!
    # For instance, assume C in A, with C inter B != 0 but C not in B
    # B -> A            C1 -> B -> A
    #      ^   ==>                 ^
    # C ---|            C2 --------|
    # In this case, C1=C inter B ; and C2=C-C1 ; we have also to add a new edge
    all_nodes=g.nodes()
    modified_residues={} #{i: region ; k: region}
    new_regions=[] #[region1,region2, ] -> new nodes max+1,max+2,max+3...
    new_nodes=[] #[ 10,11, ...]
    new_edges=[] #[3,2,..] means -> new nodes max+1 -> node3 ; max+2 -> node2 ; ...
    new_nodes_start_index=len(all_nodes) #if g already has 7 nodes from [0,...,6], then following nodes id will be 8,9,....
    for i in range(0,len(all_nodes)):
        for j in range(i+1,len(all_nodes)):
            #We check intersection between unrelated nodes only
            if (i not in recursive_predecessors(g,j) | recursive_successors(g,j)):
                intersection=np.logical_and(filled_residues[i],filled_residues[j]).astype(np.uint8)
                does_intersect=(np.sum(intersection) != 0)
                if does_intersect :
                    #First question: for new node x, do we consider new edge (x,i), or (x,j) ?
                    #The question is what is the target node for the new edge: i or j ?
                    #Considered decision rule:
                    # - the target is i if diff_i=(region_i - intersection) has a hole
                    # In other words: target is i if diff_i != filled(diff_i)
                    #Second question: which node is modified: if target is i, modified_node is j
                    diff_i=filled_residues[i]-intersection
                    test_i=not(np.array_equal(diff_i,fill_region(diff_i)))
                    if test_i :
                        target=i
                        modified_node=j
                    else:
                        target=j
                        modified_node=i
                    #New node
                    new_node_id=new_nodes_start_index+len(new_nodes)
                    new_nodes+=[new_node_id]
                    #New edge
                    new_edges+=[(new_node_id,target)]
                    #New region
                    new_regions+=[intersection]
                    #Modified region
                    modified_residues[modified_node]=residues[modified_node]-intersection


    #####################
    #Apply modifications to residues and to the graph
    all_residues=[]
    for i in range(0,len(residues)):
        if i not in modified_residues:
            all_residues+=[residues[i]]
        else:
            all_residues+=[modified_residues[i]]
    all_residues+=new_regions
    for i in range(0,len(new_nodes)):
        g.add_node(new_nodes[i])
        edge_tuple=new_edges[i]
        g.add_edge(edge_tuple[0],edge_tuple[1])

    return g,all_residues
'''








'''
def topological_graph_from_residues(residues):
    #Nodes: one node per residue
    g=IrDiGraph()
    for i in range(0,len(residues)): g.add_node(i)
    #Build edges according to inclusion (checked using the fill_holes and intersection algorithms)
    for i in range(0,len(residues)):
        for j in range(0,len(residues)):
            if i != j:
                region=sp.ndimage.morphology.binary_fill_holes(residues[j],np.ones(tuple([3]*len(residues[j].shape)))).astype(np.uint8)
                if np.array_equal(np.logical_and(residues[i],region),residues[i].astype(np.bool)) :
                    g.add_edge(i,j)

    #Transitive reduction to remove transitive closure
    reduced_g=__transitive_reduction__(g)
    return reduced_g
'''




'''
#To keep in mind how to run over isomorphism/matchings
print("Common:",list_of_commun_isomorphims)
print("Topological matchings:")
#Iterate over isomorphisms
for i in t_matcher.isomorphisms_iter():
    print("---> Isomorphism: " , i)
#Iterate over matchings
for i in t_matcher.match():
    print("---> Matching: " , i)
print("Photometric matchings:")
#Iterate over isomorphisms
for i in p_matcher.isomorphisms_iter():
    print("---> Isomorphism: " , i)
#Iterate over matchings
for i in p_matcher.match():
    print("---> Matching: " , i)
'''
'''
### Problem with nosetests !!!
@property
def targets(self):
    """
    :getter: doc getter
    :setter: doc setter
    :type: node id
    """
    return self._current_targets

def set_targets(self,t):
    self._current_targets=t
    self.update_subgraphs__()



@targets.setter
def targets(self,t):
    #TODO test that targets in graphs
    print("here")
    self._current_targets=t
    self.update_subgraphs__()
    #self._current_root_node=list(first_segmented_successors(t_graph,n))
    #nodes_belonging_roi=
'''
