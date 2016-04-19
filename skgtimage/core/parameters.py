"""
Image specific graph search algorithms for finding parameters (i.e. ROI, classes, intervals) for a given target t:
 * Region of interest
 * List of region nodes within the ROI
 * List of region nodes of distinct photometric properties within the ROI
 * List of photometric properties (intervals or "fixed value") for each distinct region belonging the ROI
"""
#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
from skgtimage.core.search_base import recursive_predecessors,recursive_successors,recursive_brothers
from skgtimage.core.search_filtered import  recursive_segmented_predecessors, \
                                            recursive_segmented_successors,\
                                            first_segmented_successors,\
                                            recursive_predecessors_until_first_segmented,\
                                            first_segmented_predecessors,\
                                            recursive_segmented_brothers
from skgtimage.core.photometry import region_stat

#################################
# RESIDUE (TOPOLOGICAL INFORMATION), REGION OF INTEREST (ROI) AND LIST OF REGIONS BELONGING TO THE ROI
#################################

def residue(g,n):
    """
    Compute the region residue of the a node. The must considered node must correspond to an already segmented
    region, otherwise None is returned

    :param g: graph
    :param n: node
    :return: numpy array or None
    """
    residue=None
    root_region=g.get_region(n)
    if root_region is not None:
        included_nodes=recursive_segmented_predecessors(g,n)
        to_remove=np.zeros(root_region.shape,np.bool)
        for i in included_nodes:
            to_remove=np.logical_or(to_remove,g.get_region(i))
        residue=np.logical_and(root_region,np.logical_not(to_remove))
    return residue

def is_residue_valid(g,n):
    """
    Return true is directly included nodes (first predecessors) are already segmented

    :param g: topological graph
    :param n: node
    :return:
    """
    direct_children=g.predecessors(n)
    is_brother_candidate=True
    for c in direct_children:
        if g.get_region(c) is None:
            is_brother_candidate=False
    return is_brother_candidate
    #if is_brother_candidate: set_of_valid_brothers |= set(b)


def root_tree_node(g):
    """

    :param g: a tree graph (topological)
    :return:
    """
    root=None
    for n in g.nodes():
        succ=recursive_successors(g,n)
        if len(succ) == 0: root=n
    return root


def roi_for_targets(g,nodes,shape):
    """
    Compute the region of interest (ROI) for several targets (to be simultaneously segmented): union of ROI for each specified node.

    :param g: graph
    :param nodes: list of nodes
    :return: roi (numpy array)
    """
    roi=np.zeros(shape)
    for n in nodes: roi=np.logical_or(roi,roi_for_target(g,n,shape))
    return roi

def roi_for_target(g,n,shape):
    """
    Compute the region of interest (ROI) for a given target. If nothing is already segmented, the
    ROI is the entire image to be analyzed (an exception is raised if no image is assigned to the graph).

    :param g: graph
    :param n: node corresponding to the target
    :return: roi (numpy array)
    """

    root=list(first_segmented_successors(g,n))
    if len(root) == 0:
        roi=np.ones(shape)
    else:
        roi=residue(g,root[0])
    return roi

def classes_for_targets(g,nodes):
    """
    Find the set of classes involved within the ROI for several targets (to be simultaneously segmented): union of set for each specified node.

    :param g: graph
    :param nodes: list of nodes
    :return: set of Ids (node names)
    """
    classes=set()
    for n in nodes: classes = classes | classes_for_target(g,n)
    return classes

def classes_for_target(g,n=None):
    """
    Find the set of classes involved within the ROI for the target n

    :param g: graph
    :param n: node corresponding to the target
    :return: set of Ids (node names)
    """
    if n is None:
        root_class=root_tree_node(g)
        classes=set([root_class]) | recursive_predecessors_until_first_segmented(g,root_class)
        return classes
    root=list(first_segmented_successors(g,n))
    if len(root) == 0:
        #raise Exception('No root region: classes cannot be determined')
        return classes_for_target(g,None)
    else:
        root_class=root[0]
        classes=set([root_class]) | recursive_predecessors_until_first_segmented(g,root_class)
        return classes

def distinct_classes(nodes,graphs):
    """


    :param nodes:
    :param graphs: typically photometric graphs
    :return:
    """
    #Nodes of interest are none similar nodes assuming that two similar nodes are simultaneously similar for each graph
    nodes_of_interest=set([nodes[0]])
    for i in range(1,len(nodes)):
        c_node=nodes[i]
        #Find common 'brothers' along each photometric graph
        c_brothers=recursive_brothers(graphs[0],c_node)
        for component in range(1,len(graphs)):
            c_brothers &= recursive_brothers(graphs[component],c_node) #common brother -> set intersection
        #If nodes of interest (in construction during this loop) do already contain a 'brother'
        #then we consider that the current node belongs to 'nodes of interest'
        if len(nodes_of_interest & c_brothers) == 0: nodes_of_interest|=set([c_node])
    return nodes_of_interest

def interval_for_classes(image,roi,nodes,t_graph,graphs):
    """

    :param nodes: only (photometrically) distinct nodes (i.e. distinct vs graphs)
    :param graphs:
    :return:
    """
    #Extremal values (min/max within ROI)
    if len(graphs) == 1: #mono-component image (grayscale)
        min_roi=[region_stat(image,roi,fct=np.min,mc=False)]
        max_roi=[region_stat(image,roi,fct=np.max,mc=False)]
    else:
        min_roi=region_stat(image,roi,fct=np.min,mc=True)
        max_roi=region_stat(image,roi,fct=np.max,mc=True)
    #Research of interval
    spatial_shape=image.shape[0:len(graphs)-1]
    class2constraint={}
    for n in nodes:
        class2constraint[n]=[]
        for current_component in range(0,len(graphs)):
            valid_brothers=find_equals(t_graph,graphs[current_component],n)
            if len(valid_brothers):
                region=np.zeros(spatial_shape)
                for b in valid_brothers:
                    region=np.logical_or(region,residue(t_graph,b))
                #value=region_stat(self.get_image(),region,component=current_component)
                if len(graphs)>1:
                    value=region_stat(image,region,mc=True)[current_component]
                else:
                    value=region_stat(image,region,mc=False)
                #print("Node ", n , " has valid brothers ", valid_brothers, " value: " ,value)
                class2constraint[n]+=[[value,value]]
            else:
                #Min boundary interval
                value_inf=min_roi[current_component]
                valid_inf=find_supinf(t_graph,graphs[current_component],n,fct=first_segmented_predecessors)
                if len(valid_inf) !=0:
                    #region=np.zeros(self.image.shape)
                    region=np.zeros(spatial_shape)
                    for b in valid_inf: region=np.logical_or(region,residue(t_graph,b))
                    if len(graphs)>1:
                        candidate_value_inf=region_stat(image,region,mc=True)[current_component]
                    else:
                        candidate_value_inf=region_stat(image,region,mc=False)
                    #candidate_value_inf=region_stat(self.get_image(),region,component=current_component)
                    if candidate_value_inf > value_inf : value_inf=candidate_value_inf
                #Max boundary interval
                value_sup=max_roi[current_component]
                valid_sup=find_supinf(t_graph,graphs[current_component],n,fct=first_segmented_successors)
                if len(valid_sup) !=0:
                    #region=np.zeros(self.image.shape)
                    region=np.zeros(spatial_shape)
                    for b in valid_sup: region=np.logical_or(region,residue(t_graph,b))
                    if len(graphs)>1:
                        candidate_value_sup=region_stat(image,region,mc=True)[current_component]
                    else:
                        candidate_value_sup=region_stat(image,region,mc=False)
                    #candidate_value_sup=region_stat(self.get_image(),region,component=current_component)
                    if candidate_value_sup < value_sup : value_sup=candidate_value_sup
                #print("Node ", n , " inf ",valid_inf)
                class2constraint[n]+=[[value_inf,value_sup]]
    return class2constraint


def find_supinf(t_graph,p_graph,node,fct=first_segmented_predecessors):
    """
    Search for either first predecessors or successors of the node parameter, where kept nodes correspond to already
    segmented regions whose topological predecessors are also already segmented. This is required
    to compute a valid photometric statistics within the a priori defined residue.

    :param t_graph: topological graph
    :param p_graph: photometric graph
    :param node: node
    :return: set of nodes satisfying constraints
    """
    #Finding first possible already segmented brothers without unsegmented subnodes
    candidates=fct(p_graph,node)
    set_of_valid_brothers=set()
    for b in candidates:
        #Search for direct topological children, and check if they are segmented
        if is_residue_valid(t_graph,b) : set_of_valid_brothers |= set([b])
    # Recursive search
    if len(set_of_valid_brothers) == 0:
        for b in candidates:
            if len(fct(p_graph,node)) != 0:
                set_of_valid_brothers |= find_supinf(t_graph,p_graph,b,fct)
    return set_of_valid_brothers


def find_equals(t_graph,p_graph,node):
    """
    Search for brother nodes of the node parameter, where kept brother nodes correspond to already
    segmented regions whose topological predecessors are also already segmented. This is required
    to compute a valid photometric statistics within the a priori defined residue.

    :param t_graph: topological graph
    :param p_graph: photometric graph
    :param node: node
    :return: set of nodes satisfying constraints
    """
    #Finding first possible already segmented brothers without unsegmented subnodes
    candidates=recursive_segmented_brothers(p_graph,node)
    valid_candidates=set()
    for b in candidates: #Search for direct topological children, and check if they are segmented
        if is_residue_valid(t_graph,b) : valid_candidates |= set([b])
    return valid_candidates

def regions_from_residues(t_graph,residues):
    regions=[]
    for i in range(0,len(residues)):
        region=np.copy(residues[i])
        #predecessors=t_graph.predecessors(i) #direct predecessors
        predecessors=recursive_predecessors(t_graph,i) #direct predecessors
        for p in predecessors: region=np.logical_or(region,residues[p])
        regions+=[region]
    return regions