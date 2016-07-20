"""
Specific graph search algorithms
"""

#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
import networkx as nx

#################################
# RECURSIVE SEARCH (SUCCESSORS AND PREDECESSORS, BROTHERS) OF NODES, AVOIDING CYCLES
#################################

def recursive_successors(g,n):
    return list(__depth_search__(g,n,nx.DiGraph.successors))

def recursive_predecessors(g,n):
    return list(__depth_search__(g,n,nx.DiGraph.predecessors))

def __depth_search__(g,n,functor=nx.DiGraph.predecessors,visited=set()):
    """
    Generic search 
    """
    to_visit=set(functor(g,n)) - visited #"s" minus already encountered
    visited = visited | to_visit #union
    #Result initialization
    result = set(to_visit)
    #Next visits
    for i in to_visit:
        result = result | set(__depth_search__(g,i,functor,visited)) #set union
    return result - set(n) #to avoid requested node

def recursive_brothers(g,n):
    p=set(recursive_predecessors(g,n))
    s=set(recursive_successors(g,n))
    brothers= p & s
    return list(brothers)

#################################
# RECURSIVE SEARCH (SUCCESSORS AND PREDECESSORS) OF NODES CORRESPONDING TO SEGMENTED REGIONS, AVOIDING CYCLES
#################################

def recursive_segmented_successors(g,n):
    """
    All encountered successors (recursive search avoiding cycles) whose nodes have a non None value for 'image' attribute.
    Return the list of nodes (empty if no matching successors).
    """
    return list(__depth_segmented_search__(g,n,nx.DiGraph.successors))

def recursive_segmented_predecessors(g,n):
    """
    All encountered predecessors (recursive search avoiding cycles) whose nodes have a non None value for 'image' attribute.
    Return the list of nodes (empty if no matching predecessors).
    """
    return list(__depth_segmented_search__(g,n,nx.DiGraph.predecessors))


def recursive_segmented_brothers(g,n):
    p=set(recursive_segmented_predecessors(g,n))
    s=set(recursive_segmented_successors(g,n))
    brothers= p & s
    return list(brothers)



def __depth_segmented_search__(g,n,functor=nx.DiGraph.predecessors,visited=set()):
    to_visit=set(functor(g,n)) - visited
    visited = visited | to_visit
    #Result initialization
    result=set()
    for e in to_visit:
        if (g.get_region(e) is not None) : result = result | set(e) #set union
    #Next visits
    for e in to_visit:
        result = result | set(__depth_segmented_search__(g,e,functor,visited)) #set union
    return result


#################################
# SEARCH (SUCCESSORS AND PREDECESSORS) OF THE FIRST NODES CORRESPONDING TO SEGMENTED REGIONS, AVOIDING CYCLES
#################################

def segmented_successors(g,n):
    """
    First encountered successors whose nodes has a non None value for 'image' attribute.
    Return a set of nodes (empty set if no matching successors).
    """
    #One excludes predecessors to avoid those being also a predecessor (in case of equivalence/cycle A <-> B)
    to_exclude=__depth_search__(g,n,functor=nx.DiGraph.predecessors)
    #Recursive search, avoiding cycles when running over successors (i.e. "visited")
    result=__first_segmented_search__(g,n,nx.DiGraph.successors,to_exclude)
    return list(result)

def segmented_predecessors(g,n):
    """
    First encountered predecessors whose nodes has a non None value for 'image' attribute.
    Return a set of nodes (empty set if no matching successors).
    """
    #As for segmented_successors
    to_exclude=__depth_search__(g,n,functor=nx.DiGraph.successors)
    #As for segmented_successors
    result=__first_segmented_search__(g,n,nx.DiGraph.predecessors,to_exclude)
    return list(result)

def __first_segmented_search__(g,n,functor=nx.DiGraph.predecessors,to_exclude=set(),visited=set()):
    to_visit=set(functor(g,n)) - visited
    visited = visited | to_visit
    #Result initialization
    result=set()
    for e in to_visit:
        if (g.get_region(e) is not None) & (e not in to_exclude): result = result | set(e) #set union
    #Next visits: nodes without "region"
    for e in (to_visit-result):
        result = result | set(__first_segmented_search__(g,e,functor,to_exclude,visited)) #set union
    return result


#################################
# RESIDUE (TOPOLOGICAL INFORMATION), REGION OF INTEREST (ROI) AND LIST OF REGIONS BELONGING TO THE ROI
#################################

def residue(g,n):
    """
    Region 'n' minus already segmented regions which are declared to be included
    """
    root_region=g.get_region(n)
    included_nodes=segmented_predecessors(g,n)
    to_remove=np.zeros(root_region.shape,np.bool)
    for i in included_nodes:
        to_remove=np.logical_or(to_remove,g.get_region(i))
    residue=np.logical_and(root_region,np.logical_not(to_remove))
    return residue

def roi(g,n):
    """
    Compute the region of interest, according to the target n
    Parameters:
    n is the target (node of the topological core g)
    """
    root=segmented_successors(g,n)
    roi=residue(g,root[0])
    return roi

def __non_segmented_predecessors__(g,n):
    result=set()
    for i in g.predecessors(n):
        if g.get_region(i) is None:
            result=result | set(i) #set union
            result=result | __non_segmented_predecessors__(g,i)
    return result


def regionlist_in_roi(g,n):
    """
    Computer the list of nodes belonging to the roi for a given target n (unsegmented regions)
    Corresponds to equation 13 of paper (Nr).
    Parameters:
    n is the target (node of the topological core g)
    """
    root=segmented_successors(g,n)[0]
    result=set(root)
    result=result | set(__non_segmented_predecessors__(g,root))
    return list(result)

