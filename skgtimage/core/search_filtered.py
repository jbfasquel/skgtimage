"""
Some recursive search algorithms within the directed graph, taking into account that some some nodes correspond to already segmented
regions.
"""
#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
import networkx as nx
from skgtimage.core.search_base import recursive_predecessors, recursive_successors,recursive_brothers

###########################
# Basic intersection between segmented nodes and recursive predecessors/successors/brothers
###########################
def recursive_segmented_successors(g,n):
    """
    Same as recursive_successors, except that only nodes corresponding to already segmented regions are considered
    :param g: graph
    :param n: node from which one searches successors (does not necessarily correspond to a segmented region)
    :return: set of node ids
    """
    """
    All encountered successors (recursive search avoiding cycles) whose nodes have a non None value for 'image' attribute.
    Return the list of nodes (empty if no matching successors).
    """
    segmented_nodes=g.segmented_nodes()
    segmented_successors = set(segmented_nodes) & recursive_successors(g,n)
    return segmented_successors
    #return __depth_segmented_search__(g,n,nx.DiGraph.successors)

def recursive_segmented_predecessors(g,n):
    """
    Same as recursive_segmented_predecessors, except that only nodes corresponding to already segmented regions are considered
    :param g: graph
    :param n: node from which one searches predecessors (does not necessarily correspond to a segmented region)
    :return: set of node ids
    """

    """
    All encountered predecessors (recursive search avoiding cycles) whose nodes have a non None value for 'image' attribute.
    Return the list of nodes (empty if no matching predecessors).
    """
    segmented_nodes=g.segmented_nodes()
    segmented_predecessors = set(segmented_nodes) & recursive_predecessors(g,n)
    return segmented_predecessors

def recursive_segmented_brothers(g,n):
    """
    Same as recursive_brothers, except that only nodes corresponding to already segmented regions are considered
    :param g: graph
    :param n: node from which one searches brothers (does not necessarily correspond to a segmented region)
    :return: set of node ids
    """
    p=set(recursive_segmented_predecessors(g,n))
    s=set(recursive_segmented_successors(g,n))
    brothers= p & s
    return list(brothers)

###########################
# First encountered successors/predecessors
###########################
def first_segmented_successors(g,n):
    """
    Recursively search, from a given node n, the first encountered successors corresponding to an already segmented region.
    One ignores already segmented brothers of n.

    :param g: graph
    :param n: node from which one searches
    :return: set of node ids
    """
    #If the
    to_exclude=recursive_brothers(g,n)

    return __first_segmented_search__(g,n,nx.DiGraph.successors,set(),to_exclude)

def first_segmented_predecessors(g,n):
    """
    Recursively search, from a given node n, the first encountered predecessors corresponding to an already segmented region
    One ignores already segmented brothers of n.s

    :param g: graph
    :param n: node from which one searches
    :return: set of node ids
    """
    to_exclude=recursive_brothers(g,n)

    return __first_segmented_search__(g,n,nx.DiGraph.predecessors,set(),to_exclude)

def __first_segmented_search__(g,n,functor=nx.DiGraph.predecessors,visited=set(),excluded=set()):
    """ Internal search function

    :param g: graph
    :param n: node from which one searches
    :param functor: direct neighbor search (either predecessors or successors)
    :param visited: internally used to avoid cycles
    :param excluded: nodes to be ignored during the search (e.g. brothers)
    :return: set of node ids
    """
    to_visit=set(functor(g,n)) - visited
    visited = visited | to_visit
    #Result initialization
    result=set()
    for e in to_visit:
        if (g.get_region(e) is not None) and (e not in excluded): result = result | set([e])
        else: result = result | set(__first_segmented_search__(g,e,functor,visited,excluded))
    #New to avoid "n" to be segmented
    if n in result: result -= set([n])
    return result

###########################
# Successors/Predecessors (recursive search) until the first one (excluded) corresponding to an already segmented region
###########################

def recursive_successors_until_first_segmented(g,n):
    """
    Recursively search for unsegmented successors until the first segmented one is encountered (excluded)

    :param g: graph
    :param n: node from which one searches
    :return: set of node ids
    """
    return __until_first_segmented_search__(g,n,nx.DiGraph.successors)

def recursive_predecessors_until_first_segmented(g,n):
    """
    Recursively search for unsegmented predecessors until the first segmented one is encountered (excluded)

    :param g: graph
    :param n: node from which one searches
    :return: set of node ids
    """
    return __until_first_segmented_search__(g,n,nx.DiGraph.predecessors)

def __until_first_segmented_search__(g,n,functor=nx.DiGraph.predecessors,visited=set()):
    to_visit=set(functor(g,n)) - visited
    visited = visited | to_visit
    #Result initialization
    result=set()
    for e in to_visit:
        if (g.get_region(e) is None) :
            result = result | set([e])
            result = result | set(__until_first_segmented_search__(g,e,functor,visited))

    return result
