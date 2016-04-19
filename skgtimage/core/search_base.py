"""
Some recursive search algorithms within the directed graph, assuming no segmentations.
Search algorithms concern successors, predecessors and brothers of a given node.
Search algorithms manage cycles.
Note that to kind of cycles may be reported:
 * If the cycle involves two nodes: these two nodes are seen as brothers.
 * For larger cycles (i.e. more than two nodes): involved nodes are not brothers.
"""
#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
import networkx as nx

def recursive_successors(g,n):
    """
    Recursively search for successors of a given node, avoiding cycles

    :param g: graph
    :param n: node from which one searches successors
    :return: set of node ids
    """
    return __depth_search__(g,n,nx.DiGraph.successors)

def recursive_predecessors(g,n):
    """
    Recursively search for predecessors of a given node, avoiding cycles

    :param g: graph
    :param n: node from which one searches predecessors
    :return: set of node ids
    """
    return __depth_search__(g,n,nx.DiGraph.predecessors)

def recursive_brothers(g,n):
    """
    Recursively search for brothers of a given node, e.g. depicting bidirectional edges with the given node n

    :param g: graph
    :param n: node from which one searches brothers
    :return: set of node ids
    """
    brothers=__depth_brother_search__(g,n) - set([n])
    return brothers

def recursive_nonbrother_successors(g,n):
    """
    Recursively search for predecessors of a given node, avoiding cycles and ignoring brothers

    :param g: graph
    :param n: node from which one searches predecessors
    :return: set of node ids
    """
    brothers=set(recursive_brothers(g,n))
    successors=__depth_search__(g,n,nx.DiGraph.successors)
    non_brothers_successors = successors - (successors & brothers)
    return non_brothers_successors

def recursive_nonbrother_predecessors(g,n):
    """
    Recursively search for predecessors of a given node, avoiding cycles and ignoring brothers

    :param g: graph
    :param n: node from which one searches predecessors
    :return: set of node ids
    """
    brothers=set(recursive_brothers(g,n))
    predecessors=__depth_search__(g,n,nx.DiGraph.predecessors)
    non_brothers_predecessors = predecessors - (predecessors & brothers)
    return non_brothers_predecessors


def __depth_brother_search__(g,n,visited=set()):
    """ Internal search function

    :param g: graph
    :param n: node
    :param visited: internally used to avoid cycles
    :return:
    """
    to_visit=set(g.successors(n)) - visited #"s" minus already encountered
    visited = visited | to_visit #union
    brothers=set()
    for i in to_visit:
        if n in set(g.successors(i)):
            brothers = brothers | set([i])
            brothers = brothers | __depth_brother_search__(g,i,visited)
    return brothers


def __depth_search__(g,n,functor=nx.DiGraph.predecessors,visited=set()):
    """ Internal search function

    :param g: graph
    :param n: node
    :param functor: direct neighbor search (either predecessors or successors)
    :param visited: internally used to avoid cycles
    :return:
    """
    to_visit=set(functor(g,n)) - visited #"s" minus already encountered
    visited = visited | to_visit #union
    #Result initialization
    result = set(to_visit)
    #Next visits
    for i in to_visit:
        result = result | set(__depth_search__(g,i,functor,visited)) #set union
    to_return=result - set([n])
    return to_return #to avoid requested node

def number_of_brother_links(g):
    """
    Determine the number of similarity links within g
    For example, A<B<C<D -> 0 ; A=B<C<D -> 1; A=B<C=D -> 2; A=B=C<D -> 2

    :param g: graph
    :return: number of similarity links (int)
    """
    number=0
    already_considered=set()
    for n in g.nodes():
        already_considered |= set([n])
        bs=recursive_brothers(g,n)
        for b in bs:
            if not(b in already_considered): number+=1
            already_considered |= set([b])

    return int(number)

def find_head(g):
    head=None
    #Search for the node (or brother nodes) without successors: the head of the graph
    for n in g.nodes():
        successors=set()
        brothers=recursive_brothers(g,n)
        #When n has no brothers
        if len(brothers)==0:
            successors=set(g.successors(n))
            if len(successors)==0:
                head=set([n])
        #When n has brothers
        else:
            group=brothers | set(n)
            for e in group:
                successors|=set(g.successors(e))
            effective_successors=successors-group
            if len(effective_successors)==0:
                head=group
    return head


def search_leaf_nodes(graph):
    result=set()
    for n in graph.nodes():
        if len(graph.predecessors(n)) == 0: result|=set([n])
    return result


def __recursive_ordering_search__(g,n,result):
    pred=set()
    for e in n:
        pred|=set(g.predecessors(e))
    pred-=n
    for r in result:
        pred-=r
    if len(pred) !=0:
        pred|=recursive_brothers(g,list(pred)[0])
        result+=[pred]
        __recursive_ordering_search__(g,pred,result)

def decreasing_ordered_nodes(g):
    """
    Only meaning full for photometric graphs
    :param g:
    :return:
    """
    head=find_head(g)
    result=[head]
    __recursive_ordering_search__(g,head,result)
    return result
