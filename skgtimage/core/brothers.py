import networkx as nx
import itertools
from skgtimage.core.search_base import recursive_brothers,find_head


def increasing_ordered_list(p_graph):
    currents = find_head(p_graph)
    my_list = []
    while len(currents) != 0:
        my_list += [currents]
        pred = set()
        for e in currents:
            pred |= (set(nx.DiGraph.predecessors(p_graph, e)) - set(recursive_brothers(p_graph, e)))
        brothers_to_add = set()
        for p in pred:
            brothers_to_add |= set(recursive_brothers(p_graph, p))
        pred |= brothers_to_add
        pred -= currents
        currents = pred

    my_list.reverse()
    return my_list

def find_groups_of_brothers(g):
    groups_of_brothers=[]
    for n in g.nodes():
        b=recursive_brothers(g,n)
        if len(b) != 0:
            b |=set([n])
            if b not in groups_of_brothers:
                groups_of_brothers+=[b]
    return groups_of_brothers

def __find_neighbor_of_each_groups_of_brothers__(g,groups_of_brothers,functor=nx.DiGraph.predecessors):
    predecessors=[]
    for group in groups_of_brothers:
        found_pred=set()
        for n in group:
            p=set(functor(g,n))
            for e in p:
                if e not in group:
                    found_pred|=set([e])
        #Include brothers of predecessor
        if len(found_pred) !=0:
            all_found_pred=recursive_brothers(g,list(found_pred)[0]) | set(found_pred)
        else:
            all_found_pred=found_pred
        predecessors+=[all_found_pred]
    return predecessors


def orderings_of_groups_of_brothers(groups):
    all_permutations=[itertools.permutations(group) for group in groups]
    all_orderings=[list(i) for i in itertools.product(*all_permutations)]
    return all_orderings






