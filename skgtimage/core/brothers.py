import networkx as nx
import itertools
from skgtimage.core.search_base import recursive_brothers,find_head

def increasing_ordered_list(p_graph):
    currents = find_head(p_graph)
    my_list = []
    # a_head=list(heads)[0]
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
    #print("predecessors:", pred)
    # my_list+=[pred]
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

def predecessors_of_each_groups_of_brothers(g,groups_of_brothers):
    return __find_neighbor_of_each_groups_of_brothers__(g,groups_of_brothers,nx.DiGraph.predecessors)
def successors_of_each_groups_of_brothers(g,groups_of_brothers):
    return __find_neighbor_of_each_groups_of_brothers__(g,groups_of_brothers,nx.DiGraph.successors)

def orderings_of_groups_of_brothers(groups):
    all_permutations=[itertools.permutations(group) for group in groups]
    all_orderings=[list(i) for i in itertools.product(*all_permutations)]
    return all_orderings

def disconnect_brothers(g,groups_of_brothers):
    for group in groups_of_brothers:
        for n in group:
            for s in g.successors(n): g.remove_edge(n,s)
            for p in g.predecessors(n): g.remove_edge(p,n)

def generate_connection(current_graph,current_ordering,groups_of_brothers,predecessors,successors):
    edges=[]
    for g_index in range(0,len(current_ordering)):
        current_group=current_ordering[g_index]
        #Connecting predecessor
        pred=predecessors[g_index]
        pred_node=None
        if len(pred) == 1:
            pred_node=list(pred)[0]
        if len(pred) > 1:
            pred_index=groups_of_brothers.index(pred)
            pred_node=current_ordering[pred_index][-1]
        if pred_node is not None:        edges+=[(pred_node,current_group[0])]
        #Connecting internal node (i.e. within group)
        for node_index in range(0,len(current_group)-1):
            edges+=[(current_group[node_index],current_group[node_index+1])]
        #Connecting successor
        succ=successors[g_index]
        succ_node=None
        if len(succ) == 1:
            succ_node=list(succ)[0]
        if len(succ) > 1:
            succ_index=groups_of_brothers.index(succ)
            succ_node=current_ordering[succ_index][0]
        if succ_node is not None: edges+=[(current_group[-1],succ_node)]

    for e in edges:
        current_graph.add_edge(e[0],e[1])


def compute_possible_graphs(g):
    #Extract preliminary informations: sets of brothers, predecessors and successors
    groups_of_brothers=find_groups_of_brothers(g)
    predecessors=predecessors_of_each_groups_of_brothers(g,groups_of_brothers)
    successors=successors_of_each_groups_of_brothers(g,groups_of_brothers)
    #FINDING ALL POSSIBLE ORDERINGS OF BROTHERS
    all_orderings=orderings_of_groups_of_brothers(groups_of_brothers)

    #GENERATING GRAPHS RELATED TO ALL SPECIFIC ORDERING
    clean_graph=g.copy()
    disconnect_brothers(clean_graph,groups_of_brothers)


    all_graphs=[]
    for i in range(0,len(all_orderings)):
        print("In compute_possible_graphs",i, " / ",len(all_orderings))
        reordered_graph=clean_graph.copy()
        current_ordering=all_orderings[i]
        generate_connection(reordered_graph,current_ordering,groups_of_brothers,predecessors,successors)
        all_graphs+=[reordered_graph]
    return all_graphs


