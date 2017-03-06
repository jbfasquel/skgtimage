import networkx as nx
from skgtimage.core.graph import transitive_closure
from skgtimage.core.isomorphism import find_subgraph_isomorphims,__find_common_isomorphims__
from skgtimage.core.brothers import find_groups_of_brothers,__find_neighbor_of_each_groups_of_brothers__,orderings_of_groups_of_brothers


def predecessors_of_each_groups_of_brothers(g,groups_of_brothers):
    return __find_neighbor_of_each_groups_of_brothers__(g,groups_of_brothers,nx.DiGraph.predecessors)
def successors_of_each_groups_of_brothers(g,groups_of_brothers):
    return __find_neighbor_of_each_groups_of_brothers__(g,groups_of_brothers,nx.DiGraph.successors)


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
        reordered_graph=clean_graph.copy()
        current_ordering=all_orderings[i]
        generate_connection(reordered_graph,current_ordering,groups_of_brothers,predecessors,successors)
        all_graphs+=[reordered_graph]
    return all_graphs

def common_subgraphisomorphisms_bf(query_graphs, ref_graphs):
    """
    Computation of common (subgraph) isomorphisms using the orginal brute force implementation of the method
    (highly combinatory): one uses unwrapped photometric graphs (in case of similarities/cycles).

    :param query_graphs:
    :param ref_graphs:
    :return:
    """
    ###########################################################################################
    #Preparing possible "permutations" within ref_graphs: managing 'brother' nodes
    ###########################################################################################
    all_ref_graphs=[]
    for rg in ref_graphs:
        nb_brothers=find_groups_of_brothers(rg)
        if len(nb_brothers) > 0:
            all_ref_graphs+=[compute_possible_graphs(rg)] #we add a list of n elements (all possible graphs)
        else:
            all_ref_graphs+=[[rg]] #we add a list of one element
    ###########################################################################################
    #Loop over query (built) graphs to be matched with a priori knowledge (ref_graphs)
    ###########################################################################################
    isomorphisms_per_graph=[]
    nb_graphs=len(query_graphs)
    for i in range(0,nb_graphs):
        related_ref_graphs=all_ref_graphs[i]
        query=transitive_closure(query_graphs[i])
        #Loop over the possible "permutations" (i.e. versus brothers/incertain relationships) of reference graphs: union of matchings
        related_isomorphisms=[]
        counter=1
        for ref in related_ref_graphs:
            #print(counter,"/",len(related_ref_graphs));counter+=1
            ref=transitive_closure(ref)
            isomorphisms=find_subgraph_isomorphims(query,ref)
            related_isomorphisms+=isomorphisms
        isomorphisms_per_graph+=[related_isomorphisms]
    ###########################################################################################
    #Common isomorphisms: intersection of matchings
    ###########################################################################################
    common_isomorphisms=__find_common_isomorphims__(isomorphisms_per_graph[0],isomorphisms_per_graph[1])
    for i in range(2,len(isomorphisms_per_graph)):
        common_isomorphisms=__find_common_isomorphims__(common_isomorphisms,isomorphisms_per_graph[i])

    return common_isomorphisms #,isomorphisms_per_graph
