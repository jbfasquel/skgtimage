#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
import networkx as nx
from skgtimage.core.graph import transitive_closure,get_ordered_nodes
from skgtimage.core.brothers import increasing_ordered_list


def nb_automorphisms(graphs):
    auto=[]
    for g in graphs:
        closed_g=transitive_closure(g)
        automorphisms=find_subgraph_isomorphims(closed_g,closed_g)
        auto+=[len(automorphisms)]
    return auto

def photometric_iso_validity(iso,image_ordering,model_ordering):
    model_ordering_from_iso = []
    for i in image_ordering:
        if i in iso:
            model_ordering_from_iso += [iso[i]]
    ###################
    model_copy=model_ordering.copy()
    image_copy=model_ordering_from_iso.copy()
    valid=True
    for m in model_copy:
        nb_similar_element=len(m)
        related_elements=set(image_copy[0:nb_similar_element])
        if m != related_elements: valid=False
        image_copy=image_copy[nb_similar_element:]

    return valid


def find_subgraph_isomorphims(query_graph,ref_graph):
    matcher=nx.isomorphism.DiGraphMatcher(query_graph,ref_graph)
    sub_isomorphisms=[i for i in matcher.subgraph_isomorphisms_iter()]
    return sub_isomorphisms

def __find_common_isomorphims__(isomorphisms_1,isomorphisms_2):
    matchings=[]
    for t in isomorphisms_1:
        for p_iso in isomorphisms_2:
            if (t == p_iso) and (t is not None) and (p_iso is not None): matchings+=[t]
    return matchings

def common_subgraphisomorphisms(query_graphs, ref_graphs, verbose=False):
    t_query,t_ref=query_graphs[0],ref_graphs[0]

    #############################
    # t_isomorphisms
    #############################
    t_isomorphisms_candidates=find_subgraph_isomorphims(transitive_closure(t_query), transitive_closure(t_ref))

    if verbose: print("Number of found isomorphisms (inclusion):",len(t_isomorphisms_candidates))
    #############################
    # p_isomorphisms
    #############################
    p_query,p_model=query_graphs[1],ref_graphs[1]
    #image_ordering=p_query.get_ordered_nodes()
    image_ordering = get_ordered_nodes(p_query)
    model_ordering=increasing_ordered_list(p_model)
    common_iso=[]
    for iso in t_isomorphisms_candidates:
        if photometric_iso_validity(iso,image_ordering,model_ordering): common_iso+=[iso]
    if verbose: print("Number of found common isomorphisms (inclusion+photometry):", len(common_iso))

    return common_iso




