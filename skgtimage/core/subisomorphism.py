#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
import numpy as np
import networkx as nx
from skgtimage.core.graph import transitive_closure
from skgtimage.core.brothers import find_groups_of_brothers,compute_possible_graphs
from skgtimage.core.search_base import decreasing_ordered_nodes

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

def common_subgraphisomorphisms(query_graphs,ref_graphs):
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
            print(counter,"/",len(related_ref_graphs));counter+=1
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

    return common_isomorphisms,isomorphisms_per_graph


def best_common_subgraphisomorphism(query_t_graph,ref_t_graph,query_p_graph,ref_p_graph,return_details=False):
    ###########################################################################################
    #1) Find common matchings
    ###########################################################################################
    common_isomorphisms,isomorphisms_per_graph=common_subgraphisomorphisms([query_t_graph,query_p_graph],[ref_t_graph,ref_p_graph])
    '''
    print("Nb t iso",len(isomorphisms_per_graph[0]))
    print("Nb p iso",len(isomorphisms_per_graph[1]))
    print("Nb common iso",len(common_isomorphisms))
    '''
    if len(common_isomorphisms) == 0: raise Exception("No common iso")
    ###########################################################################################
    #2) Compute energies regarding similarities
    ###########################################################################################
    brothers=find_groups_of_brothers(ref_p_graph)
    matching=None
    #Computing energies
    eie_sim=[]
    for c_iso in common_isomorphisms:
        eie_sim+=[energie_sim(query_p_graph,ref_p_graph,c_iso)]
    eie_dist=[]
    for c_iso in common_isomorphisms:
        eie_dist+=[energie_dist(query_p_graph,ref_p_graph,c_iso)]
    #Taking the best common isomorphisms as result
    if (len(brothers) != 0) and (len(eie_sim) !=0):
        min_eie=min(eie_sim)
        nb=eie_sim.count(min_eie)
        if nb != 1 : raise Exception("erreur")
        index_of_min=eie_sim.index(min_eie)
        matching=common_isomorphisms[index_of_min]
    else:
        max_eie=max(eie_dist)
        nb=eie_dist.count(max_eie)
        #if nb != 1 : raise Exception("erreur")
        index_of_max=eie_dist.index(max_eie)
        matching=common_isomorphisms[index_of_max]

    #print("eie_sim:", eie_sim)
    #print("eie_dist:", eie_dist)

    return matching,common_isomorphisms,isomorphisms_per_graph[0],isomorphisms_per_graph[1],eie_sim,eie_dist



def nb_automorphisms(graphs):
    auto=[]
    for g in graphs:
        closed_g=transitive_closure(g)
        automorphisms=find_subgraph_isomorphims(closed_g,closed_g)
        auto+=[len(automorphisms)]
    return auto

def oirelationships(io):
    oi={}
    for i in io:
        my_val=io[i]
        if type(my_val)==str:
            if my_val not in oi:
                oi[my_val]=set([i])
        else:
            for o in my_val:
                if o not in oi: oi[o]=set([i])
                else: oi[o] |= set([i])
    return oi

def energie_dist(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    #print(list_of_nodes)
    intensities=[]
    for i in range(0,len(list_of_nodes)):
        current_element=list_of_nodes[i]
        #When brothers: one computes the mean intensity
        if len(current_element) > 1:
            local_intensities=[]
            for b in current_element:
                corresponding_node=list(oi[b])[0]
                local_intensities+=[query_graph.get_mean_residue_intensity(corresponding_node)]
            mean_intensity=np.mean(np.asarray(local_intensities))
        #When simple node: one only retrieves the mean intensity
        else:
            tmp=list(current_element)[0]
            target=oi[tmp]
            corresponding_node=list(target)[0]
            mean_intensity=query_graph.get_mean_residue_intensity(corresponding_node)
        intensities+=[mean_intensity]
    #Compute distances:
    #print(intensities)
    eie=0
    for i in range(0,len(list_of_nodes)-1):
        eie+=np.abs(intensities[i]-intensities[i+1])
    return eie

def energie_sim(query_graph,ref_graph,iso):
    """
    Compute graph energy related to similar nodes
    :param query_graph:
    :param ref_graph:
    :param iso:
    :return:
    """
    oi=oirelationships(iso)
    grps=find_groups_of_brothers(ref_graph)
    energy_per_brother_groups=[]
    for g in grps:
        intensities=[]
        for b in g:
            corresponding_node=list(oi[b])[0]
            intensities+=[query_graph.get_mean_residue_intensity(corresponding_node)]
        eie=np.std(np.asarray(intensities))
        energy_per_brother_groups+=[eie]
    iso_energy=np.sum(np.asarray(energy_per_brother_groups))

    return iso_energy

