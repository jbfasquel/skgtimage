#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
import numpy as np
import networkx as nx
from skgtimage.core.photometry import region_stat
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

    if len(common_isomorphisms) == 0: #If no common matching, one returns
        return None,None,isomorphisms_per_graph[0],isomorphisms_per_graph[1],None
    ###########################################################################################
    #2) Compute the matching maximizing a criteria
    ###########################################################################################
    #Computing energies

    eie_per_iso=[]
    for c_iso in common_isomorphisms:
        eie_per_iso+=[matching_criterion_value(query_p_graph,ref_p_graph,c_iso)]

    #Taking the best common isomorphisms as result
    matching=None
    max_eie=max(eie_per_iso)
    index_of_max=eie_per_iso.index(max_eie)
    matching=common_isomorphisms[index_of_max]

    return matching,common_isomorphisms,isomorphisms_per_graph[0],isomorphisms_per_graph[1],eie_per_iso

def matching_criterion_value(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    mean_intensities=[]
    brothers_std_dev=[]
    #negative_intensities=[]
    for i in range(0,len(list_of_nodes)):
        current_element=list_of_nodes[i]
        #cost_brothers+=[0.0] #by default null penality
        ################################################
        #When brothers: one computes the mean intensity
        ################################################
        if len(current_element) > 1:
            #Taking region size proportions into account
            local_intensities=[]
            brother_nodes=[list(oi[b])[0] for b in current_element]
            #region=query_graph.get_region(brother_nodes[0]).astype(np.uint8)
            #local_intensities+=[region_stat(query_graph.get_image(),region,fct=np.mean)]
            for j in range(0,len(brother_nodes)):
                local_intensities+=[query_graph.get_mean_residue_intensity(brother_nodes[j])]
                #region+=query_graph.get_region(brother_nodes[j]).astype(np.uint8)

            mean_intensity=np.mean(np.asarray(local_intensities))
            stddev_intensity=np.std(np.asarray(local_intensities))
            brothers_std_dev+=[stddev_intensity]
            #local_intensities=sorted(local_intensities)
            #local_intensities_diffs=[ np.abs(local_intensities[i]-local_intensities[i-1]) for i in range(0,len(local_intensities)-1)]
            #negative_intensities+=local_intensities_diffs
            #cost_brothers[i]=-(region_stat(query_graph.get_image(),region,fct=np.std))
            #cost_brothers+=[(region_stat(query_graph.get_image(),region,fct=np.std))]
        ################################################
        #When simple node: one only retrieves the mean intensity
        ################################################
        else:
            tmp=list(current_element)[0]
            target=oi[tmp]
            corresponding_node=list(target)[0]
            mean_intensity=query_graph.get_mean_residue_intensity(corresponding_node)
        ################################################
        #Keep intensity
        ################################################
        mean_intensities+=[mean_intensity]
        #intensities+=[mean_intensity]

    intensity_diffs=[ np.abs(mean_intensities[i]-mean_intensities[i-1]) for i in range(1,len(mean_intensities)-1)]
    intensity_diffs_squared=np.asarray(intensity_diffs)**2
    mean_intensity_diff=np.mean(intensity_diffs_squared)

    eie=mean_intensity_diff
    if len(brothers_std_dev) > 0:
        brothers_std_dev_squared=np.asarray(brothers_std_dev)**2
        eie=eie/np.mean(brothers_std_dev_squared)
    '''
    for n in negative_intensities:
        eie-=n
    '''
    return eie


'''
def matching_criterion_value(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    intensities=[]
    #cost_brothers=[]
    negative_intensities=[]
    for i in range(0,len(list_of_nodes)):
        current_element=list_of_nodes[i]
        #cost_brothers+=[0.0] #by default null penality
        ################################################
        #When brothers: one computes the mean intensity
        ################################################
        if len(current_element) > 1:
            #Taking region size proportions into account
            local_intensities=[]
            brother_nodes=[list(oi[b])[0] for b in current_element]
            region=query_graph.get_region(brother_nodes[0]).astype(np.uint8)
            local_intensities+=[region_stat(query_graph.get_image(),region,fct=np.mean)]
            for j in range(1,len(brother_nodes)):
                local_intensities+=[query_graph.get_mean_residue_intensity(brother_nodes[j])]
                region+=query_graph.get_region(brother_nodes[j]).astype(np.uint8)

            mean_intensity=region_stat(query_graph.get_image(),region,fct=np.mean)
            local_intensities=sorted(local_intensities)
            local_intensities_diffs=[ np.abs(local_intensities[i]-local_intensities[i-1]) for i in range(0,len(local_intensities)-1)]
            negative_intensities+=local_intensities_diffs
            #cost_brothers[i]=-(region_stat(query_graph.get_image(),region,fct=np.std))
            #cost_brothers+=[(region_stat(query_graph.get_image(),region,fct=np.std))]
        ################################################
        #When simple node: one only retrieves the mean intensity
        ################################################
        else:
            tmp=list(current_element)[0]
            target=oi[tmp]
            corresponding_node=list(target)[0]
            mean_intensity=query_graph.get_mean_residue_intensity(corresponding_node)
        ################################################
        #Keep intensity
        ################################################
        intensities+=[mean_intensity]
    intensity_diffs=[ np.abs(intensities[i]-intensities[i-1]) for i in range(0,len(intensities)-1)]
    mean_intensity_diff=np.mean(np.asarray(intensity_diffs))

    eie=mean_intensity_diff
    for n in negative_intensities:
        eie-=n
    return eie
'''
'''
def matching_criterion_value(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    intensities=[]
    cost_brothers=[]
    for i in range(0,len(list_of_nodes)):
        current_element=list_of_nodes[i]
        #cost_brothers+=[0.0] #by default null penality
        ################################################
        #When brothers: one computes the mean intensity
        ################################################
        if len(current_element) > 1:
            #Taking region size proportions into account
            #local_intensities=[]
            brother_nodes=[list(oi[b])[0] for b in current_element]
            region=query_graph.get_region(brother_nodes[0]).astype(np.uint8)
            for j in range(1,len(brother_nodes)):
                region+=query_graph.get_region(brother_nodes[j]).astype(np.uint8)
            mean_intensity=region_stat(query_graph.get_image(),region,fct=np.mean)
            #cost_brothers[i]=-(region_stat(query_graph.get_image(),region,fct=np.std))
            cost_brothers+=[(region_stat(query_graph.get_image(),region,fct=np.std))]
        ################################################
        #When simple node: one only retrieves the mean intensity
        ################################################
        else:
            tmp=list(current_element)[0]
            target=oi[tmp]
            corresponding_node=list(target)[0]
            mean_intensity=query_graph.get_mean_residue_intensity(corresponding_node)
        ################################################
        #Keep intensity
        ################################################
        intensities+=[mean_intensity]
    intensity_diffs=[ np.abs(intensities[i]-intensities[i-1]) for i in range(0,len(intensities)-1)]
    mean_intensity_diff=np.mean(np.asarray(intensity_diffs))

    eie=mean_intensity_diff
    if len(cost_brothers) != 0:
        mean_cost_brothers=np.mean(np.asarray(cost_brothers))
        eie=float(mean_intensity_diff)/float(mean_cost_brothers)
    return eie
'''

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
