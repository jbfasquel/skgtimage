#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause


import numpy as np
import networkx as nx
from skgtimage.core.parameters import regions_from_residues
from skgtimage.core.graph import transitive_closure,transitive_reduction
from skgtimage.core.photometry import region_stat
from skgtimage.core.topology import fill_region
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

def find_common_isomorphisms(isomorphisms):
    matchings=__find_common_isomorphims__(isomorphisms[0],isomorphisms[1])
    for i in range(2,len(isomorphisms)):
        matchings=__find_common_isomorphims__(matchings,isomorphisms[i])
    return matchings

def generate_common_subgraphisomorphisms(query_graphs,ref_graphs):
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
    common_isomorphisms=find_common_isomorphisms(isomorphisms_per_graph)

    return common_isomorphisms,isomorphisms_per_graph

def nb_automorphisms(graphs):
    auto=[]
    for g in graphs:
        closed_g=transitive_closure(g)
        automorphisms=find_subgraph_isomorphims(closed_g,closed_g)
        auto+=[len(automorphisms)]
    return auto



def unrelevant_inputs(io,nodes):
    result=set(nodes)-set(io.keys())
    return result


def ambiguous_inputs(io):
    inputs_leading_to_multiple_outputs=set()
    for i in io:
        if len(io[i]) > 1: inputs_leading_to_multiple_outputs|=set([i])
    return inputs_leading_to_multiple_outputs

def remove_node(g,n):
    edges_to_add=[]
    succ=g.successors(n)
    prec=g.predecessors(n)
    for p in prec:
        for s in succ:
            edges_to_add+=[(p,s)]
    g.remove_node(n)
    for e in edges_to_add:
        g.add_edge(e[0],e[1])




def unmatched_nodes(isomorphisms,nodes):
    unmatchings=set()
    for n in nodes:
        is_involved=False
        for iso in isomorphisms:
            if n in iso.keys(): is_involved=True
        if is_involved is False: unmatchings |= set([n])
    return unmatchings
'''
def __is_surjective__(outputs2inputs):
    input_points=list(outputs2inputs.values())
    is_surjection=True
    for i in range(0,len(input_points)):
        ref_set=input_points[i]
        for j in range(0,len(input_points)):
            if i !=j:
                intersection=ref_set & input_points[j]
                if len(intersection) != 0: is_surjection=False
    return is_surjection
'''
def iorelationships(isomorphisms):
    """

    :param isomorphisms
    :return: dictionnaries where keys are outputs of the surjection and values are related inputs (possibly severals)
    """
    inputs=set()
    for i in isomorphisms:
        inputs|=set(i.keys())

    io={}
    for i in inputs:
        io[i]=set()
    for i in isomorphisms:
        for k in i:
            io[k] |= set([i[k]])
    return io
'''
def find_sub_surjection(isomorphisms):
    """

    :param isomorphisms
    :return: dictionnaries where keys are outputs of the surjection and values are related inputs (possibly severals)
    """
    inputs=set()
    for i in isomorphisms:
        inputs|=set(i.keys())

    outputs2inputs={}
    for iso in isomorphisms:
        for k in iso.keys():
            target=iso[k]
            if target not in outputs2inputs: outputs2inputs[target]=set()
            outputs2inputs[target] |= set([k])
    if __is_surjective__(outputs2inputs) is False:
        return None

    return outputs2inputs
'''
def oirelationships(io):
    oi={}
    for i in io:
        my_val=io[i]
        if type(my_val)==str:
            if my_val not in oi:
                #my_key=set(my_val])
                oi[my_val]=set([i])
        else:
            for o in my_val:
                if o not in oi: oi[o]=set([i])
                else: oi[o] |= set([i])
    return oi


def id2residues(residues,io):
    oi={}
    for i in io:
        for o in io[i]:
            if o not in oi: oi[o]=set([i])
            else: oi[o] |= set([i])

    id2r={}
    for o in oi.keys():
        targets=list(oi[o])
        current_residue=residues[targets[0]]
        for j in range(1,len(targets)):
            current_residue=np.logical_or(current_residue,residues[targets[j]])
        id2r[o]=current_residue
    return id2r

def update_graphs(graphs,residues,matching):
    id2res=id2residues(residues,matching)
    for k in id2res:
        for g in graphs:
            g.set_region(k,fill_region(id2res[k]))


'''
def update_residues(residues,t_graph,surjection,unrelated_nodes):
    """
    :param residues:
    :param t_graph:
    :param surjection:
    :return:
    """
    #MERGE UNRELATED NODES (REGION RESIDUES FIRST)
    final_residues=np.copy(residues)
    for n in unrelated_nodes:
        fathers=list(t_graph.successors(n))
        if len(fathers) > 1: raise Exception("Error")
        f=fathers[0]
        #print(np.max(final_residues[f]),np.max(final_residues[n]))
        #print(np.min(final_residues[f]),np.min(final_residues[n]))
        final_residues[f]=np.logical_or(final_residues[f],final_residues[n])
    #ASSIGN
    id2residues={}
    for i in surjection.keys():
        targets=list(surjection[i])
        current_residue=final_residues[targets[0]]
        for j in range(1,len(targets)):
            current_residue=np.logical_or(current_residue,final_residues[targets[j]])
        id2residues[i]=current_residue

    return id2residues
    #all_nodes=
    #unmatched_nodes=
'''



'''
def identify_from_labels(image,labelled_image,t_graph,p_graph,nodes,return_detailed=False):
    residues=residues_from_labels(labelled_image)
    sub_p_graph=transitive_reduction(transitive_closure(p_graph).subgraph(nodes))
    #sub_t_graph=skgti.core.transitive_reduction(skgti.core.transitive_closure(tp_model.t_graph.subgraph(l)))
    sub_t_graph=transitive_reduction(transitive_closure(t_graph).subgraph(nodes))
    built_t_graph,new_residues=topological_graph_from_residues(residues)
    n=number_of_brother_links(sub_p_graph)
    built_p_graph=photometric_graph_from_residues(image,new_residues,n)
    #Matching
    t_isomorphisms=find_subgraph_isomorphims(transitive_closure(built_t_graph),transitive_closure(sub_t_graph))
    p_isomorphisms=find_subgraph_isomorphims(transitive_closure(built_p_graph),transitive_closure(sub_p_graph))
    matchings=find_common_isomorphisms([p_isomorphisms,t_isomorphisms])
    # SURJECTION
    surj=find_sub_surjection(matchings)
    # FINAL REGIONS
    unrelated_nodes=unmatched_nodes(matchings,built_t_graph.nodes())
    target2residues=update_residues(new_residues,built_t_graph,surj,unrelated_nodes)
    if return_detailed:
        return target2residues,built_t_graph,built_p_graph,surj,matchings,t_isomorphisms,p_isomorphisms,new_residues
    else:
        return target2residues
'''


def t_filtering_v1(matching,residues,t_graph,p_graph):
    nodes=unrelevant_inputs(matching,t_graph.nodes())
    #final_residues=np.copy(residues)
    for n in nodes:
        succ=t_graph.successors(n)
        if len(succ) > 1: raise("Error")
        print(n,"->",succ)
        if len(succ) == 1 :
            father=succ[0]
            #Residue
            residues[father]=np.logical_or(residues[father],residues[n])
            #Adapt graphs
            remove_node(t_graph,n)
            remove_node(p_graph,n)

    #return final_residues,t_graph,p_graph

def p_filtering(matching,image,residues,t_graph,p_graph):
    nodes=ambiguous_inputs(matching)
    #final_residues=np.copy(residues)
    ##################
    # Photometric distances BUT between TOPOLOGICAL neighbors !!!!
    stats=[]
    for r in residues:
        stats+=[region_stat(image,r,fct=np.mean,mc=False)]
    #print(stats)

    for n in nodes:
        #list_of_adj=p_graph.successors(n)+p_graph.predecessors(n)
        list_of_adj=t_graph.successors(n)+t_graph.predecessors(n)
        list_of_dist=[]
        for a in list_of_adj:
            list_of_dist+=[abs(stats[a]-stats[n])]

        indice_of_min_dist=np.argmin(list_of_dist)
        closest_node=list_of_adj[indice_of_min_dist]
        print(list_of_adj," -> ",list_of_dist)
        print("closest:",closest_node)
        #Residue
        residues[closest_node]=np.logical_or(residues[closest_node],residues[n])
        #Adapt graphs
        remove_node(t_graph,n)
        remove_node(p_graph,n)
        #Update matching
        del matching[n]


    '''
    for n in nodes:
        #
        succ=t_graph.successors(n)
        if len(succ) > 1: raise("Error")
        father=succ[0]
        #Residue
        final_residues[father]=np.logical_or(final_residues[father],final_residues[n])
        #Adapt graphs
        remove_node(t_graph,n)
        remove_node(p_graph,n)
    '''
    #return final_residues,t_graph,p_graph

#############################################################################################
#############################################################################################
#####################      NEW                              #################################
#############################################################################################
#############################################################################################
def unambiguous_matchings(matching):
    result={}
    reverse_matching=oirelationships(matching)
    for i in matching:
        input=i
        output=matching[i]
        if (len(output)==1):
            target=list(output)[0]
            target_inputs=reverse_matching[target]
            if (len(target_inputs)==1):
                result[i]=matching[i]
    return result

def ambiguous_matchings(matching):
    result={}
    for i in matching:
        if len(matching[i])>1: result[i]=matching[i]
    return result

def sub_common_isomorphisms(common_isomorphisms,io):
    result=[]
    for iso in common_isomorphisms:
        valid=True
        for e in io:
            if e not in iso: valid=False
        if valid: result+=[iso]
    return result

def filtered_common_subgraph_isomorphisms_v1(matching,common_isomorphisms):
    okmat=unambiguous_matchings(matching)
    nokmat=ambiguous_matchings(matching)
    new_common_isomorphisms=sub_common_isomorphisms(common_isomorphisms,okmat)
    io=iorelationships(new_common_isomorphisms)
    return io,new_common_isomorphisms


def filtered_common_subgraph_isomorphisms_v2(matching,common_isomorphisms):
    okmat=unambiguous_matchings(matching)
    stats=[]
    for iso in common_isomorphisms:
        measure=0
        for e in okmat:
            if list(okmat[e])[0] == iso[e]:
                measure+=1
        stats+=[measure]
    print(stats)
    indice=stats.index(max(stats))
    new_matching=common_isomorphisms[indice]
    return new_matching



def t_filtering_v2(image,residues,matching,t_graph,p_graph):
    nodes=unrelevant_inputs(matching,t_graph.nodes())

    ##################
    # Photometric distances BUT between TOPOLOGICAL neighbors !!!!
    stats=[]
    for r in residues:
        stats+=[region_stat(image,r,fct=np.mean,mc=False)]


    #final_residues=np.copy(residues)
    for n in nodes:
        #intensity distance versus successor
        succ=t_graph.successors(n)
        if len(succ) > 1: raise("Error")
        father=succ[0]
        d_succ=abs(stats[father]-stats[n])
        #intensity distance versus successor
        pred=t_graph.predecessors(n)
        if len(pred) > 1: raise("Error")
        child=pred[0]
        d_pred=abs(stats[child]-stats[n])

        #################
        #merge with closest
        if d_succ<d_pred:
            #Residue
            residues[father]=np.logical_or(residues[father],residues[n])
            #Adapt graphs
            remove_node(t_graph,n)
            remove_node(p_graph,n)
        else:
            #Residue
            residues[child]=np.logical_or(residues[child],residues[n])
            #Adapt graphs
            remove_node(t_graph,n)
            remove_node(p_graph,n)


#############################################################################################
#############################################################################################
#####################      NEW    ENERGY                    #################################
#############################################################################################
#############################################################################################




def energie_dist(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    print(list_of_nodes)
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
    print(intensities)
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

