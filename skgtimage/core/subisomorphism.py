#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause
import numpy as np
import networkx as nx
from skgtimage.core.graph import transitive_closure
from skgtimage.core.brothers import find_groups_of_brothers,compute_possible_graphs,increasing_ordered_list
from skgtimage.core.search_base import decreasing_ordered_nodes


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

def is_isomorphism_valid(query_graph, ref_graph, isomorphism):
    matcher = nx.isomorphism.DiGraphMatcher(query_graph, ref_graph)
    # check if iso_candidate is an valid subgraph iso between query and ref
    validity=True
    for query_node in isomorphism:
        ref_node = isomorphism[query_node]
        is_ok = matcher.syntactic_feasibility(query_node, ref_node)
        if is_ok == False: validity=False

    return validity

def __find_common_isomorphims__(isomorphisms_1,isomorphisms_2):
    matchings=[]
    for t in isomorphisms_1:
        for p_iso in isomorphisms_2:
            if (t == p_iso) and (t is not None) and (p_iso is not None): matchings+=[t]
    return matchings

def check_iso_eligibility(iso,ordered_nodes,groups_of_brothers):
    """
    :param iso: input to output bijection
    :param ordered_nodes: from real image (photometric graph)
    :param groups_of_brothers: from model (photometric graph)
    :return:
    """
    validity=True
    oi=oirelationships_iso(iso)
    #print("reverse iso",oi)
    for group in groups_of_brothers:
        group_inputs=[]
        indices_of_interest=[]
        for e in group:
            related_input=oi[e]
            group_inputs+=[related_input]
            indices_of_interest+=[ordered_nodes.index(related_input)]
        #sub_ordered_nodes=ordered_nodes[min(indices_of_interest):max(indices_of_interest)+1]
        for i in range(min(indices_of_interest),max(indices_of_interest)+1):
            input_node=ordered_nodes[i]
            if (input_node not in group_inputs) and (input_node in iso.keys()): validity=False
            #print("Concerned input:",ordered_nodes[i])

        #print("Subordered:",sub_ordered_nodes)

        #print("Group",group)
    return validity

def common_subgraphisomorphisms_optimized_v2(query_graphs,ref_graphs,verbose=False):
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
    image_ordering=p_query.get_ordered_nodes()
    model_ordering=increasing_ordered_list(p_model)
    common_iso=[]
    for iso in t_isomorphisms_candidates:
        if photometric_iso_validity(iso,image_ordering,model_ordering): common_iso+=[iso]
    if verbose: print("Number of found common isomorphisms (inclusion+photometry):", len(common_iso))

    return common_iso


def common_subgraphisomorphisms_optimized(query_graphs,ref_graphs):
    """
    :param query_graphs: [t_graph,p_graph] from segmented image
    :param ref_graphs: [t_graph,p_graph] from model
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
    isomorphisms_candidates=[]
    #Versus first reference graph only (including its unwrapped versions, resulting from similarities)
    related_ref_graphs = all_ref_graphs[0]
    query = transitive_closure(query_graphs[0])
    related_isomorphisms = []
    counter = 1
    for ref in related_ref_graphs:
        #print(counter, "/", len(related_ref_graphs))
        counter += 1
        ref = transitive_closure(ref)
        isomorphisms = find_subgraph_isomorphims(query, ref)
        related_isomorphisms += isomorphisms
    isomorphisms_candidates += related_isomorphisms
    print("Nb valid t_iso v1:", isomorphisms_candidates)

    # Versus next: only check whether candidates is subiso
    ############################
    # CHECK VALIDITY VS BROTHERS
    ############################
    valid_isomorphisms=[]
    p_query_graph=query_graphs[1]
    p_ref_graph=ref_graphs[1]
    ordered_nodes = p_query_graph.get_ordered_nodes()
    groups_of_brothers = find_groups_of_brothers(p_ref_graph)
    nb_valid = 0
    print("Nb t iso v1:",len(isomorphisms_candidates))

    for iso in isomorphisms_candidates:
        validity = check_iso_eligibility(iso, ordered_nodes, groups_of_brothers)
        #print("Validity ISO1:",validity)
        if validity:
            nb_valid += 1
            valid_isomorphisms+=[iso]

    ############################
    #
    ############################

    current_valid_isomorphism=[]
    nb_graphs=len(query_graphs)
    for i in range(1,nb_graphs):
        query = transitive_closure(query_graphs[i])
        related_ref_graphs=all_ref_graphs[i]

        #Loop over the possible "permutations" (i.e. versus brothers/incertain relationships) of reference graphs: union of matchings
        related_isomorphisms=[]

        #print(valid_isomorphisms)
        iso_i=0
        for iso_candidate in valid_isomorphisms:
            iso_i+=1
            #Remove nodes from query that are not related to the isomorphism
            nodes_to_remove=set(query.nodes())-set(iso_candidate.keys())
            #print(nodes_to_remove)
            subquery=query.copy()
            for n in nodes_to_remove: subquery.remove_node(n)

            #Loop over unwrapped ref graphs: if iso candidate is valid at least for one of the unwrapped version -> the isomorphism candidate is valid
            counter = 1
            is_valid_candidate = False
            i=0
            while (is_valid_candidate == False) and (i<len(related_ref_graphs)):
                #print("ISO ", iso_i,"/",len(valid_isomorphisms),":", counter,"/",len(related_ref_graphs));counter+=1
                ref=related_ref_graphs[i]
                ref=transitive_closure(ref)
                validity = is_isomorphism_valid(subquery, ref, iso_candidate)
                # print(ref," --> ", validity)
                if validity:
                    is_valid_candidate = True
                i+=1

            '''
            for ref in related_ref_graphs:
                print("ISO ", iso_i,"/",len(valid_isomorphisms),":", counter,"/",len(related_ref_graphs));counter+=1
                ref=transitive_closure(ref)

                validity=is_isomorphism_valid(subquery, ref, iso_candidate)
                #print(ref," --> ", validity)
                if validity:
                    is_valid_candidate=True
                    break
            '''
            if is_valid_candidate:
                current_valid_isomorphism+=[iso_candidate]
        valid_isomorphisms=current_valid_isomorphism
    print("Nb t+p iso v1:", len(valid_isomorphisms))
    return valid_isomorphisms


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

'''
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
'''
def best_common_subgraphisomorphism(common_isomorphisms,query_p_graph,ref_p_graph,return_details=False):
    #Computing energies

    eie_per_iso=[]
    for c_iso in common_isomorphisms:
        eie_per_iso+=[matching_criterion_value(query_p_graph,ref_p_graph,c_iso)]
    eie2iso={}
    for i in range(0,len(common_isomorphisms)):
        eie2iso[eie_per_iso[i]]=common_isomorphisms[i]

    #Taking the best common isomorphisms as result
    matching=None
    max_eie=max(eie_per_iso)
    index_of_max=eie_per_iso.index(max_eie)
    matching=common_isomorphisms[index_of_max]

    #Isos in decreasing order of energy
    best_isos=[]
    my_list=sorted(eie2iso)
    my_list.reverse()
    for e in my_list:
        best_isos+=[eie2iso[e]]

    return matching,eie_per_iso,best_isos

def matching_criterion_value(query_graph,ref_graph,iso):
    oi=oirelationships(iso)
    list_of_nodes=decreasing_ordered_nodes(ref_graph)
    mean_intensities=[]
    brothers_std_dev_for_each=[]
    #new_brothers_std_dev_for_each = []
    for i in range(0,len(list_of_nodes)):
        current_element=list_of_nodes[i]
        ################################################
        #When brothers: one computes the mean intensity and std of similar nodes
        ################################################
        if len(current_element) > 1:
            local_intensities=[]
            brother_nodes=[list(oi[b])[0] for b in current_element]
            for j in range(0,len(brother_nodes)):
                local_intensities+=[query_graph.get_mean_residue_intensity(brother_nodes[j])]

            mean_intensity=np.mean(np.asarray(local_intensities))
            stddev_intensity=np.std(np.asarray(local_intensities))
            brothers_std_dev_for_each+=[stddev_intensity]
            #new_brothers_std_dev_for_each+=[stddev_intensity]
        ################################################
        #When simple node: one only retrieves the mean intensity
        ################################################
        else:
            tmp=list(current_element)[0]
            target=oi[tmp]
            corresponding_node=list(target)[0]
            mean_intensity=query_graph.get_mean_residue_intensity(corresponding_node)
            #new_brothers_std_dev_for_each += [0.0]
        ################################################
        #Keep intensity
        ################################################
        mean_intensities+=[mean_intensity]

    intensity_diffs=[ np.abs(mean_intensities[i]-mean_intensities[i-1]) for i in range(1,len(mean_intensities))]
    mean_intensity_diff=np.mean(intensity_diffs)
    mean_intensity_dev=np.std(intensity_diffs)
    ##############
    #Variance management
    ##############
    if mean_intensity_dev == 0 : mean_intensity_dev=1
    if len(brothers_std_dev_for_each) == 0 : brothers_std_dev=1
    else:
        brothers_std_dev=np.mean(np.asarray(brothers_std_dev_for_each))
        #new_brothers_std_dev = np.mean(np.asarray(new_brothers_std_dev_for_each))


    ##############
    #Energy: case without similarities
    ##############
    if len(brothers_std_dev_for_each) == 0 :
        eie = mean_intensity_diff / mean_intensity_dev #celui de l'article
        '''
        #new eie: ne marche pas pour liver et k-means
        max_diff=np.abs(mean_intensities[0]-mean_intensities[-1])
        #eie = max_diff / mean_intensity_diff
        eie = max_diff * mean_intensity_diff / mean_intensity_dev
        '''
    ##############
    # Energy: case with similarities
    ##############
    else:
        eie = mean_intensity_diff / brothers_std_dev

    return eie


def matching_criterion_value_old(query_graph,ref_graph,iso):
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

    intensity_diffs=[ np.abs(mean_intensities[i]-mean_intensities[i-1]) for i in range(1,len(mean_intensities))]
    intensity_diffs_squared=np.asarray(intensity_diffs)**2
    mean_intensity_diff=np.mean(intensity_diffs_squared)
    mean_intensity_dev=np.std(intensity_diffs_squared)
    diff_max = (np.abs(mean_intensities[0] - mean_intensities[len(mean_intensities) - 1])) ** 2

    eie=mean_intensity_diff
    if len(brothers_std_dev) > 0:
        brothers_std_dev_squared=np.asarray(brothers_std_dev)**2
        eie=eie/np.mean(brothers_std_dev_squared)
    #New:
    else:
        if mean_intensity_dev != 0:
            eie = diff_max*mean_intensity_diff / (mean_intensity_dev**2)
        else:
            eie = diff_max * mean_intensity_diff
            #eie = diff_max / (mean_intensity_dev ** 2)

    '''
    else:
        if mean_intensity_dev != 0:
            eie = mean_intensity_diff / (mean_intensity_dev**2)
    '''
    '''
    else:
        eie=eie/np.std(intensity_diffs_squared)
    '''
    '''
    for n in negative_intensities:
        eie-=n
    '''
    return eie

def nb_automorphisms(graphs):
    auto=[]
    for g in graphs:
        closed_g=transitive_closure(g)
        automorphisms=find_subgraph_isomorphims(closed_g,closed_g)
        auto+=[len(automorphisms)]
    return auto

def oirelationships_iso(io):
    """
    case bijection
    :param io:
    :return:
    """
    oi={}
    for i in io:
        my_val=io[i]
        oi[io[i]]=i
    return oi


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
