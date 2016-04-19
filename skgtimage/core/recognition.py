import numpy as np
import networkx as nx
from skgtimage.core.matching import generate_common_subgraphisomorphisms,iorelationships,energie_dist,energie_sim
from skgtimage.core.brothers import find_groups_of_brothers
from skgtimage.core.search_base import search_leaf_nodes


def rename_nodes(graphs,matching):
    resulting_graphs=[]
    for g in graphs:
        resulting_graphs+=[nx.relabel_nodes(g,matching)]
    return tuple(resulting_graphs)

def remove_smallest_leaf_regions(t_graph,p_graph,number=1):
    leaf_node2size=compute_node2size_map(t_graph,True)
    #print(leaf_node2size)
    increasing_sizes=sorted(leaf_node2size.values())
    #print(increasing_sizes)
    nodes_to_remove=[]
    for i in range(0,number):
        current_size=increasing_sizes[i]
        for e in leaf_node2size:
            if leaf_node2size[e]==current_size: nodes_to_remove+=[e]
    #print(nodes_to_remove)
    for n in nodes_to_remove:
        remove_leaf_node(t_graph,p_graph,n)



def remove_leaf_node(t_graph,p_graph,node):
    if len(t_graph.successors(node)) != 1: raise Exception("error")
    ###########################
    # topology
    #Top edge

    father=t_graph.successors(node)[0]
    topological_connection=(node,father)
    region=t_graph.get_region(node)

    top_edge=(node,father)
    t_graph.remove_edge(top_edge[0],top_edge[1])
    #Bottom edge
    bottom_edges=[(i,node) for i in t_graph.predecessors(node)]
    if len(bottom_edges) != 0: raise Exception("error")
    t_graph.remove_node(node)

    ###########################
    # photo
    if len(p_graph.successors(node)) !=0:
        p_graph.remove_edge(node,p_graph.successors(node)[0])
    if len(p_graph.predecessors(node)) !=0:
        p_graph.remove_edge(p_graph.predecessors(node)[0],node)
    p_graph.remove_node(node)
    #p_graph.set_region(father,new_target_region)
    t_graph.update_intensities(p_graph.get_image())
    for n in t_graph.nodes():
        p_graph.set_mean_residue_intensity(n,t_graph.get_mean_residue_intensity(n))
    ##########################
    # recompute intensity ordering
    intensity2node={}
    for n in p_graph.nodes():
        intensity2node[p_graph.get_mean_residue_intensity(n)]=n
    increas_ordered_intensities=sorted(intensity2node)
    p_graph.remove_edges_from(p_graph.edges())
    #for i in increas_ordered_intensities: print(intensity2node[i])
    for i in range(0,len(increas_ordered_intensities)-1):
        a=intensity2node[increas_ordered_intensities[i]]
        b=intensity2node[increas_ordered_intensities[i+1]]
        p_graph.add_edge(a,b)

    ###########################
    # Return modification to be able to retablish it later
    return topological_connection,region


def compute_node2size_map(graph,leaf_only=False):
    node2size={}
    considered_nodes=graph.nodes()
    if leaf_only : considered_nodes=search_leaf_nodes(graph)
    for n in considered_nodes:
        size=np.count_nonzero(graph.get_region(n))
        node2size[n]=size
    return node2size

'''
def update_graphs_from_identified_regions(graphs,regions,matching):
    """

    :param graphs: list of graph (topological+ photometric) where nodes and edges correspond to a priori knowledge
    :param regions: regions (not residues) ordered according to matching
    :param matching: matching = map[region index]=node (a priori knowledge)
    """
    #Update graphs
    for i in matching.keys():
        id=matching[i]
        region=regions[i]
        for g in graphs:
            if g.get_region(id) is None:
                g.set_region(id,region)
'''
'''
def recognize_version1(query_graphs,ref_graphs,return_details=False):
    ###########################################################################################
    #Common matchings
    ###########################################################################################
    common_isomorphisms,isomorphisms_per_graph=generate_common_subgraphisomorphisms(query_graphs,ref_graphs)
    ###########################################################################################
    #Relationships: union of all peer2peer matchings which have been discovered
    #Warning: risk of possible inconsistency
    ###########################################################################################
    io=iorelationships(common_isomorphisms)

    return io,tuple([common_isomorphisms,tuple(isomorphisms_per_graph)])
'''

def recognize_version2(query_t_graph,ref_t_graph,query_p_graph,ref_p_graph,return_details=False):
    ###########################################################################################
    #1) Find common matchings
    ###########################################################################################
    common_isomorphisms,isomorphisms_per_graph=generate_common_subgraphisomorphisms([query_t_graph,query_p_graph],[ref_t_graph,ref_p_graph])
    #print("Nb t iso",len(isomorphisms_per_graph[0]))
    #print("Nb p iso",len(isomorphisms_per_graph[1]))
    #print("Nb common iso",len(common_isomorphisms))
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
        if nb != 1 : raise Exception("erreur")
        index_of_max=eie_dist.index(max_eie)
        matching=common_isomorphisms[index_of_max]

    print("eie_sim:", eie_sim)
    print("eie_dist:", eie_dist)

    return matching,common_isomorphisms,isomorphisms_per_graph[0],isomorphisms_per_graph[1],eie_sim,eie_dist

def closest(p_graph,n,candidates):
    candidate2distance={}
    intensity_of_n=p_graph.get_mean_residue_intensity(n)
    list_of_candidates=list(candidates)
    list_of_distances=[]
    #min_distance=
    for c in list_of_candidates:
        list_of_distances+=[np.abs(intensity_of_n-p_graph.get_mean_residue_intensity(c))]
    min_distance=min(list_of_distances)
    if list_of_distances.count(min_distance) != 1 : raise Exception("Multiple candidates")
    index_of_min_distance=list_of_distances.index(min_distance)
    candidate=list_of_candidates[index_of_min_distance]
    return candidate,min_distance

def detect_conflicts(unknown2target):
    targets=list(unknown2target.values())
    conflicting_targets=set()
    for t in targets:
        if targets.count(t) > 1: conflicting_targets|=set([t])
    conflictin_sources=[] #[(a,b,c),(j,k),(l,m)]
    for t in conflicting_targets:
        conflict=[]
        for u in unknown2target:
            if unknown2target[u] == t : conflict+=[u]
        conflictin_sources+=[conflict]
    return conflictin_sources

def matching_costs(graph,s,targets):
    target2cost={}
    #min_distance=
    for t in targets:
        target2cost[t]=np.abs(graph.get_mean_residue_intensity(s)-graph.get_mean_residue_intensity(t))
    return target2cost

def resolve_conflict(graph,sources,unknown2candidates):
    for s in sources:
        print(matching_costs(graph,s,unknown2candidates[s]))
'''
def neighboring_candidates(graph,s,possible_candidates):
    """
    Possibles (valid) candidates are with star (*): 0*,2*,3*,5*,6*
    Returned neighboring_candidates are: 0* (father), 2* (brother), 3* (direct child), 6* (direct child) ; not 5* (pb 4)
    We ignore situations (return empty set) where s has no valide father
                0*
             -> ^ <-
            |   |   |
            s   1   2*
         -> ^ <-
        |   |   |
        3*  4   6*
            ^
            |
            5*

    :param graph:
    :param s:
    :param possible_candidates:
    :return:
    """
    if len(graph.successors(s)) != 1: raise Exception("Error")
    father=graph.successors(s)[0]
    if father not in possible_candidates: return set()
    #Start analysis
    candidates=set([father]) | (set(graph.predecessors(father)) & possible_candidates ) | (set(graph.predecessors(s)) & possible_candidates )
    return candidates
'''


def ordered_photometric_candidates(p_graph,n,candidates):
    distance2candidate={}
    intensity_of_n=p_graph.get_mean_residue_intensity(n)
    for c in candidates:
        distance=np.abs(intensity_of_n-p_graph.get_mean_residue_intensity(c))
        distance2candidate[distance]=c

    ordered_distances=sorted(distance2candidate)
    ordered_candidates=[distance2candidate[d] for d in ordered_distances]
    return ordered_candidates,ordered_distances

def neighboring_candidates(graph,s,possible_candidates):
    """
    Possibles (valid) candidates are with star (*): 0*,2*,3*,5*,6*
    Returned neighboring_candidates are: 0* (father), 2* (brother), 3* (direct child), 6* (direct child) ; not 5* (pb 4)
    We ignore situations (return empty set) where s has no valide father
                0*
             -> ^ <-
            |   |   |
            s   1   2*
         -> ^ <-
        |   |   |
        3*  4   6*
            ^
            |
            5*

    :param graph:
    :param s:
    :param possible_candidates:
    :return:
    """
    if len(graph.successors(s)) != 1: raise Exception("Error")
    father=graph.successors(s)[0]
    #if father not in possible_candidates: return set()
    #Start analysis
    candidates=set([father]) | set(graph.predecessors(father)) | set(graph.predecessors(s))
    candidates-=set([s])
    candidates&=possible_candidates #without this rule, fail for example 03: region
    return candidates


def apply_merging(t_graph,p_graph,source,target):
    if len(t_graph.successors(source)) != 1: raise Exception("error")
    ###########################
    # topology
    #Top edge
    top_edge=(source,t_graph.successors(source)[0])
    t_graph.remove_edge(top_edge[0],top_edge[1])
    #Bottom edge
    bottom_edges=[(i,source) for i in t_graph.predecessors(source)]
    for e in bottom_edges:
        t_graph.remove_edge(e[0],e[1])
        if e[0] != target:
            t_graph.add_edge(e[0],target)
        else:
            t_graph.add_edge(e[0],top_edge[1])

    new_target_region=np.logical_or(t_graph.get_region(source),t_graph.get_region(target)).astype(np.uint8)
    t_graph.set_region(target,new_target_region)
    t_graph.remove_node(source)
    ###########################
    # photometry
    if len(p_graph.successors(source)) !=0:
        p_graph.remove_edge(source,p_graph.successors(source)[0])
    if len(p_graph.predecessors(source)) !=0:
        p_graph.remove_edge(p_graph.predecessors(source)[0],source)
    p_graph.remove_node(source)
    p_graph.set_region(target,new_target_region)
    t_graph.update_intensities(p_graph.get_image())
    for n in t_graph.nodes():
        p_graph.set_mean_residue_intensity(n,t_graph.get_mean_residue_intensity(n))
    ##########################
    # recompute intensity ordering
    intensity2node={}
    for n in p_graph.nodes():
        intensity2node[p_graph.get_mean_residue_intensity(n)]=n
    increas_ordered_intensities=sorted(intensity2node)
    p_graph.remove_edges_from(p_graph.edges())
    #for i in increas_ordered_intensities: print(intensity2node[i])
    for i in range(0,len(increas_ordered_intensities)-1):
        a=intensity2node[increas_ordered_intensities[i]]
        b=intensity2node[increas_ordered_intensities[i+1]]
        p_graph.add_edge(a,b)
        '''
        if not(p_graph.has_edge(a,b)): print("warning")
        else: print("ok")
        '''

def check_merging_consistency(new_t_graph,new_p_graph,ref_t_graph,ref_p_graph,ref_matching):
    """
    Check that at least one "common isomorphism" is valide vs ref_matching
    :param new_t_graph: refined graph
    :param new_p_graph: refined graph
    :param ref_t_graph: reference graph
    :param ref_p_graph: reference graph
    :param ref_matching: reference matching between initial graphs and reference graphs
    :return:
    """
    #Computing common isomorphisms
    common_isomorphisms,_=generate_common_subgraphisomorphisms([new_t_graph,new_p_graph],[ref_t_graph,ref_p_graph])
    #Checking validity
    validities=[]
    for c in common_isomorphisms:
        is_ok=True
        for e in ref_matching:
            if e not in c: is_ok=False
        validities+=[is_ok]
    if len(validities)==0:
        is_valid=False
    else:
        is_valid = (max(validities) == True)
    if is_valid is False:
        a=10 #to capture debug
    return is_valid

def greedy_refinement_v3(t_graph,p_graph,ref_t_graph,ref_p_graph,ref_matching):
    """
    While remaining unknown nodes:
        1) For each r in remaining unknown nodes: find topological neighbors that are known, and compute photometric distances
        2) Merge the node of remaining unknown nodes depicting the smallest distance and then: merge=(node,target)
            2-a) If graph consistency is preserved: update the list of remaining unknown nodes.
                 Else: undo merge, and remove merge=(node,target) from possibilities

    Reestimate each time distances...
    :param t_graph:
    :param p_graph:
    :param ref_t_graph:
    :param ref_p_graph:
    :param ref_matching:
    :return:
    """
    #########
    #simplified_ref_p_graph=ref_p_graph.copy()
    simplified_ref_p_graph=hack(p_graph,ref_matching) #p_graph considered in matching, i.e. without "brothers"


    #Initially, matched nodes (identified) are those of the initial matching
    initial_matched_nodes=set(ref_matching.keys())
    remaining_nodes=set(t_graph.nodes())-initial_matched_nodes
    print("Remaining nodes",remaining_nodes)
    modification_historisation=[]
    ###################
    #Start test
    ###################
    invalid_merge=[]
    previous_t_graph=t_graph
    previous_p_graph=p_graph
    min_distance_order=0
    while len(remaining_nodes)>0:
        remaining_nodes2ordered_candidates={} #e.g. remaining_nodes2ordered_candidates[5]==[10,0]
        remaining_nodes2ordered_distances={} #e.g. remaining_nodes2ordered_distances[5]==[0.5,12.5]
        mindistance2matching={} #e.g. mindistance2matching[0.5]==(5,10)
        for e in remaining_nodes:
            #target_candidates=neighboring_candidates(t_graph,e,initial_matched_nodes)
            #ordered_target_candidates,ordered_target_distances=ordered_photometric_candidates(p_graph,e,target_candidates)
            target_candidates=neighboring_candidates(previous_t_graph,e,initial_matched_nodes)
            ordered_target_candidates,ordered_target_distances=ordered_photometric_candidates(previous_p_graph,e,target_candidates)
            if len(target_candidates) != 0:
                remaining_nodes2ordered_candidates[e]=ordered_target_candidates
                remaining_nodes2ordered_distances[e]=ordered_target_distances
                min_distance=ordered_target_distances[0]
                matching=(e,ordered_target_candidates[0])
                mindistance2matching[min_distance]=matching
        '''
        for e in remaining_nodes2ordered_candidates:
            print("Node ",e," neighbors: ",remaining_nodes2ordered_candidates[e], " (dist: ",remaining_nodes2ordered_distances[e],")")
        '''
        for d in sorted(mindistance2matching):
            print("Distance:", d, " Matching = ",mindistance2matching[d])

        #Merge
        mini=sorted(mindistance2matching)[min_distance_order]
        print("min distance",mini)
        merge=mindistance2matching[mini]
        print("Merging",merge)
        if merge in invalid_merge:
            considered_remaining_node=merge[0]
            considered_target_node=merge[1]
            do_continue=True
            offset=1
            while do_continue:
                target_index=remaining_nodes2ordered_candidates[considered_remaining_node].index(considered_target_node)
                new_target_node=remaining_nodes2ordered_candidates[considered_remaining_node][target_index+offset]
                new_merge=(considered_remaining_node,new_target_node)

                if (new_merge in invalid_merge) and (offset+1 <= len(remaining_nodes2ordered_candidates[considered_remaining_node])):
                    do_continue=True
                    offset+=1
                    print("---> New merging candidate ",merge, " : NOK")
                else:
                    do_continue=False
            merge=new_merge
            print("---> New merging candidate ",merge)
            if merge in invalid_merge:
                print("FIX ME")

        #Historisation
        current_t_graph=previous_t_graph.copy()
        current_p_graph=previous_p_graph.copy()
        #modification_historisation+=[(t_graph.copy(),p_graph.copy(),merge)]
        #Perform copies

        apply_merging(current_t_graph,current_p_graph,merge[0],merge[1])
        validity=check_merging_consistency(current_t_graph,current_p_graph,ref_t_graph,simplified_ref_p_graph,ref_matching)
        if validity == True:
            previous_t_graph=current_t_graph
            previous_p_graph=current_p_graph
            modification_historisation+=[(previous_t_graph,previous_p_graph,merge)]
            remaining_nodes=set(current_t_graph.nodes())-initial_matched_nodes
        elif validity == False:
            validity_with_brothers=check_merging_consistency(current_t_graph,current_p_graph,ref_t_graph,ref_p_graph,ref_matching)
            if validity_with_brothers==True:
                simplified_ref_p_graph=hack(current_p_graph,ref_matching) #p_graph considered in matching, i.e. without "brothers"
            else:
                invalid_merge+=[merge]
            print("Fail to merge",merge)
            print("Validity 1:", validity, " But validity 2:",validity_with_brothers)

        print("Remaining nodes: ", remaining_nodes)
    ###################
    #End test
    ###################
    return current_t_graph,current_p_graph,modification_historisation


def hack(built_p,matching):
    """
    return new ref graph correspnding to matching without amiguity
    :return:
    """
    import networkx as nx
    built_p=built_p.copy()
    #keep_only_nodes(built_p,matching.keys())
    nodes_to_remove=set(built_p.nodes())-set(matching.keys())
    for n in nodes_to_remove:
        if len(built_p.successors(n)) != 1: raise Exception("error")
        father=built_p.successors(n)[0]
        top_edge=(n,father)
        built_p.remove_edge(top_edge[0],top_edge[1])
        #Bottom edge
        bottom_edges=[(i,n) for i in built_p.predecessors(n)]
        for e in bottom_edges:
            built_p.remove_edge(e[0],e[1])
            built_p.add_edge(e[0],father)
        built_p.remove_node(n)
    #Relabelling to be conform to initial reference graph
    clean_ref_p=nx.relabel_nodes(built_p,matching)
    return clean_ref_p

