import networkx as nx
from skgtimage.core.isomorphism import find_subgraph_isomorphims,common_subgraphisomorphisms
from skgtimage.core.topology import topological_merging_candidates,merge_nodes_topology
from skgtimage.core.photometry import merge_nodes_photometry
from skgtimage.core.graph import transitive_closure

def extract_subgraph(built_p,subnodes):
    """
    Return
    :return:
    """
    built_p=built_p.copy()
    nodes_to_remove=set(built_p.nodes())-set(subnodes)
    for n in nodes_to_remove:
        if len(built_p.successors(n)) == 1:
            father=built_p.successors(n)[0]
            top_edge=(n,father)
            built_p.remove_edge(top_edge[0],top_edge[1])
            #Bottom edge
            bottom_edges=[(i,n) for i in built_p.predecessors(n)]
            for e in bottom_edges:
                built_p.remove_edge(e[0],e[1])
                built_p.add_edge(e[0],father)
            built_p.remove_node(n)
        elif len(built_p.successors(n)) == 0:
            if len(built_p.predecessors(n)) !=0:
                pred=built_p.predecessors(n)[0]
                built_p.remove_edge(pred,n)
            built_p.remove_node(n)
    return built_p


def check_merge_validity(previous_t_graph,previous_p_graph,current_t_graph,current_p_graph,ref_t_graph,ref_p_graph,ref_matching):
    ######################
    # Test 1: one checks topological constraints only.
    # If not verified, then not necessary to further investigate photometric constraints (to spare cputime): one returns "false"
    ######################
    if ref_matching not in find_subgraph_isomorphims(transitive_closure(current_t_graph),transitive_closure(ref_t_graph)): return False

    ######################
    # Test 2: one checks both topological and photometric constraints,
    # Verification performed without considering possible valid photometric relationships inversion (enabled by "brother links") -> one spare cputime
    ######################
    subgraph=extract_subgraph(previous_p_graph,ref_matching.keys()) #reference ref_p_graph considered in matching, i.e. without "brothers"
    simplified_ref_p_graph=nx.relabel_nodes(subgraph,ref_matching)
    #common_isomorphisms,_=common_subgraphisomorphisms([current_t_graph,current_p_graph],[ref_t_graph,simplified_ref_p_graph])
    common_isomorphisms=common_subgraphisomorphisms([current_t_graph, current_p_graph], [ref_t_graph, simplified_ref_p_graph])
    validity= (ref_matching in common_isomorphisms)

    ######################
    # Test 3: one checks both topological and photometric constraints, considering photometric similarities
    #If modifications have involved photometric relationships "inversions" (previous validity is therefore false)
    #we check wheter matching still exist by considering the "full" reference photometric graph with all possible "brothers" (more time consuming)
    #In such a case, ref_matching can be modified (e.g. valid photometric inversion)
    ######################
    if validity == False:
        common_isomorphisms= common_subgraphisomorphisms([current_t_graph, current_p_graph], [ref_t_graph, ref_p_graph])
        validity= (ref_matching in common_isomorphisms)

    return validity


def cost2merge(t_graph,p_graph,nodes,possible_targets):
    #d2m: distance to merge (link) hash table
    d2m={}
    ###############################
    #For each nodes e, targets candidates verifies following constraintes:
    # 1) they are "possible targets" (i.e. already matched nodes)
    # 2) they are topologically acceptable targets (versus node e): either direct predecessor or successor or brother
    #For each target, we compute the intensity difference versus the current node e
    ###############################
    for e in nodes:
        target_candidates=topological_merging_candidates(t_graph,e) & set(possible_targets)
        intensity_of_e=p_graph.get_mean_intensity(e)
        for t in target_candidates:
            intensity_of_t=p_graph.get_mean_intensity(t)
            distance=abs(intensity_of_e-intensity_of_t)
            if distance in d2m:
                d2m[distance]+=[(e,t)]
            else:
                d2m[distance]=[(e,t)]
    ###############################
    #Candidates are possible merges (link) sorted in the increasing order of the related intensity difference
    ###############################
    candidates=[]
    for d in sorted(d2m):
        candidates+=d2m[d]

    return candidates,d2m

def propagate(t_graph,p_graph,ref_t_graph,ref_p_graph,ref_matching,visual_debug=False,verbose=False):
    """
    While remaining unknown nodes:
        1) For each r in remaining unknown nodes: find topological neighbors that are known, and compute photometric distances
        2) Selection merging corresponding to the smallest distance
        3) Apply the node merging
        4) If merging preserves both topological and photometric constraints: the merge is conserved
           Else: one repeat 3) with next possible merge (less small distance)

        Note: if all possible merges lead break topological and photometric constraints, then propagation stops: failure (impossibility)

    :param t_graph:
    :param p_graph:
    :param ref_t_graph:
    :param ref_p_graph:
    :param ref_matching:
    :return:
    """
    ###################
    #Initialization
    ###################
    previous_t_graph=t_graph
    previous_p_graph=p_graph
    modification_historisation=[]
    initial_matched_nodes=set(ref_matching.keys())
    remaining_nodes=set(t_graph.nodes())-initial_matched_nodes
    ###################
    #Loop until all "unidentified nodes" are merged with identified ones
    ###################
    while len(remaining_nodes)>0:
        ###################
        #Step 1: for current remaing nodes, we compute the cost of merging each of these remaining nodes with closest matched nodes
        #The leads to a sorted list of possible merges
        #Note: ordered_merging_candidates are recomputed each time a valid merge is performed, because it modifies "costs"
        #of other merging candidates (indeed, merging modifies regions and related photometric properties on which costs
        #are computed)
        ###################
        ordered_merging_candidates,d2m=cost2merge(previous_t_graph,previous_p_graph,remaining_nodes,initial_matched_nodes)
        #ordered_merging_candidates example: [(4, 0),(4, 1),(x,y),(2, 3)]
        #d2m example: {0.59999999999999998: [(4, 1),(x,y)], 0.90000000000000002: [(2, 3)], 0.38518518518518519: [(4, 0)]}


        ###################
        #Step 2: We try to perform one merging starting with the first one (lowest cost)
        #The merge is considered as valid if resulting graphs (current_t_graph and current_p_graph)
        #still match both topological and photometric constraints, otherwise the next merge is tested... and so on
        #until: a) a valid merge is encountered, b) none is possible merges is valid (raise exception)
        ###################
        current_candidate_index=0
        stop_condition=False
        while stop_condition==False:
            ###########################
            #Retrieve the current merge
            ###########################
            merge=ordered_merging_candidates[current_candidate_index]
            ###########################
            #If "visual debug" True: plot each intermediate graph
            ###########################
            if visual_debug:
                from skgtimage.io import plot_matching,matching2links;import matplotlib.pyplot as plt
                ordered_merges=[i[2] for i in modification_historisation]
                plot_matching(t_graph, ref_t_graph, link_lists=[matching2links(ref_matching), ordered_merges, [merge]], colors=['red', 'green', 'yellow']);plt.show()

            ###########################
            #Apply merging on graph copies (temporarly node merging)
            ###########################
            current_t_graph=previous_t_graph.copy()
            current_p_graph=previous_p_graph.copy()
            merge_nodes_photometry(current_p_graph,merge[0],merge[1])
            merge_nodes_topology(current_t_graph,merge[0],merge[1])

            ###########################
            #Check that topological and photometric constraints are still verified after (temporarly node merging)
            ###########################
            validity=check_merge_validity(previous_t_graph,previous_p_graph,current_t_graph,current_p_graph,ref_t_graph,ref_p_graph,ref_matching)

            ###########################
            #Use validity to decide whether the next merge should be tested (within the "ordered_merging_candidates" list)
            ###########################
            if validity:
                if verbose: print(ordered_merging_candidates, " Merge: ", merge, " -> OK")
                previous_t_graph=current_t_graph
                previous_p_graph=current_p_graph
                modification_historisation+=[merge]
                remaining_nodes=set(current_t_graph.nodes())-initial_matched_nodes
                stop_condition=True
            else:
                if verbose: print(ordered_merging_candidates, " Merge: ", merge, " -> NOK")
                stop_condition=False
                current_candidate_index+=1
                if current_candidate_index == len(ordered_merging_candidates):
                    #print("Here",current_candidate_index)
                    #One removes the last region from candidates
                    concerned_merge=ordered_merging_candidates[current_candidate_index-1]
                    region_to_remove=concerned_merge[0]
                    if verbose: print("Removing node",region_to_remove)
                    previous_t_graph.remove_node(region_to_remove)
                    previous_p_graph.remove_node(region_to_remove)
                    #One reiterates the procedure the remove the other "invalid" candidates
                    previous_t_graph, previous_p_graph,tmp=propagate(previous_t_graph,previous_p_graph,ref_t_graph,ref_p_graph,ref_matching,visual_debug,verbose)
                    modification_historisation+=tmp
                    return previous_t_graph, previous_p_graph,modification_historisation

    ###################
    #Return
    ###################
    return previous_t_graph,previous_p_graph,modification_historisation

