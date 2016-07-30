from skgtimage.core.topology import merge_nodes_topology,topological_merging_candidates
from skgtimage.core.photometry import merge_nodes_photometry
from skgtimage.core.propagation import cost2merge
import numpy as np

def remove_smallest_regions(t_graph,p_graph,number=1):
    ###############################
    # Compute the size of each region
    ###############################
    node2size={}
    considered_nodes=t_graph.nodes() #search_leaf_nodes(graph)
    for n in considered_nodes:
        size=np.count_nonzero(t_graph.get_region(n))
        node2size[n]=size
    ###############################
    # First the "number" "smallest" regions (i.e. nodes)
    ###############################
    increasing_sizes=sorted(node2size.values())
    nodes_to_remove=[]
    for i in range(0,number):
        current_size=increasing_sizes[i]
        for e in node2size:
            if (node2size[e]==current_size) and (e not in nodes_to_remove): nodes_to_remove+=[e]
    ###############################
    # Remove the "number" "smallest" regions (i.e. nodes), by merging them with their direct topological father
    ###############################
    for n in nodes_to_remove:
        if len(t_graph.successors(n)) == 1:
            father=t_graph.successors(n)[0]
            merge_nodes_topology(t_graph,n,father)
            merge_nodes_photometry(p_graph,n,father)
        elif len(t_graph.successors(n)) == 0: #cas where node to remove is head
            c=topological_merging_candidates(t_graph,n)
            orders,d2m=cost2merge(t_graph, p_graph, set([n]), c)
            merge=orders[0]
            print(c,"merge: ", merge)
            merge_nodes_topology(t_graph,merge[0],merge[1])
            merge_nodes_photometry(p_graph,merge[0],merge[1])
        elif len(t_graph.successors(n)) > 1:
            raise Exception("error")
