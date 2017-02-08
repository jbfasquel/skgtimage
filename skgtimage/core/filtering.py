from skgtimage.core.topology import merge_nodes_topology,topological_merging_candidates
from skgtimage.core.photometry import merge_nodes_photometry
from skgtimage.core.propagation import cost2merge
from skgtimage.core.subisomorphism import common_subgraphisomorphisms_optimized_v2
from skgtimage.core.factory import photometric_graph_from_residues_refactorying,from_labelled_image
import numpy as np
import skimage; from skimage.future import graph as skimage_graph


def __weight_mean_color__(graph, src, dst, n):
    """
    From skimage
    Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff #return {'weight': diff} #probably in next version of skimage

'''
def __merge_mean_color__(graph, src, dst):
    """
    From skimage
    Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] / graph.node[dst]['pixel count'])

def rag_merge(gray_image,labelled,threshold,roi=None):
    t, p = from_labelled_image(gray_image, labelled, roi)
    tmp=t.get_labelled()
    #tmp=labelled
    #gray = skgti.utils.rgb2gray(image)
    # gray=np.dstack((gray,gray,gray))
    gray_rgb = np.dstack((gray_image, np.zeros(gray_image.shape), np.zeros(gray_image.shape)))
    # g = graph.rag_mean_color(image, label)
    rag = skimage_graph.rag_mean_color(gray_rgb, tmp, mode='distance')
    labelled=skimage_graph.merge_hierarchical(tmp, rag, threshold,rag_copy=False,in_place_merge=True,merge_func=__merge_mean_color__,weight_func=__weight_mean_color__)  # 30 OK, 50 mieux)
    return labelled
'''

def compute_dm(p_g):
    ord_n = p_g.get_ordered_nodes()
    dm = {}
    for j in range(1, len(ord_n)):
        #diff = abs(p_g.get_mean_residue_intensity(j) - p_g.get_mean_residue_intensity(j - 1))
        diff = abs(p_g.get_mean_residue_intensity(ord_n[j]) - p_g.get_mean_residue_intensity(ord_n[j-1]))
        if diff in dm:
            #dm[diff] += [(j - 1, j)]
            dm[diff] += [(ord_n[j - 1], ord_n[j])]
        else:
            #dm[diff] = [(j - 1, j)]
            dm[diff] = [(ord_n[j - 1], ord_n[j])]
    return dm



def merge_photometry_gray(image, label, nb_times=10):
    residues = [np.where(label == i, 255, 0) for i in range(0, np.max(label) + 1)]
    p_g = photometric_graph_from_residues_refactorying(image, residues)
    #dm=compute_dm(p_g)
    for i in range(0, nb_times):
        dm = compute_dm(p_g)
        min_diff = sorted(dm)[0]
        merge = dm[min_diff][0]
        #print("merge:",merge)
        if len(dm[min_diff]) == 1: dm.pop(min_diff)
        if (merge[0] in p_g.nodes()) and (merge[1] in p_g.nodes()):
            merge_nodes_photometry(p_g, merge[0], merge[1])
        #dm = compute_dm(p_g)

    new_labelled = np.zeros(image.shape, dtype=np.int16)
    nodes = p_g.nodes()
    print(nodes)
    for i in range(0, len(nodes)):
        roi = p_g.get_region(nodes[i])
        new_labelled = np.ma.masked_array(new_labelled, mask=roi).filled(i)

    return new_labelled



def merge_filtering(t_graph,p_graph,nb_times=1):
    for i in range(0,nb_times):
        remaining_nodes = t_graph.nodes()
        print("Merging - Remaining nodes", len(remaining_nodes))
        ordered_merging_candidates, d2m = cost2merge(t_graph, p_graph, remaining_nodes, remaining_nodes)
        merge = ordered_merging_candidates[0]
        # Apply merging
        merge_nodes_photometry(p_graph, merge[0], merge[1])
        merge_nodes_topology(t_graph, merge[0], merge[1])




def size_filtering(t_graph,p_graph,threshold=0,verbose=False):
    ###############################
    # Compute the size of each region
    ###############################
    node2size={}
    considered_nodes=t_graph.nodes() #search_leaf_nodes(graph)
    for n in considered_nodes:
        size=np.count_nonzero(t_graph.get_region(n))
        node2size[n]=size

    nodes_to_remove=[]
    for n in node2size:
        if node2size[n] < threshold:
            nodes_to_remove+=[n]

    if verbose:
        print("Sizes:", sorted(node2size.values()))
        print("--> ", len(nodes_to_remove)," nodes to remove:", nodes_to_remove)
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


def rag2labelled(rag,label):
    new_labelled_region=np.copy(label)
    for n in rag.node:
        labels=rag.node[n]['labels']
        if len(labels)>1:
            for l in labels:
                region=np.where(label==l,1,0)
                new_labelled_region=np.ma.masked_array(new_labelled_region,mask=region).filled(n)
    return new_labelled_region

def rag_merge_candidates(rag):
    ordered_edges = {}
    for e in rag.edge:
        src = e
        for dest in rag.edge[e]:
            diff = rag.edge[e][dest]['weight']
            # print(dest,rag.edge[e][dest]['weight'])
            if diff in ordered_edges:
                if (dest, src) not in ordered_edges[diff]:
                    ordered_edges[diff] += [(src, dest)]
            else:
                ordered_edges[diff] = [(src, dest)]
        #print(e, rag.edge[e])

    # print(sorted(ordered_edges))
    merging_candidates=[]
    for i in sorted(ordered_edges):
        merging_candidates+=ordered_edges[i]
    return merging_candidates


def rag_merge_until_commonisomorphism(t_graph,p_graph,ref_t_graph,ref_p_graph,image,roi=None,mc=False,verbose=False):
    """
    TODO:
    :param t_graph:
    :param p_graph:
    :param ref_t_graph:
    :param ref_p_graph:
    :param roi:
    :param verbose:
    :return:
    """
    ##########
    #Build labelled image from topological graph
    label,image_gray=t_graph.get_labelled(),t_graph.get_image()
    # g = graph.rag_mean_color(image, label)

    ##########
    #Build rag
    if mc is False:
        image_rgb = np.dstack((image, np.zeros(image.shape), np.zeros(image.shape)))
    else:
        image_rgb=image
    rag = skimage_graph.rag_mean_color(image_rgb, label, mode='distance')

    new_t_graph, new_p_graph=t_graph,p_graph

    common_isomorphisms = common_subgraphisomorphisms_optimized_v2([t_graph, p_graph], [ref_t_graph, ref_p_graph])
    while len(common_isomorphisms) == 0:
        ##########
        #Merge the 2 most similar regions
        merging_candidates=rag_merge_candidates(rag)
        if len(merging_candidates)==0: raise Exception("RAG merge no more nodes")
        to_merge = merging_candidates[0]
        if verbose: print("RAG merge:",to_merge)
        src, dst = to_merge[0], to_merge[1]
        rag.node[dst]['total color'] += rag.node[src]['total color']
        rag.node[dst]['pixel count'] += rag.node[src]['pixel count']
        rag.node[dst]['mean color'] = (rag.node[dst]['total color'] / rag.node[dst]['pixel count'])
        rag.merge_nodes(to_merge[0], to_merge[1], __weight_mean_color__)
        ##########
        #Update t,p graph
        new_labelled_region=rag2labelled(rag,label)
        #merge_nodes_photometry(p_graph, to_merge[0], to_merge[1])
        #merge_nodes_topology(t_graph, to_merge[0], to_merge[1])

        new_t_graph,new_p_graph=from_labelled_image(image_gray,new_labelled_region,roi,verbose=verbose)
        common_isomorphisms = common_subgraphisomorphisms_optimized_v2([new_t_graph, new_p_graph], [ref_t_graph, ref_p_graph])


    return new_t_graph,new_p_graph