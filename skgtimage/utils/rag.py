import numpy as np
from skgtimage.core.isomorphism import common_subgraphisomorphisms
from skimage.future import graph as skimage_graph
from skgtimage.core.factory import from_labelled_image


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

def rag_merge(image,labelled,threshold,mc=False,roi=None):

    rgb=image
    if mc == False:
        rgb = np.dstack((image, np.zeros(image.shape), np.zeros(image.shape)))
    #if type(rgb)==np.ma.MaskedArray:
    #    rgb=rgb.filled(0)
    if roi is not None:
        labelled=labelled+1
        labelled = np.ma.array(labelled, mask=np.logical_not(roi)).filled(0)

    # g = graph.rag_mean_color(image, label)
    rag = skimage_graph.rag_mean_color(rgb, labelled, mode='distance')
    labelled=skimage_graph.merge_hierarchical(labelled, rag, threshold,rag_copy=False,in_place_merge=True,merge_func=__merge_mean_color__,weight_func=__weight_mean_color__)  # 30 OK, 50 mieux)

    if roi is not None:
        labelled+=1
        labelled = np.ma.array(labelled, mask=np.logical_not(roi)).filled(0)
    return labelled

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


def rag2labelled(rag,label):
    new_labelled_region=np.copy(label)
    for n in rag.node:
        labels=rag.node[n]['labels']
        if len(labels)>1:
            for l in labels:
                region=np.where(label==l,1,0)
                new_labelled_region=np.ma.masked_array(new_labelled_region,mask=region).filled(n)
    return new_labelled_region


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

    common_isomorphisms = common_subgraphisomorphisms([t_graph, p_graph], [ref_t_graph, ref_p_graph])
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
        common_isomorphisms = common_subgraphisomorphisms([new_t_graph, new_p_graph], [ref_t_graph, ref_p_graph])


    return new_t_graph,new_p_graph