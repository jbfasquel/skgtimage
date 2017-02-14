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
    #t, p = from_labelled_image(gray_image, labelled, roi)
    #tmp=t.get_labelled()
    #tmp=labelled
    #gray = skgti.utils.rgb2gray(image)
    # gray=np.dstack((gray,gray,gray))
    #tmp_labelled=np.copy(labelled)
    #if roi is not None:

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
