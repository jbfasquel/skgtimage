import networkx as nx
import os
from skgtimage.core.factory import from_labelled_image
from skgtimage.core.photometry import update_photometric_graph
from skgtimage.core.subisomorphism import common_subgraphisomorphisms_optimized,common_subgraphisomorphisms_optimized_v2
from skgtimage.core.search_base import find_head
from skgtimage.core.topology import fill_region
#from skgtimage.io.with_graphviz import save_graph,save_graphregions

def background_removal_by_size(image,labelled_image,ref_t,ref_p,save_dir=None):
    """
    To remove background: this provides a ROI within which analysis is performed
    This enable to manage situation where background can not be described in the model (e.g. impossibility to assumed any
    photometric relationships with other regions).
    :param image:
    :param labelled_image:
    :return: roi (i.e. complementary of the backgroung)
    """
    pass


def background_removal_by_iso(image,t,p,ref_t,ref_p,verbose=False):
    """
    To remove background: this provides a ROI within which analysis is performed
    This enable to manage situation where background can not be described in the model (e.g. impossibility to assumed any
    photometric relationships with other regions).
    :param image:
    :param labelled_image:
    :return: roi (i.e. complementary of the backgroung)
    """

    t_cs=list(nx.weakly_connected_component_subgraphs(t))
    p_cs=[]
    nbisos=[]
    nbnodes=[]
    for i in range(0,len(t_cs)):
        t_c=t_cs[i]
        t_c.set_image(image)
        p_c=nx.subgraph(p,t_c.nodes())
        p_c.set_image(image)
        update_photometric_graph(p_c)
        p_cs+=[p_c]
        if len(t_c.nodes()) >= len(ref_t.nodes()): #to avoid useless computation
            common_isos=common_subgraphisomorphisms_optimized_v2([t_c, p_c],[ref_t, ref_p])
            nbisos+=[len(common_isos)]
        else:
            nbisos += [0]
        nbnodes+=[len(t_c.nodes())]
        if verbose: print("Connected component ", i, " : nodes:", t_c.nodes(), " -> nb isos:", nbisos[-1], " -> nb nodes:", len(t_c.nodes()))

    index_of_max=nbisos.index(max(nbisos))
    t_c_max=t_cs[index_of_max]
    p_c_max=p_cs[index_of_max]
    node_of_interest=list(find_head(t_c_max))[0]
    roi=t_c_max.get_region(node_of_interest)
    roi=fill_region(roi)

    return roi,t_c_max,p_c_max
