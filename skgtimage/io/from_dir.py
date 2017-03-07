import os,re
import numpy as np
from skgtimage.core.factory import from_string,from_regions
from skgtimage.core.graph import IrDiGraph,relabel_nodes
from skgtimage.utils.color import rgb2gray
import scipy as sp;from scipy import misc

def from_dir(directory, image=None, t_desc=None, p_desc=None, mc=False):
    #####
    # Manage image
    #####
    if image is None:
        if os.path.exists(os.path.join(directory, "image.npy")):
            image=np.load(os.path.join(directory, "image.npy"))
        elif os.path.exists(os.path.join(directory, "image.png")):
            image=sp.misc.imread(os.path.join(directory, "image.png"))
    #####
    # Manage color
    #####
    if mc: image=rgb2gray(image)
    #####
    # Manage regions
    #####
    id2region = {}
    for f in os.listdir(directory):
        if re.match("region_.*\.npy", f) is not None:
            r_name = (f.split('.')[0]).split('_')[1]
            region = np.load(os.path.join(directory, f))
            id2region[r_name]=region
        elif re.match("region_.*\.png", f) is not None:
            r_name = (f.split('.')[0]).split('_')[1]
            region = sp.misc.imread(os.path.join(directory, f))
            id2region[r_name] = region
    #####
    # Check validity (mutual exclusion)
    #####
    all_regions = list(id2region.values())
    for i in range(0, len(all_regions)):
        for j in range(0, len(all_regions)):
            if i != j:
                inter = np.logical_and(all_regions[i], all_regions[j])
                maxi = np.max(inter)
                if maxi != 0: raise RuntimeWarning("Warning: region not mutually excluded (can be ignored -> managed by from_regions() function)")

    #####
    # Build graphs
    #####
    if ( t_desc is not None ) and ( p_desc is not None ):
        t_graph = from_string(t_desc,IrDiGraph())
        p_graph = from_string(p_desc,IrDiGraph())
        t_graph.set_image(image)
        p_graph.set_image(image)
        for id in id2region:
            region=id2region[id]
            t_graph.set_region(id, region)
            p_graph.set_region(id, region)
        t_graph.update_intensities(image)
        p_graph.update_intensities(image)

    else:
        regions=id2region.values()
        t_graph, p_graph = from_regions(image, regions)
        t_graph.update_intensities(image)
        # Relabel
        node2id = {}
        for n in t_graph.nodes():
            for id in id2region:
                region_graph = t_graph.get_region(n)
                region_fs = id2region[id].astype(np.float)
                if np.max(region_graph) != 0: region_graph /= np.max(region_graph)
                if np.max(region_fs) != 0: region_fs /= np.max(region_fs)
                if np.array_equal(region_graph, region_fs):
                    node2id[n] = id
        t_graph, p_graph = relabel_nodes([t_graph, p_graph], node2id)

    ######
    # Return
    ######
    return t_graph,p_graph

