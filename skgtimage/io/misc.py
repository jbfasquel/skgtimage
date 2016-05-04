import os
import numpy as np
from skgtimage.core.factory import from_string
from skgtimage.core.graph import IrDiGraph
from skgtimage.core.search_base import find_head
import scipy as sp;from scipy import misc


def compute_intensitymap(t_graph,do_round=False):
    mapping={}
    for n in t_graph.nodes():
        #residue=t_graph.get_residue(n)
        intensity=t_graph.get_mean_residue_intensity(n)
        if do_round : intensity=np.round(intensity,0)
        mapping[n]=intensity
    return mapping

def generate_single_image(t_graph,mapping=None):
    result=np.zeros(t_graph.get_image().shape)
    for n in t_graph.nodes():
        #residue=t_graph.get_residue(n)
        residue=t_graph.get_region(n)
        if mapping is not None:
            intensity=mapping[n]
        else:
            intensity=t_graph.get_mean_residue_intensity(n)
        result=np.ma.masked_array(result, mask=residue).filled(intensity)
    return result

def from_dir(desc_t,desc_p,image,directory):
    t_graph=from_string(desc_t,IrDiGraph())
    p_graph=from_string(desc_p,IrDiGraph())
    t_graph.set_image(image)
    p_graph.set_image(image)
    if len(image.shape)==2:
        for n in t_graph.nodes():
            if os.path.exists(os.path.join(directory,"region_"+str(n)+".npy")):
                region=np.load(os.path.join(directory,"region_"+str(n)+".npy"))
            else:
                region=sp.misc.imread(os.path.join(directory,"region_"+str(n)+".png"))
            t_graph.set_region(n,region)
            p_graph.set_region(n,region)

    if len(image.shape)==3:
        for n in t_graph.nodes():
            region=np.load(os.path.join(directory,"region_"+str(n)+".npy"))
            t_graph.set_region(n,region)
            p_graph.set_region(n,region)

    t_graph.update_intensities(image)
    #Hack
    for n in t_graph.nodes():
        intensity=t_graph.get_mean_residue_intensity(n)
        p_graph.set_mean_residue_intensity(n,intensity)

    return t_graph,p_graph
