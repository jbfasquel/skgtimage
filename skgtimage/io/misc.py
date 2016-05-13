import os
import numpy as np
from skgtimage.core.factory import from_string
from skgtimage.core.graph import IrDiGraph
from skgtimage.core.topology import fill_region
from skgtimage.core.search_base import find_head
from skgtimage.utils.histogram import int_histogram
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt


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


def plot_graph_histogram(t_graph,p_graph,fullhisto=False):
    colors="bgrcmykbgrcmykbgrcmykbgrcmyk"
    image=t_graph.get_image()
    n=list(find_head(t_graph))[0]
    roi=fill_region(t_graph.get_region(n))
    h,b=int_histogram(image,roi)
    maximum=max(h)
    if fullhisto : plt.plot(b,h,'k.')

    nodes=p_graph.nodes()
    for i in range(0,len(nodes)):
        n=nodes[i]
        region=p_graph.get_region(nodes[i])
        intensity=p_graph.get_mean_residue_intensity(n)
        c=colors[i]
        plt.plot([intensity,intensity],[0,maximum],c+'-',linewidth=2.0)
        h,b=int_histogram(image,region)
        plt.plot(b,h,c+'-')
        plt.annotate(str(n),xy=(intensity,maximum))


    plt.yscale('log')


