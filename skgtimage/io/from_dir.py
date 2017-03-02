import os
import numpy as np
from skgtimage.core.factory import from_string,from_regions
from skgtimage.core.graph import IrDiGraph,rename_nodes
from skgtimage.utils.color import rgb2gray
import scipy as sp;from scipy import misc


def from_dir2(directory,color=False):
    image=None
    region_names=[]
    regions=[]
    #####
    #Discover regions
    for f in os.listdir(directory):
        if f[0]==".": continue #to skip .Dstore
        (root_name,extension)=f.split(".")
        if root_name == "image":
            if extension == "npy":
                image = np.load(os.path.join(directory, f))
            else:
                image=sp.misc.imread(os.path.join(directory,f))
            if color:
                image=rgb2gray(image)
        else:
            region_names+=[root_name[len("region_"):len(root_name)]]
            if extension == "npy":
                regions += [np.load(os.path.join(directory, f))]
            else:
                regions+=[sp.misc.imread(os.path.join(directory,f))]
    print(region_names)
    #####
    #Check
    for i in range(0,len(regions)):
        for j in range(0,len(regions)):
            if i != j:
                inter=np.logical_and(regions[i],regions[j])
                maxi=np.max(inter)
                if maxi != 0: raise Exception("Error: region not mutually excluded")

    #####
    #Build graphs
    t_graph,p_graph=from_regions(image,regions)

    #####
    #Rename nodes according to file names
    remaping={}
    for i in range(0,len(region_names)):
        remaping[i]=region_names[i]
    (t_graph,p_graph)=rename_nodes([t_graph,p_graph],remaping)

    #####
    #Update intensities
    t_graph.update_intensities(image)
    for n in t_graph.nodes():
        intensity=t_graph.get_mean_residue_intensity(n)
        p_graph.set_mean_residue_intensity(n,intensity)

    return t_graph,p_graph

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




