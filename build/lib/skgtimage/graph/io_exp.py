#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import networkx as nx
import os
import numpy as np
import scipy as sp;from scipy import misc

def save_pickle(filename,g):
    nx.write_gpickle(g, filename)

def load_pickle(filename):
    return nx.read_gpickle(filename)

def export_human_readable(directory,g,full=True):
    #clean dir
    if os.path.exists(directory) :
        for f in os.listdir(directory):
            try :
                os.remove(os.path.join(directory,f))
            except: pass
    #create dir
    if not os.path.exists(directory) : os.mkdir(directory)
    #save to pickle (to be able to load)
    filename=os.path.join(directory,"core.pkl");
    save_pickle(filename,g)
    #save to dot and svg (if possible)
    a=nx.to_agraph(g)
    a.layout(prog='neato')
    a.draw(os.path.join(directory,"core.dot"))
    a.draw(os.path.join(directory,"core.svg"))
    a.draw(os.path.join(directory,"core.png"))
    #save image: .png+.npy if 2D array, .npy only otherwise
    image=g.image()
    if len(image.shape) == 2:
        filename=os.path.join(directory,"image.png");
        sp.misc.imsave(filename, image)
    #save regions: .png+.npy if 2D array, .npy only otherwise
    for n in g.nodes():
        current_region=g.get_region(n)
        if current_region is not None:
            if len(current_region.shape) == 2:
                max=np.max(current_region)
                if max != 255 : current_region=255*(current_region/max)
                filename=os.path.join(directory,str(n)+".png");
                sp.misc.imsave(filename, current_region)
            else: raise Exception("Not a 2D image")
    
    
