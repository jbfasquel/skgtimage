#Author: Jean-Baptiste Fasquel <Jean-Baptiste.Fasquel@univ-angers.fr>, LARIS Laboratory, Angers University, France
#Copyright (C) 2015 Jean-Baptiste Fasquel
#Licence: BSD 3 clause

import numpy as np
import time
from sklearn import cluster

def kmeans(image,nb_clusters,n_seedings=100,roi=None,mc=False,verbose=False):
    """
    Kmeans
    """
    #Multi-component image
    nb_components,spatial_dim=1,len(image.shape)
    if mc:
        nb_components=image.shape[-1]
        spatial_dim-=1

    if roi is not None:
        roi_mask=np.dstack(tuple([roi for i in range(0,nb_components)]))
        roied_image=np.ma.masked_array(image,mask=np.logical_not(roi_mask))
        return kmeans(roied_image,nb_clusters,n_seedings,None,mc,verbose)

    if type(image) == np.ma.masked_array :
        reshaped_data=image.compressed()
        reshaped_data=reshaped_data.reshape(-1,nb_components).astype(np.float)
    else:
        data=image.flatten()
        reshaped_data=data.reshape(len(data),nb_components).astype(np.float)

    k=cluster.KMeans(n_clusters=nb_clusters,init="random",n_init=n_seedings)
    t0 = time.clock()
    k.fit(reshaped_data)
    t1 = time.clock()
    print("Cpu time: ",t1-t0)
    runtime=t1-t0
    labels=k.labels_

    #### Writing labels to appropriate pixels: Version 1
    result = np.zeros(image.shape[0:spatial_dim],dtype=np.uint8) #ne marche pas avec color -> vu comme 3D

    if type(image) == np.ma.masked_array :
        roi=np.logical_not(image.mask)
    else: roi=np.ones(image.shape,dtype=np.bool)
    if nb_components>1: #cas 2D RGB
        roi=roi[:,:,0]
    else:
        roi=roi

    result[roi] = labels
    result=np.ma.masked_array(result, mask=np.logical_not(roi))
    return result





