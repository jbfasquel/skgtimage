import numpy as np
import sklearn
from sklearn.cluster import MeanShift #, estimate_bandwidth
import time
from skgtimage.utils import rgb2chsv


def mean_shift_rgb(image_rgb,bandwidth=0.3,spatial_dim=2,n_features=1,roi=None,verbose=True):
    image_chsv=rgb2chsv(image_rgb)
    l_image_chsv=np.ma.array(image_chsv, mask=np.logical_not(np.dstack(tuple([roi for i in range(0,3)]))))
    segmentation=mean_shift(l_image_chsv,bandwidth=bandwidth,spatial_dim=2,n_features=3,verbose=True) #0.1 OK
    return segmentation


def mean_shift(image,bandwidth=0.3,spatial_dim=2,n_features=1,verbose=True):
    """
    Mean shift

    """
    #print image.shape
    #print image.mask.shape
    #Preparing data
    if type(image) == np.ma.masked_array :
        #reshaped_data=image.compressed()
        reshaped_data=image.compressed()
        #reshaped_data=reshaped_data.reshape(len(reshaped_data),n_features)
        reshaped_data=reshaped_data.reshape(-1,n_features)
    else:
        reshaped_data=data.reshape(len(image),n_features)

    t0=time.clock()
    #bandwidth = estimate_bandwidth(reshaped_data, quantile=quantile, n_samples=500)
    #print bandwidth
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    #ms = MeanShift(bandwidth=bandwidth)
    ms.fit(reshaped_data)
    t1=time.clock()
    if verbose==True:
        print("Cpu time (sec): " , t1-t0)
        print("nb classes mean shift : " , len(np.unique(ms.labels_)))

    labels = ms.labels_
    #ms_result = np.zeros(roi_image.shape)
    #ms_result[roi] = labels
    n_clusters_ = len(np.unique(labels))
    #n_clusters_ = len(labels_unique)

    #### Writing labels to appropriate pixels: Version 1
    result = np.zeros(image.shape[0:spatial_dim],dtype=np.uint8) #ne marche pas avec color -> vu comme 3D

    roi=np.logical_not(image.mask)
    if (spatial_dim == 2) & (n_features>1): #cas 2D RGB
        roi=roi[:,:,0]
    elif (spatial_dim == 2) & (n_features==1): #cas 2D grayscale
        roi=roi

    #roi=np.dsplit(roi, 2)
    result[roi] = labels
    result=np.ma.masked_array(result, mask=np.logical_not(roi))

    return result
