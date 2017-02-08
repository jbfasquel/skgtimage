import numpy as np
import sklearn
import skimage
from skgtimage.utils.color import rgb2chsv
from sklearn.cluster import MeanShift #, estimate_bandwidth
import time

def mean_shift(image,bandwidth,roi=None,mc=False,verbose=True,sigma=None,rgb_convert=False):
    """
    Mean shift
    """
    #Image preparation
    if mc: #color
        if sigma is not None:
            tmp=skimage.filters.gaussian(image, sigma=sigma, multichannel=True)
            return mean_shift(tmp,bandwidth=bandwidth,roi=roi,mc=mc,verbose=verbose,sigma=None,rgb_convert=rgb_convert)
        #Conversion
        if rgb_convert:
            tmp=rgb2chsv(image)
            return mean_shift(tmp, bandwidth=bandwidth, roi=roi, mc=mc, verbose=verbose, sigma=sigma,rgb_convert=False)


    nb_components,spatial_dim=1,len(image.shape)
    if mc:
        nb_components=image.shape[-1]
        spatial_dim-=1

    if roi is not None:
        roi_mask=np.dstack(tuple([roi for i in range(0,nb_components)]))
        roied_image=np.ma.masked_array(image,mask=np.logical_not(roi_mask))
        return mean_shift(roied_image,bandwidth,None,mc,verbose)
    else:
        if type(image) != np.ma.masked_array :
            roi=np.ones(image.shape[0:spatial_dim])
            roi_mask=np.dstack(tuple([roi for i in range(0,nb_components)]))
            roied_image=np.ma.masked_array(image,mask=np.logical_not(roi_mask))
            return mean_shift(roied_image,bandwidth,None,mc,verbose)


    if type(image) == np.ma.masked_array :
        reshaped_data=image.compressed().reshape(-1,nb_components)
    else:
        reshaped_data=image.reshape(-1,nb_components)

    t0=time.clock()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(reshaped_data)
    t1=time.clock()
    runtime = t1 - t0
    if verbose==True:
        print("Cpu time (sec): " , t1-t0)
        print("nb clusters : " , len(np.unique(ms.labels_)))

    labels = ms.labels_
    result = np.zeros(image.shape[0:spatial_dim],dtype=np.uint8) #ne marche pas avec color -> vu comme 3D

    roi=np.logical_not(image.mask)
    if nb_components > 1:
        roi=roi[:,:,0]
    else:
        roi=roi

    result[roi] = labels
    result=np.ma.masked_array(result, mask=np.logical_not(roi))
    return result

