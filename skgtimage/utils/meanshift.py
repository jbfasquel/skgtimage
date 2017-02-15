import numpy as np
import sklearn
import skimage
from skgtimage.utils.color import rgb2chsv
from sklearn.cluster import MeanShift
import time

def meanshift(image, bandwidth, roi=None, mc=False, sigma=None, rgb_convert=False,verbose=False):
    """
    Apply meanshif to input image (within region of interest)

    :param image: input image
    :param bandwidth: bandwidth parameter considered in scikit-learn  MeanShift
    :param roi: region of interest
    :param mc: whether image is multi-component or not (color in our case)
    :param verbose:
    :param sigma: preliminary gaussian filtering (parameter of scikit-image filters.gaussian)
    :param rgb_convert: if True and mc True, RGB image is converted HSV space
    :return: labelled image (numpy array), where each label corresponds to a specific value
    """
    #Image preparation
    if mc: #color
        if sigma is not None:
            tmp=skimage.filters.gaussian(image, sigma=sigma, multichannel=True)
            return meanshift(tmp, bandwidth=bandwidth, roi=roi, mc=mc, verbose=verbose, sigma=None, rgb_convert=rgb_convert)
        #Conversion
        if rgb_convert:
            tmp=rgb2chsv(image)
            return meanshift(tmp, bandwidth=bandwidth, roi=roi, mc=mc, verbose=verbose, sigma=sigma, rgb_convert=False)


    nb_components,spatial_dim=1,len(image.shape)
    if mc:
        nb_components=image.shape[-1]
        spatial_dim-=1

    if roi is not None:
        roi_mask=np.dstack(tuple([roi for i in range(0,nb_components)]))
        roied_image=np.ma.masked_array(image,mask=np.logical_not(roi_mask))
        return meanshift(roied_image, bandwidth, None, mc, verbose)
    else:
        if type(image) != np.ma.masked_array :
            roi=np.ones(image.shape[0:spatial_dim])
            roi_mask=np.dstack(tuple([roi for i in range(0,nb_components)]))
            roied_image=np.ma.masked_array(image,mask=np.logical_not(roi_mask))
            return meanshift(roied_image, bandwidth, None, mc, verbose)


    if type(image) == np.ma.masked_array :
        reshaped_data=image.compressed().reshape(-1,nb_components)
    else:
        reshaped_data=image.reshape(-1,nb_components)

    t0=time.clock()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(reshaped_data)
    t1=time.clock()
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

