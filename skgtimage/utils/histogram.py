import numpy as np

def int_histogram(image,roi=None):
    """
        Compute the histogram of integer arrays. Considering integers vs e.g. floats avoid
        to manage bin width (set to 1 for integers). If not None, ROI contain non null value
        where image point have to be processed. image can integrate the ROI (roi must be None is this case):
        image=np.ma.masked_array(image, mask=np.logical_not(roi))
        Return the histogram and associated values (abscissa)
    """
    #If explicit ROI (i.e. explicit as not integrated within an image of type np.ma.masked_array
    if roi is not None:
        tmp_masked_array=np.ma.masked_array(image, mask=np.logical_not(roi))
        return int_histogram(tmp_masked_array)
    #Needed because np.histogram() does not restrict computations within mask in case of np.ma.masked_array
    if type(image) == np.ma.masked_array :
        return int_histogram(image.compressed()) #compressed: return unmasked values in a 1D array
    min_image,max_image=image.min(),image.max()
    h,x = np.histogram(image, bins=max_image-min_image+1,range=(min_image,max_image+1))
    return h,x[0:x.size-1]

def float_histogram(image,nb_bins=10):
    """
        Compute the histogram of float arrays, according to the number of bins.
        Return the histogram and associated values (abscissa)
    """
    if type(image) == np.ma.masked_array :
        return float_histogram(image.compressed(),nb_bins)
    h,x = np.histogram(image, bins=nb_bins)
    return h,x[0:x.size-1]
