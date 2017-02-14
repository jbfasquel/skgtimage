import numpy as np
import os,time
import skimage; from skimage import segmentation;from skimage.future import graph
import scipy as sp;from scipy import misc;from scipy import ndimage
from skgtimage.utils.evaluation import grey_levels
from skgtimage.utils.color import rgb2gray,rgb2chsv
from skgtimage.utils.rag_merging import rag_merge
import skgtimage as skg
import matplotlib.pyplot as plt


def quickshift(image,ratio,mc=False,roi=None,verbose=True):
    """
    Apply quickshift segmentation to input image (within region of interest)

    :param image: input image
    :param ratio: ratio parameter considered in scikit-image quickshift
    :param mc: whether image is multi-component or not (color in our case)
    :param verbose:
    :return: labelled image (numpy array), where each label corresponds to a specific value
    """
    input_image=image
    if mc is False:
        input_image=np.dstack((image,image,image))

    #t0=time.clock()
    label=skimage.segmentation.quickshift(input_image,ratio=ratio)
    if roi is not None:
        label+=1
        label=np.ma.array(label, mask=np.logical_not(roi)).filled(0)

    return label