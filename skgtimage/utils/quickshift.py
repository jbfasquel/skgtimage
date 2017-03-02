import numpy as np
import time
import skimage; from skimage import segmentation


def quickshift(image,ratio,mc=False,roi=None,verbose=False):
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

    t0=time.clock()
    label=skimage.segmentation.quickshift(input_image,ratio=ratio)
    t1 = time.clock()
    if verbose:
        print("Cpu time (sec): " , t1-t0)

    if roi is not None:
        label+=1
        label=np.ma.array(label, mask=np.logical_not(roi)).filled(0)

    return label