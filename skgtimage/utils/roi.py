import numpy as np


def bounding_box(roi):
    bounding_box_raw=np.where(roi != 0 )
    dim=len(bounding_box_raw)
    bb=[]
    for i in range(0,dim):
        bb+=[min(bounding_box_raw[i])]
        bb+=[max(bounding_box_raw[i])]

    return bb

def extract_subarray_rgb(array,roi=None,bb=None,margin=0):
    """
    array: 2D RGB, roi: 2D Scalar
    """
    local_bb=bb
    if local_bb is None :
        local_bb=bounding_box(roi)

    if len(local_bb) == 4 : #2D RGB
        local_bb_with_margin=[max(local_bb[0]-margin,0),min(local_bb[1]+margin,array.shape[0]-1),
                              max(local_bb[2]-margin,0),min(local_bb[3]+margin,array.shape[1]-1)]
        return array[local_bb_with_margin[0]:local_bb_with_margin[1]+1,
                     local_bb_with_margin[2]:local_bb_with_margin[3]+1,:] #,: for dim 2,3,4 (r,g,b)


def extract_subarray(array,roi=None,bb=None,margin=0):
    """

    """
    if (roi is not None) and (array.shape != roi.shape): raise Exception("Array and roi must be of the same size")

    local_bb=bb
    if local_bb is None :
        local_bb=bounding_box(roi)

    if len(local_bb) == 4 : #2D
        local_bb_with_margin=[max(local_bb[0]-margin,0),min(local_bb[1]+margin,array.shape[0]-1),
                              max(local_bb[2]-margin,0),min(local_bb[3]+margin,array.shape[1]-1)]
        return array[local_bb_with_margin[0]:local_bb_with_margin[1]+1,
                     local_bb_with_margin[2]:local_bb_with_margin[3]+1]
    elif len(local_bb) == 6 : #3D
        local_bb_with_margin=[max(local_bb[0]-margin,0),min(local_bb[1]+margin,array.shape[0]-1),
                              max(local_bb[2]-margin,0),min(local_bb[3]+margin,array.shape[1]-1),
                              max(local_bb[4]-margin,0),min(local_bb[5]+margin,array.shape[2]-1)]
        return array[local_bb_with_margin[0]:local_bb_with_margin[1]+1,
                     local_bb_with_margin[2]:local_bb_with_margin[3]+1,
                     local_bb_with_margin[4]:local_bb_with_margin[5]+1]
    else :
        return None
