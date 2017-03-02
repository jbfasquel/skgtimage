import os
import numpy as np
import scipy as sp;from scipy import misc
import skimage; from skimage import segmentation
from skgtimage.core.photometry import grey_levels


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


def save_image_context(image,label,context_dir,roi=None,slices=[],mc=False):
    """
    save image+superimposed labels, roied_label,roied_image+superimposed labels
    :param image:
    :param label:
    :param directory:
    :param roi:
    :param slices:
    :param mc:
    :return:
    """
    if not os.path.exists(context_dir): os.mkdir(context_dir)
    nb = len(grey_levels(label, roi))
    #Case 2D grayscale
    if (len(image.shape) == 2) and (mc is False):
        tmp_image=image
        #if roi is not None: tmp_image = np.ma.array(image.astype(np.float), mask=np.logical_not(roi)).filled(np.min(image) - 1)
        if roi is not None: tmp_image = np.ma.array(image, mask=np.logical_not(roi)).filled(0)
        save_image2d_boundaries(tmp_image, label, directory=context_dir, filename="image_and_"+str(nb)+"_labels")
        __save_image2d__(tmp_image,os.path.join(context_dir,"image.png"))
        __save_image2d__(label,os.path.join(context_dir,"label.png"))
        #Crop
        if roi is not None:
            tmp_image_crop = extract_subarray(tmp_image, roi=roi)
            label_crop=extract_subarray(label, roi=roi)
            save_image2d_boundaries(tmp_image_crop, label_crop, directory=context_dir, filename="image_and_"+str(nb)+"_label_crop")
            __save_image2d__(tmp_image_crop, os.path.join(context_dir, "image_crop.png"))
            __save_image2d__(label_crop, os.path.join(context_dir, "label_crop.png"))

    #Case 2D color
    elif mc is True:
        tmp_image = image
        if roi is not None:
            tmp_roi=np.dstack(tuple([roi for i in range(0,3)]))
            tmp_image = np.ma.array(tmp_image, mask=np.logical_not(tmp_roi)).filled(0)
            __save_image2d__(tmp_image, os.path.join(context_dir, "image_roied.png"))
        save_image2d_boundaries(tmp_image, label, directory=context_dir, filename="image_and_"+str(nb)+"_label")
        if roi is not None:
            tmp_image_crop = extract_subarray_rgb(tmp_image, roi=roi)
            label_crop = extract_subarray(label, roi=roi)
            save_image2d_boundaries(tmp_image_crop, label_crop, directory=context_dir, filename="image_and_"+str(nb)+"_label_crop")
            __save_image2d__(tmp_image_crop, os.path.join(context_dir, "image_crop.png"))
            __save_image2d__(label_crop, os.path.join(context_dir, "label_crop.png"))

    #Case 3D grayscale (with slices)
    elif (len(image.shape) == 3) and (mc is False):
        __save_image3d__(image,context_dir+"image/",slices,True)
        if roi is not None:
            l_image=np.ma.array(image.astype(np.float), mask=np.logical_not(roi))
        __save_image3d__(l_image,context_dir+"image_roi/",slices,True)
        save_image3d_boundaries(l_image, label, directory=context_dir+"image_"+str(nb)+"_label/", slices=slices)


def save_image2d_boundaries(image,labelled,directory=None,filename="img_bound"):
    if not os.path.exists(directory): os.mkdir(directory)

    nb = len(grey_levels(labelled))
    if np.max(labelled) != np.min(labelled):
        tmp_labelled=(labelled.astype(np.float)-np.min(labelled))*(255.0)/(np.max(labelled)-np.min(labelled)).astype(np.uint8)
    else:
        tmp_labelled=labelled.astype(np.uint8)
    sp.misc.imsave(directory + filename + "_" + str(nb) + "_labels.png", tmp_labelled)
    if len(image.shape) == 2:
        tmp = np.dstack(tuple([image for i in range(0,3)]))
        tmp = skimage.segmentation.mark_boundaries(tmp, labelled)
    else:
        tmp = skimage.segmentation.mark_boundaries(image, labelled)
    sp.misc.imsave(directory + filename + "_" + str(nb) + "_labels_bounds.png", tmp)

def save_image3d_boundaries(image,labelled,directory=None,slices=[]):
    #Directory
    if not os.path.exists(directory) : os.mkdir(directory)

    tmp_image=image
    #Save
    for s in slices:
        current_labelled=labelled[:,:,s]
        current_slice=tmp_image[:,:,s]
        mini, maxi = np.min(current_slice), np.max(current_slice)
        if mini != maxi:
            current_slice = (current_slice.astype(np.float) - mini) * (255.0) / (maxi - mini)
        current_slice = current_slice.astype(np.uint8)
        if type(current_slice) == np.ma.MaskedArray:
            current_slice=current_slice.filled(0)
        if type(current_labelled) == np.ma.MaskedArray:
            current_labelled=current_labelled.filled(0)

        current_slice=np.rot90(current_slice)
        current_labelled = np.rot90(current_labelled)
        #filename=os.path.join(directory,"slice_"+str(s)+".png");
        save_image2d_boundaries(current_slice,current_labelled,directory,"slice_"+str(s))

def __save_image2d__(image,filename,do_rescale=True):
    mini,maxi=np.min(image),np.max(image)
    if (maxi-mini != 0) and do_rescale:
        tmp_image=(image.astype(np.float)-mini)*(255.0)/(maxi-mini)
        sp.misc.imsave(filename, tmp_image.astype(np.uint8))
    else:
        sp.misc.imsave(filename, image.astype(np.uint8))

def __save_image3d__(image,directory,slices=[],do_rescale=True):
    #Directory
    if not os.path.exists(directory) : os.mkdir(directory)
    #Rescale
    tmp_image=image
    #Save
    for s in slices:
        current_slice=tmp_image[:,:,s]
        mini, maxi = np.min(current_slice), np.max(current_slice)
        if mini != maxi:
            current_slice = (current_slice.astype(np.float) - mini) * (255.0) / (maxi - mini)
        else:
            current_slice = current_slice.astype(np.uint8)
        if type(current_slice) == np.ma.MaskedArray:
            current_slice=current_slice.filled(0)
        current_slice=np.rot90(current_slice)
        filename=os.path.join(directory,"slice_"+str(s)+".png");
        __save_image2d__(current_slice,filename,False)
