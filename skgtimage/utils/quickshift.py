import numpy as np
import os,time
import skimage; from skimage import segmentation;from skimage.future import graph
import scipy as sp;from scipy import misc;from scipy import ndimage
from skgtimage.utils.evaluation import grey_levels
from skgtimage.utils.color import rgb2gray,rgb2chsv
from skgtimage.utils.rag_merging import rag_merge
import skgtimage as skg
import matplotlib.pyplot as plt


'''
def merge_photometry_color_slow(chsv_image,label,roi,times,verbose=False):
    """
    Merge regions of similar photometry, even if there are not adjacent
    :param chsv_image:
    :param label:
    :param roi:
    :param times:
    :param verbose:
    :return:
    """
    if verbose:
        print("Slow merge_photometry_color, remaining iterations:",times)
    ################
    #Precomputing properties
    levels=grey_levels(label,roi)
    nb_levels=len(levels)
    mean_for_each_level=[]
    size_for_each_level=[]
    for i in range(0,nb_levels):
        roi_i = np.where(label == levels[i], 1, 0)
        ri = region_stat(chsv_image, roi_i, np.mean, mc=True)
        mean_for_each_level+=[ri]
        size_for_each_level+=[np.sum(roi_i)]

    ################
    #Computing adjacency matrix
    adj_matrix=np.ones((nb_levels,nb_levels)) #1 assumes to be higher than any distance
    for i in range(0,nb_levels):
        for j in range(0, nb_levels):
            if i < j:
                #roi_i=np.where(label==levels[i],0,1)
                #ri=region_stat(chsv_image,roi_i,np.mean,mc=True)
                #roi_i=np.dstack((roi_i,roi_i,roi_i))
                #ri=np.mean(np.ma.array(chsv_image, mask=roi_i).compressed().reshape(-1,3))
                ri=mean_for_each_level[i]
                #roi_j=np.where(label==levels[j],0,1)
                #roi_j = np.dstack((roi_j, roi_j, roi_j))
                #rj = region_stat(chsv_image, roi_j, np.mean, mc=True)
                rj = mean_for_each_level[j]
                #rj=np.mean(np.ma.array(chsv_image, mask=roi_j).compressed().reshape(-1,3))
                dist=np.sqrt((rj[0]-ri[0])**2+(rj[1]-ri[1])**2+(rj[2]-ri[2])**2)
                adj_matrix[i,j]=dist
                adj_matrix[j,i]=dist
    ################
    #Search minimal distance
    mini=np.min(adj_matrix)
    min_is,min_js=np.where(adj_matrix==mini)
    min_i=min_is[0]
    min_j=min_js[0]
    merging=(levels[min_i],levels[min_j])

    #Label j takes the label i
    roi_to_change=np.where(label==merging[1],1,0)
    new_label=np.ma.array(label, mask=roi_to_change).filled(merging[0])
    #plt.imshow(new_label);plt.show()
    #Conditional recursive call
    remainging_times=times-1
    if remainging_times==0:
        return new_label
    else:
        return merge_photometry_color(chsv_image,new_label,roi,remainging_times,verbose)
'''



def quickshift(image,ratio,mc=False,roi=None,verbose=True):
    '''
    tmp_save_dir=save_dir
    if save_dir is not None:
        if not os.path.exists(save_dir): os.mkdir(save_dir)
        tmp_save_dir+="Quickshift/"
        if not os.path.exists(tmp_save_dir): os.mkdir(tmp_save_dir)
    '''
    input_image=image
    if mc is False:
        input_image=np.dstack((image,image,image))

    #t0=time.clock()
    label=skimage.segmentation.quickshift(input_image,ratio=ratio)
    if roi is not None:
        label+=1
        label=np.ma.array(label, mask=np.logical_not(roi)).filled(0)
    '''
    if tmp_save_dir is not None:
        nb = len(grey_levels(label))
        sp.misc.imsave(tmp_save_dir+"_0_QS_"+str(nb)+"_labels.png",label)
        tmp=skimage.segmentation.mark_boundaries(image, label)
        sp.misc.imsave(tmp_save_dir + "_0_QS_label_bounds.png", tmp)

    if cut_threshold != 0:
        label=rag_merge(image,label,cut_threshold,mc,roi)
        t1=time.clock()
        if save_dir is not None:
            nb=len(grey_levels(label))
            sp.misc.imsave(tmp_save_dir+"_1_QS_"+str(nb)+"_labels.png",label)
            tmp=skimage.segmentation.mark_boundaries(image, label)
            sp.misc.imsave(tmp_save_dir + "_1_QS_label_bounds.png", tmp)
    if nb_expected_labels !=0 :
        nb = len(grey_levels(label))
        times=nb-nb_expected_labels
        if mc is True:
            label=skg.utils.merge_photometry_color(image,label,roi,times,verbose)
        else:
            label= skg.core.merge_photometry_gray(image,label,times)
        if save_dir is not None:
            nb = len(grey_levels(label))
            sp.misc.imsave(tmp_save_dir + "_2_QS_" + str(nb) + "_labels.png", label)
            tmp = skimage.segmentation.mark_boundaries(image, label)
            sp.misc.imsave(tmp_save_dir + "_2_QS_label_bounds.png", tmp)
    '''
    return label