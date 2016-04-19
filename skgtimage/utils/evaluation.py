# *-* coding: iso-8859-1 *-*
############################ BEGIN LICENSE BLOCK ############################
# Copyright (C)  Jean-Baptiste.Fasquel (LISA Laboratory, Angers University, France 2011)
############################ END LICENSE BLOCK ############################

import numpy as np
from skgtimage.utils.histogram import int_histogram
from skgtimage.core.search_base import find_head
import csv,os

def save_to_csv(dir,name,gcr,similarities,nodepersim=None):
    fullfilename=os.path.join(dir,name+".csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow([name])
    c_writer.writerow(['good classification rate (%)']+[gcr])
    c_writer.writerow(['similarities']+[i for i in similarities])
    if nodepersim is not None:
        c_writer.writerow(['related nodes']+[i for i in nodepersim])
    csv_file.close()

def compute_sim_between_graph_regions(result_graph,truth_graph):
    region2sim={}
    for n in truth_graph.nodes():
        true_region=truth_graph.get_region(n)
        result_region=result_graph.get_region(n)
        sim=similarity_index(result_region,true_region)
        region2sim[n]=sim
    return region2sim

def combine(graph,nodes,labels,shape=None):
    if shape is not None:
        result=np.zeros(shape)
    else:
        result=np.zeros(graph.get_image().shape)
    for i in range(0,len(nodes)):
        result=np.ma.masked_array(result, mask=graph.get_region(nodes[i])).filled(labels[i])
    return result


class Regions:
    def __init__(self):
        self.id2region={}
    def add(self,id,region):
        self.id2region[id]=region
    def combine(self,regions,labels):
        """
        Take care of topological inclusion: start by regions containing other regions !!!
        :param regions:
        :param labels:
        :return:
        """
        base=list(self.id2region.values())[0]
        truth=np.zeros(base.shape)
        truth=np.ma.masked_array(truth, mask=base).filled(labels[0]) # FIRST LEVEL COMES FROM AUTOMATED SEGMENTATION
        for i in range(1,len(regions)):
            truth=np.ma.masked_array(truth, mask=self.id2region[regions[i]]).filled(labels[i])
        return truth
    def downsample(self,factor):
        for id in self.id2region.keys():
            region=self.id2region[id]
            downsampled_region=r=sp.ndimage.interpolation.zoom(region,1/float(factor),order=0).astype(np.bool)
            self.id2region[id]=downsampled_region


def grey_levels(image,roi=None):
    #occurences,values=int_histogram(image,roi=roi)
    occurences,values=int_histogram(image,roi)
    non_zeros_indices=np.where(occurences != 0 )[0]
    grey_levels=values[non_zeros_indices].astype(np.int)
    return grey_levels

def remap_greylevels(image,dest_levels,src_levels=None):
    """
    Use rois for each level
    """
    if src_levels is None : src_levels=grey_levels(image)
    remapped_result=np.zeros(image.shape)
    for level_index in range(0,len(src_levels)):
        roi= image == src_levels[level_index]
        remapped_result=np.ma.MaskedArray(remapped_result,mask=roi).filled(dest_levels[level_index])
    if type(image) == np.ma.masked_array :
         remapped_result=np.ma.MaskedArray(remapped_result,mask=image.mask)
    return remapped_result

def goodclassification_rate(result,truth):
    """
        Measure to good classification rate between classification result and expect result (truth), assuming
        that greylevels correspond.
        Return: classification rate
        Note: greylevels are the labels to which result must be mapped in order to correspond to truth labels
        sc_result=skit.remap_greylevels2(sc_result,levels)
    """
    #initial_shape=result.shape
    #mask=None
    ########################
    #Adapt vs masked array (ROI) -> must be compressed
    #To ignore pixels outside the region of interest
    ########################
    if type(result) == np.ma.masked_array :
        result=result.compressed()
    if type(truth) == np.ma.masked_array :
        truth=truth.compressed()
    result=result.flatten()
    truth=truth.flatten()
    nb_points=len(truth)

    number_of_errors=np.count_nonzero(result-truth)
    classif=1.0-number_of_errors/nb_points
    return classif


def misclassification_rate_unknown_label_ordering(result,truth):
    """
        Measure to good classification rate between classification result and expect result (truth), assuming
        that the correspondance between list of labels are assumed to be unknown: all permutations are then tested, and
        the one leading to the lowest misclassification is considered as the result.
        Return: classification rate and greylevels
        Note: greylevels are the labels to which result must be mapped in order to correspond to truth labels
        sc_result=skit.remap_greylevels2(sc_result,levels)
    """
    #initial_shape=result.shape
    #mask=None
    ########################
    #Adapt vs masked array (ROI) -> must be compressed
    #To ignore pixels outside the region of interest
    ########################
    if type(result) == np.ma.masked_array :
        #mask=result.mask
        result=result.compressed()

    if type(truth) == np.ma.masked_array :
        truth=truth.compressed()

    result=result.flatten()
    truth=truth.flatten()
    nb_points=len(truth)
    ########################
    #REMAP TO TRUTH_LEVELS
    ########################
    truth_levels=grey_levels(truth)
    #print "truth_levels" , truth_levels
    result_levels=grey_levels(result)
    #print "result_levels" , result_levels
    remapped_result=remap_greylevels(result, truth_levels,result_levels)
    import itertools
    #print itertools.permutations(truth_levels)
    best_result_levels=None
    lowest_misclassification_rate=1.0
    for p in itertools.permutations(truth_levels):
        #print "result: " , skit.grey_levels(result)
        tmp_result=remap_greylevels(result, p,result_levels)
        #print "tmp result: " , skit.grey_levels(tmp_result)
        number_of_errors=np.count_nonzero(tmp_result-truth)
        misclassification_rate=(number_of_errors)/float(nb_points)
        #print p ," -> " , misclassification_rate
        if misclassification_rate < lowest_misclassification_rate :
            best_result_levels=p
            lowest_misclassification_rate=misclassification_rate
    return lowest_misclassification_rate,best_result_levels

def goodclassification_rate_unknown_label_ordering(r,t):
    misclassif,levels=misclassification_rate_unknown_label_ordering(r,t)
    return 1.0 -misclassif, levels


def similarity_indices(result,truth):
    """
    similarity per level (i.e. per class)
    """
    similarities=[]
    truth_levels=grey_levels(truth)
    result_levels=grey_levels(result)
    #if truth_levels.list() != result_levels.all(): raise Exception("both grey_level distributions must be identical")
    if not np.array_equal(truth_levels,result_levels): raise Exception("both grey_level distributions must be identical (use skit.remap_greylevels(sc_result,levels)) ")

    for level in result_levels:
        #print level
        #print result==level
        similarities+=[similarity_index(result==level, truth==level)]
    return similarities
'''
def find_gray_level_mapping(result,truth,roi=None):
    """
    return two lists: result_levels, and truth_levels so that both level lists corresponds.
    Correspondance is detected ccording to similarity indices of related binary regions)
    """
    truth_levels=skit.grey_levels(truth,roi)
    truth_levels_sizes=[]
    truth_levels_2_sizes={}
    for t in truth_levels:
        binary= truth == t
        #truth_levels_sizes+=[np.sum(binary)/np.max(binary)]
        truth_levels_2_sizes[t]=np.sum(binary)/np.max(binary)
    from collections import OrderedDict
    d_sorted_by_value = OrderedDict(sorted(truth_levels_2_sizes.items(), key=lambda x: x[1]))

    result_levels=skit.grey_levels(result,roi)
    map_truth_levels_2_result_levels={}
    #for t_l in truth_levels:
    for t_l in d_sorted_by_value:
        #print "current t_l : " , t_l
        corresponding_result_level=result_levels[0]
        sim=0
        for r_l in result_levels:
            if r_l not in map_truth_levels_2_result_levels.values(): #pour eviter d'avoir deux niveaux identiques
                #tmp_sim=similarity_index(result == r_l ,truth == t_l,roi)
                tmp_sim=tp(result == r_l ,truth == t_l)
                print tmp_sim , " between " , t_l , " and " , r_l
                if tmp_sim > sim:
                    sim=tmp_sim
                    corresponding_result_level=r_l
        map_truth_levels_2_result_levels[t_l]=corresponding_result_level
    result_levels=[]
    truth_levels=[]
    for i in sorted(map_truth_levels_2_result_levels):
        truth_levels+=[i]
        result_levels+=[map_truth_levels_2_result_levels[i]]
    return result_levels,truth_levels
    #print map_truth_levels_2_result_levels
'''
def tp(result,truth):
    """ True positive """
    return np.float32(np.sum(np.logical_and(result, truth)))

def tn(result,truth):
    """ True negative """
    return np.float32(np.sum(np.logical_not(np.logical_or(result, truth))))

def fp(result,truth):
    """ False positive """
    return np.float32(np.sum(np.logical_and(np.logical_not(truth), result)))

def fn(result,truth):
    """ False negative """
    return np.float32(np.sum(np.logical_and(truth,np.logical_not(result))))

def sensibility(result,truth):
    """ Sensibility : 1.0 (perfect) truth is included into result """
    if np.max(result)==0 : return 0
    tp_val=tp(result,truth)
    fn_val=fn(result,truth)
    return tp_val/(tp_val+fn_val)

def specificity(result,truth):
    """ Specificity : 1.0 (perfect) result is included into true """
    if np.max(result)==0 : return 0
    tn_val=tn(result,truth)
    fp_val=fp(result,truth)
    return tn_val/(tn_val+fp_val)


def similarity_index(result,truth,roi=None):
    """
        If roi is not none, result and truth reduced to roi are used for computations (i.e. compressed() masked array)

    """
    if type(result) == np.ma.masked_array :
        similarity_index(result.compressed(),truth.compressed())

    if roi is not None:
        result_tmp=np.ma.MaskedArray(result,mask=np.logical_not(roi)).compressed()
        truth_tmp=np.ma.MaskedArray(truth,mask=np.logical_not(roi)).compressed()
        similarity_index(result_tmp,truth_tmp)
    if (np.min(result) !=0) or (np.min(truth) !=0):
        raise Exception("Minimum must be 0")
    #print np.min(result), np.max(result)
    #print np.min(truth), np.max(truth)
    max_result=np.max(result)
    max_truth=np.max(truth)
    #intersection=np.logical_and(truth,result)
    #TODO:np.asarray(truth,dtype=np.uint16) if overflow
    if (max_result==0) or (max_truth==0):
        intersection=np.zeros(truth.shape,dtype=np.float)
        return 0
    else:
        #intersection=(truth.astype(np.float)*result.astype(np.float))/np.float(max_result*max_truth) #faster than logical and
        intersection=np.logical_and(truth,result)
        sim_index=2.0*np.sum(intersection)/(np.sum(truth)/np.float(max_truth)+np.sum(result)/np.float(max_result))
        return sim_index

if __name__ == '__main__':
    '''
    truth = np.array([      [0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,0,0,0,0,0]],np.float)
    roi = np.array([      [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [1,1,1,1,1,1],
                            [1,1,1,1,1,1],
                            [1,1,1,1,1,1]],np.float)
    '''
    '''
    result = np.array([     [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,1,1,0,0],
                            [0,0,1,1,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0]],np.float)
    print similarity_index(result,truth)
    result = np.array([     [0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,0,0,0,0,0]],np.float)
    print similarity_index(result,truth)

    result = np.array([     [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0]],np.float)
    print similarity_index(result,truth)
    print similarity_index(result,truth,roi)
    t1=np.ma.MaskedArray(result,mask=np.logical_not(roi))
    t2=np.ma.MaskedArray(truth,mask=np.logical_not(roi))
    print similarity_index(t1,t2)

    result = 255*np.array([ [0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,0,0,0,0,0]],np.float)
    print similarity_index(result,truth)
    print similarity_index(result,truth,roi)
    t1=np.ma.MaskedArray(result,mask=np.logical_not(roi))
    t2=np.ma.MaskedArray(truth,mask=np.logical_not(roi))
    print similarity_index(t1,t2)
    '''
    '''
    roi = np.array([      [0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,0,0,0,0,0]],np.float)

    truth = np.array([ [0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,2,2,0],
                            [0,1,1,2,2,0],
                            [0,3,3,3,1,0],
                            [0,0,0,0,0,0]],np.float)

    result = np.array([[0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,5,3,0],
                            [0,1,1,3,3,0],
                            [0,3,5,3,3,0],
                            [0,0,0,0,0,0]],np.float)
    print similarity_index(result,truth)
    print truth == 2.0
    print find_gray_level_mapping(result,truth,roi)
    '''
    '''
    result = np.array([     [-10,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0],
                            [0,1,1,1,1,0]],np.float)
    print similarity_index(result,truth)
    '''

    truth = np.array([ [0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,2,2,0],
                            [0,1,1,2,2,0],
                            [0,3,3,3,1,0],
                            [0,0,0,0,0,0]],np.float)

    result = np.array([[0,0,0,0,0,0],
                            [0,1,1,1,1,0],
                            [0,1,1,5,3,0],
                            [0,1,1,3,3,0],
                            [0,3,5,3,3,0],
                            [0,0,0,0,0,0]],np.float)
    roi = np.array([      [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [0,0,0,0,0,0],
                            [1,1,1,1,1,1],
                            [1,1,1,1,1,1],
                            [1,1,1,1,1,1]],np.float)
    truth=np.ma.MaskedArray(truth,mask=np.logical_not(roi))
    result=np.ma.MaskedArray(result,mask=np.logical_not(roi))

    print("truth : \n" ,truth)
    print("result : \n" ,result)
    #print misclassification_rate2(result,truth)
    classif,levels=goodclassification_rate_unknown_label_ordering(result,truth)
    result_2=remap_greylevels(result,levels)
    print(similarity_indices(result_2, truth))
    print(levels)
    print(result_2)

