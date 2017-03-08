# *-* coding: iso-8859-1 *-*
############################ BEGIN LICENSE BLOCK ############################
# Copyright (C)  Jean-Baptiste.Fasquel (LISA Laboratory, Angers University, France 2011)
############################ END LICENSE BLOCK ############################

import numpy as np
import csv,os
from skgtimage.core.graph import get_node2mean
from skgtimage.core.search_base import find_head
from skgtimage.core.topology import fill_region

def save_to_csv(dir,name,gcr,region2sim):
    nodepersim=[]
    related_sims=[]
    for n in sorted(region2sim):
        sim=np.round(region2sim[n],3)
        nodepersim+=[n]
        related_sims+=[sim]

    fullfilename=os.path.join(dir,name+".csv")
    csv_file=open(fullfilename, "w")
    c_writer = csv.writer(csv_file,dialect='excel')
    c_writer.writerow([name])
    c_writer.writerow(['good classification rate (%)']+[gcr])
    c_writer.writerow(['similarities']+[i for i in related_sims])
    if nodepersim is not None:
        c_writer.writerow(['related nodes']+[i for i in nodepersim])
    csv_file.close()

def goodclassification_rate_graphs(result_t_graph,truth_t_graph,roi=None,prec=None):
    res2int = get_node2mean(truth_t_graph,round=True)
    truth_image = truth_t_graph.get_labelled(mapping=res2int)
    result_image = result_t_graph.get_labelled(mapping=res2int)

    #####
    # GENERATING MASKED IMAGES

    #OLD uncorrect measurement: ROI is not the "filled" head.
    #Should be union of all truth regions, or the entire image (if recognition is not contrainted to the ROI
    #May be this involved uncorrect result in the paper
    '''
    head=list(find_head(truth_t_graph))[0]
    roi=fill_region(truth_t_graph.get_region(head))
    l_truth_image=np.ma.array(truth_image, mask=np.logical_not(roi))
    l_result_image=np.ma.array(result_image, mask=np.logical_not(roi))
    classif=goodclassification_rate(l_result_image,l_truth_image)
    '''
    #NEW
    if roi is not None:
        l_truth_image = np.ma.array(truth_image, mask=np.logical_not(roi))
        l_result_image = np.ma.array(result_image, mask=np.logical_not(roi))
        classif=goodclassification_rate(l_result_image,l_truth_image)
    else:
        classif = goodclassification_rate(result_image, truth_image)
    if prec is not None:
        classif = np.round(classif, prec)
    return classif,truth_image,result_image

def similarity_indices_graph_regions(result_graph, truth_graph, prec=None):
    region2sim={}
    for n in truth_graph.nodes():
        true_region=truth_graph.get_region(n)
        result_region=result_graph.get_region(n)
        sim=similarity_index(result_region,true_region)
        if prec is not None:
            sim=np.round(sim,prec)
        region2sim[n]=sim
    return region2sim


def goodclassification_rate(result,truth,prec=None):
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
    if prec is not None:
        classif=np.round(classif,prec)
    return classif

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

