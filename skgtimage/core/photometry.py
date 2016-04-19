import numpy as np
from skgtimage.core.graph import IrDiGraph
from skgtimage.core.topology import fill_region
#def region_stat(image,region,fct=np.mean,gray=True,component=None):
def region_stat(image,region,fct=np.mean,mc=False):
    """
    Compute a photometric statistics within a region of the image

    :param image: image from which the statistic is computed
    :param region: region within which statistics are computed
    :param fct: functor defining the statistics (e.g. numpy.mean or numpy.std)
    :param gray: True is grayscale image, False if color
    :param component: component in case of color image (0, 1 or 2)
    :return: statistic (value)
    """
    if mc == False:
        data=np.ma.masked_array(image,mask=np.logical_not(region)).compressed()
        result=fct(data)
        return result
    else: #color with 3 components
        nb_components=image.shape[-1]
        roi_mask=np.dstack(tuple([region for i in range(0,nb_components)]))
        #data=np.ma.masked_array(image,mask=np.logical_not(np.dstack((region,region,region)))).compressed().reshape(-1,3)
        data=np.ma.masked_array(image,mask=np.logical_not(roi_mask)).compressed().reshape(-1,nb_components)
        #if component
        #return fct(data,axis=0)[component]
        return fct(data,axis=0)

def sort_regions_by_stat(image,regions,fct=np.mean,mc=False,component=0,return_stats=False):
    """
    Sort regions in decreasing order of their photometric statistics
    Todo: sort_labels_by_stat

    :param image: image from which statistics are computed
    :param regions: list of regions within which statistics are computed
    :param fct: functor defining the statistics (e.g. numpy.mean or numpy.std)
    :param gray: True is grayscale image, False if color
    :param component: component in case of color image (0, 1 or 2)
    :param return_stats: True for returning ordered statistics
    :return: ordered list of regions (and, optionally the ordered list of related statistics)
    """
    #List of stat
    stat2region={}
    if mc==False :
        for r in regions:
            result=region_stat(image,r,fct,mc)
            stat2region[result]=r
    else:
        for r in regions:
            #result=region_stat(image,r,fct,gray,component)
            result=region_stat(image,r,fct,mc)[component]
            stat2region[result]=r
    #Ordering
    ordered_stats=sorted(stat2region.keys(),reverse=True)
    ordered_regions=[ stat2region[i] for i in ordered_stats ]
    if return_stats:
        return ordered_regions,ordered_stats
    else:
        return ordered_regions

def sort_region_indices_by_stat(image,regions,fct=np.mean,mc=False,component=0,return_stats=False):
    """
    Sort regions in decreasing order of their photometric statistics and return ordered index list (index within the input list of regions
    Todo: sort_labels_by_stat

    :param image: image from which statistics are computed
    :param regions: list of regions within which statistics are computed
    :param fct: functor defining the statistics (e.g. numpy.mean or numpy.std)
    :param gray: True is grayscale image, False if color
    :param component: component in case of color image (0, 1 or 2)
    :param return_stats: True for returning ordered statistics
    :return: ordered list of indices (and, optionally the ordered list of related statistics)
    """
    #List of stat: key<->stat and valeur<->list of indices (although unlikely, several residues may share the same stat)
    stat2index={}
    if mc == False :
        for i in range(0,len(regions)):
            result=region_stat(image,regions[i],fct,mc)
            #stat2index[result]=i
            if result in stat2index: stat2index[result]+=[i]
            else: stat2index[result]=[i]

    else:
        for i in range(0,len(regions)):
            result=region_stat(image,regions[i],fct,mc)[component]
            if result in stat2index: stat2index[result]+=[i]
            else: stat2index[result]=[i]
            #stat2index[result]=i
    #Ordering
    ordered_stats_keys=sorted(stat2index.keys(),reverse=True)
    ordered_indices=[]
    ordered_stats=[]
    for i in ordered_stats_keys:
        ordered_stats+=len(stat2index[i])*[i]
        ordered_indices+=stat2index[i]
    #ordered_indices=[ stat2index[i] for i in ordered_stats ]
    if return_stats:
        return ordered_indices,ordered_stats
    else:
        return ordered_indices


def sort_labels_by_stat(image,labelled_image,fct=np.mean,mc=False,component=0,return_stats=False):
    regions=[np.where(labelled_image==i,1,0) for i in range(1,np.max(labelled_image)+1)]
    if return_stats:
        ordered_indices,ordered_stats=sort_region_indices_by_stat(image,regions,fct=fct,mc=mc,component=component,return_stats=return_stats)
        ordered_indices=np.array(ordered_indices)+1
        return ordered_indices,ordered_stats
    else:
        ordered_indices=sort_region_indices_by_stat(image,regions,fct=fct,mc=mc,component=component,return_stats=return_stats)
        ordered_indices=np.array(ordered_indices)+1
        return ordered_indices


def photometric_graph_from_residues(image,residues):
    """
    return photometric graph where similar nodes (i.e. brothers) correspond to smallest mean intensity differences
    the number of similarity differences are

    :param image:
    :param residues:
    :param brother_links: number of brother links to consider
    :return:
    """
    #################################
    #Nodes: one node per residue
    #################################
    g=IrDiGraph(image=image)
    for i in range(0,len(residues)):
        g.add_node(i)
        filled_r=fill_region(residues[i])
        g.set_region(i,filled_r)

    #################################
    #Edges: according to photometry
    #################################
    #ordered_indices,stats=sort_region_indices_by_stat(image,residues,fct=np.mean,gray=True,return_stats=True)
    ordered_indices,stats=sort_region_indices_by_stat(image,residues,fct=np.mean,mc=False,return_stats=True)

    for i in range(0,len(ordered_indices)):
        node=ordered_indices[i]
        value=stats[i]
        g.set_mean_residue_intensity(node,value)

    increasing_ordered=ordered_indices[::-1]
    increasing_stats=stats[::-1]
    for i in range(0,len(increasing_ordered)-1):
        g.add_edge(increasing_ordered[i],increasing_ordered[i+1])



    #################################
    #Return the final graph
    return g

def build_similarities(image,residues,g,nb_brothers):
    ordered_indices,stats=sort_region_indices_by_stat(image,residues,fct=np.mean,mc=False,return_stats=True)
    increasing_ordered=ordered_indices[::-1]
    increasing_stats=stats[::-1]
    #################################
    #Brother management : if we a priori know that similarities are expected,
    #the built graph must depict the same number of similarities to facilitate further exact graph matching.
    #################################
    diff=[ np.abs(increasing_stats[i+1]-increasing_stats[i]) for i in range(0,len(increasing_stats)-1)]
    #print(diff)
    new_edges=[]
    replacement_value=np.max(diff)
    for b in range(0,nb_brothers):
        min_diff=np.argmin(diff)
        diff[min_diff]=replacement_value
        new_edge=(increasing_ordered[min_diff+1],increasing_ordered[min_diff])
        new_edges+=[new_edge]
    #Adding reverse edges
    for n in new_edges : g.add_edge(n[0],n[1])

