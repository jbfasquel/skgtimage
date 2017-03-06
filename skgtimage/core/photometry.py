import numpy as np
from skgtimage.core.graph import IrDiGraph

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
        data=np.ma.masked_array(image,mask=np.logical_not(roi_mask)).compressed().reshape(-1,nb_components)
        return fct(data,axis=0)

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

def photometric_graph_from_regions(image, regions):
    """
    return photometric graph where similar nodes (i.e. brothers) correspond to smallest mean intensity differences
    the number of similarity differences are

    :param image:
    :param regions:
    :param brother_links: number of brother links to consider
    :return:
    """
    #################################
    #Nodes: one node per residue
    #################################
    g=IrDiGraph(image=image)
    for i in range(0, len(regions)):
        g.add_node(i)
        g.set_region(i, regions[i])
    #################################
    #Edges: according to photometry
    #################################
    update_photometric_graph(g)
    #################################
    #Return the final graph
    return g

def merge_nodes_photometry(graph,source,target):
    new_region=np.logical_or(graph.get_region(source),graph.get_region(target))
    graph.remove_node(source)
    graph.set_region(target,new_region)
    update_photometric_graph(graph)

def update_photometric_graph(graph):
    """
    :param graph: assume to model photometric relationships
    :return:
    """
    ############
    #First update all region mean intensities
    for n in graph.nodes():
        region=graph.get_region(n)
        intensity=region_stat(graph.get_image(),region)
        graph.set_mean_intensity(n, intensity)
    ############
    #Second: update edges
    intensity2node={}
    for n in graph.nodes():
        intensity2node[graph.get_mean_intensity(n)]=n
    increas_ordered_intensities=sorted(intensity2node)
    graph.remove_edges_from(graph.edges()) #remove all existing edges first
    #Add new edges
    for i in range(0,len(increas_ordered_intensities)-1):
        a=intensity2node[increas_ordered_intensities[i]]
        b=intensity2node[increas_ordered_intensities[i+1]]
        graph.add_edge(a,b)



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

def grey_levels(image,roi=None):
    occurences,values=int_histogram(image,roi)
    non_zeros_indices=np.where(occurences != 0 )[0]
    grey_levels=values[non_zeros_indices].astype(np.int)
    return grey_levels
