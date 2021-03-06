import re,itertools
import numpy as np
import scipy as sp; from scipy import ndimage
import networkx as nx
from skgtimage.core.topology import topological_graph_from_regions
from skgtimage.core.photometry import photometric_graph_from_regions

def __analyze_sentence__(g,desc) :
    operators=re.findall('<|>|=',desc)
    operands=re.split('<|>|=',desc)
    multioperands=[ re.split(',',o) for o in operands]
    for operands in multioperands:
        for o in operands : g.add_node(o)
    for i in range(0,len(operators)):
        operator=operators[i]
        left_operands=multioperands[i]
        right_operands=multioperands[i+1]
        if operator == '<':
            for l,r in itertools.product(left_operands,right_operands):
                g.add_edge(l,r)
        elif operator == '=':
            for l,r in itertools.product(left_operands,right_operands):
                g.add_edge(l,r)
                g.add_edge(r,l)
        elif operator == '>':
            for l,r in itertools.product(left_operands,right_operands):
                g.add_edge(r,l)

def from_string(desc,g=None):
    if g is None: g=nx.DiGraph()
    #Remove spaces
    nospace_desc=re.sub(' ','',desc)
    #Remove == -> =
    nospace_desc=re.sub('==','=',nospace_desc)
    #Split into sentences (separator is ';')
    descs=re.split(';',nospace_desc)
    #Analyze each sub-string
    for d in descs : __analyze_sentence__(g,d)
    #Return
    return g


def from_regions(image,regions):
    built_t_graph,new_residues=topological_graph_from_regions(regions)
    built_p_graph=photometric_graph_from_regions(image, new_residues)
    built_t_graph.set_image(image);built_p_graph.set_image(image)
    return built_t_graph,built_p_graph


def from_labelled_image(image, labelled_image, roi=None, manage_bounds=False, thickness=1,verbose=False):
    """
        Generate both inclusion and photometric graphs from the input image and its labelling (over-segmentation)

        :param image: input image
        :param labelled_image: input labelling (over-segmentation)
        :param roi: region of interest considered from computing graphs (regions lying outside are ignored)
        :param manage_bounds: if a thin enveloppe is added at image or roi boundaries
        :param thickness: internal boundary thickness to be considered for computing enveloppe label
        :param verbose: if True, details of the procedure are printed
        :return: built inclusion and photometric graphs

    """
    #To remove noise at labelled_image boundaries
    if manage_bounds:
        if type(labelled_image) == np.ma.masked_array :
            roi=np.logical_not(labelled_image.mask)
        new_labelled_image,new_roi=manage_boundaries(labelled_image, roi, thickness)
        return from_labelled_image(image, new_labelled_image, new_roi, False)
    #Regions (residues) from labels
    regions=labelled_image2regions(labelled_image,roi)
    #Built graphs from regions
    if verbose: print("Start building inclusion and photometric graphs from labelled image")
    result=from_regions(image,regions)
    if verbose: print("End building inclusion and photometric graphs from labelled image")
    return result

def manage_boundaries(image, roi=None, thickness=1):
    ############
    # FIND THE ROI INNER BOUNDARY DOMINANT VALUE
    if roi is None: roi=np.ones(image.shape)
    eroded_roi=sp.ndimage.morphology.binary_erosion(roi,iterations=thickness).astype(np.uint8)
    inner_boundary=roi/np.max(roi)-eroded_roi
    inner_boundary_values=np.ma.MaskedArray(image,mask=np.logical_not(inner_boundary)).compressed()
    bins=np.arange(np.min(inner_boundary_values),np.max(inner_boundary_values)+2)
    h,b=np.histogram(inner_boundary_values,bins)
    dominant_value=b[np.argmax(h)]
    ############
    # ENLARGE THE ROI
    enlarged_roi=sp.ndimage.morphology.binary_dilation(roi,iterations=1).astype(np.uint8)
    outer_boundary=enlarged_roi-roi/np.max(roi)
    modified_image=np.ma.MaskedArray(image,mask=outer_boundary).filled(dominant_value)
    if type(image)==np.ma.MaskedArray:
        modified_image=np.ma.MaskedArray(modified_image,mask=np.logical_not(enlarged_roi))
    return modified_image,enlarged_roi


def labelled_image2regions(labelled_image,roi=None):
    """
        Generate regions from labelled image: each region correspond to a specific label
    """
    #If explicit ROI (i.e. explicit as not integrated within an image of type np.ma.masked_array
    if roi is not None:
        tmp_masked_array=np.ma.masked_array(labelled_image, mask=np.logical_not(roi))
        return labelled_image2regions(tmp_masked_array)
    #Use histogram to find labels
    regions=[]
    if type(labelled_image) == np.ma.masked_array :
        mask_roi=np.logical_not(labelled_image.mask)
        min_image,max_image=labelled_image.compressed().min(),labelled_image.compressed().max()
        hist,bins = np.histogram(labelled_image.compressed(), bins=max_image-min_image+1,range=(min_image,max_image+1))
        bins=bins[0:bins.size-1]
        for i in range(0,len(hist)):
            if hist[i] != 0:
                new_region=np.where(labelled_image==bins[i],1,0)
                new_region=np.logical_and(mask_roi,new_region)
                regions+=[new_region]
    else:
        min_image,max_image=labelled_image.min(),labelled_image.max()
        hist,bins = np.histogram(labelled_image, bins=max_image-min_image+1,range=(min_image,max_image+1))
        bins=bins[0:bins.size-1]
        for i in range(0,len(hist)):
            if hist[i] != 0: regions+=[np.where(labelled_image==bins[i],1,0)]
    return regions
