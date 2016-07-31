import re,itertools
import numpy as np
import scipy as sp; from scipy import ndimage
import networkx as nx
from skgtimage.core.graph import IrDiGraph,transitive_reduction,labelled_image2regions
from skgtimage.core.topology import topological_graph_from_residues_refactorying
from skgtimage.core.photometry import photometric_graph_from_residues,photometric_graph_from_residues_refactorying


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

def from_labelled_image(image,labelled_image,roi=None,manage_bounds=False,thickness=2):
    #To remove noise at labelled_image boundaries
    if manage_bounds:
        if type(labelled_image) == np.ma.masked_array :
            roi=np.logical_not(labelled_image.mask)
        new_labelled_image=manage_boundaries(labelled_image,roi,thickness)
        return from_labelled_image(image,new_labelled_image,roi,False)
    #Regions (residues) from labels
    regions=labelled_image2regions(labelled_image,roi)
    #Built graphs from regions
    return from_regions(image,regions)



def from_regions(image,regions):
    built_t_graph,new_residues=topological_graph_from_residues_refactorying(regions)
    built_p_graph=photometric_graph_from_residues_refactorying(image,new_residues)
    built_t_graph.set_image(image);built_p_graph.set_image(image)
    return built_t_graph,built_p_graph

def manage_boundaries(image,roi=None,thickness=2):
    if roi is None: roi=np.ones(image.shape)
    eroded_roi=sp.ndimage.morphology.binary_erosion(roi,iterations=thickness).astype(np.uint8)
    inner_boundary=roi/np.max(roi)-eroded_roi
    inner_boundary_values=np.ma.MaskedArray(image,mask=np.logical_not(inner_boundary)).compressed()
    bins=np.arange(np.min(inner_boundary_values),np.max(inner_boundary_values)+2)
    h,b=np.histogram(inner_boundary_values,bins)
    dominant_value=b[np.argmax(h)]
    modified_image=np.ma.MaskedArray(image,mask=inner_boundary).filled(dominant_value)
    if type(image)==np.ma.MaskedArray:
        modified_image=np.ma.MaskedArray(modified_image,mask=np.logical_not(roi))
    return modified_image


def from_labelled_image_v2(image,labelled_image,roi=None,manage_bounds=False,thickness=2):
    #To remove noise at labelled_image boundaries
    if manage_bounds:
        if type(labelled_image) == np.ma.masked_array :
            roi=np.logical_not(labelled_image.mask)
        new_labelled_image,new_roi=manage_boundaries_v2(labelled_image,roi,thickness)
        return from_labelled_image_v2(image,new_labelled_image,new_roi,False)
    #Regions (residues) from labels
    regions=labelled_image2regions(labelled_image,roi)
    #Built graphs from regions
    return from_regions(image,regions)

def manage_boundaries_v2(image,roi=None,thickness=2):
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
