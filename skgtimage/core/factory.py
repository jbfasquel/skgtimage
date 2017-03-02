import re,itertools
import numpy as np
import scipy as sp; from scipy import ndimage
import networkx as nx
from skgtimage.core.graph import IrDiGraph,transitive_reduction,labelled_image2regions,rename_nodes
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

def from_dir(path,multichannel=False):
    import os,re
    import scipy as sp;from scipy import misc;
    image=sp.misc.imread(os.path.join(path,'image.png'))
    regions=[]
    id2region={}
    for f in os.listdir(path):
        if re.match(".*\.png",f) is not None:
            #Any file ending with .png (image) and starting with 'region'
            if (f.split('.')[1] == 'png') & (f.split('_')[0]=='region'):
                r_name=(f.split('.')[0]).split('_')[1]
                region=sp.misc.imread(os.path.join(path,f))
                regions+=[region]
                id2region[r_name]=region
    if multichannel:
        image=0.2125 * image[:, :, 0] + 0.7154 * image[:, :, 1] + 0.0721 * image[:, :, 2]
        #image=rgb2gray(image)
    built_t,build_p=from_regions(image,regions)
    #Relabel
    node2id={}
    for n in built_t.nodes():
        for id in id2region:
            region_graph=built_t.get_region(n)
            region_fs=id2region[id].astype(np.float)
            region_graph/=np.max(region_graph)
            region_fs/=np.max(region_fs)
            if np.array_equal(region_graph,region_fs):
                node2id[n]=id

    built_t, build_p=rename_nodes([built_t,build_p],node2id)

    return built_t,build_p

def from_regions(image,regions):
    built_t_graph,new_residues=topological_graph_from_residues_refactorying(regions)
    built_p_graph=photometric_graph_from_residues_refactorying(image,new_residues)
    built_t_graph.set_image(image);built_p_graph.set_image(image)
    return built_t_graph,built_p_graph


def from_labelled_image(image, labelled_image, roi=None, manage_bounds=False, thickness=1,verbose=False):
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
