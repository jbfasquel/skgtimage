import os,pickle
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper


root_name="image03"
dir="Database/image03"
save_dir="Database/image03/meanshift_down3_region_top_test1_eie/"

#########
# KNOWLEDGE
#########
#A:contour_blanc ; B:bandeau noir ; C: texte_blanc_du_noir ; D:fond_jaune ; E:texte_du_fond_jaune ; F:fleurs
#G:trous_noirs_du_texte_de_C ; H: fond_jaune_du_text_sur_jaune_E
t_graph=skgti.core.graph_factory("C,D<B<A;E,F<D;G<C;H<E")
p_graph=skgti.core.graph_factory("G=B<E=F<H=D<A=C")

#########
# IMAGE: COLOR AND GRAY
#########
image_rgb=sp.misc.imread(os.path.join(dir,"image03.jpeg"))
roi=sp.misc.imread(os.path.join(dir,"image03_region0.png"))
#CROP
image_rgb=skgti.utils.extract_subarray_rgb(image_rgb,roi)
roi=skgti.utils.extract_subarray(roi,roi)
#DOWNSAMPLING
downsampling=3
image_rgb=image_rgb[::downsampling,::downsampling,:]
roi=roi[::downsampling,::downsampling]
#GRAYSCALE
image=skgti.utils.rgb2gray(image_rgb)

#########
# MEANSHIFT ON COLOR IMAGE -> DOES NOT WORK AND GRAY -> NOT ENOUGH DISCRIMINATION
#########
#CLUSTERING
bandwidth=0.09 #il faut au min 4 classes
segmentation=skgti.utils.mean_shift_rgb(image_rgb,bandwidth=bandwidth,spatial_dim=2,n_features=3,roi=roi,verbose=True) #0.1 OK
# BOUNDARIES
segmentation=skgti.core.manage_boundaries(segmentation,roi)
# ROI
l_image=np.ma.array(image, mask=np.logical_not(roi))
l_segmentation=np.ma.array(segmentation, mask=np.logical_not(roi))
#helper.save_initial_context(save_dir,"01_context",image,segmentation,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(l_segmentation)
built_p_graph=skgti.core.photometric_graph_from_residues(l_image,new_residues)
#helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues)

###########################################
# MATCHING
###########################################
#matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)
#helper.pickle_isos(save_dir,"02_",matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist)
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=helper.unpickle_isos(save_dir,"02_")

#helper.save_matching_details(save_dir,"03_t",built_t_graph,t_graph,matching,common_isomorphisms,None,[eie_sim,eie_dist])
#helper.save_matching_details(save_dir,"03_p",built_p_graph,p_graph,matching,common_isomorphisms)

print("merge 1")
new_t_graph,new_p_graph=skgti.core.merge1(built_t_graph,built_p_graph,t_graph,p_graph,matching)
#skgti.io.save_graph("topo",new_t_graph,nodes=None,tree=True,directory=save_dir+"t_test",save_regions=True,save_residues=True)
#skgti.io.save_graph("topo",new_p_graph,nodes=None,tree=True,directory=save_dir+"p_test",save_regions=True)

#common_isomorphisms2,isomorphisms_per_graph=skgti.core.generate_common_subgraphisomorphisms([new_t_graph,new_p_graph],[t_graph,p_graph])
#helper.pickle_isos2(save_dir,"04_",common_isomorphisms2)
common_isomorphisms2=helper.unpickle_isos2(save_dir,"04_")
print(len(common_isomorphisms2))
print(matching)
for c in common_isomorphisms2:
    is_ok=True
    for e in matching:
        if e not in c: is_ok=False
    print("Ok",is_ok)
helper.save_matching_details(save_dir,"04_t",new_t_graph,t_graph,common_isomorphisms=common_isomorphisms2)
helper.save_matching_details(save_dir,"04_p",new_p_graph,p_graph,common_isomorphisms=common_isomorphisms2)

quit()

