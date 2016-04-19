import os,pickle
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper


#########
# MISC INFORMATIONS
#########
root_name="image04"
input_dir="Database/image04"
#truth_dir="Database/image03/truth_top/"
save_dir="Database/image04/test/"

#########
# A PRIORI KNOWLEDGE
#########
'''
t_graph=skgti.core.graph_factory("C,D<B<A;E,F<D;G<C;H<E")
p_graph=skgti.core.graph_factory("G=B<E=F<H=D<A=C")
'''
#########
# IMAGE: COLOR AND GRAY
#########
#COLOR IMAGE
image_rgb=sp.misc.imread(os.path.join(input_dir,"image04_filtered.jpeg"))
image=sp.misc.imread(os.path.join(input_dir,"image04_filtered_gray.png"))
roi=sp.misc.imread(os.path.join(input_dir,"image04_region6.png"))
#########
# MEANSHIFT ON COLOR IMAGE
#########

#CLUSTERING
bandwidth=0.12 #->5 CLASSES
segmentation=skgti.utils.mean_shift_rgb(image_rgb,bandwidth=bandwidth,spatial_dim=2,n_features=3,roi=roi,verbose=True) #0.1 OK
# SAVE
#helper.save_initial_context(save_dir,"01_context",image,segmentation,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
#segmentation=skgti.core.manage_boundaries(segmentation,roi)
# ROI
l_image=np.ma.array(image, mask=np.logical_not(roi))
l_segmentation=np.ma.array(segmentation, mask=np.logical_not(roi))
# BUILD
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(l_segmentation)
built_p_graph=skgti.core.photometric_graph_from_residues(l_image,new_residues)
# SAVE
helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues)

'''
###########################################
# INITIAL MATCHING
###########################################
#matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)

#SAVE
#helper.pickle_isos(save_dir,"02_",matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist)
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=helper.unpickle_isos(save_dir,"02_")
helper.save_matching_details(save_dir,"03_t",built_t_graph,t_graph,matching,common_isomorphisms,None,[eie_sim,eie_dist])
helper.save_matching_details(save_dir,"03_p",built_p_graph,p_graph,matching,common_isomorphisms)

###########################################
# GREEDY REFINEMENT
###########################################
final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(built_t_graph,built_p_graph,t_graph,p_graph,matching)
(relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],matching)

#SAVE
helper.save_refinement_historization(save_dir,histo,t_graph,p_graph,matching)
helper.save_built_graphs(save_dir,"05_",final_t_graph,final_p_graph)
relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
helper.save_built_graphs(save_dir,"06_relabelled_",relabelled_final_t_graph,relabelled_final_p_graph)
'''