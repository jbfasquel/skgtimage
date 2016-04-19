import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper


root_name="image02"
dir="Database/image02/"
truth_dir=dir+"truth/"
save_dir="Database/image02/meanshift_bandwidth15/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)


#########
# A PRIORI KNOWLEDGE
#########
t_graph=skgti.core.graph_factory("tumor,vessel<liver")
p_graph=skgti.core.graph_factory("tumor<liver<vessel")


#########
# LOAD DATA
#########
image=np.load(os.path.join(dir,"image_filtered.npy"))
roi=np.load(os.path.join(truth_dir,"region_liver.npy"))
#CROP
#image=skgti.utils.extract_subarray(image,roi)
#roi=skgti.utils.extract_subarray(roi,roi)

#helper.save_3d_slices(save_dir,"01_context_image",image,slices=[45])
#helper.plot(image,45);quit()
#ROI
l_image=np.ma.array(image, mask=np.logical_not(roi))
#helper.save_3d_slices(save_dir,"01_context_image_roi",l_image,slices=[45])

#########
# SEGMENTATION MEANSHIFT
#########
bandwidth=15 #15->7 classes ; 20)->3 classes (artefacts) ; 19->5 classes (mauvaise segmentation)
#labelled_image=skgti.utils.mean_shift(l_image,bandwidth=bandwidth,spatial_dim=3,n_features=1,verbose=True)
#np.save(save_dir+"labelled.npy",labelled_image.filled(0))
labelled_image=np.load(save_dir+"labelled.npy");labelled_image=np.ma.array(labelled_image, mask=np.logical_not(roi))
#helper.plot(labelled_image.filled(0),45)
#helper.save_initial_context(save_dir,"01_context",image,labelled_image,t_graph,p_graph)
###########################################
# BUILDING GRAPHS
###########################################
segmentation=skgti.core.manage_boundaries(labelled_image,roi)
# BUILD
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(segmentation)
built_p_graph=skgti.core.photometric_graph_from_residues(l_image,new_residues)

#helper.save_built_graphs3D(save_dir,"02_",built_t_graph,built_p_graph,new_residues,[45])
built_t_graph.set_image(l_image)
built_p_graph.set_image(l_image)
skgti.core.remove_smallest_leaf_regions(built_t_graph,built_p_graph)
helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues,[45])

###########################################
# MATCHINGS
###########################################
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)
print("nb p_isos: ", len(p_isomorphisms))
print("nb t_isos: ", len(t_isomorphisms))
print("nb common_isos: ", len(common_isomorphisms))
#quit()

#helper.save_matching_details(save_dir,"03_t",built_t_graph,t_graph,matching,common_isomorphisms,None,[eie_sim,eie_dist])
#helper.save_matching_details(save_dir,"03_p",built_p_graph,p_graph,matching,common_isomorphisms)

final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(built_t_graph,built_p_graph,t_graph,p_graph,matching)

helper.save_refinement_historization(save_dir,histo,t_graph,p_graph,matching)

helper.save_built_graphs(save_dir,"05_",final_t_graph,final_p_graph,slices=[45])
(relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],matching)

relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
helper.save_built_graphs(save_dir,"06_relabelled_",relabelled_final_t_graph,relabelled_final_p_graph,slices=[45])

