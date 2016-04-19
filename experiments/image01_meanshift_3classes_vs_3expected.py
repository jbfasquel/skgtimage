import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

input_dir="Database/image01/truth/"
save_dir="Database/image01/meanshift_3classes_vs_3expected/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

image=sp.misc.imread(os.path.join(input_dir,"image.png"))
roi=sp.misc.imread(os.path.join(input_dir,"region_file.png"))

#########
# SEGMENTATION MEANSHIFT
#########
l_image=np.ma.array(image, mask=np.logical_not(roi))
bandwidth=15 #3 classes
labelled_image=skgti.utils.mean_shift(l_image,bandwidth=bandwidth,spatial_dim=2,n_features=1,verbose=True)

#########
# KNOWLEDGE
#########
t_graph=skgti.core.graph_factory("text<paper<file")
p_graph=skgti.core.graph_factory("text<file<paper")
helper.save_initial_context(save_dir,"01_context",image,labelled_image,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
residues=skgti.core.residues_from_labels(labelled_image)
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues)
helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues)

###########################################
# MATCHINGS
###########################################
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)
print("nb p_isos: ", len(p_isomorphisms))
print("nb t_isos: ", len(t_isomorphisms))
print("nb common_isos: ", len(common_isomorphisms))
#quit()

helper.save_matching_details(save_dir,"03_t",built_t_graph,t_graph,matching,common_isomorphisms,None,[eie_sim,eie_dist])
helper.save_matching_details(save_dir,"03_p",built_p_graph,p_graph,matching,common_isomorphisms)

final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(built_t_graph,built_p_graph,t_graph,p_graph,matching)

helper.save_refinement_historization(save_dir,histo,t_graph,p_graph,matching)

helper.save_built_graphs(save_dir,"05_",final_t_graph,final_p_graph)
(relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],matching)

relabelled_final_t_graph.set_image(image) #hack to save mixed region residues
helper.save_built_graphs(save_dir,"06_relabelled_",relabelled_final_t_graph,relabelled_final_p_graph)

