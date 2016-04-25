import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

root_name="image01"
input_dir="Database/image01/truth"
#save_dir="Database/image01/kmeans_3classes/"
save_dir="Database/image01/test_refactoryin/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)


#########
# KNOWLEDGE
#########
t_graph=skgti.core.graph_factory("text<paper<file")
p_graph=skgti.core.graph_factory("text<file<paper")

#########
# IMAGE
#########
image=sp.misc.imread(os.path.join(input_dir,"image.png"))
roi=sp.misc.imread(os.path.join(input_dir,"region_file.png"))


#########
# SEGMENTATION MEANSHIFT
#########
l_image=np.ma.array(image, mask=np.logical_not(roi))
labelled_image=skgti.utils.kmeans(l_image,3,n_seedings=30)

labelled_image=skgti.core.manage_boundaries(labelled_image,roi)
helper.save_initial_context(save_dir,"01_context",image,labelled_image,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(labelled_image)
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

###########################################
# INFLUENCE EIE
###########################################
t_desc="text<paper<file"
p_desc="text<file<paper"
helper.influence_of_commonisos(image,common_isomorphisms,eie_dist,eie_sim,built_t_graph,built_p_graph,t_graph,p_graph,t_desc,p_desc,input_dir,save_dir)