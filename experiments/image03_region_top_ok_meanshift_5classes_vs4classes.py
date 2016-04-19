import os,pickle
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper


#########
# MISC INFORMATIONS
#########
root_name="image03"
dir="Database/image03"
truth_dir="Database/image03/truth_top/"
save_dir="Database/image03/top_meanshift_ok_5classes_versus4expected/"

#########
# A PRIORI KNOWLEDGE
#########
t_graph=skgti.core.graph_factory("C,D<B<A;E,F<D;G<C;H<E")
p_graph=skgti.core.graph_factory("G=B<E=F<H=D<A=C")

#########
# IMAGE: COLOR AND GRAY
#########
#COLOR IMAGE
image_rgb=sp.misc.imread(os.path.join(dir,"image03.png"))
#ROI
roi=sp.misc.imread(os.path.join(truth_dir,"region_A.png"))
#GRAYSCALE IMAGE
image=skgti.utils.rgb2gray(image_rgb)
sp.misc.imsave(os.path.join(dir,"image_gray.png"),image)

#########
# MEANSHIFT ON COLOR IMAGE
#########
nodes=list(skgti.core.classes_for_targets(t_graph,'B'))
nodes_of_interest=skgti.core.distinct_classes(nodes,[p_graph])
nb_classes=len(nodes_of_interest)
print("Number of distinct classes: ",nb_classes)
#CLUSTERING
bandwidth=0.1 #->5 CLASSES
segmentation=skgti.utils.mean_shift_rgb(image_rgb,bandwidth=bandwidth,spatial_dim=2,n_features=3,roi=roi,verbose=True) #0.1 OK
# SAVE
helper.save_initial_context(save_dir,"01_context",image,segmentation,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
segmentation=skgti.core.manage_boundaries(segmentation,roi)
# ROI
l_image=np.ma.array(image, mask=np.logical_not(roi))
l_segmentation=np.ma.array(segmentation, mask=np.logical_not(roi))
# BUILD
built_t_graph,new_residues=skgti.core.topological_graph_from_labels(l_segmentation)
built_p_graph=skgti.core.photometric_graph_from_residues(l_image,new_residues)
# SAVE
helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues)

###########################################
# INITIAL MATCHING
###########################################
#matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)

#SAVE
#helper.pickle_isos(save_dir,"02_",matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist)
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=helper.unpickle_isos(save_dir,"02_")
print("nb p_isos: ", len(p_isomorphisms))
print("nb t_isos: ", len(t_isomorphisms))
print("nb common_isos: ", len(common_isomorphisms))
#quit()

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

###########################################
# INFLUENCE EIE
###########################################
t_desc="C,D<B<A;E,F<D;G<C;H<E"
p_desc="G=B<E=F<H=D<A=C"
helper.influence_of_commonisos(image,common_isomorphisms,eie_dist,eie_sim,built_t_graph,built_p_graph,t_graph,p_graph,t_desc,p_desc,truth_dir,save_dir)