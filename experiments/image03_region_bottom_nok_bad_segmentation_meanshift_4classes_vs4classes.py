import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

root_name="image03"
dir="Database/image03"
truth_dir="Database/image03/truth_bottom/"
save_dir="Database/image03/bottom_meanshift_nok_bad_segmentation_4classes_versus4expected/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)


#########
# KNOWLEDGE
#########
t_graph=skgti.core.graph_factory("E<D;G<F;D,F,H,I<C<B<A")
p_graph=skgti.core.graph_factory("B=F<D=H<I=E<C=A=G")
#########
# IMAGE: COLOR AND GRAY
#########
image_rgb=sp.misc.imread(os.path.join(dir,"image03.png"))
roi=sp.misc.imread(os.path.join(truth_dir,"region_A.png"))
#GRAYSCALE
image=skgti.utils.rgb2gray(image_rgb)

#########
# MEANSHIFT ON COLOR IMAGE -> DOES NOT WORK AND GRAY -> NOT ENOUGH DISCRIMINATION
#########
nodes=list(skgti.core.classes_for_targets(t_graph,'B'))
nodes_of_interest=skgti.core.distinct_classes(nodes,[p_graph])
nb_classes=len(nodes_of_interest)
print("Number of distinct classes: ",nb_classes)
#CLUSTERING
bandwidth=0.12 #il faut au min 4 classes: 0.09 ; 5 classes 0.08
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
helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues)


###########################################
# INITIAL MATCHING
###########################################
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)
print("nb p_isos: ", len(p_isomorphisms))
print("nb t_isos: ", len(t_isomorphisms))
print("nb common_isos: ", len(common_isomorphisms))
quit()

helper.pickle_isos(save_dir,"02_",matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist)
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=helper.unpickle_isos(save_dir,"02_")
helper.save_matching_details(save_dir,"03_t",built_t_graph,t_graph,matching,common_isomorphisms,None,[eie_sim,eie_dist])
helper.save_matching_details(save_dir,"03_p",built_p_graph,p_graph,matching,common_isomorphisms)

###########################################
# GREEDY REFINEMENT
###########################################
final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(built_t_graph,built_p_graph,t_graph,p_graph,matching)
for i in range(0,len(histo)):
    context=histo[i]
    current_t_graph=context[0]
    current_p_graph=context[1]
    current_matching=context[2]
    helper.save_matching_details(save_dir,"04_t_"+str(i),current_t_graph,t_graph,matching=matching)
    helper.save_matching_details(save_dir,"04_p_"+str(i),current_p_graph,p_graph,matching=matching)
    print("Merge:",current_matching)

helper.save_built_graphs(save_dir,"05_",final_t_graph,final_p_graph)
