import os
import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

save_dir="Database/image00/cas2_test1_eie/"
###########################################
# KNOWLEDGE
###########################################
image=np.array([ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 2, 2.5, 2, 2, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

segmentation=np.array([
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 2, 3, 2, 2, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

###########################################
# KNOWLEDGE
###########################################
t_graph=skgti.core.graph_factory("B<A")
p_graph=skgti.core.graph_factory("A<B")

#helper.save_initial_context(save_dir,"01_context",image,segmentation,t_graph,p_graph)
###########################################
# BUILDING GRAPHS
###########################################
built_t_graph,residues=skgti.core.topological_graph_from_labels(segmentation)
built_p_graph=skgti.core.photometric_graph_from_residues(image,residues)

#helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,residues)
###########################################
# MATCHING
###########################################
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)
#helper.pickle_isos(save_dir,"02_",matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist)
#matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=helper.unpickle_isos(save_dir,"02_")


#helper.save_matching_details(save_dir,"03_t",built_t_graph,t_graph,matching,common_isomorphisms,t_isomorphisms,[eie_sim,eie_dist])
#helper.save_matching_details(save_dir,"03_p",built_p_graph,p_graph,matching)

###########################################
# FINAL FILTERING
###########################################
skgti.core.merge1(built_t_graph,built_p_graph,matching)