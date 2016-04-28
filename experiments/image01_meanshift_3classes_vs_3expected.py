import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="Database/image01/truth/"
save_dir="Database/image01/meanshift_3classes_vs_3expected_refactorying/"


t_desc="text<paper<file"
p_desc="text<file<paper"

image=sp.misc.imread(truth_dir+"image.png")
roi=sp.misc.imread(truth_dir+"region_file.png")

# SEGMENTATION MEANSHIFT

label=skgti.utils.mean_shift(image,15,roi,False,True)
# RECOGNITION
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=False,verbose=True)

skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")


# EVALUATION
import helper
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)
# EVALUATION VS RAWSEGMENTATION
region2segmentintensities={'text':255,'paper':127,'file':0}
classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation_refactorying(save_dir+"00_context/labelled_image.png",t_desc,p_desc,image,region2segmentintensities,
                                                                                         save_dir+"06_final/",truth_dir,save_dir+"07_eval_vs_raw_seg/")
print("Raw segmentation vs truth: ",classif_rawsegmentation, "(proposed method GCR=",classif_result,")")




'''
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
'''
