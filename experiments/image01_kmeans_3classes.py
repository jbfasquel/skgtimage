import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

truth_dir="Database/image01/truth/"
save_dir="Database/image01/kmeans_3classes_refactorying/"


t_desc="text<paper<file"
p_desc="text<file<paper"

image=sp.misc.imread(truth_dir+"image.png")
roi=sp.misc.imread(truth_dir+"region_file.png")
'''
# SEGMENTATION KMEANS
label=skgti.utils.kmeans_refactorying(image,3,50,roi=roi,mc=False,verbose=True)

# RECOGNITION
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=False,verbose=True)

skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")
'''
matcher=skgti.io.unpickle_matcher(save_dir+"matcher.pkl")
# EVALUATION VS TRUTH
import helper
'''
classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")
print("Evaluation of all regions vs truth: GCR = ", classif, " ; Similarities = " , region2sim)
# EVALUATION VS RAWSEGMENTATION
region2segmentintensities={'text':127,'paper':0,'file':255}
classif_result,classif_rawsegmentation=helper.compared_with_rawsegmentation_refactorying(save_dir+"00_context/labelled_image.png",t_desc,p_desc,image,region2segmentintensities,
                                                                                         save_dir+"06_final/",truth_dir,save_dir+"07_eval_vs_raw_seg/")
print("Raw segmentation vs truth: ",classif_rawsegmentation, "(proposed method GCR=",classif_result,")")
'''
# EVALUATION VS CHOICE OF THE INITIAL COMMON ISOMORPHISM
helper.influence_of_commonisos_refactorying(matcher,image,t_desc,p_desc,truth_dir,save_dir)
'''
tmp=skgti.core.manage_boundaries(labelled_image,roi) #optional: just for display
helper.save_initial_context(save_dir,"01_context",image,tmp,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
built_t_graph,built_p_graph=skgti.core.from_labelled_image(image,labelled_image,roi,manage_bounds=True)
helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph)

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
'''