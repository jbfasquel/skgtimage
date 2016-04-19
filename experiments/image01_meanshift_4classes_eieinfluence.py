import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

input_dir="Database/image01/truth"
save_dir="Database/image01/meanshift_4classes_eieinfluence/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

image=sp.misc.imread(os.path.join(input_dir,"image.png"))
roi=sp.misc.imread(os.path.join(input_dir,"region_file.png"))

#########
# SEGMENTATION MEANSHIFT
#########
l_image=np.ma.array(image, mask=np.logical_not(roi))
bandwidth=10 #3 classes
labelled_image=skgti.utils.mean_shift(l_image,bandwidth=bandwidth,spatial_dim=2,n_features=1,verbose=True)

#########
# KNOWLEDGE
#########
t_desc="text<paper<file"
p_desc="text<file<paper"

t_graph=skgti.core.graph_factory("text<paper<file")
p_graph=skgti.core.graph_factory("text<file<paper")
#helper.save_initial_context(save_dir,"01_context",image,labelled_image,t_graph,p_graph)

###########################################
# BUILDING GRAPHS
###########################################
residues=skgti.core.residues_from_labels(labelled_image)
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues)
#helper.save_built_graphs(save_dir,"02_",built_t_graph,built_p_graph,new_residues)

###########################################
# MATCHINGS
###########################################
matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.recognize_version2(built_t_graph,t_graph,built_p_graph,p_graph,True)
print("nb p_isos: ", len(p_isomorphisms))
print("nb t_isos: ", len(t_isomorphisms))
print("nb common_isos: ", len(common_isomorphisms))

helper.influence_of_commonisos(image,common_isomorphisms,eie_dist,eie_sim,built_t_graph,built_p_graph,t_graph,p_graph,t_desc,p_desc,input_dir,save_dir)
'''
performances=[]
for i in range(0,len(common_isomorphisms)):
    current_matching=common_isomorphisms[i]
    try :
        final_t_graph,final_p_graph,histo=skgti.core.greedy_refinement_v3(built_t_graph,built_p_graph,t_graph,p_graph,current_matching)
        (relabelled_final_t_graph,relabelled_final_p_graph)=skgti.core.rename_nodes([final_t_graph,final_p_graph],current_matching)
        relabelled_final_t_graph.set_image(image) #hack to save mixed region residues

        helper.save_built_graphs(save_dir,"06_relabelled_"+str(i)+"_",relabelled_final_t_graph,relabelled_final_p_graph)
        classif,region2sim=helper.compared_with_truth(image,t_desc,p_desc,input_dir,save_dir+"06_relabelled_"+str(i)+"_built_t_graph",save_dir+"07_eval_classif_"+str(i)+"/")
        #print("Eie dis: ",eie_dist[i]," - Eie sim: ", eie_sim[i], " --> classif: " , classif)
        performances+=[classif]
    except Exception as e:
        print("exception ",e)
        performances+=["Failed"]

#####
# SAVING TO CSV
import csv
fullfilename=os.path.join(save_dir,"06_classif_vs_commoniso.csv")
csv_file=open(fullfilename, "w")
c_writer = csv.writer(csv_file,dialect='excel')
c_writer.writerow(["Result for each commoniso"])
c_writer.writerow(['Eie dist']+[i for i in eie_dist])
c_writer.writerow(['Eie sim']+[i for i in eie_sim])
c_writer.writerow(['GCR']+[i for i in performances])
csv_file.close()

print("Eies dis: ",eie_dist)
print("Eies sim: ",eie_sim)
print("performances: ",performances)
'''