import os
import numpy as np
import scipy as sp;from scipy import misc
import matplotlib.pyplot as plt
import skgtimage as skgti

root_name="image01"
dir="Database/image01/"
save_dir="Database/image01/meanshift_4classes/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

roi=sp.misc.imread(os.path.join(dir,root_name+"_region2.png"))
image=sp.misc.imread(os.path.join(dir,root_name+".png"))





#########
# SEGMENTATION MEANSHIFT
#########
l_image=np.ma.array(image, mask=np.logical_not(roi))
bandwidth=12 #3 classes
labelled_image=skgti.utils.mean_shift(l_image,bandwidth=bandwidth,spatial_dim=2,n_features=1,verbose=True)
sp.misc.imsave(os.path.join(save_dir,"segmentation.png"),labelled_image)
sp.misc.imsave(os.path.join(save_dir,"segmentation_visu.png"),(200*labelled_image.astype(np.uint8)))
#plt.imshow(labelled_image,cmap="gray",vmin=0,vmax=np.max(labelled_images),interpolation="nearest");plt.show();quit()

#########
# KNOWLEDGE
#########
sub_t_graph=skgti.core.graph_factory("text<paper<file")
sub_p_graph=skgti.core.graph_factory("text<file<paper")
t_graph=sub_t_graph
p_graph=sub_p_graph
skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()

###########################################
# BUILDING GRAPHS
###########################################
residues=skgti.core.residues_from_labels(labelled_image)
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues)

#PLOT
filled_new_residues=skgti.core.fill_regions(new_residues)
skgti.io.plot_graphs_regions([built_t_graph,built_p_graph],filled_new_residues)
plt.savefig(save_dir+'02_built_graphs.png');plt.gcf().clear()

###########################################
# MATCHINGS
###########################################
matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))=skgti.core.recognize_version1([built_t_graph,built_p_graph],[sub_t_graph,sub_p_graph],True)
'''
skgti.io.plot_graph_matchings(built_t_graph,t_graph,t_isomorphisms,nb_rows=2)
plt.savefig(save_dir+'03_t_all_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,p_isomorphisms,nb_rows=2)
plt.savefig(save_dir+'03_p_all_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_t_graph,t_graph,common_isomorphisms)
plt.savefig(save_dir+'03_t_common_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,common_isomorphisms) #;plt.show()
plt.savefig(save_dir+'03_p_common_iso.png');plt.gcf().clear()
'''
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
#skgti.io.plot_graphs_matching([built_t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_initial.png');plt.gcf().clear()

skgti.core.t_filtering_v1(matching,new_residues,built_t_graph,built_p_graph)

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_filtered1.png');plt.gcf().clear()

skgti.core.p_filtering(matching,image,new_residues,built_t_graph,built_p_graph)

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_filtered2.png');plt.gcf().clear()
###########################################
# UPDATING MATCHINGS
###########################################
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'05_final_matching.png');plt.gcf().clear()

skgti.core.update_graphs([t_graph,p_graph],new_residues,matching)

skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'06_result_graph_regions.png');plt.gcf().clear()

result=skgti.utils.combine(image.shape,t_graph,['file','paper','text'],[2,3,1])
plt.imshow(result,cmap="gray",vmin=np.min(result),vmax=np.max(result),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'06_result_single_image.png');plt.gcf().clear()


#########################
#
#########################
from image01_evaluation import eval

eval(roi,labelled_image,t_graph,save_dir)