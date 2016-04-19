import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

save_dir="Database/image00/cas0_test5/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

image=np.array([ [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 2, 3, 3, 2, 1, 1, 1],
                 [1, 1, 1, 2, 2.5, 3, 2, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

segmentation=np.array([
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 2, 3, 3, 2, 1, 1, 1],
                 [1, 1, 1, 2, 3, 3, 2, 1, 1, 1],
                 [1, 1, 1, 2, 2, 2, 2, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

#sp.misc.imsave(os.path.join(save_dir,"01_image_segmentation.png"),(80*image.astype(np.uint8)))
plt.imshow(image,cmap="gray",vmin=0,vmax=np.max(image),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'01_image_initiale.png');plt.gcf().clear()
plt.imshow(segmentation,cmap="gray",vmin=0,vmax=np.max(segmentation),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'01_image_label.png');plt.gcf().clear()

###########################################
# KNOWLEDGE
###########################################
t_graph=skgti.core.graph_factory("C<B<A")
p_graph=skgti.core.graph_factory("A<B<C")
print("Number of automorphisms:",skgti.core.nb_automorphisms([t_graph,p_graph]))
skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()

###########################################
# SEGMENTATION CONTEXT: ALREADY SEGMENTED REGIONS + TARGET + SUBGRAPHS
###########################################
t_graph.set_region("A",np.ones(image.shape))
p_graph.set_region("A",np.ones(image.shape))
nodes=('A','B','C')
residues=skgti.core.residues_from_labels(segmentation)
[sub_t_graph,sub_p_graph]=skgti.core.get_sub_graphs([t_graph,p_graph],nodes)

###########################################
# BUILDING GRAPHS
###########################################
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues)

filled_new_residues=skgti.core.fill_regions(new_residues)
skgti.io.plot_graphs_regions([built_t_graph,built_p_graph],filled_new_residues)
plt.savefig(save_dir+'02_built_graphs.png');plt.gcf().clear()

###########################################
# TEMP
###########################################
matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))=skgti.core.recognize_version1([built_t_graph,built_p_graph],[t_graph,p_graph],True)

#quit()

###########################################
# MATCHINGS
###########################################
#matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))=skgti.core.find_common_subgraph_isomorphisms([built_t_graph,built_p_graph],[sub_t_graph,sub_p_graph],True)

skgti.io.plot_graphs_regions_new([p_graph]+skgti.core.compute_possible_graphs(p_graph))
plt.savefig(save_dir+'03_p_all_permutations.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_t_graph,t_graph,t_isomorphisms,nb_rows=2)
plt.savefig(save_dir+'03_t_all_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,p_isomorphisms,nb_rows=2)
plt.savefig(save_dir+'03_p_all_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_t_graph,t_graph,common_isomorphisms)
plt.savefig(save_dir+'03_t_common_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,common_isomorphisms) #;plt.show()
plt.savefig(save_dir+'03_p_common_iso.png');plt.gcf().clear()

#skgti.io.plot_graph_matchings(built_t_graph,t_graph,t_isomorphisms,nb_rows=2);plt.show();plt.gcf().clear()
#skgti.io.plot_graph_matchings(built_p_graph,p_graph,p_isomorphisms,nb_rows=2);plt.show();plt.gcf().clear()
#skgti.io.plot_graph_matchings(built_t_graph,t_graph,common_isomorphisms,nb_rows=2);plt.show();plt.gcf().clear()
#skgti.io.plot_graph_matchings(built_p_graph,p_graph,common_isomorphisms,nb_rows=2);plt.show();plt.gcf().clear()
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
#skgti.io.plot_graphs_matching([built_t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_initial.png');plt.gcf().clear()

###########################################
# MATCHING REFINEMENT (FILTERING)
###########################################
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


result=skgti.utils.combine(image.shape,t_graph,['A','B'],[1,2])
plt.imshow(result,cmap="gray",vmin=np.min(result),vmax=np.max(result),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'06_result_single_image.png');plt.gcf().clear()

#print(nodes_with_ambiguity)


