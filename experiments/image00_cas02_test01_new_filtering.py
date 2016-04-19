import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

save_dir="Database/image00/cas2_test1_new_filtering/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
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

#sp.misc.imsave(os.path.join(save_dir,"01_image_segmentation.png"),(80*image.astype(np.uint8)))
plt.imshow(image,cmap="gray",vmin=0,vmax=np.max(segmentation),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'01_image_segmentation.png');plt.gcf().clear()


###########################################
# KNOWLEDGE
###########################################
t_graph=skgti.core.graph_factory("B<A")
p_graph=skgti.core.graph_factory("A<B")
skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()
#plt.show();


###########################################
# SEGMENTATION CONTEXT: ALREADY SEGMENTED REGIONS + TARGET + SUBGRAPHS
###########################################
t_graph.set_region("A",np.ones(image.shape))
p_graph.set_region("A",np.ones(image.shape))

# SEGMENTATION: ONE CONSIDER FLAT RESIDUES
nodes=('A','B')
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

#PLOT BUILT GRAPHS AND REGIONS
filled_new_residues=skgti.core.fill_regions(new_residues)
skgti.io.plot_graphs_regions([built_t_graph,built_p_graph],filled_new_residues)
plt.savefig(save_dir+'02_built_graphs.png');plt.gcf().clear()

###########################################
# MATCHINGS
###########################################
matching,(common_isomorphisms,(t_isomorphisms,p_isomorphisms))=skgti.core.recognize_version1([built_t_graph,built_p_graph],[sub_t_graph,sub_p_graph],True)

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
#skgti.io.plot_graphs_matching([built_t_graph,p_graph],[built_t_graph,built_p_graph],matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'04_matching_initial.png');plt.gcf().clear()

###########################################
# NEW
###########################################

new_matching,new_common_isomorphisms=skgti.core.filtered_common_subgraph_isomorphisms_v1(matching,common_isomorphisms)
skgti.io.plot_graph_matchings(built_t_graph,t_graph,new_common_isomorphisms)
plt.savefig(save_dir+'05_t_common_iso_filtered.png');plt.gcf().clear()
skgti.io.plot_graph_matching(built_t_graph,t_graph,new_matching)
plt.savefig(save_dir+'05_t_matching_filtered.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,new_common_isomorphisms)
plt.savefig(save_dir+'05_p_common_iso_filtered.png');plt.gcf().clear()
skgti.io.plot_graph_matching(built_p_graph,p_graph,new_matching)
plt.savefig(save_dir+'05_p_matching_filtered.png');plt.gcf().clear()

skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],new_matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'05_filtered_matching.png');plt.gcf().clear()


skgti.core.t_filtering_v2(image,new_residues,new_matching,built_t_graph,built_p_graph)
skgti.io.plot_graphs_matching([t_graph,p_graph],[built_t_graph,built_p_graph],new_matching,titles=["Topology","Photometry"])
plt.savefig(save_dir+'06_final_matching.png');plt.gcf().clear()

skgti.core.update_graphs([t_graph,p_graph],new_residues,new_matching)

skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'06_result_graph_regions.png');plt.gcf().clear()


result=skgti.utils.combine(image.shape,t_graph,['A','B'],[1,2])
plt.imshow(result,cmap="gray",vmin=np.min(result),vmax=np.max(result),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'06_result_single_image.png');plt.gcf().clear()
skgti.io.save_graph("topo",t_graph,nodes=None,tree=True,directory=save_dir+"final_t_graph",save_regions=True)
