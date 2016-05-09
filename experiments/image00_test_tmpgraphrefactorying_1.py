import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

truth_dir="Database/image00/test_tmp/truth/"
save_dir="Database/image00/test_tmp/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

# IMAGE
image=np.array([[0.0, 0.2, 0.0, 0.0, 0.0],
                [0.2, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.1, 1.4, 0.9, 0.0],
                [0.0, 1.0, 1.3, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.4, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0]])

label=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 4, 1, 0],
                [0, 1, 5, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 6, 0, 0],
                [0, 0, 0, 0, 0]])


t_desc="B<A";t_graph=skgti.core.from_string(t_desc)
p_desc="A<B";p_graph=skgti.core.from_string(p_desc)


built_t_graph,built_p_graph=skgti.core.from_labelled_image_refactorying(image,label)
#built_p_graph.update_intensities(image)
#skgti.core.update_photometric_graph(built_p_graph)
#skgti.io.plot_graph_with_regions_refactorying(built_t_graph);plt.show()
#Case 1
#skgti.core.merge_nodes_topology(built_t_graph,0,1)
#print("Diff nodes:",len(set([1,2,3,4])-set(built_t_graph.nodes())))
#print("Diff edges:",len(set([(2,1),(3,1),(4,1)])-set(built_t_graph.edges())))
#skgti.core.merge_nodes_topology(built_t_graph,1,0)
#skgti.core.merge_nodes_topology(built_t_graph,1,0)
#skgti.core.merge_nodes_topology(built_t_graph,2,3)
#skgti.core.merge_nodes_topology(built_t_graph,1,2)
#skgti.core.merge_nodes_topology(built_t_graph,1,4)
#skgti.core.merge_nodes_topology(built_t_graph,4,1)
#skgti.core.merge_nodes_topology(built_t_graph,4,3)

matching,common_isomorphisms,t_isomorphisms,p_isomorphisms,eie_sim,eie_dist=skgti.core.best_common_subgraphisomorphism(built_t_graph,
                                                                                                       t_graph,
                                                                                                       built_p_graph,
                                                                                                       p_graph,False)

#skgti.io.plot_graph_links(built_t_graph,t_graph,link_lists=[skgti.io.matching2links(matching)],colors=['red']);plt.show()

current_t_graph,current_p_graph,modification_historisation=skgti.core.propagate(built_t_graph,built_p_graph,t_graph,p_graph,matching)
ordered_merges=[i[2] for i in modification_historisation]
print(ordered_merges)
skgti.io.plot_graph_links(built_t_graph,t_graph,link_lists=[skgti.io.matching2links(matching),ordered_merges],colors=['red','green']);plt.show()

#skgti.io.plot_graph_with_regions_refactorying(built_t_graph);plt.show()
'''
for n in built_p_graph.nodes():
    print(n,built_p_graph.get_mean_residue_intensity(n))
skgti.core.merge_nodes_photometry(built_p_graph,2,3)
for n in built_p_graph.nodes():
    print(n,built_p_graph.get_mean_residue_intensity(n))

#skgti.io.plot_graph_with_regions_refactorying(built_t_graph);plt.show()
skgti.io.plot_graph_with_regions_refactorying(built_p_graph);plt.show()
'''
