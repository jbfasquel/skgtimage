import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti
import os
import scipy as sp;from scipy import misc

save_dir="Database/image00/cas2_test1/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)


# IMAGE
image=np.array([ [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1, 1],
                [1, 2, 4, 2, 1, 1, 1],
                [1, 2, 4, 2, 1, 3, 1],
                [1, 2, 2, 2, 1, 3, 1],
                [1, 1, 1, 1, 1, 1, 1]])

segmentation=np.array([ [1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 1, 1],
                [1, 2, 4, 2, 1, 1, 1],
                [1, 2, 4, 2, 1, 3, 1],
                [1, 2, 2, 2, 1, 3, 1],
                [1, 1, 1, 1, 1, 1, 1]])

#sp.misc.imsave(os.path.join(save_dir,"01_image_segmentation.png"),(50*image.astype(np.uint8)))
plt.imshow(image,cmap="gray",vmin=0,vmax=np.max(segmentation),interpolation="nearest");plt.axis('off');
plt.savefig(save_dir+'01_image_segmentation.png');plt.gcf().clear()

# KNOWLEDGE
t_graph=skgti.core.graph_factory("B,C<A;D<B")
p_graph=skgti.core.graph_factory("A<B<C<D")


# INITIALIZATION : SETTING THE IMAGE TO BE ANALYZED
t_graph.set_region("A",np.ones(image.shape))
p_graph.set_region("A",np.ones(image.shape))


skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()

# SEGMENTATION: ONE CONSIDER FLAT RESIDUES
nodes=('A','B','C','D')
residues=skgti.core.residues_from_labels(segmentation)
sub_p_graph=skgti.core.transitive_reduction(skgti.core.transitive_closure(p_graph).subgraph(nodes))
sub_t_graph=skgti.core.transitive_reduction(skgti.core.transitive_closure(t_graph).subgraph(nodes))

autos=skgti.core.nb_automorphisms([sub_t_graph,sub_p_graph])
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)
n=skgti.core.number_of_brother_links(sub_p_graph)
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues,n)

#PLOT BUILT GRAPHS AND REGIONS
filled_new_residues=skgti.core.fill_regions(new_residues)
skgti.io.plot_graphs_regions([built_t_graph,built_p_graph],filled_new_residues);plt.title("Surjection")
plt.savefig(save_dir+'02_built_graphs.png');plt.gcf().clear()


#Matching
t_isomorphisms=skgti.core.find_subgraph_isomorphims(skgti.core.transitive_closure(built_t_graph),skgti.core.transitive_closure(sub_t_graph))
p_isomorphisms=skgti.core.find_subgraph_isomorphims(skgti.core.transitive_closure(built_p_graph),skgti.core.transitive_closure(sub_p_graph))
matchings=skgti.core.find_common_isomorphisms([p_isomorphisms,t_isomorphisms])

#PLOT MATCHING
skgti.io.plot_graph_matchings(built_t_graph,t_graph,t_isomorphisms,nb_rows=2)
plt.savefig(save_dir+'03_t_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,p_isomorphisms)
plt.savefig(save_dir+'03_p_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_t_graph,t_graph,matchings)
plt.savefig(save_dir+'04_t_matching.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,matchings) #;plt.show()
plt.savefig(save_dir+'04_p_matching.png');plt.gcf().clear()

#SURJECTION
surj=skgti.core.find_sub_surjection(matchings)
skgti.io.plot_graph_matchings(p_graph,built_p_graph,[surj])
plt.savefig(save_dir+'05_p_surjection.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(t_graph,built_t_graph,[surj])
plt.savefig(save_dir+'05_t_surjection.png');plt.gcf().clear()

#FINAL FILTERING
unrelated_nodes=skgti.core.unmatched_nodes(matchings,built_t_graph.nodes())
target2residues=skgti.core.update_residues(new_residues,built_t_graph,surj,unrelated_nodes)
for k in target2residues:
    t_graph.set_region(k,skgti.core.fill_region(target2residues[k]))
    p_graph.set_region(k,skgti.core.fill_region(target2residues[k]))


skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'05_result.png');plt.gcf().clear()
