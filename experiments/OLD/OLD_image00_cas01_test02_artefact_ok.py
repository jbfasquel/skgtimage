import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti
import os
import scipy as sp;from scipy import misc


save_dir="Database/image00/test2/"

# IMAGE
image=np.array([ [1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 0, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 1, 1, 1, 1, 1]])

segmentation=np.array([ [1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 0, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 1, 1, 1, 1, 1]])

sp.misc.imsave(os.path.join(save_dir,"01_image_segmentation.png"),(80*image.astype(np.uint8)))

# KNOWLEDGE
t_graph=skgti.core.graph_factory("B,C<A")
p_graph=skgti.core.graph_factory("A<B<C")
print("Number of automorphisms:",skgti.core.nb_automorphisms([t_graph,p_graph]))
skgti.io.plot_graph(t_graph);plt.title("A priori T knowledge");plt.savefig(save_dir+'01_A_priori_topo.png');plt.gcf().clear()
skgti.io.plot_graph(p_graph);plt.title("A priori P knowledge");plt.savefig(save_dir+'01_A_priori_photo.png');plt.gcf().clear()

# INITIALIZATION : SETTING THE IMAGE TO BE ANALYZED
t_graph.set_region("A",np.ones(image.shape))
p_graph.set_region("A",np.ones(image.shape))


# SEGMENTATION: ONE CONSIDER FLAT RESIDUES
nodes=('A','B','C')
target2residues,built_t_graph,built_p_graph,matching,matchings,t_isomorphisms,p_isomorphisms,new_residues=skgti.core.identify_from_labels(image,segmentation,t_graph,p_graph,nodes,True)


#PLOT BUILT GRAPHS AND REGIONS
filled_new_residues=skgti.core.fill_regions(new_residues)
skgti.io.plot_graphs_regions([built_t_graph,built_p_graph],filled_new_residues);plt.title("Surjection")
plt.savefig(save_dir+'02_built_graphs.png');plt.gcf().clear()

#PLOT MATCHING
skgti.io.plot_graph_matchings(built_t_graph,t_graph,t_isomorphisms,nb_rows=2)
plt.savefig(save_dir+'03_t_iso.png');plt.gcf().clear()
skgti.io.plot_graph_matchings(built_p_graph,p_graph,p_isomorphisms)
plt.savefig(save_dir+'03_p_iso.png');plt.gcf().clear()
skgti.io.plot_graph_surjection(built_t_graph,t_graph,matching)
plt.savefig(save_dir+'04_t_matching.png');plt.gcf().clear()
skgti.io.plot_graph_surjection(built_p_graph,p_graph,matching) #;plt.show()
plt.savefig(save_dir+'04_p_matching.png');plt.gcf().clear()


for k in target2residues:
    t_graph.set_region(k,skgti.core.fill_region(target2residues[k]))
    p_graph.set_region(k,skgti.core.fill_region(target2residues[k]))


skgti.io.plot_graphs_regions_new([t_graph,p_graph],nb_rows=1)
plt.savefig(save_dir+'04_result.png');plt.gcf().clear()

