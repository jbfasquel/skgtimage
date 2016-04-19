import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti


image=np.array([ [1, 1, 1, 1],
                 [1, 3.5, 4, 1],
                 [1, 0, 0, 1],
                 [1, 1, 1, 1]])

segmentation=np.array([ [1, 1, 1, 1],
                 [1, 2, 3, 1],
                 [1, 4, 4, 1],
                 [1, 1, 1, 1]])

residues=skgti.core.residues_from_labels(segmentation)
built_t_graph,new_residues=skgti.core.topological_graph_from_residues(residues)

built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues)
#skgti.core.build_similarities(image,new_residues,built_p_graph,1)

for i in range(0,len(new_residues)):
    built_t_graph.set_region(i,skgti.core.fill_region(new_residues[i]))
    built_p_graph.set_region(i,skgti.core.fill_region(new_residues[i]))

skgti.io.plot_graphs_regions_new([built_t_graph,built_p_graph],nb_rows=1);plt.show()
