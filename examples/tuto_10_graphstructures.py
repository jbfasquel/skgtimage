import numpy as np
import skgtimage as skgti
import matplotlib.pylab as plt


# IMAGE
image=np.array([[1, 1, 1, 1, 1, 1, 1],
                [1, 2, 2, 2, 1, 3, 1],
                [1, 2, 4, 2, 1, 3, 1],
                [1, 2, 2, 2, 1, 3, 1],
                [1, 1, 1, 1, 1, 1, 1]])

r0=np.array(   [[1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1]])

r1=np.array(   [[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])

r2=np.array(   [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])

r3=np.array(   [[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]])

# GRAPH
#ipg=skgti.core.IPGraph(image);print(ipg)
ig=skgti.core.IGraph(image=image);print(ig)
skgti.core.fromstring("3<0;2<1<0",ig)
ig.set_region('0',r0)
ig.set_region('1',r1)
ig.set_region('2',r2)
ig.set_region('3',r3)
#print(ig)
#skgti.io.plot_graphs_regions_new([ig]);plt.show()
built_t_graph,new_residues=skgti.core.topological_graph_from_residues([r0,r1,r2,r3])
#print(new_residues)
built_p_graph=skgti.core.photometric_graph_from_residues(image,new_residues)
print(built_p_graph)
#ig.add_node(0)
