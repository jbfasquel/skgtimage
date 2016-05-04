import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti

############
# THEORIC GRAPH
############
'''
A graph-based image interpretation method using binary inclusion and photometric relationships: principle and implementation.
'''
############
# LABELLED IMAGE
############
'''
#Use case 1: basic
expected_graph=skgti.core.graph_factory("2<1<0")
image=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])
'''
'''
#Use case 2: connected components
expected_graph=skgti.core.graph_factory("2<1<0;3<0")
image=np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 1, 2, 1, 0, 2, 0],
                [0, 1, 1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])
'''
'''
#Use case 3: regions of similar labels
expected_graph=skgti.core.graph_factory("2<1<0")
image=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])
'''
'''
#Use case 4: similar labels + connected components
expected_graph=skgti.core.graph_factory("1<0;2,3<1;4<2")
image=np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 2, 2, 2, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 0, 2, 0, 1, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0]])
'''
'''
#Use case 5: boundary effect
expected_graph=skgti.core.graph_factory("1<0")
image=np.array([[0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 2, 2, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0]])
roi=np.where(image>=1,1,0)
image=np.ma.MaskedArray(image,mask=np.logical_not(roi))
'''

#Use case 6: 3 is included in both 1 and 2, while 1 and 2 only overlap
# -->2--
# |     |
# 3     -->0
# |     |
# -->1--
# Effect: 2 and 1 are split is separated regions
expected_graph=skgti.core.from_string("1<0")
image=np.array([[0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 2, 0, 0],
                [0, 0, 1, 2, 1, 2, 0],
                [0, 1, 2, 3, 2, 1, 0],
                [0, 2, 1, 1, 2, 1, 0],
                [0, 0, 2, 2, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]])

############
# BUILT FROM LABELLED IMAGE
############
built_g,new_residues=skgti.core.topological_graph_from_labels(image)

############
# PLOT EXPECTED AND BUILT GRAPHS
############
plt.subplot(121);skgti.io.plot_graph_refactorying(expected_graph);plt.title("Expected")
plt.subplot(122);skgti.io.plot_graph_refactorying(built_g);plt.title("Built");plt.show()
