import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3,1],
                [1,2,3,2,1,1,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]])

# KNOWLEDGE
tp_model=skgti.core.TPModel()
tp_model.set_topology("C<B<A")
tp_model.add_photometry("A<B<C")

# INITIALIZATION : SETTING THE IMAGE
tp_model.set_image(image)

# SEGMENTATION
segmentations=[ np.where(image==i,1,0) for i in [1,2,3] ]

# PLOTTING OBTAINED GRAPH VS A PRIORI ONE
#t_graph,_=skgti.core.topological_graph_from_residues(segmentations)
#plt.subplot(121);skgti.io.plot_graph(tp_model.t_graph);plt.title("A priori graph")
#plt.subplot(122);skgti.io.plot_graph(t_graph);plt.title("Obtained graph");plt.show();quit()

# IDENTIFICATION INVOLVING INEXACT GRAPH MATCHING
tp_model.identify_from_residues(segmentations)


# PLOT
skgti.io.plot_model(tp_model,segmentations)
plt.show()
