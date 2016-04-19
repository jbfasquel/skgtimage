import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti

# IMAGE
image=np.array([[1,1,1,1,1,1,1],
                [1,2,2,2,1,1,1],
                [1,2,3,2,1,3,1],
                [1,2,3,2,1,3,1],
                [1,2,2,2,1,1,1],
                [1,1,1,1,1,1,1]])

# KNOWLEDGE
tp_model=skgti.core.TPModel()
tp_model.set_topology("C<B<A;D<A")
tp_model.add_photometry("A<B<C=D")

# INITIALIZATION : SETTING THE IMAGE
tp_model.set_image(image)

# SEGMENTATION
segmentations=[ np.where(image==i,1,0) for i in [1,2,3] ]

# IDENTIFICATION
tp_model.identify_from_residues(segmentations)

# PLOT
skgti.io.plot_model(tp_model,segmentations);plt.show()
