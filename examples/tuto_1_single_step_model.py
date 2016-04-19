import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti

# IMAGE
image=np.array([ [1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 2, 2, 1, 3, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]])

# KNOWLEDGE
tp_model=skgti.core.TPModel()
tp_model.set_topology("B,C<A")
tp_model.add_photometry("A<B<C")

# INITIALIZATION : SETTING THE IMAGE TO BE ANALYZED
tp_model.set_image(image)

# SEGMENTATION: ONE CONSIDER FLAT RESIDUES
segmentations=[ np.where(image==i,1,0) for i in [1,2,3] ]

# RECOGNITION OF REGIONS A, B AND C
tp_model.identify_from_residues(segmentations)

# PLOT
skgti.io.plot_model(tp_model,segmentations);plt.show()
