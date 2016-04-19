import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti

####################
# IMAGE
####################
image=np.array([ [1, 1, 1, 1, 1, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 2, 2, 1, 5, 1],
                [1, 2, 2, 1, 6, 1],
                [1, 2, 2, 1, 1, 1],
                [1, 1, 1, 1, 1, 1]])

####################
# KNOWLEDGE
####################
tp_model=skgti.core.TPModel()
tp_model.set_topology("B,C<A")
tp_model.add_photometry("A<B<C")

####################
# INITIALIZATION : SETTING THE IMAGE TO BE ANALYZED
####################
tp_model.set_image(image)

####################
# ONE STEP SEGMENTATION USING KMEANS
####################
#ROI IMAGE
l_image=tp_model.roi_image()

#NUMBER OF CLUSTERS
nb=tp_model.number_of_clusters()

# PARAMETERS FROM KNOWLEDGE : INTERVALS
intervals=tp_model.intervals_for_clusters()

# KMEANS SEEDER
seeder=skgti.utils.KMeansSeeder(l_image,nb,intervals)

# KMEANS SEGMENTATION
labelled_image=skgti.utils.kmeans(l_image,nb,seeder=seeder)

print(labelled_image)

####################
# RECOGNITION
####################
tp_model.identify_from_labels(np.array(labelled_image))

####################
# PLOT
####################
segmentations=[np.where(labelled_image==i,1,0) for i in range(np.min(labelled_image),np.max(labelled_image)+1)]
skgti.io.plot_model(tp_model,segmentations)
plt.show()