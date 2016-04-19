import numpy as np
import scipy as sp;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

####################################
# IMAGE AND KNOWLEDGE
####################################
image=np.array([ [1, 1, 1, 1, 1, 1, 0, 0, 0],
                 [1, 2, 2, 1, 1, 1, 0, 0, 0],
                 [1, 2, 2, 1, 5, 1, 0, 5, 0],
                 [1, 2, 2, 1, 5, 1, 0, 6, 0],
                 [1, 2, 2, 1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 1, 1, 1, 0, 0, 0]])


tp_model=skgti.core.TPModel()
tp_model.set_topology("C,D<B<A;E<A")
tp_model.add_photometry("A<B<C<D=E")

####################################
# INITIALIZATION : SETTING THE IMAGE TO BE ANALYZED
####################################
tp_model.set_image(image)

####################################
# SEQUENTIAL SEGMENTATION STEP 0 - TARGET IS REGION A (THE WHOLE IMAGE)
####################################
region=np.ones(image.shape)
tp_model.set_region('A',region)

####################################
# SEQUENTIAL SEGMENTATION STEP 1: TARGETS REGIONS B AND E (THE WHOLE IMAGE) -
# ASSUMPTION THAT STDDEV LARGER FOR B THAN FOR E
####################################
# SEGMENTATION
thresholding=np.where(image>0,1,0)
labelled_image,nb_labels=sp.ndimage.measurements.label(thresholding,np.ones((3,3)))

# IDENTIFICATION
std_sorted_labels=skgti.core.sort_labels_by_stat(image,labelled_image,fct=np.std)
regionB=np.where(labelled_image==std_sorted_labels[0],1,0)
regionE=np.where(labelled_image==std_sorted_labels[1],1,0)
tp_model.set_region('B',regionB)
tp_model.set_region('E',regionE)

####################################
# SEQUENTIAL SEGMENTATION STEP 2: TARGETS REGIONS C AND D USING KMEANS
####################################
# SETTING THE TARGET(S)
tp_model.set_targets(['C','D'])

# PARAMETERS FROM KNOWLEDGE : ROI ("local image")
l_image=tp_model.roi_image()

# PARAMETERS FROM KNOWLEDGE : NUMBER OF CLUSTERS
nb=tp_model.number_of_clusters()
# PARAMETERS FROM KNOWLEDGE : INTERVALS
intervals=tp_model.intervals_for_clusters()


# KMEANS SEEDER
seeder=skgti.utils.KMeansSeeder(l_image,nb,intervals)

# KMEANS SEGMENTATION
labelled_image=skgti.utils.kmeans(l_image,nb,seeder=seeder)

print(labelled_image)


# IDENTIFICATION
tp_model.identify_from_labels(labelled_image)

####################
# PLOT
####################
skgti.io.plot_model(tp_model)
plt.show()
