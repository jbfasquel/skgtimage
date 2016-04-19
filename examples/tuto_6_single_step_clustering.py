import numpy as np
import scipy as sp;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

####################################
# IMAGE AND KNOWLEDGE
####################################
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
# SEQUENTIAL SEGMENTATION STEP 2: TARGETS REGIONS C AND D USING KMEANS
####################################
# SETTING THE TARGET(S)
#tp_model.set_targets(['C','D'])

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
print(seeder)
print(seeder.detail_info())
print(seeder)
for i in range(0,20):
    print(np.transpose(seeder.generate()))

# IDENTIFICATION
tp_model.identify_from_labels(labelled_image)

####################
# PLOT
####################
skgti.io.plot_model(tp_model)
plt.show()
