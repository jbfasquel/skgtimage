import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti

####################################
# IMAGE AND KNOWLEDGE
####################################
# IMAGE
image=np.array([[0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,0],
                [0,1,2,2,1,1,1,0],
                [0,1,2,2,1,3,1,0],
                [0,1,2,2,1,3,1,0],
                [0,1,2,2,1,1,1,0],
                [0,1,1,1,1,1,1,0],
                [0,0,0,0,0,0,0,0]])

# KNOWLEDGE
tp_model=skgti.core.TPModel()
tp_model.set_topology("C,D<B<A")
tp_model.add_photometry("A<B<C<D")

# INITIALIZATION : SETTING THE IMAGE TO BE ANALYZED AND THE ROOT REGION
tp_model.set_image(image)

####################################
# SEQUENTIAL SEGMENTATION
####################################
# STEP 0: TARGET IS REGION A (THE WHOLE IMAGE)
region=np.ones(image.shape)
tp_model.set_region('A',region)

# STEP 1: TARGET IS REGION B (GRAY LEVELS > 0)
region=np.where(image>0,1,0)
tp_model.set_region('B',region)

# STEP 2: TARGETS ARE REGIONS C AND D
tp_model.set_targets(['C','D'])
l_image=tp_model.roi_image() #Local image for local analysis
segmentations=[ np.where(l_image==i,1,0) for i in [1,2,3] ] #Segmentation

if tp_model.is_identification_feasible():
    tp_model.identify_from_residues(segmentations)

####################
# PLOT
####################
plt.subplot(331)
plt.imshow(image,cmap="gray",interpolation="nearest");plt.axis('off')
plt.title("Image")
plt.subplot(332)
skgti.io.plot_graph(tp_model.t_graph,['A','B'])
plt.title("Topological graph")
plt.subplot(333)
skgti.io.plot_graph(tp_model.p_graphs[0],['A','B'],tree=False)
plt.title("Photometric graph")
subplot_id=334
for i in range(0,3):
    plt.subplot(subplot_id+i)
    plt.title("Residue "+str(i+1))
    plt.imshow(segmentations[i],cmap="gray",interpolation="nearest");plt.axis('off')

subplot_id=337
names=['B','C','D']
for i in range(0,3):
    plt.subplot(subplot_id+i)
    name=names[i]
    region=tp_model.get_region(name)
    plt.title("Region "+name)
    plt.imshow(region,cmap="gray",vmin=0,vmax=1,interpolation="nearest");plt.axis('off')
plt.show()
