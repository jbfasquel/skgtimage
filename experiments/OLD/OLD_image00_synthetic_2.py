import numpy as np
import matplotlib.pyplot as plt
import skgtimage as skgti
import scipy as sp;from scipy import misc,ndimage

tmp_dir='Synthetic/'

#################
# IMAGE
#################
#Region A
image=np.zeros((100,100),dtype=np.float);
region_A=np.ones(image.shape)
#Region B
region_B=skgti.utils.draw_rectangle(image,(50,30),(80,30),100);sp.misc.imsave(tmp_dir+'B.png',region_B)
#Region C
skgti.utils.draw_square(image,(30,30),20,150)
#Region D
skgti.utils.draw_square(image,(70,30),20,200)
#Region E
skgti.utils.draw_rectangle(image,(50,70),(60,30),100)

#################
# Non uniform lighting
#################
for i in range(0,image.shape[0]): image[i,:]=image[i,:]+i

#################
# Oversampling
#################
#image=sp.ndimage.interpolation.zoom(image,2)

#################
# NOISE
#################
image=skgti.utils.add_gaussian_noise(image,0,5.0)
'''
plt.subplot(131)
plt.imshow(image,'gray')
plt.subplot(132)
histo,bins=skgti.utils.int_histogram(image)
#y,x=skgti.utils.float_histogram(image)
plt.plot(bins,histo)
plt.subplot(133)
plt.plot(image[:,30])
print(image[:,30])
plt.show()
'''
#################
# KNOWLEDGE
#################
tp_model=skgti.core.TPModel()
tp_model.set_topology("C,D<B;B,E<A")
tp_model.add_photometry("B=E;A<B<C<D")
tp_model.set_image(image)
tp_model.set_region('A',region_A)

#################
# CONTEXT A
#################
'''
tp_model.set_targets(['C','D'])
l_image=tp_model.roi_image()
nb=tp_model.number_of_clusters()
intervals=tp_model.intervals_for_clusters()
labelled_image=skgti.utils.kmeans(l_image,nb,intervals=intervals)

#print(labelled_image)
#plt.imshow(labelled_image,'gray');plt.show()
#tp_model.identify_from_residues(labelled_image)
#skgti.io.plot_model(tp_model);plt.show()
'''
#################
# CONTEXT A
#################
tp_model.set_region('B',region_B)
tp_model.set_targets(['C','D'])
l_image=tp_model.roi_image()
nb=tp_model.number_of_clusters()
intervals=tp_model.intervals_for_clusters()
labelled_image=skgti.utils.kmeans(l_image,nb,intervals=intervals)
plt.imshow(labelled_image,'gray');plt.show()

#print(labelled_image)
g,_=skgti.core.topological_graph_from_labels(labelled_image)
skgti.io.plot_graph(g);plt.show()
#plt.imshow(labelled_image,'gray');plt.show()
tp_model.identify_from_labels(labelled_image)
skgti.io.plot_model(tp_model);plt.show()
