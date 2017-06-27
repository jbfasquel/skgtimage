import os,sys; sys.path.append(os.path.abspath("../")) #for executation without having installed the package
import numpy as np
import skgtimage as skgti

#Initial image
image= np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 4, 4, 4, 4, 4, 0],
                  [0, 4, 6, 6, 6, 4, 0],
                  [0, 4, 4, 4, 4, 4, 0],
                  [0, 4, 1, 1, 1, 4, 0],
                  [0, 4, 1, 7, 1, 4, 0],
                  [0, 4, 1, 1, 1, 4, 0],
                  [0, 4, 4, 4, 4, 4, 0],
                  [0, 0, 0, 0, 0, 0, 0]])
#Labelling: several regions depict several connected components
#Region with label 0 correspond to image intensities 0 and 1
#Region with label 2 correspond to image intensities 6 and 7
labelling = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 2, 2, 2, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 0, 2, 0, 1, 0],
                  [0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0]])

#Building both inclusion and photometry graphs
inclusion_graph,photometry_graph=skgti.core.from_labelled_image(image,labelling)
#Retrieval of the new labelling
new_labelling=inclusion_graph.get_labelled()


import matplotlib.pyplot as plt
plt.subplot(131)
plt.title("Image")
plt.imshow(image,interpolation='nearest')
plt.axis('off')
plt.subplot(132)
plt.title("Initial labelling")
plt.imshow(labelling,interpolation='nearest')
plt.axis('off')
plt.subplot(133)
plt.title("Labelling after the discovery \nof inclusion relationships")
plt.imshow(new_labelling,interpolation='nearest')
plt.axis('off')
plt.show()


