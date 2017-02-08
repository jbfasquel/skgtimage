import os
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp;from scipy import misc
import skgtimage as skgti

save_dir="../../Database/examples/case2_sim1/"
if not os.path.exists(save_dir) : os.mkdir(save_dir)

image_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 5, 5, 5, 0, 3, 3, 3, 0],
                        [0, 5, 4, 5, 0, 3, 3, 3, 0],
                        [0, 5, 4, 5, 0, 3, 3, 3, 0],
                        [0, 5, 5, 5, 0, 3, 3, 3, 0],
                        [0, 5, 5, 5, 0, 3, 3, 3, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0], ])

image_label=sp.ndimage.interpolation.zoom(image_label,zoom=4,order=0)
print(image_label)
t_model="C<B<A;D<A"
p_model="A<C=D<B"

model_example = np.array([[0, 0, 0, 0, 0],
                          [0, 2, 2, 2, 0],
                          [0, 2, 1, 2, 0],
                          [0, 2, 2, 2, 0],
                          [0, 0, 0, 0, 0],
                          [0, 1.3, 1.3, 1.3, 0],
                          [0, 0, 0, 0, 0]])
model_example = sp.ndimage.interpolation.zoom(model_example, zoom=4, order=0)
sp.misc.imsave(save_dir + "model_example.png", model_example);
#quit()

#plt.imshow(image_label);plt.show()

id2r,matcher=skgti.core.recognize_regions(image_label,image_label,t_model,p_model,roi=None,verbose=True,bf=True)
#id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,verbose=True)

#skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.save_matcher_details(matcher,image_label,image_label,None,save_dir,True)

#COMPARISON WITH TRUTH FOR ISO INFLUENCE

truth_dir=save_dir+"truth/"
if not os.path.exists(truth_dir) : os.mkdir(truth_dir)
A=np.where(image_label==0,1,0)
sp.misc.imsave(truth_dir+"region_A.png",A)
B=np.where(image_label==5,1,0)
sp.misc.imsave(truth_dir+"region_B.png",B)
C=np.where(image_label==4,1,0)
sp.misc.imsave(truth_dir+"region_C.png",C)
D=np.where(image_label==3,1,0)
sp.misc.imsave(truth_dir+"region_D.png",D)


import helper
classif,region2sim=helper.compared_with_truth(image_label,t_model,p_model,truth_dir,save_dir+"06_final",save_dir+"07_eval_classif/")

import helper
helper.influence_of_commonisos(matcher,image_label,t_model,p_model,truth_dir,save_dir)
