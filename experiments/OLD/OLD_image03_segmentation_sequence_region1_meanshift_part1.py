import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

root_name="image03"
dir="Database/image03_seg/"

##################
##################
# KNOWLEDGE
##################
##################
tp_model=skgti.core.TPModel()
#OLD
#tp_model.set_topology("8<7<3<2<1;4,5,6<3")
#tp_model.set_photometry(["1=2=3=4;5<8=6<7","1=2=3=4<5=6=7=8","2=4<1=3=5=6=7=8"])
#NEW
tp_model.set_topology("8<7<3<2<1;5,6<3")
tp_model.set_photometry(["1=2=3;5<8=6<7","1=2=3<5=6=7=8","2<1=3=5=6=7=8"])

#skgti.io.plot_graph(tp_model.t_graph);plt.show() #;quit()
#skgti.io.plot_graph(tp_model.p_graphs[0]);plt.show();quit()
#skgti.io.plot_graph(tp_model.p_graphs[1]);plt.show();quit()
#skgti.io.plot_graph(tp_model.p_graphs[2]);plt.show();quit()

##################
##################
# IMAGE
##################
##################
image=sp.misc.imread(os.path.join(dir,root_name+"_filtered.jpeg"))
image_hsv=skgti.utils.rgb2hsv(image)
tp_model.set_image(image_hsv)
#plt.imshow(image);plt.show();quit()

##################
##################
# CONTEXT
##################
##################
region_1=sp.misc.imread(os.path.join(dir,root_name+"_region1.png"))
tp_model.set_region('1',region_1)
#for r in ['0','1','2','4','5','6']: tp_model.set_region(r,sp.misc.imread(os.path.join(dir,root_name+"_region"+r+".png")))
#skgti.io.plot_model(tp_model);plt.show()

##################
##################
# SEGMENTATION
##################
##################
tp_model.set_targets(['7'])
l_image=tp_model.roi_image() #;print(l_image[:,:,0]);print(l_image[:,:,1]);print(l_image[:,:,2]);plt.imshow(l_image.filled(0));plt.show();quit()
nb=tp_model.number_of_clusters() #;print(nb);quit()
l_image_chsv=skgti.utils.hsv2chsv(l_image)
#plt.imshow(l_image_chsv.filled(0));plt.show();quit()
labelled_image=skgti.utils.mean_shift(l_image_chsv,bandwidth=0.1,spatial_dim=2,n_features=3,verbose=True)

#plt.imshow(labelled_image);plt.show()

sp.misc.imsave(dir+'meanshift_region1.png',(labelled_image+1).filled(0).astype(np.uint8))
#plt.imshow(labelled_image);plt.show();quit()
