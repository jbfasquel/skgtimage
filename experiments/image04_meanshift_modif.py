import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti

truth_dir="Database/image04/truth2/"
save_dir="Database/image04/meanshift_modifknowledge/"

'''
#Initial a priori knowledge
t_desc="glass<glassboundary<carboundary;rims,car<carboundary<background"
p_desc="glassboundary=carboundary<background<car<glass=rims"
'''
#Initial a priori knowledge: glassboundary+carboundary=boundaries
#Implies that glass and rims can not be distinguished because
# 1) topo: glass and rims same father (and no children ???)
# 2) photo: glass and rims similar
# --> glass_rims
t_desc="glass_rims,car<boundaries;boundaries<background"
p_desc="boundaries<background<car<glass_rims"


#Segmentation
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
image=skgti.utils.rgb2gray(image_rgb)
#sp.misc.imsave("image_gray.png",image_gray)
image_chsv=skgti.utils.rgb2chsv(image_rgb)
label=skgti.utils.mean_shift(image_chsv,0.16,None,True,True) #NB cluster == 18 -> mais long cputime
#plt.imshow(label);plt.show();quit()


id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=0,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,None,save_dir,False)
