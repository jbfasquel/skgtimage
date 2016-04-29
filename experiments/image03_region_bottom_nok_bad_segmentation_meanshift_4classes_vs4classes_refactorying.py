import os
import numpy as np
import scipy as sp;from scipy import misc;from scipy import ndimage
import matplotlib.pyplot as plt
import skgtimage as skgti
import helper

truth_dir="Database/image03/truth_bottom/"
save_dir="Database/image03/bottom_meanshift_nok_bad_segmentation_4classes_versus4expected_refactorying/"

# KNOWLEDGE
t_desc="E<D;G<F;D,F,H,I<C<B<A"
p_desc="B=F<D=H<I=E<C=A=G"
# IMAGE: COLOR AND GRAY
image_rgb=sp.misc.imread(os.path.join(truth_dir,"image.png"))
roi=sp.misc.imread(os.path.join(truth_dir,"region_A.png"))
image=skgti.utils.rgb2gray(image_rgb)

# SEGMENTATION
image_chsv=skgti.utils.rgb2chsv(image_rgb)
label=skgti.utils.mean_shift(image_chsv,0.12,roi,True,True) #0.1 OK


# RECOGNITION
matcher=skgti.core.matcher_factory(image,label,t_desc,p_desc,roi=None,manage_bounds=False,thickness=2,filtering=False)
matcher.compute_maching(True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
'''
id2r,matcher=skgti.core.recognize_regions(image,label,t_desc,p_desc,roi=roi,manage_bounds=True,thickness=2,filtering=False,verbose=True)
skgti.io.save_matcher_details(matcher,image,label,roi,save_dir,False)
skgti.io.pickle_matcher(matcher,save_dir+"matcher.pkl")
matcher=skgti.io.unpickle_matcher(save_dir+"matcher.pkl")
'''
